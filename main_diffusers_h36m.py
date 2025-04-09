import errno
import os
import random
import sys
import warnings
from datetime import datetime
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import webdataset as wds
from accelerate import Accelerator
from common.arguments import parse_args
from common.diff_pipeline import DiffPipe
from common.loss import (mpjpe, mpjpe_diffusion, mpjpe_diffusion_all_min,
                         mpjpe_diffusion_median)
from common.pose_dformer import PoseTransformer
from common.visualize import chains_ixs, get_chains
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from mvn.datasets import utils as dataset_utils
from mvn.datasets.human36m import get_sample_test, get_sample_train
from mvn.models import pose_hrnet
from mvn.utils.cfg import config, update_config
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args = parse_args()

print('python ' + ' '.join(sys.argv))
print("CUDA Device Count: ", torch.cuda.device_count())
print(args)

# tensorboard
if not args.evaluate:
    writer = SummaryWriter(args.log)
    writer.add_text('command', 'python ' + ' '.join(sys.argv))


# set random seed
manualSeed = 1234
random.seed(manualSeed)
torch.manual_seed(manualSeed)
np.random.seed(manualSeed)
torch.cuda.manual_seed_all(manualSeed)

# create checkpoint directory if it does not exist
if args.checkpoint=='':
    raise ValueError('Invalid checkpoint path')
try:
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

update_config(args.config)

# set accelerator
accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="no")

# dataset loading
if not args.evaluate:
    train_dataset = wds.WebDataset(f'{args.webdataset_path}/train/'+'data-{000000..000155}.tar', shardshuffle=True)
    train_dataset = train_dataset.shuffle(1000).decode('pil').to_tuple('jpg', 'pyd').map(get_sample_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset.batched(config.train.batch_size), batch_size=None, 
                                                   num_workers=config.train.num_workers)
    val_dataset = wds.WebDataset(f'{args.webdataset_path}/validation/'+'data-{000000..000054}.tar', shardshuffle=False)
    val_dataset = val_dataset.decode('pil').to_tuple('jpg', 'pyd').map(get_sample_test)
    val_dataloader = torch.utils.data.DataLoader(val_dataset.batched(config.val.batch_size), batch_size=None, 
                                                 num_workers=config.val.num_workers)
else:
    val_dataset = wds.WebDataset(f'{args.webdataset_path}/validation/'+'data-{000000..000054}.tar', shardshuffle=False)
    val_dataset = val_dataset.decode('pil').to_tuple('jpg', 'pyd').map(get_sample_test)
    val_dataloader = torch.utils.data.DataLoader(val_dataset.batched(config.val.batch_size), batch_size=None, 
                                                 num_workers=config.val.num_workers)

joints_left = [4, 5, 6, 11, 12, 13]
joints_right = [1, 2, 3, 14, 15, 16]


# define backbone
backbone = pose_hrnet.get_pose_net(config.model.backbone)
ret = backbone.load_state_dict(torch.load("checkpoint/posehrnet/pose_hrnet_w32_256x192.pth"), strict=False)
print("Loading backbone from {}".format(config.model.backbone.checkpoint))
print("Backbone weights are fixed!")
for p in backbone.parameters():
    p.requires_grad = False
backbone.eval()

# define model
if not args.evaluate:
    model_pos_train = PoseTransformer(mask_pose=config.model.poseformer.mask_pose, 
                                      mask_context=config.model.poseformer.mask_context, 
                                      shuffle_context=config.model.poseformer.shuffle_context,
                                      drop_path_rate=0.2)
model_pos_val = PoseTransformer(mask_pose=config.model.poseformer.mask_pose, 
                                mask_context=config.model.poseformer.mask_context, 
                                shuffle_context=config.model.poseformer.shuffle_context,
                                drop_path_rate=0)

# load deformable context extraction module
if not args.evaluate:
    checkpoint_conpose = torch.load('checkpoint/human36m_nips_retrain.bin')['model']
    checkpoint_conpose = {k.replace('module.', '').replace('volume_net', 'pose_estimator'): v for k, v in checkpoint_conpose.items()}
    checkpoint_conpose = {k: v for k, v in checkpoint_conpose.items() if 'backbone' not in k}
    checkpoint_conpose = {k.replace('pose_estimator.', ''): v for k, v in checkpoint_conpose.items()}
    checkpoint_conpose = {k: v for k, v in checkpoint_conpose.items() if 'coord_embed' in k or 'feat_embed' in k or 'Spatial_pos_embed' in k or 'context_blocks' in k}
    ret = model_pos_train.load_state_dict(checkpoint_conpose, strict=False)
    print("Loading deformable context extraction...")
    model_pos_train.Spatial_pos_embed.requires_grad = False
    for p in model_pos_train.coord_embed.parameters():
        p.requires_grad = False
    for p in model_pos_train.feat_embed.parameters():
        p.requires_grad = False
    for p in model_pos_train.context_blocks.parameters():
        p.requires_grad = False

# set noise schedulers
noise_scheduler = DDPMScheduler(num_train_timesteps=100)
noise_scheduler.config.prediction_type = "epsilon" 
val_scheduler = DDIMScheduler(num_train_timesteps=100)
if not args.evaluate:
    val_scheduler.set_timesteps(5, device=accelerator.device)
else:
    val_scheduler.set_timesteps(args.timesteps, device=accelerator.device)
val_scheduler.config.prediction_type = "epsilon"

# prepare model and dataloader with accelerator
backbone = accelerator.prepare_model(backbone)
if not args.evaluate:
    model_pos_train = accelerator.prepare_model(model_pos_train)
    train_dataloader = accelerator.prepare_data_loader(train_dataloader)
model_pos_val = accelerator.prepare_model(model_pos_val)
if not args.evaluate:
    val_dataloader = accelerator.prepare_data_loader(val_dataloader)

# put models on gpu if available
if torch.cuda.is_available():
    backbone = backbone.to(accelerator.device)
    if not args.evaluate:
        model_pos_train = model_pos_train.to(accelerator.device)
    model_pos_val = model_pos_val.to(accelerator.device)

# resume training from pretrained model
if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch'] + 1))
    if not args.evaluate:
        model_pos_train.load_state_dict(checkpoint['model_pos'], strict=False)
    model_pos_val.load_state_dict(checkpoint['model_pos'], strict=False)

if not args.evaluate:
    #####################
    # TRAINING START!!! #
    #####################
    criterion = nn.MSELoss()

    lr = args.learning_rate
    optimizer = optim.AdamW(model_pos_train.parameters(), lr=lr, weight_decay=0.1)
    optimizer = accelerator.prepare_optimizer(optimizer)

    lr_decay = args.lr_decay
    losses_3d_train = []
    losses_3d_valid = []

    epoch = 0
    best_epoch = 0

    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')
        lr = checkpoint['lr']

    while epoch < args.epochs:
        model_pos_train.train()

        start_time_epoch = time()
        epoch_loss_3d_train = 0

        train_progress_bar = tqdm(desc=f"Training epoch {epoch+1}")
        it = 0
        
        for images, inputs_3d, inputs_2d, inputs_2d_crop in train_dataloader:
            with accelerator.accumulate(model_pos_train):
                images = images.to(accelerator.device)
                inputs_3d = inputs_3d.to(accelerator.device)
                inputs_2d = inputs_2d.to(accelerator.device)
                inputs_2d_crop = inputs_2d_crop.to(accelerator.device)

                inputs_2d = inputs_2d[:, None]

                # extract context features from backbone
                context_features = backbone(images)

                # reference points for deformable attention on backbone features
                inputs_2d_crop[..., :2] /= torch.tensor([192//2, 256//2], device=images.device)
                inputs_2d_crop[..., :2] -= torch.tensor([1, 1], device=images.device)

                # Sample random noise
                noise = torch.randn_like(inputs_3d)
                bsz = inputs_3d.shape[0]
                # Sample a random timestep for each sample in the batch
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=inputs_3d.device)
                timesteps = timesteps.long()
                # Add noise to the input according to the noise magnitude at each timestep
                noisy_joints_3d = noise_scheduler.add_noise(inputs_3d, noise, timesteps)

                optimizer.zero_grad()

                # predict 3D poses
                predictions = model_pos_train(inputs_2d, inputs_2d_crop, noisy_joints_3d, context_features, timesteps)

                # compute losses and backpropagate
                loss_total = criterion(predictions, noise)
                accelerator.backward(loss_total)
                optimizer.step()

                epoch_loss_3d_train += loss_total.item()

                train_progress_bar.update()
                train_progress_bar.set_postfix({
                    'noise_loss': loss_total.item()
                })

                it += 1

        losses_3d_train.append(epoch_loss_3d_train / it)

        # evaluate every 5 epochs
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                model_pos_val.load_state_dict(model_pos_train.state_dict(), strict=False)
                model_pos_val.eval()

                pipeline = DiffPipe(model_pos_val, val_scheduler).to(accelerator.device)

                epoch_loss_3d_valid = 0
                
                val_progress_bar = tqdm(desc=f"Validation epoch {epoch+1}")
                it = 0
                
                # evaluate on test set
                for images, inputs_3d, inputs_2d, inputs_2d_crop, _ in val_dataloader:
                    b, n, _, _, _ = images.shape

                    images = images.to(accelerator.device)
                    images = images.view(b*n, images.shape[2], images.shape[3], images.shape[4])
                    inputs_2d = inputs_2d.to(accelerator.device)
                    inputs_2d = inputs_2d.view(b*n, inputs_2d.shape[2], inputs_2d.shape[3])[:, None]
                    inputs_2d_crop = inputs_2d_crop.to(accelerator.device)
                    inputs_2d_crop = inputs_2d_crop.view(b*n, inputs_2d_crop.shape[2], inputs_2d_crop.shape[3])

                    inputs_3d = inputs_3d.to(accelerator.device)

                    # reference points for deformable attention on backbone features
                    inputs_2d_crop[..., :2] /= torch.tensor([192//2, 256//2], device=images.device)
                    inputs_2d_crop[..., :2] -= torch.tensor([1, 1], device=images.device)

                    # sample random noise
                    shape = (b * n, args.num_proposals, 1, 17, 3)
                    random_noise = torch.randn(shape).to(accelerator.device)

                    # predict context features
                    context_features = backbone(images)

                    # predict from random noise
                    predicted_3d_pos = pipeline(random_noise, inputs_2d, inputs_2d_crop, context_features)

                    predicted_3d_pos[:, :, :, :, 0, :] = 0
                    predicted_3d_pos = predicted_3d_pos.view(b, n,
                                                            predicted_3d_pos.shape[1], predicted_3d_pos.shape[2],
                                                            predicted_3d_pos.shape[3], predicted_3d_pos.shape[4],
                                                            predicted_3d_pos.shape[5])
                    predicted_3d_pos_flip = predicted_3d_pos[:, 1].clone()
                    predicted_3d_pos = predicted_3d_pos[:, 0]

                    # average predictions from original and flipped images
                    predicted_3d_pos_flip[..., 0] *= -1
                    predicted_3d_pos_flip[:, :, :, :, joints_left + joints_right, :] = predicted_3d_pos_flip[:, :, :, :, joints_right + joints_left, :]
                    predicted_3d_pos = torch.mean(torch.stack((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1)

                    error_total = mpjpe_diffusion_all_min(predicted_3d_pos[:, -1:], inputs_3d, mean_pos=True)
                    
                    epoch_loss_3d_valid += error_total.item()
                    
                    val_progress_bar.update()
                    val_progress_bar.set_postfix({
                        'mpjpe (mm)': error_total.item() * 1000
                    })

                    it += 1

                losses_3d_valid.append(epoch_loss_3d_valid / it)

        elapsed = (time() - start_time_epoch) / 60

        # log losses
        print('[%d] time %.2f lr %f 3d_pos_train %f 3d_pos_valid %f' % (
            epoch + 1,
            elapsed,
            lr,
            losses_3d_train[-1],
            losses_3d_valid[-1] * 1000 if (epoch + 1) % 5 == 0 else -1
        ))
        
        log_path = os.path.join(args.checkpoint, 'training_log.txt')
        f = open(log_path, mode='a')
        f.write('[%d] time %.2f lr %f 3d_pos_train %f 3d_pos_valid %f\n' % (
            epoch + 1,
            elapsed,
            lr,
            losses_3d_train[-1],
            losses_3d_valid[-1] * 1000 if (epoch + 1) % 5 == 0 else -1
        ))
        f.close()

        writer.add_scalar("Loss/3d training loss", losses_3d_train[-1], epoch+1)
        if (epoch + 1) % 5 == 0:
            writer.add_scalar("Loss/3d validation loss", losses_3d_valid[-1] * 1000, epoch+1)
        writer.add_scalar("Parameters/learning rate", lr, epoch+1)
        writer.add_scalar('Parameters/training time per epoch', elapsed, epoch+1)
        
        # decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay

        # save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'latest_epoch.bin')
            print('Saving checkpoint to', chk_path)
            torch.save({
                'epoch': epoch,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
            }, chk_path)

        epoch += 1

    ###################
    # TRAINING END!!! #
    ###################
else:
    ####################
    # TESTING START!!! #
    ####################
    print('Evaluating after training...')

    model_pos_val.eval()
    model_pos_val = accelerator.unwrap_model(model_pos_val)

    pipeline = DiffPipe(model_pos_val, val_scheduler).to(accelerator.device)

    with torch.no_grad():
        mpjpe_avg_hypothesis = 0
        mpjpe_median_hypothesis = 0
        mpjpe_best_skel_hypothesis = 0
        mpjpe_best_joint_hypothesis = 0
        last_epoch = 0

        pred_skeletons = []
        gt_skeletons = []
        
        
        it = 1
        total_elements = 0
        # evaluate on test set
        for images, inputs_3d, inputs_2d, inputs_2d_crop, img in val_dataloader:
            start = time()
                
            b, n, _, _, _ = images.shape
            total_elements += b

            images = images.to(accelerator.device)
            images = images.view(b*n, images.shape[2], images.shape[3], images.shape[4])
            inputs_2d = inputs_2d.to(accelerator.device)
            inputs_2d = inputs_2d.view(b*n, inputs_2d.shape[2], inputs_2d.shape[3])[:, None]
            inputs_2d_crop = inputs_2d_crop.to(accelerator.device)
            inputs_2d_crop = inputs_2d_crop.view(b*n, inputs_2d_crop.shape[2], inputs_2d_crop.shape[3])

            inputs_3d = inputs_3d.to(accelerator.device)

            # reference points for deformable attention on backbone features
            inputs_2d_crop[..., :2] /= torch.tensor([192//2, 256//2], device=images.device)
            inputs_2d_crop[..., :2] -= torch.tensor([1, 1], device=images.device)

            # sample random noise
            shape = (b * n, args.num_proposals, 1, 17, 3)
            random_noise = torch.randn(shape).to(accelerator.device)

            # predict context features
            context_features = backbone(images)

            # predict from random noise
            predicted_3d_pos = pipeline(random_noise, inputs_2d, inputs_2d_crop, context_features)

            predicted_3d_pos[:, :, :, :, 0, :] = 0
            predicted_3d_pos = predicted_3d_pos.view(b, n,
                                                     predicted_3d_pos.shape[1], predicted_3d_pos.shape[2],
                                                     predicted_3d_pos.shape[3], predicted_3d_pos.shape[4],
                                                     predicted_3d_pos.shape[5])
            predicted_3d_pos_flip = predicted_3d_pos[:, 1].clone()
            predicted_3d_pos = predicted_3d_pos[:, 0]

            # average predictions from original and flipped images
            predicted_3d_pos_flip[..., 0] *= -1
            predicted_3d_pos_flip[:, :, :, :, joints_left + joints_right, :] = predicted_3d_pos_flip[:, :, :, :, joints_right + joints_left, :]
            predicted_3d_pos = torch.mean(torch.stack((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1)

            batch_mpjpe_avg_hypothesis = mpjpe_diffusion_all_min(predicted_3d_pos[:, -1:], inputs_3d, mean_pos=True)
            batch_mpjpe_median_hypothesis = mpjpe_diffusion_median(predicted_3d_pos[:, -1:], inputs_3d)
            batch_mpjpe_best_skel_hypothesis = mpjpe_diffusion(predicted_3d_pos[:, -1:], inputs_3d)
            batch_mpjpe_best_joint_hypothesis = mpjpe_diffusion_all_min(predicted_3d_pos[:, -1:], inputs_3d)

            if args.visualize:
                if it % 20 == 0:
                    break

                mean_pose = torch.mean(predicted_3d_pos[:, -1:], dim=2, keepdim=False)
                target = inputs_3d.unsqueeze(1).repeat(1, predicted_3d_pos.shape[1], 1, 1, 1)
                errors = torch.norm(mean_pose - target, dim=len(target.shape) - 1)
                top_idxs = torch.topk(errors.mean(dim=[1, 2, 3]), 20, largest=False).indices.cpu().numpy()

                img = img[top_idxs].cpu().numpy()
                predicted_3d_pos = predicted_3d_pos[top_idxs, -1:].cpu().numpy()
                inputs_3d = inputs_3d[top_idxs].cpu().numpy()
                np.save(f'predictions/human_img_{it}.npy', img)
                np.save(f'predictions/human_gt_{it}.npy', inputs_3d)
                np.save(f'predictions/human_pred_{it}.npy', predicted_3d_pos)
            
            mpjpe_avg_hypothesis += batch_mpjpe_avg_hypothesis.item() * b
            mpjpe_median_hypothesis += batch_mpjpe_median_hypothesis.item() * b
            mpjpe_best_skel_hypothesis += batch_mpjpe_best_skel_hypothesis.item() * b
            mpjpe_best_joint_hypothesis += batch_mpjpe_best_joint_hypothesis.item() * b

            pred_skeletons.append(predicted_3d_pos[:, -1, :, 0, :, :].detach().cpu())
            gt_skeletons.append(inputs_3d.detach().cpu())

            print(f"[{it}] ({time() - start:.2f} seconds)"
                  f"MPJPE (Avg): {mpjpe_avg_hypothesis * 1000 / total_elements:.2f} mm |"
                  f"MPJPE (Median): {mpjpe_median_hypothesis * 1000 / total_elements:.2f} mm |"
                  f"MPJPE (Best Per-Skeleton): {mpjpe_best_skel_hypothesis * 1000 / total_elements:.2f} mm |"
                  f"MPJPE (Best Per-Joint): {mpjpe_best_joint_hypothesis * 1000 / total_elements:.2f} mm")
                
            
            it += 1
            
        
        np.save(f'predictions/human_predictions.npy', torch.vstack(pred_skeletons).numpy())
        np.save(f'predictions/human_gt.npy', torch.vstack(gt_skeletons).numpy())

            
            
        

        mpjpe_avg_hypothesis = mpjpe_avg_hypothesis / total_elements
        mpjpe_median_hypothesis = mpjpe_median_hypothesis / total_elements
        mpjpe_best_skel_hypothesis = mpjpe_best_skel_hypothesis / total_elements
        mpjpe_best_joint_hypothesis = mpjpe_best_joint_hypothesis / total_elements

        print(f'Average MPJPE (Avg Hypothesis): {mpjpe_avg_hypothesis * 1000:.2f} mm')
        print(f'Average MPJPE (Median Hypothesis): {mpjpe_median_hypothesis * 1000:.2f} mm')
        print(f'Average MPJPE (Best Per-Skeleton Hypothesis): {mpjpe_best_skel_hypothesis * 1000:.2f} mm')
        print(f'Average MPJPE (Best Per-Joint Hypothesis): {mpjpe_best_joint_hypothesis * 1000:.2f} mm')

    
    ##################
    # TESTING END!!! #
    ##################
