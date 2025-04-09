import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import prettytable
import torch
from common.loss import (mpjpe, mpjpe_diffusion, mpjpe_diffusion_all_min,
                         mpjpe_diffusion_median, p_mpjpe_diffusion,
                         p_mpjpe_diffusion_all_min, p_mpjpe_diffusion_median)
from common.visualize import chains_ixs, get_chains
from mvn.utils.img import crop_image
from tqdm import tqdm

map_actions = {
    'act_02_subact_01': 'Directions 1', 
    'act_02_subact_02': 'Directions', 
    'act_03_subact_01': 'Discussion 1', 
    'act_03_subact_02': 'Discussion 2', 
    'act_04_subact_01': 'Eating 1', 
    'act_04_subact_02': 'Eating', 
    'act_05_subact_01': 'Greeting 1', 
    'act_05_subact_02': 'Greeting', 
    'act_06_subact_01': 'Phoning 1', 
    'act_06_subact_02': 'Phoning', 
    'act_07_subact_01': 'Photo 1', 
    'act_07_subact_02': 'Photo', 
    'act_08_subact_01': 'Posing 1', 
    'act_08_subact_02': 'Posing', 
    'act_09_subact_01': 'Purchases 1', 
    'act_09_subact_02': 'Purchases', 
    'act_10_subact_01': 'Sitting 1', 
    'act_10_subact_02': 'Sitting', 
    'act_11_subact_01': 'SittingDown 1', 
    'act_11_subact_02': 'SittingDown', 
    'act_12_subact_01': 'Smoking 1', 
    'act_12_subact_02': 'Smoking', 
    'act_13_subact_01': 'Waiting 1', 
    'act_13_subact_02': 'Waiting', 
    'act_14_subact_01': 'WalkDog 1', 
    'act_14_subact_02': 'WalkDog', 
    'act_15_subact_01': 'WalkTogether 1', 
    'act_15_subact_02': 'WalkTogether', 
    'act_16_subact_01': 'Walking 1', 
    'act_16_subact_02': 'Walking'
}

actions_metrics = {
    'Directions': [],
    'Discussion': [],
    'Eating': [],
    'Greeting': [],
    'Phoning': [],
    'Photo': [],
    'Posing': [],
    'Purchases': [],
    'Sitting': [],
    'SittingDown': [],
    'Smoking': [],
    'Waiting': [],
    'WalkDog': [],
    'WalkTogether': [],
    'Walking': [],
    'All': []
}

dataset = 'mpi-inf'
metric_to_compute = 'mpjpe-pck-auc'

if dataset == 'human36m':
    gt_poses = np.load('human_gt.npy')
    pred_poses = np.load('human_predictions.npy')
    image_names = np.load('image_names.npy')
elif dataset == 'mpi-inf':
    gt_poses = np.load('mpi_gt.npy')
    pred_poses = np.load('mpi_predictions.npy')
else:
    raise ValueError('Invalid dataset')

if metric_to_compute == 'reliability':
    num_samples = len(gt_poses)
    idxs = np.arange(len(gt_poses))

 
    pose_mean_distances = []
    mpjpe_metrics = []
    
    for idx in tqdm(idxs):
        pred = pred_poses[idx]
        gt = gt_poses[idx]

        mean_dist_pose = np.mean(np.linalg.norm(pred - pred.mean(axis=0, keepdims=True), axis=2)) * 1000
        mpjpe_mean_pose = np.mean(np.linalg.norm(pred.mean(axis=0, keepdims=True) - gt, axis=-1)) * 1000

        pose_mean_distances.append(mean_dist_pose)
        mpjpe_metrics.append(mpjpe_mean_pose)
    
    pose_mean_distances = np.array(pose_mean_distances)
    mpjpe_metrics = np.array(mpjpe_metrics)

    precision_scores = []
    recall_scores = []
    if dataset == 'human36m':
        thresholds = np.arange(8, 30, 0.5)
    else:
        thresholds = np.arange(8, 60, 1.75)
    for thres in thresholds:
        precision = np.mean(mpjpe_metrics[pose_mean_distances < thres])
        good_poses = np.sum((pose_mean_distances < thres))
        recall = int((good_poses / num_samples) * 100)
        precision_scores.append(precision)
        recall_scores.append(recall)
    precision_scores.append(np.mean(mpjpe_metrics))
    recall_scores.append(100)
    plt.plot(recall_scores, precision_scores, c='r')
    plt.xlabel('% of samples')
    plt.ylabel('MPJPE (mm)')
    plt.xticks(np.arange(0, 101, 5))
    plt.gca().invert_xaxis()
    plt.savefig(f'reliability_{dataset}.pdf', dpi=300, bbox_inches='tight') 
    plt.close()



elif metric_to_compute == 'new_aggregation':
    metrics = []
    for pred, gt in tqdm(zip(pred_poses, gt_poses)):
        dist_from_mean = np.linalg.norm(pred - pred.mean(axis=0, keepdims=True), axis=(1, 2))
        sorted_hypotheses = np.argsort(dist_from_mean)
        ordered_pose_pred = pred[sorted_hypotheses]

        best_mpjpe = np.mean(np.linalg.norm(ordered_pose_pred[:1] - gt, axis=-1)) * 1000
        mean_10_poses = np.mean(ordered_pose_pred[:int(ordered_pose_pred.shape[0] * 0.1)], axis=0, keepdims=True)
        best_10_mpjpe = np.mean(np.linalg.norm(mean_10_poses - gt, axis=-1)) * 1000
        mean_20_poses = np.mean(ordered_pose_pred[:int(ordered_pose_pred.shape[0] * 0.2)], axis=0, keepdims=True)
        best_20_mpjpe = np.mean(np.linalg.norm(mean_20_poses - gt, axis=-1)) * 1000
        mean_30_poses = np.mean(ordered_pose_pred[:int(ordered_pose_pred.shape[0] * 0.3)], axis=0, keepdims=True)
        best_30_mpjpe = np.mean(np.linalg.norm(mean_30_poses - gt, axis=-1)) * 1000
        mean_40_poses = np.mean(ordered_pose_pred[:int(ordered_pose_pred.shape[0] * 0.4)], axis=0, keepdims=True)
        best_40_mpjpe = np.mean(np.linalg.norm(mean_40_poses - gt, axis=-1)) * 1000
        mean_50_poses = np.mean(ordered_pose_pred[:int(ordered_pose_pred.shape[0] * 0.5)], axis=0, keepdims=True)
        best_50_mpjpe = np.mean(np.linalg.norm(mean_50_poses - gt, axis=-1)) * 1000
        mean_60_poses = np.mean(ordered_pose_pred[:int(ordered_pose_pred.shape[0] * 0.6)], axis=0, keepdims=True)
        best_60_mpjpe = np.mean(np.linalg.norm(mean_60_poses - gt, axis=-1)) * 1000
        mean_70_poses = np.mean(ordered_pose_pred[:int(ordered_pose_pred.shape[0] * 0.7)], axis=0, keepdims=True)
        best_70_mpjpe = np.mean(np.linalg.norm(mean_70_poses - gt, axis=-1)) * 1000
        mean_80_poses = np.mean(ordered_pose_pred[:int(ordered_pose_pred.shape[0] * 0.8)], axis=0, keepdims=True)
        best_80_mpjpe = np.mean(np.linalg.norm(mean_80_poses - gt, axis=-1)) * 1000
        mean_90_poses = np.mean(ordered_pose_pred[:int(ordered_pose_pred.shape[0] * 0.9)], axis=0, keepdims=True)
        best_90_mpjpe = np.mean(np.linalg.norm(mean_90_poses - gt, axis=-1)) * 1000
        mean_100_poses = np.mean(ordered_pose_pred, axis=0, keepdims=True)
        best_100_mpjpe = np.mean(np.linalg.norm(mean_100_poses - gt, axis=-1)) * 1000

        metrics.append([best_mpjpe, best_10_mpjpe, best_20_mpjpe, best_30_mpjpe, best_40_mpjpe, best_50_mpjpe,
                        best_60_mpjpe, best_70_mpjpe, best_80_mpjpe, best_90_mpjpe, best_100_mpjpe])

    metrics = np.array(metrics)

    table = prettytable.PrettyTable()
    table.field_names = ['Hypothesis', 'MPJPE']
    table.add_row(['Best', f'{np.mean(metrics[:, 0]):.2f} mm'])
    table.add_row(['Best 10%', f'{np.mean(metrics[:, 1]):.2f} mm'])
    table.add_row(['Best 20%', f'{np.mean(metrics[:, 2]):.2f} mm'])
    table.add_row(['Best 30%', f'{np.mean(metrics[:, 3]):.2f} mm'])
    table.add_row(['Best 40%', f'{np.mean(metrics[:, 4]):.2f} mm'])
    table.add_row(['Best 50%', f'{np.mean(metrics[:, 5]):.2f} mm'])
    table.add_row(['Best 60%', f'{np.mean(metrics[:, 6]):.2f} mm'])
    table.add_row(['Best 70%', f'{np.mean(metrics[:, 7]):.2f} mm'])
    table.add_row(['Best 80%', f'{np.mean(metrics[:, 8]):.2f} mm'])
    table.add_row(['Best 90%', f'{np.mean(metrics[:, 9]):.2f} mm'])
    table.add_row(['Avg all', f'{np.mean(metrics[:, 10]):.2f} mm'])
    print(table)

    plt.plot([np.mean(metrics[:, 0]), np.mean(metrics[:, 1]), np.mean(metrics[:, 2]), np.mean(metrics[:, 3]), np.mean(metrics[:, 4]),   
            np.mean(metrics[:, 5]), np.mean(metrics[:, 6]), np.mean(metrics[:, 7]), np.mean(metrics[:, 8]), np.mean(metrics[:, 9]), 
            np.mean(metrics[:, 10])], c='g')
    plt.xticks(range(11), ['Nearest', '10%', '20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%'])
    plt.xlabel('% of Hypothesis')
    plt.ylabel('MPJPE (mm)')
    plt.savefig(f'mpjpe_hypothesis_{dataset}.pdf', dpi=300, bbox_inches='tight')
    plt.close()

elif metric_to_compute == 'mpjpe':
    for image_name, gt, pred in tqdm(zip(image_names, gt_poses, pred_poses)):
        image_name = str(image_name)

        gt = torch.from_numpy(gt[None])
        pred = torch.from_numpy(pred[None][:, None][:, :, :, None])

        mpjpe_avg_hypothesis = mpjpe_diffusion_all_min(pred, gt, mean_pos=True).item() * 1000
        mpjpe_median_hypothesis = mpjpe_diffusion_median(pred, gt).item() * 1000
        mpjpe_best_skel_hypothesis = mpjpe_diffusion(pred, gt).item() * 1000
        mpjpe_best_joint_hypothesis = mpjpe_diffusion_all_min(pred, gt).item() * 1000

        p_mpjpe_avg_hypothesis = p_mpjpe_diffusion_all_min(pred, gt, mean_pos=True).item() * 1000
        p_mpjpe_median_hypothesis = p_mpjpe_diffusion_median(pred, gt).item() * 1000
        p_mpjpe_best_skel_hypothesis = p_mpjpe_diffusion(pred, gt).item() * 1000
        p_mpjpe_best_joint_hypothesis = p_mpjpe_diffusion_all_min(pred, gt).item() * 1000

        if 'act_02_subact_01' in image_name or 'act_02_subact_02' in image_name:
            actions_metrics['Directions'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                                p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_03_subact_01' in image_name or 'act_03_subact_02' in image_name:
            actions_metrics['Discussion'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                                p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_04_subact_01' in image_name or 'act_04_subact_02' in image_name:
            actions_metrics['Eating'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                            p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_05_subact_01' in image_name or 'act_05_subact_02' in image_name:
            actions_metrics['Greeting'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                                p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_06_subact_01' in image_name or 'act_06_subact_02' in image_name:
            actions_metrics['Phoning'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                            p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_07_subact_01' in image_name or 'act_07_subact_02' in image_name:
            actions_metrics['Photo'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                            p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_08_subact_01' in image_name or 'act_08_subact_02' in image_name:
            actions_metrics['Posing'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                            p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_09_subact_01' in image_name or 'act_09_subact_02' in image_name:
            actions_metrics['Purchases'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                                p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_10_subact_01' in image_name or 'act_10_subact_02' in image_name:
            actions_metrics['Sitting'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                            p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_11_subact_01' in image_name or 'act_11_subact_02' in image_name:
            actions_metrics['SittingDown'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                                p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_12_subact_01' in image_name or 'act_12_subact_02' in image_name:
            actions_metrics['Smoking'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                            p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_13_subact_01' in image_name or 'act_13_subact_02' in image_name:
            actions_metrics['Waiting'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                            p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_14_subact_01' in image_name or 'act_14_subact_02' in image_name:
            actions_metrics['WalkDog'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                            p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_15_subact_01' in image_name or 'act_15_subact_02' in image_name:
            actions_metrics['WalkTogether'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                                    p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
        if 'act_16_subact_01' in image_name or 'act_16_subact_02' in image_name:
            actions_metrics['Walking'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                            p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])
            
        actions_metrics['All'].append([mpjpe_avg_hypothesis, mpjpe_median_hypothesis, mpjpe_best_skel_hypothesis, mpjpe_best_joint_hypothesis,
                                    p_mpjpe_avg_hypothesis, p_mpjpe_median_hypothesis, p_mpjpe_best_skel_hypothesis, p_mpjpe_best_joint_hypothesis])

    table = prettytable.PrettyTable()
    table.field_names = ['Action', 'MPJPE Avg', 'MPJPE Med', 'MPJPE BS', 'MPJPE BJ', 'P-MPJPE Avg', 'P-MPJPE Med', 'P-MPJPE BS', 'P-MPJPE BJ']
    for action in actions_metrics.keys():
        avg_metrics = np.mean(actions_metrics[action], axis=0)
        table.add_row([action, f'{avg_metrics[0]:.2f} mm', f'{avg_metrics[1]:.2f} mm', f'{avg_metrics[2]:.2f} mm', f'{avg_metrics[3]:.2f} mm',
                    f'{avg_metrics[4]:.2f} mm', f'{avg_metrics[5]:.2f} mm', f'{avg_metrics[6]:.2f} mm', f'{avg_metrics[7]:.2f} mm'])
    print(table)

elif metric_to_compute == 'mpjpe-pck-auc':
    mpjpe_scores = []
    for gt, pred in tqdm(zip(gt_poses, pred_poses)):
        gt = torch.from_numpy(gt[None])
        pred = torch.from_numpy(pred[None][:, None][:, :, :, None])
        mpjpe_scores.append(mpjpe_diffusion_all_min(pred, gt, mean_pos=True).item() * 1000)
    mpjpe_scores = np.array(mpjpe_scores)
    print(f'MPJPE: {np.mean(mpjpe_scores):.2f} mm')

    pcks = []
    pcks_auc = []
    thresholds = np.arange(0, 151, 5)
    for pred, gt in zip(pred_poses, gt_poses):
        pred = torch.from_numpy(pred[None])
        pred = torch.mean(pred, dim=1, keepdim=True)
        gt = torch.from_numpy(gt[None])

        pck = torch.mean(torch.lt(torch.norm(pred - gt, dim=-1) * 1000, 150.0).float(), dim=(0, 2))
        pcks.append(pck)

        curr_pck_auc = []
        for thres in thresholds:
            pck = torch.mean(torch.lt(torch.norm(pred - gt, dim=-1) * 1000, thres).float(), dim=(0, 2))
            curr_pck_auc.append(pck.item())
        pcks_auc.append(curr_pck_auc)
    
    pcks = torch.stack(pcks, dim=0)
    pck = pcks.sum() / len(pcks.flatten())
    print(f'PCK: {pck.item() * 100:.2f}')

    pcks_auc = np.array(pcks_auc)
    pcks_auc = np.mean(pcks_auc, axis=0)
    print(f'AUC: {np.mean(pcks_auc) * 100:.2f}')
