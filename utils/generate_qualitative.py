from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from common.visualize import chains_ixs, get_chains
from tqdm import tqdm
 

img_files = Path(f'predictions').glob('mpi_img_*.npy')
img_files = sorted(list(img_files))
pred_files = Path(f'predictions').glob('mpi_pred_*.npy')
pred_files = sorted(list(pred_files))
gt_files = Path(f'{path_}/predictions').glob('mpi_gt_*.npy')
gt_files = sorted(list(gt_files))

random_samples = np.random.choice(len(img_files), 20, replace=False)
img_files = [img_files[i] for i in random_samples]
pred_files = [pred_files[i] for i in random_samples]
gt_files = [gt_files[i] for i in random_samples]

for i, (img_file, pred_file, gt_file) in tqdm(enumerate(zip(img_files, pred_files, gt_files))):
    img = np.load(img_file)
    gt = np.load(gt_file)
    pred = np.load(pred_file)
    for idx in range(img.shape[0]):
        curr_img = cv2.resize(img[idx], (256, 256))
        cv2.imwrite(f'{path_}/qualitative/{i:04}_{idx:04}_rgb.jpg', curr_img[:, :, ::-1])

        curr_gt = gt[idx][0]
        curr_pred = np.mean(pred[idx][0, :, 0], axis=0)
        for azimuth in range(-90, 91, 10):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            chains_gt = get_chains(curr_gt, *chains_ixs)
            ax.scatter3D(curr_gt[:, 0], curr_gt[:, 2], -curr_gt[:, 1], c='green', depthshade=True)
            for c_id in range(len(chains_gt)):
                chain_gt = chains_gt[c_id]                    
                plt.plot(chain_gt[:, 0], chain_gt[:, 2], -chain_gt[:, 1], c='green')

            chains_pred = get_chains(curr_pred, *chains_ixs)
            ax.scatter3D(curr_pred[:, 0], curr_pred[:, 2], -curr_pred[:, 1], c='red', depthshade=True)
            for c_id in range(len(chains_pred)):
                chain_pred = chains_pred[c_id]                    
                plt.plot(chain_pred[:, 0], chain_pred[:, 2], -chain_pred[:, 1], c='red')
            
            xy_radius = 0.5
            radius = 0.75
            azim_delta = 70
            ax.view_init(elev=15, azim=azimuth)
            ax.set_xlim3d([-xy_radius, xy_radius])
            ax.set_zlim3d([-radius, radius])
            ax.set_ylim3d([-xy_radius, xy_radius])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.get_zaxis().set_visible(False)

            plt.savefig(f'qualitative/{i:04}_{idx:04}_az_{azimuth:04}_pred.png', dpi=300, bbox_inches="tight") 
            plt.close()