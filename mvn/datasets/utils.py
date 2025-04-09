import numpy as np
import torch

from mvn.utils.img import image_batch_to_torch

import os
import zipfile
import cv2
import random

joints_left = [4, 5, 6, 11, 12, 13] 
joints_right = [1, 2, 3, 14, 15, 16]

class data_prefetcher():
    def __init__(self, loader, device, is_train, flip_test):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.device = device
        self.is_train = is_train
        self.flip_test = flip_test
        self.mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).to(device)
        self.preload()

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            for i in range(len(self.next_batch)):
                self.next_batch[i] = self.next_batch[i].to(self.device)

            images_batch, keypoints_3d_gt, keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_crop = self.next_batch

            # BRG to RGB
            images_batch = torch.flip(images_batch, [-1])
            # normalize image
            images_batch = (images_batch / 255.0 - self.mean) / self.std

            # absolute to relative 3D keypoints
            keypoints_3d_gt[:, :, 1:] -= keypoints_3d_gt[:, :, :1]
            keypoints_3d_gt[:, :, 0] = 0

            # training data augmentation (horizontal flipping)
            if random.random() <= 0.5 and self.is_train:
                images_batch = torch.flip(images_batch, [-2])

                keypoints_2d_batch_cpn[..., 0] *= -1
                keypoints_2d_batch_cpn[..., joints_left + joints_right, :] = keypoints_2d_batch_cpn[..., joints_right + joints_left, :]

                keypoints_2d_batch_cpn_crop[:, :, 0] = 192 - keypoints_2d_batch_cpn_crop[:, :, 0] - 1
                keypoints_2d_batch_cpn_crop[:, joints_left + joints_right] = keypoints_2d_batch_cpn_crop[:, joints_right + joints_left]

                keypoints_3d_gt[:, :, :, 0] *= -1
                keypoints_3d_gt[:, :, joints_left + joints_right] = keypoints_3d_gt[:, :, joints_right + joints_left]

            # testing data augmentation (horizontal flipping)
            if (not self.is_train) and self.flip_test:
                images_batch = torch.stack([images_batch, torch.flip(images_batch,[2])], dim=1)

                keypoints_2d_batch_cpn_flip = keypoints_2d_batch_cpn.clone()
                keypoints_2d_batch_cpn_flip[..., 0] *= -1
                keypoints_2d_batch_cpn_flip[..., joints_left + joints_right, :] = keypoints_2d_batch_cpn_flip[..., joints_right + joints_left, :]
                keypoints_2d_batch_cpn = torch.stack([keypoints_2d_batch_cpn, keypoints_2d_batch_cpn_flip], dim=1)

                keypoints_2d_batch_cpn_crop_flip = keypoints_2d_batch_cpn_crop.clone()
                keypoints_2d_batch_cpn_crop_flip[:, :, 0] = 192 - keypoints_2d_batch_cpn_crop_flip[:, :, 0] - 1
                keypoints_2d_batch_cpn_crop_flip[:, joints_left + joints_right] = keypoints_2d_batch_cpn_crop_flip[:, joints_right + joints_left]
                keypoints_2d_batch_cpn_crop = torch.stack([keypoints_2d_batch_cpn_crop, keypoints_2d_batch_cpn_crop_flip], dim=1)

                del keypoints_2d_batch_cpn_flip, keypoints_2d_batch_cpn_crop_flip

            self.next_batch = [images_batch.float(), keypoints_3d_gt.float(), keypoints_2d_batch_cpn.float(), keypoints_2d_batch_cpn_crop.float()]

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)