import os
import pickle
import random

import cv2
import numpy as np
import torch
from mvn.utils.img import crop_image
from torchvision import transforms

retval = {
    'subject_names': ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11'],
    'camera_names': ['54138969', '55011271', '58860488', '60457274'],
    'action_names': [
        'Directions-1', 'Directions-2',
        'Discussion-1', 'Discussion-2',
        'Eating-1', 'Eating-2',
        'Greeting-1', 'Greeting-2',
        'Phoning-1', 'Phoning-2',
        'Posing-1', 'Posing-2',
        'Purchases-1', 'Purchases-2',
        'Sitting-1', 'Sitting-2',
        'SittingDown-1', 'SittingDown-2',
        'Smoking-1', 'Smoking-2',
        'TakingPhoto-1', 'TakingPhoto-2',
        'Waiting-1', 'Waiting-2',
        'Walking-1', 'Walking-2',
        'WalkingDog-1', 'WalkingDog-2',
        'WalkingTogether-1', 'WalkingTogether-2']
}

h36m_cameras_azimuth = [70, -70, 110, -110]


joints_left = [4, 5, 6, 11, 12, 13] 
joints_right = [1, 2, 3, 14, 15, 16]
 

def get_sample_train(sample):
    image, shot = sample

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_shape = (192, 256)

    image = np.array(image)
    image = crop_image(image, shot['center'], shot['scale'], image_shape)
    image = transform(image)

    keypoints_3d_gt = np.expand_dims(shot['joints_3d'], axis=0)

    keypoints_3d_gt[:, 1:, :] -= keypoints_3d_gt[:, :1, :]    
    keypoints_3d_gt[:, 0, :] = 0
    keypoints_3d_gt = torch.from_numpy(keypoints_3d_gt).float()

    keypoints_2d_cpn = shot['joints_2d_cpn']
    keypoints_2d_cpn = torch.from_numpy(keypoints_2d_cpn).float()
    keypoints_2d_cpn_crop = shot['joints_2d_cpn_crop']
    keypoints_2d_cpn_crop = torch.from_numpy(keypoints_2d_cpn_crop).float()

    if random.random() <= 0.5:
        image = torch.flip(image, [-1])

        keypoints_2d_cpn[:, 0] *= -1
        keypoints_2d_cpn[joints_left + joints_right, :] = keypoints_2d_cpn[joints_right + joints_left, :]

        keypoints_2d_cpn_crop[:, 0] = 192 - keypoints_2d_cpn_crop[:, 0] - 1
        keypoints_2d_cpn_crop[joints_left + joints_right, :] = keypoints_2d_cpn_crop[joints_right + joints_left, :]

        keypoints_3d_gt[:, :, 0] *= -1
        keypoints_3d_gt[:, joints_left + joints_right, :] = keypoints_3d_gt[:, joints_right + joints_left, :]

    return image, keypoints_3d_gt, keypoints_2d_cpn, keypoints_2d_cpn_crop


def get_sample_test(sample):
    image, shot = sample

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_shape = (192, 256)

    image = np.array(image)
    image = crop_image(image, shot['center'], shot['scale'], image_shape)
    image_numpy = image.copy()
    image = transform(image)

    keypoints_3d_gt = np.expand_dims(shot['joints_3d'], axis=0)
    keypoints_3d_gt[:, 1:, :] -= keypoints_3d_gt[:, :1, :]
    keypoints_3d_gt[:, 0, :] = 0
    keypoints_3d_gt = torch.from_numpy(keypoints_3d_gt).float()

    keypoints_2d_cpn = shot['joints_2d_cpn']
    keypoints_2d_cpn = torch.from_numpy(keypoints_2d_cpn).float()
    keypoints_2d_cpn_crop = shot['joints_2d_cpn_crop']
    keypoints_2d_cpn_crop = torch.from_numpy(keypoints_2d_cpn_crop).float()

    
    image_flip = torch.flip(image, [-1])
    image = torch.stack([image, image_flip], dim=0)

    keypoints_2d_cpn_flip = keypoints_2d_cpn.clone()
    keypoints_2d_cpn_flip[:, 0] *= -1
    keypoints_2d_cpn_flip[joints_left + joints_right, :] = keypoints_2d_cpn_flip[joints_right + joints_left, :]
    keypoints_2d_cpn = torch.stack([keypoints_2d_cpn, keypoints_2d_cpn_flip], dim=0)

    keypoints_2d_cpn_crop_flip = keypoints_2d_cpn_crop.clone()
    keypoints_2d_cpn_crop_flip[:, 0] = 192 - keypoints_2d_cpn_crop_flip[:, 0] - 1
    keypoints_2d_cpn_crop_flip[joints_left + joints_right, :] = keypoints_2d_cpn_crop_flip[joints_right + joints_left, :]
    keypoints_2d_cpn_crop = torch.stack([keypoints_2d_cpn_crop, keypoints_2d_cpn_crop_flip], dim=0)

    del keypoints_2d_cpn_flip, keypoints_2d_cpn_crop_flip
    
    return image, keypoints_3d_gt, keypoints_2d_cpn, keypoints_2d_cpn_crop , image_numpy #h36m_cameras_azimuth[shot['camera_id']] #shot['image']