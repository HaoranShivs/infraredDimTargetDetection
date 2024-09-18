import torch

# import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

import cv2
import os
import os.path as osp

from dataprocess.sirst import Rotate_4D_Transform

import numpy as np


class Crop_IRSTD1kDataset(Data.Dataset):
    """
    Return: Single channel
    """

    def __init__(self, base_dir, mode="train", base_size=32):
        assert mode in ["train", "test"]

        if mode == "train":
            self.data_dir = osp.join(base_dir, "trainval")
        elif mode == "test":
            self.data_dir = osp.join(base_dir, "test")
        else:
            raise NotImplementedError
        self.base_size = base_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, "crop_images", f"{base_size}")):
            if filename.endswith("png"):
                self.names.append(filename)

        if mode == "test":
            self.augment = transforms.Compose(
                [
                    # transforms.RandomResizedCrop(base_size, scale=(0.8, 1.0)),
                    transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),
                    transforms.RandomHorizontalFlip(),  # 随机水平翻转
                    Rotate_4D_Transform(),  # randomly rotate in angles: 0, 90, 180, 270
                ]
            )
        else:
            self.augment = transforms.Compose(
                [
                    # transforms.RandomResizedCrop(base_size, scale=(0.8, 1.0)),  # 在给定的scale范围内随机缩放并裁剪
                    transforms.RandomAffine(degrees=0, translate=(0.5, 0.5), shear=0),
                    transforms.RandomHorizontalFlip(),  # 随机水平翻转
                    Rotate_4D_Transform()
                ]
            )

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, "crop_images", f"{self.base_size}", name)
        label_path = osp.join(self.data_dir, "crop_masks", f"{self.base_size}", name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        data = torch.cat((img.unsqueeze(0), mask.unsqueeze(0)), dim=0)

        data_aug = self.augment(data)

        data_aug = data_aug / 255.0

        data_aug = data_aug.unsqueeze(1)

        if (self.base_size < 32):
            zeros_ = torch.zeros(2, 1, 32, 32)
            start_idx = 16-self.base_size//2
            zeros_[0,0,start_idx:-start_idx, start_idx:-start_idx] = data_aug[0,0]
            zeros_[1,0,start_idx:-start_idx, start_idx:-start_idx] = data_aug[1,0]
            data_aug = zeros_
        return data_aug[0], data_aug[1]

    def __len__(self):
        return len(self.names)


class Crop_NUDTDataset(Data.Dataset):
    """
    Return: Single channel
    """

    def __init__(self, base_dir, mode="train", base_size=256):
        assert mode in ["train", "test"]

        if mode == "train":
            self.data_dir = osp.join(base_dir, "trainval")
        elif mode == "test":
            self.data_dir = osp.join(base_dir, "test")
        else:
            raise NotImplementedError
        self.base_size = base_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, "crop_images", f"{base_size}")):
            if filename.endswith("png"):
                self.names.append(filename)

        if mode == "test":
            self.augment = transforms.Compose(
                [
                    # transforms.RandomResizedCrop(base_size, scale=(0.8, 1.0)),
                    transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),
                    transforms.RandomHorizontalFlip(),  # 随机水平翻转
                    Rotate_4D_Transform(),  # randomly rotate in angles: 0, 90, 180, 270
                ]
            )
        else:
            self.augment = transforms.Compose(
                [
                    # transforms.RandomResizedCrop(base_size, scale=(0.8, 1.0)),  # 在给定的scale范围内随机缩放并裁剪
                    transforms.RandomAffine(degrees=0, translate=(0.5, 0.5), shear=0),
                    transforms.RandomHorizontalFlip(),  # 随机水平翻转
                    Rotate_4D_Transform(),
                ]
            )

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, "crop_images", f"{self.base_size}", name)
        label_path = osp.join(self.data_dir, "crop_masks", f"{self.base_size}", name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        data = torch.cat((img.unsqueeze(0), mask.unsqueeze(0)), dim=0)

        data_aug = self.augment(data)

        data_aug = data_aug / 255.0

        data_aug = data_aug.unsqueeze(1)
        return data_aug[0], data_aug[1]

    def __len__(self):
        return len(self.names)
