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
        self.mode = mode
        if mode == "train":
            self.data_dir = osp.join(base_dir, "trainval")
        elif mode == "test":
            self.data_dir = osp.join(base_dir, "test")
        else:
            raise NotImplementedError
        self.base_size = base_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, "crop32images")):
            if filename.endswith("png"):
                self.names.append(filename)
        if mode == "train":
            self.noise_names = []
            for filename in os.listdir(osp.join(self.data_dir, "noise32")):
                if filename.endswith("png"):
                    self.noise_names.append(filename)

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
        img_path = osp.join(self.data_dir, "crop32images", name)
        label_path = osp.join(self.data_dir, "crop32masks", name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        data = torch.cat((img.unsqueeze(0), mask.unsqueeze(0)), dim=0)
        data = self.noisyImage(data, self.base_size, i)
        data_aug = self.augment(data)

        data_aug = data_aug / 255.0

        data_aug = data_aug.unsqueeze(1)

        return data_aug[0], data_aug[1]

    def __len__(self):
        return len(self.names)
    
    def noisyImage(self, img_label, target_size, noise_idx):
        img_size = img_label.shape[-1]
        if img_size == target_size:
            return img_label
        elif img_size > target_size:
            start_idx = img_size//2- target_size//2
            return img_label[..., start_idx:start_idx+target_size, start_idx:start_idx+target_size]
        padding = torch.zeros((*img_label.shape[:-2],target_size,target_size))
        if self.mode == "train":
            noise_name = self.noise_names[noise_idx%len(self.noise_names)]
            backg_path = osp.join(self.data_dir, "noise32", noise_name)
            backg = cv2.imread(backg_path, 0)
            backg = torch.from_numpy(backg).type(torch.FloatTensor)
            padding[0] = backg
        # randomly embeding the img to the noisy background
        indices = torch.randint(0, target_size-img_size,(2,))
        padding[0, indices[0]:indices[0]+img_size, indices[1]:indices[1]+img_size] = img_label[0]
        # same padding to the label
        padding[1, indices[0]:indices[0]+img_size, indices[1]:indices[1]+img_size] = img_label[1]

        return padding


class Crop_NUDTDataset(Data.Dataset):
    """
    Return: Single channel
    """

    def __init__(self, base_dir, mode="train", base_size=32):
        assert mode in ["train", "test"]
        self.mode = mode
        if mode == "train":
            self.data_dir = osp.join(base_dir, "trainval")
        elif mode == "test":
            self.data_dir = osp.join(base_dir, "test")
        else:
            raise NotImplementedError
        self.base_size = base_size

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, "crop32images")):
            if filename.endswith("png"):
                self.names.append(filename)
        if mode == "train":
            self.noise_names = []
            for filename in os.listdir(osp.join(self.data_dir, "noise32")):
                if filename.endswith("png"):
                    self.noise_names.append(filename)

        self.augment = transforms.Compose([
                transforms.RandomAffine(degrees=180, translate=(0.3, 0.3)),
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
            ]
        )

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, "crop32images", name)
        label_path = osp.join(self.data_dir, "crop32masks", name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)

        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)

        data = torch.cat((img.unsqueeze(0), mask.unsqueeze(0)), dim=0)
        data = self.noisyImage(data, self.base_size, i)
        data_aug = self.augment(data) if self.mode == "train" else data

        data_aug = data_aug / 255.0

        data_aug = data_aug.unsqueeze(1)

        return data_aug[0], data_aug[1]

    def __len__(self):
        return len(self.names)
    
    def noisyImage(self, img_label, target_size, noise_idx):
        img_size = img_label.shape[-1]
        if img_size == target_size:
            return img_label
        elif img_size > target_size:
            start_idx = img_size//2- target_size//2
            return img_label[..., start_idx:start_idx+target_size, start_idx:start_idx+target_size]
        padding = torch.zeros((*img_label.shape[:-2],target_size,target_size))
        if self.mode == "train":
            noise_name = self.noise_names[noise_idx%len(self.noise_names)]
            backg_path = osp.join(self.data_dir, "noise32", noise_name)
            backg = cv2.imread(backg_path, 0)
            backg = torch.from_numpy(backg).type(torch.FloatTensor)
            padding[0] = backg
        # randomly embeding the img to the noisy background
        indices = torch.randint(0, target_size-img_size,(2,))
        padding[0, indices[0]:indices[0]+img_size, indices[1]:indices[1]+img_size] = img_label[0]
        # same padding to the label
        padding[1, indices[0]:indices[0]+img_size, indices[1]:indices[1]+img_size] = img_label[1]

        return padding

