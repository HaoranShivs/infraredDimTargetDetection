import torch

# import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

import cv2
import os
import os.path as osp

# import sys
# import random
# import scipy.io as scio
from scipy.signal import convolve2d
import numpy as np

__all__ = ["SirstAugDataset", "IRSTD1kDataset", "NUDTDataset"]

# class SirstAugDataset(Data.Dataset):
#     '''
#     Return: Single channel
#     '''
#     def __init__(self, base_dir=r'/Users/tianfangzhang/Program/DATASETS/sirst_aug',
#                  mode='train', base_size=256):
#         assert mode in ['train', 'test']

#         if mode == 'train':
#             self.data_dir = osp.join(base_dir, 'trainval')
#         elif mode == 'test':
#             self.data_dir = osp.join(base_dir, 'test')
#         else:
#             raise NotImplementedError

#         self.base_size = base_size

#         self.names = []
#         for filename in os.listdir(osp.join(self.data_dir, 'images')):
#             if filename.endswith('png'):
#                 self.names.append(filename)
#         self.tranform = augumentation()
#         # self.transform = transforms.Compose([
#         #     transforms.ToTensor(),
#         #     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),  # Default mean and std
#         # ])

#     def __getitem__(self, i):
#         name = self.names[i]
#         img_path = osp.join(self.data_dir, 'images', name)
#         label_path = osp.join(self.data_dir, 'masks', name)

#         img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
#         img, mask = self.tranform(img, mask)
#         img = img.reshape(1, self.base_size, self.base_size) / 255.
#         if np.max(mask) > 0:
#             mask = mask.reshape(1, self.base_size, self.base_size) / np.max(mask)
#         else:
#             mask = mask.reshape(1, self.base_size, self.base_size)
#         # row, col = img.shape
#         # img = img.reshape(1, row, col) / 255.
#         # if np.max(mask) > 0:
#         #     mask = mask.reshape(1, row, col) / np.max(mask)
#         # else:
#         #     mask = mask.reshape(1, row, col)

#         img = torch.from_numpy(img).type(torch.FloatTensor)
#         mask = torch.from_numpy(mask).type(torch.FloatTensor)
#         return img, mask

#     def __len__(self):
#         return len(self.names)


class IRSTD1kDataset(Data.Dataset):
    """
    Return: Single channel
    """

    def __init__(
        self,
        base_dir=r"W:/DataSets/Infraid_datasets/IRSTD-1k",
        mode="train",
        mask_blurred=False,
        base_size=256,
    ):
        assert mode in ["train", "test"]

        if mode == "train":
            self.data_dir = osp.join(base_dir, "trainval")
        elif mode == "test":
            self.data_dir = osp.join(base_dir, "test")
        else:
            raise NotImplementedError
        self.base_size = base_size
        self.mask_blurred = mask_blurred
        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, "images")):
            if filename.endswith("png"):
                self.names.append(filename)

        self.augment = transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0),
                transforms.RandomResizedCrop(
                    base_size,
                    scale=(0.8, 1.0)
                ),  # 在给定的scale范围内随机缩放并裁剪
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
            ]
        )

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, "images", name)
        label_path = osp.join(self.data_dir, "masks", name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)

        # img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, [self.base_size, self.base_size], interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        data = torch.cat((img.unsqueeze(0), mask.unsqueeze(0)), dim=0)

        data_aug = self.augment(data)

        data_aug = data_aug / 255.0

        if self.mask_blurred:
            mask = gaussian_filter(np.array(data_aug[1]), sigma=1, kernel_size=5)
            data_aug[1] = torch.from_numpy(mask)
        data_aug = data_aug.unsqueeze(1)
        return data_aug[0], data_aug[1]

    def __len__(self):
        return len(self.names)


class NUDTDataset(Data.Dataset):
    """
    Return: Single channel
    """

    def __init__(
        self,
        base_dir=r"W:/DataSets/Infraid_datasets/NUDT-SIRST",
        mode="train",
        mask_blurred=False,
        base_size=256,
    ):
        assert mode in ["train", "test"]

        if mode == "train":
            self.data_dir = osp.join(base_dir, "trainval")
        elif mode == "test":
            self.data_dir = osp.join(base_dir, "test")
        else:
            raise NotImplementedError
        self.base_size = base_size
        self.mask_blurred = mask_blurred

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, "images")):
            if filename.endswith("png"):
                self.names.append(filename)

        self.augment = transforms.Compose(
            [
                transforms.RandomRotation(30),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0),
                transforms.RandomResizedCrop(
                    base_size, scale=(0.8, 1.0)
                ),  # 在给定的scale范围内随机缩放并裁剪
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
            ]
        )

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, "images", name)
        label_path = osp.join(self.data_dir, "masks", name)

        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        # img = cv2.resize(img, [self.base_size, self.base_size], interpolation=cv2.INTER_LINEAR)
        # mask = cv2.resize(mask, [self.base_size, self.base_size], interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        data = torch.cat((img.unsqueeze(0), mask.unsqueeze(0)), dim=0)

        data_aug = self.augment(data)

        data_aug = data_aug / 255.0

        if self.mask_blurred:
            mask = gaussian_filter(np.array(data_aug[1]), sigma=1, kernel_size=5)
            data_aug[1] = torch.from_numpy(mask)
        data_aug = data_aug.unsqueeze(1)
        return data_aug[0], data_aug[1]

    def __len__(self):
        return len(self.names)


def gaussian_filter(image, sigma=1, kernel_size=None):
    """
    deploy guass filter on the input image.

    args:
    image(numpy.array):
    sigma(float):
    kernel_size(int): must be odd num, the size of the guass kernel.
    """

    if kernel_size is None:
        # 如果没有指定kernel_size，则按照之前的逻辑计算
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    else:
        # 确保kernel_size为奇数
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number.")

    # 创建网格坐标系
    x, y = np.mgrid[
        -kernel_size // 2 + 1 : kernel_size // 2 + 1, -kernel_size // 2 + 1 : kernel_size // 2 + 1
    ]
    # 计算二维高斯函数的值
    g = np.exp(-(x**2 + y**2) / (2.0 * sigma**2))
    # 归一化高斯核
    g /= g.sum()
    # 应用高斯滤波
    pre_shape = image.shape[:-2]
    image = np.reshape(image, (image.shape[-2:]))

    blurred_image = convolve2d(image, g, mode="same", boundary="fill", fillvalue=0)
    blurred_image = np.reshape(blurred_image, (*pre_shape, *(blurred_image.shape)))
    return blurred_image
