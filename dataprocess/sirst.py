import torch

# import torch.nn as nn
import torch.utils.data as Data
import torchvision.transforms as transforms

import cv2
import os
import os.path as osp
import matplotlib.pyplot as plt

# import sys
import random
# import scipy.io as scio
from scipy.signal import convolve2d
import scipy.ndimage
import numpy as np

__all__ = ["SirstAugDataset", "IRSTD1kDataset", "NUDTDataset"]


class Rotate_4D_Transform:
    def __init__(self):
        self.angles = [0, 90, 180, 270]

    def __call__(self, img):
        random_idx = torch.randint(0, 1000, (1,))
        
        angle = self.angles[random_idx%4]
        rotated_img = self.__rotate__(img, angle)
        return rotated_img
    
    def __rotate__(self, img, angle):
        C, _, _ = img.shape
        if angle == 90:
            img = torch.transpose(img, -1, -2)
            img = torch.flip(img, dims=(-1,)) 
        elif angle == 180:
            img = torch.flip(img, dims=(-1, -2))
        elif angle == 270:
            img = torch.transpose(img, -1, -2)
            img = torch.flip(img, dims=(-2,))
        return img
        
class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
        return input.copy(), target.copy()

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
        base_size=256,
        pt_label=False,
        pseudo_label=False,
        augment=True,
        turn_num=0,
        target_mix = False,
        cfg=None
    ):
        assert mode in ["train", "test"]

        if mode == "train":
            self.data_dir = osp.join(base_dir, "trainval")
        elif mode == "test":
            self.data_dir = osp.join(base_dir, "test")
        else:
            raise NotImplementedError
        
        self.mode = mode
        self.cfg = cfg
        self.base_size = base_size
        self.pt_label = pt_label
        self.pseudo_label = pseudo_label
        self.aug = augment
        self.turn_num = turn_num
        self.target_mix = target_mix
        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, "images")):
            if filename.endswith("png"):
                self.names.append(filename)

        self.augment_test = transforms.Compose([
            transforms.Resize((self.base_size, self.base_size), interpolation=transforms.InterpolationMode.NEAREST)
        ])

        self.augment_train = transforms.Compose([
            transforms.RandomResizedCrop(
                base_size,
                scale=(0.8, 1.0)),  # 在给定的scale范围内随机缩放并裁剪
            transforms.RandomAffine(degrees=180, translate=(0.3, 0.3)),
            transforms.RandomHorizontalFlip(),  # 随机水平翻转
        ])
        self.gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)) if target_mix else None

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, "images", name)
        pseudo_label_path = osp.join(self.data_dir, f'pixel_pseudo_label{self.turn_num}', name)
        label_path = osp.join(self.data_dir, "masks", name)
        if self.pseudo_label:
            img, mask, pseudo_label = cv2.imread(img_path, 0), cv2.imread(label_path, 0), cv2.imread(pseudo_label_path, 0)

            img = torch.from_numpy(img).type(torch.float32)
            mask = torch.from_numpy(mask).type(torch.float32)
            pseudo_label = torch.from_numpy(pseudo_label).type(torch.float32)
            
            img, mask, pseudo_label = self.augment_test(img.unsqueeze(0)), self.augment_test(mask.unsqueeze(0)), self.augment_test(pseudo_label.unsqueeze(0))
            data = (img, mask, pseudo_label)
        else:
            img, mask= cv2.imread(img_path, 0), cv2.imread(label_path, 0)

            img = torch.from_numpy(img).type(torch.float32)
            mask = torch.from_numpy(mask).type(torch.float32)
            img, mask = self.augment_test(img.unsqueeze(0)), self.augment_test(mask.unsqueeze(0))

            data = (img, mask)
        data = torch.cat(data, dim=0) 

        data_aug = self.augment_train(data) if self.mode == "train" and self.aug else data

        data_aug = data_aug / 255.0

        data_aug = data_aug.unsqueeze(1)

        if self.pt_label:
            pt_label = self.__mask2point(data_aug[1])
            # data_aug = torch.concatenate((data_aug,pt_label.unsqueeze(0)),dim=0)
            data_aug[1] = pt_label
        # row_num = 4
        # col_num = 4
        # fig, axes = plt.subplots(row_num, col_num, figsize=(col_num*4, row_num*4))
        # for i in range(row_num):
        #     axes[i, 0].imshow(data_aug[0,0].numpy(), cmap='gray')
        #     axes[i, 1].imshow(data_aug[1,0].numpy(), cmap='gray')
        #     axes[i, 2].imshow(data_aug[2,0].numpy(), cmap='gray')
        #     axes[i, 3].imshow(data_aug[3,0].numpy(), cmap='gray')
        # plt.tight_layout()
        # plt.show()
        # a = input()
        elif self.target_mix:
            data_aug[0], data_aug[1] = self.__mix_target(data_aug[0], data_aug[1], i)
        if self.pseudo_label:
            return data_aug[0], data_aug[1], data_aug[2]
        else:
            return data_aug[0], data_aug[1]

    def __len__(self):
        return len(self.names)
    
    def __mask2point(self, mask):
        # 将mask转换为numpy数组以便处理
        mask_array = np.array(mask[0])
        # 使用连通组件分析找到所有独立的目标区域
        labels, num_features = scipy.ndimage.label(mask_array > 0.9)

        pts_label = torch.zeros_like(mask, dtype=torch.float32)

        for label_id in range(1, num_features + 1):
            # 获取当前连通组件的位置
            pos = np.where(labels == label_id)
            
            if len(pos[0]) == 0:
                continue 

            # 计算目标区域的边界框
            top_left_x = min(pos[1])
            top_left_y = min(pos[0])
            bottom_right_x = max(pos[1])
            bottom_right_y = max(pos[0])

            # 计算正方形的中心点, 并加随机数，产生手工标注的误差效果
            center_x = (top_left_x + bottom_right_x) // 2
            center_y = (top_left_y + bottom_right_y) // 2
            # center_x = (top_left_x + bottom_right_x) // 2 + torch.randint(-2, 3, (1,))
            # center_y = (top_left_y + bottom_right_y) // 2 + torch.randint(-2, 3, (1,))

            # center_x = np.clip(center_x, 0, 255)
            # center_y = np.clip(center_y, 0, 255)
            pts_label[0, center_y, center_x] = 1.0

        return pts_label

    def __mix_target(self, img, mask, idx):
        """
        Mix the target with image and mask.
        Target is from perferct generated pesudo label, with no dissociated pixels.
        Mixing includs the following steps:
        1. Find the proper position for the target where the img is complex. Furthermore, complex area means there are many edges.
        2. Mix the target with image in proper way which means the border of the target and background is smooth.
        3. Make mask according to the new and original target.
        """
        # Select random target
        target_path = osp.join(self.data_dir, "perfect_target")
        target_names = os.listdir(target_path)
        target_name = random.choice(target_names)
        target = cv2.imread(osp.join(target_path, target_name), 0)
        target = torch.from_numpy(target).type(torch.float32)
        target_blured = self.gaussian_blur(target.unsqueeze(0)).squeeze(0)

        edge_path = osp.join(self.data_dir, "canny_edge")
        name = self.names[idx]

        edge = cv2.imread(osp.join(edge_path, name), 0)
        edge = torch.from_numpy(edge).type(torch.float32)
        # Step 1
        # find a 32*32 area where the complexicity is mid-level.
        h_idx, w_idx = self.__random_position(edge)
        # mix the target with image(simple way)
        img[0, h_idx-16:h_idx+16, w_idx-16:w_idx+16] = target_blured[0,0]
        # mix the target with mask
        mask[0, h_idx-16:h_idx+16, w_idx-16:w_idx+16] = target
        
        return img, mask

    def __random_position(self, edge):
        """
        Randomly choose a position for the target.
        """
        edge_level = torch.nn.functional.avg_pool2d(edge.unsqueeze(0).unsqueeze(0), 2, stride=2)  # (1,1,256,256)
        edge_level = torch.nn.functional.avg_pool2d(edge_level, 2, stride=2)    # (1,1,128,128)
        edge_level = torch.nn.functional.avg_pool2d(edge_level, 2, stride=2)    # (1,1,64,64)

        edge_mean = edge_level.mean()
        edge_mid = (edge_level.max() + edge_level.min()) / 2
        condition = (edge_level > edge_mean) * (edge_level < edge_mid)  # ??? is it proper?
        _, _, H_idx, W_idx = torch.where(condition)
        print(H_idx.shape)
        
        random_idx = torch.randint(0, H_idx.shape[0], (1,))
        rh_idx, w_idx = int(H_idx[random_idx].item()), int(W_idx[random_idx].item())
        rh_idx, w_idx = rh_idx * 4 + 2, w_idx * 4 + 2
        while(rh_idx < 16 or rh_idx > 240 or w_idx < 16 or w_idx > 240):
            random_idx = torch.randint(0, H_idx.shape[0], (1,))
            rh_idx, w_idx = int(H_idx[random_idx].item()), int(W_idx[random_idx].item())
            rh_idx, w_idx = rh_idx * 4 + 2, w_idx * 4 + 2

        return rh_idx, w_idx



class NUDTDataset(Data.Dataset):
    """
    Return: Single channel
    """

    def __init__(
        self,
        base_dir=r"W:/DataSets/Infraid_datasets/NUDT-SIRST",
        mode="train",
        base_size=256,
        noise=False,
        cfg=None
    ):
        assert mode in ["train", "test"]

        if mode == "train":
            self.data_dir = osp.join(base_dir, "trainval")
        elif mode == "test":
            self.data_dir = osp.join(base_dir, "test")
        else:
            raise NotImplementedError
        self.mode = mode 
        self.base_size = base_size
        self.cfg = cfg
        self.noise = noise

        self.names = []
        for filename in os.listdir(osp.join(self.data_dir, "images")):
            if filename.endswith("png"):
                self.names.append(filename)

        self.augment = transforms.Compose([
                # transforms.RandomResizedCrop(base_size, scale=(0.8, 1.0)),  # 在给定的scale范围内随机缩放并裁剪
                transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), shear=0),
                transforms.RandomHorizontalFlip(),  # 随机水平翻转
            ])

    def __getitem__(self, i):
        name = self.names[i]
        img_path = osp.join(self.data_dir, "images", name)
        label_path = osp.join(self.data_dir, "masks", name)
        img, mask = cv2.imread(img_path, 0), cv2.imread(label_path, 0)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        data = torch.cat((img.unsqueeze(0), mask.unsqueeze(0)), dim=0)

        data_aug = self.augment(data) if self.mode == "train" else data 

        data_aug = data_aug / 255.0
        data_aug = data_aug.unsqueeze(1)

        if self.noise:
            noise_path = osp.join(self.data_dir, "noise32", name)
            noise = cv2.imread(noise_path, 0)
            noise = torch.from_numpy(noise).type(torch.float32) / 255.0
            noise = noise.unsqueeze(0)
            return data_aug[0], data_aug[1], noise

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
