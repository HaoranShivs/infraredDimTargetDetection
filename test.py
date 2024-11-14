import torch
import torch.fft
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as osp

from dataprocess.sirst import IRSTD1kDataset
# from dataprocess.sirst_point import IRSTD1kDataset
import torch.utils.data as Data
import yaml
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
import cv2

from net.attentionnet import attenMultiplyUNet_withloss

cfg_path = 'cfg.yaml'
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

trainset = IRSTD1kDataset(base_dir=r"W:/DataSets/ISTD/IRSTD-1k", mode="train", base_size=256, pseudo_label=True, cfg=cfg)
train_data_loader = Data.DataLoader(trainset, batch_size=32, shuffle=False, drop_last=False)


net_path = "W:/Projects/infraredDimTargetDetection/result/20241114T12-10-17_ws_GE_irstd1k/best.pkl"
net = attenMultiplyUNet_withloss(cfg, False)
net.load_state_dict(torch.load(net_path))
net = net.net.to('cuda')

origin_name = os.listdir("W:/DataSets/ISTD/IRSTD-1k/trainval/images")
idx = 0
pixel_pseudo_label_path = "W:/DataSets/ISTD/IRSTD-1k/trainval/pixel_pseudo_label2"
for j, (data, label) in enumerate(train_data_loader):
    data = data.to('cuda')
    preds, _, _, _ = net(data)
    preds = preds.cpu().detach()
    
    pseudo_label = (preds + label) / 2
    pseudo_label = pseudo_label * (pseudo_label > 0.5)
    
    pseudo_label = np.array(pseudo_label) * 256 # we know 'pseudo_label' is lower than 1.0, so we use 256
    for i in range(data.shape[0]):
        pseudo_label_image = Image.fromarray(pseudo_label[i, 0].astype(np.uint8), mode='L')  # 'L' 表示灰度模式
        pseudo_label_image.save(pixel_pseudo_label_path + '/' + origin_name[idx])
        idx += 1

    # label = label * (label > 0.1)
    # row_num = 4
    # col_num = 4
    # fig, axes = plt.subplots(row_num, col_num, figsize=(col_num*4, row_num*4))
    # for i in range(row_num):
    #     axes[i, 0].imshow(data[i,0], cmap='gray')
    #     # axes[i, 1].imshow(img_gradient_[i, 0], cmap='gray')
    #     # axes[i, 2].imshow(expanded_grad[i, 0], cmap='gray')
    #     # axes[i, 3].imshow(img_gradient_[i, 2], cmap='gray')
    #     # axes[i, 4].imshow(img_gradient_[i, 3], cmap='gray')
    #     # axes[i, 5].imshow(img_gradient_[i, 4], cmap='gray')
    #     # axes[i, 6].imshow(img_gradient_[i, 5], cmap='gray')
    #     # axes[i, 7].imshow(img_gradient_[i, 6], cmap='gray')
    #     # axes[i, 8].imshow(img_gradient_[i, 7], cmap='gray')
    #     axes[i, 1].imshow(label[i, 0], cmap='gray')
    #     axes[i, 2].imshow(preds[i, 0], cmap='gray')
    #     axes[i, 3].imshow(pseudo_label[i, 0], cmap='gray')
    # plt.tight_layout()
    # plt.show()
    # a = input()