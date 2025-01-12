import torch
import torch.fft

from PIL import Image

import numpy as np
import os

from dataprocess.sirst import IRSTD1kDataset
# from dataprocess.sirst_point import IRSTD1kDataset
import torch.utils.data as Data
import yaml

from net.attentionnet import attenMultiplyUNet_withloss

cfg_path = 'cfg.yaml'
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)

trainset = IRSTD1kDataset(base_dir=r"W:/DataSets/ISTD/IRSTD-1k", mode="train", base_size=256, pseudo_label=True, cfg=cfg)
train_data_loader = Data.DataLoader(trainset, batch_size=32, shuffle=False, drop_last=False)

net_path = "W:/Projects/infraredDimTargetDetection/result/20241126T04-21-13_ws_atten344_irstd1k/best.pkl"
net = attenMultiplyUNet_withloss(cfg, False)
net.load_state_dict(torch.load(net_path))
net = net.net.to('cuda')
net = net.eval()

origin_name = os.listdir("W:/DataSets/ISTD/IRSTD-1k/trainval/images")
idx = 0
pixel_pseudo_label_path = "W:/DataSets/ISTD/IRSTD-1k/trainval/pixel_pseudo_label1.3.3"
for j, (data, pixel_label, pt_labela) in enumerate(train_data_loader):
    data = data.to('cuda')
    preds, _, _, _, _ = net(data)
    preds = preds.cpu().detach()
    
    pseudo_label = (preds + pixel_label) / 2
    pseudo_label_ = torch.zeros_like(pseudo_label)
    # 优化区域，使得每一个点标签对应的一个区域灰度值最大为1，最小为0
    B, _, S, _ = data.shape
    indices = torch.where(pt_labela > 0.)
    region_size = 32
    half_region_size = region_size // 2
    for b, _, s1, s2 in zip(*indices):
        # 计算区域的边界
        start_s1 = max(0, s1 - half_region_size)
        end_s1 = min(S, s1 + half_region_size)
        start_s2 = max(0, s2 - half_region_size)
        end_s2 = min(S, s2 + half_region_size)
        # 处理区域
        region = pseudo_label[b, 0, start_s1:end_s1, start_s2:end_s2]
        region = region-torch.min(region)/(torch.max(region) - torch.min(region))
        pseudo_label_[b, 0, start_s1:end_s1, start_s2:end_s2] = region

    pseudo_label = np.array(pseudo_label_) * 255
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