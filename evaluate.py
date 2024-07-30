import os
import os.path as osp
import time
import datetime
from argparse import ArgumentParser

import yaml
import numpy as np
import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

# from models import get_model
from dataprocess.sirst import NUDTDataset, IRSTD1kDataset
from net.basenet import BaseNet1, BaseNet2, LargeBaseNet, LargeBaseNet2, BaseNet3, GaussNet
from utils.loss import SoftLoULoss
from utils.lr_scheduler import *
from utils.evaluation import SegmentationMetricTPFNFP, my_PD_FA
from utils.logger import setup_logger


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Evaluation of BaseNet')

    parser.add_argument('--gpu', type=str, default='0', help='GPU number')

    #
    # Dataset parameters
    #
    parser.add_argument('--base-size', type=int, default=256, help='base size of images')
    parser.add_argument('--crop-size', type=int, default=256, help='crop size of images')
    parser.add_argument('--dataset', type=str, default='nudt', help='choose datasets')

    #
    # Net path
    #
    parser.add_argument('--model-path', type=str, default='result/20240620T10-46-34_basenet_16-32-48-64-128_nudt',
                        help='net name: fcn')

    args = parser.parse_args()

    return args


class Evaluate(object):
    def __init__(self, args):
        self.args = args

        ## cfg file
        cfg_path = osp.join(args.model_path, 'cfg.yaml')
        with open(cfg_path) as f:
            self.cfg = yaml.safe_load(f)

        ## dataset
        if args.dataset == 'nudt':
            valset = NUDTDataset(base_dir=r'W:/DataSets/ISTD/NUDT-SIRST', mode='test', base_size=args.base_size)
        # elif args.dataset == 'sirstaug':
        #     trainset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
        #                                mode='train', base_size=args.base_size)  # base_dir=r'E:\ztf\datasets\sirst_aug'
        #     valset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
        #                              mode='test', base_size=args.base_size)  # base_dir=r'E:\ztf\datasets\sirst_aug'
        elif args.dataset == 'irstd1k':
            valset = IRSTD1kDataset(base_dir=r'W:/DataSets/ISTD/IRSTD-1k', mode='test', base_size=args.base_size) # base_dir=r'E:\ztf\datasets\IRSTD-1k'
        else:
            raise NotImplementedError

        self.val_data_loader = Data.DataLoader(valset, batch_size=32, shuffle=True)

        ## GPU
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

        ## model
        self.net = GaussNet(cfg=self.cfg)

        ## load_model
        model_path = osp.join(args.model_path, 'best.pkl')
        self.net.load_state_dict(torch.load(model_path))
        self.net = self.net.to(self.device)

        self.softiou = SoftLoULoss()

        ## evaluation metrics
        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.best_miou = 0
        self.best_fmeasure = 0
        self.eval_loss = 0  # tmp values
        self.miou = 0
        self.fmeasure = 0
        self.eval_my_PD_FA = my_PD_FA()

        ## visualization
        self.best_ori_pict = None
        self.best_seg_pict = None
        self.best_seg_label = None
    

    def validation(self):
        self.metric.reset()
        loss = torch.tensor([1,], dtype=torch.float32)
        for i, (data, labels) in enumerate(self.val_data_loader):
            with torch.no_grad():
                y_hat = self.net(data.to(self.device))
            # out_D, out_T = out_D.cpu(), out_T.cpu()
            out_T = y_hat.cpu()

            labels = (labels > 0.5).type(torch.float32)
            loss_softiou = self.softiou(out_T, labels)
            if (loss_softiou < loss):
                self.best_ori_pict = data.cpu()
                self.best_seg_pict = out_T
                self.best_seg_label = labels
                loss = loss_softiou
            # loss_mse = self.mse(out_D, data)
            # gamma = torch.Tensor([0.1]).to(self.device)
            # loss_all = loss_softiou + torch.mul(gamma, loss_mse)

            self.metric.update(labels, out_T)


        miou, prec, recall, fmeasure = self.metric.get()
        if miou > self.best_miou:
            self.best_miou = miou
        if fmeasure > self.best_fmeasure:
            self.best_fmeasure = fmeasure

        print(miou, self.best_miou, fmeasure, self.best_fmeasure)

    def visualize(self):
        oripict = np.array(self.best_ori_pict) * 255
        segpict = np.array(self.best_seg_pict) * 255
        label = np.array(self.best_seg_label) * 255

        n = segpict.shape[0]  # 图片对的数量
        shape = label.shape[-2:]

        # 创建一个n行2列的子图网格
        fig, axes = plt.subplots(n, 3, figsize=(6, 3*n))  # 调整figsize以适应你的需要

        for i in range(n):
            # 在当前行的第一列显示第1张图片
            axes[i, 0].imshow(oripict[i].reshape(*shape), cmap='gray')
            axes[i, 0].axis('off')  # 关闭坐标轴显示
            
            # 在当前行的第一列显示第2张图片
            axes[i, 1].imshow(segpict[i].reshape(*shape), cmap='gray')
            axes[i, 1].axis('off')  # 关闭坐标轴显示
            
            # 在当前行的第二列显示第3张图片
            axes[i, 2].imshow(label[i].reshape(*shape), cmap='gray')
            axes[i, 2].axis('off')  # 同样关闭坐标轴显示

        # 显示图表
        plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

        # 保存图表、
        save_path= osp.join(self.args.model_path, args.dataset + '_result.png')
        plt.savefig(save_path)


if __name__ == '__main__':
    args = parse_args()

    trainer = Evaluate(args)
    trainer.validation()
    trainer.visualize()