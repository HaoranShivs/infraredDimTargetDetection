import os
import os.path as osp
import time
import datetime
from argparse import ArgumentParser
from PIL import Image

import yaml
import numpy as np
import torch
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

# from models import get_model
from dataprocess.sirst import NUDTDataset, IRSTD1kDataset

from net.attentionnet import attenMultiplyUNet_withloss
from utils.lr_scheduler import *
from utils.evaluation import SegmentationMetricTPFNFP, my_PD_FA
from utils.logger import setup_logger


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description="Implement of BaseNet")

    #
    # Dataset parameters
    #
    parser.add_argument("--base-size", type=int, default=256, help="base size of images")
    parser.add_argument("--crop-size", type=int, default=256, help="crop size of images")
    parser.add_argument("--dataset", type=str, default="nudt", help="choose datasets")

    #
    # Training parameters
    #

    parser.add_argument("--batch-size", type=int, default=32, help="batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--warm-up-epochs", type=int, default=0, help="warm up epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--gpu", type=str, default="0", help="GPU number")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--lr-scheduler", type=str, default="poly", help="learning rate scheduler")

    #
    # Net parameters
    #
    parser.add_argument("--net-name", type=str, default="rpcanet", help="net name: fcn")
    parser.add_argument("--model-path", type=str, default="", help="load model path")
    parser.add_argument("--model-path1", type=str, default="", help="load model path")
    parser.add_argument("--model-path2", type=str, default="", help="load model path")
    # Rank parameters
    #
    # parser.add_argument('--rank', type=int, default=8,
    #                     help='rank number')

    #
    # Save parameters
    #
    parser.add_argument("--save-iter-step", type=int, default=10, help="save model per step iters")
    parser.add_argument("--log-per-iter", type=int, default=1, help="interval of logging")
    parser.add_argument("--base-dir", type=str, default="./result", help="saving dir")

    #
    # Configuration
    #
    parser.add_argument("--cfg-path", type=str, default="./cfg.yaml", help="path of cfg file")

    args = parser.parse_args()

    # Save folders
    # args.base_dir = r'D:\WFY\dun_irstd\result'
    args.time_name = time.strftime("%Y%m%dT%H-%M-%S", time.localtime(time.time()))
    args.folder_name = "{}_{}_{}".format(args.time_name, args.net_name, args.dataset)
    args.save_folder = osp.join(args.base_dir, args.folder_name)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # seed
    if args.seed != 0:
        set_seeds(args.seed)

    # logger
    args.logger = setup_logger("BaseNet test", args.save_folder, 0, filename="log.txt", mode="a")
    return args


def set_seeds(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, args):
        self.args = args

        ## cfg file
        with open(args.cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        with open(osp.join(self.args.save_folder, "cfg.yaml"), "w", encoding="utf-8") as file:
            yaml.dump(self.cfg, file, allow_unicode=True)

        ## GPU
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

        # model
        # net = BaseNet4(1, self.cfg)
        # loss_fn = SoftLoULoss()
        self.net = attenMultiplyUNet_withloss(self.cfg, False)

        # self.net.apply(self.weight_init)
        self.net = self.net.to(self.device)

        ## evaluation metrics
        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.best_miou = 0
        self.best_fmeasure = 0
        self.best_prec = 0
        self.best_recall = 0
        self.eval_loss = 0  # tmp values
        self.miou = 0
        self.fmeasure = 0
        self.eval_my_PD_FA = my_PD_FA()

        ## log info
        self.logger = args.logger
        self.logger.info(args)
        self.logger.info("Using device: {}".format(self.device))

    def train(self, epochs):
        iter_per_epoch = len(self.train_data_loader)
        max_iter = epochs * iter_per_epoch
        iter_num = 0
        # training step
        start_time = time.time()
        base_log = (
            "Epoch-Iter: [{:03d}/{:03d}]-[{:03d}/{:03d}-{:03d}]  || Lr: {:.6f} ||  Loss: {:.4f}={:.4f}+{:.4f}+{:.4f} || "
            "Cost Time: {} || Estimated Time: {}"
        )
        for epoch in range(epochs):
            for i, (data, _, label) in enumerate(self.train_data_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                label = (label > self.cfg["label_vague_threshold"]).type(torch.float32)

                _, softIoU_loss, class_loss, detail_loss, loss_128 = self.net(data, label)
                total_loss = softIoU_loss + class_loss + detail_loss + loss_128

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # for name, param in self.net.named_parameters():
                #     if "net.linear" in name:
                #         print(f"Gradient for {name}: {param.grad}")

                iter_num += 1

                cost_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                eta_seconds = ((time.time() - start_time) / iter_num) * (max_iter - iter_num)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                self.writer.add_scalar("Train Loss/Loss All", np.mean(total_loss.item()), iter_num)
                # self.writer.add_scalar("Train Loss/Loss SoftIoU", np.mean(loss_softiou.item()), iter_num)
                # self.writer.add_scalar('Train Loss/Loss MSE', np.mean(loss_mse.item()), iter_num)
                self.writer.add_scalar("Learning rate/", trainer.optimizer.param_groups[0]["lr"], iter_num)

                if iter_num % self.args.log_per_iter == 0:
                    self.logger.info(
                        base_log.format(
                            epoch + 1,
                            args.epochs,
                            iter_num % iter_per_epoch,
                            iter_per_epoch,
                            self.turn_epoch,
                            self.optimizer.param_groups[0]["lr"],
                            total_loss.item(),
                            softIoU_loss.item(),
                            class_loss.item(),
                            detail_loss.item(),
                            cost_string,
                            eta_string,
                        )
                    )

                if iter_num % iter_per_epoch == 0:
                    self.net.eval()
                    self.validation(epoch)
                    self.net.train()
                    self.scheduler(self.optimizer, i, epoch, None)

    def infer(self):
        trainset = IRSTD1kDataset(
            base_dir=r"W:/DataSets/ISTD/IRSTD-1k",
            mode="train",
            base_size=256,
            pt_label=True,
            pseudo_label=True,
            augment=False,
            turn_num=self.turn_epoch,
            cfg=self.cfg,
        )
        train_data_loader = Data.DataLoader(trainset, batch_size=32, shuffle=False, drop_last=False)
        
        net = self.net.net.eval()

        origin_name = os.listdir("W:/DataSets/ISTD/IRSTD-1k/trainval/images")
        idx = 0
        pixel_pseudo_label_path = f'W:/DataSets/ISTD/IRSTD-1k/trainval/pixel_pseudo_label{self.turn_epoch+1}'
        if not os.path.exists(pixel_pseudo_label_path):
            os.makedirs(pixel_pseudo_label_path)

        for j, (data, pt_labela, pixel_label) in enumerate(train_data_loader):
            data = data.to("cuda")
            preds, _, _, _, _ = net(data)
            preds = preds.cpu().detach()

            pseudo_label = (preds + pixel_label) / 2
            pseudo_label_ = torch.zeros_like(pseudo_label)
            # 优化区域，使得每一个点标签对应的一个区域灰度值最大为1，最小为0
            B, _, S, _ = data.shape
            indices = torch.where(pt_labela > 0.0)
            region_size = 16
            half_region_size = region_size // 2
            for b, _, s1, s2 in zip(*indices): 
                # 计算区域的边界
                start_s1 = max(0, s1 - half_region_size)
                end_s1 = min(S, s1 + half_region_size)
                start_s2 = max(0, s2 - half_region_size)
                end_s2 = min(S, s2 + half_region_size)
                # 处理区域
                region = pseudo_label[b, 0, start_s1:end_s1, start_s2:end_s2]
                region_ = (region - torch.min(region)) / (torch.max(region) - torch.min(region))
                pseudo_label_[b, 0, start_s1:end_s1, start_s2:end_s2] = region_

                # 显示结果
                print(torch.max(region_))
                plt.figure(figsize=(12, 6))
                plt.subplot(121), plt.imshow(region, cmap='gray')
                plt.subplot(122), plt.imshow(region_, cmap='gray')
                plt.show()
                a= input()
            
            # row_num = 4
            # col_num = 5
            # fig, axes = plt.subplots(row_num, col_num, figsize=(col_num*4, row_num*4))
            # for i in range(row_num):
            #     axes[i, 0].imshow(data[i,0].cpu().detach().numpy(), cmap='gray')
            #     axes[i, 1].imshow(pt_labela[i, 0].cpu().detach().numpy(), cmap='gray')
            #     axes[i, 2].imshow(pixel_label[i, 0].cpu().detach().numpy(), cmap='gray')
            #     axes[i, 3].imshow(preds[i, 0].cpu().detach().numpy(), cmap='gray')
            #     axes[i, 4].imshow(pseudo_label_[i, 0].cpu().detach().numpy(), cmap='gray')
            # plt.tight_layout()
            # plt.show()
            # a = input()

            pseudo_label = np.array(pseudo_label_) * 255
            for i in range(data.shape[0]):
                pseudo_label_image = Image.fromarray(pseudo_label[i, 0].astype(np.uint8), mode="L")  # 'L' 表示灰度模式
                pseudo_label_image.save(pixel_pseudo_label_path + "/" + origin_name[idx])
                idx += 1
        net = net.train()

    def training(self):
        self.turn_epoch = 0
        # training record
        path = osp.join(self.args.save_folder, f"{self.turn_epoch}")
        if not os.path.exists(path):
            os.makedirs(path)
        ## SummaryWriter
        self.writer = SummaryWriter(log_dir=path)
        self.writer.add_text(osp.join(self.args.folder_name, f"{self.turn_epoch}"), "Args:%s, " % self.args)
        ## dataset
        if args.dataset == "nudt":
            trainset = NUDTDataset(
                base_dir=r"W:/DataSets/ISTD/NUDT-SIRST", mode="train", base_size=args.base_size, cfg=self.cfg
            )
            valset = NUDTDataset(
                base_dir=r"W:/DataSets/ISTD/NUDT-SIRST", mode="test", base_size=args.base_size, cfg=self.cfg
            )
        # elif args.dataset == 'sirstaug':
        #     trainset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
        #                                mode='train', base_size=args.base_size)  # base_dir=r'E:\ztf\datasets\sirst_aug'
        #     valset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
        #                              mode='test', base_size=args.base_size)  # base_dir=r'E:\ztf\datasets\sirst_aug'
        elif args.dataset == "irstd1k":
            trainset = IRSTD1kDataset(
                base_dir=r"W:/DataSets/ISTD/IRSTD-1k",
                mode="train",
                base_size=args.base_size,
                pseudo_label=True,
                turn_num=0,
                cfg=self.cfg,
            )
            valset = IRSTD1kDataset(
                base_dir=r"W:/DataSets/ISTD/IRSTD-1k", mode="test", base_size=args.base_size, cfg=self.cfg
            )
        else:
            raise NotImplementedError

        self.train_data_loader = Data.DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2
        )
        self.val_data_loader = Data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2
        )

        ## lr scheduler
        self.scheduler = LR_Scheduler_Head(args.lr_scheduler, 0.001, 200, len(self.train_data_loader), lr_step=10)

        ## optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.infer()

        self.train(1)

        net_path = osp.join(self.args.save_folder, f"{self.turn_epoch}", "best.pkl")
        self.net.load_state_dict(torch.load(net_path))

        self.infer()

        for cycle_epoch in range(1, self.cfg["cycle_num"]):
            self.turn_epoch = cycle_epoch
            # training record
            path = osp.join(self.args.save_folder, f"{self.turn_epoch}")
            if not os.path.exists(path):
                os.makedirs(path)
            ## SummaryWriter
            self.writer = SummaryWriter(log_dir=path)
            self.writer.add_text(osp.join(self.args.folder_name, f"{self.turn_epoch}"), "Args:%s, " % self.args)
            # dataset
            if args.dataset == "nudt":
                trainset = NUDTDataset(
                    base_dir=r"W:/DataSets/ISTD/NUDT-SIRST", mode="train", base_size=args.base_size, cfg=self.cfg
                )
            # elif args.dataset == 'sirstaug':
            #     trainset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
            #                                mode='train', base_size=args.base_size)  # base_dir=r'E:\ztf\datasets\sirst_aug'
            elif args.dataset == "irstd1k":
                trainset = IRSTD1kDataset(
                    base_dir=r"W:/DataSets/ISTD/IRSTD-1k",
                    mode="train",
                    base_size=args.base_size,
                    cfg=self.cfg,
                    turn_num=self.turn_epoch,
                    pseudo_label=False,
                )
            else:
                raise NotImplementedError

            self.train_data_loader = Data.DataLoader(
                trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2
            )

            ## lr scheduler
            self.scheduler = LR_Scheduler_Head(args.lr_scheduler, 0.0001, 100, len(self.train_data_loader), lr_step=10)

            ## optimizer
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
            self.infer()

            self.train(1)

            # net_path = osp.join(self.args.save_folder, f"{self.turn_epoch}", "best.pkl")
            # self.net.load_state_dict(torch.load(net_path))

            self.infer()

    def validation(self, epoch):
        self.metric.reset()
        # self.eval_my_PD_FA.reset()
        base_log = "Data: {:s}, mIoU: {:.4f}/{:.4f}, prec: {:.4f}/{:.4f}, recall: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f} "
        # base_log = "Data: {:s}, mIoU: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f}, Pd:{:.4f}, Fa:{:.8f} "
        for i, (data, labels) in enumerate(self.val_data_loader):
            with torch.no_grad():
                # noise = torch.zeros((data.shape[0], 1, 32, 32), device=data.device)
                pred, _, _, _, _ = self.net.net(data.to(self.device))
            out_T = pred.cpu()

            labels = (labels > self.cfg["label_vague_threshold"]).type(torch.float32)
            self.metric.update(labels, out_T)
        miou_all, prec_all, recall_all, fmeasure_all = self.metric.get()

        torch.save(self.net.state_dict(), osp.join(self.args.save_folder, f"{self.turn_epoch}", "latest.pkl"))
        if miou_all > self.best_miou:
            self.best_miou = miou_all
            torch.save(self.net.state_dict(), osp.join(self.args.save_folder, f"{self.turn_epoch}", "best.pkl"))
        if fmeasure_all > self.best_fmeasure:
            self.best_fmeasure = fmeasure_all
        if prec_all > self.best_prec:
            self.best_prec = prec_all
        if recall_all > self.best_recall:
            self.best_recall = recall_all

        self.writer.add_scalar("Test/mIoU", miou_all, epoch)
        self.writer.add_scalar("Test/F1", fmeasure_all, epoch)
        self.writer.add_scalar("Best/mIoU", self.best_miou, epoch)
        self.writer.add_scalar("Best/Fmeasure", self.best_fmeasure, epoch)

        self.logger.info(
            base_log.format(
                self.args.dataset,
                miou_all,
                self.best_miou,
                prec_all,
                self.best_prec,
                recall_all,
                self.best_recall,
                fmeasure_all,
                self.best_fmeasure,
            )
        )

    def load_model(self, model_path: str = "", model_path1: str = "", model_path2: str = ""):
        if model_path != "":
            model_path = osp.join(model_path, "best.pkl")
            self.net.load_state_dict(torch.load(model_path))
        if model_path1 != "":
            model_path1 = osp.join(model_path1, "best.pkl")
            self.net.net_heatmap.load_state_dict(torch.load(model_path1))
        if model_path2 != "":
            model_path2 = osp.join(model_path2, "best.pkl")
            self.net.net_localseg.load_state_dict(torch.load(model_path2))


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)
    trainer.load_model(args.model_path, args.model_path1, args.model_path2)
    trainer.training()

    # print('Best mIoU: %.5f, Best Fmeasure: %.5f\n\n' % (trainer.best_miou, trainer.best_fmeasure))
