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
from dataprocess.croped_sirst import Crop_IRSTD1kDataset, Crop_NUDTDataset
# from net.basenet import BaseNet1, BaseNet2, BaseNet3, LargeBaseNet, LargeBaseNet2, GaussNet, GaussNet2, GaussNet3, GaussNet4, SigmoidNet
from net.twotasknet import Heatmap_net, LocalSegment, HeatMaptoImg, UnitLabels
from utils.loss import SoftLoULoss, ImageRecoverLoss, Heatmap_SoftIoU, downsampleImg
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
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.iter_num = 0

        ## cfg file
        with open(args.cfg_path) as f:
            self.cfg = yaml.safe_load(f)
        with open(osp.join(self.args.save_folder, "cfg.yaml"), "w", encoding="utf-8") as file:
            yaml.dump(self.cfg, file, allow_unicode=True)

        # dataset
        if args.dataset == "nudt":
            trainset = NUDTDataset(
                base_dir=r"W:/DataSets/ISTD/NUDT-SIRST", mode="train", base_size=args.base_size
            )
            valset = NUDTDataset(
                base_dir=r"W:/DataSets/ISTD/NUDT-SIRST", mode="test", base_size=args.base_size
            )
        # elif args.dataset == 'sirstaug':
        #     trainset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
        #                                mode='train', base_size=args.base_size)  # base_dir=r'E:\ztf\datasets\sirst_aug'
        #     valset = SirstAugDataset(base_dir=r'./datasets/sirst_aug',
        #                              mode='test', base_size=args.base_size)  # base_dir=r'E:\ztf\datasets\sirst_aug'
        elif args.dataset == "irstd1k":
            trainset = IRSTD1kDataset(
                base_dir=r"W:/DataSets/ISTD/IRSTD-1k", mode="train", base_size=args.base_size, cfg=self.cfg
            )
            valset = IRSTD1kDataset(
                base_dir=r"W:/DataSets/ISTD/IRSTD-1k", mode="test", base_size=args.base_size, cfg=self.cfg
            )
        # if args.dataset == "irstd1k":
        #     trainset = Crop_IRSTD1kDataset(
        #         base_dir=r"W:/DataSets/ISTD/IRSTD-1k", mode="train", base_size=args.base_size
        #     )
        #     valset = Crop_IRSTD1kDataset(
        #         base_dir=r"W:/DataSets/ISTD/IRSTD-1k", mode="test", base_size=args.base_size
        #     )
        # elif args.dataset == "nudt":
        #     trainset = Crop_NUDTDataset(
        #         base_dir=r"W:/DataSets/ISTD/NUDT-SIRST", mode="train", base_size=args.base_size
        #     )
        #     valset = Crop_NUDTDataset(
        #         base_dir=r"W:/DataSets/ISTD/NUDT-SIRST", mode="test", base_size=args.base_size
        #     )
        else:
            raise NotImplementedError

        self.train_data_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        self.val_data_loader = Data.DataLoader(valset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        self.iter_per_epoch = len(self.train_data_loader)
        self.max_iter = args.epochs * self.iter_per_epoch

        ## GPU
        if torch.cuda.is_available():
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        self.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

        ## model
        self.net_heatmap = Heatmap_net(cfg=self.cfg)

        self.net_localseg = LocalSegment(cfg=self.cfg)

        # self.net.apply(self.weight_init)
        self.net_heatmap = self.net_heatmap.to(self.device)

        self.net_localseg = self.net_localseg.to(self.device)

        ## criterion
        self.heatmap_softiou = Heatmap_SoftIoU()
        self.softiou = SoftLoULoss()
        # self.recov_loss = ImageRecoverLoss(self.cfg["multiscale_recov_loss_weight"])
        # self.mse = torch.nn.MSELoss()

        ## lr scheduler
        self.scheduler = LR_Scheduler_Head(
            args.lr_scheduler, args.lr, args.epochs, len(self.train_data_loader), lr_step=10
        )

        ## optimizer
        # self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=args.learning_rate,
        #                                  momentum=0.9, weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(list(self.net_heatmap.parameters()) +list( self.net_localseg.parameters()), lr=args.lr)
        # self.optimizer = torch.optim.Adam(self.net_heatmap.parameters(), lr=args.lr)

        ## evaluation metrics
        self.metric = SegmentationMetricTPFNFP(nclass=1)
        self.best_miou = 0
        self.best_fmeasure = 0
        self.best_prec = 0
        self.best_recall = 1
        self.eval_loss = 0  # tmp values
        self.miou = 0
        self.fmeasure = 0
        self.eval_my_PD_FA = my_PD_FA()

        # # SummaryWriter
        self.writer = SummaryWriter(log_dir=args.save_folder)
        self.writer.add_text(args.folder_name, "Args:%s, " % args)

        ## log info
        self.logger = args.logger
        self.logger.info(args)
        self.logger.info("Using device: {}".format(self.device))

    def training(self):
        # training step
        start_time = time.time()
        base_log = "Epoch-Iter: [{:03d}/{:03d}]-[{:03d}/{:03d}]  || Lr: {:.6f} || Loss: {:.4f}={:.4f}+{:.4f} || " \
                   "Cost Time: {} || Estimated Time: {}"
        for epoch in range(args.epochs):
            for i, (data, labels) in enumerate(self.train_data_loader):
                data = data.to(self.device)
                labels = labels.to(self.device)

                target = self.net_heatmap(data)

                labels = (labels > 0.01).type(torch.float32)
                multiScale_labels = downsampleImg(labels, self.cfg)

                loss_heatmap = self.heatmap_softiou(target, multiScale_labels)

                multiscale_imgs, bboxs = HeatMaptoImg(target, data, 0.1)

                for i in range(len(multiscale_imgs)-1, -1, -1):
                    for j in range(len(multiscale_imgs[0])):
                        for z in range(multiscale_imgs[i][j].shape[0]):
                            ori_pict = np.array(data.cpu())
                            ori_pict[j,0, bboxs[i][j][z][0]:bboxs[i][j][z][2], bboxs[i][j][z][1]:bboxs[i][j][z][3]] = 0 
                            target_pict = np.array(multiscale_imgs[i][j][z].cpu())

                            fig, axes = plt.subplots(2)
                            axes[0].imshow(ori_pict[j,0], cmap='gray')
                            axes[0].axis('off')  # 关闭坐标轴显示
                                
                            axes[1].imshow(target_pict[0], cmap='gray')
                            axes[1].axis('off')  # 同样关闭坐标轴显示
                                # plt.title('Blurred Image')
                            plt.tight_layout()
                            plt.show()
                            a = input()
                
                pred = self.net_localseg(multiscale_imgs)

                pred = UnitLabels(pred, bboxs, 256)

                loss_softiou = self.softiou(pred, labels)

                # loss_softiou = torch.tensor([0,], device=loss_heatmap.device)
                loss_all = loss_softiou + loss_heatmap

                self.optimizer.zero_grad()
                # loss_softiou.backward()
                loss_all.backward()
                self.optimizer.step()

                # for name, param in self.net.named_parameters():
                #     if "heatmap.heatmaps" in name:
                #         self.logger.info(f"{name} 's grad: {param.grad} at epoch {epoch}, step {i}")

                self.iter_num += 1

                cost_string = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                eta_seconds = ((time.time() - start_time) / self.iter_num) * (self.max_iter - self.iter_num)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                self.writer.add_scalar('Train Loss/Loss All', np.mean(loss_all.item()), self.iter_num)
                # self.writer.add_scalar("Train Loss/Loss SoftIoU", np.mean(loss_softiou.item()), self.iter_num)
                # self.writer.add_scalar('Train Loss/Loss MSE', np.mean(loss_mse.item()), self.iter_num)
                self.writer.add_scalar(
                    "Learning rate/", trainer.optimizer.param_groups[0]["lr"], self.iter_num
                )

                if self.iter_num % self.args.log_per_iter == 0:
                    self.logger.info(
                        base_log.format(
                            epoch + 1,
                            args.epochs,
                            self.iter_num % self.iter_per_epoch,
                            self.iter_per_epoch,
                            self.optimizer.param_groups[0]["lr"],
                            loss_all.item(), loss_softiou.item(), loss_heatmap.item(),
                            cost_string,
                            eta_string,
                        )
                    )

                if self.iter_num % self.iter_per_epoch == 0:
                    self.net_heatmap.eval()
                    self.validation()
                    self.net_heatmap.train()
                    self.scheduler(self.optimizer, i, epoch, None)

    def validation(self):
        self.metric.reset()
        # self.eval_my_PD_FA.reset()
        base_log = "Data: {:s}, mIoU: {:.4f}/{:.4f}, prec: {:.4f}/{:.4f}, recall: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f} "
        # base_log = "Data: {:s}, mIoU: {:.4f}/{:.4f}, F1: {:.4f}/{:.4f}, Pd:{:.4f}, Fa:{:.8f} "
        for i, (data, labels) in enumerate(self.val_data_loader):
            with torch.no_grad():
                # generP, target = self.net(data.to(self.device))
                target = self.net_heatmap(data.to(self.device))
            # out_D, out_T = out_D.cpu(), out_T.cpu()
            # out_T = target.cpu()

            # loss_softiou = self.softiou(out_T, labels)
            # loss_mse = self.mse(out_D, data)
            # gamma = torch.Tensor([0.1]).to(self.device)
            # loss_all = loss_softiou + torch.mul(gamma, loss_mse)
            
            labels = (labels > 0.5).type(torch.float32)
            labels = downsampleImg(labels, self.cfg)
            miou_all, prec_all, recall_all, fmeasure_all = 0., 0., 0., 0.
            for i in range(len(labels)):
                self.metric.update(labels[i], target[2-i].cpu())
                miou, prec, recall, fmeasure = self.metric.get()
                miou_all += miou
                prec_all += prec
                recall_all += recall
                fmeasure_all += fmeasure
            miou_all, prec_all, recall_all, fmeasure_all = miou_all/3, prec_all/3, recall_all/3, fmeasure_all/3
        torch.save(self.net_heatmap.state_dict(), osp.join(self.args.save_folder, "latest.pkl"))
        # torch.save(self.net.state_dict(), osp.join(self.args.save_folder, "latest.pkl"))
        if miou_all > self.best_miou:
            self.best_miou = miou_all
            torch.save(self.net_heatmap.state_dict(), osp.join(self.args.save_folder, "best.pkl"))
            # torch.save(self.net.state_dict(), osp.join(self.args.save_folder, "best.pkl"))
        if fmeasure_all > self.best_fmeasure:
            self.best_fmeasure = fmeasure_all
        if prec_all > self.best_prec:
            self.best_prec = prec_all
        if recall_all < self.best_recall:
            self.best_recall = recall_all

        # print(miou, self.best_miou, fmeasure, self.best_fmeasure)

        self.writer.add_scalar("Test/mIoU", miou, self.iter_num)
        self.writer.add_scalar("Test/F1", fmeasure, self.iter_num)
        self.writer.add_scalar("Best/mIoU", self.best_miou, self.iter_num)
        self.writer.add_scalar("Best/Fmeasure", self.best_fmeasure, self.iter_num)

        self.logger.info(
            base_log.format(self.args.dataset, miou_all, self.best_miou, prec_all, self.best_prec, recall_all, self.best_recall, fmeasure_all, self.best_fmeasure)
        )

    def load_model(self, model_path1: str = "", model_path2: str = ""):
        if model_path1 != "":
            model_path1 = osp.join(model_path1, "best.pkl")
            self.net_heatmap.load_state_dict(torch.load(model_path1))
        if model_path2 != "":
            model_path2 = osp.join(model_path2, "best.pkl")
            self.net_localseg.load_state_dict(torch.load(model_path2))


if __name__ == "__main__":
    args = parse_args()

    trainer = Trainer(args)
    trainer.load_model(args.model_path1, args.model_path2)
    trainer.training()

    # print('Best mIoU: %.5f, Best Fmeasure: %.5f\n\n' % (trainer.best_miou, trainer.best_fmeasure))
