import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftLoULoss(nn.Module):
    def __init__(self):
        super(SoftLoULoss, self).__init__()

    def forward(self, pred, target):
        smooth = 0.1
        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        loss = (intersection_sum + smooth) / \
               (pred_sum + target_sum - intersection_sum + smooth)

        loss = 1 - torch.mean(loss)
        return loss


class SoftIoUL1NromLoss(nn.Module):
    def __init__(self, lambda_iou=0.8, lambda_l1=0.2):
        super(SoftIoUL1NromLoss, self).__init__()
        self.softiou = SoftLoULoss()
        self.lambda_iou = lambda_iou
        self.lambda_l1 = lambda_l1

    def forward(self, pred, target):
        iouloss = self.softiou(pred, target)

        batch_size, C, height, width = pred.size()
        pred = (pred > 0).float()
        l1loss = torch.sum(pred) / (batch_size * C * height * width)

        loss = self.lambda_iou * iouloss + self.lambda_l1 * l1loss
        return loss


class Heatmap_SoftIoU(nn.Module):
    def __init__(self, cfg):
        super(Heatmap_SoftIoU, self).__init__()
        self.softIouLoss = SoftLoULoss()
        self.cfg = cfg

    def forward(self, preds, label):
        labels = self.downsampleImg(label)
        loss1 = self.softIouLoss(preds[0], labels[2])
        loss2 = self.softIouLoss(preds[1], labels[1])
        loss3 = self.softIouLoss(preds[2], labels[0])
        loss = (loss1 + loss2 + loss3) / 3 
        return loss
    
    def downsampleImg(self, img):
        downsampler = nn.MaxPool2d(2, 2)

        downsampled_imgs = []
        downsampled_img = img

        for i in range(len(self.cfg["multiscalefeature_outchannel"]), 0, -1):
            downsampled_img = downsampler(downsampled_img)
            if i <= 3:
                downsampled_imgs.append(downsampled_img)
        
        return downsampled_imgs


class Heatmap_MSE(nn.Module):
    def __init__(self, cfg):
        super(Heatmap_MSE, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.cfg = cfg

    def forward(self, preds, label):
        # make the vague caused by augment transform clear
        label = (label > self.cfg["label_vague_threshold"]).type(torch.float32)
        labels = self.downsampleImg(label)
        loss1 = self.mse_loss(preds[0], labels[2])
        loss2 = self.mse_loss(preds[1], labels[1])
        loss3 = self.mse_loss(preds[2], labels[0])
        loss = (loss1 + loss2 + loss3) / 3 
        return loss
    
    def downsampleImg(self, img):
        downsampler = nn.AvgPool2d(2, 2)

        downsampled_imgs = []
        downsampled_img = img

        for i in range(len(self.cfg["multiscalefeature_outchannel"]), 0, -1):
            downsampled_img = downsampler(downsampled_img)
            if i <= 3:
                downsampled_imgs.append(downsampled_img)
        
        return downsampled_imgs


class ImageRecoverLoss(nn.Module):
    def __init__(self, loss_weight_per_scale):
        super(ImageRecoverLoss, self).__init__()
        self.weight = loss_weight_per_scale
        self.avg_pool = nn.AvgPool2d(2, 2, 0)
        

    def forward(self, generP, oriP):

        z = torch.cat((generP, oriP), 1)

        loss = 0
        for i in self.weight:
            z = self.avg_pool(z)
            loss_ = torch.abs(z[:,0,...] - z[:,1,...]).sum() / (z.shape[0] * z.shape[2] * z.shape[3])
            loss += loss_ * i
        
        return loss