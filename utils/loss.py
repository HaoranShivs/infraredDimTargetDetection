import torch
import torch.nn as nn
import torch.nn.functional as F


class  SoftLoULoss(nn.Module):
    def __init__(self):
        super(SoftLoULoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
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


def downsampleImg(img, cfg_dict):
    downsampler = nn.MaxPool2d(2, 2)

    downsampled_imgs = []
    downsampled_img = img

    for i in range(len(cfg_dict["multiscalefeature_outchannel"]), 0, -1):
        downsampled_img = downsampler(downsampled_img)
        if i <= 3:
            downsampled_imgs.append(downsampled_img)
    
    return downsampled_imgs


class Heatmap_SoftIoU(nn.Module):
    def __init__(self):
        super(Heatmap_SoftIoU, self).__init__()
        self.softIouLoss = SoftLoULoss()

    def forward(self, pred, target):
        loss1 = self.softIouLoss(pred[0], target[2])
        loss2 = self.softIouLoss(pred[1], target[1])
        loss3 = self.softIouLoss(pred[2], target[0])
        loss = (loss1 + loss2 + loss3) / 3 
        return loss


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