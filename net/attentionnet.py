import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from net.basenet import Resconv, ShallowFeatureExtractor, DeepFeatureExtractor, UpScaler, Conv2d_Bn_Relu, DetectNet1, BaseNet3, ResBlock
from net.twotasknet import ConvT2d_Bn_Relu
from utils.loss import SoftLoULoss

# 设置pytorch打印选项
torch.set_printoptions(
    threshold=512,         # 最大显示元素数量为10
    # linewidth=320,        # 每行的最大字符数为120
    precision=4,          # 小数点后的数字精度为4
    edgeitems=20,          # 每行显示的边缘元素数量为5
    sci_mode=False        # 不使用科学计数法
)


class AttentionFromdeeper(nn.Module): 
    def __init__(self, in_channel):
        super(AttentionFromdeeper, self).__init__()
        self.conv1 = DeepFeatureExtractor(in_channel, in_channel, 3, 1)
        self.conv2 = nn.Conv2d(in_channel, 1, 3, 1, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        B, _, S, _ = inputs.shape
        x = self.conv1(inputs)
        x = self.conv2(x)
        res = self.normalize_tensor(self.softmax(x.view(B, 1, S * S))).view(B, 1, S, S)
        return res
    
    def normalize_tensor(self, tensor):
        """
        对形状为 [B, 1, S] 的张量进行归一化，最后一维张量的最大值为1，最小值为 0。
        
        参数:
        tensor (torch.Tensor): 形状为 [B, 1, S] 的张量
        
        返回:
        torch.Tensor: 归一化后的张量
        """
        # 获取最小值和最大值
        min_vals, _ = tensor.min(dim=-1, keepdim=True)
        max_vals, _ = tensor.max(dim=-1, keepdim=True)
        
        # 防止除零错误
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1e8  # 如果最大值和最小值相等，设为1避免除零错误
        
        # 归一化
        normalized_tensor = (tensor - min_vals) / range_vals
        
        return normalized_tensor


class ConvDownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvDownSample, self).__init__()
        self.conv1 = Conv2d_Bn_Relu(in_channel, out_channel, 3, 1, 1)
        self.conv2 = Conv2d_Bn_Relu(out_channel, out_channel, 3, 1, 1)
        self.conv_downsampler = Conv2d_Bn_Relu(out_channel, out_channel, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.conv_downsampler(x)
        return x


class ConvUpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvUpSample, self).__init__()
        # self.conv_upsampler = ConvT2d_Bn_Relu(in_channel, out_channel, 3)
        self.atten_proj = AttentionFromdeeper(in_channel)
        self.shallow_proj = DeepFeatureExtractor(out_channel, out_channel, 3, 1)
        self.conv = ShallowFeatureExtractor(out_channel, out_channel, 3, 1)
 
    def forward(self, x_deep, x_shallow=None, ratio=0.1):
        # x = self.conv_upsampler(x_deep)
        x_shallow = self.shallow_proj(x_shallow)
        atten = F.interpolate(self.atten_proj(x_deep), scale_factor=2, mode='bilinear')
        # x = torch.where(atten > 0.5, x_shallow, x)
        x = (atten*(1-ratio) + ratio) * x_shallow
        x = self.conv(x)
        return x, atten


class interact(nn.Module):
    def __init__(self, dist):
        super(interact, self).__init__()
        self.dist = dist

    def forward(self, feature):
        avg_pooled_feature = F.interpolate(F.avg_pool2d(feature, 3, 2, 1), scale_factor=2, mode='nearest')
        top_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=self.dist*2, dims=-2)).pow(2), dim=1, keepdim=True)
        top_right_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=[self.dist*2, -self.dist*2], dims=[-2, -1])).pow(2), dim=1, keepdim=True)
        top_left_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=[self.dist*2, self.dist*2], dims=[-2, -1])).pow(2), dim=1, keepdim=True)
        bot_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=-self.dist*2, dims=-2)).pow(2), dim=1, keepdim=True)
        bot_right_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=[-self.dist*2, -self.dist*2], dims=[-2, -1])).pow(2), dim=1, keepdim=True)
        bot_left_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=[-self.dist*2, self.dist*2], dims=[-2, -1])).pow(2), dim=1, keepdim=True)
        right_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=-self.dist*2, dims=-1)).pow(2), dim=1, keepdim=True)
        left_f = torch.sum((feature - torch.roll(avg_pooled_feature, shifts=self.dist*2, dims=-1)).pow(2), dim=1, keepdim=True)
        diff = torch.concatenate((top_left_f, top_f, top_right_f, left_f, right_f, bot_left_f, bot_f, bot_right_f), dim=1)
        diff = torch.max(diff, dim=1, keepdim=True).values
        return diff


# class attenMultiplyUNet(nn.Module):
#     def __init__(self, cfg):
#         super(attenMultiplyUNet, self).__init__()

#         channel_list = cfg["multiscalefeature_channellist"]

#         encoder_list = []
#         for i in range(1, len(channel_list)):
#             encoder_list.append(ConvDownSample(channel_list[i - 1], channel_list[i]))
#         self.encoder = nn.ModuleList(encoder_list)

#         decoder_list = []
#         for i in range(len(channel_list) - 1, 1, -1):
#             decoder_list.append(ConvUpSample(channel_list[i], channel_list[i - 1]))
#         decoder_list.append(ConvUpSample(channel_list[1], cfg["target_detection_size"]))
#         self.decoder = nn.ModuleList(decoder_list)
        
#         self.linear = Conv2d_Bn_Relu(cfg["target_detection_size"], 1, 1)
#         self.sigmoid = nn.Sigmoid()

#         # self.conv_debug = nn.Conv2d(channel_list[-1], 1, 1)

#     def forward(self, img):
#         features = [img,]
#         for i in range(len(self.encoder)):
#             features.append(self.encoder[i](features[i]))

#         print(features[-1])
#         # res = self.conv_debug(features[-1])
#         feature_num = len(features)
#         x_deep = features[feature_num-1]
#         for j in range(len(self.decoder)-1):
#             x_deep = self.decoder[j](x_deep, features[feature_num-2-j])
#         x = self.decoder[-1](x_deep)

#         res = self.sigmoid(self.linear(x))
#         return res


class attenMultiplyUNet(nn.Module):
    def __init__(self, cfg):
        super(attenMultiplyUNet, self).__init__()

        self.conv = Resconv(1, cfg["learning_conv_outchannel"])

        self.ds1 = ConvDownSample(cfg["learning_conv_outchannel"] , 32)  #128
        self.ds2 = ConvDownSample(32,32)  #64
        self.ds3 = ConvDownSample(32,64) #32
        self.ds4 = ConvDownSample(64,64) #16
        self.ds5 = ConvDownSample(64, 128)   #8
        self.ds6 = ConvDownSample(128, 128)  #4
        self.ds7 = ConvDownSample(128, 256)  #2

        self.us7 = ConvUpSample(256, 128)    #4
        self.us6 = ConvUpSample(128, 128)    #8
        self.us5 = ConvUpSample(128, 64)
        self.us4 = ConvUpSample(64, 64)
        self.us3 = ConvUpSample(64, 32)
        self.us2 = ConvUpSample(32, 32)
        self.us1 = ConvUpSample(32, cfg["learning_conv_outchannel"])

        self.linear = DetectNet1(cfg["learning_conv_outchannel"], 1)

    def forward(self, img):
        x_256 = self.conv(img)
        x_128 = self.ds1(x_256)
        x_64 = self.ds2(x_128)
        x_32 = self.ds3(x_64)
        x_16 = self.ds4(x_32)
        x_8 = self.ds5(x_16)
        x_4 = self.ds6(x_8)
        x_2 = self.ds7(x_4)

        seg_4 = self.us7(x_2, x_4)
        seg_8 = self.us6(seg_4, x_8)
        seg_16 = self.us5(seg_8, x_16)
        seg_32 = self.us4(seg_16, x_32)
        seg_64 = self.us3(seg_32, x_64)
        seg_128 = self.us2(seg_64, x_128)
        seg_256 = self.us1(seg_128, x_256)

        res = self.linear(seg_256)
        return res, seg_256


class attenMultiplyUNet2(nn.Module):
    def __init__(self, cfg):
        super(attenMultiplyUNet2, self).__init__()

        self.conv = ShallowFeatureExtractor(1, cfg["learning_conv_outchannel"])

        self.ds1 = nn.Sequential(ConvDownSample(cfg["learning_conv_outchannel"] , 32), ResBlock(32,32)) # 128
        self.ds2 = nn.Sequential(ConvDownSample(32 , 64), ResBlock(64,64))  # 64
        self.ds3 = nn.Sequential(ConvDownSample(64, 128), ResBlock(128,128))    # 32
        self.ds4 = nn.Sequential(ConvDownSample(128, 256), ResBlock(256,256))   # 16

        self.us4 = ConvUpSample(256, 128, 0.5)
        self.us3 = ConvUpSample(128, 64, 0.4)
        self.us2 = ConvUpSample(64, 32, 0.3)
        self.us1 = ConvUpSample(32, cfg["learning_conv_outchannel"], 0.2)

        self.linear = DetectNet1(cfg["learning_conv_outchannel"], 1)

    def forward(self, img):
        x_256 = self.conv(img)
        x_128 = self.ds1(x_256)
        x_64 = self.ds2(x_128)
        x_32 = self.ds3(x_64)
        x_16 = self.ds4(x_32)

        seg_32, atten_32 = self.us4(x_16, x_32)
        seg_64, atten_64 = self.us3(seg_32, x_64)
        seg_128, atten_128 = self.us2(seg_64, x_128)
        seg_256, atten_256 = self.us1(seg_128, x_256)

        res = self.linear(seg_256)

        return res, seg_256, (atten_32, atten_64, atten_128, atten_256)


class attenMultiplyUNet_withloss(nn.Module):
    def __init__(self, cfg, feature_map=False):
        super(attenMultiplyUNet_withloss, self).__init__()
        self.net = attenMultiplyUNet2(cfg)
        self.loss_fn = SoftLoULoss()
        self.cfg = cfg
        self.feature_map = feature_map

    def forward(self, img, label):
        res, _feature_map, atten_maps = self.net(img)
        loss = self.loss_fn(res, label)
        # 显示图片
        row_num = 4
        fig, axes = plt.subplots(row_num, 7, figsize=(7*3, row_num*3))
        for i in range(row_num):
            axes[i, 0].imshow(img[i,0].cpu().detach().numpy(), cmap='gray')
            axes[i, 1].imshow(atten_maps[0][i,0].cpu().detach().numpy(), cmap='gray')
            axes[i, 2].imshow(atten_maps[1][i,0].cpu().detach().numpy(), cmap='gray')
            axes[i, 3].imshow(atten_maps[2][i,0].cpu().detach().numpy(), cmap='gray')
            axes[i, 4].imshow(atten_maps[3][i,0].cpu().detach().numpy(), cmap='gray')
            axes[i, 5].imshow(res[i,0].cpu().detach().numpy(), cmap='gray')
            axes[i, 6].imshow(label[i,0].cpu().detach().numpy(), cmap='gray')
        plt.tight_layout()
        plt.show()
        a = input()
        if self.feature_map:
            return res, loss, _feature_map
        return res, loss
    
    def pseudolabel_point2segment(self, label, loss=1.0, feature_map=None):
        """ 
        according to the loss, we decide the weight of impacts of position and feature, to got a segment pseudo-label
        Args:
            loss:float
            label: point label, (B, 1, S, S)
            feature_map: (B, C, S, S)
        Returns:
            pseudo-label: (B, 1, S, S)
        """
        distances = self.__find_nearest_distance_to_highlights(label) + 1e-10
        pseudo_label = torch.clamp_max(distances ** -1 - loss + 0.5, 1)
        # if feature_map is not None:
        #     feature_distances = self.__compute_feature_distance(feature_map, label).to(label.device)    #(B, 1, S, S)
        #     feature_similirity_score = torch.exp(-feature_distances)
        #     # feature_similirity_score = (feature_similirity_score - torch.min(feature_similirity_score))/(torch.max)
        #     # print(feature_similirity_score)
        #     # a = input("feature_similirity_score show above")
        #     pseudo_label = feature_similirity_score * (1-loss) + pseudo_label * loss
        pseudo_label = (pseudo_label > 0.5).type(torch.float32)
        return pseudo_label
        
    def __find_nearest_distance_to_highlights(self, image_batch):
        B, _, S, _ = image_batch.size()
        # 创建一个张量来保存结果
        distance_map = torch.zeros_like(image_batch)
        
        for b in range(B):
            # 获取当前图像
            img = image_batch[b, 0]

            # 找到所有的亮点位置
            highlight_indices = torch.nonzero(img, as_tuple=True)
            
            # 如果没有亮点，跳过该图像
            if len(highlight_indices[0]) == 0:
                continue
            
            # 将亮点坐标转换为 (N, 2) 的张量
            highlight_coords = torch.stack(highlight_indices, dim=1)
            
            # 创建一个网格，表示图像中的所有像素点
            grid_y, grid_x = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')
            all_pixel_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).to(image_batch.device)
            
            # 计算所有像素点到所有亮点的距离
            # 使用广播机制来计算距离
            # all_pixel_coords: (S*S, 2)
            # highlight_coords: (N, 2)
            # distances: (S*S, N)
            distances = torch.cdist(all_pixel_coords.float(), highlight_coords.float())
            
            # 取最小距离
            min_distances = distances.min(dim=1).values
            
            # 将结果存入 distance_map
            distance_map[b, 0] = min_distances.view(S, S)

        return distance_map

    def __compute_feature_distance(self, feature_map, label):
        B, C, S, _ = feature_map.size()
        
        # 创建一个全零的张量来保存结果
        distance_map = torch.zeros(B, 1, S, S, device=feature_map.device)
        
        for b in range(B):
            # 获取当前图像和标签
            img_features = feature_map[b]  # (C, S, S)
            img_label = label[b, 0]  # (S, S)

            # 找到所有的亮点位置
            highlight_indices = torch.nonzero(img_label, as_tuple=True)  # (nonzero_count, 2)
            
            # 如果没有亮点，跳过该图像
            if len(highlight_indices[0]) == 0:
                continue
            
            # 提取亮点位置对应的特征向量
            reference_features = img_features[:, highlight_indices[0].cpu(), highlight_indices[1].cpu()]  # (C, nonzero_count)
            reference_features = reference_features.permute(1, 0)  # (nonzero_count, C)
            
            # 创建一个网格，表示图像中的所有像素点
            grid_y, grid_x = torch.meshgrid(torch.arange(S), torch.arange(S), indexing='ij')
            all_pixel_coords = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1)  # (S*S, 2)
            
            # 将所有像素点的特征向量展平为 (S*S, C)
            all_pixel_features = img_features.view(C, -1).permute(1, 0)  # (S*S, C)
            # print(all_pixel_features)
            # a = input("all_pixel_features show above")
            
            # 计算所有像素点到所有参考特征向量的距离
            # 使用广播机制来计算距离
            # all_pixel_features: (S*S, C)
            # reference_features: (nonzero_count, C)
            # distances: (S*S, nonzero_count)
            distances = torch.cdist(all_pixel_features, reference_features, p=2)
            
            # 取最小距离
            min_distances, _ = torch.min(distances, dim=1)  # (S*S,)
            
            # 将结果存入 distance_map
            distance_map[b, 0] = min_distances.view(S, S)

        return distance_map
