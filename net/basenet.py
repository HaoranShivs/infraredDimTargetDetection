import torch
import torch.nn as nn


class Conv2d_Bn_Relu(nn.Module):
    def __init__(
        self,
        in_channel: int = 32,
        out_channel: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
    ):
        super(Conv2d_Bn_Relu, self).__init__()
        self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv2d(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel: int = 32,
        hide_channel: int = 32,
        out_channel: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super(ResBlock, self).__init__()
        self.conv2d1 = nn.Conv2d(in_channel, hide_channel, kernel_size, stride, 1)
        self.conv2d2 = nn.Conv2d(hide_channel, out_channel, kernel_size, stride, 1)
        # self.conv2d3 = nn.Conv2d(out_channel, out_channel, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # self.bn3 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv2d1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2d2(x)
        x = self.bn2(x)
        # x = self.relu(x)
        # x = self.bn3(self.conv2d3(x))
        x = x + inputs
        x = self.relu(x)
        return x


class DeepFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channel: int = 32,
        out_channel: int = 32,
        kernel_size: int = 3,
        stride: int = 1,
    ):
        super(DeepFeatureExtractor, self).__init__()
        self.conv2d1 = Conv2d_Bn_Relu(in_channel, out_channel, kernel_size, stride, 1)
        self.conv2d2 = Conv2d_Bn_Relu(out_channel, out_channel, 1, 1)

    def forward(self, inputs):
        x = self.conv2d1(inputs)
        x = self.conv2d2(x)
        return x


class Resconv(nn.Module):
    def __init__(self, in_channel=1, out_channel=32, down_sampler=None):
        super(Resconv, self).__init__()

        self.block1 = ResBlock(in_channel, out_channel, out_channel)
        self.block2 = ResBlock(out_channel, out_channel, out_channel)
        self.block3 = ResBlock(out_channel, out_channel, out_channel)

        self.down_sampler = down_sampler

    def forward(self, img):
        """
        args:
            img(torch.tensor): (B, C1, H, W)
        output:
            reconstructed_img(torch.tensor): ( B, C2, H, W)
        """
        x = self.block1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.down_sampler(x) if self.down_sampler is not None else x
        return x


class MultiScaleFeatureNet(nn.Module):
    def __init__(self, in_channel=None, out_channel_list=None, downsampler=None):
        super(MultiScaleFeatureNet, self).__init__()
        if out_channel_list is None:
            raise ValueError("parameter 'out_channel_list' is not given")

        self.scale1 = nn.Sequential(
            Conv2d_Bn_Relu(in_channel, out_channel_list[0], 3, 1, 1),
            Conv2d_Bn_Relu(out_channel_list[0], out_channel_list[0], 3, 1, 1),
            downsampler,
        )

        self.scale2 = nn.Sequential(
            Conv2d_Bn_Relu(out_channel_list[0], out_channel_list[1], 3, 1, 1),
            Conv2d_Bn_Relu(out_channel_list[1], out_channel_list[1], 3, 1, 1),
            downsampler,
        )

        self.scale3 = nn.Sequential(
            Conv2d_Bn_Relu(out_channel_list[1], out_channel_list[2], 3, 1, 1),
            Conv2d_Bn_Relu(out_channel_list[2], out_channel_list[2], 3, 1, 1),
            downsampler,
        )

    def forward(self, shallow_feature):
        """
        args:
            shallow_feature(torch.tensor): (B, C1, H, W)
        output:
            (...,H/2, W/2)
            (...,H/4, W/4)
            (...,H/8, W/8)
        """
        x1 = self.scale1(shallow_feature)  # 64
        x2 = self.scale2(x1)  # 32
        x3 = self.scale3(x2)  # 16
        return x1, x2, x3


class FeatureFusionNet1(nn.Module):
    def __init__(self, in_channel_list=None, out_channel=256, ratio_list=None):
        super(FeatureFusionNet1, self).__init__()
        self.comp1 = Conv2d_Bn_Relu(in_channel_list[0], out_channel, 1, 1)
        self.comp2 = Conv2d_Bn_Relu(in_channel_list[1], out_channel, 1, 1)
        self.comp3 = Conv2d_Bn_Relu(in_channel_list[2], out_channel, 1, 1)
        self.comp4 = Conv2d_Bn_Relu(in_channel_list[3], out_channel, 1, 1)

        self.transform = Conv2d_Bn_Relu(out_channel, out_channel, 1, 1)

    def forward(self, feature1, feature2, feature3, feature4):
        f1 = self.comp1(feature1)
        f2 = self.comp2(feature2).repeat(1, 1, 2, 2)
        f3 = self.comp3(feature3).repeat(1, 1, 4, 4)
        f4 = self.comp4(feature4).repeat(1, 1, 8, 8)

        mixf = f1 + f2 + f3 + f4

        fusion_feature = self.transform(mixf)

        return fusion_feature


class FeatureFusionNet2(nn.Module):
    def __init__(self, in_channel_list=None, out_channel=256, ratio_list=None):
        super(FeatureFusionNet2, self).__init__()
        # sum(in_channel_list[-1:]), sum(in_channel_list[-1:])
        self.cf1 = DeepFeatureExtractor(sum(in_channel_list[:1]), sum(in_channel_list[:1]), 3, 1)
        self.cf2 = DeepFeatureExtractor(sum(in_channel_list[:2]), sum(in_channel_list[:2]), 3, 1)
        self.cf3 = DeepFeatureExtractor(sum(in_channel_list[:3]), sum(in_channel_list[:3]), 3, 1)
        self.cf4 = DeepFeatureExtractor(sum(in_channel_list[:4]), sum(in_channel_list[:4]), 3, 1)

        self.transform = Conv2d_Bn_Relu(sum(in_channel_list[:4]), out_channel, 1, 1)

    def forward(self, feature1, feature2, feature3, feature4):
        f1 = self.cf1(feature1)
        f2 = torch.cat((f1, feature2.repeat(1, 1, 2, 2)), dim=1)
        f2 = self.cf2(f2)
        f3 = torch.cat((f2, feature3.repeat(1, 1, 4, 4)), dim=1)
        f3 = self.cf3(f3)
        f4 = torch.cat((f3, feature4.repeat(1, 1, 8, 8)), dim=1)
        f4 = self.cf4(f4)

        fusion_feature = self.transform(f4)

        return fusion_feature


class DetectNet1(nn.Module):
    def __init__(self, in_channel=None, out_channel=None):
        super(DetectNet1, self).__init__()
        self.upscale = nn.Conv2d(in_channel, out_channel, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        B, C, H, W = feature.shape
        result = self.upscale(feature)
        result = result.reshape((B, 2, 2, H, W)).permute(0, 3, 1, 4, 2).reshape((B, 1, 2 * H, 2 * W))
        result = self.sigmoid(result)
        return result


class DetectNet2(nn.Module):
    def __init__(self, in_channel=None, out_channel=None):
        super(DetectNet2, self).__init__()
        self.upscale = nn.ConvTranspose2d(in_channel, out_channel, 3, 2, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        result = self.upscale(feature)
        result = self.sigmoid(result)
        return result


class DetectNet3(nn.Module):
    def __init__(self, in_channel=None, out_channel=None):
        super(DetectNet3, self).__init__()
        self.final = nn.Conv2d(in_channel, out_channel, 3, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        feature = feature.repeat(1, 1, 2, 2)
        result = self.final(feature)
        result = self.sigmoid(result)
        return result


class DetectNet4(nn.Module):
    def __init__(self, in_channel=None, out_channel=None):
        super(DetectNet4, self).__init__()
        # self.final1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1)
        self.final = nn.Conv2d(in_channel, out_channel, 1, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, feature):
        # feature = self.final1(feature)
        result = self.final(feature)
        result = self.sigmoid(result)
        return result


class BaseNet1(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(BaseNet1, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        self.resconv = Resconv(in_channel, cfg["resconv_outchannel"])

        downsampler = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"], cfg["multiscalefeature_outchannel"], downsampler
        )
        self.ffusion = FeatureFusionNet1(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"],
            cfg["featurefusion_outchannel"],
            cfg["ratio_list"],
        )
        self.detect = DetectNet1(cfg["featurefusion_outchannel"], cfg["channel4upscale"])

    def forward(self, img):
        x = self.resconv(img)
        x11, x21, x31 = self.multiscalef(x)
        xf1 = self.ffusion(x, x11, x21, x31)  # (B, C, 128, 128)
        result = self.detect(xf1)
        return result


class BaseNet2(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(BaseNet2, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        self.resconv = Resconv(in_channel, cfg["resconv_outchannel"])

        downsampler = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"], cfg["multiscalefeature_outchannel"], downsampler
        )
        self.ffusion = FeatureFusionNet2(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"],
            cfg["featurefusion_outchannel"],
            cfg["ratio_list"],
        )
        self.detect = DetectNet2(cfg["featurefusion_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)
        x11, x21, x31 = self.multiscalef(x)
        xf1 = self.ffusion(x, x11, x21, x31)  # (B, C, 128, 128)
        result = self.detect(xf1)
        return result


class LargeBaseNet(nn.Module):
    def __init__(self, in_channel: int = 1, cfg=None):
        super(LargeBaseNet, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler = nn.MaxPool2d(2, 2)
        self.resconv = Resconv(in_channel, cfg["resconv_outchannel"])

        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"], cfg["multiscalefeature_outchannel"], downsampler
        )
        self.ffusion = FeatureFusionNet2(
            [cfg["resconv_outchannel"]] + cfg["multiscalefeature_outchannel"],
            cfg["featurefusion_outchannel"],
            cfg["ratio_list"],
        )
        self.detect = DetectNet4(cfg["featurefusion_outchannel"], 1)

    def forward(self, img):
        x = self.resconv(img)  # (B, C, 256, 256)
        x11, x21, x31 = self.multiscalef(x)
        xf1 = self.ffusion(x, x11, x21, x31)  # (B, C, 256, 256)
        result = self.detect(xf1)
        return result
