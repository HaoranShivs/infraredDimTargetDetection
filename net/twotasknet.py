import torch
import torch.nn as nn

from net.basenet import Resconv, ShallowFeatureExtractor, DeepFeatureExtractor, UpScaler, Conv2d_Bn_Relu, DetectNet1


class MultiScaleFeatureNet(nn.Module):
    def __init__(self, in_channel, out_channel_list, downsampler=None):
        super(MultiScaleFeatureNet, self).__init__()

        out_channel_list = [in_channel] + out_channel_list

        pyramid_layers = []
        for i in range(1, len(out_channel_list)):
            layer = ShallowFeatureExtractor(out_channel_list[i - 1], out_channel_list[i], 3, 1, downsampler)
            pyramid_layers.append(layer)

        self.pyramid = nn.ModuleList(pyramid_layers)

    def forward(self, shallow_feature):
        """
        args:
            shallow_feature(torch.tensor): (B, C1, H, W)
        output:
            (...,H/2, W/2)
            (...,H/4, W/4)
            (...,H/8, W/8)
            ...
        """
        outputs = []
        x = shallow_feature
        for i in self.pyramid:
            x = i(x)
            outputs.append(x)
        return outputs


class FusionNet_upscale(nn.Module):
    def __init__(self, in_channel_list):
        super(FusionNet_upscale, self).__init__()

        SRer_layers = []
        for i in range(len(in_channel_list) - 1, 0, -1):
            SRer_layers.append(UpScaler(in_channel_list[i], in_channel_list[i - 1], 3))

        self.SRer = nn.ModuleList(SRer_layers)

    def forward(self, inputs):
        if len(inputs) != len(self.SRer) + 1:
            print(len(inputs), len(self.SRer))
            raise ValueError("inputs of FusionNet is not same with cfg['multiscalefeature_outchannel']")

        outputs = []
        xd = inputs[len(inputs) - 1]
        for i in range(0, len(self.SRer)):
            xs = inputs[len(inputs) - 2 - i]
            xd = self.SRer[i](xd, xs)
            outputs.append(xd)
        return outputs


class Heatmap(nn.Module):
    def __init__(self, in_channel_list):
        super(Heatmap, self).__init__()

        heatmaps = []
        for i in range(len(in_channel_list) - 1, -1, -1):
            heatmaps.append(
                nn.Sequential(
                    DeepFeatureExtractor(in_channel_list[i], in_channel_list[i], 3),
                    nn.Conv2d(in_channel_list[i], 1, 1),
                )
            )

        self.heatmaps = nn.ModuleList(heatmaps)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        if len(inputs) != len(self.heatmaps):
            raise ValueError("inputs of FusionNet is not same with cfg['multiscalefeature_outchannel']")

        outputs = []
        for i in range(0, len(self.heatmaps)):
            x = inputs[i]
            res = self.heatmaps[i](x)
            res = self.sigmoid(res)
            outputs.append(res)
        return outputs


class Heatmap_net(nn.Module):
    """
    args:
        in_channel:
        cfg: dict of configuration information
    inputs:
        img: (B, 1, H, W)
    output:
        (B, 1, 8, 8)
        (B, 1, 16, 16)
        (B, 1, 32, 32)
    """

    def __init__(self, cfg=None):
        super(Heatmap_net, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        downsampler1 = nn.MaxPool2d(3, 2, 1)
        self.resconv = Resconv(1, cfg["resconv_outchannel"], downsampler1)

        downsampler2 = nn.MaxPool2d(2, 2)
        self.multiscalef = MultiScaleFeatureNet(
            cfg["resconv_outchannel"],
            cfg["multiscalefeature_outchannel"],
            downsampler2,
        )
        self.ffusion = FusionNet_upscale(cfg["multiscalefeature_outchannel"][-4:])
        self.heatmap = Heatmap(cfg["multiscalefeature_outchannel"][-4:-1])

    def forward(self, img):
        x = self.resconv(img)
        outputs_f = self.multiscalef(x)
        xf1 = self.ffusion(outputs_f[-4:])  # (B, 32, 256, 256)
        target = self.heatmap(xf1)
        return target


def HeatMaptoImg(heatmap, img, thre):
    """
    optimize the heatmap to decrese the overlap area of different scale of heatmap.
    Args:
        heatmap:bool
            (B, 1, 8, 8)
            (B, 1, 16, 16)
            (B, 1, 32, 32)
        img:(B, 1, H, W)
    Returns:
        cutted_imgs:
            [scale:(32,16,8)[B batchs[(M, 32, 32)or(N, 16, 16)or(X, 8, 8)]]]
        bboxs:
            [scale:[batchs:M tensor(x1, y1, x2, y2)]](32, 16, 8)
    """
    # sift area
    sumer = sum_Conv(2, 2, 0)
    sumer = sumer.to(img.device)
    for i in range(len(heatmap) - 2, -1, -1):
        heatmap_ = heatmap[i + 1] > thre
        heatmap_ = heatmap_.type(torch.float32)
        heatmap[i] = heatmap[i] > thre
        heatmap_ = sumer(heatmap_)
        heatmap[i] = heatmap[i] & torch.where(
            heatmap_ == 1, torch.zeros_like(heatmap[i], dtype=torch.bool), torch.ones_like(heatmap[i], dtype=torch.bool)
        )
        heatmap_ = heatmap_.repeat(1, 1, 2, 2)
        heatmap[i + 1] = heatmap[i + 1] > thre
        heatmap[i + 1] = heatmap[i + 1] & torch.where(
            heatmap_ > 1,
            torch.zeros_like(heatmap[i + 1], dtype=torch.bool),
            torch.ones_like(heatmap[i + 1], dtype=torch.bool),
        )

    # bbox [scale:[batchs:tensor(x1, y1, x2, y2)]]
    B, _, _, _ = heatmap[0].shape
    bboxs = [[[] for i in range(B)] for j in range(len(heatmap))]
    scale = 8
    for i in range(len(heatmap) - 1, -1, -1):
        batch_indices, _, row_indices, col_indices = torch.where(heatmap[i] == 1)
        for idx in range(batch_indices.shape[0]):
            coor = (
                torch.tensor([row_indices[idx], col_indices[idx], row_indices[idx] + 1, col_indices[idx] + 1]) * scale
            )
            bboxs[i][batch_indices[idx]].append(coor)
        scale = scale * 2

    # cutting out the target area [scale:[batch:, M*(H*W)]]
    multiscale = [[[] for i in range(B)] for j in range(len(heatmap))]
    for i in range(len(bboxs)):
        for j in range(B):
            for z in bboxs[i][j]:
                target_area = img[j, 0, z[0] : z[2], z[1] : z[3]].unsqueeze(0).unsqueeze(0)
                multiscale[i][j].append(target_area)
            multiscale[i][j] = torch.concatenate(multiscale[i][j], 0) if len(multiscale[i][j]) > 0 else torch.tensor([])

    return multiscale, bboxs


def UnifyShape(img, to_size=32):
    """
    reunit pieces of labels into one.
    Args:
        img:
            (B, M, 1, 8, 8)
            (B, N, 1, 16, 16)
            (B, Y, 1, 32, 32)
        to_size:
        label_size: int
    Returns:
        img:
            all croped_img transformed to 32*32.(num, 1, 32, 32)
        shape_idx:
            original shape of img, to make sure we reconstrcut the same consitution of img.
    """
    S, B = len(img), len(img[0])
    shape_idx = [[] for i in range(S)]
    img_tensor = torch.zeros((1, 1, to_size, to_size), device="cuda")
    for i in range(S):
        for j in range(B):
            if img[i][j].shape[0] == 0:
                shape_idx[i].append({"num": 0, "row_idx": 0, "col_idx": 0})
                continue
            num, _, size, _ = img[i][j].shape
            if size < to_size:
                zeros_ = torch.zeros((num, 1, to_size, to_size), device="cuda")
                start_idx = torch.randint(0, to_size - size, (2,), dtype=torch.int64)  # (row_idx, col_idx)
                zeros_[:, 0:, start_idx[0] : size + start_idx[0], start_idx[1] : size + start_idx[1]] = img[i][j]
                img_tensor = torch.concatenate((img_tensor, zeros_), dim=0)
                shape_idx[i].append({"num": num, "row_idx": start_idx[0], "col_idx": start_idx[1]})
            else:
                shape_idx[i].append({"num": num, "row_idx": 0, "col_idx": 0})
                img_tensor = torch.concatenate((img_tensor, img[i][j]), dim=0)
    return img_tensor[1:], shape_idx


def UnitLabels(multiscale_labels, bboxs, label_size):
    """
    reunit pieces of labels into one.
    Args:
        multiscale_labels:bool
            (B, M, 1, 8, 8)
            (B, N, 1, 16, 16)
            (B, Y, 1, 32, 32)
        bboxs:
            [scale:[batchs:M tensor(x1, y1, x2, y2)]]
        label_size: int
    Returns:
        label:
            (B,1,label_size,label_size)
    """
    B = len(multiscale_labels[0])
    label = torch.zeros(B, 1, label_size, label_size, device="cuda")
    for i in range(len(multiscale_labels) - 1, -1, -1):
        for j in range(B):
            for z in range(len(multiscale_labels[i][j])):
                label[
                    j, 0, bboxs[i][j][z][1] : bboxs[i][j][z][3], bboxs[i][j][z][0] : bboxs[i][j][z][2]
                ] += multiscale_labels[i][j][z, 0]
    return label


class sum_Conv(nn.Module):
    """
    sum all value of the region
    inputs:
        x:(B, C, H, W)
    """

    def __init__(self, kernel, stride, padding):
        super(sum_Conv, self).__init__()
        self.sumer = nn.Conv2d(1, 1, kernel, stride, padding, bias=False)
        self.sumer.weight.data = torch.ones((1, 1, kernel, kernel))
        self.sumer.weight.requires_grad = False  # 不需要梯度更新

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B * C, 1, H, W)
        res = self.sumer(x)
        res = res.view(B, C, res.shape[-2], res.shape[-1])
        return res


class ConvT2d_Bn_Relu(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int = 3):
        super(ConvT2d_Bn_Relu, self).__init__()
        self.convT = nn.ConvTranspose2d(in_channel, out_channel, kernel_size, 2, 1, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.convT(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Learning_Block1(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Learning_Block1, self).__init__()
        self.conv1 = Conv2d_Bn_Relu(in_channel, out_channel, kernel_size, 2, 1)  # (B,C,16,16)
        self.conv2 = Conv2d_Bn_Relu(out_channel, out_channel * 2, kernel_size, 2, 1)  # (B,C,8,8)
        self.conv3 = Conv2d_Bn_Relu(out_channel * 2, out_channel * 4, kernel_size, 2, 1)  # (B,C,4,4)
        self.conv4 = Conv2d_Bn_Relu(out_channel * 4, out_channel * 8, kernel_size, 2, 1)  # (B,C,2,2)

        self.convT4 = ConvT2d_Bn_Relu(out_channel * 8, out_channel * 4, kernel_size)  # (B,C,8,8)
        self.convT3 = ConvT2d_Bn_Relu(out_channel * 4, out_channel * 2, kernel_size)  # (B,C,8,8)
        self.convT2 = ConvT2d_Bn_Relu(out_channel * 2, out_channel, kernel_size)  # (B,C,16,16)
        self.convT1 = ConvT2d_Bn_Relu(out_channel, in_channel, kernel_size)  # (B,C,32,32)

    def forward(self, img):
        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        xT3 = self.convT4(x4)
        x3 = torch.min(xT3, x3)
        xT2 = self.convT3(x3)
        x2 = torch.min(xT2, x2)
        xT1 = self.convT2(x2)
        x1 = torch.min(xT1, x1)
        xT0 = self.convT1(x1)
        res = torch.min(xT0, img)
        return res


class Learning_Block2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(Learning_Block2, self).__init__()
        self.conv1 = Conv2d_Bn_Relu(in_channel, out_channel, kernel_size, 2, 1)  # (B,C,16,16)
        self.conv2 = Conv2d_Bn_Relu(out_channel, out_channel * 2, kernel_size, 2, 1)  # (B,C,8,8)
        self.conv3 = Conv2d_Bn_Relu(out_channel * 2, out_channel * 4, kernel_size, 2, 1)  # (B,C,4,4)

    def forward(self, img):
        x1 = self.conv1(img)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x2 = torch.min(x3.repeat(1, 1, 2, 2), x2)
        x1 = torch.min(x2.repeat(1, 1, 2, 2), x1)
        res = torch.min(x1.repeat(1, 1, 2, 2), img)
        return res


class LearningConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(LearningConv, self).__init__()
        self.linear1 = Conv2d_Bn_Relu(in_channel, 8, kernel_size, 1, padding=1)
        self.block2 = Learning_Block1(8, 16, kernel_size)
        self.block3 = Learning_Block1(8, 16, kernel_size)
        self.block4 = Learning_Block1(8, 16, kernel_size)
        # self.block5 = Learning_Block1(8, 16, kernel_size)
        self.linear6 = Conv2d_Bn_Relu(8, out_channel, kernel_size, 1, 1)

    def forward(self, img):
        x = self.linear1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)
        x = self.linear6(x)
        return x


class headLarge(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3):
        super(LearningConv, self).__init__()
        self.linear1 = Conv2d_Bn_Relu(in_channel, 8, kernel_size, 1, padding=1)
        self.block2 = Learning_Block1(8, 16, kernel_size)
        self.block3 = Learning_Block1(8, 16, kernel_size)
        self.block4 = Learning_Block1(8, 16, kernel_size)
        # self.block5 = Learning_Block1(8, 16, kernel_size)
        self.linear6 = Conv2d_Bn_Relu(8, out_channel, kernel_size, 1, 1)

    def forward(self, img):
        x = self.linear1(img)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        # x = self.block5(x)
        x = self.linear6(x)
        # x = self.block5(x)
        # x = self.block6(x)
        # x = self.block7(x)
        # x = self.block8(x)
        return x


class LocalSegment(nn.Module):
    """
    args:
        cfg: dict of configuration information
    inputs:
        targets:(bool)
            (B, 1, 32, 32)
            (B, 1, 16, 16)
            (B, 1, 8, 8)
    output:
        (B, 1, 32, 32)
        (B, 1, 16, 16)
        (B, 1, 8, 8)
    """

    def __init__(self, cfg=None):
        super(LocalSegment, self).__init__()
        if cfg is None:
            raise ValueError("parameter 'cfg' is not given")

        self.resconv = LearningConv(1, cfg["learning_conv_outchannel"])

        self.linear = DetectNet1(cfg["learning_conv_outchannel"], 1)

    def forward(self, img):
        results = [[[] for j in range(len(img[i]))] for i in range(len(img))]

        img, shape_idx = UnifyShape(img, 32)

        x = self.resconv(img)
        res = self.linear(x)
        # reconstruct the list strcuture
        cnt = 0
        for i in range(len(shape_idx)):
            pred_size = 2 ** (5 - i)
            for j in range(len(shape_idx[i])):
                cnt_new = shape_idx[i][j]["num"] + cnt
                results[i][j] = res[cnt:cnt_new]
                results[i][j] = results[i][j][
                    :,
                    0:,
                    shape_idx[i][j]["row_idx"] : pred_size + shape_idx[i][j]["row_idx"],
                    shape_idx[i][j]["col_idx"] : pred_size + shape_idx[i][j]["col_idx"],
                ]
                cnt = cnt_new
        return results
