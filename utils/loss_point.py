import torch
import torch.nn as nn
import torch.nn.functional as F


def find_nearest_distance_to_highlights(image_batch):
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
            distance_map[b,0] = torch.ones([S, S], dtype=torch.float32) * 362.1
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

def img_gradient(image_batch):
    # 定义 Sobel 算子
    sobel = torch.tensor([[[[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]]],
                          [[[ 1., 2., 1.],
                            [ 0., 0., 0.],
                            [-1.,-2.,-1.]]],
                          [[[ 2., 1., 0.],
                            [ 1., 0.,-1.],
                            [ 0.,-1.,-2.]]],
                          [[[ 0., 1., 2.],
                            [-1., 0., 1.],
                            [-2.,-1., 0.]]]], dtype=image_batch.dtype, device=image_batch.device, requires_grad=False)
    # 找到四个方向上最大的梯度
    _gradient = F.conv2d(image_batch, sobel, padding=1)  # (B, 4, S, S)
    _gradient = torch.max(torch.abs(_gradient), dim=1, keepdim=True).values
    _gradient[:, :, 0, :] = 0
    _gradient[:, :, -1,:] = 0
    _gradient[:, :, :, 0] = 0
    _gradient[:, :, :,-1] = 0
    return _gradient


class ptlabel_loss(nn.Module):
    """
    for point label training. the loss consists of distances of preding target from the point, the gradients of corresponding original picture area.
    """
    def __init__(self):
        super(ptlabel_loss, self).__init__()

    def forward(self, pred, pt_label, ori_img, loss=1.0):
        B, _, S, _ = pred.shape
        distance = find_nearest_distance_to_highlights(pt_label)
        # loss:(0.3,1.0)
        distance_thre = int((1.046-loss)/0.046)
        _distance = torch.where(distance>distance_thre, distance-distance_thre, torch.zeros_like(distance))  #(B, 1, S, S)
        _sum = torch.max(_distance.view(_distance.shape[0], _distance.shape[1], -1), dim=2).values
        _distance = _distance / _sum.unsqueeze(-1).unsqueeze(-1)
        distance_loss = torch.sum(pred * _distance) / (S * S * B)
        # distance_mask = torch.where(distance>distance_thre, torch.zeros_like(distance), torch.ones_like(distance))
        img = ori_img * pred
        gradient = img_gradient(img)
        gradient_loss = torch.sum(gradient) / (S * S * B)
        return distance_loss, gradient_loss