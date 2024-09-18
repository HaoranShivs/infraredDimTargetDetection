import os
import shutil

from PIL import Image
import numpy as np
import itertools
import scipy.ndimage

def move_files(src_dir, file_names, dest_dir):
    # 确保目标目录存在
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 遍历源目录中的所有文件
    for filename in file_names:
        filename = filename + ".png"
        src_file = os.path.join(src_dir, filename)
        # 构建新的文件名
        dest_file = os.path.join(dest_dir, filename)
        
        # 移动并重命名文件
        shutil.move(src_file, dest_file)


def assign_side_length(a, targets):
    # 遍历目标数值列表
    for target in targets:
        if a < target:
            return target
    return targets[-1]


def crop_objects(root_dir, crop_size=32):
    # set path
    images_path = root_dir + "/images"
    masks_path = root_dir + "/masks"
    output_images_dir = root_dir + "/crop_images"
    output_masks_dir = root_dir + "/crop_masks"

    # 确保输出目录存在
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    if not os.path.exists(output_images_dir + f'/{crop_size}'):
        os.makedirs(output_images_dir + f'/{crop_size}')
    if not os.path.exists(output_masks_dir):
        os.makedirs(output_masks_dir)
    if not os.path.exists(output_masks_dir + f'/{crop_size}'):
        os.makedirs(output_masks_dir + f'/{crop_size}')

    for i in os.listdir(images_path):
        # 加载图像和mask
        image = Image.open(images_path + '/' + i).convert('L')
        mask = Image.open(masks_path + '/' + i).convert('L')  # 转换为灰度图

        # 将mask转换为numpy数组以便处理
        mask_array = np.array(mask)
        
        # 使用连通组件分析找到所有独立的目标区域
        labels, num_features = scipy.ndimage.label(mask_array > 0)

        for label_id in range(1, num_features + 1):
            # 获取当前连通组件的位置
            pos = np.where(labels == label_id)
            
            if len(pos[0]) == 0:
                continue 

            # 计算目标区域的边界框
            top_left_x = min(pos[1])
            top_left_y = min(pos[0])
            bottom_right_x = max(pos[1])
            bottom_right_y = max(pos[0])

             # 计算正方形的中心点
            center_x = (top_left_x + bottom_right_x) // 2
            center_y = (top_left_y + bottom_right_y) // 2

            # 计算正方形的边长
            # square_side = max(bottom_right_x - top_left_x, bottom_right_y - top_left_y)
            # if square_side < 3:
            #     continue
            # square_side = assign_side_length(square_side, [8,16,32,64])
            square_side = crop_size

            # 调整裁剪区域中心点坐标以适应图像边界
            if center_x - square_side // 2 < 0:
                center_x = square_side // 2
            if center_y - square_side // 2 < 0:
                center_y = square_side // 2
            if center_x + square_side // 2 >= image.width:
                center_x = image.width - square_side // 2 - 1
                if center_x - square_side // 2 < 0:
                    continue
            if center_y + square_side // 2 >= image.height:
                center_y = image.height - square_side // 2 - 1
                if center_y - square_side // 2 < 0:
                    continue
            
            # 定义裁剪的范围
            left = center_x - square_side // 2
            upper = center_y - square_side // 2
            right = center_x + square_side // 2
            lower = center_y + square_side // 2

            # 裁剪图像和mask
            cropped_image = image.crop((left, upper, right, lower))
            cropped_mask = mask.crop((left, upper, right, lower))

            # 获取文件名
            base_name, ext = os.path.splitext(i)
            
            # 生成带有编号的新文件名
            new_filename = f"{base_name}_{str(label_id).zfill(2)}{ext}"

            # print(output_images_dir + '/' + new_filename)
            # a = input()
            
            # 保存裁剪后的图像和mask
            cropped_image.save(output_images_dir + f'/{square_side}/' + new_filename)
            cropped_mask.save(output_masks_dir + f'/{square_side}/' + new_filename)


if __name__ == '__main__':
    # source_directory = "W:/DataSets/IRSTD-1k"
    # trainval_directory = "W:/DataSets/IRSTD-1k/trainval"
    # test_directory = "W:/DataSets/IRSTD-1k/test"

    # file_names = []
    # with open("W:/DataSets/IRSTD-1k/test.txt", "r") as file:
    #     for line in file:
    #         file_names.append(line.strip())

    # move_files(os.path.join(source_directory, 'IRSTD1k_Img'), file_names, os.path.join(test_directory, 'images'))
    # move_files(os.path.join(source_directory, 'IRSTD1k_Label'), file_names, os.path.join(test_directory, 'masks'))
    root_dir = "W:/DataSets/ISTD/NUDT-SIRST/trainval"
    crop_objects(root_dir, 16)
