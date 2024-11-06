import os
import random
import numpy as np
from PIL import Image

def load_image_and_mask(image_path, mask_path):
    """
    加载图像和掩膜。
    
    参数:
        image_path (str): 图像文件路径。
        mask_path (str): 掩膜文件路径。
        
    返回:
        Tuple[np.ndarray, np.ndarray]: 图像数组和掩膜数组。
    """
    image = Image.open(image_path).convert('L')  # 转换为灰度图
    mask = Image.open(mask_path).convert('L')  # 转换为灰度图
    image_array = np.array(image)
    mask_array = np.array(mask)
    return image_array, mask_array

def sample_noise_patch(image_array, mask_array, size=32):
    """
    从图像中随机选取一个大小为 size 的子图像，
    该子图像不包含任何检测目标。
    
    参数:
        image_array (np.ndarray): 图像数组。
        mask_array (np.ndarray): 掩膜数组。
        size (int): 子图像的大小，默认为 32。
        
    返回:
        np.ndarray: 不包含检测目标的子图像。
    """
    height, width = image_array.shape[:2]
    
    while True:
        # 随机选取子图像的位置
        top = random.randint(0, height - size)
        left = random.randint(0, width - size)
        
        # 获取子图像和掩膜
        sub_image = image_array[top:top+size, left:left+size]
        sub_mask = mask_array[top:top+size, left:left+size]
        
        # 检查掩膜中是否有非零像素
        if not np.any(sub_mask):
            return sub_image

def main(root_dir, size=32):
    """
    主函数，用于从图像数据集中选取不包含目标的子图像。
    
    参数:
        image_path (str): 图像文件路径。
        mask_path (str): 掩膜文件路径。
        output_path (str): 输出子图像的路径。
        size (int): 子图像的大小，默认为 32。
    """
    # set path
    images_path = root_dir + "/images"
    masks_path = root_dir + "/masks"
    output_images_dir = root_dir + f"/noise{size}"
    
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)

    for i in os.listdir(images_path):
        # 加载图像和掩膜
        image_array, mask_array = load_image_and_mask(images_path+'/'+i, masks_path+'/'+i)

        # 选取不包含目标的子图像
        noise_patch = sample_noise_patch(image_array, mask_array, size=size)

        # 保存子图像
        Image.fromarray(noise_patch).save(output_images_dir+'/'+i)

# 示例使用
if __name__ == "__main__":
    root_dir = "W:/DataSets/ISTD/NUDT-SIRST/test"
    
    main(root_dir, 32)