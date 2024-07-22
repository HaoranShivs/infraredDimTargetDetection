import os
import shutil

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


if __name__ == '__main__':
    source_directory = "W:/DataSets/IRSTD-1k"
    trainval_directory = "W:/DataSets/IRSTD-1k/trainval"
    test_directory = "W:/DataSets/IRSTD-1k/test"

    file_names = []
    with open("W:/DataSets/IRSTD-1k/test.txt", "r") as file:
        for line in file:
            file_names.append(line.strip())

    move_files(os.path.join(source_directory, 'IRSTD1k_Img'), file_names, os.path.join(test_directory, 'images'))
    move_files(os.path.join(source_directory, 'IRSTD1k_Label'), file_names, os.path.join(test_directory, 'masks'))