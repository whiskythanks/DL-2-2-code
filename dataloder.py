import os
from PIL import Image
import cv2
import numpy as np
from sqlalchemy import true
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms

image_size = 128

def Prepare(image_dir, is_save: bool):
    save_image_dir = 'data/Image'
    save_label_dir = 'data/Label'
    if (not os.path.exists(save_image_dir)) and is_save: os.makedirs(save_image_dir)
    if (not os.path.exists(save_label_dir)) and is_save: os.makedirs(save_label_dir)
    images = []
    labels = []
    for classes in ['DCM', 'HCM', 'NOR']:
        for types in ['Image', 'Label']:
            for root, dirs, files in os.walk(os.path.join(image_dir, 'Image_' + classes, 'png', types)):
                for file in files:
                    old_name = os.path.join(root, file)
                    # 如果对图像进行保存
                    if is_save:
                        image = Image.open(old_name)
                        new_name = os.path.join(save_image_dir if types == 'Image' else save_label_dir,  # 路径
                                                classes + root[-2:] + file)  # 文件名
                        image.save(new_name)
                        # 返回新的路径
                        images.append(new_name) if types == 'Image' else labels.append(new_name)
                    # 不保存则返回原路径
                    else:
                        images.append(old_name) if types == 'Image' else labels.append(old_name)

    return images, labels

Prepare(image_dir=r"./Heart Data",is_save=true)