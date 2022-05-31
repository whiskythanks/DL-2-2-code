import os
from PIL import Image
import cv2
import numpy as np
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


class Train_Dataset(Dataset):
    def __init__(self, image_dir='Heart Data'):
        self.images, self.labels = Prepare(image_dir, False)
        # image：单通道黑白图像
        self.image_transform = transforms.Compose([transforms.CenterCrop(image_size),
                                                   transforms.ToTensor(), ])
        # label：单通道黑白图像，0、85、170、255
        self.label_transform = transforms.Compose([transforms.CenterCrop(image_size),
                                                   # transforms.PILToTensor(),
                                                   transforms.ToTensor(),
                                                   transforms.Lambda(lambda y: y.to(dtype=torch.int64).squeeze())])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 打开相应image、label
        # image = Image.open(self.images[idx])
        # label = Image.open(self.labels[idx])
        image = Image.open(self.images[idx]).convert('RGB')
        label = Image.open(self.labels[idx])
        # image = Image.open(self.images[idx])
        # label = Image.open(self.labels[idx])
        # 图像变化
        image = self.image_transform(image)
        label = self.label_transform(label)
        # 将颜色0、85、170、255分别转化为类0、1、2、3
        label[label == 0], label[label == 85], label[label == 170], label[label == 255] = 0, 1, 2, 3
        return image, label
