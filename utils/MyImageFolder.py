# -*- coding: utf-8 -*-
"""
@Time ： 2022/8/14 10:09
@Auth ： Fanteng Meng
@File ：imgfolder.py
@IDE ：PyCharm
@Github : https://github.com/FT115
"""
import torch
import random
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
normalize = transforms.Normalize(mean=[.5, .5, .5],
                                 std=[.5, .5, .5])

train_transform = transforms.Compose([])
train_transform.transforms.append(transforms.Resize((224, 224)))
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(transforms.RandomHorizontalFlip(p=0.8))
train_transform.transforms.append(normalize)

val_transform = transforms.Compose([])
val_transform.transforms.append(transforms.Resize((224, 224)))
val_transform.transforms.append(transforms.ToTensor())
val_transform.transforms.append(normalize)


class MyImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform):
        super(MyImageFolder, self).__init__(root, transform)


    def __getitem__(self, index):
        path, target = self.samples[index]
        img_name=os.path.splitext(os.path.basename(path))[0]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, img_name

    def __len__(self) -> int:
        return len(self.samples)

