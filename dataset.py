import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import random
import glob
import os
from PIL import Image

from utils import IMAGENET_VAL_DATA, NIPS_DATA, load_ground_truth

class ClassSamples15000(torch.utils.data.Dataset):
    '''
    Randomly sample 15000 examples from ImageNet Validation Dataset.
    They are used as attacked examples.
    '''
    def __init__(self, auto_assign_target=True, valdir=IMAGENET_VAL_DATA):
        normalize = transforms.Normalize(mean=[0., 0., 0.],
                                     std=[1., 1., 1.])
        self.auto_assign_target = auto_assign_target
        self.val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,]))
        predict_df = pd.read_csv('./val_predict_label.csv')
        valid_inds = predict_df[(predict_df['densenet121'] == predict_df['gt']) & \
                   (predict_df['vgg19_bn'] == predict_df['gt']) & \
                   (predict_df['resnet50'] == predict_df['gt'])].index.tolist()
        random.seed(1024)
        random.shuffle(valid_inds)
        self.inds = valid_inds[:15000]
        

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, idx):
        image, label = self.val_dataset[self.inds[idx]]
        if self.auto_assign_target:
            target_label = [i for i in range(1000) if i!=label]
            random.seed(idx)
            random.shuffle(target_label)
            target_label = target_label[0]
            return image, label, target_label
        else:
            return image, label, None

class NIPSDataset(torch.utils.data.Dataset):
    '''
    Randomly sample 15000 examples from ImageNet Validation Dataset.
    They are used as attacked examples.
    '''
    def __init__(self, data_path=NIPS_DATA, part=1, part_index=1):
        image_id_list, self.label_ori_list, self.label_tar_list = load_ground_truth()
        self.image_paths = []
        for image_id in image_id_list:
            path = os.path.join(data_path, 'images', '{}.png'.format(image_id))
            self.image_paths.append(path)
        if part == 1:
            pass
        else:
            length = len(self.image_paths)
            part_len = int(length / part)
            self.image_paths = self.image_paths[(part_index-1)*part_len:part_index*part_len]

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            ])        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path)
        image = self.transforms(image)
        return image, self.label_ori_list[idx], self.label_tar_list[idx]


class NIPSDataset_TwoImages(torch.utils.data.Dataset):
    '''
    Randomly sample 15000 examples from ImageNet Validation Dataset.
    They are used as attacked examples.
    '''
    def __init__(self, data_path=NIPS_DATA):
        image_id_list, self.label_ori_list, self.label_tar_list = load_ground_truth()
        self.image_paths = []
        for image_id in image_id_list:
            path = os.path.join(data_path, 'images', '{}.png'.format(image_id))
            self.image_paths.append(path)
        
        self.ori_transforms = transforms.Compose([
            transforms.ToTensor(),
            ])        
        self.local_transforms = transforms.Compose([
            transforms.RandomResizedCrop(299, scale=(0.2,0.5)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        ori_image = Image.open(path)
        image = self.ori_transforms(ori_image)
        local_image = self.local_transforms(ori_image)
        return [image, local_image], self.label_ori_list[idx], self.label_tar_list[idx]