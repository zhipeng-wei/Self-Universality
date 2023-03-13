import torch
import torch.nn as nn
import csv
import os

OPT_PATH = ''
NIPS_DATA = ''
IMAGENET_VAL_DATA = ''
# ADV_IMAGE_MODEL_CKPT = '/share_io02_ssd/zhipeng/adv_image_models_ckpt/{}.ckpt'
# ADV_MODELS = ['adv_inception_resnet_v2', 'adv_inception_v3', 'ens3_adv_inception_v3', 'ens4_adv_inception_v3', 'ens_adv_inception_resnet_v2']

class Normalize(nn.Module):
    def __init__(self, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()

    def forward(self, x):
        return (x - self.mean.to(x.device)[None, :, None, None]
                ) / self.std.to(x.device)[None, :, None, None]

def load_ground_truth(csv_filename=os.path.join(NIPS_DATA, 'images.csv')):
    image_id_list = []
    label_ori_list = []
    label_tar_list = []

    with open(csv_filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            image_id_list.append( row['ImageId'] )
            label_ori_list.append( int(row['TrueLabel']) - 1 )
            label_tar_list.append( int(row['TargetClass']) - 1 )

    return image_id_list, label_ori_list, label_tar_list