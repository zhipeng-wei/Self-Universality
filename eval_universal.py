import argparse
import os
from tqdm import tqdm
import torch.nn as nn
import torch
import torchvision
from torchvision import models
import pandas as pd
import numpy as np
import json
import glob
from utils import Normalize, load_ground_truth, OPT_PATH, NIPS_DATA
import dataset

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='input batch size for reference (default: 10)')
    parser.add_argument('--dataset', type=str, default='NIPSDataset', help='class in dataset.py')
    parser.add_argument('--file_tailor', type=str, default='zp', help='')
    args = parser.parse_args()
    args.adv_path = os.path.join(OPT_PATH, args.file_tailor)
    return args

if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.manual_seed(1024)
    torch.backends.cudnn.deterministic = True

    # loading models
    all_models = ['inception_v3', 'resnet50', 'densenet121', 'vgg16_bn']
    eval_models = []
    for model_name in all_models:
        if model_name == 'inception_v3':
            this_model = nn.Sequential(Normalize(), getattr(models, model_name)(pretrained=True, transform_input=True)).eval()
        else:
            this_model =  nn.Sequential(Normalize(), getattr(models, model_name)(pretrained=True)).eval()
        for param in this_model[1].parameters():
            param.requires_grad = False  
        this_model.cuda()
        eval_models.append(this_model)

    # loading attacked images
    attack_dataset = getattr(dataset, args.dataset)()
    print ('length of attack dataset', len(attack_dataset))
    dataloader = torch.utils.data.DataLoader(
                    attack_dataset, batch_size=args.batch_size,
                    num_workers=4, shuffle=False, pin_memory=True)

    # loading paths of generated perturbations
    perts = glob.glob(os.path.join(args.adv_path, 'pert-from_*_to_*.npy'))

    # evaluation
    result_matrix = np.zeros((len(all_models), len(perts), len(attack_dataset)))
    flag = 0
    pert_to_OriImageInd = {}
    for pert_path in tqdm(perts):
        pert = torch.from_numpy(np.load(pert_path))
        pert = torch.unsqueeze(pert, dim=0)
        ori_label = int(pert_path.split('/')[-1].split('-')[-1].split('_')[1])
        tar_label = int(pert_path.split('/')[-1].split('-')[-1].split('_')[3].strip('.npy'))
        image_ind = int(pert_path.split('/')[-1].split('-')[-1].split('_')[4].strip('.npy'))
        res = []
        for inputs, ori_labels, target_labels in dataloader:
            adv = (inputs + pert).cuda()
            tar_labels = torch.ones_like(target_labels)*tar_label
            for model in eval_models:
                with torch.no_grad():
                    this_logit = model(adv)
                preds = (torch.argmax(this_logit, dim=1) == tar_label).type(torch.int).cpu().numpy()
                res.append(preds)
        for model_ind in range(len(all_models)):
            model_res = res[model_ind::len(all_models)]
            model_res = np.concatenate(model_res, axis=0)
            result_matrix[model_ind, flag] = model_res
        pert_to_OriImageInd[flag] = image_ind
        flag += 1

    # save
    np.save(os.path.join(args.adv_path, 'eval_universal'), result_matrix)
    with open(os.path.join(args.adv_path, 'pert_to_OriImageInd.json'), 'w') as opt:
        json.dump(pert_to_OriImageInd, opt)