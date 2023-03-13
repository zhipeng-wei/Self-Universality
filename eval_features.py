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
import pickle as pkl

activations = []
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

def split_batch(used_list, batch_size):
    groups = zip(*(iter(used_list), ) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(used_list) % batch_size
    if count != 0:
        end_list.append(used_list[-count:])
    else:
        pass
    return end_list

def forward_hook(module, input, output):
    global activations
    activations += [output]
    return None

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
    # register
    eval_models[0][1].Mixed_7c.register_forward_hook(forward_hook)
    eval_models[1][1].layer3[-1].register_forward_hook(forward_hook)
    eval_models[2][1].features.denseblock3.register_forward_hook(forward_hook)
    eval_models[3][1].features[20].register_forward_hook(forward_hook)

    # loading attacked images
    attack_dataset = getattr(dataset, args.dataset)()
    print ('length of attack dataset', len(attack_dataset))
    

    # loading paths of generated perturbations
    perts = glob.glob(os.path.join(args.adv_path, 'pert-from_*_to_*.npy'))
    # ids are randomly generated in local.ipynb
    used_perts_ids = [770, 789, 863, 961, 818, 861, 785, 778, 797, 824]
    used_perts = []
    for pert_id in used_perts_ids:
        used_perts.append(perts[pert_id])
    # obtain their correct predicted samples
    path = glob.glob(os.path.join('/share_io02_ssd/zhipeng/iterative_attack', '*BaseCE*True*/eval_universal.npy'))
    eval_universal = np.load(path[0])

    # calculate the feature similarity between images before and after adding perturbations
    similarity_matrixs = {}
    pert_flag = 0
    for pert_path in tqdm(used_perts):
        pert = torch.from_numpy(np.load(pert_path))
        pert = torch.unsqueeze(pert, dim=0).cuda()
        ori_label = int(pert_path.split('/')[-1].split('-')[-1].split('_')[1])
        tar_label = int(pert_path.split('/')[-1].split('-')[-1].split('_')[3].strip('.npy'))
        image_ind = int(pert_path.split('/')[-1].split('-')[-1].split('_')[4].strip('.npy'))
        pert_id = used_perts_ids[pert_flag]
        pert_flag += 1
        for model_id, model in enumerate(eval_models):
            dataset_ids = np.array([i for i in range(1000)])
            used_dataset_ids = dataset_ids[eval_universal[model_id, pert_id] == 1]
            batch_ids = split_batch(used_dataset_ids, args.batch_size)
            label_to_benign_features = {}
            label_to_adv_features = {}
            for batch_id in batch_ids:
                images = []
                ori_labels = []
                for dataset_id in batch_id:
                    image, ori_label, target_label = attack_dataset[dataset_id]
                    images.append(image)
                    ori_labels.append(ori_label)
                images = torch.stack(images, dim=0).cuda()
                activations = []
                with torch.no_grad():
                    this_logit = model(images)
                for label_id, ori_label in enumerate(ori_labels):
                    if ori_label not in label_to_benign_features.keys():
                        label_to_benign_features[ori_label] = activations[0][label_id]
                
                activations = []
                with torch.no_grad():
                    adv = (images + pert)
                    this_logit = model(adv)
                for label_id, ori_label in enumerate(ori_labels):
                    if ori_label not in label_to_adv_features.keys():
                        label_to_adv_features[ori_label] = activations[0][label_id]
            length = len(label_to_benign_features.values())
            benign_results = np.zeros((length, length))
            adv_results = np.zeros((length, length))
            used_ori_labels = list(label_to_benign_features.keys())
            ori_css = []
            adv_css = []
            for one_label in range(length):
                for ano_label in range(length):
                    if one_label > ano_label:
                        one_fea = label_to_benign_features[used_ori_labels[one_label]].view(1,-1)
                        ano_fea = label_to_benign_features[used_ori_labels[ano_label]].view(1,-1)
                        ori_cs = torch.nn.functional.cosine_similarity(one_fea, ano_fea).cpu().item()
                        one_fea = label_to_adv_features[used_ori_labels[one_label]].view(1,-1)
                        ano_fea = label_to_adv_features[used_ori_labels[ano_label]].view(1,-1)
                        adv_cs = torch.nn.functional.cosine_similarity(one_fea, ano_fea).cpu().item()
                        ori_css.append(ori_cs)
                        adv_css.append(adv_cs)
            # print ('used length.', len(used_ori_labels))
            if pert_id not in similarity_matrixs.keys():
                similarity_matrixs[pert_id] = {}
            similarity_matrixs[pert_id][model_id] = [ori_css, adv_css]
            
    # save
    with open(os.path.join(args.adv_path, 'cosine_similarity.json'), 'wb') as opt:
        pkl.dump(similarity_matrixs, opt)