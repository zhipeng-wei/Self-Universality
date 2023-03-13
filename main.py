import argparse
import os
from tqdm import tqdm
import torch.nn as nn
import torch
import torchvision
from torchvision import models
import pandas as pd
import numpy as np
from utils import Normalize, load_ground_truth, OPT_PATH, NIPS_DATA
import dataset
import attacks
import time

def arg_parse():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device.')
    parser.add_argument('--white_box', type=str, default='resnet152', help='inception_v3, resnet50, densenet121, vgg16_bn')
    parser.add_argument('--batch_size', type=int, default=10, metavar='N',
                    help='input batch size for reference (default: 10)')
    parser.add_argument('--dataset', type=str, default='NIPSDataset', help='class in dataset.py')
    parser.add_argument('--attack', type=str, default='DTMI', help='class in attacks.py')
    parser.add_argument('--loss_fn', type=str, default='CE', help='CE or Logit')
    
    parser.add_argument('--steps', type=int, default=300, help='CE or Logit')
    
    parser.add_argument('--target', action='store_true')
    parser.add_argument('--no-target', dest='target', action='store_false')
    parser.set_defaults(target=True)
    parser.add_argument('--MI', action='store_true')
    parser.add_argument('--no-MI', dest='MI', action='store_false')
    parser.set_defaults(MI=True)
    parser.add_argument('--TI', action='store_true')
    parser.add_argument('--no-TI', dest='TI', action='store_false')
    parser.set_defaults(TI=True)
    parser.add_argument('--DI', action='store_true')
    parser.add_argument('--no-DI', dest='DI', action='store_false')
    parser.set_defaults(DI=True)

    # hyper-parameters used in this method
    parser.add_argument('--scale_start', type=float, default=0.1, help='the lower bound for the local image')
    parser.add_argument('--scale_interval', type=float, default=0.1, help='scale_start+scale_interval determines the upper bound')
    parser.add_argument('--fsl_coef', type=float, default=0.1, help='the coefficient of feature similarity loss')
    parser.add_argument('--depth', type=int, default=3, help='the layer used to extract features')
    
    parser.add_argument('--part', type=int, default=1, help='the layer used to extract features')
    parser.add_argument('--part_index', type=int, default=1, help='the layer used to extract features')

    parser.add_argument('--local_size', type=int, default=100, help='the coefficient of feature similarity loss')
    parser.add_argument('--local_shape', type=int, default=30, help='the layer used to extract features')

    parser.add_argument('--saveperts', action='store_true')
    parser.add_argument('--no-saveperts', dest='saveperts', action='store_false')
    parser.set_defaults(saveperts=False)

    parser.add_argument('--usecenter', action='store_true')
    parser.add_argument('--no-usecenter', dest='usecenter', action='store_false')
    parser.set_defaults(usecenter=False)

    parser.add_argument('--file_tailor', type=str, default='experiment', help='')
    args = parser.parse_args()
    args.adv_path = os.path.join(OPT_PATH, 'Paper-{}-{}-{}-Target_{}-{}'.format(args.white_box, args.dataset, args.attack, args.loss_fn, args.file_tailor))
    if not os.path.exists(args.adv_path):
        os.makedirs(args.adv_path)
    print (args.adv_path)
    return args


if __name__ == '__main__':
    args = arg_parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if os.path.exists(os.path.join(args.adv_path, 'result.csv')):
        print ('Exist', os.path.join(args.adv_path, 'result.csv'))
    else:
        torch.manual_seed(1024)
        torch.backends.cudnn.deterministic = True

        # loading models
        all_models = ['inception_v3', 'resnet50', 'densenet121', 'vgg16_bn']
        if args.white_box != 'inception_v3':
            white_model = nn.Sequential(Normalize(), getattr(models, args.white_box)(pretrained=True)).eval()
        else:
            white_model = nn.Sequential(Normalize(), getattr(models, args.white_box)(pretrained=True, transform_input=True)).eval()
        for param in white_model[1].parameters():
            param.requires_grad = False  
        white_model.cuda()

        black_models = [i for i in all_models if i != args.white_box]
        eval_models = []
        for model_name in black_models:
            if model_name == 'inception_v3':
                this_model = nn.Sequential(Normalize(), getattr(models, model_name)(pretrained=True, transform_input=True)).eval()
            else:
                this_model =  nn.Sequential(Normalize(), getattr(models, model_name)(pretrained=True)).eval()
            for param in this_model[1].parameters():
                param.requires_grad = False  
            this_model.cuda()
            eval_models.append(this_model)

        # loading dataset
        attack_dataset = getattr(dataset, args.dataset)(part=args.part, part_index=args.part_index)
        print ('length of attack dataset', len(attack_dataset))

        dataloader = torch.utils.data.DataLoader(
                        attack_dataset, batch_size=args.batch_size,
                        num_workers=4, shuffle=False, pin_memory=True)

        # attack
        if args.attack in ['DTMI', 'DTEMI', 'DTMI_SI',  'DTMI_Admix']:
            adversor = getattr(attacks, args.attack)(white_model, eval_models, loss_fn=args.loss_fn, eval_steps=[args.steps], steps=args.steps, \
                target=args.target, DI=args.DI, TI=args.TI, MI=args.MI)
        elif args.attack in ['DTMI_Local_FeatureSimilarityLoss', 'DTMI_Random_FeatureSimilarityLoss', 'DTMI_Local_FeatureSimilarityLoss_EMI', 'DTMI_Local_FeatureSimilarityLoss_Admix', 'DTMI_Local_FeatureSimilarityLoss_SI']:
            scale = (args.scale_start, args.scale_interval)
            adversor = getattr(attacks, args.attack)(args.white_box, args.depth, args.fsl_coef, scale, white_model, eval_models, loss_fn=args.loss_fn, eval_steps=[i+20 for i in range(0, args.steps, 20)], steps=args.steps, \
                target=args.target, DI=args.DI, TI=args.TI, MI=args.MI)
        elif args.attack in ['DTMI_Local_FeatureSimilarityLoss_Center']:
            scale = (args.scale_start, args.scale_interval)
            adversor = getattr(attacks, args.attack)(args.usecenter, args.white_box, args.depth, args.fsl_coef, scale, white_model, eval_models, loss_fn=args.loss_fn, eval_steps=[i+20 for i in range(0, args.steps, 20)], steps=args.steps, \
                target=args.target, DI=args.DI, TI=args.TI, MI=args.MI)
        elif args.attack in ['TMI_ODI', 'DTMI_Local_FeatureSimilarityLoss_ODI']:
            import odi_attack
            if args.attack == 'TMI_ODI':
                adversor = getattr(odi_attack, args.attack)(white_model, eval_models, loss_fn=args.loss_fn, eval_steps=[i+20 for i in range(0, args.steps, 20)], steps=args.steps, \
                    target=args.target, DI=args.DI, TI=args.TI, MI=args.MI)
            elif args.attack == 'DTMI_Local_FeatureSimilarityLoss_ODI':
                scale = (args.scale_start, args.scale_interval)
                adversor = getattr(odi_attack, args.attack)(args.white_box, args.depth, args.fsl_coef, scale, white_model, eval_models, loss_fn=args.loss_fn, eval_steps=[i+20 for i in range(0, args.steps, 20)], steps=args.steps, \
                    target=args.target, DI=args.DI, TI=args.TI, MI=args.MI)
        else:
            print (args.attack)
        # main loop
        result = None
        save_flag = 0
        gradient_norm_matrix = []
        for inputs, ori_labels, target_labels in tqdm(dataloader):
            if isinstance(inputs, list):
                for ind in range(len(inputs)):
                    inputs[ind] = inputs[ind].cuda()
            else:
                inputs = inputs.cuda()
            ori_labels = ori_labels.cuda()
            target_labels = target_labels.cuda()

            if 'Admix' in args.attack:
                admix_images = []
                for i in torch.randperm(len(attack_dataset))[:8]:
                    a_image,_,_ = attack_dataset[i]
                    admix_images.append(a_image)
                admix_images = torch.stack(admix_images, dim=0).cuda()
                re, perturbations, norm = adversor.perturb(inputs, ori_labels, target_labels, admix_images)
            else:
                re, perturbations, norm = adversor.perturb(inputs, ori_labels, target_labels)
            if result is None:
                result = re
            else:
                result += re
            
            gradient_norm_matrix.append(norm)

            if args.saveperts:
                for per_ind in range(args.batch_size):
                    ori_l, tar_l = ori_labels[per_ind].item(), target_labels[per_ind].item()
                    tmp_pert = perturbations[per_ind].detach().clone().cpu().numpy()
                    np.save(os.path.join(args.adv_path, 'pert-from_{}_to_{}_{}'.format(ori_l, tar_l, save_flag)), tmp_pert)
                    save_flag += 1
            torch.cuda.empty_cache() 

        attack_time = time.time()-adversor.begin_time

        # save result
        df = pd.DataFrame(columns = ['iter'] + black_models)
        for ind, itr in enumerate(adversor.eval_steps):
            df.loc[ind] = [itr] + list(result[:,ind])
        df.to_csv(os.path.join(args.adv_path, 'result_{}_{}.csv'.format(attack_time, args.part_index)), index=False)

        gradient_norm_matrix = np.concatenate(gradient_norm_matrix, axis=1)
        np.save(os.path.join(args.adv_path, 'gradient_magnitude'), gradient_norm_matrix)

