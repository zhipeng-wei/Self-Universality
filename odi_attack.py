import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.stats as st
import torchvision
from torch.distributions.multivariate_normal import MultivariateNormal

import time
from losses import LogitLoss
from odi import render_3d_aug_input, Render3D

class Base(object):
    def __init__(self, loss_fn, eval_steps, steps, target=False, random_start=False, epsilon=16./255., alpha=2./255.):
        self.eval_steps = eval_steps   
        self.steps = steps
        self.target = target
        self.random_start = random_start
        self.epsilon = epsilon
        self.alpha = alpha
        if loss_fn == 'CE':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_fn == 'Logit':
            self.loss_fn = LogitLoss()

        self.begin_time = time.time()

    def _DI(self, X_in):
        rnd = np.random.randint(299, 330,size=1)[0]
        h_rem = 330 - rnd
        w_rem = 330 - rnd
        pad_top = np.random.randint(0, h_rem,size=1)[0]
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem,size=1)[0]
        pad_right = w_rem - pad_left

        c = np.random.rand(1)
        if c <= 0.7:
            X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
            return  X_out 
        else:
            return  X_in

    def _DI_WO_Prob(self, X_in):
        rnd = np.random.randint(299, 330,size=1)[0]
        h_rem = 330 - rnd
        w_rem = 330 - rnd
        pad_top = np.random.randint(0, h_rem,size=1)[0]
        pad_bottom = h_rem - pad_top
        pad_left = np.random.randint(0, w_rem,size=1)[0]
        pad_right = w_rem - pad_left

        X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_top,pad_right,pad_bottom),mode='constant', value=0)
        X_out = F.interpolate(X_out, size=(X_in.shape[-2], X_in.shape[-1]))
        return  X_out 

    def _TI_kernel(self):
        def gkern(kernlen=15, nsig=3):
            x = np.linspace(-nsig, nsig, kernlen)
            kern1d = st.norm.pdf(x)
            kernel_raw = np.outer(kern1d, kern1d)
            kernel = kernel_raw / kernel_raw.sum()
            return kernel
        channels=3
        kernel_size=5
        kernel = gkern(kernel_size, 3).astype(np.float32)
        gaussian_kernel = np.stack([kernel, kernel, kernel])
        gaussian_kernel = np.expand_dims(gaussian_kernel, 1)
        gaussian_kernel = torch.from_numpy(gaussian_kernel).cuda()
        return gaussian_kernel
    
    def _target_layer(self, model_name, depth):
        '''
        'inception_v3', 'resnet50', 'densenet121', 'vgg16_bn'
        depth: [1, 2, 3, 4]
        '''
        if model_name == 'resnet50':
            return getattr(self.model[1], 'layer{}'.format(depth))[-1]
        elif model_name == 'vgg16_bn':
            depth_to_layer = {1:12,2:22,3:32,4:42}
            return getattr(self.model[1], 'features')[depth_to_layer[depth]]
        elif model_name == 'densenet121':
            return getattr(getattr(self.model[1], 'features'), 'denseblock{}'.format(depth))
        elif model_name == 'inception_v3':
            depth_to_layer = {1:'Conv2d_4a_3x3', 2:'Mixed_5d', 3:'Mixed_6e', 4:'Mixed_7c'}
            return getattr(self.model[1], '{}'.format(depth_to_layer[depth]))

    def perturb(self, images, ori_labels, target_labels):
        raise NotImplementedError

class TMI_ODI(Base):
    '''
    Use TI, MI and DI as the baseline attack.
    '''
    def __init__(self, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(TMI_ODI, self).__init__(loss_fn, eval_steps, steps, target)
        self.model = model
        self.eval_models = eval_models

        self.DI, self.TI, self.MI = DI, TI, MI
        if self.TI:
            self.gaussian_kernel = self._TI_kernel()
        # parameters from https://github.com/dreamflake/ODI python eval_attacks.py --config_idx=101
        self.render = Render3D(101, 0)
        self.prob = 1.0

    def perturb(self, images, ori_labels, target_labels):
        adv_images = images.clone()
        batch_size = adv_images.shape[0]
        if self.target:
            used_label = target_labels
            used_coef = -1
        else:
            used_label = ori_labels
            used_coef = 1
        
        # if self.random_start:
        #     adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
        #     adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        #     delta = torch.tensor((adv_images - images).clone().detach(), requires_grad=True).cuda()
        # else:
        #     delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        norm_matrix = np.zeros((self.steps, batch_size))

        x_min = torch.clamp(adv_images - self.epsilon, 0.0, 1.0)
        x_max = torch.clamp(adv_images + self.epsilon, 0.0, 1.0)

        iter_flag = 0
        if self.MI:
            grad_pre = 0
        # begin_time = time.time()
        for itr in range(self.steps):
            adv_images.requires_grad_()
            with torch.enable_grad():
                if self.DI:
                    odi_input = render_3d_aug_input(adv_images, renderer=self.render, prob=self.prob)
                    logits = self.model(odi_input)
                    loss_label = used_label
                else:
                    logits = self.model(adv_images)
                    loss_label = used_label
                loss = self.loss_fn(logits, loss_label)
            grad_c = torch.autograd.grad(loss, adv_images,
                                        retain_graph=False, create_graph=False)[0]
            # loss.backward(retain_graph=True)
            # grad_c = delta.grad.clone()
            
            # grad_norm = torch.norm(grad_c.view(batch_size, -1), dim=1).cpu().numpy()
            # norm_matrix[itr] = grad_norm

            if self.TI:
                grad_c = F.conv2d(grad_c, self.gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
            else:
                pass
            if self.MI:
                grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre
                grad_pre = grad_a
            else:
                grad_a = grad_c.clone()

            # delta.grad.zero_()
            # delta.data = delta.data + used_coef * self.alpha * torch.sign(grad_a)
            # delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            # delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
            pert = used_coef * self.alpha * grad_a.sign()
            adv_images = adv_images.detach() + pert
            adv_images = torch.clamp(adv_images, x_min, x_max)


            if itr+1 in self.eval_steps:
                for m_id, model in enumerate(self.eval_models):
                    with torch.no_grad():
                        this_logit = model(adv_images)
                    if self.target:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) == used_label).cpu().item()
                    else:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) != used_label).cpu().item()
                    result_matrix[m_id, iter_flag] = success_nums
                iter_flag += 1
            torch.cuda.empty_cache() 
        # end_time = time.time()
        # print (end_time-begin_time)
        return result_matrix, adv_images-images, norm_matrix

class DTMI_Local_FeatureSimilarityLoss_ODI(Base):
    '''
    Incorporate locality of images and the feature similarity loss between global features and local features.
    '''
    def __init__(self, model_name, depth, coef, scale, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI_Local_FeatureSimilarityLoss_ODI, self).__init__(loss_fn, eval_steps, steps, target)
        self.start, self.interval = scale
        self.coef = coef
        self.model_name = model_name
        self.depth = depth
        self.model = model
        self.eval_models = eval_models

        self.DI, self.TI, self.MI = DI, TI, MI
        if self.TI:
            self.gaussian_kernel = self._TI_kernel()

        self.local_transform = torchvision.transforms.RandomResizedCrop(299, scale=(self.start, self.start+self.interval))
        self._register_forward()

        # parameters from https://github.com/dreamflake/ODI python eval_attacks.py --config_idx=101
        self.render = Render3D(101, 0)
        self.prob = 1.0

    def _register_forward(self):
        '''
        'inception_v3', 'resnet50', 'densenet121', 'vgg16_bn'
        '''
        self.activations = []
        def forward_hook(module, input, output):
            self.activations += [output]
            return None
        target_layer = self._target_layer(self.model_name, self.depth)    
        target_layer.register_forward_hook(forward_hook)

    def _LI_WO_prob(self, X_in):
        return self.local_transform(X_in)

    def perturb(self, images, ori_labels, target_labels):
        adv_images = images.clone()
        batch_size = adv_images.shape[0]
        if self.target:
            used_label = target_labels
            used_coef = -1
        else:
            used_label = ori_labels
            used_coef = 1
        
        if self.random_start:
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
            delta = torch.Tensor(adv_images - images, requires_grad=True).cuda()
        else:
            delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        norm_matrix = np.zeros((self.steps, batch_size))

        x_min = torch.clamp(adv_images - self.epsilon, 0.0, 1.0)
        x_max = torch.clamp(adv_images + self.epsilon, 0.0, 1.0)

        iter_flag = 0
        if self.MI:
            grad_pre = 0
        # begin_time = time.time()
        for itr in range(self.steps):
            adv_images.requires_grad_()
            with torch.enable_grad():
                if self.DI:
                    odi_input = render_3d_aug_input(adv_images, renderer=self.render, prob=self.prob)
                    li_inputs = self._LI_WO_prob(odi_input)
                    accom_inputs = torch.cat([odi_input, li_inputs], dim=0)
                    logits = self.model(accom_inputs)
                    loss_label = torch.cat([used_label, used_label], dim=0)
                classifier_loss = self.loss_fn(logits, loss_label)
                fs_loss = torch.nn.functional.cosine_similarity(self.activations[0][:batch_size].view(batch_size, -1), self.activations[0][-batch_size:].view(batch_size, -1))
                fs_loss = torch.mean(fs_loss)
                loss = classifier_loss + self.coef * used_coef * fs_loss
            grad_c = torch.autograd.grad(loss, adv_images,
                                        retain_graph=False, create_graph=False)[0]

            if self.TI:
                grad_c = F.conv2d(grad_c, self.gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
            else:
                pass
            if self.MI:
                grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre
                grad_pre = grad_a
            else:
                grad_a = grad_c.clone()

            pert = used_coef * self.alpha * grad_a.sign()
            adv_images = adv_images.detach() + pert
            adv_images = torch.clamp(adv_images, x_min, x_max)

            if itr+1 in self.eval_steps:
                for m_id, model in enumerate(self.eval_models):
                    with torch.no_grad():
                        this_logit = model(adv_images+delta)
                    if self.target:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) == used_label).cpu().item()
                    else:
                        success_nums = torch.sum(torch.argmax(this_logit, dim=1) != used_label).cpu().item()
                    result_matrix[m_id, iter_flag] = success_nums
                iter_flag += 1
        return result_matrix, adv_images-images, norm_matrix