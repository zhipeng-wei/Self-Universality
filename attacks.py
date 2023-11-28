import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy.stats as st
import torchvision
from torch.distributions.multivariate_normal import MultivariateNormal
import math

import time
from losses import LogitLoss

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
            X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_right,pad_top,pad_bottom),mode='constant', value=0)
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

        X_out = F.pad(F.interpolate(X_in, size=(rnd,rnd)),(pad_left,pad_right,pad_top,pad_bottom),mode='constant', value=0)
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

class DTMI(Base):
    '''
    Use TI, MI and DI as the baseline attack.
    '''
    def __init__(self, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI, self).__init__(loss_fn, eval_steps, steps, target)
        self.model = model
        self.eval_models = eval_models

        self.DI, self.TI, self.MI = DI, TI, MI
        if self.TI:
            self.gaussian_kernel = self._TI_kernel()

    def perturb(self, images, ori_labels, target_labels):
        adv_images = images.clone()
        batch_size = adv_images.shape[0]
        if self.target:
            used_label = target_labels
            used_coef = -1
        else:
            used_label = ori_labels
            used_coef = 1
        
        delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        norm_matrix = np.zeros((self.steps, batch_size))

        iter_flag = 0
        if self.MI:
            grad_pre = 0
        # begin_time = time.time()
        for itr in range(self.steps):
            if self.DI:
                logits = self.model(self._DI(adv_images+delta))
                loss_label = used_label
            else:
                logits = self.model(adv_images+delta)
                loss_label = used_label
            loss = self.loss_fn(logits, loss_label)
            loss.backward()
            grad_c = delta.grad.clone()
            
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

            delta.grad.zero_()
            delta.data = delta.data + used_coef * self.alpha * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
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
        # end_time = time.time()
        # print (end_time-begin_time)
        return result_matrix, delta.data, norm_matrix

class DTMI_SI(Base):
    '''
    Use TI, MI and DI as the baseline attack.
    '''
    def __init__(self, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI_SI, self).__init__(loss_fn, eval_steps, steps, target)
        self.model = model
        self.eval_models = eval_models

        self.DI, self.TI, self.MI = DI, TI, MI
        if self.TI:
            self.gaussian_kernel = self._TI_kernel()

    def _si(self, inputs):
        inputs = torch.cat([inputs/(2**i) for i in range(5)], dim=0)
        return inputs

    def perturb(self, images, ori_labels, target_labels):
        adv_images = images.clone()
        batch_size = adv_images.shape[0]
        if self.target:
            used_label = target_labels
            used_coef = -1
        else:
            used_label = ori_labels
            used_coef = 1
        
        delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        norm_matrix = np.zeros((self.steps, batch_size))

        iter_flag = 0
        if self.MI:
            grad_pre = 0
        # begin_time = time.time()
        for itr in range(self.steps):
            augmented_inputs = self._si(adv_images+delta)
            loss_label = torch.cat([used_label]*int(augmented_inputs.shape[0]/batch_size))
            if self.DI:
                logits = self.model(self._DI(augmented_inputs))
            else:
                logits = self.model(augmented_inputs)
            loss = self.loss_fn(logits, loss_label)
            loss.backward()
            grad_c = delta.grad.clone()
            
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

            delta.grad.zero_()
            delta.data = delta.data + used_coef * self.alpha * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
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
        # end_time = time.time()
        # print (end_time-begin_time)
        return result_matrix, delta.data, norm_matrix

class DTMI_Local_FeatureSimilarityLoss_SI(Base):
    '''
    Incorporate locality of images and the feature similarity loss between global features and local features.
    '''
    def __init__(self, model_name, depth, coef, scale, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI_Local_FeatureSimilarityLoss_SI, self).__init__(loss_fn, eval_steps, steps, target)
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

    def _si(self, inputs):
        inputs = torch.cat([inputs/(2**i) for i in range(5)], dim=0)
        return inputs

    def perturb(self, images, ori_labels, target_labels):
        adv_images = images.clone()
        batch_size = adv_images.shape[0]
        if self.target:
            used_label = target_labels
            used_coef = -1
        else:
            used_label = ori_labels
            used_coef = 1
        
        delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        norm_matrix = np.zeros((self.steps, batch_size))
        iter_flag = 0
        if self.MI:
            grad_pre = 0
        for itr in range(self.steps):
            self.activations = []
            if self.DI:
                li_inputs = self._LI_WO_prob(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                augmented_inputs = self._si(accom_inputs)
                logits = self.model(self._DI(augmented_inputs))
                loss_label = torch.cat([used_label]*int(augmented_inputs.shape[0]/batch_size))
            else:
                li_inputs = self._LI_WO_prob(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                augmented_inputs = self._si(accom_inputs)
                logits = self.model(augmented_inputs)
                loss_label = torch.cat([used_label]*int(augmented_inputs.shape[0]/batch_size))
            
            classifier_loss = self.loss_fn(logits, loss_label)
            fs_losses = []
            for si_ind in range(5):
                fs_loss = torch.nn.functional.cosine_similarity(self.activations[0][si_ind*2*batch_size:(si_ind+1)*2*batch_size][:batch_size].view(batch_size, -1), self.activations[0][si_ind*2*batch_size:(si_ind+1)*2*batch_size][-batch_size:].view(batch_size, -1))
                fs_losses.append(fs_loss)
            fs_losses = torch.cat(fs_losses).mean()
            loss = classifier_loss + self.coef * used_coef * fs_losses
            loss.backward()

            grad_c = delta.grad.clone()
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

            delta.grad.zero_()
            delta.data = delta.data + used_coef * self.alpha * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
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
        return result_matrix, delta.data, norm_matrix

class DTMI_Admix(Base):
    '''
    Use TI, MI and DI as the baseline attack.
    '''
    def __init__(self, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI_Admix, self).__init__(loss_fn, eval_steps, steps, target)
        self.model = model
        self.eval_models = eval_models

        self.DI, self.TI, self.MI = DI, TI, MI
        if self.TI:
            self.gaussian_kernel = self._TI_kernel()

    def _admix(self, inputs, admix_images):
        inputs = torch.cat([inputs + 0.2 * admix_images[torch.randperm(admix_images.size(0))[0]].unsqueeze(dim=0) for i in range(3)], dim=0)
        inputs = torch.cat([inputs/(2**i) for i in range(5)], dim=0)
        return inputs

    def perturb(self, images, ori_labels, target_labels, admix_images):
        adv_images = images.clone()
        batch_size = adv_images.shape[0]
        if self.target:
            used_label = target_labels
            used_coef = -1
        else:
            used_label = ori_labels
            used_coef = 1
        
        delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        norm_matrix = np.zeros((self.steps, batch_size))

        iter_flag = 0
        if self.MI:
            grad_pre = 0
        # begin_time = time.time()
        for itr in range(self.steps):
            augmented_inputs = self._admix(adv_images+delta, admix_images)
            loss_label = torch.cat([used_label]*int(augmented_inputs.shape[0]/batch_size))
            
            if self.DI:
                logits = self.model(self._DI(augmented_inputs))
            else:
                logits = self.model(augmented_inputs)
            loss = self.loss_fn(logits, loss_label)
            loss.backward()
            grad_c = delta.grad.clone()
            
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

            delta.grad.zero_()
            delta.data = delta.data + used_coef * self.alpha * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
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
        # end_time = time.time()
        # print (end_time-begin_time)
        return result_matrix, delta.data, norm_matrix

class DTMI_Local_FeatureSimilarityLoss_Admix(Base):
    '''
    Incorporate locality of images and the feature similarity loss between global features and local features.
    '''
    def __init__(self, model_name, depth, coef, scale, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI_Local_FeatureSimilarityLoss_Admix, self).__init__(loss_fn, eval_steps, steps, target)
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

    def _admix(self, inputs, admix_images):
        inputs = torch.cat([inputs + 0.2 * admix_images[torch.randperm(admix_images.size(0))[0]].unsqueeze(dim=0) for i in range(3)], dim=0)
        inputs = torch.cat([inputs/(2**i) for i in range(5)], dim=0)
        return inputs

    def perturb(self, images, ori_labels, target_labels, admix_images):
        adv_images = images.clone()
        batch_size = adv_images.shape[0]
        if self.target:
            used_label = target_labels
            used_coef = -1
        else:
            used_label = ori_labels
            used_coef = 1
        
        delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        norm_matrix = np.zeros((self.steps, batch_size))
        iter_flag = 0
        if self.MI:
            grad_pre = 0
        for itr in range(self.steps):
            self.activations = []
            if self.DI:
                li_inputs = self._LI_WO_prob(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                augmented_inputs = self._admix(accom_inputs, admix_images)
                logits = self.model(self._DI(augmented_inputs))
                loss_label = torch.cat([used_label]*int(augmented_inputs.shape[0]/batch_size))
            else:
                li_inputs = self._LI_WO_prob(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                augmented_inputs = self._admix(accom_inputs, admix_images)
                logits = self.model(augmented_inputs)
                loss_label = torch.cat([used_label]*int(augmented_inputs.shape[0]/batch_size))
            
            classifier_loss = self.loss_fn(logits, loss_label)
            fs_losses = []
            for si_ind in range(15):
                fs_loss = torch.nn.functional.cosine_similarity(self.activations[0][si_ind*2*batch_size:(si_ind+1)*2*batch_size][:batch_size].view(batch_size, -1), self.activations[0][si_ind*2*batch_size:(si_ind+1)*2*batch_size][-batch_size:].view(batch_size, -1))
                fs_losses.append(fs_loss)
            fs_losses = torch.cat(fs_losses).mean()
            loss = classifier_loss + self.coef * used_coef * fs_losses
            loss.backward()

            grad_c = delta.grad.clone()
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

            delta.grad.zero_()
            delta.data = delta.data + used_coef * self.alpha * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
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
        return result_matrix, delta.data, norm_matrix


class DTEMI(Base):
    '''
    Use TI, EMI and DI as the baseline attack.
    '''
    def __init__(self, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTEMI, self).__init__(loss_fn, eval_steps, steps, target)
        self.model = model
        self.eval_models = eval_models

        self.DI, self.TI, self.MI = DI, TI, MI
        if self.TI:
            self.gaussian_kernel = self._TI_kernel()

        # parameters from https://github.com/JHL-HUST/EMI/blob/main/emi_fgsm.py
        sampling_number = 11
        sampling_interval = 7
        self.factors = np.linspace(-sampling_interval, sampling_interval, num=sampling_number)

    def perturb(self, images, ori_labels, target_labels):
        ori_b,_,_,_ = images.shape
        adv_images = images.clone()
        batch_size = adv_images.shape[0]
        if self.target:
            used_label = target_labels
            used_coef = -1
        else:
            used_label = ori_labels
            used_coef = 1
        
        delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        norm_matrix = np.zeros((self.steps, batch_size))

        iter_flag = 0
        if self.MI:
            grad_pre = 0
        # begin_time = time.time()
        for itr in range(self.steps):
            adv_lookaheads = [adv_images+delta+factor*self.alpha*grad_pre for factor in self.factors]
            avg_grad = None
            for adv_ind in range(len(self.factors)):
                adv_inputs = adv_lookaheads[adv_ind]
                if self.DI:
                    logits = self.model(self._DI(adv_inputs))
                    loss_label = used_label
                else:
                    logits = self.model(adv_inputs)
                    loss_label = used_label
                loss = self.loss_fn(logits, loss_label)
                loss.backward()
                grad_c = delta.grad.clone()
                delta.grad.zero_()
                if avg_grad == None:
                    avg_grad = grad_c
                else:
                    avg_grad += grad_c
            avg_grad = avg_grad/len(self.factors)
            # grad_norm = torch.norm(grad_c.view(batch_size, -1), dim=1).cpu().numpy()
            # norm_matrix[itr] = grad_norm

            if self.TI:
                grad_c = F.conv2d(avg_grad, self.gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
            else:
                pass
            if self.MI:
                grad_a = avg_grad / torch.mean(torch.abs(avg_grad), (1, 2, 3), keepdim=True) + 1 * grad_pre
                grad_pre = grad_a
            else:
                grad_a = avg_grad.clone()

            delta.grad.zero_()
            delta.data = delta.data + used_coef * self.alpha * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
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
        # end_time = time.time()
        # print (end_time-begin_time)
        return result_matrix, delta.data, norm_matrix

class DTMI_Local_FeatureSimilarityLoss_EMI(Base):
    '''
    Incorporate locality of images and the feature similarity loss between global features and local features.
    '''
    def __init__(self, model_name, depth, coef, scale, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI_Local_FeatureSimilarityLoss_EMI, self).__init__(loss_fn, eval_steps, steps, target)
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

        # parameters from https://github.com/JHL-HUST/EMI/blob/main/emi_fgsm.py
        sampling_number = 11
        sampling_interval = 7
        self.factors = np.linspace(-sampling_interval, sampling_interval, num=sampling_number)

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
        
        delta = torch.zeros_like(adv_images, requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        norm_matrix = np.zeros((self.steps, batch_size))
        iter_flag = 0
        if self.MI:
            grad_pre = 0
        for itr in range(self.steps):
            avg_grad = None
            for adv_ind, factor in enumerate(self.factors):
                self.activations = []
                adv_inputs = adv_images+delta+factor*self.alpha*grad_pre
                if self.DI:
                    li_inputs = self._LI_WO_prob(adv_inputs)
                    accom_inputs = torch.cat([adv_inputs, li_inputs], dim=0)
                    logits = self.model(self._DI(accom_inputs))
                    loss_label = torch.cat([used_label, used_label], dim=0)
                else:
                    logits = self.model(adv_inputs)
                    loss_label = used_label
                classifier_loss = self.loss_fn(logits, loss_label)
                fs_loss = torch.nn.functional.cosine_similarity(self.activations[0][:batch_size].view(batch_size, -1), self.activations[0][-batch_size:].view(batch_size, -1))
                fs_loss = torch.mean(fs_loss)
                loss = classifier_loss + self.coef * used_coef * fs_loss
                loss.backward()

                grad_c = delta.grad.clone()
                delta.grad.zero_()
                if avg_grad == None:
                    avg_grad = grad_c
                else:
                    avg_grad += grad_c
            avg_grad = avg_grad/len(self.factors)

            grad_c = avg_grad
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

            delta.grad.zero_()
            delta.data = delta.data + used_coef * self.alpha * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
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
        return result_matrix, delta.data, norm_matrix


class DTMI_Local_FeatureSimilarityLoss(Base):
    '''
    Incorporate locality of images and the feature similarity loss between global features and local features.
    '''
    def __init__(self, model_name, depth, coef, scale, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI_Local_FeatureSimilarityLoss, self).__init__(loss_fn, eval_steps, steps, target)
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
        
        delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        norm_matrix = np.zeros((self.steps, batch_size))
        iter_flag = 0
        if self.MI:
            grad_pre = 0
        for itr in range(self.steps):
            self.activations = []
            if self.DI:
                li_inputs = self._LI_WO_prob(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                logits = self.model(self._DI(accom_inputs))
                loss_label = torch.cat([used_label, used_label], dim=0)
            else:
                li_inputs = self._LI_WO_prob(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                logits = self.model(accom_inputs)
                loss_label = torch.cat([used_label, used_label], dim=0)
            
            classifier_loss = self.loss_fn(logits, loss_label)
            
            fs_loss = torch.nn.functional.cosine_similarity(self.activations[0][:batch_size].view(batch_size, -1), self.activations[0][-batch_size:].view(batch_size, -1))
            fs_loss = torch.mean(fs_loss)
            loss = classifier_loss + self.coef * used_coef * fs_loss
            loss.backward()

            grad_c = delta.grad.clone()
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

            delta.grad.zero_()
            delta.data = delta.data + used_coef * self.alpha * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
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
        return result_matrix, delta.data, norm_matrix

class DTMI_Random_FeatureSimilarityLoss(Base):
    '''
    Incorporate locality of images and the feature similarity loss between global features and local features.
    '''
    def __init__(self, model_name, depth, coef, scale, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI_Random_FeatureSimilarityLoss, self).__init__(loss_fn, eval_steps, steps, target)
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

    def _random_noise(self, X_in):
        return torch.empty_like(X_in).uniform_(0, 1)

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

        delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        iter_flag = 0
        if self.MI:
            grad_pre = 0
        for itr in range(self.steps):
            self.activations = []
            if self.DI:
                li_inputs = self._random_noise(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                logits = self.model(self._DI(accom_inputs))
                loss_label = torch.cat([used_label, used_label], dim=0)
            else:
                li_inputs = self._random_noise(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                logits = self.model(accom_inputs)
                loss_label = torch.cat([used_label, used_label], dim=0)
            
            classifier_loss = self.loss_fn(logits, loss_label)
            
            fs_loss = torch.nn.functional.cosine_similarity(self.activations[0][:batch_size].view(batch_size, -1), self.activations[0][-batch_size:].view(batch_size, -1))
            fs_loss = torch.mean(fs_loss)
            loss = classifier_loss + self.coef * used_coef * fs_loss
            loss.backward()

            grad_c = delta.grad.clone()
            if self.TI:
                grad_c = F.conv2d(grad_c, self.gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
            else:
                pass
            if self.MI:
                grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre
                grad_pre = grad_a
            else:
                grad_a = grad_c.clone()

            delta.grad.zero_()
            delta.data = delta.data + used_coef * self.alpha * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
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
        return result_matrix, delta.data


class DTMI_Local_FeatureSimilarityLoss_Center(Base):
    '''
    Incorporate locality of images and the feature similarity loss between global features and local features.
    '''
    def __init__(self, UseCenter, model_name, depth, coef, scale, model, eval_models, loss_fn, eval_steps, steps, target, DI=False, TI=False, MI=False):
        super(DTMI_Local_FeatureSimilarityLoss_Center, self).__init__(loss_fn, eval_steps, steps, target)
        self.start, self.interval = scale
        self.UseCenter = UseCenter
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
        b,_,_,_ = X_in.shape
        ious = 0.0
        while ious<0.5:
            i,j,h,w = self.local_transform.get_params(X_in, self.local_transform.scale, self.local_transform.ratio)
            mask = torch.zeros_like(X_in)
            mask[:,:,i:i+h,j:j+w] = 1
            ious = torch.sum(mask * self.center)/(b*h*w)
        return torchvision.transforms.functional.resized_crop(X_in, i, j, h, w, self.local_transform.size, self.local_transform.interpolation)

    def perturb(self, images, ori_labels, target_labels):
        self.center = torch.zeros_like(images)
        if self.UseCenter:
            self.center[:,:,44:255,44:255] = 1
        else:
            self.center[:,:,44:255,44:255] = 1
            self.center = 1-self.center
        adv_images = images.clone()
        batch_size = adv_images.shape[0]
        if self.target:
            used_label = target_labels
            used_coef = -1
        else:
            used_label = ori_labels
            used_coef = 1
        
        delta = torch.zeros_like(adv_images,requires_grad=True).cuda()

        result_matrix = np.zeros((len(self.eval_models), len(self.eval_steps)))
        iter_flag = 0
        if self.MI:
            grad_pre = 0
        for itr in range(self.steps):
            self.activations = []
            if self.DI:
                li_inputs = self._LI_WO_prob(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                logits = self.model(self._DI(accom_inputs))
                loss_label = torch.cat([used_label, used_label], dim=0)
            else:
                li_inputs = self._LI_WO_prob(adv_images)
                accom_inputs = torch.cat([adv_images+delta, li_inputs+delta], dim=0)
                logits = self.model(accom_inputs)
                loss_label = torch.cat([used_label, used_label], dim=0)
            
            classifier_loss = self.loss_fn(logits, loss_label)
            
            fs_loss = torch.nn.functional.cosine_similarity(self.activations[0][:batch_size].view(batch_size, -1), self.activations[0][-batch_size:].view(batch_size, -1))
            fs_loss = torch.mean(fs_loss)
            loss = classifier_loss + self.coef * used_coef * fs_loss
            loss.backward()

            grad_c = delta.grad.clone()
            if self.TI:
                grad_c = F.conv2d(grad_c, self.gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
            else:
                pass
            if self.MI:
                grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True) + 1 * grad_pre
                grad_pre = grad_a
            else:
                grad_a = grad_c.clone()

            delta.grad.zero_()
            delta.data = delta.data + used_coef * self.alpha * torch.sign(grad_a)
            delta.data = delta.data.clamp(-self.epsilon, self.epsilon) 
            delta.data = ((adv_images + delta.data).clamp(0,1)) - adv_images
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
        return result_matrix, delta.data
