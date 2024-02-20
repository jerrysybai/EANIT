# Copyright (c) Microsoft, Inc. 2020 
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from torch import nn as nn

def KL(input, target, noise_pos, normal_pos, reduction="sum"):
    bsz, _, _ = input.shape
    input = input[torch.arange(bsz).unsqueeze(-1), noise_pos[:, :]]
    target = target[torch.arange(bsz).unsqueeze(-1), normal_pos[:, :]]
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32), F.softmax(target, dim=-1, dtype=torch.float32), reduction=reduction)
    return loss, 0

def SKL(logit, target, epsilon=1e-8):
    logit = logit.view(-1, logit.size(-1)).float()
    target = target.view(-1, target.size(-1)).float()
    #bs = logit.size(0)
    p = F.log_softmax(logit, 1).exp()
    y = F.log_softmax(target, 1).exp()
    rp = -(1.0/(p + epsilon) -1 + epsilon).detach().log()
    ry = -(1.0/(y + epsilon) -1 + epsilon).detach().log()
    return (p* (rp- ry) * 2).sum()


class AdvMaskedLmLoss(object):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, ignore_index):
        self.args = args
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def adv_project(self, grad, norm_type='inf', eps=1e-6):
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction
    
    # def Sim_cos(self, logit, target, adv_step_size = None):
    #     cos_sim = F.cosine_similarity(logit,target,-1).unsqueeze(-1).expand_as(logit)
    #     if adv_step_size is not None:
    #         cos_sim = torch.clamp(cos_sim, -adv_step_size, adv_step_size)
    #     return cos_sim
    
    def get_loss(self, logits, target_mask, input_ids):
        # 将labels中不属于target的部分，设为ignore_index，只计算target部分的loss
        labels = torch.where(target_mask == 1, input_ids, self.ignore_index)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        #计算正常样本的损失
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, labels

    def __call__(self, model, inputs, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        target_mask = inputs['target_mask']
        normal_pos = inputs['normal_pos']
        noise_pos = inputs['noise_pos']
        #计算l(x,\thete)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]

        #计算正常样本的损失
        loss, labels = self.get_loss(logits, target_mask, input_ids)
        
        if self.args.add_nosie:
            # output = model(x)
            # loss = get_loss(output, y)
            
                # class noise_restore->
                # for i in model.layers:
                    # if i is embedding_layer:
                    #     加noise, store(embedding_layer.weight)
                
            # model = noise-restor(model).noise
            # adv_loss = get_loss(model)
            # adv_loss.backward()
            # g = adv_loss.gradient
            # model = noise_restore(model).restore
            
            # loss.backward()
            
            # input: [b, l, 4096] new_input = [b, l, 4096] + [b, l, 1] = [b, l, 4096]
            #构建对抗样本
            model.zero_grad()
            embed = outputs['hidden_states']
            dims = torch.tensor(input_ids.size(1) * 4096)
            mag_norm = 10/torch.sqrt(dims)
            noise = torch.zeros(embed.size()).uniform_(-mag_norm, mag_norm).to(input_ids.device) #* self.args.noise_var
            noise.requires_grad_() #噪声向量初始化
            adv_outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, noise_adv = noise)#, noise_mask = inputs["ins_mask"]
            adv_logits = adv_outputs["logits"] if isinstance(adv_outputs, dict) else adv_outputs[0]
            adv_loss, _ = self.get_loss(adv_logits, target_mask.detach(), input_ids.detach()) if not self.args.useKL else KL(adv_logits, logits.detach(), noise_pos, normal_pos)
            
            # adv_loss.backward(retain_graph=True)
            adv_loss.backward()
            delta_grad = noise.grad #对噪声向量计算梯度
            norm = delta_grad.norm()
            model.zero_grad()
            
            if (torch.isnan(norm) or torch.isinf(norm)):  #会出现nan值的处理，这里我觉得很有必要，因为我自己搞的时候就发现很容易出现nan值
                # skim this batch
                return loss, outputs, labels
            
            new_noise = noise + (delta_grad / norm) * self.args.adv_step_size  #更新噪声向量
            # line 6 projection
            # new_noise = self.adv_project(noise, norm_type=self.args.project_norm_type, eps=self.args.noise_gamma)
            adv_outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, noise_adv = new_noise)#, noise_mask = inputs["ins_mask"]) #再走一遍网络，一共走三遍
            adv_logits = adv_outputs["logits"] if isinstance(adv_outputs, dict) else adv_outputs[0]
            adv_loss, _ = self.get_loss(adv_logits, target_mask, input_ids) if not self.args.useKL else KL(adv_logits, logits.detach(), noise_pos, normal_pos)
            # line 8 symmetric KL
            
            loss = loss + 1.0 * adv_loss
            
        return loss, outputs, labels
    
    

class AdvMaskedLmLoss_2steps(object):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, ignore_index):
        self.args = args
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def adv_project(self, grad, norm_type='inf', eps=1e-6):
        if norm_type == 'l2':
            direction = grad / (torch.norm(grad, dim=-1, keepdim=True) + eps)
        elif norm_type == 'l1':
            direction = grad.sign()
        else:
            direction = grad / (grad.abs().max(-1, keepdim=True)[0] + eps)
        return direction
    
    # def Sim_cos(self, logit, target, adv_step_size = None):
    #     cos_sim = F.cosine_similarity(logit,target,-1).unsqueeze(-1).expand_as(logit)
    #     if adv_step_size is not None:
    #         cos_sim = torch.clamp(cos_sim, -adv_step_size, adv_step_size)
    #     return cos_sim
    
    def get_loss(self, logits, target_mask, input_ids):
        # 将labels中不属于target的部分，设为ignore_index，只计算target部分的loss
        labels = torch.where(target_mask == 1, input_ids, self.ignore_index)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        #计算正常样本的损失
        loss = self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        return loss, labels

    def __call__(self, model, inputs, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        target_mask = inputs['target_mask']
        normal_pos = inputs['normal_pos']
        noise_pos = inputs['noise_pos']
        #计算l(x,\thete)
        
        if self.args.add_nosie:
            dims = torch.tensor(input_ids.size(1) * 4096)
            mag_norm = 1/torch.sqrt(dims)
            noise = torch.zeros([input_ids.size(0), input_ids.size(1), 4096]).uniform_(-mag_norm, mag_norm).to(input_ids.device) #* self.args.noise_var
            noise.requires_grad_() #噪声向量初始化
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, noise_adv = noise)#, noise_mask = inputs["ins_mask"]
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            loss, labels = self.get_loss(logits, target_mask.detach(), input_ids.detach()) 
            
            # adv_loss.backward(retain_graph=True)
            loss.backward(retain_graph=True)
            delta_grad = noise.grad #对噪声向量计算梯度
            norm = delta_grad.norm()
            model.zero_grad()
            
            if (torch.isnan(norm) or torch.isinf(norm)):  #会出现nan值的处理，这里我觉得很有必要，因为我自己搞的时候就发现很容易出现nan值
                # skim this batch
                return loss, outputs, labels
            
            new_noise = noise + (delta_grad / norm) * self.args.adv_step_size  #更新噪声向量
            # line 6 projection
            # new_noise = self.adv_project(noise, norm_type=self.args.project_norm_type, eps=self.args.noise_gamma)
            adv_outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, noise_adv = new_noise)#, noise_mask = inputs["ins_mask"]) #再走一遍网络，一共走三遍
            adv_logits = adv_outputs["logits"] if isinstance(adv_outputs, dict) else adv_outputs[0]
            adv_loss, _ = self.get_loss(adv_logits, target_mask, input_ids) if not self.args.useKL else KL(adv_logits, logits.detach(), noise_pos, normal_pos)
            # line 8 symmetric KL
            
            loss = loss + 1.0 * adv_loss
            
        return loss, outputs, labels