import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import copy
import random

from torch.optim.lr_scheduler import _LRScheduler

# -----------------------------------------------------------------------------
# Define Polynomial Decay
# -----------------------------------------------------------------------------

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]
    
# -----------------------------------------------------------------------------
# Define EMA: Mean Teacher Framework
# -----------------------------------------------------------------------------

class EMA(object):
    def __init__(self, model, alpha):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def update(self, model):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1
        
# -----------------------------------------------------------------------------
# Define supervised training loss
# -----------------------------------------------------------------------------

def compute_supervised_loss(predict, target, reduction=True):
    assert not target.requires_grad
    assert predict.size(0)==target.size(0)
    assert predict.size(2)==target.size(1)
    assert predict.size(3)==target.size(2)

    if reduction:
        loss = F.cross_entropy(predict, target, ignore_index=-1)
    else:
        loss = F.cross_entropy(predict, target, ignore_index=-1, reduction='none')
    return loss

# -----------------------------------------------------------------------------
# Define pseudo-supervised training loss
# Confidence weighting - unsup_weight
# -----------------------------------------------------------------------------

def compute_unsupervised_loss(predict, target, logits, unsup_weight= 0, strong_threshold=0.97):
    batch_size = predict.shape[0]
    valid_mask = (target >= 0).float()   # only count valid pixels

    weighting = logits.view(batch_size, -1).ge(strong_threshold).sum(-1) / valid_mask.view(batch_size, -1).sum(-1)
    
    loss = F.cross_entropy(predict, target, reduction='none', ignore_index=-1)
    if unsup_weight > 0:
        loss = loss * logits
        
    weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None] * loss, loss > 0))
    return weighted_loss

# -----------------------------------------------------------------------------
# Define replay buffer
# Realistically, we already load all images into the dataset array
# So this buffer only stores the array locations for sampling later
# -----------------------------------------------------------------------------

class pseudo_replay_buffer:
    def __init__(self, max_buffer_length=100):
        self.locindexs_replay = []
        self.max_buffer_length = max_buffer_length
        
    def add(self, x):
        self.locindexs_replay.extend(x.tolist())
        random.shuffle(self.locindexs_replay)
        
        if len(self.locindexs_replay) > self.max_buffer_length:
            self.locindexs_replay = self.locindexs_replay[:self.max_buffer_length]
        
    def sample(self, len_samples=4):
        chosenones = [self.locindexs_replay.pop() for runs in range(len_samples)]
        return chosenones
    
    def __len__(self):
        return len(self.locindexs_replay)