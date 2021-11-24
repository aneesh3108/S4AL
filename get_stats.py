import os
import os.path as osp

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.enabled = True

import argparse
import matplotlib.pylab as plt

import numpy as np
import random
from datetime import datetime
import copy

from networks import get_deeplab
from helpers import BuildDataLoader

from helpers import fast_hist, calculate_iou, get_stats

parser = argparse.ArgumentParser(description='Supervised Segmentation with Perfect Labels')

parser.add_argument('--seed', default=3108, type=int)
parser.add_argument('--dataset', default='camvid', type=str)

parser.add_argument('--backbone', default='mbv2', type=str)
parser.add_argument('--dilate_scale', default=16, type=int)

parser.add_argument('--pathtomodel', default = None, type=str)

args = parser.parse_args()

data_loader_main = BuildDataLoader(dataset=args.dataset)
print("Defined dataloader for {}".format(args.dataset))

test_loader = data_loader_main.build_testloader()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = 'cpu'

model1 = get_deeplab(backbone=args.backbone, num_classes=data_loader_main.classes, dilate_scale=args.dilate_scale)
print("Initialized model with {}".format(args.backbone))

# working_ckpt = 'logs/mbv2/camvid/camvid_s4al/checkpoints/epoch_gen2_200.pt'
working_ckpt = args.pathtomodel
all_weights = torch.load(working_ckpt, map_location=torch.device('cpu'))

try:
    model1.load_state_dict(all_weights['teacher_state_dict'])
except:
    model1.load_state_dict(all_weights['model_state_dict'])
    
model1.to(device)    
 
test_epoch = len(test_loader)
test_dataset = iter(test_loader)
    
hist  = 0

with torch.no_grad():
    model1.eval()
    
    for iter_idx in tqdm(range(test_epoch)):
        
        data_obj_v = test_dataset.next()
        imgs_v, segmaps_v = data_obj_v['img'].to(device), data_obj_v['segmap'].to(device)
        
        predmaps_v = model1(imgs_v)
        predmaps_v_xl = F.interpolate(predmaps_v, size = segmaps_v.shape[1:], mode='bilinear', align_corners=True)
        
        _hist = fast_hist(pred=predmaps_v_xl.argmax(1).flatten().cpu().numpy(),
                              gtruth=segmaps_v.flatten().cpu().numpy(),
                              num_classes=data_loader_main.classes)
        hist += _hist
        
iu, _ = calculate_iou(hist)

get_stats(hist, iu, dataset = args.dataset)