import os
import os.path as osp

from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
cudnn.enabled = True

import argparse

import numpy as np
import random
from datetime import datetime

from networks import get_deeplab
from helpers import get_camvid_label, get_cityscapes_label
from helpers import BuildDataLoader, PolyLR
from helpers import compute_supervised_loss

from helpers import fast_hist, calculate_iou

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Supervised Segmentation with Perfect Labels - Only Labeled Set')

parser.add_argument('--seed', default=3108, type=int)
parser.add_argument('--dataset', default='camvid', type=str)

parser.add_argument('--backbone', default='mbv2', type=str)
parser.add_argument('--dilate_scale', default=16, type=int)

parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--epochs', default=100, type=int)

parser.add_argument('--save_dir', default='dummy', help='lost runs are a result of your laziness')

args = parser.parse_args()

os.makedirs(osp.join('logs', args.backbone, args.save_dir, 'checkpoints'), exist_ok=True)
    
tensorboard_writer = SummaryWriter(osp.join('logs', args.backbone, args.save_dir))

if args.dataset == 'camvid':
    mask_label_mapper = get_camvid_label()
elif args.dataset == 'cityscapes':
    mask_label_mapper = get_cityscapes_label()
else:
    raise NotImplementedError("Dataset not defined")
    
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

data_loader_main = BuildDataLoader(dataset=args.dataset)
print("Defined dataloader for {}".format(args.dataset))

train_al_loader, val_loader, test_loader = data_loader_main.build_supervised_al()

if torch.cuda.is_available() and not args.onlycpu:
    device = torch.device("cuda")
else:
    device = 'cpu'
    
model = get_deeplab(backbone=args.backbone, num_classes=data_loader_main.classes, dilate_scale=args.dilate_scale).to(device)
print("Initialized model with {}".format(args.backbone))

optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
scheduler = PolyLR(optimizer, args.epochs, power=0.9)

torch.save({
            'epoch': 0,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, osp.join('logs', args.backbone, args.save_dir, 'checkpoints', 'epoch_0.pt'))

train_epoch = len(train_al_loader)
val_epoch = len(val_loader)
test_epoch = len(test_loader)

iteration = 1

for epoch_idx in range(args.epochs):
    
    # Train
    
    train_al_dataset = iter(train_al_loader)
    model.train()
    
    for iter_idx in range(train_epoch):
        optimizer.zero_grad()
        
        data_obj_al = train_al_dataset.next()
        imgs, segmaps = data_obj_al['img'].to(device), data_obj_al['segmap'].to(device)
        
        predmaps = model(imgs)
        predmaps_xl = F.interpolate(predmaps, size = segmaps.shape[1:], mode='bilinear', align_corners=True)

        sup_loss = compute_supervised_loss(predmaps_xl, segmaps)
        
        loss = sup_loss
        
        loss.backward()
        optimizer.step()
        
        if (iteration)%5 == 0:
            print("{} | EPOCH {:02d} of {:02d} | ITER {:05d} | SUP_LOSS {:.4f} | TRAIN LOSS {:.4f}".\
                  format(datetime.now().strftime("%H:%M:%S"), epoch_idx, args.epochs, iteration, sup_loss, loss))
            tensorboard_writer.add_scalar('supervised loss', sup_loss.item(), iteration)
            tensorboard_writer.add_scalar('total loss', loss.item(), iteration)
            
        iteration += 1
        
    # If epoch is a multiple of 10, save model and compute scores on validation
    # and test sets
            
    if (epoch_idx + 1)%10==0:
        
        torch.save({
            'epoch': (epoch_idx + 1),
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            }, osp.join('logs', args.backbone, args.save_dir, 'checkpoints', 'epoch_{}.pt'.format(epoch_idx + 1)))
        
        # Validation
        
        iou_acc  = 0
        with torch.no_grad():
            model.eval()
            val_dataset = iter(val_loader)
            
            for iter_idx in tqdm(range(val_epoch)):
                
                data_obj_v = val_dataset.next()
                imgs_v, segmaps_v = data_obj_v['img'].to(device), data_obj_v['segmap'].to(device)
                
                predmaps_v = model(imgs_v)
                predmaps_v_xl = F.interpolate(predmaps_v, size = segmaps_v.shape[1:], mode='bilinear', align_corners=True)
                
                _iou_acc = fast_hist(pred=predmaps_v_xl.argmax(1).flatten().cpu().numpy(),
                                      gtruth=segmaps_v.flatten().cpu().numpy(),
                                      num_classes=data_loader_main.classes)
                iou_acc += _iou_acc

        c_iou, c_acc = calculate_iou(iou_acc)
        m_iou = np.nanmean(c_iou)
        
        print("{} | EPOCH {:02d} of {:02d} | VAL MPCA {:.4f} | VAL MIOU {:.4f}".format(datetime.now().strftime("%H:%M:%S"), epoch_idx, args.epochs, c_acc, m_iou))
        for c_idx in range(len(c_iou)):
            tensorboard_writer.add_scalar('val_{}'.format(mask_label_mapper[c_idx]), c_iou[c_idx], (epoch_idx + 1))
        tensorboard_writer.add_scalar('val_miou', m_iou, (epoch_idx + 1))
        tensorboard_writer.add_scalar('val_mpca', c_acc, (epoch_idx + 1))
        
        # Test
        
        iou_acc  = 0
        with torch.no_grad():
            model.eval()
            test_dataset = iter(test_loader)
            
            for iter_idx in tqdm(range(test_epoch)):
                
                data_obj_v = test_dataset.next()
                imgs_v, segmaps_v = data_obj_v['img'].to(device), data_obj_v['segmap'].to(device)
                
                predmaps_v = model(imgs_v)
                predmaps_v_xl = F.interpolate(predmaps_v, size = segmaps_v.shape[1:], mode='bilinear', align_corners=True)
                
                _iou_acc = fast_hist(pred=predmaps_v_xl.argmax(1).flatten().cpu().numpy(),
                                      gtruth=segmaps_v.flatten().cpu().numpy(),
                                      num_classes=data_loader_main.classes)
                iou_acc += _iou_acc

        c_iou, c_acc = calculate_iou(iou_acc)
        m_iou = np.nanmean(c_iou)
        
        print("{} | EPOCH {:02d} of {:02d} | TEST MPCA {:.4f} | TEST MIOU {:.4f}".format(datetime.now().strftime("%H:%M:%S"), epoch_idx, args.epochs, c_acc, m_iou))
        for c_idx in range(len(c_iou)):
            tensorboard_writer.add_scalar('test_{}'.format(mask_label_mapper[c_idx]), c_iou[c_idx], (epoch_idx + 1))
        tensorboard_writer.add_scalar('test_miou', m_iou, (epoch_idx + 1))
        tensorboard_writer.add_scalar('test_mpca', c_acc, (epoch_idx + 1))
        
    scheduler.step()
    
tensorboard_writer.close()

