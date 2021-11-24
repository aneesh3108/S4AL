import os
import os.path as osp

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data.sampler as sampler

cudnn.benchmark = True
cudnn.enabled = True

import argparse

import numpy as np
import random
from datetime import datetime

from networks import get_deeplab
from helpers import get_camvid_label, get_cityscapes_label
from helpers import BuildDataLoader, BuildDatasetAL, PolyLR, EMA
from helpers import compute_supervised_loss, compute_unsupervised_loss
from helpers import batch_transform, generate_unsup_data, generate_unsup_data_biased, pseudo_replay_buffer

from helpers import fast_hist, calculate_iou

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Semi-Supervised Segmentation with Perfect Labels')

parser.add_argument('--seed', default=3108, type=int)
parser.add_argument('--dataset', default='camvid', type=str)

parser.add_argument('--backbone', default='mbv2', type=str)
parser.add_argument('--dilate_scale', default=16, type=int)

parser.add_argument('--lr', default=1e-2, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--generations', default=5, type=int, help='this is the number of times to run active learning loop')
parser.add_argument('--unsup_weight', default=0, type=float)

parser.add_argument('--coldstart', action='store_true')
parser.add_argument('--strong_threshold', default=0.97, type=float)

parser.add_argument('--apply_aug', default='classmix', type=str, help='apply semi-supervised method: cutout cutmix classmix')
parser.add_argument('--region_size', default=30, type=int)
parser.add_argument('--num_regions', default=4, type=int)

parser.add_argument('--start_adaptive', default=1, type=int)
parser.add_argument('--max_buffer_length', default=50, type=int)

parser.add_argument('--save_dir', default='dummy', help='lost runs are a result of your laziness')

args = parser.parse_args()

os.makedirs(osp.join('logs', args.backbone, args.save_dir, 'checkpoints'), exist_ok=True)
  
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

# Define unlabeled dataset

train_ul_dataset_core = BuildDatasetAL(root=data_loader_main.root,
                                        dataset=data_loader_main.dataset,
                                        idx_list='train_al_unlab',
                                        crop_size=data_loader_main.crop_size,
                                        scale_size=data_loader_main.scale_size,
                                        augmentation_flip=True,
                                        augmentation_color=False,
                                        tensor_tx=True,
                                        trainval=True
                                        )

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = 'cpu'
    
softmax_fx = nn.Softmax2d()

if args.dataset == 'camvid':
    adaptive_pool_op_size = (int(360/args.region_size), int(480/args.region_size))
elif args.dataset == 'cityscapes':
    adaptive_pool_op_size = (int(688/args.region_size), int(688/args.region_size))
else:
    raise NotImplementedError("Dataset not in supported list of datasets")
    
for generation_idx in range(args.generations+1):
    
    if generation_idx == args.generations:
        total_number_of_epochs = args.epochs * 2
    else:
        total_number_of_epochs = args.epochs
    
    # Replay buffer
    
    myreplaybuffer = pseudo_replay_buffer(max_buffer_length=args.max_buffer_length)
    
    tensorboard_writer = SummaryWriter(osp.join('logs', args.backbone, args.save_dir), filename_suffix='gen_{}'.format(generation_idx))
    
    num_samples = data_loader_main.batch_size * data_loader_main.batch_iters//2
    
    train_ul_loader = torch.utils.data.DataLoader(dataset=train_ul_dataset_core,
                                                  batch_size=data_loader_main.batch_size,
                                                  sampler=sampler.RandomSampler(data_source=train_ul_dataset_core,
                                                                                replacement=True,
                                                                                num_samples=num_samples),
                                                  drop_last=True,
                                                  num_workers=4
                                                  )
    
    model = get_deeplab(backbone=args.backbone, num_classes=data_loader_main.classes, dilate_scale=args.dilate_scale).to(device)
    print("Initialized model with {}".format(args.backbone))
    
    ema = EMA(model, 0.99)
    print("Initialized teacher model with {}".format(args.backbone))
    
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=True)
    scheduler = PolyLR(optimizer, total_number_of_epochs, power=0.9)
    
    torch.save({
                'epoch': 0,
                'model_state_dict': model.state_dict(),
                'teacher_state_dict': ema.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, osp.join('logs', args.backbone, args.save_dir, 'checkpoints', 'epoch_gen_{}_0.pt'.format(generation_idx)))
    
    train_epoch = len(train_al_loader)
    val_epoch = len(val_loader)
    test_epoch = len(test_loader)
    
    iteration = 1
    
    for epoch_idx in range(total_number_of_epochs):
        
        # Train
        
        train_al_dataset = iter(train_al_loader)
        train_ul_dataset = iter(train_ul_loader)
        
        model.train()
        ema.model.train()
        
        for iter_idx in range(train_epoch):
            
            optimizer.zero_grad()
    
            data_obj_al = train_al_dataset.next()
            imgs_al, segmaps_al = data_obj_al['img'].to(device), data_obj_al['segmap'].to(device)
            
            predmaps_al = model(imgs_al)
            predmaps_al_xl = F.interpolate(predmaps_al, size = segmaps_al.shape[1:], mode='bilinear', align_corners=True)
            
            sup_loss = compute_supervised_loss(predmaps_al_xl, segmaps_al)
            
            if (epoch_idx<10) and args.coldstart:
                unsup_loss = torch.tensor(0.0).to(device)
                adaptive_unsup_loss = torch.tensor(0.0).to(device)
                sup_loss = 1.5 * sup_loss
                
            else:
                
                data_obj_ul = train_ul_dataset.next()
                imgs_ul, segmaps_ul, segmaps_mask_ul = data_obj_ul['img'].to(device), data_obj_ul['segmap'].to(device), data_obj_ul['segmap_mask'].to(device)
                
                # generate pseudo-labels from teacher model
                
                with torch.no_grad():
                    predmaps_ul = ema.model(imgs_ul)    
                    
                    predmaps_ul_xl = F.interpolate(predmaps_ul, size = segmaps_al.shape[1:], mode='bilinear', align_corners=True)
                    
                    pseudo_logits, pseudo_labels = torch.max(torch.softmax(predmaps_ul_xl, dim=1), dim=1)
                    
                    # for sampled regions, replace pseudo labels with annotated ground truths
                    
                    pseudo_labels[segmaps_mask_ul>0] = segmaps_ul[segmaps_mask_ul>0]
                    pseudo_logits[segmaps_mask_ul>0] = 1.0
    
                    # random scale images first
                    
                    train_u_aug_imgs, train_u_aug_segmaps, train_u_aug_logits = \
                        batch_transform(imgs_ul, pseudo_labels, pseudo_logits,
                                        data_loader_main.crop_size, data_loader_main.scale_size, apply_augmentation=False)
        
                    # apply mixing strategy: cutout, cutmix or classmix
                    
                    train_u_aug_imgs, train_u_aug_segmaps, train_u_aug_logits = \
                        generate_unsup_data(train_u_aug_imgs, train_u_aug_segmaps, train_u_aug_logits, mode=args.apply_aug)
        
                    # apply augmentation: flip + color jitter

                    train_u_aug_imgs, train_u_aug_segmaps, train_u_aug_logits = \
                        batch_transform(train_u_aug_imgs, train_u_aug_segmaps, train_u_aug_logits,
                                        data_loader_main.crop_size, (1.0, 1.0), apply_augmentation=True)
                        
                # train with pseudo-labels from teacher model on student model
            
                predmaps_ul = model(train_u_aug_imgs)
                predmaps_ul_xl = F.interpolate(predmaps_ul, size=segmaps_al.shape[1:], mode='bilinear', align_corners=True)
                
                unsup_loss = compute_unsupervised_loss(predict=predmaps_ul_xl,
                                                        target=train_u_aug_segmaps,
                                                        logits=train_u_aug_logits,
                                                        unsup_weight=args.unsup_weight,
                                                        strong_threshold=args.strong_threshold)
                
                if generation_idx >= args.start_adaptive:
                    
                    if (epoch_idx>29):
                        # TODO: Make this an argument as well
                        
                        # Sample indices and corresponding tensor arrays from replay buffer
                        
                        adaptive_classmix_locations = myreplaybuffer.sample(len_samples = data_loader_main.batch_size)
                        
                        imgs_ul2 = torch.stack(([train_ul_dataset_core.__getitem__(i)['img'] for i in adaptive_classmix_locations])).to(device)
                        segmaps_ul2 = torch.stack(([train_ul_dataset_core.__getitem__(i)['segmap'] for i in adaptive_classmix_locations])).to(device)
                        segmaps_mask_ul2 = torch.stack(([train_ul_dataset_core.__getitem__(i)['segmap_mask'] for i in adaptive_classmix_locations])).to(device)
                        
                        # generate pseudo-labels from teacher model

                        with torch.no_grad():
                            predmaps_ul2 = ema.model(imgs_ul2)
                            
                            predmaps_ul_xl2 = F.interpolate(predmaps_ul2, size = segmaps_al.shape[1:], mode='bilinear', align_corners=True)
                            
                            pseudo_logits2, pseudo_labels2 = torch.max(torch.softmax(predmaps_ul_xl2, dim=1), dim=1)
                            
                            # for sampled regions, replace pseudo labels with annotated ground truths

                            pseudo_labels2[segmaps_mask_ul2>0] = segmaps_ul2[segmaps_mask_ul2>0]
                            pseudo_logits2[segmaps_mask_ul2>0] = 1.0
                            
                            # stack replay buffer images with current mini-batch
                          
                            train_u_aug_imgs = torch.cat((imgs_ul, imgs_ul2))
                            train_u_aug_segmaps = torch.cat((pseudo_labels, pseudo_labels2))
                            train_u_aug_logits = torch.cat((pseudo_logits, pseudo_logits2))
                            
                            # apply mixing strategy: cutout, cutmix or classmix
                            
                            train_u_aug_imgs, train_u_aug_segmaps, train_u_aug_logits = \
                                generate_unsup_data_biased(train_u_aug_imgs, train_u_aug_segmaps, train_u_aug_logits, mode=args.apply_aug, dataset=args.dataset)
                
                            # apply augmentation: flip + color jitter
                            
                            train_u_aug_imgs, train_u_aug_segmaps, train_u_aug_logits = \
                                batch_transform(train_u_aug_imgs, train_u_aug_segmaps, train_u_aug_logits,
                                                data_loader_main.crop_size, (1.0, 1.0), apply_augmentation=True)
                                
                        # train with pseudo-labels from teacher model on student model
                            
                        predmaps_ul2 = model(train_u_aug_imgs)
                        predmaps_ul_xl2 = F.interpolate(predmaps_ul2, size=segmaps_al.shape[1:], mode='bilinear', align_corners=True)
                    
                        adaptive_unsup_loss = compute_unsupervised_loss(predict=predmaps_ul_xl2,
                                                                        target=train_u_aug_segmaps,
                                                                        logits=train_u_aug_logits,
                                                                        unsup_weight=args.unsup_weight,
                                                                        strong_threshold=args.strong_threshold)
                        
                    else:
                        adaptive_unsup_loss = torch.tensor(0.0).to(device)
                        
                    myreplaybuffer.add(data_obj_ul['index'])
                    
                else:
                    adaptive_unsup_loss = torch.tensor(0.0).to(device)
                
            loss = sup_loss + unsup_loss + adaptive_unsup_loss
            
            loss.backward()
            optimizer.step()
            
            ema.update(model)
            
            if (iteration)%5 == 0:
                print("{} | GEN {:02d} of {:02d} | EPOCH {:02d} of {:02d} | ITER {:05d} | SUP_LOSS {:.4f} | UNSUP_LOSS {:.4f} | AD_UNSUP_LOSS {:.4f} | TRAIN LOSS {:.4f}".\
                      format(datetime.now().strftime("%H:%M:%S"), generation_idx, args.generations+1, epoch_idx, total_number_of_epochs, iteration,\
                              sup_loss, unsup_loss, adaptive_unsup_loss, loss))
                tensorboard_writer.add_scalar('supervised loss', sup_loss.item(), iteration)
                tensorboard_writer.add_scalar('unsupervised loss', unsup_loss.item(), iteration)
                tensorboard_writer.add_scalar('adaptive unsupervised loss', adaptive_unsup_loss.item(), iteration)
                tensorboard_writer.add_scalar('total loss', loss.item(), iteration)
                
            iteration += 1
                
        if (epoch_idx + 1)%10==0:
            
            torch.save({
                'epoch': (epoch_idx + 1),
                'model_state_dict': model.state_dict(),
                'teacher_state_dict': ema.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                }, osp.join('logs', args.backbone, args.save_dir, 'checkpoints', 'epoch_gen{}_{}.pt'.format(generation_idx,epoch_idx + 1)))
                        
            # Validation
            
            model.eval()
            ema.model.eval()

            iou_acc  = 0
            with torch.no_grad():
                
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
            
            print("{} | GEN {:02d} of {:02d} | EPOCH {:02d} of {:02d} | VAL MPCA {:.4f} | VAL MIOU {:.4f}".format(
                datetime.now().strftime("%H:%M:%S"), generation_idx, args.generations+1, epoch_idx, total_number_of_epochs, c_acc, m_iou))
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
            
            print("{} | GEN {:02d} of {:02d} | EPOCH {:02d} of {:02d} | TEST MPCA {:.4f} | TEST MIOU {:.4f}".format(
                datetime.now().strftime("%H:%M:%S"), generation_idx, args.generations+1, epoch_idx, total_number_of_epochs, c_acc, m_iou))
            for c_idx in range(len(c_iou)):
                tensorboard_writer.add_scalar('test_{}'.format(mask_label_mapper[c_idx]), c_iou[c_idx], (epoch_idx + 1))
            tensorboard_writer.add_scalar('test_miou', m_iou, (epoch_idx + 1))
            tensorboard_writer.add_scalar('test_mpca', c_acc, (epoch_idx + 1))
                
        scheduler.step()
        
    tensorboard_writer.close()
    
    
    # Active Learning Loop - here, with softmax entropy
    
    if generation_idx < args.generations:
        print("Labeling - Active Learning - in process..")
        
        model.eval()
        ema.model.eval()
        
        train_ul_dataset_core.augmentation_flip = False
        train_ul_dataset_core.scale_size = (1.0, 1.0)
        
        print("Changed augmentation_flip: {} - Assert".format(train_ul_dataset_core.augmentation_flip))
        print("Changed scale_size: {} - Assert".format(train_ul_dataset_core.scale_size))
    
        train_ul_core_loader = torch.utils.data.DataLoader(dataset=train_ul_dataset_core,
                                                            batch_size=4,
                                                            shuffle=False,
                                                            num_workers=4
                                                            )
        
        test_dataset = iter(train_ul_core_loader)
        test_epoch = len(train_ul_core_loader)
        
        for idx in tqdm(range(test_epoch)):
        
            data_obj_v = test_dataset.next()
            imgs_v, segmaps_v = data_obj_v['img'].to(device), data_obj_v['segmap'].to(device)
            data_indxs = data_obj_v['index']
        
            with torch.no_grad():
                predmaps_v = model(imgs_v)
                
            predmaps_v_xl = F.interpolate(predmaps_v, size = segmaps_v.shape[1:], mode='bilinear', align_corners=True)
                    
            predlogits = softmax_fx(predmaps_v_xl)
            
            pred_ent = (-predlogits * torch.log(predlogits)).sum(dim=1)
            pred_ent[data_obj_v['segmap_mask'] > 0] = 0
            
            pred_avg = F.adaptive_avg_pool2d(pred_ent, output_size=adaptive_pool_op_size)
            
            for sub_idx in range(len(pred_avg)):
            
                v, k = torch.topk(pred_avg[sub_idx].flatten(),args.num_regions)
                indices = np.array(np.unravel_index(k.cpu().numpy(), pred_avg[sub_idx].shape)).T
        
                for len_indices in range(len(indices)):
                    x1 = indices[len_indices,1] * args.region_size
                    x2 = indices[len_indices,1] * args.region_size + args.region_size
                    y1 = indices[len_indices,0] * args.region_size
                    y2 = indices[len_indices,0] * args.region_size + args.region_size
                    train_ul_dataset_core.segmap_mask_array[data_indxs[sub_idx],y1:y2,x1:x2] = 1
                    
        print("Total number of pixels labeled: {}".format(train_ul_dataset_core.segmap_mask_array.sum()))
                        
        torch.cuda.empty_cache()
        
        train_ul_dataset_core.augmentation_flip = True
        train_ul_dataset_core.scale_size = data_loader_main.scale_size
        
        print("Changed augmentation_flip: {} - Assert".format(train_ul_dataset_core.augmentation_flip))
        print("Changed scale_size: {} - Assert".format(train_ul_dataset_core.scale_size))
    
    else:
        continue