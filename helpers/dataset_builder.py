import os
import os.path as osp

from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import torch

from tqdm import tqdm

import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

from .dataset_utils import *
from .dataset_colormap_utils import *
from .transform_utils import *

# -----------------------------------------------------------------------------
# Create dataset in PyTorch format - Supervised
# -----------------------------------------------------------------------------

class BuildDataset(Dataset):
    def __init__(self, root, dataset, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0),
                  augmentation_flip=False, augmentation_color=False, tensor_tx=False,
                  trainval=True,use_custom_idx=False):
        self.root = root
        self.dataset = dataset
        
        if use_custom_idx:
            self.idx_list = idx_list
            idx_list = 'custom'
        else:
            idx_list_filename = osp.join(self.root, self.dataset, 'imagesets', idx_list + '.txt')
            with open(idx_list_filename, 'r') as txtfile:
                self.idx_list = txtfile.read().splitlines()
        
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.augmentation_flip = augmentation_flip
        self.augmentation_color = augmentation_color
        self.tensor_tx = tensor_tx
        self.trainval = trainval
        
        self.image_array = []
        self.segmap_array = []
        self.image_name = []
        
        if self.dataset == 'CamVid':
            self.init_dataset_camvid()
        elif self.dataset == 'CityScapes':
            self.init_dataset_cityscapes()
            
        self.image_array = np.stack(self.image_array)
        self.segmap_array = np.stack(self.segmap_array)
            
        print("Dataset: {}, Set: {}, Length: {} is initialized.".format(self.dataset, idx_list, len(self.image_array)))
        
    def init_dataset_camvid(self):
        for index in tqdm(range(len(self.idx_list))):
            image = Image.open(osp.join(self.root, self.dataset, 'images', '{}.png'.format(self.idx_list[index])))
            segmap = Image.open(osp.join(self.root, self.dataset, 'annots', '{}.png'.format(self.idx_list[index])))
            
            image = np.asarray(image)
            segmap = np.asarray(segmap)
            segmap = np.where(segmap == 11, 255, segmap)
            
            self.image_array.append(image)
            self.segmap_array.append(segmap)
            self.image_name.append('{}'.format(self.idx_list[index]))
            
    def init_dataset_cityscapes(self):
        if self.trainval:
            folder_loc = 'train'
        else:
            folder_loc = 'val'
            
        for index in tqdm(range(len(self.idx_list))):
            image = Image.open(osp.join(self.root, self.dataset, 'images', folder_loc, '{}.png'.format(self.idx_list[index])))
            segmap = Image.open(osp.join(self.root, self.dataset, 'annots', folder_loc, '{}.png'.format(self.idx_list[index])))
            
            segmap = Image.fromarray(cityscapes_class_map(np.array(segmap)))
            
            image = np.asarray(image)
            segmap = np.asarray(segmap)
            
            self.image_array.append(image)
            self.segmap_array.append(segmap)
            self.image_name.append('{}'.format(self.idx_list[index]))
            
    def __getitem__(self, index):
        
        image = self.image_array[index]
        segmap = self.segmap_array[index]
        
        image, segmap = transform_image(image,
                                        segmap,
                                        None,
                                        self.crop_size,
                                        self.scale_size,
                                        self.augmentation_flip,
                                        self.augmentation_color,
                                        self.tensor_tx)
        
        
        if self.tensor_tx:
            return {'img': image,
                    'segmap': segmap.squeeze(0),
                    'name': self.image_name[index],
                    'index': index}
        else:
            return {'img': image,
                    'segmag': segmap,
                    'name': self.image_name[index],
                    'index': index}
    
    def __len__(self):
        return len(self.idx_list)
    
# -----------------------------------------------------------------------------
# Create dataset in PyTorch format - with a dedicated array for region masks
# -----------------------------------------------------------------------------

class BuildDatasetAL(Dataset):
    def __init__(self, root, dataset, idx_list, crop_size=(512, 512), scale_size=(0.5, 2.0),
                  augmentation_flip=False, augmentation_color=False, tensor_tx=False,
                  trainval=True, use_custom_idx=False):
        self.root = root
        self.dataset = dataset
        
        if use_custom_idx:
            self.idx_list = idx_list
            idx_list = 'custom'
        else:
            idx_list_filename = osp.join(self.root, self.dataset, 'imagesets', idx_list + '.txt')
            with open(idx_list_filename, 'r') as txtfile:
                self.idx_list = txtfile.read().splitlines()
        
        self.crop_size = crop_size
        self.scale_size = scale_size
        self.augmentation_flip = augmentation_flip
        self.augmentation_color = augmentation_color
        self.tensor_tx = tensor_tx
        self.trainval = trainval
        
        self.image_array = []
        self.segmap_array = []
        self.image_name = []
        
        self.segmap_mask_array = []
        
        if self.dataset == 'CamVid':
            self.init_dataset_camvid()
        elif self.dataset == 'CityScapes':
            self.init_dataset_cityscapes()
            
        self.image_array = np.stack(self.image_array)
        self.segmap_array = np.stack(self.segmap_array)
        self.segmap_mask_array = np.stack(self.segmap_mask_array)
            
        print("Dataset: {}, Set: {}, Length: {} is initialized.".format(self.dataset, idx_list, len(self.image_array)))
        
    def init_dataset_camvid(self):
        for index in tqdm(range(len(self.idx_list))):
            image = Image.open(osp.join(self.root, self.dataset, 'images', '{}.png'.format(self.idx_list[index])))
            segmap = Image.open(osp.join(self.root, self.dataset, 'annots', '{}.png'.format(self.idx_list[index])))
            
            image = np.asarray(image)
            segmap = np.asarray(segmap)
            segmap = np.where(segmap == 11, 255, segmap)
            segmap_mask = np.zeros(segmap.shape, dtype=bool)

            self.image_array.append(image)
            self.segmap_array.append(segmap)
            self.image_name.append('{}'.format(self.idx_list[index]))
            self.segmap_mask_array.append(segmap_mask)
            
    def init_dataset_cityscapes(self):
        if self.trainval:
            folder_loc = 'train'
        else:
            folder_loc = 'val'
            
        for index in tqdm(range(len(self.idx_list))):
            image = Image.open(osp.join(self.root, self.dataset, 'images', folder_loc, '{}.png'.format(self.idx_list[index])))
            segmap = Image.open(osp.join(self.root, self.dataset, 'annots', folder_loc, '{}.png'.format(self.idx_list[index])))
            
            segmap = Image.fromarray(cityscapes_class_map(np.array(segmap)))
            
            image = np.asarray(image)
            segmap = np.asarray(segmap)
            segmap_mask = np.zeros(segmap.shape, dtype=bool)
            
            self.image_array.append(image)
            self.segmap_array.append(segmap)
            self.image_name.append('{}'.format(self.idx_list[index]))
            self.segmap_mask_array.append(segmap_mask)
            
    def __getitem__(self, index):
        
        image = self.image_array[index]
        segmap = self.segmap_array[index]
        segmap_mask = self.segmap_mask_array[index]
            
        image, segmap, segmap_mask = transform_image(image,
                                                     segmap,
                                                     segmap_mask,
                                                     self.crop_size,
                                                     self.scale_size,
                                                     self.augmentation_flip,
                                                     self.augmentation_color,
                                                     self.tensor_tx)

        
        if self.tensor_tx:
            return {'img': image,
                    'segmap': segmap.squeeze(0),
                    'segmap_mask': segmap_mask.squeeze(0),
                    'name': self.image_name[index],
                    'index': index}
        else:
            return {'img': image,
                    'segmag': segmap,
                    'segmap_mask': segmap_mask,
                    'name': self.image_name[index],
                    'index': index}
    
    def __len__(self):
        return len(self.idx_list)

# -----------------------------------------------------------------------------
# Create data loader in PyTorch format
# -----------------------------------------------------------------------------

class BuildDataLoader:
    def __init__(self, dataset):
        
        if dataset == 'camvid':
            self.root = 'datasets'
            self.dataset = 'CamVid'
            self.im_size = [360, 480]
            self.crop_size = [360, 480]
            self.scale_size = (0.75, 1.25)
            self.batch_size = 4
            self.batch_iters = 100
            self.classes = 11
            
        elif dataset == 'cityscapes':
            self.root = 'datasets'
            self.dataset = 'CityScapes'
            self.im_size = [688, 688]
            self.crop_size = [688, 688]
            self.scale_size = (0.5, 2.0)
            self.batch_size = 4
            self.batch_iters = 200
            self.classes = 19

    def build_supervised(self):
        train_l_dataset = BuildDataset(root=self.root,
                                       dataset=self.dataset,
                                       idx_list='train',
                                       crop_size=self.crop_size,
                                       scale_size=self.scale_size,
                                       augmentation_flip=True,
                                       augmentation_color=True,
                                       tensor_tx=True,
                                       trainval=True
                                       )
        
        val_dataset = BuildDataset(root=self.root,
                                   dataset=self.dataset,
                                   idx_list='val',
                                   crop_size=self.im_size,
                                   scale_size=(1.0, 1.0),
                                   augmentation_flip=False,
                                   augmentation_color=False,
                                   tensor_tx=True,
                                   trainval=True
                                   )
        
        test_dataset = BuildDataset(root=self.root,
                                    dataset=self.dataset,
                                    idx_list='test',
                                    crop_size=self.im_size,
                                    scale_size=(1.0, 1.0),
                                    augmentation_flip=False,
                                    augmentation_color=False,
                                    tensor_tx=True,
                                    trainval=False
                                    )
        
        num_samples = self.batch_size * self.batch_iters

        train_l_loader = torch.utils.data.DataLoader(
            dataset=train_l_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset,
                                          replacement=True,
                                          num_samples=num_samples),
            drop_last=True,
            num_workers=4
            )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4
            )
        
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4
            )
        
        return train_l_loader, val_loader, test_loader
    
    def build_supervised_al(self):
        train_l_dataset = BuildDataset(root=self.root,
                                       dataset=self.dataset,
                                       idx_list='train_al_lab',
                                       crop_size=self.crop_size,
                                       scale_size=self.scale_size,
                                       augmentation_flip=True,
                                       augmentation_color=True,
                                       tensor_tx=True,
                                       trainval=True
                                       )
        
        val_dataset = BuildDataset(root=self.root,
                                   dataset=self.dataset,
                                   idx_list='val',
                                   crop_size=self.im_size,
                                   scale_size=(1.0, 1.0),
                                   augmentation_flip=False,
                                   augmentation_color=False,
                                   tensor_tx=True,
                                   trainval=True
                                   )
        
        test_dataset = BuildDataset(root=self.root,
                                    dataset=self.dataset,
                                    idx_list='test',
                                    crop_size=self.im_size,
                                    scale_size=(1.0, 1.0),
                                    augmentation_flip=False,
                                    augmentation_color=False,
                                    tensor_tx=True,
                                    trainval=False
                                    )
        
        num_samples = self.batch_size * self.batch_iters//2

        train_l_loader = torch.utils.data.DataLoader(
            dataset=train_l_dataset,
            batch_size=self.batch_size,
            sampler=sampler.RandomSampler(data_source=train_l_dataset,
                                          replacement=True,
                                          num_samples=num_samples),
            drop_last=True,
            num_workers=4
            )

        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4
            )
        
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4
            )
        
        return train_l_loader, val_loader, test_loader
    
    def build_testloader(self):
        test_dataset = BuildDataset(root=self.root,
                                    dataset=self.dataset,
                                    idx_list='test',
                                    crop_size=self.im_size,
                                    scale_size=(1.0, 1.0),
                                    augmentation_flip=False,
                                    augmentation_color=False,
                                    tensor_tx=True,
                                    trainval=False
                                    )
        
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=4
            )
        
        return test_loader