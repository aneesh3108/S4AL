from PIL import Image
import torch
import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f

from .dataset_utils import *
from .dataset_colormap_utils import *

# Credit: https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/transforms.py
try:
    from torchvision.transforms.functional import InterpolationMode
    map_nearest = InterpolationMode.NEAREST
    map_bilinear = InterpolationMode.BILINEAR
    map_bicubic = InterpolationMode.BICUBIC
except ImportError:
    map_nearest = Image.NEAREST
    map_bilinear = Image.BILINEAR
    map_bicubic = Image.BICUBIC


# -----------------------------------------------------------------------------
# Define data augmentation - image, label, and region-masks
# -----------------------------------------------------------------------------

def transform_image(image, label, label_mask=None, crop_size=(512, 512), scale_size=(0.8, 1.0),
                    augmentation_flip=False, augmentation_color=False, tensor_tx=False,
                    to_pil=True):
    
    if to_pil:
        image = transforms_f.to_pil_image(image)
        label = transforms_f.to_pil_image(label)
        if label_mask is not None:
            label_mask = transforms_f.to_tensor(label_mask)
    
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, map_bilinear)
    label = transforms_f.resize(label, resized_size, map_nearest)
    if label_mask is not None:
        label_mask=transforms_f.resize(label_mask, resized_size, map_nearest)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if label_mask is not None:
            label_mask=transforms_f.pad(label_mask, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label = transforms_f.crop(label, i, j, h, w)
    if label_mask is not None:
        label_mask = transforms_f.crop(label_mask, i, j, h, w)

    if augmentation_color:
        # Random color jitter
        if torch.rand(1) > 0.5:
            color_transform = transforms.ColorJitter(brightness=(0.75, 1.25), 
                                                      contrast=(0.75, 1.25))
            image = color_transform(image)
            
    if augmentation_flip:
        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if label_mask is not None:
                label_mask = transforms_f.hflip(label_mask)

    if tensor_tx:
        # Transform to tensor
        image = transforms_f.to_tensor(image)
        label = (transforms_f.to_tensor(label) * 255).long()
        label[label == 255] = -1  # invalid pixels are re-mapped to index -1

        # if label_mask is not None:
        #     label_mask = transforms_f.to_tensor(label_mask)

        # Apply (ImageNet) normalisation
        image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if label_mask is not None:
        return image, label, label_mask
    else:
        return image, label
    
# -----------------------------------------------------------------------------
# Define data augmentation - image, label, and logits
# -----------------------------------------------------------------------------

def transform_logits(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0),
                     augmentation_flip=False, augmentation_color=False, tensor_tx=False,
                     to_pil=True):
    
    if to_pil:
        image = transforms_f.to_pil_image(image)
        label = transforms_f.to_pil_image(label)
    
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, map_bilinear)
    label = transforms_f.resize(label, resized_size, map_nearest)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, map_nearest)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)
    
    if augmentation_color:
        # Random color jitter
        if torch.rand(1) > 0.5:
            color_transform = transforms.ColorJitter(brightness=(0.75, 1.25), 
                                                      contrast=(0.75, 1.25))
            image = color_transform(image)
            
    if augmentation_flip:
        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if logits is not None:
                logits = transforms_f.hflip(logits)

    if tensor_tx:
        # Transform to tensor
        image = transforms_f.to_tensor(image)
        label = (transforms_f.to_tensor(label) * 255).long()
        label[label == 255] = -1  # invalid pixels are re-mapped to index -1

        if logits is not None:
            logits = transforms_f.to_tensor(logits)
        
        # Apply (ImageNet) normalisation
        image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if logits is not None:
        return image, label, logits
    else:
        return image, label

# -----------------------------------------------------------------------------
# Apply transforms to a group of images
# TODO: make this GPU friendly
#       - converting to PIL everytime is a waste of full GPU utilization
# -----------------------------------------------------------------------------

def batch_transform(data, label, logits, crop_size, scale_size, apply_augmentation):
    data_list, label_list, logits_list = [], [], []
    device = data.device

    for k in range(data.shape[0]):
        data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
        aug_data, aug_label, aug_logits = transform_logits(data_pil, label_pil, logits_pil,
                                                           crop_size=crop_size,
                                                           scale_size=scale_size,
                                                           augmentation_flip=apply_augmentation,
                                                           augmentation_color=apply_augmentation,
                                                           tensor_tx=True,
                                                           to_pil=False)

        data_list.append(aug_data.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)

    data_trans, label_trans, logits_trans = \
        torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device)
    return data_trans, label_trans, logits_trans
