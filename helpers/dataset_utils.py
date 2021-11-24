import torch
import torch.nn.functional as F
import numpy as np
import os
import copy

import torchvision.transforms.functional as transforms_f

def denormalise(x, imagenet=True):
    if imagenet:
        x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2
    
def tensor_to_pil(im, label, logits):
    im = denormalise(im)
    im = transforms_f.to_pil_image(im.cpu())

    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())

    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
    return im, label, logits

# -----------------------------------------------------------------------------
# Define semi-supervised methods (based on data augmentation)
# -----------------------------------------------------------------------------

def generate_cutout_mask(img_size, ratio=2):
    cutout_area = img_size[0] * img_size[1] / ratio

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.float()

def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)  # all unique labels
    labels_select = labels[torch.randperm(len(labels))][:len(labels) // 2]  # randomly select half of labels

    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()

def generate_unsup_data(data, target, logits, mode='cutmix'):
    batch_size, _, im_h, im_w = data.shape
    device = data.device

    new_data = []
    new_target = []
    
    new_logits = []
    for i in range(batch_size):
        if mode == 'cutmix':
            mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
        if mode == 'classmix':
            mix_mask = generate_class_mask(target[i]).to(device)

        new_data.append((data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_target.append((target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

    new_data, new_target, new_logits = torch.cat(new_data), torch.cat(new_target), torch.cat(new_logits)
    return new_data, new_target.long(), new_logits

# -----------------------------------------------------------------------------
# Define semi-supervised methods - Adaptive ClassMix samplings
# -----------------------------------------------------------------------------

def generate_class_mask_biased(pseudo_labels, dataset):
    if dataset == 'camvid':
        labels_head = torch.tensor([0,1,3,4,5])
        labels_tail = torch.tensor([2,6,7,8,9,10])
        
        labels_select_head = labels_head[torch.randperm(len(labels_head))][:3]
        labels_select_tail = labels_tail[torch.randperm(len(labels_tail))][:2]

    elif dataset == 'cityscapes':
        labels_head = torch.tensor([0,1,2,8,10,13])
        labels_tail = torch.tensor([3,4,5,6,7,9,11,12,14,15,16,17,18])
   
        labels_select_head = labels_head[torch.randperm(len(labels_head))][:3]
        labels_select_tail = labels_tail[torch.randperm(len(labels_tail))][:3]

    labels_select = torch.cat((labels_select_head, labels_select_tail)).to(pseudo_labels.device)
    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()

def generate_unsup_data_biased(data, target, logits=None, mode='classmix', dataset ='camvid'):
    batch_size, _, im_h, im_w = data.shape
    device = data.device

    new_data = []
    new_target = []
    
    if logits is not None:
        new_logits = []
        
    for i in range(batch_size//2):
        mix_mask = generate_class_mask_biased(target[i + batch_size//2], dataset).to(device)

        new_data.append((data[i] * (1 - mix_mask) + data[(i + batch_size//2)] * mix_mask).unsqueeze(0))
        new_target.append((target[i] * (1 - mix_mask) + target[(i + batch_size//2)] * mix_mask).unsqueeze(0))
        
        if logits is not None:
            new_logits.append((logits[i] * (1 - mix_mask) + logits[(i + batch_size//2)] * mix_mask).unsqueeze(0))

    new_data, new_target = torch.cat(new_data), torch.cat(new_target)
    
    if logits is not None:
        new_logits = torch.cat(new_logits)
        
    if logits is not None:
        return new_data, new_target.long(), new_logits
    else:
        return new_data, new_target.long()