import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
from tabulate import tabulate

from torch.optim.lr_scheduler import _LRScheduler
from sklearn.metrics import confusion_matrix
from .dataset_colormap_utils import get_camvid_label, get_cityscapes_label

# -----------------------------------------------------------------------------
# Ref: https://github.com/NVIDIA/semantic-segmentation
# -----------------------------------------------------------------------------

def fast_hist(pred, gtruth, num_classes):
    # mask indicates pixels we care about
    mask = (gtruth >= 0) & (gtruth < num_classes)

    # stretch ground truth labels by num_classes
    #   class 0  -> 0
    #   class 1  -> 19
    #   class 18 -> 342
    #
    # TP at 0 + 0, 1 + 1, 2 + 2 ...
    #
    # TP exist where value == num_classes*class_id + class_id
    # FP = row[class].sum() - TP
    # FN = col[class].sum() - TP
    hist = np.bincount(num_classes * gtruth[mask].astype(int) + pred[mask],
                       minlength=num_classes ** 2)
    hist = hist.reshape(num_classes, num_classes)
    return hist

# -----------------------------------------------------------------------------
# Ref: https://github.com/NVIDIA/semantic-segmentation
# -----------------------------------------------------------------------------

def calculate_iou(hist_data):
    # acc = np.diag(hist_data).sum() / hist_data.sum()
    acc_cls = np.diag(hist_data) / (hist_data.sum(axis=1) + 1e-10)
    acc_cls = np.nanmean(acc_cls)
    divisor = hist_data.sum(axis=1) + hist_data.sum(axis=0) - np.diag(hist_data)
    iou = np.diag(hist_data) / (divisor + 1e-10)
    return iou, acc_cls

# -----------------------------------------------------------------------------
# Ref: https://github.com/NVIDIA/semantic-segmentation
# (Slightly modified)
# -----------------------------------------------------------------------------

def get_stats(hist, iu, dataset):
    
    iu_FP = hist.sum(axis=1) - np.diag(hist)
    iu_FN = hist.sum(axis=0) - np.diag(hist)
    iu_TP = np.diag(hist)
    
    if dataset == 'camvid':
        id2cat = get_camvid_label()
    else:
        id2cat = get_cityscapes_label()
        
    tabulate_data = []
    
    header = ['Id', 'label', 'iU']
    header.extend(['Precision', 'Recall'])
    
    for class_id in range(len(iu)):
        class_data = []
        class_data.append(class_id)
        class_name = "{}".format(id2cat[class_id]) if class_id in id2cat else ''
        class_data.append(class_name)
        class_data.append(iu[class_id] * 100)
    
        # total_pixels = hist.sum()
        class_data.append((iu_TP[class_id] / (iu_TP[class_id] + iu_FP[class_id])) * 100)
        class_data.append((iu_TP[class_id] / (iu_TP[class_id] + iu_FN[class_id])) * 100)
        tabulate_data.append(class_data)
        
    print_str = str(tabulate((tabulate_data), headers=header, floatfmt='1.2f'))
    print(print_str)
    print("mIoU = {:.2f}".format(np.nanmean(iu) * 100))