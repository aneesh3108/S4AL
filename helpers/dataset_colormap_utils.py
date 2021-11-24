import numpy as np

# -----------------------------------------------------------------------------
# General
# -----------------------------------------------------------------------------

def get_colored_output(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]
    return np.uint8(color_mask)

# -----------------------------------------------------------------------------
# CamVid
# -----------------------------------------------------------------------------

def get_camvid_label():
    mask_labels = {0: 'sky',
                   1: 'building',
                   2: 'column_pole',
                   3: 'road',
                   4: 'sidewalk',
                   5: 'tree',
                   6: 'sign',
                   7: 'fence',
                   8: 'car',
                   9: 'pedestrian',
                   10: 'byciclist'
                   }
    return mask_labels

def create_camvid_label_colormap():
  """Creates a label colormap used in CamVid segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 128, 128]
  colormap[1] = [128, 0, 0]
  colormap[2] = [192, 192, 128]
  colormap[3] = [128, 64, 128]
  colormap[4] = [0, 0, 192]
  colormap[5] = [128, 128, 0]
  colormap[6] = [192, 128, 128]
  colormap[7] = [64, 64, 128]
  colormap[8] = [64, 0, 128]
  colormap[9] = [64, 64, 0]
  colormap[10] = [0, 128, 192]
  colormap[-1] = [0, 0, 0]
  return colormap

# -----------------------------------------------------------------------------
# CityScapes
# -----------------------------------------------------------------------------

def get_cityscapes_label():
    mask_labels = {0: 'road',
                   1: 'sidewalk',
                   2: 'building',
                   3: 'wall',
                   4: 'fence',
                   5: 'pole',
                   6: 'traffic light',
                   7: 'traffic sign',
                   8: 'vegetation',
                   9: 'terrain',
                   10: 'sky',
                   11: 'pedestrian',
                   12: 'rider',
                   13: 'car',
                   14: 'truck',
                   15: 'bus',
                   16: 'train',
                   17: 'motorcycle',
                   18: 'bicycle'
                   }
    return mask_labels
    
def create_cityscapes_label_colormap():
  """Creates a label colormap used in CityScapes segmentation benchmark.
  Returns:
    A colormap for visualizing segmentation results.
  """
  colormap = np.zeros((256, 3), dtype=np.uint8)
  colormap[0] = [128, 64, 128]
  colormap[1] = [244, 35, 232]
  colormap[2] = [70, 70, 70]
  colormap[3] = [102, 102, 156]
  colormap[4] = [190, 153, 153]
  colormap[5] = [153, 153, 153]
  colormap[6] = [250, 170, 30]
  colormap[7] = [220, 220, 0]
  colormap[8] = [107, 142, 35]
  colormap[9] = [152, 251, 152]
  colormap[10] = [70, 130, 180]
  colormap[11] = [220, 20, 60]
  colormap[12] = [255, 0, 0]
  colormap[13] = [0, 0, 142]
  colormap[14] = [0, 0, 70]
  colormap[15] = [0, 60, 100]
  colormap[16] = [0, 80, 100]
  colormap[17] = [0, 0, 230]
  colormap[18] = [119, 11, 32]
  colormap[-1] = [0, 0, 0]
  return colormap

def cityscapes_class_map(mask):
    # source: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    mask_map = np.zeros_like(mask)
    mask_map[np.isin(mask, [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30])] = 255
    mask_map[np.isin(mask, [7])] = 0
    mask_map[np.isin(mask, [8])] = 1
    mask_map[np.isin(mask, [11])] = 2
    mask_map[np.isin(mask, [12])] = 3
    mask_map[np.isin(mask, [13])] = 4
    mask_map[np.isin(mask, [17])] = 5
    mask_map[np.isin(mask, [19])] = 6
    mask_map[np.isin(mask, [20])] = 7
    mask_map[np.isin(mask, [21])] = 8
    mask_map[np.isin(mask, [22])] = 9
    mask_map[np.isin(mask, [23])] = 10
    mask_map[np.isin(mask, [24])] = 11
    mask_map[np.isin(mask, [25])] = 12
    mask_map[np.isin(mask, [26])] = 13
    mask_map[np.isin(mask, [27])] = 14
    mask_map[np.isin(mask, [28])] = 15
    mask_map[np.isin(mask, [31])] = 16
    mask_map[np.isin(mask, [32])] = 17
    mask_map[np.isin(mask, [33])] = 18
    return mask_map

