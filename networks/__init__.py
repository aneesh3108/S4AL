from .mobilenetv2 import *
from .resnet import *
from .aspp import *
from .deeplabv3_mb import *
from .deeplabv3_resnet import *
from .drnseg import *

def get_deeplab(backbone='mbv2',
                num_classes=11,
                dilate_scale=16,
                pretrained_flag=True
                ):
    
    if backbone=='mbv2':
        model=DeepLabv3Plus_MB(network_mbv2=mobilenet_v2(pretrained=pretrained_flag),
                               num_classes=num_classes
                               )
    elif backbone=='r50' and dilate_scale==8:
        model=DeepLabv3Plus_RN(network_rn=resnet50_d8(pretrained=pretrained_flag),
                               num_classes=num_classes
                               )
    elif backbone=='r50' and dilate_scale==16:
        model=DeepLabv3Plus_RN(network_rn=resnet50_d16(pretrained=pretrained_flag),
                               num_classes=num_classes
                               )
    elif backbone =='drn':
        model=DRNSeg(classes=num_classes,pretrained=pretrained_flag)
    else:
        raise NotImplementedError("Backbone not supported at this time")
        
    return model

