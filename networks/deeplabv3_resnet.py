import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DeepLabv3Plus_RN']

from .aspp import *
from .resnet import *


class DeepLabv3Plus_RN(nn.Module):
    def __init__(self, network_rn, num_classes=11, dilate_scale=16):
        super(DeepLabv3Plus_RN, self).__init__()

        if dilate_scale == 16:
            aspp_atrous_rates = [6, 12, 18]
        elif dilate_scale == 8:
            aspp_atrous_rates = [12,24,36]
        else:
            raise NotImplementedError("only supporting rate of 8,16 right now")

        self.resnet_conv1 = network_rn.conv1
        self.resnet_bn1 = network_rn.bn1
        self.resnet_relu = network_rn.relu
        self.resnet_maxpool = network_rn.maxpool

        self.resnet_layer1 = network_rn.layer1
        self.resnet_layer2 = network_rn.layer2
        self.resnet_layer3 = network_rn.layer3
        self.resnet_layer4 = network_rn.layer4

        self.aspp_dl = ASPPModule(
            in_channels=2048, atrous_rates=aspp_atrous_rates)

        self.low_level_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=48,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        self.deeplabhead_classifier = nn.Sequential(
            nn.Conv2d(in_channels=(256+48), out_channels=256,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
        )
        
    def forward(self, imgs):
        x = self.resnet_conv1(imgs)
        x = self.resnet_bn1(x)
        x = self.resnet_relu(x)
        x = self.resnet_maxpool(x)

        feat_low = self.resnet_layer1(x)
        x = self.resnet_layer2(feat_low)
        x = self.resnet_layer3(x)
        feat_high = self.resnet_layer4(x)

        feat_aspp = self.aspp_dl(feat_high)

        feat_low = self.low_level_1x1(feat_low)

        feat_aspp_4x = F.interpolate(
            feat_aspp, size=feat_low.shape[2:], mode='bilinear', align_corners=True)

        prediction = self.deeplabhead_classifier(
            torch.cat([feat_aspp_4x, feat_low], dim=1))
        
        return prediction


if __name__ == "__main__":
    r50 = resnet50(pretrained=True)
    model = DeepLabv3Plus_RN(network_rn=r50)
    model.eval()

    # Check CamVid
    print("--------CAMVID--------")
    input_tensor = torch.randn(1, 3, 360, 480)
    prediction = model(input_tensor)
    print("Pred Shape before interp: {}".format(prediction.shape))
    prediction = F.interpolate(
        prediction, size=input_tensor.shape[2:], mode='bilinear', align_corners=True)
    print("Pred Shape before interp: {}".format(prediction.shape))