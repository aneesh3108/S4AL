import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['DeepLabv3Plus_MB']

from .aspp import *
from .mobilenetv2 import *


class DeepLabv3Plus_MB(nn.Module):
    def __init__(self, network_mbv2, dilate_scale=16, num_classes=11):
        super(DeepLabv3Plus_MB, self).__init__()

        if dilate_scale == 16:
            aspp_atrous_rates = [6, 12, 18]
        else:
            raise NotImplementedError("only supporting rate of 16 right now")

        self.network = network_mbv2.features[:-1]

        self.aspp_dl = ASPPModule(
            in_channels=320, atrous_rates=aspp_atrous_rates)

        self.low_level_1x1 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=48,
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
        feat_low = self.network[0:4](imgs)
        feat_high = self.network[4:](feat_low)

        feat_aspp = self.aspp_dl(feat_high)

        feat_low = self.low_level_1x1(feat_low)

        feat_aspp_4x = F.interpolate(
            feat_aspp, size=feat_low.shape[2:], mode='bilinear', align_corners=True)

        prediction = self.deeplabhead_classifier(
            torch.cat([feat_aspp_4x, feat_low], dim=1))
        
        return prediction


if __name__ == "__main__":
    mbv2 = mobilenet_v2(pretrained=True)
    model = DeepLabv3Plus_MB(network_mbv2=mbv2)
    model.eval()

    # Check CamVid
    print("--------CAMVID--------")
    input_tensor = torch.randn(1, 3, 360, 480)
    prediction = model(input_tensor)
    print("Pred Shape before interp: {}".format(prediction.shape))
    prediction = F.interpolate(
        prediction, size=input_tensor.shape[2:], mode='bilinear', align_corners=True)
    print("Pred Shape before interp: {}".format(prediction.shape))