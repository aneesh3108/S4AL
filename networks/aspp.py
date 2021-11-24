import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ASPPConv', 'ASPPPooling', 'ASPPModule']


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super(ASPPConv, self).__init__(*modules)
        
        
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    
    
class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPPModule, self).__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)
    
if __name__ == '__main__':
    model = ASPPModule(in_channels=320, atrous_rates=[6,12,18])
    model.eval()
    
    #Check CamVid
    print("--------CAMVID--------")
    input_tensor = torch.randn(1, 320, 23, 30)
    y = model(input_tensor)
    print("ASPP feat shapes: {}".format(y.shape))
    
    #Check CityScapes
    print("--------CityScapes--------")
    input_tensor = torch.randn(1, 320, 43, 43)
    y = model(input_tensor)
    print("ASPP feat shapes: {}".format(y.shape))