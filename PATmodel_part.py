import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=[1,3,5,7], reduction=16, group=1, L=32):
        super().__init__()
        self.out_channels = out_channels
        self.d = max(in_channels // reduction, L)
        self.convs = nn.ModuleList([])
        for i in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(in_channels, out_channels, kernel_size=i, padding=i // 2, groups=group)),
                    ('bn',nn.BatchNorm2d(out_channels)),
                    ('relu',nn.ReLU(inplace=True))
                ]))
            )
        self.fc = nn.Linear(out_channels, self.d)
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(
                nn.Linear(self.d, out_channels)
            )
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        batch_size, c, _, _ = x.size()
        conv_outs = []
        # 1.split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)
        # 2.fuse
        feats_U = sum(conv_outs)
        feats_S = feats_U.mean(-1).mean(-1)
        feats_Z = self.fc(feats_S)
        # 3.select
        weights = []
        for fc in self.fcs:
            weight = fc(feats_Z)
            weights.append(weight.view(batch_size, self.out_channels, 1, 1))
        attention_weights = torch.stack(weights, 0)
        attention_weights = self.softmax(attention_weights)
        # 4.fuse
        out=(attention_weights*feats).sum(0)
        return out
        
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            SKConv(in_channels, out_channels),
            SKConv(out_channels, out_channels),
        )

    def forward(self, x):
        return self.double_conv(x)
        
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

        self.in_channels = in_channels

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        #x2 = x2.narrow(1, self.in_channels // 4 + 1, self.in_channels // 2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up2(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Skipconnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Skipconnet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 10), stride=(1, 10), padding=(1,0)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SingleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

