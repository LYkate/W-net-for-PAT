import torch.nn.functional as F

from PATmodel_part import *

class PATModel(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super(PATModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.up1 = Up1(256, 128, bilinear)
        self.up2 = Up1(128, 64, bilinear)
        self.up3 = Up1(64, 32, bilinear)
        self.up4 = Up2(256, 128, bilinear)
        self.up5 = Up2(128, 64, bilinear)
        self.up6 = Up2(64, 32, bilinear)
        self.sc1 = Skipconnet(256, 256)
        self.sc2 = Skipconnet(128, 128)
        self.sc3 = Skipconnet(64, 64)
        self.sc4 = Skipconnet(32, 32)
        self.conv = SingleConv(32,out_channels)
        self.down4 = Down(32, 64)
        self.down5 = Down(64, 128)
        self.down6 = Down(128, 256)
        self.conv2 = SingleConv(64,out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.sc4(x1)
        x6 = self.sc3(x2)
        x7 = self.sc2(x3)
        x8 = self.sc1(x4)
        x = self.up1(x8, x7)
        x = self.up2(x, x6)
        x9 = self.up3(x, x5)
        x10 = self.down4(x9)
        x11 = self.down5(x10)
        x12 = self.down6(x11)
        x = self.up4(x12, x11)
        x = self.up5(x, x10)
        x = self.up6(x, x9)
        logits = self.conv(x)
        return logits

net = PATModel(in_channels=1, out_channels=1)
net = net.cuda()
