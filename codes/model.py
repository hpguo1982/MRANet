import pvtv2
from resnet import resnet50
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st
from utils import cus_sample

class Conv2D(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1, dilation=1, bias=True, act=True):
        super().__init__()

        self.act = act
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.act == True:
            x = self.relu(x)
        return x

class  ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_c)
        )

    def forward(self, inputs):
        x1 = self.conv(inputs)
        x2 = self.shortcut(inputs)
        x = self.relu(x1 + x2)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, scale):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c, out_c)

    def forward(self, inputs):
        x = self.up(inputs)
        x = self.r1(x)
        return x

class RSFM(nn.Module):
    def __init__(self, in_c, out_c, is_up=True):
        super().__init__()
        self.is_up = is_up
        if self.is_up:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.r1 = ResidualBlock(in_c+out_c, out_c)

    def forward(self, x, s):
        if self.is_up:
            x = self.up(x)
        x = torch.cat([x, s], axis=1)
        x = self.r1(x)
        return x


class SpatialAttention(nn.Module):  #SA
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)

class ChannelAttention(nn.Module):   #CA
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class RSAB(nn.Module):
    def __init__(self, channels, padding=0, groups=1, matmul_norm=True):
        super(RSAB, self).__init__()
        self.channels = channels
        self.padding = padding
        self.groups = groups
        self.matmul_norm = matmul_norm
        self._channels = channels//8

        self.conv_query = nn.Conv2d(in_channels=channels, out_channels=self._channels, kernel_size=1, groups=groups)
        self.conv_key = nn.Conv2d(in_channels=channels, out_channels=self._channels, kernel_size=1, groups=groups)
        self.conv_value = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1, groups=groups)

        self.conv_output = Conv2D(in_c=channels, out_c=channels, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Get query, key, value tensors
        query = self.conv_query(x).view(batch_size, -1, height*width)
        key = self.conv_key(x).view(batch_size, -1, height*width)
        value = self.conv_value(x).view(batch_size, -1, height*width)

        # Apply transpose to swap dimensions for matrix multiplication
        query = query.permute(0, 2, 1).contiguous()  # (batch_size, height*width, channels//8)
        value = value.permute(0, 2, 1).contiguous()  # (batch_size, height*width, channels)

        # Compute attention map
        attention_map = torch.matmul(query, key)
        if self.matmul_norm:
            attention_map = (self._channels**-.5) * attention_map
        attention_map = torch.softmax(attention_map, dim=-1)

        # Apply attention
        out = torch.matmul(attention_map, value)
        out = out.permute(0, 2, 1).contiguous().view(batch_size, channels, height, width)

        # Apply output convolution
        out = self.conv_output(out)
        out = out + x

        return out


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CMKD(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.c1 = Conv2D(in_c, out_c, kernel_size=1, padding=0)
        self.c2 = Conv2D(in_c, out_c, kernel_size=3, padding=1)
        self.c3 = Conv2D(in_c, out_c, kernel_size=7, padding=3)
        self.c4 = Conv2D(in_c, out_c, kernel_size=11, padding=5)
        self.s1 = Conv2D(out_c*4, out_c, kernel_size=1, padding=0)

        self.d1 = Conv2D(out_c, out_c, kernel_size=3, padding=1, dilation=1)
        self.d2 = Conv2D(out_c, out_c, kernel_size=3, padding=3, dilation=3)
        self.d3 = Conv2D(out_c, out_c, kernel_size=3, padding=7, dilation=7)
        self.d4 = Conv2D(out_c, out_c, kernel_size=3, padding=11, dilation=11)
        self.s2 = Conv2D(out_c*4, out_c, kernel_size=1, padding=0, act=False)
        self.s3 = Conv2D(in_c, out_c, kernel_size=1, padding=0, act=False)

        self.ca = ChannelAttention(out_c)
        self.sa = SpatialAttention()

    def forward(self, x):
        x0 = x
        x1 = self.c1(x)
        x2 = self.c2(x)
        x3 = self.c3(x)
        x4 = self.c4(x)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.s1(x)

        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        x4 = self.d4(x)
        x = torch.cat([x1, x2, x3, x4], axis=1)
        x = self.s2(x)
        s = self.c3(x0)

        x = self.relu(x+s)
        x = x * self.ca(x)
        x = x * self.sa(x)

        return x


class ASIM(nn.Module):
    def __init__(self, left_channel):
        super(ASIM, self).__init__()
        self.conv1 = BasicConv2d(left_channel, left_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(left_channel, left_channel, kernel_size=3, stride=1, padding=1)
        self.conv_cat = BasicConv2d(2*left_channel, left_channel, kernel_size=3, stride=1, padding=1)
        self.conv_cat_2 = BasicConv2d(3*left_channel, left_channel, kernel_size=3, stride=1, padding=1)
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample = cus_sample

    def forward(self, left, right):
        if right.shape != left.shape:
            right = self.upsample(right, scale_factor=2)  # right 上采样
        else:
            right = right

        right = self.conv1(right)
        out = self.conv_cat(torch.cat((left, right), 1))
        out_g = self.conv2(out)
        out_g = self.avg_pool(out_g)
        out_g = self.sigmoid(out_g)
        right = self.conv2(right) * out_g
        left = self.conv2(left) * out_g
        out = self.conv_cat_2(torch.cat((left, right, out), 1))
        out_g2 = self.sigmoid(self.conv2(out - self.avg_pool(out)))
        out = self.conv2(out) * out_g2

        return out

class MRANet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.backbone = pvtv2.pvt_v2_b3()  ## [64, 128, 320, 512]
        path = 'pvt_v2_b3.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        """ Convolutions Multiple Kernel Dilation +SAB"""
        self.cmkd1 = CMKD(64, 64)
        self.cmkd2 = CMKD(128, 128)
        self.cmkd3 = CMKD(320, 320)
        self.rsab = RSAB(512)

        """ Channel Reduction """
        self.c1 = Conv2D(64, 64, kernel_size=1, padding=0)
        self.c2 = Conv2D(128, 64, kernel_size=1, padding=0)
        self.c3 = Conv2D(320, 64, kernel_size=1, padding=0)
        self.c4 = Conv2D(512, 64, kernel_size=1, padding=0)

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.d1 = ASIM(64)
        self.d2 = ASIM(64)
        self.d3 = RSFM(64, 64)
        self.d4 = UpBlock(64, 64, 4)

        self.r1 = ResidualBlock(64, 64)
        self.y = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        pvt1 = self.backbone(inputs)
        e1 = pvt1[0]  ## [-1, 64, h/4, w/4]
        e2 = pvt1[1]  ## [-1, 128, h/8, w/8]
        e3 = pvt1[2]  ## [-1, 320, h/16, w/16]
        e4 = pvt1[3]  ## [-1, 512, h/32, w/32]

        ae1 = self.cmkd1(e1)
        ae2 = self.cmkd2(e2)
        ae3 = self.cmkd3(e3)
        ae4 = self.rsab(e4)

        c1 = self.c1(ae1)
        c2 = self.c2(ae2)
        c3 = self.c3(ae3)
        c4 = self.c4(ae4)

        c41 = self.up(c4)
        d1 = self.d1(c41, c3)

        d11 = self.up(d1)
        d2 = self.d2(d11, c2)
        d3 = self.d3(d2, c1)

        x = self.d4(d3)
        x = self.r1(x)
        y = self.y(x)

        return y
if __name__ == "__main__":
    x = torch.randn((4, 3, 256, 256))
    model = MRANet()
    y = model(x)
    print(y.shape)