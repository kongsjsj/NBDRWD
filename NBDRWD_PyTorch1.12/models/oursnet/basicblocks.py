# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from models.commonblocks import (conv, sequential, DownSampleWithDWT, UpSampleWithInvDWT, ResConvBlock)


class DnCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64

        layers = list()

        # First layer for Conv+ReLU
        layers.append(nn.Conv2d(in_channels=in_channels,
                                out_channels=features,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=False))

        layers.append(nn.ReLU(inplace=True))

        # Hidden layer for Conv+BN+ReLU
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features,
                                    out_channels=features,
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    bias=False))

            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))

        # Finally layer for Conv
        layers.append(nn.Conv2d(in_channels=features,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                padding=padding,
                                bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


class ResWUNet(nn.Module):
    """
    Implement Residual U-Net with dwt
    """

    def __init__(self, in_channels=64, out_channels=64, n=1):
        """
        :param in_channels:  Input channels
        :param out_channels: Output channels
        :param n: the number of residual block
        """
        super().__init__()
        #
        # --------------------------------------------------
        # first layer(from left to right and top to bottom)
        # --------------------------------------------------
        #
        self.conv_first = conv(in_channels, 64, kernel_size=3, padding=1, mode='C')
        # res block1 (Conv + ReLU + Conv + ReLU)
        self.res_blk1_1 = sequential(*[ResConvBlock(in_channels=64, out_channels=64) for _ in range(n)])
        # down 1
        self.down1_1 = DownSampleWithDWT()
        # conv1 1x1, 128 ---> 16
        self.conv1x1_1_1 = conv(64 * 2, 16, kernel_size=1, padding=0, mode='CR')
        # down 2
        self.down1_2 = DownSampleWithDWT()
        # conv1 3x3, 16 ---> 64
        self.conv16_64_1_1 = conv(16, 64, mode='CR')
        # res block2
        self.res_blk1_2 = sequential(*[ResConvBlock(in_channels=64, out_channels=64) for _ in range(n)])
        # conv2 1x1, 128 ---> 16
        self.conv1x1_1_2 = conv(64 * 2, 16, kernel_size=1, padding=0, mode='CR')
        # down 3
        self.down1_3 = DownSampleWithDWT()
        # conv2 3x3, 16 ---> 64
        self.conv16_64_1_2 = conv(16, 64, mode='CR')
        # res block3
        self.res_blk1_3 = sequential(*[ResConvBlock(in_channels=64, out_channels=64) for _ in range(n)])
        # conv3 1x1, 128 ---> 64
        self.conv1x1_1_3 = conv(64 * 2, 64, kernel_size=1, padding=0, mode='CR')
        # res block4
        self.res_blk1_4 = sequential(*[ResConvBlock(in_channels=64, out_channels=64) for _ in range(n)])
        # last conv for reconstruct
        self.conv_last = conv(64, out_channels, kernel_size=3, padding=1, mode='C')

        #
        # --------------------------------------------------
        # second layer(from left to right and top to bottom)
        # --------------------------------------------------
        #
        # conv1 1x1, 256 ---> 64
        self.conv256_64_2_1 = conv(256, 64, kernel_size=1, padding=0, mode='CR')
        # res block1
        self.res_blk2_1 = sequential(*[ResConvBlock(in_channels=64, out_channels=64) for _ in range(n)])
        # down 1
        self.down2_1 = DownSampleWithDWT()
        # conv2 3x3, 64 ---> 256
        self.conv64_256_2_2 = conv(64, 256, mode='CR')
        # up 1
        self.up2_1 = UpSampleWithInvDWT()
        # conv1 1x1, 256 ---> 64
        self.conv1x1_2_1 = conv(256, 64, kernel_size=1, padding=0, mode='CR')
        # conv2 1x1, 64*3 ---> 16
        self.conv1x1_2_2 = conv(64 * 3, 16, kernel_size=1, padding=0, mode='CR')
        # down 2
        self.down2_2 = DownSampleWithDWT()
        # conv3 3x3, 16 ---> 64
        self.conv16_64_2_3 = conv(16, 64, mode='CR')
        # res block2
        self.res_blk2_2 = sequential(*[ResConvBlock(in_channels=64, out_channels=64) for _ in range(n)])
        # conv4 3x3, 64 ---> 256
        self.conv64_256_2_4 = conv(64, 256, mode='CR')
        # up 2
        self.up2_2 = UpSampleWithInvDWT()
        # conv3 1x1, 256 ---> 64
        self.conv1x1_2_3 = conv(256, 64, kernel_size=1, padding=0, mode='CR')
        # conv4 1x1, 64*3 ---> 64
        self.conv1x1_2_4 = conv(64 * 3, 64, kernel_size=1, padding=0, mode='CR')
        # res block3
        self.res_blk2_3 = sequential(*[ResConvBlock(in_channels=64, out_channels=64) for _ in range(n)])
        # conv5 3x3, 64 ---> 256
        self.conv64_256_2_5 = conv(64, 256, mode='CR')
        # up 3
        self.up2_3 = UpSampleWithInvDWT()

        #
        # --------------------------------------------------
        # third layer(from left to right and top to bottom)
        # --------------------------------------------------
        #
        # conv1 1x1, 256 ---> 64
        self.conv256_64_3_1 = conv(256, 64, kernel_size=1, padding=0, mode='CR')
        # res block1
        self.res_blk3_1 = sequential(*[ResConvBlock(in_channels=64, out_channels=64) for _ in range(n)])
        # down 1
        self.down3_1 = DownSampleWithDWT()
        # conv2 3x3, 64 ---> 256
        self.conv64_256_3_2 = conv(64, 256, mode='CR')
        # up 1
        self.up3_1 = UpSampleWithInvDWT()
        # conv1 1x1, 256 ---> 64
        self.conv1x1_3_1 = conv(256, 64, kernel_size=1, padding=0, mode='CR')
        # conv2 1x1, 64*3 ---> 64
        self.conv1x1_3_2 = conv(64 * 3, 64, kernel_size=1, padding=0, mode='CR')
        # res block2
        self.res_blk3_2 = sequential(*[ResConvBlock(in_channels=64, out_channels=64) for _ in range(n)])
        # conv3 3x3, 64 ---> 256
        self.conv64_256_3_3 = conv(64, 256, mode='CR')
        # up 2
        self.up3_2 = UpSampleWithInvDWT()

        #
        # --------------------------------------------------
        # fourth layer(from left to right and top to bottom)
        # --------------------------------------------------
        #
        # conv1 1x1, 256 ---> 64
        self.conv256_64_4_1 = conv(256, 64, kernel_size=1, padding=0, mode='CR')
        self.res_blk4_1 = sequential(*[ResConvBlock(in_channels=64, out_channels=64) for _ in range(n)])
        # conv1 3x3, 64 ---> 256
        self.conv64_256_4_1 = conv(64, 256, mode='CR')
        # up 1
        self.up4_1 = UpSampleWithInvDWT()

    def forward(self, inputs):
        h, w = inputs.size()[-2:]
        padding_bot = int(np.ceil(h / 8) * 8 - h)
        padding_right = int(np.ceil(w / 8) * 8 - w)
        x = nn.ReplicationPad2d((0, padding_right, 0, padding_bot))(inputs)
        # main forward step
        # Notation: xi is the ith layer's features
        x = self.conv_first(x)
        x1 = self.res_blk1_1(x)

        x2 = self.down1_1(x1)
        x2 = self.conv256_64_2_1(x2)
        x2 = self.res_blk2_1(x2)

        x3 = self.down2_1(x2)
        x2 = self.conv64_256_2_2(x2)
        x1_temp = self.up2_1(x2)
        x2 = self.conv1x1_2_1(x2)

        x1 = torch.cat((x1, x1_temp), dim=1)
        x1 = self.conv1x1_1_1(x1)
        x2_temp_down = self.down1_2(x1)
        x1 = self.conv16_64_1_1(x1)
        x1 = self.res_blk1_2(x1)

        x3 = self.conv256_64_3_1(x3)
        x3 = self.res_blk3_1(x3)

        x4 = self.down3_1(x3)
        x4 = self.conv256_64_4_1(x4)
        x4 = self.res_blk4_1(x4)
        x4 = self.conv64_256_4_1(x4)
        x3_temp_up = self.up4_1(x4)

        x3 = self.conv64_256_3_2(x3)
        x2_temp_up = self.up3_1(x3)
        x3 = self.conv1x1_3_1(x3)  # N * 64 * h/4 * w/4

        x2 = torch.cat((x2, x2_temp_down, x2_temp_up), dim=1)
        x2 = self.conv1x1_2_2(x2)  # N * 16 * h/2 * w/2
        x3_temp_down = self.down2_2(x2)
        x2 = self.conv16_64_2_3(x2)
        x2 = self.res_blk2_2(x2)
        x2 = self.conv64_256_2_4(x2)  # N * 256 * h/2 * w/2
        x1_temp = self.up2_2(x2)
        x2 = self.conv1x1_2_3(x2)  # N * 64 * h/2 * w/2

        x3 = torch.cat((x3, x3_temp_down, x3_temp_up), dim=1)
        x3 = self.conv1x1_3_2(x3)  # N * 64 * h/4 * w/4
        x3 = self.res_blk3_2(x3)
        x3 = self.conv64_256_3_3(x3)  # N * 256 * h/4 * w/4
        x2_temp_up = self.up3_2(x3)

        x1 = torch.cat((x1, x1_temp), dim=1)
        x1 = self.conv1x1_1_2(x1)  # N * 16 * h * w
        x2_temp_down = self.down1_3(x1)  # N * 64 * h/2 * w/2
        x1 = self.conv16_64_1_2(x1)
        x1 = self.res_blk1_3(x1)

        x2 = torch.cat((x2, x2_temp_down, x2_temp_up), dim=1)
        x2 = self.conv1x1_2_4(x2)
        x2 = self.res_blk2_3(x2)
        x2 = self.conv64_256_2_5(x2)  # N * 256 * h/2 * w/2
        x1_temp = self.up2_3(x2)

        x1 = torch.cat((x1, x1_temp), dim=1)  # N * 128 * h * w
        x1 = self.conv1x1_1_3(x1)
        x1 = self.res_blk1_4(x1)  # N * 64 * h * w
        x1 = self.conv_last(x1)
        output = x1[..., :h, :w]

        return output



class FTheta(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.dcnn = ResWUNet(in_channels=in_channels, out_channels=out_channels, n=1)

    def forward(self, inputs):
        sum_bk = inputs
        for k in range(1):
            bk = self.dcnn(inputs)
            sum_bk = sum_bk + (-1) ** (k + 1) * bk
            inputs = bk

        return sum_bk


"""
Following code come from https://github.com/dongjxjx/dwdn
"""


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x


def wiener_filter_para(_input_blur):
    median_filter = MedianPool2d(kernel_size=3, padding=1)(_input_blur)
    diff = median_filter - _input_blur
    num = (diff.shape[2]*diff.shape[3])
    mean_n = torch.sum(diff, (2, 3)).unsqueeze(dim=-1).unsqueeze(dim=-1)/num
    var_n = torch.sum((diff - mean_n) * (diff - mean_n), (2, 3))/(num-1)
    mean_input = torch.sum(_input_blur, (2, 3)).unsqueeze(dim=-1).unsqueeze(dim=-1)/num
    var_s2 = (torch.sum((_input_blur-mean_input)*(_input_blur-mean_input), (2, 3))/(num-1)) ** 0.5
    # NSR = var_n / var_s2 * 8.0 / 3.0 / 10.0
    NSR = var_n / var_s2
    NSR = torch.mean(NSR, dim=1)
    NSR = NSR.view(-1, 1, 1, 1)
    return NSR


if __name__ == '__main__':
    # device = torch.device('cuda')
    # x = torch.randn(1, 16, 128, 128).cuda()
    # net = NonLinOperatorSubNet(16)
    # net.to(device)
    # out = net(x)
    # print(out.shape)

    device = torch.device('cuda')
    x = torch.randn(1, 16, 128, 128).cuda()
    pass
