# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Author: Shengjiang Kong,School of mathematics and statistics, Xidian University.
Email: sjkongxd@gmail.com.
This Module implement ours proposed network: NBDRWD.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from .basicblocks import FTheta, wiener_filter_para
from models.commonblocks import (p2o,
                                 cconj,
                                 r2c,
                                 cabs2,
                                 cmul,
                                 csum,
                                 rfft,
                                 cdiv,
                                 irfft,
                                 Conv,
                                 Deconv,
                                 ResidualBlock)
from utils.othertools import variable_to_cv2_image


class DWDNPlus(nn.Module):
    """Implementation DWDN+ model
    """

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: int, the number of channels of input
        :param out_channels: the number of channels of output
        """
        super().__init__()
        n_resblock = 3
        n_feats1 = 16
        n_feats = 32
        kernel_size = 5

        FeatureBlock = [Conv(in_channels, n_feats1, kernel_size, padding=2, act=True),
                        ResidualBlock(Conv, n_feats1, kernel_size, padding=2),
                        ResidualBlock(Conv, n_feats1, kernel_size, padding=2),
                        ResidualBlock(Conv, n_feats1, kernel_size, padding=2)]
        # ------------------------------------------------------------------
        # Implements DWDN+: multi-scale cascaded encoder-decoder network
        # ------------------------------------------------------------------
        # ------------------------------------------------------------------
        # Cascaded 1
        # ------------------------------------------------------------------
        in_block_cascaded_1_1 = [Conv(n_feats1, n_feats, kernel_size, padding=2, act=True),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2)]

        in_block_cascaded_1_2 = [Conv(n_feats1 + n_feats, n_feats, kernel_size, padding=2, act=True),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2)]

        encoder_first_cascaded_1 = [Conv(n_feats, n_feats * 2, kernel_size, padding=2, stride=2, act=True),
                                    ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2)]
        encoder_second_cascaded_1 = [Conv(n_feats * 2, n_feats * 4, kernel_size, padding=2, stride=2, act=True),
                                     ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2),
                                     ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2),
                                     ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2)]
        encoder_third_cascaded_1 = [Conv(n_feats * 4, n_feats * 8, kernel_size, padding=2, stride=2, act=True),
                                    ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2)]
        # decoder3
        decoder_third_cascaded_1 = [ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2) for _ in range(n_resblock)]
        decoder_third_cascaded_1.append(Deconv(n_feats * 8, n_feats * 4, kernel_size=3, padding=1, output_padding=1, act=True))
        # decoder2
        decoder_second_cascaded_1 = [ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2) for _ in range(n_resblock)]
        decoder_second_cascaded_1.append(Deconv(n_feats * 4, n_feats * 2, kernel_size=3, padding=1, output_padding=1, act=True))
        # decoder1
        decoder_first_cascaded_1 = [ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2) for _ in range(n_resblock)]
        decoder_first_cascaded_1.append(Deconv(n_feats * 2, n_feats, kernel_size=3, padding=1, output_padding=1, act=True))

        out_block_cascaded_1 = [ResidualBlock(Conv, n_feats, kernel_size, padding=2) for _ in range(n_resblock)]
        out_block_cascaded_1_1 = [Conv(n_feats, n_feats1, kernel_size, padding=2)]
        out_block_cascaded_1_2 = [Conv(n_feats, n_feats1 + n_feats, kernel_size, padding=2)]

        # ------------------------------------------------------------------
        # Cascaded 2
        # ------------------------------------------------------------------
        in_block_cascaded_2_1 = [Conv(n_feats1, n_feats, kernel_size, padding=2, act=True),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2)]

        in_block_cascaded_2_2 = [Conv(n_feats1 + n_feats, n_feats, kernel_size, padding=2, act=True),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2)]

        encoder_first_cascaded_2 = [Conv(n_feats, n_feats * 2, kernel_size, padding=2, stride=2, act=True),
                                    ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2)]
        encoder_second_cascaded_2 = [Conv(n_feats * 2, n_feats * 4, kernel_size, padding=2, stride=2, act=True),
                                     ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2),
                                     ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2),
                                     ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2)]
        encoder_third_cascaded_2 = [Conv(n_feats * 4, n_feats * 8, kernel_size, padding=2, stride=2, act=True),
                                    ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2)]
        # decoder3
        decoder_third_cascaded_2 = [ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2) for _ in range(n_resblock)]
        decoder_third_cascaded_2.append(
            Deconv(n_feats * 8, n_feats * 4, kernel_size=3, padding=1, output_padding=1, act=True))
        # decoder2
        decoder_second_cascaded_2 = [ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2) for _ in
                                     range(n_resblock)]
        decoder_second_cascaded_2.append(
            Deconv(n_feats * 4, n_feats * 2, kernel_size=3, padding=1, output_padding=1, act=True))
        # decoder1
        decoder_first_cascaded_2 = [ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2) for _ in range(n_resblock)]
        decoder_first_cascaded_2.append(
            Deconv(n_feats * 2, n_feats, kernel_size=3, padding=1, output_padding=1, act=True))

        out_block_cascaded_2 = [ResidualBlock(Conv, n_feats, kernel_size, padding=2) for _ in range(n_resblock)]
        out_block_cascaded_2_1 = [Conv(n_feats, n_feats1, kernel_size, padding=2)]
        out_block_cascaded_2_2 = [Conv(n_feats, n_feats1 + n_feats, kernel_size, padding=2)]

        # ------------------------------------------------------------------
        # Cascaded 3 (final)
        # ------------------------------------------------------------------
        InBlock1 = [Conv(n_feats1, n_feats, kernel_size, padding=2, act=True),
                    ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                    ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                    ResidualBlock(Conv, n_feats, kernel_size, padding=2)]
        InBlock2 = [Conv(n_feats1 + n_feats, n_feats, kernel_size, padding=2, act=True),
                    ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                    ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                    ResidualBlock(Conv, n_feats, kernel_size, padding=2)]

        # encoder1
        Encoder_first = [Conv(n_feats, n_feats * 2, kernel_size, padding=2, stride=2, act=True),
                         ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2),
                         ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2),
                         ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2)]
        # encoder2
        Encoder_second = [Conv(n_feats * 2, n_feats * 4, kernel_size, padding=2, stride=2, act=True),
                          ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2),
                          ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2),
                          ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2)]
        # decoder2
        Decoder_second = [ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2) for _ in range(n_resblock)]
        Decoder_second.append(Deconv(n_feats * 4, n_feats * 2, kernel_size=3, padding=1, output_padding=1, act=True))
        # decoder1
        Decoder_first = [ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2) for _ in range(n_resblock)]
        Decoder_first.append(Deconv(n_feats * 2, n_feats, kernel_size=3, padding=1, output_padding=1, act=True))

        OutBlock = [ResidualBlock(Conv, n_feats, kernel_size, padding=2) for _ in range(n_resblock)]
        OutBlock2 = [Conv(n_feats, out_channels, kernel_size, padding=2)]

        self.FeatureBlock = nn.Sequential(*FeatureBlock)
        # ---------------------------------------------------------
        # Cascaded 1
        # ---------------------------------------------------------
        self.in_block_cascaded_1_1 = nn.Sequential(*in_block_cascaded_1_1)
        self.in_block_cascaded_1_2 = nn.Sequential(*in_block_cascaded_1_2)
        self.encoder_first_cascaded_1 = nn.Sequential(*encoder_first_cascaded_1)
        self.encoder_second_cascaded_1 = nn.Sequential(*encoder_second_cascaded_1)
        self.encoder_third_cascaded_1 = nn.Sequential(*encoder_third_cascaded_1)
        self.decoder_third_cascaded_1 = nn.Sequential(*decoder_third_cascaded_1)
        self.decoder_second_cascaded_1 = nn.Sequential(*decoder_second_cascaded_1)
        self.decoder_first_cascaded_1 = nn.Sequential(*decoder_first_cascaded_1)
        self.out_block_cascaded_1 = nn.Sequential(*out_block_cascaded_1)
        self.out_block_cascaded_1_1 = nn.Sequential(*out_block_cascaded_1_1)
        self.out_block_cascaded_1_2 = nn.Sequential(*out_block_cascaded_1_2)
        # ---------------------------------------------------------
        # Cascaded 2
        # ---------------------------------------------------------
        self.in_block_cascaded_2_1 = nn.Sequential(*in_block_cascaded_2_1)
        self.in_block_cascaded_2_2 = nn.Sequential(*in_block_cascaded_2_2)
        self.encoder_first_cascaded_2 = nn.Sequential(*encoder_first_cascaded_2)
        self.encoder_second_cascaded_2 = nn.Sequential(*encoder_second_cascaded_2)
        self.encoder_third_cascaded_2 = nn.Sequential(*encoder_third_cascaded_2)
        self.decoder_third_cascaded_2 = nn.Sequential(*decoder_third_cascaded_2)
        self.decoder_second_cascaded_2 = nn.Sequential(*decoder_second_cascaded_2)
        self.decoder_first_cascaded_2 = nn.Sequential(*decoder_first_cascaded_2)
        self.out_block_cascaded_2 = nn.Sequential(*out_block_cascaded_2)
        self.out_block_cascaded_2_1 = nn.Sequential(*out_block_cascaded_2_1)
        self.out_block_cascaded_2_2 = nn.Sequential(*out_block_cascaded_2_2)
        # ---------------------------------------------------------
        # Cascaded 3
        # ---------------------------------------------------------
        self.inBlock1 = nn.Sequential(*InBlock1)
        self.inBlock2 = nn.Sequential(*InBlock2)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.OutBlock = nn.Sequential(*OutBlock)
        self.OutBlock2 = nn.Sequential(*OutBlock2)
        # ---------------------------------------------------------
        # Non-Linear Sub-network
        # ---------------------------------------------------------
        # self.neumann_net = NeumannNet(n_feats1, n_feats1, n_iter)

    def forward(self, x, k):
        """
        :param x:  tensor, N x C x W x H
        :param k: tensor, tensor, N x (1,3) x w x h, Blur kernel
        :return: Output tensor N X C X H X W
        """
        # ---------------------------------------------------------
        # F(.) and F^(-1)(.) denote FFT and inverse FFT,
        # F(.)_bar denotes complex conjugate of F(.)
        # ---------------------------------------------------------
        x_feature = self.FeatureBlock(x)
        ks = k.shape[2:]
        dim = (ks[0], ks[0], ks[1], ks[1])
        clear_features = torch.zeros_like(x_feature)
        x_feature = F.pad(x_feature, dim, "replicate")

        for i in range(x_feature.shape[1]):
            blur_feature_ch = x_feature[:, i:i + 1, :, :]
            clear_feature_ch = wiener_deblur(blur_feature_ch, k)
            clear_features[:, i:i + 1, :, :] = clear_feature_ch[:, :, ks[1]:-ks[1], ks[0]:-ks[0]]

        n_levels = 2
        scale = 0.5
        output = []
        input_pre = None

        for level in range(n_levels):
            per_scale = scale ** (n_levels - level - 1)
            n, c, h, w = x.shape
            hi = int(round(h * per_scale))
            wi = int(round(w * per_scale))
            # hi = int(math.floor(h * scale))
            # wi = int(math.floor(w * scale))
            if level == 0:
                input_clear = F.interpolate(clear_features, (hi, wi), mode='bicubic', align_corners=True)
                inp_all = input_clear.cuda()

                # Cascaded 1
                first_scale_inblock = self.in_block_cascaded_1_1(inp_all)
                encoder_1_cascaded_1 = self.encoder_first_cascaded_1(first_scale_inblock)
                encoder_2_cascaded_1 = self.encoder_second_cascaded_1(encoder_1_cascaded_1)
                encoder_3_cascaded_1 = self.encoder_third_cascaded_1(encoder_2_cascaded_1)
                decoder_3_cascaded_1 = self.decoder_third_cascaded_1(encoder_3_cascaded_1)
                hh, ww = encoder_2_cascaded_1.shape[-2:]
                decoder_3_cascaded_1 = decoder_3_cascaded_1[..., :hh, :ww]
                decoder_2_cascaded_1 = self.decoder_second_cascaded_1(decoder_3_cascaded_1 + encoder_2_cascaded_1)
                hh, ww = encoder_1_cascaded_1.shape[-2:]
                decoder_2_cascaded_1 = decoder_2_cascaded_1[..., :hh, :ww]
                decoder_1_cascaded_1 = self.decoder_first_cascaded_1(decoder_2_cascaded_1 + encoder_1_cascaded_1)
                decoder_1_cascaded_1 = decoder_1_cascaded_1[..., :hi, :wi]
                out_1_cascaded_1 = self.out_block_cascaded_1(decoder_1_cascaded_1 + first_scale_inblock)
                out_2_cascaded_1 = self.out_block_cascaded_1_1(out_1_cascaded_1)
                first_scale_inblock_1 = inp_all + out_2_cascaded_1

                # Cascaded 2
                first_scale_inblock_1_out = self.in_block_cascaded_2_1(first_scale_inblock_1)
                encoder_1_cascaded_2 = self.encoder_first_cascaded_2(first_scale_inblock_1_out)
                encoder_2_cascaded_2 = self.encoder_second_cascaded_2(encoder_1_cascaded_2)
                encoder_3_cascaded_2 = self.encoder_third_cascaded_2(encoder_2_cascaded_2)
                decoder_3_cascaded_2 = self.decoder_third_cascaded_2(encoder_3_cascaded_2)
                hh, ww = encoder_2_cascaded_2.shape[-2:]
                decoder_3_cascaded_2 = decoder_3_cascaded_2[..., :hh, :ww]
                decoder_2_cascaded_2 = self.decoder_second_cascaded_2(decoder_3_cascaded_2 + encoder_2_cascaded_2)
                hh, ww = encoder_1_cascaded_2.shape[-2:]
                decoder_2_cascaded_2 = decoder_2_cascaded_2[..., :hh, :ww]
                decoder_1_cascaded_2 = self.decoder_first_cascaded_2(decoder_2_cascaded_2 + encoder_1_cascaded_2)
                decoder_1_cascaded_2 = decoder_1_cascaded_2[..., :hi, :wi]
                out_1_cascaded_2 = self.out_block_cascaded_2(decoder_1_cascaded_2 + first_scale_inblock_1_out)
                out_2_cascaded_2 = self.out_block_cascaded_2_1(out_1_cascaded_2)
                first_scale_inblock_2 = first_scale_inblock_1 + out_2_cascaded_2
                init_input = self.inBlock1(first_scale_inblock_2)
            else:
                input_clear = F.interpolate(clear_features, (hi, wi), mode='bicubic', align_corners=True)
                input_pred = F.interpolate(input_pre, (hi, wi), mode='bicubic')
                inp_all = torch.cat((input_clear.cuda(), input_pred), 1)

                # Cascaded 1
                first_scale_inblock = self.in_block_cascaded_1_2(inp_all)
                encoder_1_cascaded_1 = self.encoder_first_cascaded_1(first_scale_inblock)
                encoder_2_cascaded_1 = self.encoder_second_cascaded_1(encoder_1_cascaded_1)
                encoder_3_cascaded_1 = self.encoder_third_cascaded_1(encoder_2_cascaded_1)
                decoder_3_cascaded_1 = self.decoder_third_cascaded_1(encoder_3_cascaded_1)
                hh, ww = encoder_2_cascaded_1.shape[-2:]
                decoder_3_cascaded_1 = decoder_3_cascaded_1[..., :hh, :ww]
                decoder_2_cascaded_1 = self.decoder_second_cascaded_1(decoder_3_cascaded_1 + encoder_2_cascaded_1)
                hh, ww = encoder_1_cascaded_1.shape[-2:]
                decoder_2_cascaded_1 = decoder_2_cascaded_1[..., :hh, :ww]
                decoder_1_cascaded_1 = self.decoder_first_cascaded_1(decoder_2_cascaded_1 + encoder_1_cascaded_1)
                decoder_1_cascaded_1 = decoder_1_cascaded_1[..., :hi, :wi]
                out_1_cascaded_1 = self.out_block_cascaded_1(decoder_1_cascaded_1 + first_scale_inblock)
                out_2_cascaded_1 = self.out_block_cascaded_1_2(out_1_cascaded_1)
                first_scale_inblock_1 = inp_all + out_2_cascaded_1

                # Cascaded 2
                first_scale_inblock_1_out = self.in_block_cascaded_2_2(first_scale_inblock_1)
                encoder_1_cascaded_2 = self.encoder_first_cascaded_2(first_scale_inblock_1_out)
                encoder_2_cascaded_2 = self.encoder_second_cascaded_2(encoder_1_cascaded_2)
                encoder_3_cascaded_2 = self.encoder_third_cascaded_2(encoder_2_cascaded_2)
                decoder_3_cascaded_2 = self.decoder_third_cascaded_2(encoder_3_cascaded_2)
                hh, ww = encoder_2_cascaded_2.shape[-2:]
                decoder_3_cascaded_2 = decoder_3_cascaded_2[..., :hh, :ww]
                decoder_2_cascaded_2 = self.decoder_second_cascaded_2(decoder_3_cascaded_2 + encoder_2_cascaded_2)
                hh, ww = encoder_1_cascaded_2.shape[-2:]
                decoder_2_cascaded_2 = decoder_2_cascaded_2[..., :hh, :ww]
                decoder_1_cascaded_2 = self.decoder_first_cascaded_2(decoder_2_cascaded_2 + encoder_1_cascaded_2)
                decoder_1_cascaded_2 = decoder_1_cascaded_2[..., :hi, :wi]
                out_1_cascaded_2 = self.out_block_cascaded_2(decoder_1_cascaded_2 + first_scale_inblock_1_out)
                out_2_cascaded_2 = self.out_block_cascaded_2_2(out_1_cascaded_2)
                first_scale_inblock_2 = first_scale_inblock_1 + out_2_cascaded_2
                init_input = self.inBlock2(first_scale_inblock_2)

            # Cascaded 3
            first_scale_encoder_first = self.encoder_first(init_input)
            first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)
            first_scale_decoder_second = self.decoder_second(first_scale_encoder_second)

            # Modified by Shengjiang Kong
            hh, ww = first_scale_encoder_first.shape[-2:]
            first_scale_decoder_second = first_scale_decoder_second[..., :hh, :ww]
            first_scale_decoder_first = self.decoder_first(first_scale_decoder_second + first_scale_encoder_first)

            # Modified by Shengjiang Kong
            first_scale_decoder_first = first_scale_decoder_first[..., :hi, :wi]

            input_pre = self.OutBlock(first_scale_decoder_first + init_input)
            out = self.OutBlock2(input_pre)
            output.append(out)

        return output[-1]


class NBDRWD(nn.Module):
    """Our proposed NBDRWD For Image  Non-blind Deblurring
    """

    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: int, the number of channels of input
        :param out_channels: the number of channels of output
        """
        super().__init__()
        n_resblock = 3
        n_feats1 = 16
        n_feats = 32
        kernel_size = 5

        FeatureBlock = [Conv(in_channels, n_feats1, kernel_size, padding=2, act=True),
                        ResidualBlock(Conv, n_feats1, kernel_size, padding=2),
                        ResidualBlock(Conv, n_feats1, kernel_size, padding=2),
                        ResidualBlock(Conv, n_feats1, kernel_size, padding=2)]
        # ------------------------------------------------------------------
        # Implements DWDN+: multi-scale cascaded encoder-decoder network
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Cascaded 1
        # ------------------------------------------------------------------
        in_block_cascaded_1_1 = [Conv(n_feats1, n_feats, kernel_size, padding=2, act=True),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2)]

        in_block_cascaded_1_2 = [Conv(n_feats1 + n_feats, n_feats, kernel_size, padding=2, act=True),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2)]

        encoder_first_cascaded_1 = [Conv(n_feats, n_feats * 2, kernel_size, padding=2, stride=2, act=True),
                                    ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2)]
        encoder_second_cascaded_1 = [Conv(n_feats * 2, n_feats * 4, kernel_size, padding=2, stride=2, act=True),
                                     ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2),
                                     ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2),
                                     ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2)]
        encoder_third_cascaded_1 = [Conv(n_feats * 4, n_feats * 8, kernel_size, padding=2, stride=2, act=True),
                                    ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2)]
        # decoder3
        decoder_third_cascaded_1 = [ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2) for _ in range(n_resblock)]
        decoder_third_cascaded_1.append(Deconv(n_feats * 8, n_feats * 4, kernel_size=3, padding=1, output_padding=1, act=True))
        # decoder2
        decoder_second_cascaded_1 = [ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2) for _ in range(n_resblock)]
        decoder_second_cascaded_1.append(Deconv(n_feats * 4, n_feats * 2, kernel_size=3, padding=1, output_padding=1, act=True))
        # decoder1
        decoder_first_cascaded_1 = [ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2) for _ in range(n_resblock)]
        decoder_first_cascaded_1.append(Deconv(n_feats * 2, n_feats, kernel_size=3, padding=1, output_padding=1, act=True))

        out_block_cascaded_1 = [ResidualBlock(Conv, n_feats, kernel_size, padding=2) for _ in range(n_resblock)]
        out_block_cascaded_1_1 = [Conv(n_feats, n_feats1, kernel_size, padding=2)]
        out_block_cascaded_1_2 = [Conv(n_feats, n_feats1 + n_feats, kernel_size, padding=2)]

        # ------------------------------------------------------------------
        # Cascaded 2
        # ------------------------------------------------------------------
        in_block_cascaded_2_1 = [Conv(n_feats1, n_feats, kernel_size, padding=2, act=True),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2)]

        in_block_cascaded_2_2 = [Conv(n_feats1 + n_feats, n_feats, kernel_size, padding=2, act=True),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                                 ResidualBlock(Conv, n_feats, kernel_size, padding=2)]

        encoder_first_cascaded_2 = [Conv(n_feats, n_feats * 2, kernel_size, padding=2, stride=2, act=True),
                                    ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2)]
        encoder_second_cascaded_2 = [Conv(n_feats * 2, n_feats * 4, kernel_size, padding=2, stride=2, act=True),
                                     ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2),
                                     ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2),
                                     ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2)]
        encoder_third_cascaded_2 = [Conv(n_feats * 4, n_feats * 8, kernel_size, padding=2, stride=2, act=True),
                                    ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2),
                                    ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2)]
        # decoder3
        decoder_third_cascaded_2 = [ResidualBlock(Conv, n_feats * 8, kernel_size, padding=2) for _ in range(n_resblock)]
        decoder_third_cascaded_2.append(
            Deconv(n_feats * 8, n_feats * 4, kernel_size=3, padding=1, output_padding=1, act=True))
        # decoder2
        decoder_second_cascaded_2 = [ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2) for _ in
                                     range(n_resblock)]
        decoder_second_cascaded_2.append(
            Deconv(n_feats * 4, n_feats * 2, kernel_size=3, padding=1, output_padding=1, act=True))
        # decoder1
        decoder_first_cascaded_2 = [ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2) for _ in range(n_resblock)]
        decoder_first_cascaded_2.append(
            Deconv(n_feats * 2, n_feats, kernel_size=3, padding=1, output_padding=1, act=True))

        out_block_cascaded_2 = [ResidualBlock(Conv, n_feats, kernel_size, padding=2) for _ in range(n_resblock)]
        out_block_cascaded_2_1 = [Conv(n_feats, n_feats1, kernel_size, padding=2)]
        out_block_cascaded_2_2 = [Conv(n_feats, n_feats1 + n_feats, kernel_size, padding=2)]

        # ------------------------------------------------------------------
        # Cascaded 3 (final)
        # ------------------------------------------------------------------
        InBlock1 = [Conv(n_feats1, n_feats, kernel_size, padding=2, act=True),
                    ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                    ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                    ResidualBlock(Conv, n_feats, kernel_size, padding=2)]
        InBlock2 = [Conv(n_feats1 + n_feats, n_feats, kernel_size, padding=2, act=True),
                    ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                    ResidualBlock(Conv, n_feats, kernel_size, padding=2),
                    ResidualBlock(Conv, n_feats, kernel_size, padding=2)]

        # encoder1
        Encoder_first = [Conv(n_feats, n_feats * 2, kernel_size, padding=2, stride=2, act=True),
                         ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2),
                         ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2),
                         ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2)]
        # encoder2
        Encoder_second = [Conv(n_feats * 2, n_feats * 4, kernel_size, padding=2, stride=2, act=True),
                          ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2),
                          ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2),
                          ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2)]
        # decoder2
        Decoder_second = [ResidualBlock(Conv, n_feats * 4, kernel_size, padding=2) for _ in range(n_resblock)]
        Decoder_second.append(Deconv(n_feats * 4, n_feats * 2, kernel_size=3, padding=1, output_padding=1, act=True))
        # decoder1
        Decoder_first = [ResidualBlock(Conv, n_feats * 2, kernel_size, padding=2) for _ in range(n_resblock)]
        Decoder_first.append(Deconv(n_feats * 2, n_feats, kernel_size=3, padding=1, output_padding=1, act=True))

        OutBlock = [ResidualBlock(Conv, n_feats, kernel_size, padding=2) for _ in range(n_resblock)]
        OutBlock2 = [Conv(n_feats, out_channels, kernel_size, padding=2)]

        self.FeatureBlock = nn.Sequential(*FeatureBlock)
        # ---------------------------------------------------------
        # Cascaded 1
        # ---------------------------------------------------------
        self.in_block_cascaded_1_1 = nn.Sequential(*in_block_cascaded_1_1)
        self.in_block_cascaded_1_2 = nn.Sequential(*in_block_cascaded_1_2)
        self.encoder_first_cascaded_1 = nn.Sequential(*encoder_first_cascaded_1)
        self.encoder_second_cascaded_1 = nn.Sequential(*encoder_second_cascaded_1)
        self.encoder_third_cascaded_1 = nn.Sequential(*encoder_third_cascaded_1)
        self.decoder_third_cascaded_1 = nn.Sequential(*decoder_third_cascaded_1)
        self.decoder_second_cascaded_1 = nn.Sequential(*decoder_second_cascaded_1)
        self.decoder_first_cascaded_1 = nn.Sequential(*decoder_first_cascaded_1)
        self.out_block_cascaded_1 = nn.Sequential(*out_block_cascaded_1)
        self.out_block_cascaded_1_1 = nn.Sequential(*out_block_cascaded_1_1)
        self.out_block_cascaded_1_2 = nn.Sequential(*out_block_cascaded_1_2)
        # ---------------------------------------------------------
        # Cascaded 2
        # ---------------------------------------------------------
        self.in_block_cascaded_2_1 = nn.Sequential(*in_block_cascaded_2_1)
        self.in_block_cascaded_2_2 = nn.Sequential(*in_block_cascaded_2_2)
        self.encoder_first_cascaded_2 = nn.Sequential(*encoder_first_cascaded_2)
        self.encoder_second_cascaded_2 = nn.Sequential(*encoder_second_cascaded_2)
        self.encoder_third_cascaded_2 = nn.Sequential(*encoder_third_cascaded_2)
        self.decoder_third_cascaded_2 = nn.Sequential(*decoder_third_cascaded_2)
        self.decoder_second_cascaded_2 = nn.Sequential(*decoder_second_cascaded_2)
        self.decoder_first_cascaded_2 = nn.Sequential(*decoder_first_cascaded_2)
        self.out_block_cascaded_2 = nn.Sequential(*out_block_cascaded_2)
        self.out_block_cascaded_2_1 = nn.Sequential(*out_block_cascaded_2_1)
        self.out_block_cascaded_2_2 = nn.Sequential(*out_block_cascaded_2_2)
        # ---------------------------------------------------------
        # Cascaded 3
        # ---------------------------------------------------------
        self.inBlock1 = nn.Sequential(*InBlock1)
        self.inBlock2 = nn.Sequential(*InBlock2)
        self.encoder_first = nn.Sequential(*Encoder_first)
        self.encoder_second = nn.Sequential(*Encoder_second)
        self.decoder_second = nn.Sequential(*Decoder_second)
        self.decoder_first = nn.Sequential(*Decoder_first)
        self.OutBlock = nn.Sequential(*OutBlock)
        self.OutBlock2 = nn.Sequential(*OutBlock2)
        # ---------------------------------------------------------
        # Sub-network: F_theta
        # ---------------------------------------------------------
        self.neumann_net = FTheta(n_feats1, n_feats1)

    def forward(self, x, k):
        """
        :param x:  tensor, N x C x W x H
        :param k: tensor, tensor, N x (1,3) x w x h, Blur kernel
        :return: Output tensor N X C X H X W
        """
        # ---------------------------------------------------------
        # F(.) and F^(-1)(.) denote FFT and inverse FFT,
        # F(.)_bar denotes complex conjugate of F(.)
        # ---------------------------------------------------------
        x_feature = self.FeatureBlock(x)
        ks = k.shape[2:]
        dim = (ks[0], ks[0], ks[1], ks[1])
        clear_features = torch.zeros_like(x_feature)
        x_feature = F.pad(x_feature, dim, "replicate")

        for i in range(x_feature.shape[1]):
            blur_feature_ch = x_feature[:, i:i + 1, :, :]
            clear_feature_ch = wiener_deblur(blur_feature_ch, k)
            clear_features[:, i:i + 1, :, :] = clear_feature_ch[:, :, ks[1]:-ks[1], ks[0]:-ks[0]]

        clear_features = self.neumann_net(clear_features)

        n_levels = 2
        scale = 0.5
        output = []
        input_pre = None

        for level in range(n_levels):
            per_scale = scale ** (n_levels - level - 1)
            n, c, h, w = x.shape
            hi = int(round(h * per_scale))
            wi = int(round(w * per_scale))
            # hi = int(math.floor(h * scale))
            # wi = int(math.floor(w * scale))
            if level == 0:
                input_clear = F.interpolate(clear_features, (hi, wi), mode='bicubic', align_corners=True)
                inp_all = input_clear.cuda()

                # Cascaded 1
                first_scale_inblock = self.in_block_cascaded_1_1(inp_all)
                encoder_1_cascaded_1 = self.encoder_first_cascaded_1(first_scale_inblock)
                encoder_2_cascaded_1 = self.encoder_second_cascaded_1(encoder_1_cascaded_1)
                encoder_3_cascaded_1 = self.encoder_third_cascaded_1(encoder_2_cascaded_1)
                decoder_3_cascaded_1 = self.decoder_third_cascaded_1(encoder_3_cascaded_1)
                hh, ww = encoder_2_cascaded_1.shape[-2:]
                decoder_3_cascaded_1 = decoder_3_cascaded_1[..., :hh, :ww]
                decoder_2_cascaded_1 = self.decoder_second_cascaded_1(decoder_3_cascaded_1 + encoder_2_cascaded_1)
                hh, ww = encoder_1_cascaded_1.shape[-2:]
                decoder_2_cascaded_1 = decoder_2_cascaded_1[..., :hh, :ww]
                decoder_1_cascaded_1 = self.decoder_first_cascaded_1(decoder_2_cascaded_1 + encoder_1_cascaded_1)
                decoder_1_cascaded_1 = decoder_1_cascaded_1[..., :hi, :wi]
                out_1_cascaded_1 = self.out_block_cascaded_1(decoder_1_cascaded_1 + first_scale_inblock)
                out_2_cascaded_1 = self.out_block_cascaded_1_1(out_1_cascaded_1)
                first_scale_inblock_1 = inp_all + out_2_cascaded_1

                # Cascaded 2
                first_scale_inblock_1_out = self.in_block_cascaded_2_1(first_scale_inblock_1)
                encoder_1_cascaded_2 = self.encoder_first_cascaded_2(first_scale_inblock_1_out)
                encoder_2_cascaded_2 = self.encoder_second_cascaded_2(encoder_1_cascaded_2)
                encoder_3_cascaded_2 = self.encoder_third_cascaded_2(encoder_2_cascaded_2)
                decoder_3_cascaded_2 = self.decoder_third_cascaded_2(encoder_3_cascaded_2)
                hh, ww = encoder_2_cascaded_2.shape[-2:]
                decoder_3_cascaded_2 = decoder_3_cascaded_2[..., :hh, :ww]
                decoder_2_cascaded_2 = self.decoder_second_cascaded_2(decoder_3_cascaded_2 + encoder_2_cascaded_2)
                hh, ww = encoder_1_cascaded_2.shape[-2:]
                decoder_2_cascaded_2 = decoder_2_cascaded_2[..., :hh, :ww]
                decoder_1_cascaded_2 = self.decoder_first_cascaded_2(decoder_2_cascaded_2 + encoder_1_cascaded_2)
                decoder_1_cascaded_2 = decoder_1_cascaded_2[..., :hi, :wi]
                out_1_cascaded_2 = self.out_block_cascaded_2(decoder_1_cascaded_2 + first_scale_inblock_1_out)
                out_2_cascaded_2 = self.out_block_cascaded_2_1(out_1_cascaded_2)
                first_scale_inblock_2 = first_scale_inblock_1 + out_2_cascaded_2
                init_input = self.inBlock1(first_scale_inblock_2)
            else:
                input_clear = F.interpolate(clear_features, (hi, wi), mode='bicubic', align_corners=True)
                input_pred = F.interpolate(input_pre, (hi, wi), mode='bicubic')
                inp_all = torch.cat((input_clear.cuda(), input_pred), 1)

                # Cascaded 1
                first_scale_inblock = self.in_block_cascaded_1_2(inp_all)
                encoder_1_cascaded_1 = self.encoder_first_cascaded_1(first_scale_inblock)
                encoder_2_cascaded_1 = self.encoder_second_cascaded_1(encoder_1_cascaded_1)
                encoder_3_cascaded_1 = self.encoder_third_cascaded_1(encoder_2_cascaded_1)
                decoder_3_cascaded_1 = self.decoder_third_cascaded_1(encoder_3_cascaded_1)
                hh, ww = encoder_2_cascaded_1.shape[-2:]
                decoder_3_cascaded_1 = decoder_3_cascaded_1[..., :hh, :ww]
                decoder_2_cascaded_1 = self.decoder_second_cascaded_1(decoder_3_cascaded_1 + encoder_2_cascaded_1)
                hh, ww = encoder_1_cascaded_1.shape[-2:]
                decoder_2_cascaded_1 = decoder_2_cascaded_1[..., :hh, :ww]
                decoder_1_cascaded_1 = self.decoder_first_cascaded_1(decoder_2_cascaded_1 + encoder_1_cascaded_1)
                decoder_1_cascaded_1 = decoder_1_cascaded_1[..., :hi, :wi]
                out_1_cascaded_1 = self.out_block_cascaded_1(decoder_1_cascaded_1 + first_scale_inblock)
                out_2_cascaded_1 = self.out_block_cascaded_1_2(out_1_cascaded_1)
                first_scale_inblock_1 = inp_all + out_2_cascaded_1

                # Cascaded 2
                first_scale_inblock_1_out = self.in_block_cascaded_2_2(first_scale_inblock_1)
                encoder_1_cascaded_2 = self.encoder_first_cascaded_2(first_scale_inblock_1_out)
                encoder_2_cascaded_2 = self.encoder_second_cascaded_2(encoder_1_cascaded_2)
                encoder_3_cascaded_2 = self.encoder_third_cascaded_2(encoder_2_cascaded_2)
                decoder_3_cascaded_2 = self.decoder_third_cascaded_2(encoder_3_cascaded_2)
                hh, ww = encoder_2_cascaded_2.shape[-2:]
                decoder_3_cascaded_2 = decoder_3_cascaded_2[..., :hh, :ww]
                decoder_2_cascaded_2 = self.decoder_second_cascaded_2(decoder_3_cascaded_2 + encoder_2_cascaded_2)
                hh, ww = encoder_1_cascaded_2.shape[-2:]
                decoder_2_cascaded_2 = decoder_2_cascaded_2[..., :hh, :ww]
                decoder_1_cascaded_2 = self.decoder_first_cascaded_2(decoder_2_cascaded_2 + encoder_1_cascaded_2)
                decoder_1_cascaded_2 = decoder_1_cascaded_2[..., :hi, :wi]
                out_1_cascaded_2 = self.out_block_cascaded_2(decoder_1_cascaded_2 + first_scale_inblock_1_out)
                out_2_cascaded_2 = self.out_block_cascaded_2_2(out_1_cascaded_2)
                first_scale_inblock_2 = first_scale_inblock_1 + out_2_cascaded_2
                init_input = self.inBlock2(first_scale_inblock_2)

            # Cascaded 3
            first_scale_encoder_first = self.encoder_first(init_input)
            first_scale_encoder_second = self.encoder_second(first_scale_encoder_first)
            first_scale_decoder_second = self.decoder_second(first_scale_encoder_second)

            # Modified by Shengjiang Kong
            hh, ww = first_scale_encoder_first.shape[-2:]
            first_scale_decoder_second = first_scale_decoder_second[..., :hh, :ww]
            first_scale_decoder_first = self.decoder_first(first_scale_decoder_second + first_scale_encoder_first)

            # Modified by Shengjiang Kong
            first_scale_decoder_first = first_scale_decoder_first[..., :hi, :wi]

            input_pre = self.OutBlock(first_scale_decoder_first + init_input)
            out = self.OutBlock2(input_pre)
            output.append(out)

        return output[-1]


def wiener_deblur(inputs, kernel):
    """
    De-conv by wiener filter
    :param inputs: N X C X H X W, tensor
    :param kernel: 1 X 1 X h X w, tensor
    :return: N X C X H X W, tensor
    """
    w, h = inputs.shape[-2:]
    fk = p2o(kernel, (w, h))  # F(k)
    fkc = cconj(fk, inplace=False)  # F(k)_bar
    fk2 = r2c(cabs2(fk))  # F(k)_bar * F(k)
    snr = wiener_filter_para(inputs)
    fk2 = csum(fk2, snr)
    fkc_fy = cmul(fkc, rfft(inputs))  # F(k)_bar * F(y)
    fkc_fy_div_fk2 = cdiv(fkc_fy, fk2)  # N X C X H X W x 2
    outputs = irfft(fkc_fy_div_fk2)

    return outputs
