# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch
from models import (NBDRWD, DWDNPlus)


def return_model_info(model_select_name='NBDRWD', num_channels=3):
    """
    :param model_select_name:
    (1)'NBDRWD' Proposed method for non-blind deblurring with uniform or non-uniform AWGN.
    (2)'NBDRWD_JPEG' Proposed method for non-blind deblurring with uniform AWGN and JPEG compression.
    (3)'DWDNPlus' Retrained DWDN+ model adopt same training strategy as NBDRWD.
    (4)'DWDNPlus_JPEG' Retrained DWDN+ for non-blind deblurring with uniform AWGN and JPEG compression.
    :param num_channels:
    :param iter_num: 1 or 3.
    :return:
    """
    model_name = None
    net = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_select_name == 'NBDRWD':
        model_name = 'NBDRWD.pth'
        net = NBDRWD(num_channels, num_channels).to(device=device)
    elif model_select_name == 'NBDRWD_JPEG':
        model_name = 'NBDRWD_JPEG.pth'
        net = NBDRWD(num_channels, num_channels).to(device=device)
    elif model_select_name == 'DWDNPlus':
        model_name = 'DWDNPlus.pth'
        net = DWDNPlus(num_channels, num_channels).to(device=device)
    elif model_select_name == 'DWDNPlus_JPEG':
        model_name = 'DWDNPlus_JPEG.pth'
        net = DWDNPlus(num_channels, num_channels).to(device=device)

    return model_name, net
