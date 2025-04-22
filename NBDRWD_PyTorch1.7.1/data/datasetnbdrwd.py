# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

import random
import cv2
import numpy as np
from torch.utils.data import Dataset
from scipy import ndimage
from utils import (get_image_paths,
                   imread_uint,
                   augment_img,
                   uint2single,
                   single2tensor3,
                   uint2tensor3,
                   blurkernel_synthesis)


class DatasetNBDRWD(Dataset):
    """
    Dataset of NBDRWD for non-blind deblurring.
    """

    def __init__(
            self,
            root_dir='',
            patch_size=256,
            kernels=None,
            num_channels=3,
            sigma_max=25,
            nl=2.55,
            kernel_index=0,
            is_train=True
    ):
        super().__init__()
        self.is_train = is_train
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.sigma_max = sigma_max
        self.kernels = kernels
        self.kernel_index = kernel_index
        self.val_noise_level = nl
        self.high_img_paths = get_image_paths(root_dir)

    def __getitem__(self, index):
        # Get high quality image
        h_path = self.high_img_paths[index]
        h_img = imread_uint(h_path, self.num_channels)
        if self.is_train:
            height, width, _ = h_img.shape
            # ----------------------------
            #  1) randomly crop the patch
            # ----------------------------
            rnd_h = random.randint(0, max(0, height - self.patch_size))
            rnd_w = random.randint(0, max(0, width - self.patch_size))
            h_patch = h_img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            # ---------------------------
            # augmentation - flip, rotate
            # ---------------------------
            h_patch = augment_img(h_patch, mode=np.random.randint(0, 8))
            # ---------------------------
            #  2) kernel
            # ---------------------------
            kernel = blurkernel_synthesis(h=25)  # motion blur
            kernel /= np.sum(kernel)
            # Set noise level
            noise_level = np.random.uniform(0., self.sigma_max) / 255.0
            # ---------------------------
            # Low-quality image
            # ---------------------------
            # blur and down-sample
            l_img = ndimage.filters.convolve(h_patch, np.expand_dims(kernel, axis=2), mode='wrap')
            # add Gaussian noise
            l_img = uint2single(l_img) + np.random.normal(0., noise_level, l_img.shape)
            h_img = h_patch
        else:
            kernel = self.kernels[0, self.kernel_index].astype(np.float64)  # validation kernel
            kernel /= np.sum(kernel)
            noise_level = self.val_noise_level / 255.0  # validation noise level
            l_img = ndimage.filters.convolve(h_img, np.expand_dims(kernel, axis=2), mode='wrap')  # blur
            l_img = uint2single(l_img) + np.random.normal(0., noise_level, l_img.shape)

        kernel = single2tensor3(np.expand_dims(np.float32(kernel), axis=2))
        h_img, l_img = uint2tensor3(h_img), single2tensor3(l_img)

        return h_img, l_img, kernel

    def __len__(self):
        return len(self.high_img_paths)


class DatasetNBDRWDJPEGCompression(Dataset):
    """
    Dataset of NBDRWD for non-blind deblurring with JPEGCompression.
    """

    def __init__(
            self,
            root_dir='',
            patch_size=256,
            kernels=None,
            num_channels=3,
            kernel_index=0,
            is_train=True
    ):
        super().__init__()
        self.is_train = is_train
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.kernels = kernels
        self.kernel_index = kernel_index
        self.high_img_paths = get_image_paths(root_dir)

    def __getitem__(self, index):
        # Get high quality image
        h_path = self.high_img_paths[index]
        h_img = imread_uint(h_path, self.num_channels)
        if self.is_train:
            height, width, _ = h_img.shape
            # ----------------------------
            #  1) randomly crop the patch
            # ----------------------------
            rnd_h = random.randint(0, max(0, height - self.patch_size))
            rnd_w = random.randint(0, max(0, width - self.patch_size))
            h_patch = h_img[rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size, :]
            # ---------------------------
            # augmentation - flip, rotate
            # ---------------------------
            h_patch = augment_img(h_patch, mode=np.random.randint(0, 8))
            # ---------------------------
            #  2) kernel
            # ---------------------------
            kernel = blurkernel_synthesis(h=25)  # motion blur
            kernel /= np.sum(kernel)
            # Set noise level
            # noise_level = np.random.uniform(0., self.sigma_max) / 255.0
            noise_level = 2.55 / 255.0
            # ---------------------------
            # Low-quality image
            # ---------------------------
            # blur and down-sample
            l_img = ndimage.filters.convolve(h_patch, np.expand_dims(kernel, axis=2), mode='wrap')
            # add Gaussian noise
            l_img = uint2single(l_img) + np.random.normal(0., noise_level, l_img.shape)
            # add JPEG compression.
            jpeg_quality_factors = [50, 60, 70, 80, 90]
            qf_value = np.random.randint(0, len(jpeg_quality_factors))
            jpeg_quality_factor = jpeg_quality_factors[qf_value]
            _, encimg = cv2.imencode('.jpg', 255.0 * l_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality_factor])
            l_img = cv2.imdecode(encimg, 3)
            l_img = l_img / 255.
            h_img = h_patch
        else:
            kernel = self.kernels[0, self.kernel_index].astype(np.float64)  # validation kernel
            kernel /= np.sum(kernel)
            l_img = ndimage.filters.convolve(h_img, np.expand_dims(kernel, axis=2), mode='wrap')  # blur
            jpeg_quality_factor = 80
            l_img = uint2single(l_img) + np.random.normal(0., 2.55 / 255., l_img.shape)
            _, encimg = cv2.imencode('.jpg', 255.0 * l_img, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality_factor])
            l_img = cv2.imdecode(encimg, 3)
            l_img = l_img / 255.

        kernel = single2tensor3(np.expand_dims(np.float32(kernel), axis=2))
        h_img, l_img = uint2tensor3(h_img), single2tensor3(l_img)

        return h_img, l_img, kernel

    def __len__(self):
        return len(self.high_img_paths)
