# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

from abc import ABC, abstractmethod
import os
import cv2
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from scipy import ndimage
from tqdm import tqdm
from models.modeloption import return_model_info
from trainnet import train_deblurring_net
from utils import generate_gauss_kernel_mix, sincos_kernel, peaks
from utils.othertools import (weights_init_kaiming,
                              batch_psnr,
                              batch_ssim,
                              init_logger_ipol,
                              imread_uint8,
                              remove_data_parallel_wrapper,
                              variable_to_cv2_image,
                              extract_file_name,
                              make_dirs)


class AlgorithmBase(ABC):
    """
    Basic class for all algorithms
    """

    def __init__(self, params):
        self.params = params  # parameters for algorithms

    @abstractmethod
    def train(self, *args):
        pass

    @abstractmethod
    def test(self, *args):
        pass

    @abstractmethod
    def test_real(self, *args):
        pass

    @abstractmethod
    def test_nonuniform(self, *args):
        pass

    @abstractmethod
    def test_jpeg_compression(self, *args):
        pass


class AlgorithmNonBlindDeblurring(AlgorithmBase):
    """
    Main Algorithm for Non-Blind Deblurring.
    """

    def __init__(self, params):
        super().__init__(params)

    def train(self, img_deblur_handle):
        """
        Training Network
        :param img_deblur_handle: the object of ImageDeblur class.
        :return:
        """
        # -------------------------------------------
        # Build network
        # -------------------------------------------
        # num_channels = self.params.num_channels
        net = self.get_net(self.params.num_channels)
        print(net)
        # -------------------------------------------
        # Initialize network
        # -------------------------------------------
        net.apply(weights_init_kaiming)
        # criterion = nn.L1Loss(reduction='sum').to(device=self.params.device)
        # print(criterion)
        criterion = nn.L1Loss().to(device=self.params.device)
        # -------------------------------------------
        # Move to GPU
        # -------------------------------------------
        model = nn.DataParallel(net, device_ids=self.params.device_ids).cuda()
        criterion.cuda()

        # -------------------------------------------
        # Optimizer
        # -------------------------------------------
        optimizer = optim.Adam(model.parameters(), lr=self.params.lr)

        # -------------------------------------------
        # training
        # -------------------------------------------
        train_set = img_deblur_handle.train_set
        validate_set = img_deblur_handle.validate_set
        train_deblurring_net(model, train_set, validate_set, criterion, optimizer, self.params)

    def test(self, img_deblur_handle):
        """
        Test trained Network
        :param : img_deblur_handle, the object of ImageDeblur class.
        :return:
        """
        print('testing on {} data ...\n'.format(self.params.test_data))
        # ------------------------------------------------------
        # 1) Initial
        # ------------------------------------------------------
        psnr_test = 0
        ssim_test = 0
        all_psnr_test = 0
        all_ssim_test = 0
        # -------------------------------------------------------
        # 2) load test dataset
        # -------------------------------------------------------
        test_set = img_deblur_handle.test_set
        # -------------------------------------------------------
        # 3) Absolute path to model file
        # -------------------------------------------------------
        net_file = self.get_net_file()
        # -------------------------------------------------------
        # 4) Create model and Load saved weights
        # -------------------------------------------------------
        num_channels = self.params.num_channels
        model = self.get_model(net_file, num_channels)
        # --------------------------------------------------------
        # load kernel shape: (1, n)
        # --------------------------------------------------------
        blur_kernels = img_deblur_handle.kernels
        concreate_dir = ''.join([self.params.test_data, '_', str(self.params.test_noiseL)])
        make_dirs(os.path.join('results', concreate_dir))

        log_all_psnr_ssim_name = os.path.join('results', concreate_dir, 'logger_summary_PSNR_SSIM_' + self.params.model_name.split('.')[0] + '.txt')
        logger_all = init_logger_ipol(file_name=log_all_psnr_ssim_name, obj_name='all')
        logger_all.info('Testing on {}\'s dataset'.format(self.params.test_data))

        for kk in range(blur_kernels.size):
            # make save directory
            make_dirs(os.path.join('results', concreate_dir, ''.join(['kernel', str(kk)])))

            # Init logger
            # Single kernel logger
            log_file_name = os.path.join('results',
                                         concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         str(kk) + '_logger_' + self.params.model_name.split('.')[0] + '.txt')
            logger = init_logger_ipol(file_name=log_file_name, obj_name=''.join(['avg', str(kk)]))

            print('*' * (len(net_file) + 10))
            print('Testing on {}th kernel'.format(kk))
            print('*' * (len(net_file) + 10))

            logger.info("\n")
            logger.info('*' * (len(net_file) + 10))
            logger.info('Testing on {}th kernel'.format(kk))
            logger.info('*' * (len(net_file) + 10))
            logger_all.info("\n")
            logger_all.info('*' * (len(net_file) + 10))
            logger_all.info('Testing on {}th kernel'.format(kk))
            logger_all.info('*' * (len(net_file) + 10))

            for f in test_set:
                np.random.seed(seed=0)  # for reproducibility
                # Open image
                img_clean = imread_uint8(f, n_channels=num_channels)  # HR image, int8,RGB
                img_clean = img_clean / 255.
                # generate degraded LR image
                blur_kernel = blur_kernels[0, kk].astype(np.float64)
                blur_kernel /= np.sum(blur_kernel)
                img_blur = ndimage.filters.convolve(img_clean, np.expand_dims(blur_kernel, axis=2), mode='wrap')  # blur
                # -----------------------------------------------------------------
                # add A_W_G_N noise
                # -----------------------------------------------------------------
                test_noise_level = self.params.test_noiseL / 255.0
                img_noise = img_blur + np.random.normal(0.0, test_noise_level, img_blur.shape)
                h, w = img_noise.shape[:2]
                # expand dim
                img_clean_ = torch.from_numpy(img_clean).float().permute(2, 0, 1).unsqueeze(dim=0)
                img_noise_ = torch.from_numpy(img_noise).float().permute(2, 0, 1).unsqueeze(dim=0)
                blur_kernel_ = torch.from_numpy(np.array(blur_kernel[np.newaxis, ...])).float().unsqueeze(dim=0)

                if torch.cuda.is_available():
                    img_clean_, img_noise_, blur_kernel_ = \
                        [ii.cuda() for ii in [img_clean_, img_noise_, blur_kernel_]]

                # Test
                with torch.no_grad():
                    img_denoised = torch.clamp(model(img_noise_, blur_kernel_), 0., 1.)

                # resize to original
                img_denoised = img_denoised[..., :h, :w]
                img_noise_ = img_noise_[..., :h, :w]

                if self.params.is_gray_scale:
                    logger.info("*** Gray-scale debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))
                    img_denoised = (img_denoised[:, 0, :, :] * 0.299 + img_denoised[:, 1, :, :] * 0.587 + img_denoised[:, 2, :,:] * 0.114).unsqueeze(dim=0)
                    img_clean_ = (img_clean_[:, 0, :, :] * 0.299 + img_clean_[:, 1, :, :] * 0.587 + img_clean_[:, 2,:,:] * 0.114).unsqueeze(dim=0)
                    img_noise_ = (img_noise_[:, 0, :, :] * 0.299 + img_noise_[:, 1, :, :] * 0.587 + img_noise_[:, 2,:,:] * 0.114).unsqueeze(dim=0)
                else:
                    logger.info("*** sRGB debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))

                psnr = batch_psnr(img_denoised, img_clean_, 1.)
                ssim = batch_ssim(img_denoised, img_clean_)
                psnr_noisy = batch_psnr(img_noise_, img_clean_, 1.)
                ssim_noisy = batch_ssim(img_noise_, img_clean_)
                logger.info("\tPSNR deblurred {0:0.2f} dB".format(psnr))
                logger.info("\tSSIM {0:0.4f}".format(ssim))
                logger.info("\tPSNR noisy blurry {0:0.2f} dB".format(psnr_noisy))
                logger.info("\tSSIM noisy blurry {0:0.4f}".format(ssim_noisy))
                logger.info("-" * 60)
                logger.info("\n")
                # Save images
                img_noise_ = variable_to_cv2_image(img_noise_)
                img_denoised = variable_to_cv2_image(img_denoised)
                file_name = extract_file_name(f)
                file_name_noisy = '{}_{}_nb.png'.format(concreate_dir, file_name)
                file_name_denoised = '{}_{}_{}_deblurred.png'.format(concreate_dir,
                                                                    file_name,
                                                                    self.params.model_name)

                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_noisy), img_noise_)
                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_denoised), img_denoised)

                # Sum all psnr and compute avg
                psnr_test += psnr
                ssim_test += ssim
                print("%s PSNR %0.2f SSIM %0.4f" % (f, psnr, ssim))

            psnr_test /= len(test_set)
            ssim_test /= len(test_set)
            print("\nPSNR on test data {0:0.2f}dB".format(psnr_test))
            print("SSIM on test data {0:0.4f}".format(ssim_test))
            logger.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger.info("SSIM on test data {0:0.4f}\n".format(ssim_test))
            logger_all.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger_all.info("SSIM on test data {0:0.4f}".format(ssim_test))

            # Statistic all psnr and ssim
            all_psnr_test += psnr_test
            all_ssim_test += ssim_test

            # Reset psnr_test and ssim_test
            psnr_test = 0
            ssim_test = 0

        # Calculate average psnr and ssim for all kernels
        all_psnr_test /= blur_kernels.size
        all_ssim_test /= blur_kernels.size
        print('Average PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        print('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))
        logger_all.info('\nAverage PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        logger_all.info('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))

    def test_real(self, img_deblur_handle):
        print('testing on {} data ...\n'.format(self.params.test_data))
        # -------------------------------------------------------
        # 1) load test dataset
        # -------------------------------------------------------
        test_set = img_deblur_handle.test_set
        # -------------------------------------------------------
        # 2) Absolute path to model file
        # -------------------------------------------------------
        net_file = self.get_net_file()
        # -------------------------------------------------------
        # 3) Create model and Load saved weights
        # -------------------------------------------------------
        num_channels = self.params.num_channels
        model = self.get_model(net_file, num_channels)
        # --------------------------------------------------------
        # load kernel shape: (1, n)
        # --------------------------------------------------------
        blur_kernels = img_deblur_handle.kernels
        # --------------------------------------------------------
        # 4) main test for specific kernel
        # --------------------------------------------------------
        # make save directory
        make_dirs(os.path.join('results', self.params.test_data, 'realDeblur'))

        with tqdm(total=len(test_set), desc='De-blurring', ncols=100) as bar:
            for image_name in range(len(test_set)):
                bar.update(1)
                # Open image
                f = test_set[image_name]
                img_blur = imread_uint8(f, n_channels=num_channels)  # HR image, int8,RGB
                img_blur = img_blur / 255.

                # blur kernel
                blur_kernel = cv2.imread(blur_kernels[image_name], 0)  # cv2.IMREAD_GRAYSCALE
                blur_kernel = (blur_kernel / 255.).astype(np.float64)
                blur_kernel /= np.sum(blur_kernel)

                h, w = img_blur.shape[:2]

                # expand dim
                img_blur_ = torch.from_numpy(img_blur).float().permute(2, 0, 1).unsqueeze(dim=0)
                blur_kernel_ = torch.from_numpy(np.array(blur_kernel[np.newaxis, ...])).float().unsqueeze(dim=0)

                if torch.cuda.is_available():
                    img_blur_, blur_kernel_ = [ii.cuda() for ii in [img_blur_, blur_kernel_]]

                # Test
                with torch.no_grad():
                    img_deblurred = torch.clamp(model(img_blur_, blur_kernel_), 0., 1.)

                # resize to original
                img_deblurred = img_deblurred[..., :h, :w]
                # Save images
                img_deblurred = variable_to_cv2_image(img_deblurred)
                file_name = extract_file_name(f)
                file_name_deblurred = '{}_{}_deblurred.png'.format(file_name, self.params.model_name)

                cv2.imwrite(os.path.join('results', self.params.test_data,
                                         'realDeblur',
                                         file_name_deblurred), img_deblurred)

    def test_nonuniform(self, img_deblur_handle):
        print('testing on {} data ...\n'.format(self.params.test_data))
        # ------------------------------------------------------
        # 1) Initial
        # ------------------------------------------------------
        psnr_test = 0
        ssim_test = 0
        all_psnr_test = 0
        all_ssim_test = 0
        # -------------------------------------------------------
        # 2) load test dataset
        # -------------------------------------------------------
        test_set = img_deblur_handle.test_set
        # -------------------------------------------------------
        # 3) Absolute path to model file
        # -------------------------------------------------------
        net_file = self.get_net_file()
        # -------------------------------------------------------
        # 4) Create model and Load saved weights
        # -------------------------------------------------------
        num_channels = self.params.num_channels
        model = self.get_model(net_file, num_channels)
        # --------------------------------------------------------
        # load kernel shape: (1, n)
        # --------------------------------------------------------
        blur_kernels = img_deblur_handle.kernels
        concreate_dir = ''.join([self.params.test_data, '_', str(self.params.sigma_map_type)])
        make_dirs(os.path.join('results', concreate_dir))
        log_all_psnr_ssim_name = os.path.join('results', concreate_dir, 'logger_summary_PSNR_SSIM_' +
                                              self.params.model_name.split('.')[0] + '_'
                                              + str(self.params.sigma_map_av) + '_'
                                              + str(self.params.sigma_map_index) + '.txt')
        logger_all = init_logger_ipol(file_name=log_all_psnr_ssim_name, obj_name='all')
        logger_all.info('Testing on {}\'s dataset'.format(self.params.test_data))

        for kk in range(blur_kernels.size):
            # make save directory
            make_dirs(os.path.join('results', concreate_dir, ''.join(['kernel', str(kk)])))
            # Init logger
            # Single kernel logger
            log_file_name = os.path.join('results',
                                         concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         str(kk) + '_logger_' + self.params.model_name.split('.')[0] + '_' + \
                                         str(self.params.sigma_map_av) + '_' + \
                                         str(self.params.sigma_map_index) + '.txt')
            logger = init_logger_ipol(file_name=log_file_name, obj_name=''.join(['avg', str(kk)]))

            print('*' * (len(net_file) + 10))
            print('Testing on {}th kernel'.format(kk))
            print('*' * (len(net_file) + 10))

            logger.info("\n")
            logger.info('*' * (len(net_file) + 10))
            logger.info('Testing on {}th kernel'.format(kk))
            logger.info('*' * (len(net_file) + 10))
            logger_all.info("\n")
            logger_all.info('*' * (len(net_file) + 10))
            logger_all.info('Testing on {}th kernel'.format(kk))
            logger_all.info('*' * (len(net_file) + 10))

            for f in test_set:
                np.random.seed(seed=0)  # for reproducibility
                # Open image
                img_clean = imread_uint8(f, n_channels=num_channels)  # HR image, int8,RGB
                img_clean = img_clean / 255.
                # generate degraded LR image
                blur_kernel = blur_kernels[0, kk].astype(np.float64)
                blur_kernel /= np.sum(blur_kernel)
                img_blur = ndimage.filters.convolve(img_clean, np.expand_dims(blur_kernel, axis=2), mode='wrap')  # blur
                # -----------------------------------------------------------------
                # Add non-uniform Gaussian noise
                # -----------------------------------------------------------------
                sigma_map = None
                if self.params.sigma_map_type == 'peaks':
                    sigma_map = peaks(256)
                elif self.params.sigma_map_type == 'gauss':
                    sigma_map = sincos_kernel()
                elif self.params.sigma_map_type == 'gauss_mix':
                    sigma_map = generate_gauss_kernel_mix(256, 256)

                sig_av = 5
                sigma_map = sig_av / 255.0 + (sigma_map - sigma_map.min()) / (sigma_map.max() - sigma_map.min()) * (
                            (25 - sig_av) / 255.0)

                h, w, c = img_blur.shape
                sigma_map = cv2.resize(sigma_map, (w, h))
                noise = np.random.randn(h, w, c) * sigma_map[:, :, np.newaxis]
                img_noise = img_blur + noise
                # expand dim
                img_clean_ = torch.from_numpy(img_clean).float().permute(2, 0, 1).unsqueeze(dim=0)
                img_noise_ = torch.from_numpy(img_noise).float().permute(2, 0, 1).unsqueeze(dim=0)
                blur_kernel_ = torch.from_numpy(np.array(blur_kernel[np.newaxis, ...])).float().unsqueeze(dim=0)

                if torch.cuda.is_available():
                    img_clean_, img_noise_, blur_kernel_ = \
                        [ii.cuda() for ii in [img_clean_, img_noise_, blur_kernel_]]

                # Test
                with torch.no_grad():
                    img_denoised = torch.clamp(model(img_noise_, blur_kernel_), 0., 1.)

                # resize to original
                img_denoised = img_denoised[..., :h, :w]
                img_noise_ = img_noise_[..., :h, :w]

                if self.params.is_gray_scale:
                    logger.info("*** Gray-scale debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))
                    img_denoised = (img_denoised[:, 0, :, :] * 0.299 + img_denoised[:, 1, :, :] * 0.587 + img_denoised[:, 2,:,:] * 0.114).unsqueeze(dim=0)
                    img_clean_ = (img_clean_[:, 0, :, :] * 0.299 + img_clean_[:, 1, :, :] * 0.587 + img_clean_[:, 2, :,:] * 0.114).unsqueeze(dim=0)
                    img_noise_ = (img_noise_[:, 0, :, :] * 0.299 + img_noise_[:, 1, :, :] * 0.587 + img_noise_[:, 2, :,:] * 0.114).unsqueeze(dim=0)
                else:
                    logger.info("*** sRGB debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))

                psnr = batch_psnr(img_denoised, img_clean_, 1.)
                ssim = batch_ssim(img_denoised, img_clean_)
                psnr_noisy = batch_psnr(img_noise_, img_clean_, 1.)
                ssim_noisy = batch_ssim(img_noise_, img_clean_)
                logger.info("\tPSNR deblurred {0:0.2f} dB".format(psnr))
                logger.info("\tSSIM {0:0.4f}".format(ssim))
                logger.info("\tPSNR noisy blurry {0:0.2f} dB".format(psnr_noisy))
                logger.info("\tSSIM noisy blurry {0:0.4f}".format(ssim_noisy))
                logger.info("-" * 60)
                logger.info("\n")

                # Save images
                img_noise_ = variable_to_cv2_image(img_noise_)
                img_denoised = variable_to_cv2_image(img_denoised)
                file_name = extract_file_name(f)
                file_name_noisy = '{}_{}_nonuniform_av{}_index{}_nb.png'.format(concreate_dir,
                                                                                   file_name,
                                                                                   self.params.sigma_map_av,
                                                                                   self.params.sigma_map_index)
                file_name_denoised = '{}_{}_{}_nonuniform_av{}_index{}_deblurred.png'.format(concreate_dir,
                                                                                            file_name,
                                                                                            self.params.model_name,
                                                                                            self.params.sigma_map_av,
                                                                                            self.params.sigma_map_index)

                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_noisy), img_noise_)
                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_denoised), img_denoised)

                # Sum all psnr and compute avg
                psnr_test += psnr
                ssim_test += ssim
                print("%s PSNR %0.2f SSIM %0.4f" % (f, psnr, ssim))

            psnr_test /= len(test_set)
            ssim_test /= len(test_set)
            print("\nPSNR on test data {0:0.2f}dB".format(psnr_test))
            print("SSIM on test data {0:0.4f}".format(ssim_test))
            logger.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger.info("SSIM on test data {0:0.4f}\n".format(ssim_test))
            logger_all.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger_all.info("SSIM on test data {0:0.4f}".format(ssim_test))

            # Statistic all psnr and ssim
            all_psnr_test += psnr_test
            all_ssim_test += ssim_test

            # Reset psnr_test and ssim_test
            psnr_test = 0
            ssim_test = 0

        # Calculate average psnr and ssim for all kernels
        all_psnr_test /= blur_kernels.size
        all_ssim_test /= blur_kernels.size
        print('Average PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        print('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))
        logger_all.info('\nAverage PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        logger_all.info('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))

    def test_jpeg_compression(self, img_deblur_handle):
        print('testing on {} data ...\n'.format(self.params.test_data))
        print('jpeg_quality_factor...{}'.format(self.params.jpeg_quality_factor))
        # ------------------------------------------------------
        # 1) Initial
        # ------------------------------------------------------
        psnr_test = 0
        ssim_test = 0
        all_psnr_test = 0
        all_ssim_test = 0
        # -------------------------------------------------------
        # 2) load test dataset
        # -------------------------------------------------------
        test_set = img_deblur_handle.test_set
        # -------------------------------------------------------
        # 3) Absolute path to model file
        # -------------------------------------------------------
        net_file = self.get_net_file()
        # -------------------------------------------------------
        # 4) Create model and Load saved weights
        # -------------------------------------------------------
        num_channels = self.params.num_channels
        model = self.get_model(net_file, num_channels)
        # --------------------------------------------------------
        # load kernel shape: (1, n)
        # --------------------------------------------------------
        blur_kernels = img_deblur_handle.kernels

        concreate_dir = ''.join([self.params.test_data, '_', str(self.params.jpeg_quality_factor)])
        make_dirs(os.path.join('results', concreate_dir))

        log_all_psnr_ssim_name = os.path.join('results', concreate_dir,
                                              'logger_summary_PSNR_SSIM_' + self.params.model_name.split('.')[
                                                  0] + '.txt')
        logger_all = init_logger_ipol(file_name=log_all_psnr_ssim_name, obj_name='all')
        logger_all.info('Testing on {}\'s dataset'.format(self.params.test_data))

        for kk in range(blur_kernels.size):
            # make save directory
            make_dirs(os.path.join('results', concreate_dir, ''.join(['kernel', str(kk)])))
            # Init logger
            # Single kernel logger
            log_file_name = os.path.join('results',
                                         concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         str(kk) + '_logger_' + self.params.model_name.split('.')[0] + '.txt')
            logger = init_logger_ipol(file_name=log_file_name, obj_name=''.join(['avg', str(kk)]))

            print('*' * (len(net_file) + 10))
            print('Testing on {}th kernel'.format(kk))
            print('*' * (len(net_file) + 10))

            logger.info("\n")
            logger.info('*' * (len(net_file) + 10))
            logger.info('Testing on {}th kernel'.format(kk))
            logger.info('*' * (len(net_file) + 10))
            logger_all.info("\n")
            logger_all.info('*' * (len(net_file) + 10))
            logger_all.info('Testing on {}th kernel'.format(kk))
            logger_all.info('*' * (len(net_file) + 10))

            for f in test_set:
                np.random.seed(seed=0)  # for reproducibility
                # Open image
                img_clean = imread_uint8(f, n_channels=num_channels)  # HR image, int8,RGB
                img_clean = img_clean / 255.

                # generate degraded LR image
                blur_kernel = blur_kernels[0, kk].astype(np.float64)
                blur_kernel /= np.sum(blur_kernel)
                img_blur = ndimage.filters.convolve(img_clean, np.expand_dims(blur_kernel, axis=2), mode='wrap')  # blur
                test_noise_level = self.params.test_noiseL / 255.0
                img_noise = img_blur + np.random.normal(0.0, test_noise_level, img_blur.shape)
                # -----------------------------------------------------------------
                # JPEG Compression with quality_factor (50, 70, 90)
                # -----------------------------------------------------------------
                _, encimg = cv2.imencode('.jpg', 255.0*img_noise, [int(cv2.IMWRITE_JPEG_QUALITY), self.params.jpeg_quality_factor])
                img_noise = cv2.imdecode(encimg, 3)

                h, w = img_noise.shape[:2]
                img_noise = img_noise / 255.
                # expand dim
                img_clean_ = torch.from_numpy(img_clean).float().permute(2, 0, 1).unsqueeze(dim=0)
                img_noise_ = torch.from_numpy(img_noise).float().permute(2, 0, 1).unsqueeze(dim=0)
                blur_kernel_ = torch.from_numpy(np.array(blur_kernel[np.newaxis, ...])).float().unsqueeze(dim=0)

                if torch.cuda.is_available():
                    img_clean_, img_noise_, blur_kernel_ = \
                        [ii.cuda() for ii in [img_clean_, img_noise_, blur_kernel_]]

                # Test
                with torch.no_grad():
                    img_denoised = torch.clamp(model(img_noise_, blur_kernel_), 0., 1.)

                # resize to original
                img_denoised = img_denoised[..., :h, :w]
                img_noise_ = img_noise_[..., :h, :w]

                if self.params.is_gray_scale:
                    logger.info("*** Gray-scale debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))
                    img_denoised = (img_denoised[:, 0, :, :] * 0.299 + img_denoised[:, 1, :, :] * 0.587 + img_denoised[:, 2, :, :] * 0.114).unsqueeze(dim=0)
                    img_clean_ = (img_clean_[:, 0, :, :] * 0.299 + img_clean_[:, 1, :, :] * 0.587 + img_clean_[:, 2, :, :] * 0.114).unsqueeze(dim=0)
                    img_noise_ = (img_noise_[:, 0, :, :] * 0.299 + img_noise_[:, 1, :, :] * 0.587 + img_noise_[:, 2, :, :] * 0.114).unsqueeze(dim=0)
                else:
                    logger.info("*** sRGB debluring ***")
                    logger.info("{} with {} noise level".format(f, self.params.test_noiseL))

                psnr = batch_psnr(img_denoised, img_clean_, 1.)
                ssim = batch_ssim(img_denoised, img_clean_)
                psnr_noisy = batch_psnr(img_noise_, img_clean_, 1.)
                ssim_noisy = batch_ssim(img_noise_, img_clean_)
                logger.info("\tPSNR deblurred {0:0.2f} dB".format(psnr))
                logger.info("\tSSIM {0:0.4f}".format(ssim))
                logger.info("\tPSNR noisy blurry {0:0.2f} dB".format(psnr_noisy))
                logger.info("\tSSIM noisy blurry {0:0.4f}".format(ssim_noisy))
                logger.info("-" * 60)
                logger.info("\n")

                # Save images
                img_noise_ = variable_to_cv2_image(img_noise_)
                img_denoised = variable_to_cv2_image(img_denoised)

                file_name = extract_file_name(f)
                file_name_noisy = '{}_{}_nb.png'.format(concreate_dir, file_name)
                file_name_denoised = '{}_{}_{}_deblurred.png'.format(concreate_dir,
                                                                    file_name,
                                                                    self.params.model_name)

                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_noisy), img_noise_)
                cv2.imwrite(os.path.join('results', concreate_dir,
                                         ''.join(['kernel', str(kk)]),
                                         file_name_denoised), img_denoised)

                # Sum all psnr and compute avg
                psnr_test += psnr
                ssim_test += ssim
                print("%s PSNR %0.2f SSIM %0.4f" % (f, psnr, ssim))

            psnr_test /= len(test_set)
            ssim_test /= len(test_set)
            print("\nPSNR on test data {0:0.2f}dB".format(psnr_test))
            print("SSIM on test data {0:0.4f}".format(ssim_test))
            logger.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger.info("SSIM on test data {0:0.4f}\n".format(ssim_test))
            logger_all.info("PSNR on test data {0:0.2f}dB".format(psnr_test))
            logger_all.info("SSIM on test data {0:0.4f}".format(ssim_test))

            # Statistic all psnr and ssim
            all_psnr_test += psnr_test
            all_ssim_test += ssim_test
            # Reset psnr_test and ssim_test
            psnr_test = 0
            ssim_test = 0

        # Calculate average psnr and ssim for all kernels
        all_psnr_test /= blur_kernels.size
        all_ssim_test /= blur_kernels.size
        print('Average PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        print('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))
        logger_all.info('\nAverage PSNR on all kernels {0:0.2f}dB'.format(all_psnr_test))
        logger_all.info('Average SSIM on all kernels {0:0.4f}'.format(all_ssim_test))


    def get_net(self, num_channels):
        _, net = return_model_info(model_select_name=self.params.model_select, num_channels=num_channels)
        return net

    def get_net_file(self):
        file_name = str(self.params.model_name)
        net_file = os.path.join(self.params.model_save_file, ''.join(file_name))
        print('-' * (len(net_file) + 10))
        print('net file: {}'.format(net_file))
        print('-' * (len(net_file) + 10))

        return net_file

    def get_model(self, net_file, num_channels):
        net = self.get_net(num_channels)
        device_ids = [0]
        # -------------------------------------------------------
        # Load saved weights
        # -------------------------------------------------------
        if torch.cuda.is_available():
            state_dict = torch.load(net_file)
            model = nn.DataParallel(net, device_ids=device_ids).cuda()
        else:
            state_dict = torch.load(net_file, map_location=torch.device('cpu'))
            # CPU mode: remove the DataParallel wrapper
            state_dict = remove_data_parallel_wrapper(state_dict)
            model = net

        model.load_state_dict(state_dict)
        model.eval()

        return model
