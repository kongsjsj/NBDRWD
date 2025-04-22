# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

import os
import argparse
import json
from time import perf_counter
from torch.utils.data import DataLoader
from scipy.io import loadmat

import torch
from data.datasetnbdrwd import (DatasetNBDRWD, DatasetNBDRWDJPEGCompression)
from imagerestorationtask import ImageNonBlindDeblurring
from algorithms import AlgorithmNonBlindDeblurring
from utils.othertools import make_dirs, set_random_seed
from utils import get_image_paths
from models.modeloption import return_model_info


def read_json_file(args):
    with open(args.train_config, 'r') as f:
        train_config = json.load(f)

    with open(args.test_config, 'r') as f:
        test_config = json.load(f)

    model_name, _ = return_model_info(model_select_name=train_config['model']['model_select'])
    # Train Params set
    params = argparse.Namespace
    params.model_name = model_name
    params.model_save_file = train_config['model']['model_save_file']
    params.model_select = train_config['model']['model_select']
    params.num_workers = train_config['train_dataset']['num_workers'],
    params.batch_size = train_config['train_dataset']['batch_size'],
    params.is_train = train_config['train_dataset']['is_train']
    params.seed = train_config['train_options']['seed']
    params.root_dir_train = train_config['train_dataset']['root_path']
    params.root_dir_val = train_config['val_dataset']['root_path']
    params.kernels_val = loadmat(train_config['val_dataset']['kernel_path'])['kernels']  # for validation
    params.num_channels = train_config['train_dataset']['num_channels']
    params.patch_size = train_config['train_dataset']['patch_size']
    params.sigma_max = train_config['train_dataset']['sigma_max']
    params.val_noise_level = train_config['val_dataset']['noise_level']
    params.kernel_index = train_config['val_dataset']['kernel_index']
    params.device = None
    if torch.cuda.is_available():
        params.device = torch.device('cuda')
    else:
        params.device = torch.device('cpu')
    params.device_ids = train_config['train_options']['device_ids']
    params.lr = train_config['optimizer']['lr']
    params.resume = train_config['train_options']['resume']
    params.is_train_gauss = train_config['train_options']['is_train_gauss']
    params.is_train_jpeg_compression = train_config['train_options']['is_train_jpeg_compression']
    params.resume_model_name = train_config['train_options']['resume_model_name']
    params.epochs = train_config['train_options']['epochs']
    params.milestone = train_config['multi_step_lr']['milestone']

    # Test Params set
    params.root_dir_test = test_config['test_dataset']['root_path']
    params.test_data = test_config['test_dataset']['test_data']
    params.test_kernel = test_config['test_dataset']['kernel_path']
    params.test_noiseL = test_config['test_dataset']['test_noise_level']
    params.is_gray_scale = test_config['test_dataset']['is_gray_scale']
    params.is_test_real = test_config['test_dataset']['is_test_real']
    params.is_test_nonuniform = test_config['test_dataset']['is_test_nonuniform']
    params.is_test_gauss = test_config['test_dataset']['is_test_gauss']
    params.is_test_jpeg_compression = test_config['test_dataset']['is_test_jpeg_compression']

    if params.is_test_nonuniform:
        params.sigma_map_type = test_config['test_dataset']['sigma_map_type']  # gauss | gauss_mix | peaks
        params.sigma_map_path = test_config['test_dataset']['sigma_map_path']
        params.sigma_map_av = test_config['test_dataset']['sigma_map_av']
        params.sigma_map_index = test_config['test_dataset']['sigma_map_index']
        params.test_noiseL = 0

    if params.is_test_jpeg_compression:
        params.jpeg_quality_factor = test_config['test_dataset']['jpeg_quality_factor']

    return params


class DatasetFactory:

    def __init__(self):
        pass

    @staticmethod
    def create_train_dataset(params):
        loader_train = None
        dataset_train = None
        # ---------------------------------------------------
        # load train dataset
        # ---------------------------------------------------
        if params.is_train_gauss:
            dataset_train = DatasetNBDRWD(
                root_dir=params.root_dir_train,
                patch_size=params.patch_size,
                num_channels=params.num_channels,
                sigma_max=params.sigma_max,
                nl=params.val_noise_level,
                is_train=params.is_train
            )
            loader_train = DataLoader(
                dataset=dataset_train,
                num_workers=params.num_workers[0],
                batch_size=params.batch_size[0],
                shuffle=True,
                drop_last=True,
                pin_memory=True
            )
            # print('Current training is: Gaussian noise removal!')

        elif params.is_train_jpeg_compression:
            dataset_train = DatasetNBDRWDJPEGCompression(
                root_dir=params.root_dir_train,
                patch_size=params.patch_size,
                num_channels=params.num_channels,
                is_train=params.is_train
            )
            loader_train = DataLoader(
                dataset=dataset_train,
                num_workers=params.num_workers[0],
                batch_size=params.batch_size[0],
                shuffle=True,
                drop_last=True,
                pin_memory=True
            )
            # print('Current training is: JPEG_Compression!')
        else:
            print('Load train dataset failed: Not implementation!')

        print("# of training samples: %d\n" % int(len(dataset_train)))

        return loader_train

    @staticmethod
    def create_val_dataset(params):
        dataset_validation = None
        # ---------------------------------------------------
        # # load validation dataset
        # ---------------------------------------------------
        if params.is_train_gauss:
            dataset_validation = DatasetNBDRWD(
                root_dir=params.root_dir_val,
                kernels=params.kernels_val,
                num_channels=params.num_channels,
                nl=params.val_noise_level,
                kernel_index=params.kernel_index,
                is_train=False
            )

        elif params.is_train_jpeg_compression:
            dataset_validation = DatasetNBDRWDJPEGCompression(
                root_dir=params.root_dir_val,
                kernels=params.kernels_val,
                num_channels=params.num_channels,
                kernel_index=params.kernel_index,
                is_train=False
            )
        else:
            print('Load validation dataset failed: Not implementation!')

        return dataset_validation


def main(params):
    make_dirs(params.model_save_file)
    set_random_seed(params.seed)

    dataset_fac = DatasetFactory()
    # ---------------------------------------------------
    # load train dataset
    # ---------------------------------------------------
    loader_train = dataset_fac.create_train_dataset(params)

    # ---------------------------------------------------
    # # load validation dataset
    # ---------------------------------------------------
    dataset_validation = dataset_fac.create_val_dataset(params)

    # ---------------------------------------------------
    # load test dataset
    # ---------------------------------------------------
    root_dir_test = params.root_dir_test
    test_data = params.test_data
    dataset_test = get_image_paths(os.path.join(root_dir_test, test_data))
    dataset_test.sort()

    # ---------------------------------------------------
    # load test kernels
    # ---------------------------------------------------
    if not params.is_test_real:
        kernels = loadmat(params.test_kernel)['kernels']
    else:
        kernels = get_image_paths(params.test_kernel)
        kernels.sort()
    # ---------------------------------------------------
    # Construct the object of ImageNonBlindDeblurring
    # ---------------------------------------------------
    img_deblur = ImageNonBlindDeblurring(loader_train,
                                         dataset_validation,
                                         dataset_test,
                                         kernels)

    # ---------------------------------------------------
    # Assign algorithm
    # ---------------------------------------------------
    img_deblur.algorithm_handle = AlgorithmNonBlindDeblurring(params)

    if params.is_train:
        # Train by specific algorithm
        start_time = perf_counter()  # tic
        img_deblur.train()
        end_time = perf_counter()  # toc
        all_time = end_time - start_time
        print("Training time was: {} hours".format(all_time / 3600))
    else:
        # Test by the trained network
        start_time = perf_counter()  # tic
        if not params.is_test_real:
            if not params.is_test_nonuniform:
                if params.is_test_gauss:
                    img_deblur.test()
                elif params.is_test_jpeg_compression:
                    img_deblur.test_jpeg_compression()
                else:
                    print('Not implementation...')
            else:
                img_deblur.test_nonuniform()
        else:
            img_deblur.test_real()

        end_time = perf_counter()  # toc
        all_time = end_time - start_time
        print("Test time was: {} seconds".format(all_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_config', default='./configs/test_realworld.json')
    parser.add_argument('--train_config', default='./configs/traincolor.json')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    main(read_json_file(args))
