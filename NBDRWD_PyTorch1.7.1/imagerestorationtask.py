# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

from abc import ABC, abstractmethod


class ImageRestorationBase(ABC):
    """
    The main interface for the image restoration task
    """

    def __init__(self, train_set, validate_set, test_set):
        self.__train_set = train_set
        self.__validate_set = validate_set
        self.__test_set = test_set
        self.__algorithm_handle = None

    @property
    def train_set(self):
        return self.__train_set

    @train_set.setter
    def train_set(self, val):
        self.__train_set = val

    @property
    def validate_set(self):
        return self.__validate_set

    @validate_set.setter
    def validate_set(self, val):
        self.__validate_set = val

    @property
    def test_set(self):
        return self.__test_set

    @test_set.setter
    def test_set(self, val):
        self.__test_set = val

    @property
    def algorithm_handle(self):
        return self.__algorithm_handle

    @algorithm_handle.setter
    def algorithm_handle(self, val):
        self.__algorithm_handle = val

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass

    @abstractmethod
    def test_real(self):
        pass

    @abstractmethod
    def test_nonuniform(self):
        pass

    @abstractmethod
    def test_jpeg_compression(self):
        pass


class ImageNonBlindDeblurring(ImageRestorationBase):
    """
    Non-blind Image Deblurring
    """

    def __init__(self, train_set, validate_set, test_set, kernels, has_noise_level_map=False):
        super().__init__(train_set, validate_set, test_set)
        self.kernels = kernels
        self.has_noise_level_map = has_noise_level_map

    def train(self):
        """
        Training a network
        :return:
        """
        self.algorithm_handle.train(self)

    def test(self):
        """
        Test a trained network for a blurred image in the presence of uniform Gaussian noise.
        :return:
        """
        self.algorithm_handle.test(self)

    def test_real(self):
        """
        Test a pre-trained network for a real-world blurred image.
        :return:
        """
        self.algorithm_handle.test_real(self)

    def test_nonuniform(self):
        """
        Test a trained network for a blurred image in the presence of non-uniform noise.
        :return:
        """
        self.algorithm_handle.test_nonuniform(self)

    def test_jpeg_compression(self):
        """
        Test a trained network for a blurred image in the presence of
         Gaussian noise and JPEG Compression.
        :return:
        """
        self.algorithm_handle.test_jpeg_compression(self)
