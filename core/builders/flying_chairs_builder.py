import os

import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from config import PATH_TO_IMAGES, NUMBER_OF_SAMPLES_TO_LOAD
from core.builders.data_augmentation import Compose, RandomCrop, RandomTranslate, RandomRotate, RandomHorizontalFlip, \
    RandomVerticalFlip, Normalize
from utils.utils import  read_flo_file,read_image


class FlyingChairsDataGenerator(Sequence):
    def _set_validation_parameters(self) -> None:
        """
        function sets number of samples, indexes and transformations based on type of data generation
        validation = True
        training = False
        """
        validation_split_idx = int(NUMBER_OF_SAMPLES_TO_LOAD * 0.9) - int(NUMBER_OF_SAMPLES_TO_LOAD * 0.9) % 8
        if self.validation:
            self.number_of_samples = int(NUMBER_OF_SAMPLES_TO_LOAD * 0.1) - int(NUMBER_OF_SAMPLES_TO_LOAD * 0.9) % 8
            self.files_indexes = np.arange(validation_split_idx + 1, NUMBER_OF_SAMPLES_TO_LOAD + 1)
            self.both_transform = Compose([RandomCrop(373, 501)])
        else:
            self.number_of_samples = int(NUMBER_OF_SAMPLES_TO_LOAD * 0.9) - int(NUMBER_OF_SAMPLES_TO_LOAD * 0.9) % 8
            self.files_indexes = np.arange(1, validation_split_idx + 1)
            self.both_transform = Compose([
                RandomTranslate(10, 10),
                RandomRotate(10, 5),
                RandomCrop(373, 501),
                RandomVerticalFlip(),
                RandomHorizontalFlip()
            ])

    def __init__(self, batch_size: int, validation=False) -> None:
        """
        :param batch_size: number of samples per step
        :param validation: bool if this data loader is for validation set or training set
        """
        self.batch_size = batch_size
        self.flow_file_name = "{:05d}_flow.flo"
        self.first_img_name = "{:05d}_img1.ppm"
        self.second_img_name = "{:05d}_img2.ppm"
        self.files_indexes = []
        self.validation = validation
        self.image_transform = [
            Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            Normalize(mean=[0.45, 0.432, 0.411], std=[1, 1, 1])
        ]
        self.flow_transform = [
            Normalize(mean=[0, 0], std=[20, 20])
        ]
        self._set_validation_parameters()
        self.on_epoch_end()

    def __len__(self)->int:
        """return how many steps are in data loader, step = samples/batch_size"""
        return int(np.ceil(self.number_of_samples / self.batch_size))

    def on_epoch_end(self) -> none:
        """
        On end of each epoch we shuffle all samples
        """
        np.random.shuffle(self.files_indexes)

    def __getitem__(self, index):
        """

        :param index:
        :return:
        """
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        flows = []
        images = []

        for i in range(start_index, end_index):
            # get index from shuffled indexes array
            files_index = self.files_indexes[i]
            with open(os.path.join(PATH_TO_IMAGES, self.flow_file_name.format(files_index)), 'rb') as f:
                header = f.read(4)
                if header.decode("utf-8") != 'PIEH':
                    raise Exception('Flow file header does not contain PIEH')

                width = np.fromfile(f, np.int32, 1).squeeze()
                height = np.fromfile(f, np.int32, 1).squeeze()

                flow = np.fromfile(f, np.float32, int(width) * int(height) * 2).reshape((int(height), int(width), 2))

            img_pair = np.array(np.array(Image.open(os.path.join(PATH_TO_IMAGES
                                                                 , self.first_img_name.format(files_index)))).astype(
                np.float32),
                np.array(Image.open(os.path.join(PATH_TO_IMAGES
                                                 , self.second_img_name.format(files_index)))).astype(
                    np.float32))

            img_pair, flow = self.both_transform(img_pair, flow)

            for transformation in self.image_transform:
                img_pair[0] = transformation(img_pair[0])
                img_pair[1] = transformation(img_pair[1])

            for transformation in self.flow_transform:
                flow = transformation(flow)
            img_pair[0] = img_pair[0][:373, :501, :]
            img_pair[1] = img_pair[1][:373, :501, :]
            flow = flow[:373, :501, :]
            flows.append(flow)
            images.append(np.concatenate([img_pair[0], img_pair[1]], axis=-1))
        return np.array(images), np.array(flows)
