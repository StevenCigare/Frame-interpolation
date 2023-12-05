import os

import numpy as np
from PIL import Image
from core.builders.data_augmentation import SequentialDataTransform, RandomCrop


class ValidationDataLoader:
    def __init__(self, number_of_samples_to_load: int, path_to_images: str):
        self.path_to_images = path_to_images
        self.flow_file_name = "{:05d}_flow.flo"
        self.first_img_name = "{:05d}_img1.ppm"
        self.second_img_name = "{:05d}_img2.ppm"
        validation_split_idx = int(number_of_samples_to_load * 0.9) - int(number_of_samples_to_load * 0.9) % 8
        self.files_indexes = np.arange(validation_split_idx + 1, number_of_samples_to_load + 1)

        self.crop = SequentialDataTransform([RandomCrop((373, 501))])

    def load_data(self):
        flows = []
        images = []

        for idx in self.files_indexes:
            flow = self._read_flow_data(idx)
            img_pair = self._read_ppm_images(idx)

            img_pair_cropped, flow_cropped = self.crop(img_pair, flow)

            flows.append(flow_cropped)
            images.append(np.concatenate([img_pair_cropped[0], img_pair_cropped[1]], axis=-1))
        return np.array(images), np.array(flows)

    def _read_flow_data(self, files_index: int):
        path = os.path.join(self.path_to_images, self.flow_file_name.format(files_index))
        with open(path, 'rb') as f:
            header = f.read(4)
            if header.decode("utf-8") != 'PIEH':
                raise Exception('Flow file header does not contain PIEH')

            width = np.fromfile(f, np.int32, 1).squeeze()
            height = np.fromfile(f, np.int32, 1).squeeze()

            flow = np.fromfile(f, np.float32, int(width) * int(height) * 2).reshape((int(height), int(width), 2))
        return flow

    def _read_ppm_images(self, files_index: int) -> list[np.ndarray]:
        path_to_first_image = os.path.join(self.path_to_images, self.first_img_name.format(files_index))
        path_to_second_image = os.path.join(self.path_to_images, self.second_img_name.format(files_index))
        img_1 = np.array(Image.open(path_to_first_image)).astype(np.float32)
        img_2 = np.array(Image.open(path_to_second_image)).astype(np.float32)

        return [img_1, img_2]
