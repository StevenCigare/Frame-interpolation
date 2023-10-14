import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
from config import PATH_TO_IMAGES, NUMBER_OF_SAMPLES_TO_LOAD
from sklearn.utils import shuffle


class FlyingChairsDataGenerator(Sequence):
    def __init__(self, batch_size):
        self.data_directory = PATH_TO_IMAGES
        self.batch_size = batch_size
        self.number_of_samples = NUMBER_OF_SAMPLES_TO_LOAD
        # arrays of flow and image cases indexes
        # from 1 to n+1 where N is number of image pairs and corresponding flow
        self.files_indexes = np.arange(1, self.number_of_samples + 1)
        self.flow_file_name = "{:05d}_flow.flo"
        self.first_img_name = "{:05d}_img1.ppm"
        self.second_img_name = "{:05d}_img2.ppm"
        # shuffle files indexes
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.number_of_samples / self.batch_size))

    def on_epoch_end(self):
        np.random.shuffle(self.files_indexes)

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        flows = []
        images = []

        for i in range(start_index, end_index):
            # get index from shuffled indexes array
            files_index = self.files_indexes[i]
            with open(os.path.join(self.data_directory, self.flow_file_name.format(files_index)), 'rb') as f:
                header = f.read(4)
                if header.decode("utf-8") != 'PIEH':
                    raise Exception('Flow file header does not contain PIEH')

                width = np.fromfile(f, np.int32, 1).squeeze()
                height = np.fromfile(f, np.int32, 1).squeeze()

                data = np.fromfile(f, np.float32, int(width) * int(height) * 2).reshape((int(height), int(width), 2))

            flows.append(data)
            images.append(
                np.concatenate([
                    Image.open(os.path.join(self.data_directory, self.first_img_name.format(files_index))),
                    Image.open(os.path.join(self.data_directory, self.second_img_name.format(files_index)))
                ], axis=-1
                )
            )

        return np.array(images), np.array(flows)
