import os
import numpy as np

from PIL import Image

from config import PATH_TO_IMAGES, NUMBER_OF_SAMPLES_TO_LOAD


class InputDataBuilder:
    def __init__(self):
        self.images = []

    def build(self):
        flows, images = self._load_data()
        print(flows.shape)
        print(images.shape)
        return self

    def _split_train_test(self):
        pass

    @staticmethod
    def _load_data() -> tuple[np.ndarray, np.ndarray]:
        files_names = os.listdir(PATH_TO_IMAGES)
        flows = []
        images = []
        for i in range(0, NUMBER_OF_SAMPLES_TO_LOAD, 3):
            print(files_names[i])
            with open(f"{PATH_TO_IMAGES}/{files_names[i]}", 'rb') as f:
                header = f.read(4)
                if header.decode("utf-8") != 'PIEH':
                    raise Exception('Flow file header does not contain PIEH')

                width = np.fromfile(f, np.int32, 1).squeeze()
                height = np.fromfile(f, np.int32, 1).squeeze()

                data = np.fromfile(f, np.float32, int(width) * int(height) * 2).reshape((int(height), int(width), 2))

            flows.append(data)
            images.append(
                np.expand_dims(
                    np.concatenate([
                        Image.open(f"{PATH_TO_IMAGES}/{files_names[i+1]}"),
                        Image.open(f"{PATH_TO_IMAGES}/{files_names[i+2]}")
                    ], axis=-1
                    ), axis=0)
            )

        return np.array(flows), np.array(images)
