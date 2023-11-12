import os

import cv2
import keras.models

from config import TRAINING, PATH_TO_IMAGES, MODEL_EPOCHS, MODEL_STEPS_PER_EPOCH
from core.builders.flying_chairs_builder import FlyingChairsDataGenerator
from models.flow_nets import FlowNet
from utils.utils import write_flo_file,read_flo_file
import numpy as np

if __name__ == '__main__':
    # if out of gpu memory error, try smaller batch_size
    flow_net = FlowNet()
    flow_net.create_model()
    if TRAINING:
        data_generator = FlyingChairsDataGenerator(batch_size=8)
        validation_generator = FlyingChairsDataGenerator(batch_size=8, validation=True)
        flow_net.train(data_generator, validation_generator, epochs=MODEL_EPOCHS, steps_per_epoch=MODEL_STEPS_PER_EPOCH)
    else:
        flow_net.model.load_weights('10_24_2023__22_33_41.keras')
        images = []
        flow_file_name = "{:05d}_flow.flo"
        first_img_name = "{:05d}_img1.ppm"
        second_img_name = "{:05d}_img2.ppm"
        files_index = 21370
        flow = flow_net.generate_flow(PATH_TO_IMAGES + first_img_name.format(files_index),
                                      PATH_TO_IMAGES + second_img_name.format(files_index))[0]
        upscaled_flow = cv2.resize(flow, (512, 384), interpolation=cv2.INTER_CUBIC)
        current_path = os.path.abspath(os.getcwd()) + "\\output\\"
        write_flo_file(current_path, upscaled_flow)
        os.system("python -m flowiz " + current_path + "\\predicted.flo")
        os.system(
            "python -m flowiz " + PATH_TO_IMAGES + flow_file_name.format(files_index) + " --outdir " + current_path
        )
        gt_flow = read_flo_file(PATH_TO_IMAGES + flow_file_name.format(files_index))
        du = upscaled_flow[:, :, 0] - gt_flow[:, :, 0]
        dv = upscaled_flow[:, :, 1] - gt_flow[:, :, 1]
        endpoint_error = np.sum(np.sqrt(du ** 2 + dv ** 2))/(gt_flow.shape[0]*gt_flow.shape[1])
        print(endpoint_error)