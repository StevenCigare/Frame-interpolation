import os

import cv2
import keras.models
import numpy as np
import tensorflow as tf
from config import TRAINING, PATH_TO_IMAGES
from core.builders.flying_chairs_builder import FlyingChairsDataGenerator
from models.flow_nets import FlowNet
from utils.utils import write_flo_file, read_flo_file
from scripts.validation_data_loader import ValidationDataLoader
from PIL import Image
import numpy as np

if __name__ == '__main__':
    # if out of gpu memory error, try smaller batch_size
    flow_net = FlowNet()
    flow_net.create_model()
    if TRAINING:
        flow_net.model.load_weights('saved_models/best_model.keras')
        data_generator = FlyingChairsDataGenerator(batch_size=8)
        validation_generator = FlyingChairsDataGenerator(batch_size=8, validation=True)
        flow_net.train(data_generator, validation_generator, epochs=75)
    else:
        images = []
        flow_file_name = "{:05d}_flow.flo"
        first_img_name = "{:05d}_img1.ppm"
        second_img_name = "{:05d}_img2.ppm"
        files_index = 18624
        #        flow_net.model.load_weights('epoch_104_best_no_l2.keras')
        flow_net.model.load_weights('saved_models/best_model.keras')
        img_1 = np.array(Image.open(f"{PATH_TO_IMAGES}/{files_index}_img1.ppm")).astype(np.float32)[:373,:501,:]
        img_2 = np.array(Image.open(f"{PATH_TO_IMAGES}/{files_index}_img2.ppm")).astype(np.float32)[:373,:501,:]
        flow = flow_net.generate_flow([img_1, img_2])[0]
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
