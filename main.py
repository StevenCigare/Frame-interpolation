import os

import cv2

from config import TRAINING, PATH_TO_IMAGES
from core.builders.flying_chairs_builder import FlyingChairsDataGenerator
from models.flow_nets import FlowNet
from utils.utils import write_flo_file

if __name__ == '__main__':
    # if out of gpu memory error, try smaller batch_size
    flow_net = FlowNet()
    flow_net.create_model()
    if TRAINING:
        data_generator = FlyingChairsDataGenerator(batch_size=16)

        flow_net.train(data_generator, epochs=10)
    else:
        flow_net.model.load_weights('C:\\Users\\micha\\Downloads\\Frame-interpolation\\10_14_2023__15_55_11.keras')
        images = []
        flow_file_name = "{:05d}_flow.flo"
        first_img_name = "{:05d}_img1.ppm"
        second_img_name = "{:05d}_img2.ppm"
        files_index = 18624
        flow = flow_net.generate_flow(PATH_TO_IMAGES + first_img_name.format(files_index),
                                      PATH_TO_IMAGES + second_img_name.format(files_index))[0]
        upscaled_flow = cv2.resize(flow, (512, 384), interpolation=cv2.INTER_CUBIC)
        current_path = os.path.abspath(os.getcwd()) + "\\output\\"
        write_flo_file(current_path + "\\predicted.flo", upscaled_flow)
        os.system("python -m flowiz " + current_path + "\\predicted.flo")
        os.system(
            "python -m flowiz " + PATH_TO_IMAGES + flow_file_name.format(files_index) + " --outdir " + current_path
        )
