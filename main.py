import os

import cv2
import numpy as np
from PIL import Image

from config import PATH_TO_IMAGES, TRAINING
from core.builders.flying_chairs_builder import FlyingChairsDataGenerator
from models.flow_nets import FlowNet
from utils.utils import read_flo_file, write_flo_file,run_flow_estimation_model,run_flow_estimation_cv2,calculate_epe,save_results

if __name__ == "__main__":
    flow_net = FlowNet()  # type: ignore
    flow_net.create_model()  # type: ignore
    if TRAINING:
        flow_net.model.load_weights("saved_models/best_model.keras")
        data_generator = FlyingChairsDataGenerator(batch_size=8)
        validation_generator = FlyingChairsDataGenerator(batch_size=8, validation=True)
        flow_net.train(data_generator, validation_generator, epochs=75)  # type: ignore
    else:
        FLOW_FILE_NAME = "{:05d}_flow.flo"
        SAMPLE_IDX = 22529

        GT_FLOW = read_flo_file(PATH_TO_IMAGES + FLOW_FILE_NAME.format(SAMPLE_IDX))  # type: ignore
        OUTPUT_FLOW_PATH = os.path.abspath(os.getcwd()) + "\\output\\"
        flow_net.model.load_weights("best_model.keras")

        model_flow = run_flow_estimation_model(flow_net,SAMPLE_IDX)
        save_results(OUTPUT_FLOW_PATH,"\\model_flow.flo",model_flow,SAMPLE_IDX)
        print("model epe: ")
        print(calculate_epe(GT_FLOW,model_flow))

        cv_flow = run_flow_estimation_cv2(SAMPLE_IDX)
        save_results(OUTPUT_FLOW_PATH,"\\cv_flow.flo",cv_flow,SAMPLE_IDX)
        print("opencv epe: ")
        print(calculate_epe(GT_FLOW,cv_flow))