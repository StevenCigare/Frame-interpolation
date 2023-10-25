from datetime import datetime as d

import cv2
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Concatenate, Conv2DTranspose
from tensorflow.keras.initializers import VarianceScaling, Constant
from tensorflow.keras.constraints import UnitNorm, NonNeg

from config import MODEL_INPUT_SHAPE, TRAINING, PATH_TO_IMAGES
from utils import Visualizer
from utils.utils import conv2d_leaky_relu, conv2d_transpose_leaky_relu, crop_like, flow_to_color


class Epe(tf.keras.losses.Loss):

    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def call(self, y_true, y_pred):
        return self.ratio * tf.keras.backend.sqrt(
            tf.keras.backend.sum(
                tf.keras.backend.square(tf.keras.preprocessing.image.smart_resize(y_true, y_pred.shape[1:3]) - y_pred),
                axis=1, keepdims=True))


class PatchCallback(tf.keras.callbacks.Callback):
    def __init__(self, model: keras.Sequential) -> None:
        super().__init__()
        self.model = model
        self.images = np.expand_dims(
            np.concatenate([
                Image.open(f"{PATH_TO_IMAGES}/21000_img1.ppm"),
                Image.open(f"{PATH_TO_IMAGES}/21000_img2.ppm")
            ], axis=-1
            ), axis=0
        )

        self.visualizer = Visualizer()

    def on_epoch_end(self, epoch, logs=None) -> None:
        flow = self.model.predict(self.images)[6][0]

        resized = cv2.resize(flow, (512, 384), interpolation=cv2.INTER_CUBIC)
        #self.visualizer.draw_flow(flow_to_color(resized))


class FlowNet:
    def __init__(self):
        self._model = None

    @property
    def model(self) -> tf.keras.Sequential:
        """Check if model exists.

        Returns:
            keras model.
        """
        if self._model is None:
            print("Model not present.")
        else:
            return self._model

    def create_model(self):
        # todo: try next training with leakyrelu
        # x = LeakyReLU(alpha=0.1)(conv1)
        input_layer = tf.keras.layers.Input(shape=MODEL_INPUT_SHAPE)
        conv1 = conv2d_leaky_relu(input_layer, 64, (7, 7), padding=(3, 3), strides=(2, 2))
        conv2 = conv2d_leaky_relu(conv1, 128, (5, 5), padding=(2, 2), strides=(2, 2))
        conv3 = conv2d_leaky_relu(conv2, 256, (5, 5), padding=(2, 2), strides=(2, 2))
        conv3_1 = conv2d_leaky_relu(conv3, 256, (3, 3), padding=(1, 1))
        conv4 = conv2d_leaky_relu(conv3_1, 512, (3, 3), padding=(1, 1), strides=(2, 2))
        conv4_1 = conv2d_leaky_relu(conv4, 512, (3, 3), padding=(1, 1))
        conv5 = conv2d_leaky_relu(conv4_1, 512, (3, 3), padding=(1, 1), strides=(2, 2))
        conv5_1 = conv2d_leaky_relu(conv5, 512, (3, 3), padding=(1, 1))
        conv6 = conv2d_leaky_relu(conv5_1, 1024, (3, 3), padding=(1, 1), strides=(2, 2))
        conv6_1 = conv2d_leaky_relu(conv6, 1024, (3, 3), padding=(1, 1))
        """ The paper itself doesn't have this documented but all implementations, including the original authors, use an extra flow path in the code. """

        predict_6 = Conv2D(name='predict_6', filters=2, kernel_size=3, strides=(1, 1), activation=None, use_bias=False)(
            conv6_1)

        upconv5 = crop_like(conv2d_transpose_leaky_relu(conv6, 512, (4, 4), (1, 1), (2, 2)), conv5_1)
        flow6 = crop_like(conv2d_transpose_leaky_relu(predict_6, 2, (4, 4), (1, 1), (2, 2)), conv5_1)
        concat5 = Concatenate(axis=-1)([upconv5, conv5_1, flow6])
        predict5 = Conv2D(name='predict_5', filters=2, kernel_size=3, strides=(1, 1), activation=None, use_bias=False)(
            concat5)
        # try use_bias = false in predict
        upconv4 = crop_like(conv2d_transpose_leaky_relu(concat5, 256, (4, 4), (1, 1), (2, 2)), conv4_1)
        flow5 = crop_like(conv2d_transpose_leaky_relu(predict5, 2, (4, 4), (1, 1), (2, 2)), conv4_1)
        concat4 = Concatenate(axis=-1)([upconv4, conv4_1, flow5])
        predict4 = Conv2D(name='predict_4', filters=2, kernel_size=3, strides=(1, 1), activation=None, use_bias=False)(
            concat4)

        upconv3 = crop_like(conv2d_transpose_leaky_relu(concat4, 128, (4, 4), (1, 1), (2, 2)), conv3_1)
        flow4 = crop_like(conv2d_transpose_leaky_relu(predict4, 2, (4, 4), (1, 1), (2, 2)), conv3_1)
        concat3 = Concatenate(axis=-1)([upconv3, conv3_1, flow4])
        predict3 = Conv2D(name='predict_3', filters=2, kernel_size=3, strides=(1, 1), activation=None, use_bias=False)(
            concat3)

        upconv2 = crop_like(conv2d_transpose_leaky_relu(concat3, 64, (4, 4), (1, 1), (2, 2)), conv2)
        flow3 = crop_like(conv2d_transpose_leaky_relu(predict3, 2, (4, 4), (1, 1), (2, 2)), conv2)
        concat2 = Concatenate(axis=-1)([upconv2, conv2, flow3])
        predict2 = Conv2D(name='predict_2', filters=2, kernel_size=3, strides=(1, 1), activation=None, use_bias=False)(
            concat2)

        upconv1 = crop_like(conv2d_transpose_leaky_relu(concat2, 64, (4, 4), (1, 1), (2, 2)), conv1)
        flow2 = crop_like(conv2d_transpose_leaky_relu(predict2, 2, (4, 4), (1, 1), (2, 2)), conv1)
        concat1 = Concatenate(axis=-1)([upconv1, conv1, flow2])
        predict1 = Conv2D(name='predict_1', filters=2, kernel_size=3, strides=(1, 1), activation=None, use_bias=False)(
            concat1)

        upconv0 = crop_like(conv2d_transpose_leaky_relu(concat1, 64, (4, 4), (1, 1), (2, 2)), input_layer)
        flow1 = crop_like(conv2d_transpose_leaky_relu(predict1, 2, (4, 4), (1, 1), (2, 2)), input_layer)
        concat0 = Concatenate(axis=-1)([upconv0, input_layer, flow1])
        predict0 = Conv2D(name='predict_0', filters=2, kernel_size=3, strides=(1, 1), activation=None, use_bias=False)(
            concat0)

        if TRAINING:
            self._model = tf.keras.Model(
                inputs=input_layer,
                outputs=[predict_6, predict5, predict4, predict3, predict2, predict1, predict0]
            )
        else:
            self._model = tf.keras.Model(inputs=input_layer, outputs=predict0)

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                           loss=[Epe(1 / 2), Epe(1 / 4), Epe(1 / 8), Epe(1 / 16), Epe(1 / 32), Epe(1 / 64),
                                 Epe(1 / 128)])
        self.model.summary()
        # Define custom initializers and constraints

    def train(self, data_generator, validation_generator=None, epochs=10, steps_per_epoch=None):
        checkpoint = ModelCheckpoint(
            filepath='20_01_24_10.keras',  # The filename to save the best model
            monitor='val_loss',  # The metric to monitor (e.g., validation loss)
            save_best_only=True,  # Save only the best model
            mode='min',  # 'min' for loss, 'max' for accuracy, 'auto' to infer
            verbose=1  # Set to 1 to see messages when saving
        )
        self.model.fit(
            data_generator,
            validation_data=validation_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            callbacks=[checkpoint, PatchCallback(self.model)]
        )
        date_time = d.now().strftime("%m_%d_%Y__%H_%M_%S") + ".keras"
        self.model.save(date_time)

    def generate_flow(self, first_image_path: str, second_image_path: str):
        images = np.concatenate([
            Image.open(first_image_path),
            Image.open(second_image_path)
        ], axis=-1
        )

        return self.model.predict(np.expand_dims(images, axis=0))
