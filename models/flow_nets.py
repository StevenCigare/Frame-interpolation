import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Concatenate
from config import MODEL_INPUT_SHAPE, TRAINING
from datetime import datetime as d
from PIL import Image

from utils.utils import conv2d_leaky_relu, conv2d_transpose_leaky_relu, crop_like


class EndPointError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.keras.backend.sqrt(
            tf.keras.backend.sum(
                tf.keras.backend.square(tf.keras.preprocessing.image.smart_resize(y_true, y_pred.shape[1:3]) - y_pred),
                axis=1, keepdims=True))


class FlowNet:
    def __init__(self):
        self._model: tf.keras.Sequential | None = None

    @property
    def model(self) -> tf.keras.Sequential:
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

        predict_6 = Conv2D(name='predict_6', filters=2, kernel_size=3, strides=(1, 1), activation=None)(conv6_1)

        upconv5 = crop_like(conv2d_transpose_leaky_relu(conv6, 512, (4, 4), (1, 1), (2, 2)), conv5_1)
        flow6 = crop_like(conv2d_transpose_leaky_relu(predict_6, 2, (4, 4), (1, 1), (2, 2)), conv5_1)
        concat5 = Concatenate(axis=-1)([upconv5, conv5_1, flow6])
        predict5 = Conv2D(
            name='predict_5',
            filters=2,
            kernel_size=3,
            strides=(1, 1),
            activation=None
        )(concat5)

        upconv4 = crop_like(conv2d_transpose_leaky_relu(concat5, 256, (4, 4), (1, 1), (2, 2)), conv4_1)
        flow5 = crop_like(conv2d_transpose_leaky_relu(predict5, 2, (4, 4), (1, 1), (2, 2)), conv4_1)
        concat4 = Concatenate(axis=-1)([upconv4, conv4_1, flow5])
        predict4 = Conv2D(name='predict_4', filters=2, kernel_size=3, strides=(1, 1), activation=None)(concat4)

        upconv3 = crop_like(conv2d_transpose_leaky_relu(concat4, 128, (4, 4), (1, 1), (2, 2)), conv3_1)
        flow4 = crop_like(conv2d_transpose_leaky_relu(predict4, 2, (4, 4), (1, 1), (2, 2)), conv3_1)
        concat3 = Concatenate(axis=-1)([upconv3, conv3_1, flow4])
        predict3 = Conv2D(name='predict_3', filters=2, kernel_size=3, strides=(1, 1), activation=None)(concat3)

        upconv2 = crop_like(conv2d_transpose_leaky_relu(concat3, 64, (4, 4), (1, 1), (2, 2)), conv2)
        flow3 = crop_like(conv2d_transpose_leaky_relu(predict3, 2, (4, 4), (1, 1), (2, 2)), conv2)
        concat2 = Concatenate(axis=-1)([upconv2, conv2, flow3])
        predict2 = Conv2D(name='predict_2', filters=2, kernel_size=3, strides=(1, 1), activation=None)(concat2)

        upconv1 = crop_like(conv2d_transpose_leaky_relu(concat2, 64, (4, 4), (1, 1), (2, 2)), conv1)
        flow2 = crop_like(conv2d_transpose_leaky_relu(predict2, 2, (4, 4), (1, 1), (2, 2)), conv1)
        concat1 = Concatenate(axis=-1)([upconv1, conv1, flow2])
        predict1 = Conv2D(name='predict_1', filters=2, kernel_size=3, strides=(1, 1), activation=None)(concat1)

        upconv0 = crop_like(conv2d_transpose_leaky_relu(concat1, 64, (4, 4), (1, 1), (2, 2)), input_layer)
        flow1 = crop_like(conv2d_transpose_leaky_relu(predict1, 2, (4, 4), (1, 1), (2, 2)), input_layer)
        concat0 = Concatenate(axis=-1)([upconv0, input_layer, flow1])
        predict0 = Conv2D(name='predict_0', filters=2, kernel_size=3, strides=(1, 1), activation=None)(concat0)

        if TRAINING:
            self._model = tf.keras.Model(
                inputs=input_layer,
                outputs=[predict_6, predict5, predict4, predict3, predict2, predict1, predict0] #, predict1
            )
        else:
            self._model = tf.keras.Model(inputs=input_layer, outputs=predict1)
        epe = EndPointError()

        self.model.compile(optimizer="adam", loss=[epe, epe, epe, epe, epe, epe])
        self.model.summary()

    def train(self, data_generator, epochs=10, steps_per_epoch=None, validation_data=None):
        self.model.fit(
            data_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data
        )
        date_time = d.now().strftime("%m_%d_%Y__%H_%M_%S") + ".keras"
        self.model.save("\\models\\" + date_time)

    def generate_flow(self, first_image_path: str, second_image_path: str):
        images = np.concatenate([
            Image.open(first_image_path),
            Image.open(second_image_path)
        ], axis=-1
        )

        return self._model.predict(np.expand_dims(images, axis=0))




