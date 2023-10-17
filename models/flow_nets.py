import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Concatenate
from config import MODEL_INPUT_SHAPE, TRAINING
from datetime import datetime as d
from PIL import Image
class EndPointError(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.keras.backend.sqrt(
            tf.keras.backend.sum(
                tf.keras.backend.square(tf.keras.preprocessing.image.smart_resize(y_true, y_pred.shape[1:3]) - y_pred),
                axis=1, keepdims=True))


class FlowNet:
    def __init__(self):
        self._model = None

    @property
    def model(self) -> tf.keras.Sequential:
        if self._model is None:
            print("Model not present.")
        else:
            return self._model

    def create_model(self):
        # todo: try next training with leakyrelu
        # x = LeakyReLU(alpha=0.1)(conv1)
        input_layer = Input(shape=MODEL_INPUT_SHAPE)
        conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
        conv2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu')(conv2)
        conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv3)
        conv4 = Conv2D(512, (3, 3), padding='same', activation='relu', strides=(2, 2))(conv3_1)
        conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv4)
        conv5 = Conv2D(512, (3, 3), padding='same', activation='relu', strides=(2, 2))(conv4_1)
        conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv5)
        conv6 = Conv2D(1024, (3, 3), padding='same', activation='relu', strides=(2, 2))(conv5_1)
        conv6_1 = Conv2D(1024, (3, 3), padding='same', activation='relu')(conv6)
        """ The paper itself doesn't have this documented but all implementations, including the original authors, use an extra flow path in the code. """

        predict_6 = Conv2D(name='predict_6', filters=2, kernel_size=3, strides=(1, 1), padding='same',
                           activation=None)(conv6_1)

        upconv5 = Conv2DTranspose(name='upconv_5', filters=512, kernel_size=(4, 4), strides=(2, 2),
                                  padding='same', activation='relu')(conv6)
        flow6 = Conv2DTranspose(name='flow_6', filters=2, kernel_size=(4, 4), strides=(2, 2),
                                padding='same', activation='relu')(predict_6)
        concat5 = Concatenate(name='concat_5', axis=-1)([upconv5, conv5_1, flow6])
        predict5 = Conv2D(name='predict_5', filters=2, kernel_size=3, strides=(1, 1), padding='same',
                          activation=None)(concat5)

        upconv4 = Conv2DTranspose(name='upconv_4', filters=256, kernel_size=(4, 4), strides=(2, 2),
                                  padding='same', activation='relu')(concat5)
        flow5 = Conv2DTranspose(name='flow_5', filters=2, kernel_size=(4, 4), strides=(2, 2),
                                padding='same', activation='relu')(predict5)
        concat4 = Concatenate(name='concat_4', axis=-1)([upconv4, conv4_1, flow5])
        predict4 = Conv2D(name='predict_4', filters=2, kernel_size=3, strides=(1, 1), padding='same',
                          activation=None)(concat4)

        upconv3 = Conv2DTranspose(name='upconv_3', filters=128, kernel_size=(4, 4), strides=(2, 2),
                                  padding='same', activation='relu')(concat4)
        flow4 = Conv2DTranspose(name='flow_4', filters=2, kernel_size=(4, 4), strides=(2, 2),
                                padding='same', activation='relu')(predict4)
        concat3 = Concatenate(axis=-1)([upconv3, conv3_1, flow4])
        predict3 = Conv2D(name='predict_3',filters=2, kernel_size=3, strides=(1, 1), padding='same',
                          activation=None)(concat3)

        upconv2 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                  padding='same', activation='relu')(concat3)
        flow3 = Conv2DTranspose(filters=2, kernel_size=(4, 4), strides=(2, 2),
                                padding='same', activation='relu')(predict3)
        concat2 = Concatenate(axis=-1)([upconv2, conv2, flow3])
        predict2 = Conv2D(name='predict_2',filters=2, kernel_size=3, strides=(1, 1), padding='same',
                          activation=None)(concat2)

        upconv1 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2),
                                  padding='same', activation='relu')(concat2)
        flow2 = Conv2DTranspose(filters=2, kernel_size=(4, 4), strides=(2, 2),
                                padding='same', activation='relu')(predict2)
        concat1 = Concatenate(axis=-1)([upconv1, conv1, flow2])
        predict1 = Conv2D(name='predict_1', filters=2, kernel_size=3, strides=(1, 1), padding='same',
                          activation=None)(concat1)
        if TRAINING:
            self._model = tf.keras.Model(
                inputs=input_layer,
                outputs=[predict_6, predict5, predict4, predict3, predict2, predict1]
            )
        else:
            self._model = tf.keras.Model(inputs=input_layer, outputs=predict1)
        epe = EndPointError()

        self.model.compile(loss=[epe, epe, epe, epe, epe, epe])
        self.model.summary()

    def train(self, data_generator, epochs=10, steps_per_epoch=None, validation_data=None):
        self.model.fit(
            data_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=validation_data
        )
        date_time = d.now().strftime("%m_%d_%Y__%H_%M_%S") + ".keras"
        self.model.save("\\models\\"+date_time)

    def generate_flow(self, first_image_path, second_image_path):
        images = np.concatenate([
            Image.open(first_image_path),
            Image.open(second_image_path)
        ], axis=-1
        )
        new_shape = (1,)+images.shape
        print(new_shape)
        images = np.reshape(images, new_shape)
        print(images.shape)
        return self._model.predict(images)

    # def train(self, x_train, y_train):
    #     self.model.fit(
    #         x=x_train,
    #         y=y_train,
    #         epochs=10,
    #         shuffle=True,
    #         validation_split=0.1,
    #         steps_per_epoch=len(x_train),
    #     )

