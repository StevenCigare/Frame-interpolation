import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, LeakyReLU, Concatenate
from keras import activations
from config import MODEL_INPUT_SHAPE, TRAINING


class FlowNet:
    def __init__(self):
        self._model: keras.Sequential | None = None
    @property
    def model(self) -> keras.Sequential:
        if self._model is None:
            print("Model not present.")
        else:
            return self._model

    def create_model(self):
        # todo: try next training with leakyrelu
        # x = LeakyReLU(alpha=0.1)(conv1)
        input_layer = Input(shape=MODEL_INPUT_SHAPE)
        conv1 = Conv2D(filters=64, kernel_size=(7, 7), strids=(2, 2), padding='same', activation='relu')(input_layer)
        conv2 = Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(256, (5, 5), strides=(2, 2), padding='same', activation='relu')(conv2)
        conv3_1 = Conv2D(256, (3, 3), padding='same', activation='relu')(conv3)
        conv4 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv3_1)
        conv4_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv4)
        conv5 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv4_1)
        conv5_1 = Conv2D(512, (3, 3), padding='same', activation='relu')(conv5)
        conv6 = Conv2D(1024, (3, 3), padding='same', activation='relu')(conv5_1)
        conv6_1 = Conv2D(1024, (3, 3), padding='same', activation='relu')(conv6)
        """ The paper itself doesn't have this documented but all implementations, including the original authors, use an extra flow path in the code. """
        predict_6 = Conv2D(name='predict_6', filters=2, kernel_size=3, strides=1, padding='same',
                                           activation=None)(conv6_1)

        """ Expanding part """
        upconv5 = Conv2DTranspose(name='upconv_5', filters=512, kernel_size=(4, 4), strides=2,
                                                   padding='same', activation='relu')(conv6)
        flow6 = Conv2DTranspose(name='flow_6', filters=2, kernel_size=(4, 4), strides=2,
                                                 padding='same', activation='relu')(predict_6)
        concat5 = Concatenate(name='concat_5', axis=-1)([upconv5, conv5_1, flow6])
        predict5 = Conv2D(name='predict_5', filters=2, kernel_size=3, strides=1, padding='same',
                                           activation=None)(concat5)

        upconv4 = Conv2DTranspose(name='upconv_4', filters=256, kernel_size=(4, 4), strides=2,
                                                   padding='same', activation='relu')(concat5)
        flow5 = Conv2DTranspose(name='flow_5', filters=2, kernel_size=(4, 4), strides=2,
                                                 padding='same', activation='relu')(predict5)
        concat4 = Concatenate(name='concat_4', axis=-1)([upconv4, conv4_1, flow5])
        predict4 = Conv2D(name='predict_4', filters=2, kernel_size=3, strides=1, padding='same',
                                           activation=None)(concat4)

        upconv3 = Conv2DTranspose(name='upconv_3', filters=128, kernel_size=(4, 4), strides=2,
                                                   padding='same', activation='relu')(concat4)
        flow4 = Conv2DTranspose(name='flow_4', filters=2, kernel_size=(4, 4), strides=2,
                                                 padding='same', activation='relu')(predict4)
        concat3 = Concatenate(name='concat_3', axis=-1)([upconv3, conv3_1, flow4])
        predict3 = Conv2D(name='predict_3', filters=2, kernel_size=3, strides=1, padding='same',
                                           activation=None)(concat3)

        upconv2 = Conv2DTranspose(name='upconv_2', filters=64, kernel_size=(4, 4), strides=2,
                                                   padding='same', activation='relu')(concat3)
        flow3 = Conv2DTranspose(name='flow_3', filters=2, kernel_size=(4, 4), strides=2,
                                                 padding='same', activation='relu')(predict3)
        concat2 = Concatenate(name='concat_2', axis=-1)([upconv2, conv2, flow3])
        predict2 = Conv2D(name='predict_2', filters=2, kernel_size=3, strides=1, padding='same',
                                           activation=None)(concat2)

        upconv1 = Conv2DTranspose(name='upconv_1', filters=64, kernel_size=(4, 4), strides=2,
                                                   padding='same', activation='relu')(concat2)
        flow2 = Conv2DTranspose(name='flow_2', filters=2, kernel_size=(4, 4), strides=2,
                                                 padding='same', activation='relu')(predict2)
        concat1 = Concatenate(name='concat_1', axis=-1)([upconv1, conv1, flow2])
        predict1 = Conv2D(name='predict_1', filters=2, kernel_size=3, strides=1, padding='same',
                                           activation=None)(concat1)

        if TRAINING:
            self._model = keras.Model(
                inputs=input_layer,
                outputs=[predict_6, predict5, predict4, predict3, predict2, predict1]
            )

        self._model = keras.Model(inputs=input_layer, outputs=predict1)
        #refinement part
        #self.model = model