import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Dropout, ReLU
from keras.models import Model
from keras.src.applications import imagenet_utils

from PostRes import PostRes


def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="tf"
    )


class DetectionModel(Model):
    def __init__(self, input_shape=(512, 512, 3), **kwargs):
        super(DetectionModel, self).__init__(**kwargs)

        # things to consider: Conv2D output sizes
        self.preBlock = Sequential([
            Conv2D(24, kernel_size=3, padding="same"),
            BatchNormalization(),
            ReLU(),
            Conv2D(24, kernel_size=3, padding="same"),
            BatchNormalization(),
            ReLU()])

        num_blocks_forw = [2, 2, 3, 3]
        num_blocks_back = [3, 3]
        self.featureNum_forw = [24, 32, 64, 64, 64]
        self.featureNum_back = [128, 64, 64]

        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i + 1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i + 1], self.featureNum_forw[i + 1]))
            setattr(self, 'forw' + str(i + 1), Sequential([*blocks]))

        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i == 0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i + 1] + self.featureNum_forw[i + 2] + addition,
                                          self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 1), Sequential([*blocks]))

        # maxpool0 happens right after of preblock, maxpool 1 after forward block 1, etc.
        self.maxpool0 = MaxPooling2D(pool_size=2, strides=2)
        self.maxpool1 = MaxPooling2D(pool_size=2, strides=2)
        self.maxpool2 = MaxPooling2D(pool_size=2, strides=2)
        self.maxpool3 = MaxPooling2D(pool_size=2, strides=2)

        self.path1 = Sequential([
            Conv2DTranspose(filters=64, kernel_size=2, strides=2),
            BatchNormalization(),
            ReLU()])
        self.path2 = Sequential([
            Conv2DTranspose(filters=64, kernel_size=2, strides=2),
            BatchNormalization(),
            ReLU()])
        self.drop = Dropout(rate=0.5)

        # I don't actually understand the size input for this one in the original code, set to 50 here
        self.final_output = Sequential([Conv2D(64, kernel_size=1),
                                        ReLU(),
                                        Conv2D(50, kernel_size=1)])

    def call(self, inputs):
        # inputs = tf.expand_dims(inputs, axis=0)
        out = self.preBlock(inputs)
        outpool = self.maxpool0(out)
        out1 = self.forw1(outpool)
        out1pool = self.maxpool1(out1)
        out2 = self.forw2(out1pool)
        # out2 = self.drop(out2) is commented out in the original documentation
        out2pool = self.maxpool2(out2)
        out3 = self.forw3(out2pool)
        out3pool = self.maxpool3(out3)
        out4 = self.forw4(out3pool)
        # out4 = self.drop(out4)

        reversed1 = self.path1(out4)
        combined1 = self.back2(tf.concat([reversed1, out3], axis=-1))
        # comb3 = self.drop(comb3)

        reversed2 = self.path2(combined1)
        combined2 = self.back1(tf.concat([reversed2, out2], axis=-1))
        combined2 = self.drop(combined2)

        out = self.final_output(combined2)

        return out
