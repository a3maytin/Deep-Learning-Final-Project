import tensorflow as tf
from tensorflow import keras


class PostRes(tf.keras.Model):
    """
    Class representing a PostRes block in a neural network.
    This block is the simplest building block used in the detection network

    Args:
        n_in (int): Number of input channels.
        n_out (int): Number of output channels.
        stride (int, optional): Stride value for the convolutional layers. Defaults to 1.

    Attributes:
        conv1 (tf.keras.layers.Conv2D): Convolutional layer 1.
        bn1 (tf.keras.layers.BatchNormalization): Batch Normalization layer 1.
        relu (tf.keras.layers.ReLU): ReLU activation layer.
        conv2 (tf.keras.layers.Conv2D): Convolutional layer 2.
        bn2 (tf.keras.layers.BatchNormalization): Batch Normalization layer 2.
        shortcut (tf.keras.layers.Sequential or None): Shortcut connection.

    Methods:
        forward: Performs forward pass through the PostRes block.

    """

    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(n_out, kernel_size=3, strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(n_out, kernel_size=3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        # Corrected "shortcut" initialzation here
        if stride != 1 or n_out != n_in:
            self.shortcut = tf.keras.Sequential([
                tf.keras.layers.Conv2D(n_out, kernel_size=1, strides=stride),
                tf.keras.layers.BatchNormalization()
            ])
        else:
            self.shortcut = None

    def call(self, x, training=None, mask=None):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out
