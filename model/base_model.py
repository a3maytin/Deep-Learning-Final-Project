from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten
from keras.src.applications import imagenet_utils


def create_base_model(input_shape=(512, 512, 3)):
    input_layer = Input(shape=input_shape)

    # Add convolutional and pooling layers
    x = Conv2D(32, (3, 3), activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Flatten the output to prepare it for the fully connected layers
    x = Flatten()(x)

    # Create the model
    base_model = Model(inputs=input_layer, outputs=x)

    return base_model


def preprocess_input(x, data_format=None):
    return imagenet_utils.preprocess_input(
        x, data_format=data_format, mode="tf"
    )
