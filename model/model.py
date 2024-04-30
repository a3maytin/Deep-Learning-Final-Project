import argparse

import keras
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model, save_model
from keras.optimizers.legacy import Adam, SGD
from sklearn.model_selection import train_test_split
from tensorflow import keras

from base_model import create_base_model, preprocess_input, create_detection_model
from dataGenerator import ImageDataGenerator

# I want to rename this to create_predictions or something 
def create_detector(input_model, class_count=4, learning_rate=0.00025):
    """
    This function takes an input model and returns a new model that predicts both the bounding box and class of
    tumors in lung x-rays.

    Parameters:
    input_model (Model): The base model to build upon.
    class_count (int): The number of classes to predict. Default is 4.

    Returns:
    Model: The detector model that predicts both bounding box and class of tumors.
    """
    intermediate_output = input_model.output

    if len(intermediate_output.shape) == 2:
        intermediate_output = keras.layers.Reshape((1, 1, intermediate_output.shape[1]))(intermediate_output)

    intermediate_output = GlobalAveragePooling2D()(intermediate_output)

    # Predicting the location of the tumor
    bbox_branch = Dense(128, activation='relu')(intermediate_output)
    bbox_branch = Dense(64, activation='relu')(bbox_branch)
    bbox_branch = Dense(32, activation='relu')(bbox_branch)
    bbox_prediction = Dense(4, activation='sigmoid', name="bbox")(bbox_branch)

    # Predicting the type of the tumor
    label_branch = Dense(512, activation="relu")(intermediate_output)
    label_branch = Dropout(0.5)(label_branch)
    label_branch = Dense(512, activation="relu")(label_branch)
    label_branch = Dropout(0.5)(label_branch)
    label_prediction = Dense(class_count, activation="softmax", name="label")(label_branch)

    detector_model = Model(inputs=input_model.input, outputs=(bbox_prediction, label_prediction))

    for layer in input_model.layers:
        layer.trainable = False

    # Compiling the model
    detector_model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
                           metrics={"label": "accuracy", "bbox": "mse", },
                           loss={"label": "categorical_crossentropy", "bbox": "mean_squared_error", },
                           loss_weights={"label": 1.0, "bbox": 1.0})

    return detector_model


class ModelTrainer:
    """
    Module: ModelTrainer

    ModelTrainer is a class that provides functionality for training a machine learning model.

    Attributes:
        - model (object): The machine learning model to be trained.
        - train_data_gen (object): The generator object for the training data.
        - test_data_gen (object): The generator object for the test/validation data.
        - early_stopping (object): The early stopping callback for the training process.
        - checkpointer (object): The model checkpoint callback for saving the best model weights.

    Methods:
        - __init__(model, train_data_gen, test_data_gen, early_stopping, checkpointer)
            Initializes the ModelTrainer object with the given parameters.

        - _train_model(epochs)
            Trains the model for the specified number of epochs.

        - _recompile_model()
            Recompiles the bounding box prediction model after training its upper layers.

        - _plot_common_graphs(epochs, history, ax)
            Prepares the plot of graphs common to both training sessions.

        - first_train(epochs=4)
            Performs the first model training.

        - second_train(epochs=16)
            Performs the second training of the model.
    """

    def __init__(self, model, train_data_gen, test_data_gen, early_stopping_instance, checkpoint_callback):
        """
        Initializes a new instance of the class.

        :param model: The machine learning model to train.
        :param train_data_gen: The data generator for training data.
        :param test_data_gen: The data generator for test data.
        :param early_stopping_instance: The early stopping callback.
        :param checkpoint_callback: The model checkpoint callback.
        """
        self.model = model
        self.train_data_gen = train_data_gen
        self.test_data_gen = test_data_gen
        self.early_stopping = early_stopping_instance
        self.checkpointer = checkpoint_callback

    def _train_model(self, epochs):
        """Trains the model for the specified number of epochs.

        :param epochs: The number of epochs to train the model.
        :return: The training process object.
        """
        training_process = self.model.fit(
            x=self.train_data_gen,
            validation_data=self.test_data_gen,
            epochs=epochs,
            callbacks=[self.early_stopping, self.checkpointer]
        )
        return training_process

    def _recompile_model(self):
        """
        Recompiles the model with specified optimizer, metrics, loss functions, and loss weights.

        :return: None
        """
        self.model.compile(
            optimizer=SGD(learning_rate=0.0001, momentum=0.9),
            metrics={
                "label": ["accuracy", keras.metrics.Precision(name="precision-metric"),
                          keras.metrics.Recall(name="recall-metric")],
                "bbox": "mse"
            },
            loss={
                "label": "categorical_crossentropy",
                "bbox": "mean_squared_error"
            },
            loss_weights={
                "label": 1.0,
                "bbox": 1.0
            }
        )

    def first_train(self, epochs=4):
        """
        Train the model for a specified number of epochs.

        :param epochs: The number of epochs to train the model (default is 4).
        :return: The result of training the model.
        """
        return self._train_model(epochs)

    def second_train(self, epochs=16):
        """
        Method to perform a second round of training on the model.

        :param epochs: Number of epochs to train the model. Default is 16.
        :return: The training history of the model.
        """
        self._recompile_model()

        # history = self._train_model(epochs)
        # history = history.history

        return self._train_model(epochs)


def split_data(data_filepath, test_size=0.1, random_state=42):
    """
    Split the data into training, testing, and validation sets.

    :param data_filepath: The file path of the data.
    :param test_size: The proportion of the data to use for testing (default is 0.1).
    :param random_state: The seed for random number generation (default is 42).
    :return: A tuple containing the training, testing, and validation data generators.
    """
    annotations = pd.read_csv(f"{data_filepath}annotations.csv")

    features, target = annotations.drop("class", axis=1), annotations["class"]

    features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=test_size,
                                                                                random_state=random_state,
                                                                                stratify=target)

    features_train, features_val, target_train, target_val = train_test_split(features_train, target_train,
                                                                              test_size=1 / 9,
                                                                              random_state=random_state,
                                                                              stratify=target_train)

    train_gen = ImageDataGenerator(pd.concat([features_train, target_train], axis=1), data_filepath)
    test_gen = ImageDataGenerator(pd.concat([features_test, target_test], axis=1), data_filepath)
    val_gen = ImageDataGenerator(pd.concat([features_val, target_val], axis=1), data_filepath)

    return train_gen, test_gen, val_gen


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the model.')
    parser.add_argument('--first_epochs', type=int, default=1, help='Number of epochs for the first training')
    parser.add_argument('--second_epochs', type=int, default=1, help='Number of epochs for the second training')
    args = parser.parse_args()

    train_gen, test_gen, val_gen = split_data("../data/")

    base_model = create_detection_model(input_shape=(512, 512, 3))
    hate_cancer_model = create_detector(base_model)

    train_gen.map = preprocess_input
    test_gen.map = preprocess_input
    val_gen.map = preprocess_input


    def get_early_stopping(patience=6):
        """
        Create an instance of the EarlyStopping object for the given parameters.

        :param patience: Number of epochs with no improvement after which training will be stopped. Default is 6.
        :type patience: int

        :return: An instance of the EarlyStopping object with the specified parameters.
        :rtype: EarlyStopping
        """
        return EarlyStopping(monitor='loss', min_delta=0, patience=patience, verbose=2, mode='auto')


    def get_checkpointer(model_name):
        """
        Constructs a ModelCheckpoint object to save the best model weights during training.

        :param model_name: The name of the model.
        :type model_name: str
        :return: The ModelCheckpoint object.
        :rtype: keras.callbacks.ModelCheckpoint
        """
        return ModelCheckpoint(filepath="../backups/" + model_name + '.{epoch:02d}.hdf5', verbose=2,
                               save_best_only=True, save_weights_only=True)


    early_stopping = get_early_stopping()
    checkpointer = get_checkpointer("hate_cancer_model")

    trainer = ModelTrainer(hate_cancer_model, train_gen, val_gen, early_stopping, checkpointer)

    print("First training\n")
    train_history = trainer.first_train(epochs=args.first_epochs)

    for layer in hate_cancer_model.layers[:105]:
        layer.trainable = False

    for layer in hate_cancer_model.layers[105:]:
        layer.trainable = True

    print("\n\nSecond training\n")
    train_history = trainer.second_train(epochs=args.second_epochs)

    print(hate_cancer_model.evaluate(test_gen))

    save_model(hate_cancer_model, "../saved_models/hate_cancer_model.h5")