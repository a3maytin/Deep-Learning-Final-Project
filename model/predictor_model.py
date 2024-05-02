import keras
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.models import Model
from keras.optimizers.legacy import Adam
from sklearn.ensemble import GradientBoostingClassifier
from tensorflow import keras


def create_predictor(input_model, class_count=4, learning_rate=0.00025):
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


class ClassifierModel(Model):
    def __init__(self, input_model, class_count=4):
        super(ClassifierModel, self).__init__()

        # Layers for tumor location prediction
        self.dense1 = Dense(128, activation='relu')
        self.dense2 = Dense(64, activation='relu')
        self.dense3 = Dense(32, activation='relu')
        self.dense4 = Dense(4, activation='sigmoid', name="bbox")

        # Create a Gradient Boosting Machine for label prediction
        self.gbm_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                                         random_state=42)

    def call(self, inputs, labels):
        # model recieves intermediatefeatures as inputs

        # Flatten intermediate features if needed
        if len(inputs.shape) == 4:
            inputs = Flatten()(inputs)

        # Train GBM
        self.gbm_classifier.fit(inputs.numpy(), labels)

        # Predict labels using GBM
        label_prediction = self.gbm_classifier.predict(inputs.numpy())

        return label_prediction
