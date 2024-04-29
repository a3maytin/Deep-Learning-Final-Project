import math
import os

import numpy as np
from tensorflow.image import resize
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence, to_categorical


class ImageDataGenerator(Sequence):
    def __init__(self, annotations, data_dir, batch_size=64, input_shape=(512, 512, 3), shuffle=True,
                 image_processing_func=None):
        """
        :param annotations: Pandas DataFrame containing annotations for the images. Must have the following columns: patient_id, image_path, class
        :param data_dir: Directory path where the images are stored.
        :param batch_size: (optional) Number of samples in each batch. Default is 64.
        :param input_shape: (optional) Shape of the input images. Default is (512, 512, 3).
        :param shuffle: (optional) Whether to shuffle the data before each epoch. Default is True.
        :param image_processing_func: (optional) Function to preprocess the images. Default is None.

        Initializes the instance of the class and prepares the data for training or evaluation.

        Usage:
        annotations = pd.DataFrame(...)
        data_dir = 'path/to/images'
        obj = ClassName(annotations, data_dir)

        """
        self.annotations = annotations.copy()
        self.annotations["filename"] = self.annotations.patient_id + "/" + self.annotations.image_path.astype(
            "str") + ".jpg"
        self.annotations.drop(axis=1, labels=["patient_id", "image_path"], inplace=True)
        self._normalize_bounding_box(input_shape)
        self._map_classes_to_integers()
        self.data_dir = data_dir if data_dir[-1] == '/' else data_dir + '/'
        self.image_processing_func = image_processing_func if image_processing_func else lambda x: x
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.num_samples = len(self.annotations)
        self.num_classes = self.annotations["class"].nunique()

    def _normalize_bounding_box(self, input_shape):
        """
        Normalizes the bounding box coordinates based on the input shape.

        :param input_shape: A tuple containing the input shape of the image in the format (height, width).
        :type input_shape: tuple
        :return: None
        :rtype: None
        """
        self.annotations.xmin /= input_shape[0]
        self.annotations.xmax /= input_shape[1]
        self.annotations.ymin /= input_shape[0]
        self.annotations.ymax /= input_shape[1]

    def _map_classes_to_integers(self):
        """
        Maps classes in the annotations to integers.

        :returns: None
        """
        self.annotations["class"] = self.annotations["class"].map({
            label: idx for idx, label in enumerate(self.annotations["class"].unique())
        })

    def __len__(self):
        """
        Return the number of batches in the dataset.

        :return: The number of batches in the dataset.
        :rtype: int
        """
        return math.ceil(self.num_samples / self.batch_size)

    def on_epoch_end(self):
        """
        This method is called at the end of each epoch during training.

        :return: None
        """
        if self.shuffle:
            self.annotations = self.annotations.sample(frac=1).reset_index(drop=True)

    def _load_and_preprocess_image(self, data):
        """
        Load and preprocess an image.

        :param data: A data object containing the filename of the image.
        :return: The preprocessed image as a numpy array.
        """
        img = load_img(os.path.join(self.data_dir, data.filename), color_mode="rgb")
        img = img_to_array(img)
        img = resize(img, (self.input_shape[0], self.input_shape[1])).numpy()
        return img / 255.0

    def _get_output(self, data):
        """
        :param data: input data object containing bounding box coordinates and class label
        :return: a list containing the bounding box coordinates and the one-hot encoded label array
        """
        bbox = np.array([data.xmin, data.ymin, data.xmax, data.ymax])
        label = to_categorical(data["class"], self.num_classes)
        return [bbox, label]

    def __getitem__(self, idx):
        """
        Get the item at index 'idx' from the dataset.

        :param idx: The index of the item to retrieve.
        :return: A tuple of input data and output data.
        """
        batch_data = self.annotations[idx * self.batch_size: (idx + 1) * self.batch_size]
        processed_images = np.array([self._load_and_preprocess_image(row) for _, row in batch_data.iterrows()])
        input_data = self.image_processing_func(processed_images)
        output_data = np.array([self._get_output(row) for _, row in batch_data.iterrows()])
        return input_data, {"bbox": output_data[:, 0, :], "label": output_data[:, 1, :]}
