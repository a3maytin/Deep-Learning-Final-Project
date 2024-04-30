# Deep-Learning-Final-Project

This project is a deep learning architecture modeled off of the paper "DeepLung: Deep 3D Dual Path Nets for Automated
Pulmonary Nodule Detection and Classification". The project is implemented in Python and uses the Keras library for
building and training the deep learning models.

## Project Structure

The project is divided into two main directories: `model` and `preprocessing`.

### Model

The `model` directory contains the following files:

- `base_model.py`: This file contains the function `create_base_model` which creates a base model with convolutional and
  pooling layers. It also contains the `preprocess_input` function for preprocessing the input data.

- `dataGenerator.py`: This file contains the `ImageDataGenerator` class which is a custom data generator for loading and
  preprocessing image data for the model.

- `model.py`: This file contains the main model architecture. It includes the `create_detector` function for creating
  the final model, and the `ModelTrainer` class for training the model.

### Preprocessing

The `preprocessing` directory contains the following files:

- `__init__.py`: This file is used to initialize the `preprocessing` package.

- `preprocess.py`: This file contains functions for preprocessing the data. It includes functions for getting the files,
  saving image and annotations, and the main script for preprocessing the data.

- `split_data.py`: This file contains functions for splitting the data into training, testing, and validation sets. It
  also includes functions for saving the annotations and data.

- `utilities.py`: This file contains utility functions used in preprocessing. It includes functions for extracting data
  from XML files, getting image properties, converting images to RGB, displaying directory, checking if a directory is
  valid, replicating folders, and building a dictionary of DICOM files.

## Usage

To use this project, you need to first preprocess the data using the scripts in the `preprocessing` directory. The data
can be found at [the cancer imaging archive.](https://www.cancerimagingarchive.net/collection/lung-pet-ct-dx/) Both the
data itself and the annotations need to be downloaded. After preprocessing, you can train the model using the scripts in
the `model` directory.

## Requirements

This project requires Python and the following Python libraries installed:

- Keras
- TensorFlow
- pandas
- numpy
- sklearn
- cv2
- pydicom
- SimpleITK

## License

This project is licensed under the MIT License.