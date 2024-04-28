import os

import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(data_dir, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.

    :param data_dir: The directory containing the data.
    :param test_size: The proportion of the dataset to include in the test split.
    :param random_state: The seed used by the random number generator.

    :return: The training and testing sets.
    """
    annotations = pd.read_csv(os.path.join(data_dir, 'annotations.csv'))

    train, test = train_test_split(annotations, test_size=test_size, random_state=random_state,
                                   stratify=annotations['class'])

    test, val = train_test_split(test, test_size=0.5, random_state=random_state,
                                 stratify=test['class'])

    return train, test, val


def save_annotations(row, file_name):
    """
    Saves annotations in a specific format to a file.

    :param row: A list containing information about the annotation.
    :type row: list
    :param file_name: The name of the file to write the annotations to.
    :type file_name: file object
    :return: None
    :rtype: None
    """
    x_min, y_min, x_max, y_max = row[2], row[3], row[4], row[5]
    formatted = [((x_min + x_max) / (2 * 512)), ((y_min + y_max) / (2 * 512)), (x_max - x_min) / 512,
                 (y_max - y_min) / 512]

    class_name_to_index = {class_name: i for i, class_name in enumerate(['A', 'B', 'E', 'G'])}

    file_name.write(f"{class_name_to_index[row[6]]} {formatted[0]} {formatted[1]} {formatted[2]} {formatted[3]}\n")


def save_data(train_test_or_val, name, data_dir):
    """
    ``save_data(train_test_or_val, name, data_dir)``

    This method is used to save data to a specified directory. It takes in three parameters:

    - ``train_test_or_val``: A pandas DataFrame or Series object containing the data to be saved.
    - ``name``: A string representing the name of the data.
    - ``data_dir``: A string representing the directory where the data should be saved.

    The method does the following:

    1. Checks if the directory ``data_dir + name`` exists. If it does not, it creates the directory.
    2. Creates two subdirectories within the main directory called "images" and "labels".
    3. Iterates over the values in ``train_test_or_val`` and copies corresponding image files to the "images" subdirectory using the ``cp`` command.
    4. For each line, it creates a text file in the "labels" subdirectory and writes the annotations using the ``save_annotations()`` function.

    This method does not return any value.

    Example Usage:

    ```python
    import pandas as pd

    data = pd.DataFrame({
        'category': ['cat', 'dog', 'cat'],
        'image_name': ['image1', 'image2', 'image3']
    })

    save_data(data, 'train', '/path/to/data')
    ```
    """
    if not os.path.exists(data_dir + name):
        os.makedirs(data_dir + name)
        os.makedirs(data_dir + name + '/images')
        os.makedirs(data_dir + name + '/labels')

    for line in train_test_or_val.values:
        os.system('cp ../data/{}/{}.jpg ../data/{}/images/{}_{}.jpg'.format(line[0], line[1], name, line[0], line[1]))

        with open('../data/{}/labels/{}_{}.txt'.format(name, line[0], line[1]), 'w') as file:
            save_annotations(line, file)


if __name__ == '__main__':
    data_dir = '../data/'

    train, test, val = split_data(data_dir)

    save_data(train, 'train', data_dir)
    save_data(test, 'test', data_dir)
    save_data(val, 'val', data_dir)
