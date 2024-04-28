import os
from xml.etree import ElementTree

import cv2
import numpy as np
from pydicom import dicomio
import SimpleITK as sitk

# this gets the annotations
class XML_preprocessor(object):

    def __init__(self, data_path, num_classes, normalize=False):
        self.path_prefix = data_path
        self.num_classes = num_classes
        self.normalization = normalize
        self.data = dict()
        self.extract_data_from_xml()

    def extract_data_from_xml(self):
        """
        Extracts data from XML files in the specified path and stores it in the `data` attribute.

        :return: None
        """
        filenames = os.listdir(self.path_prefix)
        for filename in filenames:
            filepath = os.path.join(self.path_prefix, filename)
            try:
                tree = ElementTree.parse(filepath)
            except FileNotFoundError:
                print(f'Error: file {filename} not found in path: {self.path_prefix}')
                continue
            except ElementTree.ParseError:
                print(f'Error: file {filename} is not a valid XML file')
                continue

            root = tree.getroot()
            size_tree = root.find('size')
            width = float(size_tree.find('width').text)
            height = float(size_tree.find('height').text)

            bounding_boxes = []
            one_hot_classes = []
            # find coordinates of box w/ tumor, make bounding box
            for object_tree in root.findall('object'):
                for bounding_box in object_tree.iter('bndbox'):
                    xmin = float(bounding_box.find('xmin').text)
                    ymin = float(bounding_box.find('ymin').text)
                    xmax = float(bounding_box.find('xmax').text)
                    ymax = float(bounding_box.find('ymax').text)

                    if self.normalization:
                        xmin /= width
                        ymin /= height
                        xmax /= width
                        ymax /= height

                    bounding_boxes.append([xmin, ymin, xmax, ymax])
                    class_name = object_tree.find('name').text

                    # this makes one hot encoding of label
                    one_hot_classes.append(self._generate_one_hot_vector(class_name))

            image_data = np.hstack((np.asarray(bounding_boxes), np.asarray(one_hot_classes)))
            self.data[filename] = image_data

    # one hot encoding of 4 labels
    def _generate_one_hot_vector(self, name):
        """
        Generate a one-hot vector for the given name.

        :param name: The name to generate the one-hot vector for.
        :type name: str
        :return: The one-hot vector representing the name.
        :rtype: list[int]
        """
        name = name.upper()
        one_hot_vector = [0] * self.num_classes
        label_dict = {'A': 0, 'B': 1, 'E': 2, 'G': 3}

        if name in label_dict:
            one_hot_vector[label_dict[name]] = 1
        else:
            print(f'unknown label: {name}')
            print(self.path_prefix)

        return one_hot_vector


def image_properties(filename):
    """
    Retrieves properties of an image file.

    :param filename: The path or filename of the image file.
    :type filename: str
    :return: A tuple containing the image array, frame number, width, height, and number of channels.
    :rtype: tuple[numpy.ndarray, int, int, int, int]
    """
    ds = sitk.ReadImage(filename)
    img_array = sitk.GetArrayFromImage(ds)
    frame_num, width, height = img_array.shape[:3]
    ch = 1 if len(img_array.shape) == 3 else img_array.shape[3]
    return img_array, frame_num, width, height, ch


def parse_dicom_file(filename):
    """Parse DICOM File and Extract Information

    :param filename: The path of the DICOM file to be parsed
    :return: A dictionary containing the extracted DICOM information

    """
    ds = dicomio.read_file(filename, force=True)
    information = {'dicom_num': ds.SOPInstanceUID}
    return information


def image_to_rgb(data, ch):
    """
    Convert an image to RGB format.

    :param data: ndarray
        The input image data.
    :param ch: int
        The number of channels in the image.
    :return: ndarray
        The image data converted to RGB format.
    """
    conversion_dict = {
        3: lambda data: cv2.cvtColor(data, cv2.COLOR_BGR2RGB),
        1: lambda data: (data + 1024) * 0.125
    }

    if ch in conversion_dict:
        img_rgb = conversion_dict[ch](data).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported number of channels: {ch}")

    return img_rgb


def display_directory(path, depth=0):
    """
    Display Directory

    Displays the content of a directory recursively.

    Parameters:
        :param path: The path of the directory to display.
        :param depth: The depth level of the current directory (default 0).

    Returns:
        None

    Example usage:
        >>> display_directory('/home/user/documents', 2)
        root:[/home/user/documents]
        |      +--folder1
        |      |      +--file1.txt
        |      +--folder2
        |      |      +--file2.txt
        |      |      +--file3.txt
        |      +--file4.txt

    Note:
        - The "path" parameter should be a valid directory path.
        - The "depth" parameter is used for indentation purposes and should not be manually set.
    """
    prefix = "|      " * depth
    if depth == 0:
        print(f"root:[{path}]")

    for item in os.listdir(path):
        if '.git' not in item:
            print(f"{prefix}+--{item}")

            newitem = os.path.join(path, item)
            if os.path.isdir(newitem):
                display_directory(newitem, depth + 1)


def is_valid_directory(directory):
    """
    Check if the given directory is valid and not a Subversion directory.

    :param directory: The path of the directory to check.
    :type directory: str
    :return: True if the directory is valid and not a Subversion directory, False otherwise.
    :rtype: bool
    """
    return os.path.isdir(directory) and directory != '.svn'


def replicate_folders(src, tar):
    """
    Recursively replicates folders from source directory to target directory.

    :param src: the source directory path
    :param tar: the target directory path
    :return: None
    """
    paths = [os.path.join(src, name) for name in os.listdir(src) if is_valid_directory(os.path.join(src, name))]

    for path in paths:
        _, filename = os.path.split(path)
        targetpath = os.path.join(tar, filename)

        if not os.path.isdir(targetpath):
            os.mkdir(targetpath)

        replicate_folders(path, targetpath)
    else:
        return


def build_dicom_dict(path):
    """
    Builds a dictionary of DICOM files with their corresponding paths.

    :param path: The path to the directory containing DICOM files.
    :return: The dictionary of DICOM files with their corresponding paths.
    """
    dicom_dict = {}
    date_list = os.listdir(path)

    for date in date_list:
        date_path = os.path.join(path, date)
        series_list = sorted(os.listdir(date_path))

        for series in series_list:
            series_path = os.path.join(date_path, series)
            dicom_list = sorted(os.listdir(series_path))

            for dicom in dicom_list:
                dicom_path = os.path.join(series_path, dicom)
                info = parse_dicom_file(dicom_path)
                dicom_dict[info['dicom_num']] = (dicom_path, dicom)

    return dicom_dict

