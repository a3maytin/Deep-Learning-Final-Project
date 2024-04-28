import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.utilities import image_properties, image_to_rgb, XML_preprocessor, build_dicom_dict

# import files
def get_files():
    """
    Get the patient IDs, images, and annotations.

    :return: A tuple containing the patient IDs, images, and annotations.
    """

    # os.listdir() used to get the list of all files and directories in the specified directory
    patient_ids = [file.split('-')[1] for file in
                   sorted(list(os.listdir("../manifest-1608669183333/Lung-PET-CT-Dx/")))[1:]]

    images = ["../manifest-1608669183333/Lung-PET-CT-Dx/Lung_Dx-" + file for file in patient_ids]

    annotations = ["../Lung-PET-CT-Dx-Annotations-XML-Files-rev12222020/Annotation/" +
                   file_id for file_id in patient_ids]

    return patient_ids, images, annotations


def save_image_and_annotations(patient, img_name, img_np, img_data, label_list, dataframe):
    """
    :param patient: The patient ID associated with the image and annotations
    :param img_name: The name of the image file
    :param img_np: The image data in NumPy array format
    :param img_data: The list of bounding box annotations for the image
    :param label_list: The list of labels corresponding to the annotations
    :param dataframe: The pandas DataFrame to save the annotations

    :return: The updated pandas DataFrame with the new annotations
    """
    patient_dir = f"../data/{patient}"
    # os.makedirs() creates a directory recursively
    os.makedirs(patient_dir, exist_ok=True)

    for rect in img_data:
        # why does A0005 only have 2 images?

        # gives corners of box where tumor is
        xmin, ymin, xmax, ymax = map(int, rect[:4])
        label = label_list[int(np.where(rect[4:] == 1)[0])]

        # patient, image, tumor box, & label
        new_row = pd.DataFrame([{"patient_id": patient, "image_path": img_name, "xmin": xmin,
                                 "ymin": ymin, "xmax": xmax, "ymax": ymax, "class": label}],
                               columns=dataframe.columns)

        dataframe = pd.concat([dataframe, new_row], ignore_index=True)

    cv2.imwrite(f"{patient_dir}/{img_name}.jpg", img_np)
    return dataframe


if __name__ == '__main__':
    patient_ids, images, annotations = get_files()

    image_dataframe = pd.DataFrame(columns=['patient_id', 'image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'class'])

    # A' were diagnosed with Adenocarcinoma, 'B' with Small Cell Carcinoma, 'E' with Large Cell Carcinoma, and 'G' with Squamous Cell Carcinoma.
    classes = ['A,', 'B', 'E', 'G']
    num_classes = len(classes)

    zip_list = zip(patient_ids, images, annotations)

    # go through each patient in iterable list
    for patientID, dicom_path, annotation_path in tqdm(zip_list, desc="Processing patients"):

        # path to image dictionary
        uid_path_dict = build_dicom_dict(dicom_path)

        if os.path.isdir(annotation_path):
            # annotations say where tumor is & what type of cancer it is

            annotations = XML_preprocessor(annotation_path, num_classes=num_classes).data

            for i, (k, v) in enumerate(list(annotations.items())):
                try:
                    # get the path and the file name
                    dcm_path, dcm_name = uid_path_dict[k[:-4]]
                    # returns tuple containing the image array, (frame number, width, height,) and number of channels
                    matrix, _, _, _, ch = image_properties(os.path.join(dcm_path))
                    # take image array (matrix) converts from DICOM to RGB file
                    img_bitmap = image_to_rgb(matrix[0], ch)
                    # add information to dataframe
                    image_dataframe = save_image_and_annotations(
                        patient=patientID,
                        img_name=str(i),
                        img_np=img_bitmap,
                        img_data=v,
                        label_list=classes,
                        dataframe=image_dataframe
                    )

                    # why do we cut it off at 9?
                    if i >= 9:
                        break
                except Exception as e:
                    pass

    print(image_dataframe.head())
    image_dataframe.to_csv("../data/annotations.csv", index=False)
