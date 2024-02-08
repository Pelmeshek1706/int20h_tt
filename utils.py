from typing import Optional, Union, Tuple
import pydicom
from PIL import Image
import os
from tqdm import tqdm
import subprocess
import json
from zipfile import ZipFile
import random
import shutil

random.seed(42)

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
RSNA_DATASET_URL = "https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/"


def download_rsna_dataset():
    """
    Download and unpack RSNA dataset from Kaggle

    :return: None
    """
    # ask user for kaggle credentials
    print("Please provide your Kaggle credentials. You must participate in the competition to download the dataset.")
    kaggle_username = input("Enter Kaggle username: ")
    kaggle_api_key = input("Enter Kaggle API key: ")

    os.makedirs(".kaggle/", exist_ok=True)
    os.environ["KAGGLE_CONFIG_DIR"] = ".kaggle/"

    # write credentials into json file
    with open(".kaggle/kaggle.json", 'w') as kaggle_json:
        kaggle_creds = {"username": kaggle_username,
                        "key": kaggle_api_key}
        kaggle_json.write(json.dumps(kaggle_creds))

    print("Downloading RSNA dataset...")
    subprocess.run(["kaggle", "competitions", "download", "-c", "rsna-pneumonia-detection-challenge"])

    print("Extracting RSNA dataset...")
    with ZipFile("rsna-pneumonia-detection-challenge.zip", 'r') as archive:
        archive.extractall("data/RSNA_data/")
    os.remove("rsna-pneumonia-detection-challenge.zip")
    os.rename("data/RSNA_data/stage_2_train_images/", "data/RSNA_data/train_dicom/")
    os.rename("data/RSNA_data/stage_2_test_images/", "data/RSNA_data/test_dicom/")


def convert_dicom_to_jpeg(dicom_folder_path: str,
                          jpeg_folder_path: str) -> None:
    """
    Convert DICOM images to JPEG. If annotation_folder_path is specified,
    only images that have annotations will be converted.

    :param dicom_folder_path: Path to the folder with DICOM images
    :param jpeg_folder_path: Path to save the converted images
    :param annotation_folder_path: Path to .txt files with annotations in YOLOv8 format
    :return: None
    """
    if not os.path.exists(jpeg_folder_path):
        os.makedirs(jpeg_folder_path)

    for dcm_image in tqdm(os.listdir(dicom_folder_path)):
        image_id = dcm_image.split(".")[0]

        dcm_data = pydicom.read_file(f"{dicom_folder_path}/{dcm_image}")
        im = dcm_data.pixel_array
        im = Image.fromarray(im)
        im.save(f"{jpeg_folder_path}/{image_id}.jpeg", 'JPEG')


def train_val_test_split(images_folder_path: str,
                         annotation_folder_path: str,
                         val_frac: float):
    os.makedirs("data/train/images/", exist_ok=True)
    os.makedirs("data/train/labels/", exist_ok=True)
    os.makedirs("data/val/images/", exist_ok=True)
    os.makedirs("data/val/labels/", exist_ok=True)
    os.makedirs("data/test/images/", exist_ok=True)

    annotated_images_ids = list(map(lambda x: x.split(".")[0],
                                    os.listdir(annotation_folder_path)))
    random.shuffle(annotated_images_ids)  # shuffle annotated files

    # split ids
    val_size = int(val_frac * len(annotated_images_ids))
    val_ids = annotated_images_ids[:val_size]
    test_ids = annotated_images_ids[val_size:(2*val_size)]
    train_ids = annotated_images_ids[2*val_size:]

    # move files to the corresponding folders
    for image in tqdm(os.listdir(images_folder_path)):
        image_id = image.split(".")[0]
        if image_id in val_ids:
            shutil.move(f"{images_folder_path}/{image}", "data/val/images/")
            shutil.move(f"{annotation_folder_path}/{image_id}.txt", "data/val/labels/")
        elif image_id in test_ids:
            shutil.move(f"{images_folder_path}/{image}", "data/test/images/")
        elif image_id in train_ids:
            shutil.move(f"{images_folder_path}/{image}", "data/train/images/")
            shutil.move(f"{annotation_folder_path}/{image_id}.txt", "data/train/labels/")
    # some unannotated files will be moved to the test folder
    for image in tqdm(os.listdir(images_folder_path)):
        if len(os.listdir("data/test/images/")) <= 2*val_size:
            shutil.move(f"{images_folder_path}/{image}", "data/test/images/")
        else:
            break


def __convert_to_yolo_format(x: Union[int, float],
                             y: Union[int, float],
                             width: Union[int, float],
                             height: Union[int, float],
                             img_width: Union[int, float],
                             img_height: Union[int, float]) -> Tuple[float, float, float, float]:
    """
    Convert annotations to YOLOv8 format

    :param x: The upper-left x coordinate of the bounding box
    :param y: The upper-left x coordinate of the bounding box
    :param width: The width of the bounding box
    :param height: The height of the bounding box
    :param img_width: The width of the image
    :param img_height: The height of the image
    :return: Normalized bounding box coordinates in YOLOv8 format
    """
    center_x = x + width / 2
    center_y = y + height / 2
    # coordinates normalization
    center_x /= img_width
    center_y /= img_height
    width /= img_width
    height /= img_height
    return center_x, center_y, width, height


def create_annotation_folder_from_csv(csv_annotation_path: str,
                                      annotation_folder_path: str) -> None:
    """
    Create annotations in YOLOv8 format from CSV

    :param csv_annotation_path: The path to the CSV file with annotations
    :param annotation_folder_path: The path to store annotation files in YOLOv8 format
    :return: None
    """
    if not os.path.exists(annotation_folder_path):
        os.makedirs(annotation_folder_path)

    with open(csv_annotation_path, "r") as csv_annotations:
        # skip header
        csv_annotations.readline()

        while True:
            line = csv_annotations.readline()

            if not line:
                break

            # parse the line by splitting
            patient_id, x, y, width, height, target = line.strip().split(",")
            if target == "0":
                continue

            # convert coordinates to yolo format
            norm_x, norm_y, norm_width, norm_height = __convert_to_yolo_format(x=float(x),
                                                                               y=float(y),
                                                                               width=float(width),
                                                                               height=float(height),
                                                                               img_width=IMAGE_WIDTH,
                                                                               img_height=IMAGE_HEIGHT)

            if os.path.isfile(f"{annotation_folder_path}/{patient_id}.txt"):
                access_mode = "a"
            else:
                access_mode = "w"

            with open(f"{annotation_folder_path}/{patient_id}.txt", access_mode) as annotation_file:
                annotation_file.write(f"{target} {norm_x} {norm_y} {norm_width} {norm_height}\n")
