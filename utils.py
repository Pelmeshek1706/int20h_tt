from typing import Union, Tuple, List
import pydicom
from PIL import Image
import os
from tqdm import tqdm
import subprocess
from zipfile import ZipFile
import random
import shutil

random.seed(42)

IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
RSNA_DATASET_URL = "https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/"


def download_rsna_dataset(kaggle_creds: str) -> None:
    """
    Download and unpack RSNA dataset from Kaggle

    :param kaggle_creds: Path to folder containing Kaggle credentials as kaggle.json file
    :return: None
    """
    os.environ["KAGGLE_CONFIG_DIR"] = kaggle_creds

    print("Downloading RSNA dataset...")
    subprocess.run(["kaggle", "competitions", "download", "-c", "rsna-pneumonia-detection-challenge"])

    print("Extracting RSNA dataset...")
    with ZipFile("rsna-pneumonia-detection-challenge.zip", 'r') as archive:
        archive.extractall("./datasets/RSNA_data/")
    os.remove("rsna-pneumonia-detection-challenge.zip")
    os.rename("./datasets/RSNA_data/stage_2_train_images/", "./datasets/RSNA_data/train_dicom/")
    os.rename("./datasets/RSNA_data/stage_2_test_images/", "./datasets/RSNA_data/test_dicom/")


def convert_dicom_to_jpeg(dicom_folder_path: str,
                          jpeg_folder_path: str) -> None:
    """
    Convert DICOM images to JPEG. If annotation_folder_path is specified,
    only images that have annotations will be converted.

    :param dicom_folder_path: Path to the folder with DICOM images
    :param jpeg_folder_path: Path to save the converted images
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
                         val_frac: float) -> None:
    """
    Split the original train dataset into train, validation and test sets.
    Train and validation sets will contain only images with bboxes, test set will contain images with and without bboxes

    :param images_folder_path: Path to the folder with train images (in JPEG format)
    :param annotation_folder_path: Path to the folder with annotation files (in YOLOv8 format)
    :param val_frac: Fraction of the annotated train images to include in the validation set
    :return: None
    """
    os.makedirs("./datasets/train/images/", exist_ok=True)
    os.makedirs("./datasets/train/labels/", exist_ok=True)
    os.makedirs("./datasets/val/images/", exist_ok=True)
    os.makedirs("./datasets/val/labels/", exist_ok=True)
    os.makedirs("./datasets/test/images/", exist_ok=True)

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
            shutil.move(f"{images_folder_path}/{image}", "./datasets/val/images/")
            shutil.move(f"{annotation_folder_path}/{image_id}.txt", "./datasets/val/labels/")
        elif image_id in test_ids:
            shutil.move(f"{images_folder_path}/{image}", "./datasets/test/images/")
        elif image_id in train_ids:
            shutil.move(f"{images_folder_path}/{image}", "./datasets/train/images/")
            shutil.move(f"{annotation_folder_path}/{image_id}.txt", "./datasets/train/labels/")
    # some unannotated files will be moved to the test folder
    for image in tqdm(os.listdir(images_folder_path)):
        if len(os.listdir("./datasets/test/images/")) <= 2*val_size:
            shutil.move(f"{images_folder_path}/{image}", "./datasets/test/images/")
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


def __convert_to_rsna_format(x: float,
                             y: float,
                             width: float,
                             height: float) -> Tuple[float, float]:
    """
    Convert inference results from YOLOv8 format (without normalization) to original RSNA dataset format

    :param x: The center x coordinate of the bounding box
    :param y: The center y coordinate of the bounding box
    :param width: The width of the predicted bounding box
    :param height: The height of the predicted bounding box
    :return: Bounding box coordinates in original RSNA dataset format
    """
    upper_left_x = x - width/2
    upper_left_y = y - height/2
    return upper_left_x, upper_left_y


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


def process_inference_results(confidences: List, bboxes_xywh: List) -> str:
    """
    Return string representation of inference results

    :param confidences: List of confidence scores for each bounding box
    :param bboxes_xywh: List of bounding boxes coordinates for in YOLOv8 format (without normalization)
    :return: Prediction string with processed inference results
    """
    output_results = []
    for conf, xywh in zip(confidences, bboxes_xywh):
        output_results.append(conf)
        x, y, width, height = xywh
        upper_left_x, upper_left_y = __convert_to_rsna_format(x, y, width, height)
        output_results.extend([upper_left_x, upper_left_y, width, height])
    return " ".join(list(map(str, output_results)))
