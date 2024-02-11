from ultralytics import YOLO
from ultralytics import settings
from argparse import ArgumentParser
from utils import download_rsna_dataset, create_annotation_folder_from_csv, convert_dicom_to_jpeg, train_val_test_split
import os

DEFAULT_DATASET_DIR = os.getcwd() + "/datasets"
DEFAULT_CSV_ANNOTATIONS_PATH = "./datasets/RSNA_data/stage_2_train_labels.csv"
DEFAULT_YOLO_ANNOTATIONS_PATH = "./datasets/RSNA_data/labels/"
DEFAULT_TRAIN_IMAGES_PATH = "./datasets/RSNA_data/train_dicom/"
DEFAULT_TEST_IMAGES_PATH = "./datasets/RSNA_data/test_dicom/"
DEFAULT_JPEG_IMAGES_PATH = "./datasets/RSNA_data/images/"

# change default YOLO dataset directory
settings.update({'datasets_dir': DEFAULT_DATASET_DIR})


def main():
    parser = ArgumentParser(description="Train YOLOv8")
    # data options
    parser.add_argument('--download_dataset',
                        action="store_true",
                        default=False,
                        help="Whether to download RSNA dataset. "
                             "If specified, you should provide Kaggle credentials as well. "
                             "You should also participate in RSNA Pneumonia Detection Challenge")
    parser.add_argument('--train_images',
                        type=str,
                        default=DEFAULT_TRAIN_IMAGES_PATH,
                        help="Path to folder containing training images in DICOM format")
    parser.add_argument('--csv_annotations',
                        type=str,
                        default=DEFAULT_CSV_ANNOTATIONS_PATH,
                        help="Path to file containing annotations for training images")
    parser.add_argument("--warm_start",
                        action="store_true",
                        default=False,
                        help="If specified, model will be trained without data preparation. "
                             "Use only if data is already extracted and processed")
    parser.add_argument("--kaggle_creds",
                        type=str,
                        default=".kaggle/",
                        help="Path to folder containing Kaggle credentials as kaggle.json file.")

    # train options
    parser.add_argument('--epochs', type=int, default=20, help="Number of epochs to train for")
    parser.add_argument('--imgsz', type=int, default=640, help="Size of input images as integer")
    parser.add_argument('--model_name', type=str, default='n',
                        help="Single letter to indicate YOLOv8 model name and size (can be on of the `n`, `s`, `m`, `l`, `x`")

    args = parser.parse_args()

    # download RSNA dataset
    if args.download_dataset:
        download_rsna_dataset(kaggle_creds=args.kaggle_creds)

    if not args.warm_start:
        # convert annotations to YOLOv8 format
        print("Creating YOLOv8 annotations for training data...")
        create_annotation_folder_from_csv(csv_annotation_path=args.csv_annotations,
                                          annotation_folder_path=DEFAULT_YOLO_ANNOTATIONS_PATH)
        # convert DICOM images to JPEG
        print("Converting DICOM images to JPEG...")
        convert_dicom_to_jpeg(dicom_folder_path=args.train_images,
                              jpeg_folder_path=DEFAULT_JPEG_IMAGES_PATH)
        # split train data into train, val and test
        print("Splitting train data...")
        train_val_test_split(images_folder_path=DEFAULT_JPEG_IMAGES_PATH,
                             annotation_folder_path=DEFAULT_YOLO_ANNOTATIONS_PATH,
                             val_frac=0.1)

    # define YOLOv8 model
    model = YOLO(f"yolov8{args.model_name}.yaml")    
    results = model.train(data="data.yaml", 
                          epochs=args.epochs, 
                          pretrained=False, 
                          imgsz=args.imgsz, 
                          verbose=True, 
                          single_cls=True)    
  

if __name__ == '__main__':
    main()
