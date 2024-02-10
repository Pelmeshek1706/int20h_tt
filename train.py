from ultralytics import YOLO
from argparse import ArgumentParser
from utils import download_rsna_dataset, create_annotation_folder_from_csv, convert_dicom_to_jpeg, train_val_test_split


DEFAULT_CSV_ANNOTATIONS_PATH = "./datasets/RSNA_data/stage_2_train_labels.csv"
DEFAULT_YOLO_ANNOTATIONS_PATH = "./datasets/RSNA_data/labels/"
DEFAULT_TRAIN_IMAGES_PATH = "./datasets/RSNA_data/train_dicom/"
DEFAULT_TEST_IMAGES_PATH = "./datasets/RSNA_data/test_dicom/"
DEFAULT_JPEG_IMAGES_PATH = "./datasets/RSNA_data/images/"


def main():
    parser = ArgumentParser(description="Train YOLOv8")
    # data options
    parser.add_argument('--download_dataset', action="store_true", default=False)
    parser.add_argument('--train_images', type=str, default=DEFAULT_TRAIN_IMAGES_PATH)
    parser.add_argument('--csv_annotations', type=str, default=DEFAULT_CSV_ANNOTATIONS_PATH)
    parser.add_argument("--warm_start", action="store_true", default=False)

    # train options
    # parser.add_argument()
    args = parser.parse_args()

    # download RSNA dataset
    if args.download_dataset:
        download_rsna_dataset()

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

    # TODO: fit YOLOv8 with data.yaml file
    # define YOLOv8 model
    model = YOLO("yolov8m.yaml")
    results = model.train(data="data.yaml", epochs=40, pretrained=False, imgsz=640, verbose=True, single_cls=True)

if __name__ == '__main__':
    main()