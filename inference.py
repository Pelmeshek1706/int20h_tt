from ultralytics import YOLO
from argparse import ArgumentParser
from utils import convert_dicom_to_jpeg, process_inference_results
import os
import shutil

JPEG_IMAGES_PATH = "./tmp_jpeg_images/"


def main():
    parser = ArgumentParser("Get predictions with trained YOLOv8 model")
    parser.add_argument("--data_path",
                        type=str,
                        required=True,
                        help="Path to data directory with DICOM files")
    parser.add_argument("--weights",
                        type=str,
                        default="weights/yolov8.pt",
                        help="Path to weights file")
    parser.add_argument("--output_file",
                        type=str,
                        default="submission.csv",
                        help="Path to file with inference results")
    args = parser.parse_args()

    # convert DICOM images to JPEG
    print("Converting DICOM images to JPEG...")
    convert_dicom_to_jpeg(dicom_folder_path=args.data_path,
                          jpeg_folder_path=JPEG_IMAGES_PATH)

    # define YOLOv8 model and load weights
    model = YOLO(args.weights)

    # run predictions
    results = model.predict(JPEG_IMAGES_PATH, save=True)

    with open(args.output_file, "a", encoding="utf-8") as submission_file:
        # add header
        submission_file.write("patientId,PredictionString\n")
        for result in results:
            head, tail = os.path.split(result.path)
            patient_id = tail.split(".")[0]

            prediction_string = process_inference_results(result.boxes.conf.tolist(), result.boxes.xywh.tolist())

            submission_file.write(f"{patient_id},{prediction_string}\n")

    shutil.rmtree(JPEG_IMAGES_PATH)


if __name__ == "__main__":
    main()
