# INT20H Test Task

*Made by XDBoobs team for INT20H*

## Table of content
- [Overview](#overview)
- - [Data preparation](#data-preparation)
- - [Training pipeline and configurations](#training-pipeline-and-configurations)
- - [Evaluations](#evaluations)
- - [Observations](#observations)
- [Usage guide](#usage-guide)
- - [Training](#training)
- - [Inference](#inference)
- [Team members](#team-members)

## Overview
This repository harnesses the power of YOLOv8 by [Ultralytics](https://docs.ultralytics.com/),
a state-of-the-art object detection algorithm, to identify pneumonia  in chest radiographs with
high accuracy and efficiency using
[RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/overview)
dataset. Leveraging the robustness and speed of YOLOv8, this project provides a streamlined
solution for healthcare professionals to quickly screen large volumes of chest X-ray images for signs of pneumonia.

### Data preparation
To train YOLOv8 model on the RSNA data, we converted training images from DICOM format to JPEG
To reduce training time, we selected only images that contain bounding boxes from original training set (6012 images
in total). This data was split into train, validation and test sets:
- Train - 80% within labeled images (4810 images in total)
- Validation - 10% within labeled images (601 images in total)
- Test - 10% within labeled images (601 image) + the same number of images from unlabeled data (1203 images in total)

### Training pipeline and configurations
The full training process consists of downloading the RSNA data, unpacking, processing it (converting from DICOM
format to JPEG), splitting and, finally, training YOLOv8 model. We used default training parameter
values except for the number of epochs (which was set to 40) and model size (which was set to m). You may specify
your own training parameters, for more information see [Usage Guide](#training)

### Evaluations
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/903c7549-8a82-4677-b290-17797dc0b06d)
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/ced683e0-3aff-43be-8972-478daeed8d2f)

### Observations
**Valid Labels**
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/6fc5bc28-5d57-4d21-a830-82cef934f027)

**Valid Predictions**
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/66a0b531-6eb3-4508-8bea-cb727668e663)

----------

## Usage guide
Python version: 3.11
1. Clone the repository:
```bash
git clone https://github.com/Pelmeshek1706/int20h_tt.git
```
2. Install all necessary packages:
```bash
pip install --no-cache-dir -r requirements.txt
```

### Training

Run train script:
```bash
python train.py [OPTIONS]
```
You may provide the following options to run the script:
>| Key                | Type               | Default                                         | Description                                                                                                                                                                      |
>|--------------------|--------------------|-------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
>| `download_dataset` | Optional, flag     | `False`                                         | Whether to download RSNA dataset from Kaggle. You If specified, you should provide Kaggle credentials as well. You should also participate in RSNA Pneumonia Detection Challenge |
>| `train_images`     | Optional, argument | `./datasets/RSNA_data/train_dicom/`             | Path to folder containing training images in DICOM format. Only needed when custom training data is used                                                                         |
>| `csv_annotations`  | Optional, argument | `./datasets/RSNA_data/stage_2_train_labels.csv` | Path to file containing annotations for training images. Only needed when custom training data is used                                                                           |
>| `warm_start`       | Optional, flag     | `False`                                         | If specified, model will be trained without data preparation. Use only if data is already extracted and processed                                                                |
>| `kaggle_creds`     | Optional, argument | `.kaggle/`                                      | Path to folder containing Kaggle credentials as kaggle.json file. Required if `download_dataset` is specified                                                                    |
>| `epochs`           | Optional, argument | `20`                                            | Number of epochs to train for                                                                                                                                                    |
>| `imgsz`            | Optional, argument | `640`                                           | Size of input images as integer                                                                                                                                                  |
>| `model_name`       | Optional, argument | `m`                                             | Single letter to indicate YOLOv8 model name and size (can be on of the `n`, `s`, `m`, `l`, `x`)                                                                                  |                                                                                  |

Examples:
```bash
# download dataset and train model with default training parameters
python train.py --download_dataset --kaggle_creds kaggle/
```
```bash
# train model in case the data is already downloaded and unpacked
python train.py --warm_start
```

### Inference

Run inference script:
```bash
python inference.py [OPTIONS]
```

You may provide the following options to run the script:
>| Key           | Type               | Default             | Description                                         |
>|---------------|--------------------|---------------------|-----------------------------------------------------|
>| `data_path`   | Required, argument | -                   | Path to data directory with DICOM files             |
>| `weights`     | Optional, argument | `weights/yolov8.pt` | Path to custom weights file                         |
>| `output_file` | Optional, argument | `submission.csv`    | Path to file where inference results will be stored |

Examples:
```bash
python inference.py --data_path ./data/images --weights ./weights/yolo8l.pt --output_file ./results/submission.csv
```

After inference script is done, images with detected pneumonia will be stored in `./runs/detect/predict/`

## Team members
### Team lead
[Github](https://github.com/Pelmeshek1706)
[Telegram](https://t.me/pelmeshek1706)

### The guy who hates Tensorflow
[Github](https://github.com/poopaandloopa)
[Telegram](https://t.me/poopaandloopa)
[LinkedIn](https://www.linkedin.com/in/andrii-krasnyh-a4828a205/)

### The guy who hates PyTorch
[Github](https://github.com/poluidol2)
[Telegra](https://t.me/poluidol)