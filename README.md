# int20h_tt

*Made by XDBoobs for int20h*

## In this task we should build algorithm that authomatically detects potential pneumonia cases
We want to create quick solution so we choosed YOLOv8 from [Ultralytics](https://docs.ultralytics.com/modes/train/).
As we know, for model training we need to use only labeled data from [dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview), so in our case, all datasize is **6012**
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/f2060400-089b-4d94-afbe-48465179807f)

### Data split:
- Train - 80%
- Validation - 10%
- Test - 10%

### Model config: #still can changing 
- Random_seed = 42
- Model size = m
- Epochs = 40

### Evaluations: 
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/903c7549-8a82-4677-b290-17797dc0b06d)
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/ced683e0-3aff-43be-8972-478daeed8d2f)

### Observations:
**Valid Labels**
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/6fc5bc28-5d57-4d21-a830-82cef934f027)

**Valid Predictions**
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/66a0b531-6eb3-4508-8bea-cb727668e663)

----------

# Usage Guide
## For Training model: 
1) <code>pip install --no-cache-dir -r requirements.txt </code>
2) <code>python train.py --download_data --kaggle_creds /content/kaggle</code> (please set up your Kaggle credentials into folder)
3) <code>python train.py --warm_start</code> (if you already have splitted data)

## For evaluating / inferance usage
1) You should set up folder with your images in <code>.dcm</code> format, like this:
   
<img width="213" alt="image" src="https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/0188487b-3635-42fa-b42d-beec8dd04a2c">
   
2)<code>python inference.py --data_path ./datasets/RSNA_data/stage_2_test_images</code> 
3) You can also add <code>--output_file</code> to specify the folder where the submission.csv file will be created (by default the file will be in the project folder)
