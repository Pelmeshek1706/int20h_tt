# int20h_tt

*Made by XDBoobs for int20h*

## In this task we should build algorithm that authomatically detects potential pneumonia cases
We want to create quick solution so we choosed YOLOv8 from [Ultralytics](https://docs.ultralytics.com/modes/train/).
As we know, for model training we need to use only labeled data from [dataset](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview), so in our case, all datasize is **6012**
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/f2060400-089b-4d94-afbe-48465179807f)

### Data split:
Train - 80%
Validation - 10%
Test - 10%

### Model config: #still can changing 
Random_seed = 42
Model size = n
Epochs = 30 

### Evaluations: 
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/20d4faeb-a360-4a7d-982e-4d19452a9434)
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/5cf87826-bb1d-464a-bf0f-c18bcc325655)

### Observations:
**Valid Labels**
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/6fc5bc28-5d57-4d21-a830-82cef934f027)

**Valid Predictions**
![image](https://github.com/Pelmeshek1706/int20h_tt/assets/94761102/66a0b531-6eb3-4508-8bea-cb727668e663)


