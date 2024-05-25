
# Geobag Counting Project

This project involves detecting and counting geobags using the YOLOv8 model. The total counts are displayed on the resulting images. The dataset was collected from a remote location in Bangladesh. This README file provides instructions on how to set up and run the project. The dataset, which has been uploaded for practice, was manually annotated and converted to the YOLO format. Google Colab was used to access a free GPU for the computations.

## The dataset has been uploaded for practice.

## Requirements

- Google Colab
- Google Drive
- Python libraries:
  - numpy
  - pandas
  - matplotlib
  - tensorflow
  - cv2 (OpenCV)
  - ultralytics (for YOLOv8)

## Setup

1. **Mount Google Drive**: Mount  Google Drive to access the dataset and save the model checkpoints.
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

2. **Install Required Libraries**: Install the `ultralytics` package.
    ```python
    !pip install ultralytics
    ```

3. **Import Libraries**: Import necessary libraries for the project.
    ```python
    import numpy as np
    import pandas as pd
    import os
    from glob import glob
    import matplotlib.pyplot as plt
    import random
    import tensorflow as tf
    import cv2
    from ultralytics import YOLO
    from IPython.display import Javascript
    from PIL import Image, ImageFont, ImageDraw
    from google.colab.patches import cv2_imshow
    ```

## Dataset

The dataset is located in  Google Drive  the path:
```/content/drive/MyDrive/Colab Notebooks/(1) Computer Vision (2023)/yolo_data ```

## Configuration Files

### Training Configuration

Create a training configuration file `train_config.yaml` for the YOLOv8 model.

```yaml
path: '/content/drive/MyDrive/Colab Notebooks/(1) Computer Vision (2023)/yolo_data'
train: data/images/train  # train images (relative to 'path')
val: data/images/test  # val images (relative to 'path')

# Classes
names:
  0: geobag
```
### Validation Configuration
Create a validation configuration file val_config.yaml for the YOLOv8 model.
```yaml
path: '/content/drive/MyDrive/Colab Notebooks/(1) Computer Vision (2023)/yolo_data'
train: data/images/train  # train images (relative to 'path')
val: data/images/test  # val images (relative to 'path')
```


## Before and After Images

Below are the before and after images of geobag detection and counting.

<div style="display: flex; justify-content: space-between;">
  <div style="flex: 1; text-align: center;">
    <h3>Before</h3>
    <img src="https://github.com/AsadShibli/Geobag-counting/assets/119102237/62fe2c82-4af8-438a-b6f2-e5bc87fb0d6e" alt="Before Image" style="width: 100%; max-width: 400px;">
  </div>
  <div style="flex: 1; text-align: center;">
    <h3>After</h3>
    <img src="https://github.com/AsadShibli/Geobag-counting/assets/119102237/928d7a41-123e-42ce-a07a-b6dd7cbae079" alt="After Image" style="width: 100%; max-width: 400px;">
  </div>
</div>




