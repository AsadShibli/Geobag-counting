
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

### Before

![Before Image]g)![31](https://github.com/AsadShibli/Geobag-counting/assets/119102237/f5674d5b-4850-46f0-9e13-62f84fb16168)


### After

![After Image](path/to/after_image.jpg)![download](https://github.com/AsadShibli/Geobag-counting/assets/119102237/f02369f0-8e4c-4d69-9690-b410e6cb43a5)


