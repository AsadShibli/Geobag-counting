
# Geobag Counting Project

This project involves detecting and counting geobags using the YOLOv8 model. The total counts are displayed on the resulting images. The dataset was collected from a remote location in Bangladesh. This README file provides instructions on how to set up and run the project. The dataset, which has been uploaded for practice, was manually annotated and converted to the YOLO format. Google Colab was used to access a free GPU for the computations.

## Before and After Images

Below are the before and after images of geobag detection and counting.

<div style="display: flex; justify-content: center;">
  <div style="flex: 1; text-align: center; margin: 10px;">
    <h3>Before</h3>
    <img src="https://github.com/AsadShibli/Geobag-counting/assets/119102237/62fe2c82-4af8-438a-b6f2-e5bc87fb0d6e" alt="Before Image" style="width: 40%; max-width: 400px;">
  </div>
  <div style="flex: 1; text-align: center; margin: 10px;">
    <h3>After</h3>
    <img src="https://github.com/AsadShibli/Geobag-counting/assets/119102237/928d7a41-123e-42ce-a07a-b6dd7cbae079" alt="After Image" style="width: 40%; max-width: 400px;">
  </div>
</div>

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

2. **Install Required Libraries**: Install the `ultralytics` package.
    ```python
    !pip install ultralytics


3. **Navigate to the Project Directory**
Change the working directory to your project folder.
    ```python
    from google.colab import drive
    %cd "/content/drive/MyDrive/Projects/yolo_count"

4. **Check Ultralytics Installation**
Import the ```ultralytics``` package and run a check to ensure everything is set up correctly.
    ```python
    from IPython import display
    display.clear_output()
    
    import ultralytics
    ultralytics.checks()

5. **Dataset Configuration**
The ```data.yml``` file should have the following structure:
    ```yaml
    train: /content/drive/MyDrive/Projects/yolo_count/data/train/images
    val: /content/drive/MyDrive/Projects/yolo_count/data/val/images
    
    nc: 1  # number of classes
    names: ['geobag']  # class names

### Training the Model
Train the YOLOv8 model on your dataset.
    ```python
    
    !yolo task=detect mode=train model=yolov8s.pt data=/content/drive/MyDrive/Projects/yolo_count/data.yml epochs=50 imgsz=512 plots=True

### Viewing Training Results
Display a sample prediction from the validation set to see how the model is performing.
    ```python
    
    from IPython.display import display, Image
    Image(filename='/content/drive/MyDrive/Projects/yolo_count/runs/detect/train14/val_batch0_pred.jpg', width=1200)

### Evaluating the Model
Validate the model to see its performance on the validation set.
    ```python
    
        from IPython.display import display, Image
        !yolo task=detect mode=val model=/content/drive/MyDrive/Projects/yolo_count/runs/detect/train14/weights/best.pt data=/content/drive/MyDrive/Projects/yolo_count/data.yml
    

### Running Inference
Use the trained model to make predictions on the test set.
    ```python
    
    !yolo task=detect mode=predict model=/content/drive/MyDrive/Projects/yolo_count/runs/detect/train14/weights/best.pt data=/content/drive/MyDrive/Projects/yolo_count/data.yml conf=0.25 source=/content/drive/MyDrive/Projects/yolo_count/data/test/images

### Custom Inference with YOLOv8
Load the custom model weights and run inference on a source image.
    ```python
    
        from ultralytics import YOLO
    
        # Load your custom model weights
        custom_weights_path = '/content/drive/MyDrive/Projects/yolo_count/runs/detect/train14/weights/best.pt'
        yolo_model = YOLO(model=custom_weights_path)
        
        source = "/content/drive/MyDrive/Projects/yolo_count/data/test/images"
        # Run inference on the source
        results = yolo_model(source, stream=True)  # generator of Results objects
        
        # Iterate over the generator to view the results
        for result in results:
            print(result)

### Displaying Results
Display the results using OpenCV and PIL.
    ```python
    
      import cv2
      from google.colab.patches import cv2_imshow
      from PIL import Image  # Importing the Image module from the PIL library
      
      # Run batched inference on a list of images
      model = YOLO('/content/drive/MyDrive/Projects/yolo_count/runs/detect/train14/weights/best.pt')
      
      results = model.predict(source='/content/Copy of 31.JPG', conf=0.4, show_labels=True)


### Adding Text to Images
Define a function to add text to an image and display the processed results.L.
    ```python
    
    from PIL import Image, ImageFont, ImageDraw
    
    def add_text(im, text, topleft, size, color):
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", size)
        draw = ImageDraw.Draw(im)
        draw.text(topleft, text, font=font, fill=color) # put the text on the image
        return im
    
    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        c = len(result.boxes.data)  # counting total box

    save_path = '/content/result.jpg'  # Saving result to a fixed file name
    result.save(filename=save_path)  # Save to disk
    c_img = Image.open(save_path)
    c_img = add_text(c_img, "Total {}".format(c), (100, 100), 200, (255, 0, 0))
    c_img.save(save_path)
    img = cv2.imread(save_path)
    img = cv2.resize(img, (800, 600))
    cv2_imshow(img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

## Conclusion

This README file provides a comprehensive guide to setting up and running the geobag detection and counting project using YOLOv8. Follow the steps outlined to train the model, evaluate its performance, and visualize the results.

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests for any improvements or bug fixes.
