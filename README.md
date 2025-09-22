# Computer Vision for Road Safety: A KITTI-based Object Detection System

This project develops an object detection system for road safety using the **YOLOv8** deep learning model. The system is trained and evaluated on the **KITTI Vision Benchmark Suite**, a widely used dataset for autonomous driving research. By leveraging pre-trained weights from `yolov8n.pt`, this solution aims to accurately identify and classify critical road objects—such as cars, pedestrians, and cyclists—from camera images, a fundamental task for building robust autonomous driving and advanced driver-assistance systems. The repository includes scripts for data preparation, model training, and inference on both images and videos.

## Setup Procedure

### 1. Clone the Repository
```bash
git clone https://github.com/sriz99/kitti_object_detection.git
```

### 2. Create Conda Environment and install the required libraries
```bash
conda create -n kitti_det python==3.10 -y
conda activate kitti_det
pip install -r requirements.txt
```
### 3. Dataset structure
Download the left camera images dataset from the following link and split the dataset into training and validation set.<br />
https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip <br />
When loading a dataset, the data must be organized as follows:

```
data/
├── train/
    ├── images/   -- Directory containing kitti training images (7000 images)
    ├── labels/    -- Directory containing corresponding labels of the training images in YOLO Format.
├── val/ 
    ├── images/   -- Directory containing kitti validation images (481 images)
    ├── labels/    -- Directory containing corresponding labels of the validation images in YOLO Format.
├── data.yaml -- File containing Data paths necessary for training and validation (Change the paths according to your folder paths)
```

### 4. YOLO Inference on Streamlit APP

For quick interface inference run the following command

```bash
streamlit run app.py ./ckpts/obj_det.pt
```

### 5. Training 

Run the following script to start training on the dataset
```bash
python train.py ./ckpts/yolov8n.pt
```
Once the training has been completed the trained weights are saved under `runs/detect/train/weights/best.pt`. Then run your own inference based on the trained model.

### 6. Evaluation on the trained model

Run the following script to get the evaluation results of the model.  Use the file from previous step to perform inference on the dataset or you can also use `obj_det.pt` file for quick evaluation results.

```bash
python evaluate.py ./ckpts/best.pt
```
