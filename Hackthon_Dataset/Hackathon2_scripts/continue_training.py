import os
import sys

# change to the correct directory
os.chdir(r"C:\Users\Khushi\VoidVision\Hackthon_Dataset\Hackathon2_scripts")

# add the conda environment to path
sys.path.insert(0, r"C:\Users\Khushi\Anaconda3\envs\EDU\Lib\site-packages")

from ultralytics import YOLO

print("starting training from train11 best model...")

# load the best model from train11
best_model_path = os.path.join(os.getcwd(), "runs", "detect", "train11", "weights", "best.pt")
if os.path.exists(best_model_path):
    print(f"continuing training from best model: {best_model_path}")
    model = YOLO(best_model_path)
else:
    print("using base yolov8s model")
    model = YOLO(os.path.join(os.getcwd(), "yolov8s.pt"))

# train for 5 more epochs
results = model.train(
    data=os.path.join(os.getcwd(), "yolo_params.yaml"), 
    epochs=5,
    device='cpu',
    single_cls=False, 
    mosaic=0.8,
    optimizer='AdamW', 
    lr0=0.001, 
    lrf=0.01, 
    momentum=0.937
)

print("training completed!")
