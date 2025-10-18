import os
os.chdir(r'C:\Users\Khushi\VoidVision\Hackthon_Dataset\Hackathon2_scripts')
from ultralytics import YOLO

print("Loading best model from train6 (68.9% mAP)...")
model = YOLO('runs/detect/train6/weights/best.pt')

print("Starting training for 5 more epochs...")
print("Target: 68.9% → 78%+ mAP@0.5")

results = model.train(
    data='yolo_params.yaml', 
    epochs=5, 
    device='cpu',
    mosaic=0.8,
    optimizer='AdamW',
    lr0=0.001,
    lrf=0.01,
    momentum=0.937
)

print("Training completed!")
print("Check the new training folder for results.")
