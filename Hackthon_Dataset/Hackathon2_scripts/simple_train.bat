@echo off
echo starting training...
cd /d "C:\Users\Khushi\VoidVision\Hackthon_Dataset\Hackathon2_scripts"
C:\Users\Khushi\Anaconda3\envs\EDU\python.exe -c "
import os
os.chdir(r'C:\Users\Khushi\VoidVision\Hackthon_Dataset\Hackathon2_scripts')
from ultralytics import YOLO
print('loading best model from train6...')
model = YOLO('runs/detect/train6/weights/best.pt')
print('starting training for 5 epochs...')
results = model.train(data='yolo_params.yaml', epochs=5, device='cpu')
print('training completed!')
"
echo training finished!
pause
