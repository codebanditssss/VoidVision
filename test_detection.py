#!/usr/bin/env python3
"""
Test the fixed detection
"""
import os
from ultralytics import YOLO

def test_detection():
    print("🧪 Testing VoidVision Detection")
    print("=" * 40)
    
    # Check for trained models
    model_paths = [
        "runs/detect/voidvision_quick/weights/best.pt",
        "Hackthon_Dataset/Hackathon2_scripts/runs/detect/voidvision_fixed/weights/best.pt",
        "Hackthon_Dataset/Hackathon2_scripts/runs/detect/voidvision_working/weights/best.pt",
        "Hackthon_Dataset/Hackathon2_scripts/runs/detect/train11/weights/best.pt"
    ]
    
    model = None
    for path in model_paths:
        if os.path.exists(path):
            try:
                model = YOLO(path)
                print(f"✅ Found trained model: {path}")
                print(f"📋 Classes: {model.names}")
                break
            except:
                continue
    
    if model is None:
        print("⚠️  No trained model found - using base model")
        model = YOLO('yolov8s.pt')
        print("🔧 This explains the detection issues!")
    
    print(f"📋 Model classes: {model.names}")
    
    # Check if it's the right model
    if 'OxygenTank' in str(model.names) or 'FireAlarm' in str(model.names):
        print("✅ Model knows about safety equipment!")
    else:
        print("❌ Model doesn't know about safety equipment - this is the problem!")
        print("🔧 Training in progress to fix this...")

if __name__ == "__main__":
    test_detection()
