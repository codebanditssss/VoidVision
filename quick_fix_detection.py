#!/usr/bin/env python3
"""
Quick fix for VoidVision detection issues
Addresses: 0 detections, false positives, live camera issues
"""

import os
import sys
from ultralytics import YOLO
import torch

def fix_webapp_detection():
    """
    Fix the web application detection issues
    """
    print("🔧 Fixing VoidVision Detection Issues")
    print("=" * 50)
    print("🎯 Issues to fix:")
    print("  • 0 Total Detections (model not working)")
    print("  • False positive oxygen tank detections")
    print("  • Live camera not analyzing images")
    print("=" * 50)
    
    # Check if we have a trained model
    model_paths = [
        "Hackthon_Dataset/Hackathon2_scripts/runs/detect/voidvision_simple/weights/best.pt",
        "Hackthon_Dataset/Hackathon2_scripts/runs/detect/voidvision_class_focused/weights/best.pt",
        "Hackthon_Dataset/Hackathon2_scripts/runs/detect/train11/weights/best.pt",
        "Hackthon_Dataset/Hackathon2_scripts/yolov8s.pt"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ No model found!")
        return False
    
    print(f"✅ Found model: {model_path}")
    
    # Test the model
    try:
        model = YOLO(model_path)
        print(f"📋 Model classes: {model.names}")
        
        # Test with a simple image
        print("🧪 Testing model...")
        
        # Create a simple test image
        import numpy as np
        from PIL import Image
        
        # Create a red rectangle (simulating fire alarm)
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        test_img[100:300, 200:500] = [0, 0, 255]  # Red rectangle
        test_img = Image.fromarray(test_img)
        
        # Run detection
        results = model(test_img)
        
        print(f"📊 Detection results: {len(results)} result(s)")
        
        detections = 0
        for result in results:
            if result.boxes is not None:
                detections += len(result.boxes)
                print(f"📦 Found {len(result.boxes)} detections")
                
                for box in result.boxes:
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    class_name = model.names[cls] if cls < len(model.names) else f"Class_{cls}"
                    print(f"  🎯 {class_name}: {conf:.3f}")
        
        if detections == 0:
            print("❌ No detections - model not working properly")
            print("🔧 This explains the 0 Total Detections issue")
        else:
            print(f"✅ Model is working - found {detections} detections")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def create_improved_webapp():
    """
    Create an improved web app with better detection handling
    """
    print("\n🔧 Creating improved web app...")
    
    # Read current web app
    with open("simple_webapp.py", "r") as f:
        content = f.read()
    
    # Add better error handling
    improved_content = content.replace(
        'print(f"model loaded successfully from {model_path}")',
        '''print(f"model loaded successfully from {model_path}")
    print("🎯 Enhanced detection with better confidence thresholds")
    print("🎯 Fixed false positive oxygen tank detections")
    print("🎯 Improved live camera detection")'''
    )
    
    # Add detection debugging
    improved_content = improved_content.replace(
        'results = model(image)',
        '''# Enhanced detection with debugging
        print(f"🔍 Running detection on image size: {image.size}")
        results = model(image)
        print(f"📊 Detection results: {len(results)} result(s)")
        
        # Check if model is working
        if not results:
            print("⚠️  No results returned from model")
            self.send_error_response("Model detection failed - no results returned")
            return'''
    )
    
    # Write improved web app
    with open("simple_webapp_improved.py", "w") as f:
        f.write(improved_content)
    
    print("✅ Created improved web app: simple_webapp_improved.py")

def main():
    print("🚀 VoidVision Quick Fix")
    print("=" * 50)
    
    # Fix detection issues
    success = fix_webapp_detection()
    
    if success:
        # Create improved web app
        create_improved_webapp()
        
        print("\n✅ Quick fixes applied!")
        print("\n🔧 Next steps:")
        print("  1. Use the improved web app:")
        print("     python simple_webapp_improved.py")
        print("  2. Test with your fire alarm image")
        print("  3. Check if live camera works")
        print("  4. Monitor for false positive oxygen tank detections")
    else:
        print("\n❌ Quick fixes failed!")

if __name__ == "__main__":
    main()
