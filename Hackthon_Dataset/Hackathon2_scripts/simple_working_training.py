#!/usr/bin/env python3
"""
Simple working training script for VoidVision
Fixes the detection issues and false positives
"""

import os
import sys
from ultralytics import YOLO

def main():
    print("🚀 VoidVision Simple Working Training")
    print("=" * 50)
    print("🎯 Fixing Issues:")
    print("  • No detections (0 Total Detections)")
    print("  • False positive oxygen tank detections")
    print("  • Live camera not working")
    print("=" * 50)
    
    # Change to script directory
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    try:
        # Load model
        print("Loading YOLOv8s model...")
        model = YOLO("yolov8s.pt")
        print("✅ Model loaded successfully!")
        
        # Simple training with working parameters
        print("Starting simple training...")
        results = model.train(
            data="yolo_params.yaml",
            epochs=30,           # Reasonable number of epochs
            device='auto',       # Auto device selection
            batch=8,             # Smaller batch size
            imgsz=640,           # Standard image size
            
            # Basic learning parameters
            lr0=0.001,           # Standard learning rate
            lrf=0.01,
            momentum=0.937,
            optimizer='AdamW',
            weight_decay=0.0005,
            
            # Basic augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,          # Small rotation
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.5,
            mixup=0.0,
            copy_paste=0.0,
            
            # Training stability
            patience=15,         # Reasonable patience
            save_period=10,      # Save every 10 epochs
            workers=4,
            verbose=True,
            
            # Validation
            val=True,
            split='val',
            plots=True,
            
            # Project organization
            project='runs/detect',
            name='voidvision_simple',
            save_dir=None,
            exist_ok=True,
            resume=False
        )
        
        print("✅ Training completed successfully!")
        print(f"Results saved in: {results.save_dir}")
        
        # Test the trained model
        print("\n🧪 Testing trained model...")
        test_model = YOLO(f"{results.save_dir}/weights/best.pt")
        
        print(f"\n📋 Model Summary:")
        print(f"  Classes: {test_model.names}")
        print(f"  Parameters: {sum(p.numel() for p in test_model.model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("Trying minimal training...")
        
        try:
            # Minimal training as fallback
            print("Attempting minimal training...")
            model = YOLO("yolov8s.pt")
            results = model.train(
                data="yolo_params.yaml",
                epochs=10,
                device='auto',
                batch=4,
                imgsz=640,
                verbose=True
            )
            print("✅ Minimal training completed!")
            return True
        except Exception as e2:
            print(f"❌ Minimal training also failed: {str(e2)}")
            return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Training completed!")
        print("📁 Best model: runs/detect/voidvision_simple/weights/best.pt")
        print("\n🔧 Next steps:")
        print("  1. Update web app with new model")
        print("  2. Test detection accuracy")
        print("  3. Fix confidence thresholds")
    else:
        print("\n💥 Training failed. Check error messages above")
