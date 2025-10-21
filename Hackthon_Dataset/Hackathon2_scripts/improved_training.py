#!/usr/bin/env python3
"""
Improved training script for VoidVision with better accuracy
Addresses orientation issues and class confusion
"""

import os
import sys
import argparse
from ultralytics import YOLO
import torch

def main():
    # Enhanced training parameters for better accuracy
    EPOCHS = 100  # More epochs for better learning
    MOSAIC = 0.5  # Reduced mosaic to preserve object details
    OPTIMIZER = 'AdamW'
    MOMENTUM = 0.937
    LR0 = 0.0005  # Lower learning rate for stable training
    LRF = 0.01
    BATCH_SIZE = 16  # Smaller batch size for better gradient updates
    IMGSZ = 640
    
    # Data augmentation parameters for orientation robustness
    HFLIP = 0.5  # Horizontal flip
    VFLIP = 0.0  # No vertical flip (safety equipment orientation matters)
    ROTATE = 15  # Rotation augmentation
    TRANSLATE = 0.1  # Translation augmentation
    SCALE = 0.5  # Scale augmentation
    SHEAR = 0.0  # No shear (preserves object shape)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=IMGSZ, help='Image size')
    parser.add_argument('--device', type=str, default='auto', help='Device (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Change to script directory
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    print("🚀 VoidVision Improved Training")
    print("=" * 60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch}")
    print(f"Image Size: {args.imgsz}")
    print(f"Device: {args.device}")
    print("=" * 60)
    print("🎯 Focus: Fire Alarm vs Emergency Phone distinction")
    print("🎯 Focus: First Aid Box vs Fire Alarm distinction")
    print("🎯 Focus: Orientation robustness (landscape/portrait)")
    print("=" * 60)
    
    try:
        # Load model
        print("Loading YOLOv8s model...")
        model = YOLO("yolov8s.pt")
        print("✅ Model loaded successfully!")
        
        # Enhanced training with better parameters
        print("Starting enhanced training...")
        results = model.train(
            data="yolo_params.yaml",
            epochs=args.epochs,
            device=args.device,
            batch=args.batch,
            imgsz=args.imgsz,
            
            # Learning parameters
            lr0=LR0,
            lrf=LRF,
            momentum=MOMENTUM,
            optimizer=OPTIMIZER,
            weight_decay=0.0005,
            
            # Data augmentation for orientation robustness
            hsv_h=0.015,  # HSV augmentation
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=ROTATE,  # Rotation for orientation robustness
            translate=TRANSLATE,
            scale=SCALE,
            shear=SHEAR,
            perspective=0.0,  # No perspective (preserves object shape)
            flipud=0.0,  # No vertical flip
            fliplr=HFLIP,  # Horizontal flip for orientation robustness
            mosaic=MOSAIC,
            mixup=0.0,  # No mixup (preserves object identity)
            copy_paste=0.0,  # No copy-paste (preserves context)
            
            # Training stability
            patience=30,  # Early stopping patience
            save_period=10,  # Save every 10 epochs
            workers=4,
            verbose=True,
            
            # Validation
            val=True,
            split='val',
            plots=True,
            
            # Class-specific improvements
            cls=0.5,  # Class loss weight
            box=7.5,  # Box loss weight
            dfl=1.5,  # DFL loss weight
            
            # Advanced training techniques
            amp=True,  # Automatic mixed precision
            fraction=1.0,  # Use full dataset
            profile=False,
            freeze=None,  # No freezing
            multi_scale=False,  # No multi-scale (better for safety equipment)
            overlap_mask=True,
            mask_ratio=4,
            drop_path=0.0,
            
            # Resume training if interrupted
            resume=False,
            exist_ok=True,
            
            # Project and name for organization
            project='runs/detect',
            name='voidvision_improved',
            save_dir=None,
        )
        
        print("✅ Training completed successfully!")
        print(f"Results saved in: {results.save_dir}")
        
        # Print final metrics
        if hasattr(results, 'results_dict'):
            print("\n📊 Final Training Metrics:")
            for key, value in results.results_dict.items():
                if 'map' in key.lower():
                    print(f"  {key}: {value:.4f}")
        
        # Test the trained model
        print("\n🧪 Testing trained model...")
        test_model = YOLO(f"{results.save_dir}/weights/best.pt")
        
        # Print model summary
        print(f"\n📋 Model Summary:")
        print(f"  Classes: {test_model.names}")
        print(f"  Parameters: {sum(p.numel() for p in test_model.model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        print("Trying fallback training approach...")
        
        try:
            # Fallback: simpler training
            print("Attempting fallback training...")
            model = YOLO("yolov8s.pt")
            results = model.train(
                data="yolo_params.yaml",
                epochs=50,
                device=args.device,
                batch=8,
                imgsz=640,
                verbose=True,
                patience=20,
                save_period=10
            )
            print("✅ Fallback training completed!")
            return True
        except Exception as e2:
            print(f"❌ Fallback training also failed: {str(e2)}")
            return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Training completed! Check runs/detect/voidvision_improved/ for results")
        print("📁 Best model: runs/detect/voidvision_improved/weights/best.pt")
    else:
        print("\n💥 Training failed. Check error messages above")
