#!/usr/bin/env python3
"""
Class-specific training improvements for VoidVision
Addresses specific confusion between Fire Alarm, Emergency Phone, and First Aid Box
"""

import os
import sys
import argparse
from ultralytics import YOLO
import torch
import yaml

def create_class_weights_config():
    """
    Create class weights to address specific confusion issues
    """
    # Higher weights for classes that are commonly confused
    class_weights = {
        'FireAlarm': 2.0,      # Higher weight - often confused with EmergencyPhone
        'EmergencyPhone': 2.0, # Higher weight - often confused with FireAlarm  
        'FirstAidBox': 2.0,    # Higher weight - often confused with FireAlarm
        'OxygenTank': 1.0,
        'NitrogenTank': 1.0,
        'SafetySwitchPanel': 1.0,
        'FireExtinguisher': 1.0
    }
    
    return class_weights

def create_focused_training_config():
    """
    Create training configuration focused on problematic classes
    """
    config = {
        'path': '../train_1',
        'train': 'train1/images',
        'val': 'val1/images', 
        'test': '../test3/images',
        'nc': 7,
        'names': [
            'OxygenTank',
            'NitrogenTank',
            'FirstAidBox', 
            'FireAlarm',
            'SafetySwitchPanel',
            'EmergencyPhone',
            'FireExtinguisher'
        ],
        
        # Class-specific training parameters
        'class_weights': create_class_weights_config(),
        
        # Focus on problematic classes
        'focus_classes': ['FireAlarm', 'EmergencyPhone', 'FirstAidBox'],
        
        # Enhanced augmentation for orientation
        'augment': {
            'hsv_h': 0.015,
            'hsv_s': 0.7, 
            'hsv_v': 0.4,
            'degrees': 15,      # Rotation for orientation robustness
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,       # No shear - preserves object shape
            'perspective': 0.0,  # No perspective - preserves object shape
            'flipud': 0.0,      # No vertical flip
            'fliplr': 0.5,      # Horizontal flip for orientation
            'mosaic': 0.5,      # Reduced mosaic
            'mixup': 0.0,       # No mixup - preserves object identity
            'copy_paste': 0.0   # No copy-paste - preserves context
        }
    }
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Class-specific training for VoidVision')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='auto', help='Device')
    parser.add_argument('--data', type=str, default='yolo_params.yaml', help='Data config file')
    
    args = parser.parse_args()
    
    # Change to script directory
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    print("🎯 VoidVision Class-Specific Training")
    print("=" * 60)
    print("🔧 Focus Areas:")
    print("  • Fire Alarm vs Emergency Phone distinction")
    print("  • First Aid Box vs Fire Alarm distinction") 
    print("  • Orientation robustness (landscape/portrait)")
    print("  • Class-specific loss weighting")
    print("=" * 60)
    
    try:
        # Load model
        print("Loading YOLOv8s model...")
        model = YOLO("yolov8s.pt")
        print("✅ Model loaded successfully!")
        
        # Create focused training configuration
        config = create_focused_training_config()
        config_file = 'focused_training_config.yaml'
        
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"📝 Created focused config: {config_file}")
        
        # Enhanced training with class-specific improvements
        print("Starting class-specific training...")
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            device=args.device,
            batch=args.batch,
            imgsz=args.imgsz,
            
            # Learning parameters optimized for class distinction
            lr0=0.0003,          # Lower learning rate for stable learning
            lrf=0.01,
            momentum=0.937,
            optimizer='AdamW',
            weight_decay=0.0005,
            
            # Class-specific loss weights
            cls=1.0,             # Higher class loss weight
            box=7.5,             # Box loss weight
            dfl=1.5,             # DFL loss weight
            
            # Enhanced augmentation for orientation robustness
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=15,          # Rotation for orientation
            translate=0.1,
            scale=0.5,
            shear=0.0,           # No shear - preserves shape
            perspective=0.0,      # No perspective - preserves shape
            flipud=0.0,          # No vertical flip
            fliplr=0.5,          # Horizontal flip for orientation
            mosaic=0.5,          # Reduced mosaic
            mixup=0.0,           # No mixup - preserves identity
            copy_paste=0.0,      # No copy-paste - preserves context
            
            # Training stability
            patience=40,         # Higher patience for convergence
            save_period=15,      # Save every 15 epochs
            workers=4,
            verbose=True,
            
            # Validation and monitoring
            val=True,
            split='val',
            plots=True,
            
            # Advanced techniques
            amp=True,            # Mixed precision
            fraction=1.0,        # Full dataset
            profile=False,
            freeze=None,         # No freezing
            multi_scale=False,   # No multi-scale for safety equipment
            overlap_mask=True,
            mask_ratio=4,
            
            # Project organization
            project='runs/detect',
            name='voidvision_class_focused',
            save_dir=None,
            exist_ok=True,
            resume=False
        )
        
        print("✅ Class-specific training completed!")
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
        
        print(f"\n📋 Model Summary:")
        print(f"  Classes: {test_model.names}")
        print(f"  Parameters: {sum(p.numel() for p in test_model.model.parameters()):,}")
        
        # Create evaluation script
        eval_script = f"""
# Evaluation script for the trained model
from ultralytics import YOLO

# Load the trained model
model = YOLO('{results.save_dir}/weights/best.pt')

# Test on specific problematic cases
test_images = [
    'path/to/fire_alarm_image.jpg',
    'path/to/emergency_phone_image.jpg', 
    'path/to/first_aid_box_image.jpg'
]

for img_path in test_images:
    results = model(img_path)
    print(f"Image: {{img_path}}")
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[class_id]
            print(f"  Detected: {{class_name}} (confidence: {{confidence:.3f}})")
"""
        
        with open('evaluate_model.py', 'w') as f:
            f.write(eval_script)
        
        print("\n📝 Created evaluation script: evaluate_model.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 Class-specific training completed!")
        print("📁 Best model: runs/detect/voidvision_class_focused/weights/best.pt")
        print("\n🔧 Next steps:")
        print("  1. Test the model on problematic images")
        print("  2. Run: python evaluate_model.py")
        print("  3. Update the web app to use the new model")
    else:
        print("\n💥 Training failed. Check error messages above")
