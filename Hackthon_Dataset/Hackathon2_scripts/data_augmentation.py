#!/usr/bin/env python3
"""
Data augmentation script for VoidVision
Creates orientation variations to improve model accuracy
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import yaml
import shutil
from pathlib import Path

def rotate_image_and_annotations(image_path, label_path, output_dir, angles=[90, 180, 270]):
    """
    Rotate images and update bounding box annotations
    """
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return
    
    h, w = image.shape[:2]
    
    # Read annotations
    annotations = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    annotations.append([class_id, x_center, y_center, width, height])
    
    for angle in angles:
        # Rotate image
        if angle == 90:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            new_h, new_w = w, h
        elif angle == 180:
            rotated_image = cv2.rotate(image, cv2.ROTATE_180)
            new_h, new_w = h, w
        elif angle == 270:
            rotated_image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
            new_h, new_w = w, h
        
        # Update annotations for rotation
        rotated_annotations = []
        for ann in annotations:
            class_id, x_center, y_center, width, height = ann
            
            if angle == 90:
                # 90 degrees clockwise: (x,y) -> (1-y, x)
                new_x_center = 1.0 - y_center
                new_y_center = x_center
                new_width = height
                new_height = width
            elif angle == 180:
                # 180 degrees: (x,y) -> (1-x, 1-y)
                new_x_center = 1.0 - x_center
                new_y_center = 1.0 - y_center
                new_width = width
                new_height = height
            elif angle == 270:
                # 270 degrees: (x,y) -> (y, 1-x)
                new_x_center = y_center
                new_y_center = 1.0 - x_center
                new_width = height
                new_height = width
            
            rotated_annotations.append([
                class_id, new_x_center, new_y_center, new_width, new_height
            ])
        
        # Save rotated image
        base_name = Path(image_path).stem
        rotated_image_path = os.path.join(output_dir, f"{base_name}_rot{angle}.jpg")
        cv2.imwrite(rotated_image_path, rotated_image)
        
        # Save rotated annotations
        rotated_label_path = os.path.join(output_dir, f"{base_name}_rot{angle}.txt")
        with open(rotated_label_path, 'w') as f:
            for ann in rotated_annotations:
                f.write(f"{ann[0]} {ann[1]:.6f} {ann[2]:.6f} {ann[3]:.6f} {ann[4]:.6f}\n")

def augment_dataset(dataset_dir, output_dir):
    """
    Augment the entire dataset with rotations
    """
    print("🔄 Starting dataset augmentation...")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels"), exist_ok=True)
    
    # Process train and val sets
    for split in ["train", "val"]:
        images_dir = os.path.join(dataset_dir, f"{split}_1", split + "1", "images")
        labels_dir = os.path.join(dataset_dir, f"{split}_1", split + "1", "labels")
        
        if not os.path.exists(images_dir):
            print(f"⚠️  Images directory not found: {images_dir}")
            continue
            
        print(f"Processing {split} set...")
        
        # Copy original files
        for filename in os.listdir(images_dir):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Copy original image
                src_img = os.path.join(images_dir, filename)
                dst_img = os.path.join(output_dir, "images", filename)
                shutil.copy2(src_img, dst_img)
                
                # Copy original label if exists
                label_filename = filename.rsplit('.', 1)[0] + '.txt'
                src_label = os.path.join(labels_dir, label_filename)
                dst_label = os.path.join(output_dir, "labels", label_filename)
                if os.path.exists(src_label):
                    shutil.copy2(src_label, dst_label)
                
                # Create rotated versions
                rotate_image_and_annotations(
                    src_img, 
                    src_label if os.path.exists(src_label) else None,
                    os.path.join(output_dir, "images"),
                    angles=[90, 180, 270]
                )
                
                # Create rotated labels
                if os.path.exists(src_label):
                    base_name = filename.rsplit('.', 1)[0]
                    for angle in [90, 180, 270]:
                        label_filename = f"{base_name}_rot{angle}.txt"
                        src_rot_label = os.path.join(labels_dir, label_filename)
                        dst_rot_label = os.path.join(output_dir, "labels", label_filename)
                        if os.path.exists(src_rot_label):
                            shutil.copy2(src_rot_label, dst_rot_label)
    
    print("✅ Dataset augmentation completed!")
    print(f"Augmented dataset saved to: {output_dir}")

def create_improved_yaml_config(dataset_dir, output_file):
    """
    Create improved YAML configuration for training
    """
    config = {
        'path': dataset_dir,
        'train': 'images',
        'val': 'images',  # Use same for now, can be split later
        'test': 'images',
        'nc': 7,
        'names': [
            'OxygenTank',
            'NitrogenTank', 
            'FirstAidBox',
            'FireAlarm',
            'SafetySwitchPanel',
            'EmergencyPhone',
            'FireExtinguisher'
        ]
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ YAML config created: {output_file}")

def main():
    # Dataset paths
    original_dataset = "../train_1"
    augmented_dataset = "../augmented_dataset"
    config_file = "improved_yolo_params.yaml"
    
    print("🚀 VoidVision Data Augmentation")
    print("=" * 50)
    print("🎯 Adding orientation variations")
    print("🎯 Improving Fire Alarm vs Emergency Phone distinction")
    print("🎯 Improving First Aid Box vs Fire Alarm distinction")
    print("=" * 50)
    
    # Augment dataset
    augment_dataset(original_dataset, augmented_dataset)
    
    # Create improved config
    create_improved_yaml_config(augmented_dataset, config_file)
    
    print("\n📊 Augmentation Summary:")
    print(f"  Original dataset: {original_dataset}")
    print(f"  Augmented dataset: {augmented_dataset}")
    print(f"  Config file: {config_file}")
    print("\n🎯 Next steps:")
    print("  1. Run: python improved_training.py --data improved_yolo_params.yaml")
    print("  2. Test the trained model")
    print("  3. Deploy the improved model")

if __name__ == "__main__":
    main()
