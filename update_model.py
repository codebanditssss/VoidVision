#!/usr/bin/env python3
"""
Update VoidVision web app to use improved trained model
"""

import os
import shutil
from pathlib import Path

def update_webapp_model():
    """
    Update the web application to use the improved trained model
    """
    print(" Updating VoidVision web app with improved model...")
    
    # Paths
    script_dir = os.path.dirname(__file__)
    training_dir = os.path.join(script_dir, "Hackthon_Dataset", "Hackathon2_scripts")
    
    # Look for the best trained model
    possible_model_paths = [
        os.path.join(training_dir, "runs", "detect", "voidvision_class_focused", "weights", "best.pt"),
        os.path.join(training_dir, "runs", "detect", "voidvision_improved", "weights", "best.pt"),
        os.path.join(training_dir, "runs", "detect", "train11", "weights", "best.pt"),
        os.path.join(training_dir, "yolov8s.pt")  # Fallback to base model
    ]
    
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print(" No trained model found!")
        return False
    
    print(f" Found model: {model_path}")
    
    # Update the web app code
    webapp_file = os.path.join(script_dir, "simple_webapp.py")
    
    if not os.path.exists(webapp_file):
        print(f" Web app file not found: {webapp_file}")
        return False
    
    # Read the current web app
    with open(webapp_file, 'r') as f:
        content = f.read()
    
    # Update the model loading section
    new_model_loading = f'''# Load the trained model
model_path = "{model_path}"
if os.path.exists(model_path):
    model = YOLO(model_path)
    print(f" Improved model loaded successfully from {{model_path}}")
    print(" Enhanced accuracy for Fire Alarm vs Emergency Phone")
    print(" Enhanced accuracy for First Aid Box vs Fire Alarm") 
    print(" Enhanced orientation robustness")
else:
    print("  Improved model not found, using base model")
    model = YOLO('yolov8s.pt')
    print("using base yolov8s model")'''
    
    # Replace the model loading section
    import re
    pattern = r'# Load the trained model.*?print\("using base yolov8s model"\)'
    content = re.sub(pattern, new_model_loading, content, flags=re.DOTALL)
    
    # Write the updated web app
    with open(webapp_file, 'w') as f:
        f.write(content)
    
    print(" Web app updated with improved model!")
    
    # Create a backup
    backup_file = webapp_file + ".backup"
    shutil.copy2(webapp_file, backup_file)
    print(f" Backup created: {backup_file}")
    
    return True

def create_model_evaluation_script():
    """
    Create a script to evaluate the model's performance
    """
    eval_script = '''#!/usr/bin/env python3
"""
Model evaluation script for VoidVision
Tests accuracy on specific problematic cases
"""

import os
import sys
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

def evaluate_model(model_path, test_images_dir):
    """
    Evaluate the model on test images
    """
    print("🧪 Evaluating VoidVision model...")
    
    # Load model
    model = YOLO(model_path)
    print(f" Model loaded: {model_path}")
    
    # Test images
    test_cases = [
        "fire_alarm_horizontal.jpg",
        "fire_alarm_vertical.jpg", 
        "emergency_phone_horizontal.jpg",
        "emergency_phone_vertical.jpg",
        "first_aid_box_horizontal.jpg",
        "first_aid_box_vertical.jpg"
    ]
    
    results = {}
    
    for test_image in test_cases:
        image_path = os.path.join(test_images_dir, test_image)
        if not os.path.exists(image_path):
            print(f"  Test image not found: {image_path}")
            continue
            
        print(f"\\n🔍 Testing: {test_image}")
        
        # Run detection
        results_det = model(image_path)
        
        # Process results
        detections = []
        for r in results_det:
            if r.boxes is not None:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[class_id]
                    detections.append({
                        'class': class_name,
                        'confidence': confidence
                    })
        
        results[test_image] = detections
        
        # Print results
        if detections:
            for det in detections:
                print(f"   {det['class']}: {det['confidence']:.3f}")
        else:
            print("   No detections")
    
    return results

def main():
    # Model path
    model_path = "Hackthon_Dataset/Hackathon2_scripts/runs/detect/voidvision_class_focused/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return
    
    # Test images directory
    test_dir = "test_images"
    if not os.path.exists(test_dir):
        print(f"  Test directory not found: {test_dir}")
        print("Create test_images/ directory with test images")
        return
    
    # Evaluate model
    results = evaluate_model(model_path, test_dir)
    
    print("\\n📊 Evaluation Summary:")
    print("=" * 50)
    
    # Analyze results
    correct_detections = 0
    total_detections = 0
    
    for image, detections in results.items():
        print(f"\\n{image}:")
        for det in detections:
            total_detections += 1
            # Check if detection is correct based on filename
            expected_class = image.split('_')[0].replace('first', 'FirstAidBox').replace('fire', 'FireAlarm').replace('emergency', 'EmergencyPhone')
            if det['class'] == expected_class and det['confidence'] > 0.5:
                correct_detections += 1
                print(f"   Correct: {det['class']} ({det['confidence']:.3f})")
            else:
                print(f"  Incorrect: {det['class']} ({det['confidence']:.3f})")
    
    accuracy = correct_detections / total_detections if total_detections > 0 else 0
    print(f"\\n🎯 Overall Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()
'''
    
    with open("evaluate_model.py", 'w') as f:
        f.write(eval_script)
    
    print(" Created model evaluation script: evaluate_model.py")

def main():
    print(" VoidVision Model Update")
    print("=" * 50)
    
    # Update web app
    success = update_webapp_model()
    
    if success:
        # Create evaluation script
        create_model_evaluation_script()
        
        print("\n Model update completed!")
        print("\n🔧 Next steps:")
        print("  1. Train the improved model:")
        print("     cd Hackthon_Dataset/Hackathon2_scripts")
        print("     python class_specific_training.py")
        print("  2. Test the model:")
        print("     python evaluate_model.py")
        print("  3. Restart the web app:")
        print("     python simple_webapp.py")
    else:
        print("\n Model update failed!")

if __name__ == "__main__":
    main()
