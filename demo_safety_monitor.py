#!/usr/bin/env python3
"""
voidvision space station safety monitor - demo script
this script demonstrates the safety monitoring capabilities
"""

import os
import sys
from ultralytics import YOLO
import cv2
import json
from datetime import datetime

def load_model():
    """load the trained yolov8 model"""
    try:
        model_path = "yolov8s.pt"
        if os.path.exists(model_path):
            print("loading yolov8 model...")
            model = YOLO(model_path)
            print("model loaded successfully!")
            return model
        else:
            print("error: model file not found")
            return None
    except Exception as e:
        print(f"error loading model: {str(e)}")
        return None

def detect_safety_equipment(model, image_path, confidence=0.5):
    """detect safety equipment in an image"""
    try:
        # load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"error: could not load image {image_path}")
            return None
        
        # run detection
        results = model(image, conf=confidence)
        
        # process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    class_name = model.names[cls]
                    bbox = box.xyxy[0].tolist()
                    
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': bbox
                    })
        
        return detections
    
    except Exception as e:
        print(f"error during detection: {str(e)}")
        return None

def check_equipment_status(detections, required_equipment):
    """check if required equipment is present"""
    detected_classes = [det['class'].lower() for det in detections]
    
    status = {}
    for equipment in required_equipment:
        status[equipment] = equipment.lower() in detected_classes
    
    return status

def generate_safety_report(detections, equipment_status, required_equipment):
    """generate a safety report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_detections': len(detections),
        'detected_equipment': [det['class'] for det in detections],
        'equipment_status': equipment_status,
        'missing_equipment': [eq for eq in required_equipment if not equipment_status.get(eq, False)],
        'safety_score': len([eq for eq in required_equipment if equipment_status.get(eq, False)]) / len(required_equipment) * 100
    }
    
    return report

def main():
    """main demo function"""
    print("🚀 voidvision space station safety monitor - demo")
    print("=" * 50)
    
    # load model
    model = load_model()
    if model is None:
        print("failed to load model. exiting.")
        return
    
    # required safety equipment
    required_equipment = [
        'oxygentank',
        'nitrogentank', 
        'firstaidbox',
        'firealarm',
        'safetyswitchpanel',
        'emergencyphone',
        'fireextinguisher'
    ]
    
    print(f"\nmonitoring {len(required_equipment)} types of safety equipment:")
    for i, equipment in enumerate(required_equipment, 1):
        print(f"  {i}. {equipment}")
    
    # demo with test images (if available)
    test_images = [
        "test_image1.jpg",
        "test_image2.jpg", 
        "sample_space_station.jpg"
    ]
    
    print(f"\n🔍 scanning for test images...")
    available_images = [img for img in test_images if os.path.exists(img)]
    
    if available_images:
        print(f"found {len(available_images)} test images")
        
        for image_path in available_images:
            print(f"\n📸 analyzing: {image_path}")
            
            # detect equipment
            detections = detect_safety_equipment(model, image_path)
            
            if detections:
                print(f"detected {len(detections)} objects:")
                for det in detections:
                    print(f"  - {det['class']} (confidence: {det['confidence']:.2f})")
                
                # check equipment status
                equipment_status = check_equipment_status(detections, required_equipment)
                
                # generate report
                report = generate_safety_report(detections, equipment_status, required_equipment)
                
                print(f"\n📊 safety report:")
                print(f"  safety score: {report['safety_score']:.1f}%")
                print(f"  detected equipment: {len(report['detected_equipment'])}")
                
                if report['missing_equipment']:
                    print(f"  ⚠️  missing equipment: {', '.join(report['missing_equipment'])}")
                else:
                    print(f"  ✅ all required equipment detected")
                
                # save report
                report_file = f"safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_file, 'w') as f:
                    json.dump(report, f, indent=2)
                print(f"  report saved to: {report_file}")
                
            else:
                print("  no safety equipment detected")
    else:
        print("no test images found. demo completed.")
        print("\nto test with real images:")
        print("1. place test images in the current directory")
        print("2. run this script again")
        print("3. or use the streamlit app: streamlit run safety_monitor_app.py")
    
    print(f"\n🎯 demo completed!")
    print(f"for interactive monitoring, run: streamlit run safety_monitor_app.py")

if __name__ == "__main__":
    main()
