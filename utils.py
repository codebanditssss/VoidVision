import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import json
from datetime import datetime

class SafetyEquipmentDetector:
    def __init__(self, model_path):
        """
        initialize the safety equipment detector
        
        args:
            model_path (str): path to the trained yolov8 model
        """
        self.model_path = model_path
        self.model = None
        self.equipment_classes = [
            "oxygenTank", "nitrogenTank", "firstAidBox", 
            "fireAlarm", "safetySwitchPanel", "emergencyPhone", "fireExtinguisher"
        ]
        self.load_model()
    
    def load_model(self):
        """load the yolov8 model"""
        try:
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                print(f"model loaded successfully from {self.model_path}")
            else:
                print(f"model file not found at {self.model_path}")
                # try alternative path
                alt_path = "Hackthon_Dataset/Hackathon2_scripts/runs/detect/train11/weights/best.pt"
                if os.path.exists(alt_path):
                    self.model = YOLO(alt_path)
                    print(f"model loaded from alternative path: {alt_path}")
                else:
                    raise FileNotFoundError("model file not found")
        except Exception as e:
            print(f"error loading model: {str(e)}")
            self.model = None
    
    def detect_equipment(self, image, confidence_threshold=0.5):
        """
        detect safety equipment in an image
        
        args:
            image: pil image or numpy array
            confidence_threshold (float): minimum confidence for detections
            
        returns:
            dict: detection results
        """
        if self.model is None:
            return {"error": "model not loaded"}
        
        try:
            # run detection
            results = self.model(image, conf=confidence_threshold)
            
            # process results
            detections = []
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.data.cpu().numpy()
                
                for box in boxes:
                    x1, y1, x2, y2, conf, cls = box
                    class_name = self.equipment_classes[int(cls)]
                    
                    detections.append({
                        "equipment": class_name,
                        "confidence": float(conf),
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "area": float((x2 - x1) * (y2 - y1))
                    })
            
            # get annotated image
            annotated_image = None
            if len(results) > 0:
                annotated_image = results[0].plot()
            
            return {
                "detections": detections,
                "total_detections": len(detections),
                "annotated_image": annotated_image,
                "success": True
            }
            
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def calculate_safety_score(self, detections):
        """
        calculate safety score based on detected equipment
        
        args:
            detections (list): list of detection dictionaries
            
        returns:
            dict: safety score and analysis
        """
        if not detections:
            return {"score": 0, "status": "critical", "message": "no equipment detected"}
        
        # count equipment by type
        equipment_counts = {}
        for detection in detections:
            equipment = detection["equipment"]
            equipment_counts[equipment] = equipment_counts.get(equipment, 0) + 1
        
        # expected equipment counts (realistic for space station module)
        expected_counts = {
            "oxygenTank": 1,
            "nitrogenTank": 1,
            "firstAidBox": 1,
            "fireAlarm": 1,
            "safetySwitchPanel": 1,
            "emergencyPhone": 1,
            "fireExtinguisher": 2
        }
        
        # calculate score
        total_expected = sum(expected_counts.values())
        total_detected = sum(equipment_counts.values())
        
        # base score from detection ratio
        detection_ratio = total_detected / total_expected if total_expected > 0 else 0
        
        # penalty for missing critical equipment
        missing_penalty = 0
        critical_equipment = ["fireAlarm", "fireExtinguisher", "emergencyPhone"]
        
        for equipment in critical_equipment:
            expected = expected_counts.get(equipment, 0)
            detected = equipment_counts.get(equipment, 0)
            if detected < expected:
                missing_penalty += (expected - detected) * 0.1
        
        # final score
        safety_score = max(0, min(100, (detection_ratio * 100) - (missing_penalty * 100)))
        
        # determine status
        if safety_score >= 80:
            status = "excellent"
        elif safety_score >= 60:
            status = "good"
        elif safety_score >= 40:
            status = "fair"
        else:
            status = "poor"
        
        return {
            "score": round(safety_score, 1),
            "status": status,
            "detection_ratio": round(detection_ratio, 3),
            "equipment_counts": equipment_counts,
            "missing_equipment": self._get_missing_equipment(equipment_counts, expected_counts),
            "message": f"safety score: {safety_score:.1f}% ({status})"
        }
    
    def _get_missing_equipment(self, detected, expected):
        """get list of missing equipment"""
        missing = []
        for equipment, expected_count in expected.items():
            detected_count = detected.get(equipment, 0)
            if detected_count < expected_count:
                missing.append({
                    "equipment": equipment,
                    "expected": expected_count,
                    "detected": detected_count,
                    "missing": expected_count - detected_count
                })
        return missing
    
    def get_detection_summary(self, detections):
        """get summary statistics for detections"""
        if not detections:
            return {"total": 0, "by_class": {}, "avg_confidence": 0}
        
        by_class = {}
        total_confidence = 0
        
        for detection in detections:
            equipment = detection["equipment"]
            confidence = detection["confidence"]
            
            if equipment not in by_class:
                by_class[equipment] = {"count": 0, "avg_confidence": 0, "confidences": []}
            
            by_class[equipment]["count"] += 1
            by_class[equipment]["confidences"].append(confidence)
            total_confidence += confidence
        
        # calculate average confidences
        for equipment in by_class:
            confidences = by_class[equipment]["confidences"]
            by_class[equipment]["avg_confidence"] = round(sum(confidences) / len(confidences), 3)
            del by_class[equipment]["confidences"]  # remove raw confidences
        
        return {
            "total": len(detections),
            "by_class": by_class,
            "avg_confidence": round(total_confidence / len(detections), 3)
        }

# utility functions
def validate_image(image):
    """validate uploaded image"""
    try:
        if isinstance(image, str):
            image = Image.open(image)
        
        # check image size
        width, height = image.size
        if width < 100 or height < 100:
            return False, "image too small (minimum 100x100 pixels)"
        
        if width > 4000 or height > 4000:
            return False, "image too large (maximum 4000x4000 pixels)"
        
        return True, "valid image"
        
    except Exception as e:
        return False, f"invalid image: {str(e)}"

def format_detection_results(detections):
    """format detection results for display"""
    if not detections:
        return pd.DataFrame()
    
    formatted_data = []
    for detection in detections:
        formatted_data.append({
            "equipment": detection["equipment"],
            "confidence": f"{detection['confidence']:.3f}",
            "bbox": f"({detection['bbox'][0]:.0f}, {detection['bbox'][1]:.0f}, {detection['bbox'][2]:.0f}, {detection['bbox'][3]:.0f})",
            "area": f"{detection['area']:.0f} px²"
        })
    
    return pd.DataFrame(formatted_data)
