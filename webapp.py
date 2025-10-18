#!/usr/bin/env python3
"""
voidvision web app - space station safety monitor
simple flask-based web application for safety equipment detection
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import cv2
import numpy as np

# add the utils to path
sys.path.append(os.path.dirname(__file__))
from utils import SafetyEquipmentDetector, validate_image, format_detection_results

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# initialize detector
detector = SafetyEquipmentDetector("Hackthon_Dataset/Hackathon2_scripts/runs/detect/train11/weights/best.pt")

@app.route('/')
def index():
    """main page"""
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    """detect safety equipment in uploaded image"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'no image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'no file selected'}), 400
        
        # get confidence threshold
        confidence = float(request.form.get('confidence', 0.5))
        
        # read image
        image = Image.open(file.stream)
        
        # validate image
        is_valid, message = validate_image(image)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # run detection
        results = detector.detect_equipment(image, confidence)
        
        if not results.get('success', False):
            return jsonify({'error': results.get('error', 'detection failed')}), 500
        
        detections = results['detections']
        
        # calculate safety score
        safety_analysis = detector.calculate_safety_score(detections)
        
        # format results
        detection_data = format_detection_results(detections)
        
        # convert annotated image to base64
        annotated_image_b64 = None
        if results['annotated_image'] is not None:
            img_buffer = BytesIO()
            Image.fromarray(results['annotated_image']).save(img_buffer, format='PNG')
            img_buffer.seek(0)
            annotated_image_b64 = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'detections': detection_data.to_dict('records'),
            'safety_analysis': safety_analysis,
            'total_detections': len(detections),
            'annotated_image': annotated_image_b64,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status')
def status():
    """get system status"""
    return jsonify({
        'model_loaded': detector.model is not None,
        'equipment_classes': detector.equipment_classes,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 starting voidvision web app...")
    print("📊 model status:", "✅ loaded" if detector.model is not None else "❌ failed")
    print("🌐 web app will be available at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
