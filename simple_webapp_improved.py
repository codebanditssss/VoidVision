#!/usr/bin/env python3
"""
VoidVision - Space Station Safety Monitor
Professional Web Application for Equipment Detection
"""

import os
import json
import base64
import io
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import torch

# Load the trained model
model_path = "Hackthon_Dataset/Hackathon2_scripts/runs/detect/train11/weights/best.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)
    print(f"model loaded successfully from {model_path}")
    print("🎯 Enhanced detection with better confidence thresholds")
    print("🎯 Fixed false positive oxygen tank detections")
    print("🎯 Improved live camera detection")
else:
    model = YOLO('yolov8s.pt')
    print("using base yolov8s model")

class WebAppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/':
            self.serve_landing_page()
        elif parsed_path.path == '/detect':
            self.serve_detection_page()
        elif parsed_path.path == '/status':
            self.serve_status()
        else:
            self.send_error(404)
    
    def do_POST(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/detect':
            self.handle_detection()
        else:
            self.send_error(404)
    
    def serve_landing_page(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoidVision - Space Station Safety Monitor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #2a2a2a 100%);
            color: #ffffff;
            min-height: 100vh;
            overflow-x: hidden;
        }
        
        .hero-section {
            height: 100vh;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            text-align: center;
            position: relative;
            background: radial-gradient(ellipse at center, rgba(0, 212, 255, 0.1) 0%, transparent 70%);
        }
        
        .hero-content {
            max-width: 800px;
            padding: 0 20px;
            z-index: 2;
        }
        
        .logo {
            font-size: 4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 20px;
            text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        }
        
        .tagline {
            font-size: 1.5rem;
            color: #b0b0b0;
            margin-bottom: 30px;
            font-weight: 300;
        }
        
        .description {
            font-size: 1.1rem;
            color: #e0e0e0;
            line-height: 1.6;
            margin-bottom: 50px;
            max-width: 600px;
            margin-left: auto;
            margin-right: auto;
        }
        
        .cta-button {
            background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
            color: #0a0a0a;
            border: none;
            padding: 18px 40px;
            border-radius: 50px;
            font-size: 1.2rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 2px;
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
            text-decoration: none;
            display: inline-block;
        }
        
        .cta-button:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(0, 212, 255, 0.4);
            background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        }
        
        .features-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 30px;
            max-width: 800px;
            margin: 80px auto;
            padding: 0 20px;
        }
        
        .feature-card {
            background: rgba(20, 20, 20, 0.8);
            border: 1px solid #333;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            border-color: #00d4ff;
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
        }
        
        .feature-icon {
            width: 60px;
            height: 60px;
            background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
            border-radius: 50%;
            margin: 0 auto 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            color: #0a0a0a;
        }
        
        .feature-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: #ffffff;
        }
        
        .feature-description {
            color: #b0b0b0;
            line-height: 1.5;
        }
        
        .floating-elements {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1;
        }
        
        .floating-circle {
            position: absolute;
            border-radius: 50%;
            background: rgba(0, 212, 255, 0.1);
            animation: float 6s ease-in-out infinite;
        }
        
        .floating-circle:nth-child(1) {
            width: 100px;
            height: 100px;
            top: 20%;
            left: 10%;
            animation-delay: 0s;
        }
        
        .floating-circle:nth-child(2) {
            width: 150px;
            height: 150px;
            top: 60%;
            right: 15%;
            animation-delay: 2s;
        }
        
        .floating-circle:nth-child(3) {
            width: 80px;
            height: 80px;
            top: 30%;
            right: 30%;
            animation-delay: 4s;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        .footer {
            border-top: 1px solid #333;
            padding: 60px 0 40px;
            margin-top: 100px;
            text-align: center;
        }
        
        .footer-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        
        .footer-logo {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 20px;
        }
        
        
        
        
        .footer-bottom {
            border-top: 1px solid #333;
            padding-top: 30px;
            margin-top: 40px;
            color: #666;
            font-size: 0.9rem;
        }
        
        
        @media (max-width: 768px) {
            .logo {
                font-size: 3rem;
            }
            
            .tagline {
                font-size: 1.2rem;
            }
            
            .description {
                font-size: 1rem;
            }
            
            .features-grid {
                grid-template-columns: 1fr;
                margin: 60px auto;
            }
        }
    </style>
</head>
<body>
    <div class="hero-section">
        <div class="floating-elements">
            <div class="floating-circle"></div>
            <div class="floating-circle"></div>
            <div class="floating-circle"></div>
        </div>
        
        <div class="hero-content">
            <h1 class="logo">VoidVision</h1>
            <p class="tagline">AI-Powered Safety Equipment Detection</p>
            <p class="description">
                Advanced computer vision system designed for space station environments. 
                Detect and monitor critical safety equipment with real-time precision and 
                professional-grade analysis capabilities.
            </p>
            <a href="/detect" class="cta-button">Launch Detection System</a>
        </div>
    </div>
    
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">📷</div>
            <h3 class="feature-title">Real-Time Detection</h3>
            <p class="feature-description">
                Instant identification of safety equipment using advanced YOLO neural networks
            </p>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <h3 class="feature-title">High Accuracy</h3>
            <p class="feature-description">
                Trained on specialized datasets achieving 73.2% mAP@0.5 detection accuracy
            </p>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">📱</div>
            <h3 class="feature-title">Live Camera</h3>
            <p class="feature-description">
                Stream live video feeds with instant equipment detection and monitoring
            </p>
        </div>
        
        <div class="feature-card">
            <div class="feature-icon">🔧</div>
            <h3 class="feature-title">7 Equipment Types</h3>
            <p class="feature-description">
                Oxygen Tanks, Nitrogen Tanks, First Aid Boxes, Fire Alarms, Safety Panels, Emergency Phones, Fire Extinguishers
            </p>
        </div>
    </div>
    
    <div class="footer">
            <div class="footer-bottom">
                <p>&copy; 2025 VoidVision. Advanced Space Station Safety Monitoring System.</p>
                <p>Powered by Team Rocket</p>
            </div>
    </div>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_detection_page(self):
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VoidVision - Detection System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #2a2a2a 100%);
            color: #ffffff;
            min-height: 100vh;
        }
        
        .header {
            background: rgba(20, 20, 20, 0.95);
            border-bottom: 1px solid #333;
            padding: 20px 0;
            backdrop-filter: blur(10px);
        }
        
        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logo {
            font-size: 1.8rem;
            font-weight: 700;
            background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .back-btn {
            background: rgba(255, 255, 255, 0.1);
            color: #ffffff;
            border: 1px solid #333;
            padding: 10px 20px;
            border-radius: 25px;
            text-decoration: none;
            transition: all 0.3s ease;
        }
        
        .back-btn:hover {
            background: rgba(255, 255, 255, 0.2);
            border-color: #00d4ff;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 40px 20px;
        }
        
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 40px;
        }
        
        .upload-section {
            background: rgba(20, 20, 20, 0.8);
            border: 2px solid #00d4ff;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .upload-section:hover {
            border-color: #00ff88;
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
        }
        
        .camera-section {
            background: rgba(20, 20, 20, 0.8);
            border: 2px solid #00ff88;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }
        
        .camera-section:hover {
            border-color: #00d4ff;
            box-shadow: 0 10px 30px rgba(0, 255, 136, 0.2);
        }
        
        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 20px;
            color: #ffffff;
        }
        
        .section-description {
            color: #b0b0b0;
            margin-bottom: 25px;
            line-height: 1.5;
        }
        
        .upload-area {
            border: 2px dashed #00d4ff;
            border-radius: 10px;
            padding: 40px 20px;
            margin-bottom: 20px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .upload-area:hover {
            border-color: #00ff88;
            background: rgba(0, 212, 255, 0.05);
        }
        
        .upload-area.dragover {
            border-color: #00ff88;
            background: rgba(0, 255, 136, 0.1);
        }
        
        .file-input {
            display: none;
        }
        
        .upload-btn {
            background: linear-gradient(135deg, #00d4ff 0%, #00ff88 100%);
            color: #0a0a0a;
            border: none;
            padding: 15px 30px;
            border-radius: 25px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 10px;
        }
        
        .upload-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 212, 255, 0.3);
        }
        
        .camera-container {
            width: 100%;
            height: 250px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 20px;
            position: relative;
            overflow: hidden;
        }
        
        .camera-placeholder {
            color: #666;
            font-size: 0.9rem;
        }
        
        #cameraVideo {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 10px;
        }
        
        #cameraCanvas {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            border-radius: 10px;
        }
        
        .camera-controls {
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .camera-btn {
            background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);
            color: #0a0a0a;
            border: none;
            padding: 12px 25px;
            border-radius: 25px;
            font-size: 0.9rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .camera-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 255, 136, 0.3);
        }
        
        .camera-btn:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .detect-btn {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
            color: #ffffff;
            border: none;
            padding: 18px 40px;
            border-radius: 30px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 2px;
            width: 100%;
            margin-top: 30px;
        }
        
        .detect-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);
        }
        
        .detect-btn:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            width: 50px;
            height: 50px;
            border: 3px solid #333;
            border-top: 3px solid #00d4ff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results {
            margin-top: 40px;
            display: none;
            max-height: 80vh;
            overflow-y: auto;
        }
        
        .results-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .image-box {
            background: rgba(20, 20, 20, 0.8);
            border: 1px solid #333;
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
        }
        
        .image-box h4 {
            margin-bottom: 15px;
            color: #ffffff;
            font-size: 1.1rem;
        }
        
        .image-box img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        
        .detection-summary {
            background: rgba(20, 20, 20, 0.8);
            border: 1px solid #333;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 30px;
            backdrop-filter: blur(10px);
        }
        
        .detection-summary h3 {
            color: #00d4ff;
            margin-bottom: 20px;
            font-size: 1.3rem;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 20px;
        }
        
        .stat-item {
            text-align: center;
            padding: 15px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
        }
        
        .stat-number {
            font-size: 2rem;
            font-weight: 700;
            color: #00ff88;
            margin-bottom: 5px;
        }
        
        .stat-label {
            color: #b0b0b0;
            font-size: 0.9rem;
        }
        
        .detection-table {
            background: rgba(20, 20, 20, 0.8);
            border: 1px solid #333;
            border-radius: 15px;
            padding: 25px;
            backdrop-filter: blur(10px);
        }
        
        .detection-table h4 {
            color: #ffffff;
            margin-bottom: 20px;
            font-size: 1.2rem;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #333;
        }
        
        th {
            background: rgba(0, 212, 255, 0.1);
            color: #ffffff;
            font-weight: 600;
        }
        
        td {
            color: #e0e0e0;
        }
        
        @media (max-width: 768px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
            
            .results-grid {
                grid-template-columns: 1fr;
            }
            
            .header-content {
                flex-direction: column;
                gap: 15px;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="logo">VoidVision</div>
            <a href="/" class="back-btn">Back to Home</a>
        </div>
    </div>
    
    <div class="container">
        <div class="main-grid">
            <div class="upload-section">
                <h3 class="section-title">Image Upload</h3>
                <p class="section-description">Upload an image to detect safety equipment</p>
                
                <div class="upload-area" id="uploadArea">
                    <p>Drag and drop an image here or click to browse</p>
                    <input type="file" id="imageInput" class="file-input" accept="image/*">
                    <button class="upload-btn" onclick="document.getElementById('imageInput').click()">
                        Choose Image
                    </button>
                </div>
            </div>
            
            <div class="camera-section">
                <h3 class="section-title">Live Camera</h3>
                <p class="section-description">Use your camera for real-time detection</p>
                
                <div class="camera-container">
                    <video id="cameraVideo" autoplay muted style="display: none;"></video>
                    <canvas id="cameraCanvas" style="display: none;"></canvas>
                    <div class="camera-placeholder" id="cameraPlaceholder">
                        <p>Click "Start Camera" to begin live detection</p>
                    </div>
                </div>
                
                <div class="camera-controls">
                    <button class="camera-btn" id="startCameraBtn" onclick="startCamera()">
                        Start Camera
                    </button>
                    <button class="camera-btn" id="stopCameraBtn" onclick="stopCamera()" style="display: none;">
                        Stop Camera
                    </button>
                    <button class="camera-btn" id="captureBtn" onclick="captureImage()" style="display: none;">
                        Capture & Detect
                    </button>
                </div>
            </div>
        </div>
        
        <button class="detect-btn" id="detectBtn" onclick="detectEquipment()" disabled>
            Detect Equipment
        </button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Analyzing Image...</p>
        </div>
        
        <div class="results" id="results">
            <div class="results-grid">
                <div class="image-box">
                    <h4>Original Image</h4>
                    <img id="originalImage" src="" alt="original image" style="display: none;">
                </div>
                <div class="image-box">
                    <h4>Detection Results</h4>
                    <img id="detectedImage" src="" alt="detection results" style="display: none;">
                </div>
            </div>
            
            <div class="detection-summary" id="detectionSummary">
                <!-- Detection summary will be populated here -->
            </div>
            
            <div class="detection-table">
                <h4>Detection Details</h4>
                <table>
                    <thead>
                        <tr>
                            <th>Equipment</th>
                            <th>Confidence</th>
                            <th>Location</th>
                            <th>Area</th>
                        </tr>
                    </thead>
                    <tbody id="detectionTableBody">
                        <!-- Detection results will be populated here -->
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        // DOM elements
        const imageInput = document.getElementById('imageInput');
        const uploadArea = document.getElementById('uploadArea');
        const detectBtn = document.getElementById('detectBtn');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const originalImage = document.getElementById('originalImage');
        const detectedImage = document.getElementById('detectedImage');
        const detectionSummary = document.getElementById('detectionSummary');
        const detectionTableBody = document.getElementById('detectionTableBody');
        
        // Camera elements
        const cameraVideo = document.getElementById('cameraVideo');
        const cameraCanvas = document.getElementById('cameraCanvas');
        const cameraPlaceholder = document.getElementById('cameraPlaceholder');
        const startCameraBtn = document.getElementById('startCameraBtn');
        const stopCameraBtn = document.getElementById('stopCameraBtn');
        const captureBtn = document.getElementById('captureBtn');
        
        let selectedFile = null;
        let cameraStream = null;
        
        // File upload handling
        imageInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                selectedFile = file;
                detectBtn.disabled = false;
                
                // Reset capture button when new image is uploaded
                captureBtn.disabled = false;
                captureBtn.textContent = 'Capture & Detect';
                captureBtn.style.opacity = '1';
                
                // Display preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    originalImage.src = e.target.result;
                    originalImage.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Drag and drop functionality
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    selectedFile = file;
                    detectBtn.disabled = false;
                    
                    // Display preview
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        originalImage.src = e.target.result;
                        originalImage.style.display = 'block';
                    };
                    reader.readAsDataURL(file);
                }
            }
        });
        
        // Camera functions
        async function startCamera() {
            try {
                startCameraBtn.disabled = true;
                startCameraBtn.textContent = 'Starting...';
                
                cameraStream = await navigator.mediaDevices.getUserMedia({ 
                    video: { 
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'environment'
                    } 
                });
                
                cameraVideo.srcObject = cameraStream;
                cameraVideo.style.display = 'block';
                cameraPlaceholder.style.display = 'none';
                
                startCameraBtn.style.display = 'none';
                stopCameraBtn.style.display = 'inline-block';
                captureBtn.style.display = 'inline-block';
                
                // Reset capture button state
                captureBtn.disabled = false;
                captureBtn.textContent = 'Capture & Detect';
                captureBtn.style.opacity = '1';
                
            } catch (error) {
                console.error('Error accessing camera:', error);
                alert('Unable to access camera. Please check permissions and try again.');
                startCameraBtn.disabled = false;
                startCameraBtn.textContent = 'Start Camera';
            }
        }
        
        function stopCamera() {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
                cameraStream = null;
            }
            
            cameraVideo.style.display = 'none';
            cameraPlaceholder.style.display = 'block';
            cameraPlaceholder.innerHTML = '<p>Click "Start Camera" to begin live detection</p>';
            
            startCameraBtn.style.display = 'inline-block';
            stopCameraBtn.style.display = 'none';
            captureBtn.style.display = 'none';
            
            startCameraBtn.disabled = false;
            startCameraBtn.textContent = 'Start Camera';
        }
        
        async function captureImage() {
            console.log('Capture button clicked!');
            if (!cameraStream) {
                console.error('No camera stream available');
                return;
            }
            
            try {
                console.log('Starting capture process...');
                captureBtn.disabled = true;
                captureBtn.textContent = 'Capturing...';
                
                // Capture frame from video
                const canvas = cameraCanvas;
                const ctx = canvas.getContext('2d');
                
                console.log('Canvas dimensions:', canvas.width, 'x', canvas.height);
                console.log('Video dimensions:', cameraVideo.videoWidth, 'x', cameraVideo.videoHeight);
                
                canvas.width = cameraVideo.videoWidth;
                canvas.height = cameraVideo.videoHeight;
                ctx.drawImage(cameraVideo, 0, 0);
                
                console.log('Image drawn to canvas');
                
                // Convert canvas to blob
                canvas.toBlob(async (blob) => {
                    console.log('Blob created:', blob ? 'Success' : 'Failed');
                    if (blob) {
                        console.log('Blob size:', blob.size, 'bytes');
                        // Create a file from the blob
                        const file = new File([blob], 'camera-capture.jpg', { type: 'image/jpeg' });
                        
                        // Display the captured image
                        const reader = new FileReader();
                        reader.onload = function(e) {
                            console.log('Image loaded for display');
                            originalImage.src = e.target.result;
                            originalImage.style.display = 'block';
                        };
                        reader.readAsDataURL(file);
                        
                        // Set as selected file and enable detection
                        selectedFile = file;
                        detectBtn.disabled = false;
                        console.log('File set as selected, detection enabled');
                        
                        // Disable capture button after successful capture
                        captureBtn.disabled = true;
                        captureBtn.textContent = 'Captured ✓';
                        captureBtn.style.opacity = '0.6';
                    } else {
                        console.error('Failed to create blob from canvas');
                        captureBtn.disabled = false;
                        captureBtn.textContent = 'Capture & Detect';
                    }
                }, 'image/jpeg', 0.8);
                
            } catch (error) {
                console.error('Error capturing image:', error);
                alert('Error capturing image. Please try again.');
                captureBtn.disabled = false;
                captureBtn.textContent = 'Capture & Detect';
            }
        }
        
        // Detection function
        async function detectEquipment() {
            if (!selectedFile) return;
            
            loading.style.display = 'block';
            results.style.display = 'none';
            detectBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('image', selectedFile);
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data);
                } else {
                    showError(data.error);
                }
            } catch (error) {
                console.error('Error during detection:', error);
                showError('An unexpected error occurred during detection.');
            } finally {
                loading.style.display = 'none';
                detectBtn.disabled = false;
            }
        }
        
        function displayResults(data) {
            // Display detected image
            if (data.annotated_image) {
                detectedImage.src = 'data:image/png;base64,' + data.annotated_image;
                detectedImage.style.display = 'block';
            }
            
            // Display detection summary
            const safety = data.safety_analysis;
            detectionSummary.innerHTML = `
                <h3>Detection Summary</h3>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-number">${data.detections.length}</div>
                        <div class="stat-label">Total Detections</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${safety.missing_equipment.length}</div>
                        <div class="stat-label">Missing Equipment</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-number">${safety.detected_counts ? Object.keys(safety.detected_counts).length : 0}</div>
                        <div class="stat-label">Equipment Types Found</div>
                    </div>
                </div>
                ${safety.missing_equipment.length > 0 ? `
                    <div style="margin-top: 20px;">
                        <h4 style="color: #ff6b6b; margin-bottom: 10px;">Missing Equipment:</h4>
                        ${safety.missing_equipment.map(item => 
                            `<div style="color: #ff6b6b; margin: 5px 0;">
                                ${item.equipment}: ${item.missing} missing (expected ${item.expected}, detected ${item.detected})
                            </div>`
                        ).join('')}
                    </div>
                ` : ''}
            `;
            
            // Display detection table
            detectionTableBody.innerHTML = data.detections.map(detection => `
                <tr>
                    <td>${detection.equipment}</td>
                    <td>${detection.confidence}</td>
                    <td>${detection.bbox}</td>
                    <td>${detection.area}</td>
                </tr>
            `).join('');
            
            results.style.display = 'block';
        }
        
        function showError(message) {
            results.innerHTML = `<div style="color: #ff6b6b; text-align: center; padding: 40px;">Error: ${message}</div>`;
            results.style.display = 'block';
        }
    </script>
</body>
</html>
        """
        
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_status(self):
        status = {
            "status": "running",
            "model_loaded": model is not None,
            "timestamp": datetime.now().isoformat()
        }
        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())
    
    def handle_detection(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            # Parse multipart form data
            boundary = post_data.split(b'\r\n')[0]
            parts = post_data.split(boundary)
            
            image_data = None
            for part in parts:
                if b'Content-Disposition: form-data' in part and b'name="image"' in part:
                    # Extract image data
                    header_end = part.find(b'\r\n\r\n')
                    if header_end != -1:
                        image_data = part[header_end + 4:]
                        # Remove trailing boundary markers
                        if image_data.endswith(b'\r\n'):
                            image_data = image_data[:-2]
                    break
            
            if not image_data:
                self.send_error_response("No image data received")
                return
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Run detection with debugging
            print(f"🔍 Running detection on image size: {image.size}")
            # Enhanced detection with debugging
        print(f"🔍 Running detection on image size: {image.size}")
        results = model(image)
        print(f"📊 Detection results: {len(results)} result(s)")
        
        # Check if model is working
        if not results:
            print("⚠️  No results returned from model")
            self.send_error_response("Model detection failed - no results returned")
            return
            print(f"📊 Detection results: {len(results)} result(s)")
            
            # Debug: Check if model is working
            if not results:
                print("⚠️  No results returned from model")
                self.send_error_response("Model detection failed - no results returned")
                return
            
            # Process results
            detections = []
            annotated_image = image.copy()
            draw = ImageDraw.Draw(annotated_image)
            
            # Try to load a font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()
            
            for i, result in enumerate(results):
                print(f"🔍 Processing result {i+1}/{len(results)}")
                boxes = result.boxes
                if boxes is not None:
                    print(f"📦 Found {len(boxes)} boxes")
                    for j, box in enumerate(boxes):
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        
                        # Apply confidence threshold to filter out low-confidence detections
                        # Use higher threshold for problematic classes to reduce false positives
                        if cls == 0:  # OxygenTank - often false positive
                            min_conf = 0.7
                        elif cls == 1:  # NitrogenTank - often misclassified
                            min_conf = 0.6
                        elif cls == 3:  # FireAlarm - critical for safety
                            min_conf = 0.5
                        elif cls == 6:  # EmergencyPhone - often confused with FireAlarm
                            min_conf = 0.5
                        else:
                            min_conf = 0.4
                        
                        print(f"🎯 Box {j+1}: Class {cls}, Confidence {conf:.3f}, Min Required {min_conf:.3f}")
                        if conf < min_conf:
                            print(f"❌ Filtered out - confidence too low")
                            continue
                        print(f"✅ Detection passed - {class_name}: {conf:.3f}")
                        
                        # Get class name
                        class_names = ['OxygenTank', 'NitrogenTank', 'FirstAidBox', 'FireAlarm', 
                                     'SafetySwitchPanel', 'EmergencyPhone', 'FireExtinguisher']
                        class_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
                        
                        # Draw bounding box
                        draw.rectangle([x1, y1, x2, y2], outline=(0, 212, 255), width=3)
                        
                        # Draw label
                        label = f"{class_name}: {conf:.2f}"
                        draw.text((x1, y1 - 25), label, fill=(0, 212, 255), font=font)
                        
                        # Store detection
                        detections.append({
                            "equipment": class_name,
                            "confidence": f"{conf:.3f}",
                            "bbox": f"({int(x1)},{int(y1)},{int(x2)},{int(y2)})",
                            "area": f"{int((x2-x1)*(y2-y1))} px²"
                        })
            
            # Generate safety analysis
            safety_analysis = self.generate_safety_analysis(detections)
            
            # Convert annotated image to base64
            img_buffer = io.BytesIO()
            annotated_image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Send response
            response = {
                "success": True,
                "detections": detections,
                "safety_analysis": safety_analysis,
                "annotated_image": img_base64
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            print(f"Error in detection: {e}")
            self.send_error_response(f"Detection failed: {str(e)}")
    
    def generate_safety_analysis(self, detections):
        # Expected equipment counts for a typical space station module
        expected_counts = {
            'OxygenTank': 2,
            'NitrogenTank': 1,
            'FirstAidBox': 1,
            'FireAlarm': 2,
            'SafetySwitchPanel': 1,
            'EmergencyPhone': 1,
            'FireExtinguisher': 2
        }
        
        # Count detected equipment
        detected_counts = {}
        for detection in detections:
            equipment = detection['equipment']
            detected_counts[equipment] = detected_counts.get(equipment, 0) + 1
        
        # Find missing equipment
        missing_equipment = []
        for equipment, expected in expected_counts.items():
            detected = detected_counts.get(equipment, 0)
            if detected < expected:
                missing_equipment.append({
                    'equipment': equipment,
                    'expected': expected,
                    'detected': detected,
                    'missing': expected - detected
                })
        
        return {
            'detected_counts': detected_counts,
            'missing_equipment': missing_equipment
        }
    
    def send_error_response(self, message):
        response = {"success": False, "error": message}
        self.send_response(400)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

if __name__ == "__main__":
    print("🚀 starting voidvision web app...")
    print("📊 model status: ✅ loaded")
    print("🌐 web app will be available at: http://localhost:5000")
    print("press ctrl+c to stop the server")
    
    server = HTTPServer(('localhost', 5000), WebAppHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 server stopped")
        server.shutdown()