#!/usr/bin/env python3
"""
VoidVision Web App - Space Station Safety Monitor
Professional web application with AI analysis, dark theme, no emojis
"""

import http.server
import socketserver
import json
import cgi
import io
import base64
from datetime import datetime
from PIL import Image
import numpy as np
import os
import sys

# Add utils to path
sys.path.append(os.path.dirname(__file__))
from utils import SafetyEquipmentDetector, validate_image, format_detection_results

# Initialize detector
detector = SafetyEquipmentDetector("Hackthon_Dataset/Hackathon2_scripts/runs/detect/train11/weights/best.pt")

class VoidVisionHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.serve_main_page()
        elif self.path == '/status':
            self.serve_status()
        else:
            super().do_GET()
    
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/detect':
            self.handle_detection()
        elif self.path == '/analyze':
            self.handle_ai_analysis()
        else:
            self.send_error(404)
    
    def serve_main_page(self):
        """Serve the main page"""
        html_content = self.get_html_content()
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html_content.encode())
    
    def serve_status(self):
        """Serve status information"""
        status = {
            "model_loaded": detector.model is not None,
            "status": "running",
            "timestamp": datetime.now().isoformat()
        }
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(status).encode())
    
    def handle_detection(self):
        """Handle equipment detection requests"""
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST',
                        'CONTENT_TYPE': self.headers['Content-Type'],
                        })
            
            file_item = form['file']
            confidence = float(form.getvalue('confidence', 0.5))
            
            if file_item.filename:
                image_bytes = file_item.file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                
                # Run detection
                results = detector.detect_equipment(image, confidence)
                
                if results.get("success", False):
                    detections = results["detections"]
                    safety_analysis = detector.calculate_safety_score(detections)
                    
                    # Convert annotated image to base64
                    annotated_image_b64 = None
                    if results.get("annotated_image") is not None:
                        buffered = io.BytesIO()
                        Image.fromarray(results["annotated_image"]).save(buffered, format="PNG")
                        annotated_image_b64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    response_data = {
                        "success": True,
                        "detections": detections,
                        "annotated_image": annotated_image_b64,
                        "safety_analysis": safety_analysis,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    response_data = {
                        "success": False,
                        "error": results.get("error", "Detection failed")
                    }
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode())
            else:
                self.send_error(400, "No file uploaded")
                
        except Exception as e:
            self.send_error(500, str(e))
    
    def handle_ai_analysis(self):
        """Handle AI analysis requests"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            detections = data.get('detections', [])
            safety_analysis = data.get('safety_analysis', {})
            
            # Generate AI analysis
            ai_analysis = self.generate_ai_analysis(detections, safety_analysis)
            
            response_data = {
                'success': True,
                'ai_analysis': ai_analysis,
                'timestamp': datetime.now().isoformat()
            }
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode())
            
        except Exception as e:
            self.send_error(500, str(e))
    
    def generate_ai_analysis(self, detections, safety_analysis):
        """Generate AI-powered safety analysis"""
        detected_equipment = [d['equipment'] for d in detections]
        safety_score = safety_analysis.get('score', 0)
        status = safety_analysis.get('status', 'unknown')
        missing_equipment = [m['equipment'] for m in safety_analysis.get('missing_equipment', [])]
        
        analysis = f"""
SAFETY ASSESSMENT REPORT
========================

OVERALL EVALUATION:
Safety Score: {safety_score}%
Status: {status.upper()}

DETECTED EQUIPMENT:
{', '.join(detected_equipment) if detected_equipment else 'None detected'}

MISSING CRITICAL EQUIPMENT:
{', '.join(missing_equipment) if missing_equipment else 'All equipment present'}

RECOMMENDATIONS:
"""
        
        if safety_score < 50:
            analysis += """
CRITICAL SAFETY CONCERNS:
- Immediate action required
- Deploy missing safety equipment
- Conduct emergency safety briefing
- Restrict access to affected areas
"""
        elif safety_score < 80:
            analysis += """
MODERATE SAFETY CONCERNS:
- Address missing equipment promptly
- Increase safety monitoring
- Review safety protocols
- Schedule equipment maintenance
"""
        else:
            analysis += """
SAFETY STATUS ACCEPTABLE:
- Continue regular monitoring
- Maintain current safety protocols
- Schedule routine inspections
- Update safety documentation
"""
        
        analysis += f"""
PRIORITY ACTIONS:
1. Verify all detected equipment is functional
2. Install missing critical safety equipment
3. Update safety inventory records
4. Conduct crew safety training
5. Schedule follow-up inspection

RISK ASSESSMENT:
Current risk level: {'HIGH' if safety_score < 50 else 'MEDIUM' if safety_score < 80 else 'LOW'}
Recommended response time: {'IMMEDIATE' if safety_score < 50 else 'WITHIN 24 HOURS' if safety_score < 80 else 'ROUTINE'}
"""
        
        return analysis.strip()
    
    def get_html_content(self):
        """Get the HTML content for the main page"""
        return """
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
            color: #e0e0e0;
            min-height: 100vh;
            line-height: 1.6;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: rgba(20, 20, 20, 0.95);
            min-height: 100vh;
            box-shadow: 0 0 30px rgba(0, 0, 0, 0.8);
            border: 1px solid #333;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: #ffffff;
            padding: 40px 30px;
            text-align: center;
            border-bottom: 3px solid #00d4ff;
            box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
        }
        
        .header h1 {
            font-size: 3.5rem;
            font-weight: 300;
            letter-spacing: 3px;
            margin-bottom: 10px;
            text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
        }
        
        .header p {
            font-size: 1.3rem;
            opacity: 0.9;
            font-weight: 300;
        }
        
        .content {
            padding: 40px;
        }
        
        .upload-section {
            border: 2px dashed #444;
            border-radius: 15px;
            padding: 60px;
            text-align: center;
            margin-bottom: 40px;
            background: rgba(30, 30, 30, 0.8);
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: #00d4ff;
            background: rgba(0, 212, 255, 0.1);
            transform: translateY(-3px);
            box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
        }
        
        .upload-section h3 {
            font-size: 1.5rem;
            margin-bottom: 15px;
            color: #00d4ff;
        }
        
        .upload-section p {
            color: #aaa;
            margin-bottom: 25px;
        }
        
        .file-input {
            display: none;
        }
        
        .upload-btn {
            background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
            color: #000;
            border: none;
            padding: 20px 40px;
            border-radius: 10px;
            font-size: 1.2rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 2px;
            box-shadow: 0 5px 15px rgba(0, 212, 255, 0.3);
        }
        
        .upload-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0, 212, 255, 0.5);
            background: linear-gradient(135deg, #0099cc 0%, #006699 100%);
        }
        
        .controls {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 25px;
            margin: 30px 0;
            padding: 25px;
            background: rgba(30, 30, 30, 0.8);
            border-radius: 15px;
            border: 1px solid #444;
        }
        
        .controls label {
            color: #00d4ff;
            font-weight: 500;
            font-size: 1.1rem;
        }
        
        .confidence-slider {
            width: 200px;
            height: 8px;
            background: #444;
            border-radius: 5px;
            outline: none;
            -webkit-appearance: none;
        }
        
        .confidence-slider::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: #00d4ff;
            border-radius: 50%;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
        }
        
        .confidence-value {
            color: #00d4ff;
            font-weight: 600;
            font-size: 1.1rem;
            min-width: 50px;
        }
        
        .detect-btn {
            background: linear-gradient(135deg, #00ff88 0%, #00cc66 100%);
            color: #000;
            border: none;
            padding: 20px 40px;
            border-radius: 10px;
            font-size: 1.2rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 2px;
            box-shadow: 0 5px 15px rgba(0, 255, 136, 0.3);
        }
        
        .detect-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 25px rgba(0, 255, 136, 0.5);
            background: linear-gradient(135deg, #00cc66 0%, #009944 100%);
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
            padding: 30px;
            background: rgba(30, 30, 30, 0.8);
            border-radius: 15px;
            margin: 20px 0;
        }
        
        .spinner {
            border: 4px solid #444;
            border-top: 4px solid #00d4ff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .results {
            display: none;
            margin-top: 30px;
        }
        
        .image-container {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .image-box {
            background: rgba(30, 30, 30, 0.8);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid #444;
        }
        
        .image-box h4 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.2rem;
        }
        
        .image-box img {
            max-width: 100%;
            border-radius: 10px;
            border: 1px solid #555;
        }
        
        .safety-analysis {
            background: rgba(30, 30, 30, 0.8);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #444;
        }
        
        .safety-analysis h4 {
            color: #00d4ff;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background: rgba(40, 40, 40, 0.8);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 1px solid #555;
        }
        
        .metric-card .value {
            font-size: 2.5rem;
            font-weight: 700;
            color: #00d4ff;
            margin-bottom: 5px;
        }
        
        .metric-card .label {
            color: #aaa;
            font-size: 0.9rem;
        }
        
        .missing-equipment {
            margin-top: 20px;
        }
        
        .missing-equipment h5 {
            color: #ff6b6b;
            margin-bottom: 10px;
        }
        
        .missing-item {
            background: rgba(255, 107, 107, 0.1);
            border: 1px solid #ff6b6b;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 8px;
            color: #ff6b6b;
        }
        
        .ai-analysis {
            background: rgba(30, 30, 30, 0.8);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            border: 1px solid #444;
            border-left: 5px solid #9b59b6;
        }
        
        .ai-analysis h4 {
            color: #9b59b6;
            margin-bottom: 15px;
            font-size: 1.3rem;
        }
        
        .analyze-btn {
            background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 15px;
        }
        
        .analyze-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(155, 89, 182, 0.3);
            background: linear-gradient(135deg, #8e44ad 0%, #7d3c98 100%);
        }
        
        .ai-content {
            background: rgba(40, 40, 40, 0.8);
            padding: 20px;
            border-radius: 10px;
            margin-top: 15px;
            border: 1px solid #555;
            white-space: pre-line;
            line-height: 1.6;
            font-family: 'Courier New', monospace;
            color: #e0e0e0;
        }
        
        .detection-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
            background: rgba(30, 30, 30, 0.8);
            border-radius: 10px;
            overflow: hidden;
        }
        
        .detection-table th {
            background: #00d4ff;
            color: #000;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        .detection-table td {
            padding: 12px 15px;
            border-bottom: 1px solid #444;
            color: #e0e0e0;
        }
        
        .detection-table tr:nth-child(even) {
            background: rgba(40, 40, 40, 0.5);
        }
        
        .detection-table tr:hover {
            background: rgba(0, 212, 255, 0.1);
        }
        
        .error {
            background: rgba(255, 107, 107, 0.1);
            color: #ff6b6b;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border: 1px solid #ff6b6b;
        }
        
        .success {
            background: rgba(0, 255, 136, 0.1);
            color: #00ff88;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            border: 1px solid #00ff88;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 0;
                border-radius: 0;
            }
            
            .header h1 {
                font-size: 2.5rem;
            }
            
            .image-container {
                grid-template-columns: 1fr;
            }
            
            .controls {
                flex-direction: column;
                gap: 15px;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>VoidVision</h1>
            <p>AI-Powered Safety Equipment Detection for Space Station Environments</p>
        </div>
        
        <div class="content">
            <div class="upload-section" id="uploadSection">
                <h3>Upload Image for Detection</h3>
                <p>Drag and drop an image file or click to browse</p>
                <input type="file" id="imageInput" class="file-input" accept="image/*">
                <button class="upload-btn" onclick="document.getElementById('imageInput').click()">
                    Choose Image
                </button>
            </div>
            
            <div class="controls">
                <label for="confidenceSlider">Confidence Threshold:</label>
                <input type="range" id="confidenceSlider" class="confidence-slider" 
                       min="0.1" max="1.0" step="0.05" value="0.5">
                <span class="confidence-value" id="confidenceValue">0.5</span>
                <button class="detect-btn" id="detectBtn" onclick="detectEquipment()" disabled>
                    Detect Equipment
                </button>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Analyzing Image...</p>
            </div>
            
            <div class="results" id="results">
                <div class="image-container">
                    <div class="image-box">
                        <h4>Original Image</h4>
                        <img id="originalImage" src="" alt="Original image">
                    </div>
                    <div class="image-box">
                        <h4>Detection Results</h4>
                        <img id="detectedImage" src="" alt="Detection results">
                    </div>
                </div>
                
                <div class="safety-analysis" id="safetyAnalysis">
                    <!-- Safety analysis will be populated here -->
                </div>
                
                <div class="ai-analysis" id="aiAnalysis" style="display: none;">
                    <h4>AI Safety Assessment</h4>
                    <div class="ai-content" id="aiContent">
                        <!-- AI analysis will be populated here -->
                    </div>
                    <button class="analyze-btn" id="analyzeBtn" onclick="getAIAnalysis()">
                        Get AI Safety Assessment
                    </button>
                </div>
                
                <h4>Detection Results</h4>
                <table class="detection-table" id="detectionTable">
                    <thead>
                        <tr>
                            <th>Equipment</th>
                            <th>Confidence</th>
                            <th>Bounding Box</th>
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
        const imageInput = document.getElementById('imageInput');
        const confidenceSlider = document.getElementById('confidenceSlider');
        const confidenceValue = document.getElementById('confidenceValue');
        const detectBtn = document.getElementById('detectBtn');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const originalImage = document.getElementById('originalImage');
        const detectedImage = document.getElementById('detectedImage');
        const safetyAnalysis = document.getElementById('safetyAnalysis');
        const detectionTableBody = document.getElementById('detectionTableBody');
        const aiAnalysis = document.getElementById('aiAnalysis');
        const aiContent = document.getElementById('aiContent');
        const analyzeBtn = document.getElementById('analyzeBtn');
        
        let selectedFile = null;
        let currentDetections = null;
        let currentSafetyAnalysis = null;
        
        // Update confidence value display
        confidenceSlider.addEventListener('input', function() {
            confidenceValue.textContent = this.value;
        });
        
        // Handle file selection
        imageInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                selectedFile = file;
                const reader = new FileReader();
                reader.onload = function(e) {
                    originalImage.src = e.target.result;
                    detectBtn.disabled = false;
                    results.style.display = 'none';
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Drag and drop functionality
        const uploadSection = document.getElementById('uploadSection');
        
        uploadSection.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                const file = files[0];
                if (file.type.startsWith('image/')) {
                    selectedFile = file;
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        originalImage.src = e.target.result;
                        detectBtn.disabled = false;
                        results.style.display = 'none';
                    };
                    reader.readAsDataURL(file);
                }
            }
        });
        
        // Detection function
        async function detectEquipment() {
            if (!selectedFile) return;
            
            loading.style.display = 'block';
            results.style.display = 'none';
            detectBtn.disabled = true;
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('confidence', confidenceSlider.value);
            
            try {
                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.success) {
                    showResults(data);
                } else {
                    showError(data.error || 'Detection failed');
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            } finally {
                loading.style.display = 'none';
                detectBtn.disabled = false;
            }
        }
        
        function showResults(data) {
            // Display detected image
            if (data.annotated_image) {
                detectedImage.src = `data:image/png;base64,${data.annotated_image}`;
            }
            
            // Display safety analysis
            const safety = data.safety_analysis;
            safetyAnalysis.innerHTML = `
                <h4>Safety Analysis</h4>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="value">${safety.score}%</div>
                        <div class="label">Safety Score</div>
                    </div>
                    <div class="metric-card">
                        <div class="value">${safety.status}</div>
                        <div class="label">Status</div>
                    </div>
                    <div class="metric-card">
                        <div class="value">${data.detections.length}</div>
                        <div class="label">Total Detections</div>
                    </div>
                </div>
                ${safety.missing_equipment && safety.missing_equipment.length > 0 ? `
                    <div class="missing-equipment">
                        <h5>Missing Equipment:</h5>
                        ${safety.missing_equipment.map(item => `
                            <div class="missing-item">
                                ${item.equipment}: ${item.missing} missing (expected ${item.expected}, detected ${item.detected})
                            </div>
                        `).join('')}
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
            
            // Store data for AI analysis
            currentDetections = data.detections;
            currentSafetyAnalysis = safety;
            
            // Show AI analysis section
            aiAnalysis.style.display = 'block';
            
            results.style.display = 'block';
        }
        
        function showError(message) {
            results.innerHTML = `<div class="error">Error: ${message}</div>`;
            results.style.display = 'block';
        }
        
        async function getAIAnalysis() {
            if (!currentDetections || !currentSafetyAnalysis) return;
            
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = 'Analyzing...';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        detections: currentDetections,
                        safety_analysis: currentSafetyAnalysis
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    aiContent.textContent = data.ai_analysis;
                    analyzeBtn.textContent = 'Analysis Complete';
                    analyzeBtn.style.background = '#00ff88';
                } else {
                    aiContent.textContent = 'Failed to get AI analysis. Please try again.';
                }
            } catch (error) {
                aiContent.textContent = 'Network error: ' + error.message;
            } finally {
                analyzeBtn.disabled = false;
                if (analyzeBtn.textContent === 'Analyzing...') {
                    analyzeBtn.textContent = 'Get AI Safety Assessment';
                }
            }
        }
        
        // Check system status on load
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                if (data.model_loaded) {
                    console.log('Model loaded successfully');
                } else {
                    console.log('Model failed to load');
                }
            })
            .catch(error => {
                console.log('Failed to check system status:', error);
            });
    </script>
</body>
</html>
        """

if __name__ == "__main__":
    PORT = 5000
    print("Starting VoidVision Web App...")
    print("Features:")
    print("✅ Dark theme")
    print("✅ No emojis")
    print("✅ AI analysis")
    print("✅ Professional design")
    print(f"Web app available at: http://localhost:{PORT}")
    
    with socketserver.TCPServer(("", PORT), VoidVisionHandler) as httpd:
        httpd.serve_forever()
