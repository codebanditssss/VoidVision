import streamlit as st
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
from utils import SafetyEquipmentDetector, validate_image, format_detection_results

# page configuration
st.set_page_config(
    page_title="voidvision - space station safety monitor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom css for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .alert-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #f44336;
    }
    .success-box {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
</style>
""", unsafe_allow_html=True)

# app title and description
st.markdown('<h1 class="main-header">🚀 voidvision</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <h3>ai-powered safety equipment detection for space station environments</h3>
    <p>detect and monitor critical safety equipment using advanced yolov8 object detection</p>
</div>
""", unsafe_allow_html=True)

# initialize detector
@st.cache_resource
def load_detector():
    model_path = "Hackthon_Dataset/Hackathon2_scripts/runs/detect/train11/weights/best.pt"
    return SafetyEquipmentDetector(model_path)

detector = load_detector()

# sidebar configuration
st.sidebar.title("⚙️ configuration")
st.sidebar.markdown("---")

# confidence threshold
confidence_threshold = st.sidebar.slider(
    "confidence threshold", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.5, 
    step=0.05,
    help="minimum confidence score for detections"
)

# equipment classes
equipment_classes = detector.equipment_classes

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 target equipment")
for i, equipment in enumerate(equipment_classes):
    st.sidebar.write(f"{i+1}. {equipment}")

# model status
st.sidebar.markdown("---")
if detector.model is not None:
    st.sidebar.success("✅ model loaded successfully")
else:
    st.sidebar.error("❌ model failed to load")

# main content area
tab1, tab2, tab3, tab4 = st.tabs(["🔍 detection", "📊 dashboard", "📈 analytics", "📋 reports"])

with tab1:
    st.header("🔍 real-time detection")
    
    # image upload section
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 upload image")
        uploaded_file = st.file_uploader(
            "choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="upload an image to detect safety equipment"
        )
        
        if uploaded_file is not None:
            # validate and display uploaded image
            image = Image.open(uploaded_file)
            is_valid, message = validate_image(image)
            
            if is_valid:
                st.image(image, caption="uploaded image", use_column_width=True)
                
                # detection button
                if st.button("🔍 detect equipment", type="primary"):
                    if detector.model is None:
                        st.error("model not loaded. please check the model path.")
                    else:
                        with st.spinner("analyzing image..."):
                            # run detection
                            results = detector.detect_equipment(image, confidence_threshold)
                            
                            if results.get("success", False):
                                st.success("detection completed!")
                                
                                detections = results["detections"]
                                
                                if detections:
                                    st.subheader("🎯 detection results")
                                    
                                    # display results table
                                    df = format_detection_results(detections)
                                    st.dataframe(df, use_container_width=True)
                                    
                                    # display annotated image
                                    if results["annotated_image"] is not None:
                                        st.image(results["annotated_image"], caption="detection results", use_column_width=True)
                                    
                                    # safety score
                                    safety_analysis = detector.calculate_safety_score(detections)
                                    st.subheader("🛡️ safety analysis")
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("safety score", f"{safety_analysis['score']}%")
                                    with col2:
                                        st.metric("status", safety_analysis['status'])
                                    with col3:
                                        st.metric("total detections", len(detections))
                                    
                                    # missing equipment alerts
                                    if safety_analysis['missing_equipment']:
                                        st.subheader("⚠️ missing equipment")
                                        for missing in safety_analysis['missing_equipment']:
                                            st.warning(f"{missing['equipment']}: {missing['missing']} missing (expected {missing['expected']}, detected {missing['detected']})")
                                
                                else:
                                    st.warning("no safety equipment detected in the image")
                                    
                            else:
                                st.error(f"detection failed: {results.get('error', 'unknown error')}")
            else:
                st.error(f"invalid image: {message}")
    
    with col2:
        st.subheader("📷 camera input")
        st.info("camera functionality will be added in future updates")
        
        # placeholder for camera input
        st.image("https://via.placeholder.com/400x300/cccccc/666666?text=Camera+Input+Placeholder", 
                caption="camera input placeholder")

with tab2:
    st.header("📊 safety equipment dashboard")
    
    # equipment status cards
    st.subheader("🎯 equipment status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("oxygen tanks", "2 detected", "1 missing")
    with col2:
        st.metric("nitrogen tanks", "1 detected", "0 missing")
    with col3:
        st.metric("first aid boxes", "1 detected", "0 missing")
    with col4:
        st.metric("fire alarms", "3 detected", "0 missing")
    
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.metric("safety panels", "1 detected", "1 missing")
    with col6:
        st.metric("emergency phones", "2 detected", "0 missing")
    with col7:
        st.metric("fire extinguishers", "4 detected", "1 missing")
    
    # safety score
    st.subheader("🛡️ safety assessment")
    
    safety_score = 85  # placeholder
    st.progress(safety_score / 100)
    st.markdown(f"**overall safety score: {safety_score}%**")
    
    if safety_score >= 80:
        st.markdown('<div class="success-box">✅ safety status: excellent</div>', unsafe_allow_html=True)
    elif safety_score >= 60:
        st.markdown('<div class="alert-box">⚠️ safety status: good</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-box">❌ safety status: needs attention</div>', unsafe_allow_html=True)

with tab3:
    st.header("📈 performance analytics")
    
    # detection statistics
    st.subheader("📊 detection statistics")
    
    # sample data for demonstration
    detection_stats = pd.DataFrame({
        "equipment": equipment_classes,
        "detected": [2, 1, 1, 3, 1, 2, 4],
        "missing": [1, 0, 0, 0, 1, 0, 1],
        "accuracy": [0.815, 0.796, 0.756, 0.711, 0.679, 0.676, 0.691]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # bar chart for detections
        fig = px.bar(detection_stats, x="equipment", y="detected", 
                    title="equipment detection count")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # accuracy chart
        fig = px.bar(detection_stats, x="equipment", y="accuracy", 
                    title="detection accuracy by equipment")
        st.plotly_chart(fig, use_container_width=True)
    
    # confusion matrix placeholder
    st.subheader("🎯 confusion matrix")
    st.info("confusion matrix visualization will be added in future updates")

with tab4:
    st.header("📋 reports & export")
    
    st.subheader("📄 generate report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**detection summary**")
        st.write("- total detections: 14")
        st.write("- confidence threshold: 0.5")
        st.write("- processing time: 2.3s")
        st.write("- model accuracy: 73.2%")
    
    with col2:
        st.markdown("**export options**")
        
        # export buttons
        if st.button("📊 export csv"):
            st.success("csv export functionality will be added")
        
        if st.button("📄 export pdf"):
            st.success("pdf export functionality will be added")
        
        if st.button("📋 export json"):
            st.success("json export functionality will be added")

# footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>voidvision - space station safety monitor | powered by yolov8 | hackathon project</p>
</div>
""", unsafe_allow_html=True)
