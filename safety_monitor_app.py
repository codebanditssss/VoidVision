import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image
import time
import json

# page configuration
st.set_page_config(
    page_title="voidvision - space station safety monitor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# custom css for space theme
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #00ffff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px #00ffff;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #00ffff;
    }
    .alert-box {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid #ff0000;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .equipment-status {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        margin: 0.25rem 0;
        border-radius: 5px;
        background: rgba(0, 255, 255, 0.1);
    }
    .detected { background: rgba(0, 255, 0, 0.2); }
    .missing { background: rgba(255, 0, 0, 0.2); }
</style>
""", unsafe_allow_html=True)

# initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'equipment_status' not in st.session_state:
    st.session_state.equipment_status = {
        'oxygentank': False,
        'nitrogentank': False,
        'firstaidbox': False,
        'firealarm': False,
        'safetyswitchpanel': False,
        'emergencyphone': False,
        'fireextinguisher': False
    }

# load model
@st.cache_resource
def load_model():
    try:
        model_path = "yolov8s.pt"
        if os.path.exists(model_path):
            return YOLO(model_path)
        else:
            st.error("model file not found. please ensure yolov8s.pt is in the current directory.")
            return None
    except Exception as e:
        st.error(f"error loading model: {str(e)}")
        return None

# main header
st.markdown('<h1 class="main-header">🚀 voidvision space station safety monitor</h1>', unsafe_allow_html=True)

# sidebar
st.sidebar.title("🛠️ control panel")

# model loading
if st.sidebar.button("load model"):
    with st.spinner("loading yolov8 model..."):
        st.session_state.model = load_model()
    if st.session_state.model:
        st.sidebar.success("model loaded successfully!")
    else:
        st.sidebar.error("failed to load model")

# detection settings
st.sidebar.subheader("detection settings")
confidence_threshold = st.sidebar.slider("confidence threshold", 0.1, 1.0, 0.5, 0.05)
iou_threshold = st.sidebar.slider("iou threshold", 0.1, 1.0, 0.7, 0.05)

# equipment monitoring settings
st.sidebar.subheader("equipment monitoring")
required_equipment = st.sidebar.multiselect(
    "required safety equipment",
    ['oxygentank', 'nitrogentank', 'firstaidbox', 'firealarm', 'safetyswitchpanel', 'emergencyphone', 'fireextinguisher'],
    default=['oxygentank', 'firstaidbox', 'fireextinguisher']
)

alert_enabled = st.sidebar.checkbox("enable alerts for missing equipment", value=True)

# main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 real-time detection")
    
    # image upload or camera
    detection_mode = st.radio(
        "detection mode",
        ["upload image", "camera input"],
        horizontal=True
    )
    
    if detection_mode == "upload image":
        uploaded_file = st.file_uploader(
            "upload space station image",
            type=['jpg', 'jpeg', 'png'],
            help="upload an image of space station interior for safety equipment detection"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="uploaded image", use_column_width=True)
            
            if st.button("detect safety equipment") and st.session_state.model:
                with st.spinner("analyzing image..."):
                    # convert pil to cv2
                    img_array = np.array(image)
                    img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # run detection
                    results = st.session_state.model(img_cv2, conf=confidence_threshold, iou=iou_threshold)
                    
                    # process results
                    detected_objects = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                conf = box.conf[0].item()
                                cls = int(box.cls[0].item())
                                class_name = st.session_state.model.names[cls]
                                detected_objects.append({
                                    'class': class_name,
                                    'confidence': conf,
                                    'bbox': box.xyxy[0].tolist()
                                })
                    
                    # update equipment status
                    for obj in detected_objects:
                        if obj['class'].lower() in st.session_state.equipment_status:
                            st.session_state.equipment_status[obj['class'].lower()] = True
                    
                    # display results
                    if detected_objects:
                        st.success(f"detected {len(detected_objects)} safety equipment items")
                        
                        # show detected objects
                        for obj in detected_objects:
                            st.write(f"🔍 **{obj['class']}** (confidence: {obj['confidence']:.2f})")
                        
                        # draw bounding boxes on image
                        annotated_img = results[0].plot()
                        st.image(annotated_img, caption="detection results", use_column_width=True)
                    else:
                        st.warning("no safety equipment detected in the image")
    
    elif detection_mode == "camera input":
        st.info("camera input mode - upload an image for now")
        st.write("in a real deployment, this would connect to space station cameras")

with col2:
    st.subheader("📊 equipment status")
    
    # equipment status display
    for equipment in required_equipment:
        status = st.session_state.equipment_status.get(equipment, False)
        status_text = "✅ detected" if status else "❌ missing"
        status_class = "detected" if status else "missing"
        
        st.markdown(f"""
        <div class="equipment-status {status_class}">
            <span><strong>{equipment.replace('_', ' ').title()}</strong></span>
            <span>{status_text}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # alerts
    if alert_enabled:
        missing_equipment = [eq for eq in required_equipment if not st.session_state.equipment_status.get(eq, False)]
        if missing_equipment:
            st.markdown(f"""
            <div class="alert-box">
                <h4>⚠️ safety alert</h4>
                <p>missing critical equipment: {', '.join(missing_equipment)}</p>
                <p>immediate action required!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("✅ all required safety equipment detected")

# detection history
st.subheader("📈 detection history")
if st.session_state.detection_history:
    for i, detection in enumerate(st.session_state.detection_history[-5:]):  # show last 5
        st.write(f"**{detection['timestamp']}**: {detection['summary']}")
else:
    st.info("no detection history yet")

# footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    <p>voidvision space station safety monitor | powered by yolov8 | duality ai hackathon</p>
    <p>ensuring astronaut safety through ai-powered equipment monitoring</p>
</div>
""", unsafe_allow_html=True)
