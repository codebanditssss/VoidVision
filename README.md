# voidvision - space station safety monitor

> ai-powered safety equipment detection for space station environments

[![python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![yolov8](https://img.shields.io/badge/yolo-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![streamlit](https://img.shields.io/badge/streamlit-app-red.svg)](https://streamlit.io/)
[![hackathon](https://img.shields.io/badge/hackathon-duality%20ai-orange.svg)](https://duality.ai/)

## overview

voidvision is an advanced object detection system designed to identify critical safety equipment in space station environments. using yolov8 architecture and synthetic data from duality ai's falcon platform, we've created a comprehensive safety monitoring solution that ensures astronaut safety through automated equipment detection.

### hackathon challenge
**duality ai space station challenge #2** - develop an ai system to detect safety equipment in space station environments using synthetic data from digital twin platforms.

## key features

- **real-time detection** of 7 critical safety equipment types
- **interactive web interface** with space-themed design
- **automated safety scoring** and equipment status monitoring
- **missing equipment alerts** with immediate notifications
- **comprehensive reporting** with json export capabilities
- **confidence threshold adjustment** for fine-tuned detection
- **detection history tracking** for safety audits

## safety equipment detected

| equipment | purpose | criticality |
|-----------|---------|-------------|
| oxygen tank | life support | critical |
| nitrogen tank | life support | critical |
| first aid box | emergency medical | high |
| fire alarm | fire safety | high |
| safety switch panel | control interface | medium |
| emergency phone | communication | medium |
| fire extinguisher | fire suppression | high |

## performance metrics

### final results (train11 - 5 epochs)
- **map@0.5**: 73.18% (significant improvement from baseline)
- **precision**: 87.96% (excellent accuracy)
- **recall**: 64.45% (good detection coverage)
- **map@0.5-95**: 58.26% (strong overall performance)
- **training time**: ~6 hours (cpu-optimized)

### dataset statistics
- **total images**: 3,511 synthetic images
- **training set**: 1,767 images
- **validation set**: 336 images
- **test set**: 1,408 images
- **classes**: 7 safety equipment types

## quick start

### prerequisites
- python 3.10+
- anaconda/miniconda
- 8gb+ ram recommended
- windows 10/11 (tested on windows 10)

### installation

1. **clone the repository**
```bash
git clone https://github.com/codebanditssss/VoidVision.git
cd VoidVision
```

2. **setup environment**
```bash
# create conda environment
conda create --name EDU python=3.10 -y
conda activate EDU

# install dependencies
pip install -r requirements.txt
```

3. **download model weights**
```bash
# yolov8s model will be downloaded automatically
# or manually download: wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
```

4. **run the application**
```bash
# start the safety monitor app
streamlit run safety_monitor_app.py

# or run the demo script
python demo_safety_monitor.py
```

## step-by-step instructions

### 1. environment setup

#### option a: using provided scripts
```bash
cd Hackthon_Dataset/Hackathon2_scripts/ENV_SETUP
# run the batch file to create environment
create_env.bat
# activate the environment
activate.bat
# install packages
install_packages.bat
```

#### option b: manual setup
```bash
# create conda environment
conda create --name EDU python=3.10 -y
conda activate EDU

# install core dependencies
pip install ultralytics>=8.3.0
pip install torch>=2.5.0
pip install torchvision>=0.20.0
pip install streamlit>=1.28.0
pip install opencv-python>=4.8.0
pip install pillow>=10.0.0
pip install numpy>=1.24.0
```

### 2. training the model

#### basic training
```bash
cd Hackthon_Dataset/Hackathon2_scripts
python train.py
```

#### advanced training with custom parameters
```bash
python train.py --epochs 50 --lr0 0.001 --mosaic 0.8 --optimizer AdamW
```

#### training parameters
- `--epochs`: number of training epochs (default: 50)
- `--lr0`: initial learning rate (default: 0.001)
- `--lrf`: final learning rate (default: 0.01)
- `--mosaic`: mosaic augmentation probability (default: 0.8)
- `--optimizer`: optimizer type (default: AdamW)
- `--momentum`: sgd momentum (default: 0.937)

### 3. running predictions

#### test set evaluation
```bash
cd Hackthon_Dataset/Hackathon2_scripts
python predict.py
```

this will:
- load the best trained model
- process all test images
- save predictions with bounding boxes
- generate evaluation metrics
- create output directories: `predictions/images/` and `predictions/labels/`

#### single image prediction
```python
from ultralytics import YOLO
import cv2

# load model
model = YOLO('runs/detect/train11/weights/best.pt')

# predict on single image
results = model.predict('path/to/image.jpg', conf=0.5)

# display results
for result in results:
    img = result.plot()
    cv2.imshow('Prediction', img)
    cv2.waitKey(0)
```

### 4. using the web application

#### launch streamlit app
```bash
streamlit run safety_monitor_app.py
```

#### features available:
- **real-time detection**: upload images or use camera input
- **equipment status**: live monitoring of all safety equipment
- **alert system**: immediate notifications for missing equipment
- **confidence adjustment**: fine-tune detection sensitivity (0.1-0.9)
- **history tracking**: view past detection results
- **json export**: download detection results for analysis

### 5. demo script usage

```bash
python demo_safety_monitor.py
```

this script provides:
- automated batch processing of test images
- comprehensive safety reports
- json export of detection results
- performance metrics calculation

## project structure

```
voidvision/
├── readme.md                    # project documentation
├── requirements.txt             # python dependencies
├── performance_report.md        # comprehensive analysis report
├── progress_tracking.md         # development progress
├── safety_monitor_app.py        # main streamlit application
├── demo_safety_monitor.py       # demo script
├── train_optimized.py           # optimized training script
├── predict.py                   # prediction script
├── visualize.py                 # visualization utilities
├── yolo_params.yaml             # model configuration
├── classes.txt                  # class definitions
└── hackthon_dataset/            # training data and scripts
    └── hackathon2_scripts/
        ├── yolov8s.pt           # pre-trained model weights
        ├── runs/                # training results
        │   └── detect/
        │       ├── train11/     # best training run (5 epochs, 73.18% mAP@0.5)
        │       │   ├── weights/
        │       │   │   ├── best.pt # best model weights
        │       │   │   └── last.pt # last epoch weights
        │       │   ├── results.csv # training metrics
        │       │   └── *.png       # training plots
        │       └── train6/      # previous training run (68.98% mAP@0.5)
        ├── env_setup/           # environment setup scripts
        └── predictions/         # prediction outputs
```

## technical approach

### model architecture
- **base model**: yolov8s (small variant for efficiency)
- **input size**: 640x640 pixels
- **classes**: 7 safety equipment categories
- **optimizer**: adamw with learning rate 0.001
- **augmentation**: mosaic (0.8), flip (0.5), color jitter

### training process
- **dataset**: synthetic data from falcon digital twin platform
- **device**: cpu-optimized training
- **epochs**: 5 (final model) with validation every epoch
- **batch size**: 16
- **validation**: real-time validation during training

### optimization strategies
- **hyperparameter tuning**: learning rate, batch size, epochs
- **data augmentation**: enhanced mosaic, flip, rotation
- **model variants**: yolov8s testing with different configurations
- **loss functions**: box loss (7.5), class loss (0.5), dfl loss (1.5)

## how to reproduce results

### reproducing final results (train11)

1. **setup environment**
```bash
conda create --name EDU python=3.10 -y
conda activate EDU
pip install -r requirements.txt
```

2. **navigate to scripts directory**
```bash
cd Hackthon_Dataset/Hackathon2_scripts
```

3. **run training with exact parameters**
```bash
python train.py --epochs 5 --lr0 0.001 --lrf 0.01 --mosaic 0.8 --optimizer AdamW --momentum 0.937
```

4. **expected training output**
- training time: ~6 hours on cpu
- final map@0.5: ~73%
- final precision: ~88%
- final recall: ~64%

5. **verify results**
```bash
# check training results
cat runs/detect/train11/results.csv

# run evaluation
python predict.py
```

### reproducing baseline results

1. **run single epoch training**
```bash
python train.py --epochs 1 --lr0 0.0001 --mosaic 0.4
```

2. **expected baseline performance**
- map@0.5: ~36%
- precision: ~47%
- recall: ~34%

## expected outputs and interpretation

### training outputs

#### results csv (`results.csv`)
```csv
epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
1,2474.77,1.02452,1.95507,1.15731,0.63697,0.39349,0.4367,0.32088,0.97209,1.44367,1.08529,0.0672973,0.00033033,0.00033033
```

**interpretation:**
- `metrics/precision(B)`: accuracy of detections (higher = fewer false positives)
- `metrics/recall(B)`: coverage of actual objects (higher = fewer missed objects)
- `metrics/mAP50(B)`: mean average precision at iou 0.5 (primary metric)
- `metrics/mAP50-95(B)`: mean average precision across iou 0.5-0.95 (more strict)

#### training plots
- `results.png`: combined training metrics
- `BoxP_curve.png`: precision curve
- `BoxR_curve.png`: recall curve
- `BoxPR_curve.png`: precision-recall curve
- `confusion_matrix.png`: class-wise performance

### prediction outputs

#### image predictions (`predictions/images/`)
- original images with bounding boxes drawn
- color-coded by class
- confidence scores displayed

#### label files (`predictions/labels/`)
```txt
0 0.5 0.3 0.2 0.4
1 0.7 0.6 0.15 0.25
```
format: `class_id x_center y_center width height` (normalized coordinates)

### web application outputs

#### detection results
- real-time bounding box visualization
- equipment status dashboard
- confidence scores for each detection
- missing equipment alerts

#### json export format
```json
{
  "timestamp": "2024-01-01T12:00:00",
  "image_path": "test_image.jpg",
  "detections": [
    {
      "class": "OxygenTank",
      "confidence": 0.85,
      "bbox": [100, 150, 200, 300],
      "status": "detected"
    }
  ],
  "safety_score": 85,
  "missing_equipment": ["FireExtinguisher"]
}
```

## troubleshooting

### common issues

#### 1. cuda/gpu issues
```bash
# force cpu usage
export CUDA_VISIBLE_DEVICES=""
# or modify training script to use device='cpu'
```

#### 2. memory issues
```bash
# reduce batch size
python train.py --batch 8
# or reduce image size
python train.py --imgsz 416
```

#### 3. package compatibility
```bash
# reinstall with specific versions
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics==8.3.0
```

#### 4. file path issues
```bash
# ensure you're in the correct directory
cd Hackthon_Dataset/Hackathon2_scripts
# check file paths in yolo_params.yaml
```

### performance optimization

#### for cpu training
- use smaller batch sizes (8-16)
- reduce image size to 416x416
- limit workers to 4-8
- use mixed precision training (`amp=True`)

#### for better results
- increase training epochs (50-100)
- use higher learning rates (0.001-0.01)
- enable more data augmentation
- try different model sizes (yolov8m, yolov8l)

## hackathon deliverables

### completed
- [x] **baseline model training** with yolov8s
- [x] **optimized model training** (5 epochs, 73.18% map@0.5)
- [x] **comprehensive performance report** (8 pages)
- [x] **space station safety monitor app** (streamlit)
- [x] **real-time detection interface**
- [x] **alert system for missing equipment**
- [x] **demo script and documentation**
- [x] **step-by-step reproduction guide**

### future enhancements
- [ ] **extended training** (50-100 epochs)
- [ ] **model ensemble** methods
- [ ] **real-time video processing**
- [ ] **falcon platform integration**

## contributing

we welcome contributions to improve voidvision:

1. **fork the repository**
2. **create a feature branch**
3. **make your changes**
4. **submit a pull request**

### areas for contribution
- **model optimization**: improve detection accuracy
- **ui/ux enhancement**: better user interface
- **documentation**: improve guides and examples
- **testing**: add comprehensive test coverage

## license

this project is developed for the duality ai space station hackathon challenge. all rights reserved.

## acknowledgments

- **duality ai** for providing the synthetic dataset and falcon platform
- **ultralytics** for the yolov8 framework
- **streamlit** for the web application framework
- **hackathon community** for inspiration and support

## contact

- **team**: voidvision
- **hackathon**: duality ai space station challenge #2
- **repository**: https://github.com/codebanditssss/VoidVision

---

> **ensuring astronaut safety through ai-powered equipment monitoring**
