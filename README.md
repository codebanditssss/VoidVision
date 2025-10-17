# 🚀 voidvision - space station safety monitor

> ai-powered safety equipment detection for space station environments

[![python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![yolov8](https://img.shields.io/badge/yolo-v8-green.svg)](https://github.com/ultralytics/ultralytics)
[![streamlit](https://img.shields.io/badge/streamlit-app-red.svg)](https://streamlit.io/)
[![hackathon](https://img.shields.io/badge/hackathon-duality%20ai-orange.svg)](https://duality.ai/)

## 🌟 overview

voidvision is an advanced object detection system designed to identify critical safety equipment in space station environments. using yolov8 architecture and synthetic data from duality ai's falcon platform, we've created a comprehensive safety monitoring solution that ensures astronaut safety through automated equipment detection.

### 🎯 hackathon challenge
**duality ai space station challenge #2** - develop an ai system to detect safety equipment in space station environments using synthetic data from digital twin platforms.

## ✨ key features

- **real-time detection** of 7 critical safety equipment types
- **interactive web interface** with space-themed design
- **automated safety scoring** and equipment status monitoring
- **missing equipment alerts** with immediate notifications
- **comprehensive reporting** with json export capabilities
- **confidence threshold adjustment** for fine-tuned detection
- **detection history tracking** for safety audits

## 🛠️ safety equipment detected

| equipment | purpose | criticality |
|-----------|---------|-------------|
| 🫁 oxygen tank | life support | critical |
| 🧊 nitrogen tank | life support | critical |
| 🏥 first aid box | emergency medical | high |
| 🔥 fire alarm | fire safety | high |
| ⚡ safety switch panel | control interface | medium |
| 📞 emergency phone | communication | medium |
| 🧯 fire extinguisher | fire suppression | high |

## 📊 performance metrics

### baseline results
- **mAP@0.5**: 35.7% (solid foundation for optimization)
- **precision**: 46.7% (good accuracy when detecting)
- **recall**: 34.4% (room for improvement)
- **training time**: 29.8 minutes (cpu-optimized)

### dataset statistics
- **total images**: 3,511 synthetic images
- **training set**: 1,767 images
- **validation set**: 336 images
- **test set**: 1,408 images
- **classes**: 7 safety equipment types

## 🚀 quick start

### prerequisites
- python 3.10+
- anaconda/miniconda
- 8gb+ ram recommended

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

## 🖥️ application interface

### space station safety monitor
- **real-time detection**: upload images or use camera input
- **equipment status**: live monitoring of all safety equipment
- **alert system**: immediate notifications for missing equipment
- **confidence adjustment**: fine-tune detection sensitivity
- **history tracking**: view past detection results

### demo script
- **automated detection**: batch processing of test images
- **safety reporting**: generate comprehensive safety reports
- **json export**: structured data for further analysis

## 📁 project structure

```
voidvision/
├── 📄 readme.md                    # project documentation
├── 📄 requirements.txt             # python dependencies
├── 📄 performance_report.md        # comprehensive analysis report
├── 📄 progress_tracking.md         # development progress
├── 🚀 safety_monitor_app.py        # main streamlit application
├── 🔧 demo_safety_monitor.py       # demo script
├── 🤖 train.py                     # model training script
├── 📊 predict.py                   # prediction script
├── 🎨 visualize.py                 # visualization utilities
├── ⚙️ yolo_params.yaml             # model configuration
├── 🏷️ classes.txt                  # class definitions
└── 📁 hackthon_dataset/            # training data and scripts
    └── hackathon2_scripts/
        ├── 🎯 yolov8s.pt           # trained model weights
        ├── 📈 runs/                # training results
        └── 📁 env_setup/            # environment setup
```

## 🔬 technical approach

### model architecture
- **base model**: yolov8s (small variant for efficiency)
- **input size**: 640x640 pixels
- **classes**: 7 safety equipment categories
- **optimizer**: adamw with learning rate 0.0001
- **augmentation**: mosaic, flip, color jitter

### training process
- **dataset**: synthetic data from falcon digital twin platform
- **device**: cpu-optimized training
- **epochs**: 1 (baseline) with plans for 50-100 epochs
- **validation**: real-time validation during training

### optimization strategies
- **hyperparameter tuning**: learning rate, batch size, epochs
- **data augmentation**: enhanced mosaic, flip, rotation
- **model variants**: yolov8s, yolov8m, yolov8l testing
- **ensemble methods**: multiple model combination

## 📈 results and analysis

### baseline performance
our initial training achieved:
- **mAP@0.5**: 35.7% - solid foundation for optimization
- **precision**: 46.7% - good accuracy when detecting objects
- **recall**: 34.4% - indicates room for improvement
- **training stability**: consistent loss reduction

### optimization potential
identified improvement areas:
- **extended training**: increase epochs to 50-100
- **learning rate optimization**: test higher learning rates
- **data augmentation**: implement advanced techniques
- **model architecture**: test larger yolov8 variants

## 🎯 hackathon deliverables

### ✅ completed
- [x] **baseline model training** with yolov8s
- [x] **comprehensive performance report** (8 pages)
- [x] **space station safety monitor app** (streamlit)
- [x] **real-time detection interface**
- [x] **alert system for missing equipment**
- [x] **demo script and documentation**

### 🔄 in progress
- [ ] **model optimization** (extended training)
- [ ] **performance visualization** (confusion matrix, loss curves)
- [ ] **falcon integration strategy**

## 🚀 future enhancements

### immediate improvements
- **extended training**: 50-100 epochs for better convergence
- **hyperparameter optimization**: systematic grid search
- **advanced augmentation**: rotation, scaling, color space
- **model ensemble**: combine multiple model predictions

### advanced features
- **real-time video processing**: continuous monitoring
- **edge deployment**: optimize for space station hardware
- **continuous learning**: falcon platform integration
- **multi-camera support**: distributed monitoring system

### real-world deployment
- **space station integration**: falcon platform deployment
- **astronaut interface**: simplified monitoring dashboard
- **emergency protocols**: automated safety procedures
- **maintenance scheduling**: predictive equipment monitoring

## 🤝 contributing

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

## 📄 license

this project is developed for the duality ai space station hackathon challenge. all rights reserved.

## 🙏 acknowledgments

- **duality ai** for providing the synthetic dataset and falcon platform
- **ultralytics** for the yolov8 framework
- **streamlit** for the web application framework
- **hackathon community** for inspiration and support

## 📞 contact

- **team**: voidvision
- **hackathon**: duality ai space station challenge #2
- **repository**: https://github.com/codebanditssss/VoidVision

---

> **ensuring astronaut safety through ai-powered equipment monitoring** 🚀
