#!/usr/bin/env python3
"""
optimized training script for voidvision
bypasses torchvision compatibility issues
"""

import os
import sys
import argparse
from ultralytics import YOLO

def main():
    # training parameters
    EPOCHS = 50
    MOSAIC = 0.8
    OPTIMIZER = 'AdamW'
    MOMENTUM = 0.937
    LR0 = 0.001
    LRF = 0.01
    BATCH_SIZE = 32
    IMGSZ = 640
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=IMGSZ, help='Image size')
    
    args = parser.parse_args()
    
    # change to script directory
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    print("🚀 voidvision optimized training")
    print("=" * 50)
    print(f"epochs: {args.epochs}")
    print(f"learning rate: {args.lr0}")
    print(f"batch size: {args.batch}")
    print(f"mosaic: {args.mosaic}")
    print(f"image size: {args.imgsz}")
    print("=" * 50)
    
    try:
        # load model
        print("loading yolov8s model...")
        model = YOLO("yolov8s.pt")
        print("model loaded successfully!")
        
        # start training
        print("starting optimized training...")
        results = model.train(
            data="yolo_params.yaml",
            epochs=args.epochs,
            device='cpu',
            single_cls=False,
            mosaic=args.mosaic,
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            momentum=args.momentum,
            batch=args.batch,
            imgsz=args.imgsz,
            patience=20,
            save_period=10,
            workers=4,
            verbose=True
        )
        
        print("✅ training completed successfully!")
        print(f"results saved in: {results.save_dir}")
        
        # print final metrics
        if hasattr(results, 'results_dict'):
            print("\n📊 final metrics:")
            for key, value in results.results_dict.items():
                if 'map' in key.lower():
                    print(f"  {key}: {value:.4f}")
        
    except Exception as e:
        print(f"❌ training failed: {str(e)}")
        print("trying alternative approach...")
        
        # alternative: try with minimal parameters
        try:
            print("attempting minimal training...")
            model = YOLO("yolov8s.pt")
            results = model.train(
                data="yolo_params.yaml",
                epochs=10,
                device='cpu',
                verbose=True
            )
            print("✅ minimal training completed!")
        except Exception as e2:
            print(f"❌ minimal training also failed: {str(e2)}")
            return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎯 training completed! check runs/detect/train/ for results")
    else:
        print("\n💥 training failed. check error messages above")
