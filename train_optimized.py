EPOCHS = 50
MOSAIC = 0.8
OPTIMIZER = 'AdamW'
MOMENTUM = 0.937
LR0 = 0.001
LRF = 0.01
SINGLE_CLS = False
BATCH_SIZE = 32
IMGSZ = 640

import argparse
from ultralytics import YOLO
import os
import sys

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # epochs
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    # mosaic
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    # optimizer
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    # momentum
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    # lr0
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    # lrf
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    # single_cls
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    # batch size
    parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='Batch size')
    # image size
    parser.add_argument('--imgsz', type=int, default=IMGSZ, help='Image size')
    
    args = parser.parse_args()
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    print(f"Starting optimized training with:")
    print(f"- Epochs: {args.epochs}")
    print(f"- Learning Rate: {args.lr0}")
    print(f"- Batch Size: {args.batch}")
    print(f"- Mosaic: {args.mosaic}")
    print(f"- Image Size: {args.imgsz}")
    
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"), 
        epochs=args.epochs,
        device='cpu',
        single_cls=args.single_cls, 
        mosaic=args.mosaic,
        optimizer=args.optimizer, 
        lr0=args.lr0, 
        lrf=args.lrf, 
        momentum=args.momentum,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=10,
        save_period=10,
        workers=4
    )
