#!/usr/bin/env python3
"""Stagewise training: short freeze then unfreeze.
Usage:
  python scripts/train_stagewise.py --data configs/data.yaml --weights yolov11n.pt
"""
import argparse, os
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data', default='configs/data.yaml')
    p.add_argument('--weights', default='yolov11n.pt')
    p.add_argument('--freeze_epochs', type=int, default=10)
    p.add_argument('--finetune_epochs', type=int, default=40)
    p.add_argument('--imgsz', type=int, default=1024)
    args=p.parse_args()

    model = YOLO(args.weights)
    print('[INFO] Stage 1: train head (freeze backbone)')
    model.train(data=args.data, epochs=args.freeze_epochs, imgsz=args.imgsz, project='runs/belarus_signs', name='stage1_freeze', freeze=[0])
    stage1 = 'runs/belarus_signs/stage1_freeze/weights/best.pt'
    if not os.path.exists(stage1):
        stage1 = args.weights
    model2 = YOLO(stage1)
    print('[INFO] Stage 2: unfreeze and finetune')
    model2.train(data=args.data, epochs=args.finetune_epochs, imgsz=args.imgsz, project='runs/belarus_signs', name='stage2_unfreeze')

if __name__=='__main__':
    main()
