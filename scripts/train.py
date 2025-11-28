#!/usr/bin/env python3
"""Simple training wrapper using ultralytics YOLO.
Usage:
  python scripts/train.py --data configs/data.yaml --weights yolov11n.pt --epochs 50 --imgsz 1024
If specified weights not found locally, Ultralytics will download official weights automatically.
"""

'''запуск:
& C:/Users/kiril/anaconda3/envs/MLops_main/python.exe "c:/Users/kiril/OneDrive/Рабочий стол/BTSF/scripts/train.py" --weights yolo11n.pt


'''
import argparse, os
from ultralytics import YOLO

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--data', default='configs/data.yaml')
    p.add_argument('--weights', default='yolov11n.pt', help='model name or path (ultralytics will download if missing)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--project', default='runs/belarus_signs')
    p.add_argument('--name', default='yolov11_train')
    return p.parse_args()

def main():
    args=parse_args()
    print(f"[INFO] Loading model {args.weights}")
    model = YOLO(args.weights)  
    print("[INFO] Starting training...")
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=args.project, name=args.name)
    print('[INFO] Training finished.')

if __name__=='__main__':
    main()
