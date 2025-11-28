#!/usr/bin/env python3
"""Inference wrapper for Ultralytics model.
Usage:
  python scripts/infer.py --weights runs/.../best.pt --source dataset/images/test --save
"""
import argparse, os
from ultralytics import YOLO

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--weights', default='models/best.pt')
    p.add_argument('--source', default='data/datasets/bts/images/test')
    p.add_argument('--save', action='store_true')
    p.add_argument('--conf', type=float, default=0.25)
    args=p.parse_args()

    model = YOLO(args.weights)
    results = model(args.source, conf=args.conf)
    if args.save:
        out='preds'
        os.makedirs(out, exist_ok=True)
        for i,r in enumerate(results):
            r.save(os.path.join(out,f'pred_{i}.jpg'))
        print('[INFO] Saved predictions to', out)
    else:
        print('[INFO] Inference done. Use --save to save images.')

if __name__=='__main__':
    main()
