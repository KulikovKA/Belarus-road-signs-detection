#!/usr/bin/env python3
"""Convert classification-folder-structured dataset into YOLO full-image bbox dataset.
Example: a folder with subfolders per class, each containing full-frame photos of a sign.
Usage: python scripts/convert_from_folders.py --src "Road signs" --out data/datasets/bts
"""
import os, shutil, random, argparse
from pathlib import Path
from PIL import Image

def ensure(p): os.makedirs(p, exist_ok=True)

def load_classes_from_folders(src):
    return sorted([d.name for d in Path(src).iterdir() if d.is_dir()])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--src', default='Road signs')
    p.add_argument('--out', default='data/datasets/bts')
    p.add_argument('--split', default='0.8,0.1,0.1')  # train,val,testex
    p.add_argument('--seed', type=int, default=42)
    args=p.parse_args()

    src=Path(args.src)
    out=Path(args.out)
    train_p,val_p,test_p = map(float, args.split.split(','))
    random.seed(args.seed)

    classes = load_classes_from_folders(src)
    class_to_id = {c:i for i,c in enumerate(classes)}
    print('Detected classes:', class_to_id)

    for sp in ['train','val','test']:
        ensure(out / 'images' / sp)
        ensure(out / 'labels' / sp)

    for cls in classes:
        folder = src / cls
        imgs = [p for p in folder.glob('*.*') if p.suffix.lower() in ['.jpg','.jpeg','.png']]
        random.shuffle(imgs)
        n=len(imgs)
        n_train = int(n*train_p)
        n_val = int(n*val_p)
        train = imgs[:n_train]
        val = imgs[n_train:n_train+n_val]
        test = imgs[n_train+n_val:]
        for im in train:
            shutil.copy(im, out / 'images' / 'train' / im.name)
            with Image.open(im) as I:
                w,h = I.size
            with open(out / 'labels' / 'train' / (im.stem + '.txt'), 'w', encoding='utf-8') as f:
                # full-image bbox
                f.write(f"{class_to_id[cls]} 0.5 0.5 1.0 1.0\n")
        for im in val:
            shutil.copy(im, out / 'images' / 'val' / im.name)
            with Image.open(im) as I:
                w,h = I.size
            with open(out / 'labels' / 'val' / (im.stem + '.txt'), 'w', encoding='utf-8') as f:
                f.write(f"{class_to_id[cls]} 0.5 0.5 1.0 1.0\n")
        for im in test:
            shutil.copy(im, out / 'images' / 'test' / im.name)
            with Image.open(im) as I:
                w,h = I.size
            with open(out / 'labels' / 'test' / (im.stem + '.txt'), 'w', encoding='utf-8') as f:
                f.write(f"{class_to_id[cls]} 0.5 0.5 1.0 1.0\n")

    # write configs/data.yaml
    with open('configs/data.yaml','w', encoding='utf-8') as f:
        f.write('path: data/datasets/bts\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('test: images/test\n')
        f.write('names:\n')
        for i,c in enumerate(classes):
            f.write(f'  {i}: {c}\n')
    print('Conversion done.')
if __name__=='__main__':
    main()
