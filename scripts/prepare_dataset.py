#!/usr/bin/env python3
"""prepare_dataset.py
Converts a raw folder with images and optional annotations into the YOLO dataset layout.
Usage:
  python scripts/prepare_dataset.py --raw data/raw --out data/datasets/bts --format yolo|voc
"""
import argparse, shutil, os
from pathlib import Path
import random
import xml.etree.ElementTree as ET

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def voc_to_yolo(xml_file, classes):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    size = root.find('size')
    w = float(size.find('width').text)
    h = float(size.find('height').text)
    lines = []
    for obj in root.iter('object'):
        name = obj.find('name').text
        if name not in classes:
            continue
        cls_id = classes.index(name)
        bb = obj.find('bndbox')
        xmin = float(bb.find('xmin').text)
        ymin = float(bb.find('ymin').text)
        xmax = float(bb.find('xmax').text)
        ymax = float(bb.find('ymax').text)
        x = ((xmin + xmax) / 2) / w
        y = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        lines.append(f"{cls_id} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}")
    return lines

def load_classes(path='configs/classes.txt'):
    classes = []
    with open(path, encoding='utf-8') as f:
        for ln in f:
            ln=ln.strip()
            if not ln: continue
            if ':' in ln:
                _, name = ln.split(':',1)
                classes.append(name.strip())
            else:
                classes.append(ln)
    return classes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/raw')
    parser.add_argument('--out', default='data/datasets/bts')
    parser.add_argument('--format', choices=['voc','yolo'], default='yolo')
    parser.add_argument('--val', type=float, default=0.2)
    parser.add_argument('--test', type=float, default=0.05)
    args=parser.parse_args()

    raw=Path(args.raw)
    out=Path(args.out)
    classes = load_classes()

    imgs = list((raw / 'images').glob('*.*'))
    imgs = [p for p in imgs if p.suffix.lower() in ['.jpg','.jpeg','.png']]
    random.shuffle(imgs)
    n=len(imgs)
    n_test = int(n*args.test)
    n_val = int(n*args.val)
    n_train = n - n_val - n_test

    parts = {'train': imgs[:n_train], 'val': imgs[n_train:n_train+n_val], 'test': imgs[n_train+n_val:]}
    for part, files in parts.items():
        (out / 'images' / part).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / part).mkdir(parents=True, exist_ok=True)
        for img in files:
            shutil.copy(img, out / 'images' / part / img.name)
            stem = img.stem
            if args.format == 'voc':
                xml = raw / 'labels' / (stem + '.xml')
                if xml.exists():
                    lines = voc_to_yolo(xml, classes)
                    with open(out / 'labels' / part / (stem + '.txt'), 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))
                else:
                    open(out / 'labels' / part / (stem + '.txt'), 'w', encoding='utf-8').close()
            else:
                # assume raw/labels has yolo txt files
                txt = raw / 'labels' / (stem + '.txt')
                if txt.exists():
                    shutil.copy(txt, out / 'labels' / part / (stem + '.txt'))
                else:
                    open(out / 'labels' / part / (stem + '.txt'), 'w', encoding='utf-8').close()

    # write data.yaml
    with open('configs/data.yaml','w', encoding='utf-8') as f:
        f.write('path: data/datasets/bts\n')
        f.write('train: images/train\n')
        f.write('val: images/val\n')
        f.write('test: images/test\n')
        f.write('names:\n')
        for i,c in enumerate(classes):
            f.write(f'  {i}: {c}\n')
    print('Done. dataset prepared at', out)

if __name__=='__main__':
    main()
