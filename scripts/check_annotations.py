#!/usr/bin/env python3
# quick annotation checker
import argparse, sys
from pathlib import Path
from PIL import Image

def load_classes(path):
    classes=[]
    for ln in open(path,encoding='utf-8'):
        ln=ln.strip()
        if not ln: continue
        if ':' in ln:
            _,name=ln.split(':',1)
            classes.append(name.strip())
        else:
            classes.append(ln)
    return classes

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--labels', default='data/datasets/bts/labels')
    p.add_argument('--images', default='data/datasets/bts/images')
    p.add_argument('--classes', default='configs/classes.txt')
    args=p.parse_args()

    classes = load_classes(args.classes)
    n=len(classes)
    issues=0
    for split in ['train','val','test']:
        lbl_dir = Path(args.labels)/split
        img_dir = Path(args.images)/split
        if not lbl_dir.exists():
            print('Missing', lbl_dir); issues+=1; continue
        for txt in lbl_dir.glob('*.txt'):
            s=txt.read_text(encoding='utf-8').strip()
            if s=='':
                print('Empty label:', txt); issues+=1; continue
            for i,line in enumerate(s.splitlines(),1):
                parts=line.split()
                if len(parts)!=5:
                    print('Bad format', txt, 'line', i); issues+=1; continue
                cid = int(parts[0])
                if cid<0 or cid>=n:
                    print('Bad class id', cid, 'in', txt); issues+=1
    if issues==0:
        print('No issues found.')
    else:
        print('Found', issues, 'issues.')

if __name__=='__main__':
    main()
