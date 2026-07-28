
from __future__ import annotations
import argparse, json, py_compile
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.common.config import load_config

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path); a=p.parse_args(); c=load_config(a.config)
    required=[c.images,c.labels/'train_labels.csv',c.labels/'val_labels.csv',c.labels/'test_labels.csv',c.labels/'unlabeled_images.txt',c.repo,c.repo/'train.py',c.repo/'detect.py']
    missing=[x for x in required if not x.exists()]
    if missing: raise FileNotFoundError('Missing:\n'+'\n'.join(map(str,missing)))
    for f in (c.project_root/'src').rglob('*.py'): py_compile.compile(str(f),doraise=True)
    print('PREFLIGHT PASS'); print('Python:',c.python); print('Data:',c.data_root); print('EfficientTree:',c.repo); print('Output:',c.output_root)
if __name__=='__main__': main()
