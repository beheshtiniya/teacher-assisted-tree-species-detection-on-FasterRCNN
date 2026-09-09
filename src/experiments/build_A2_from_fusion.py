
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import pandas as pd
BASE=['filename','class','xmin','ymin','xmax','ymax']; MAIN=BASE+['confidence']
def expert(path):
    d=pd.read_csv(path,encoding='utf-8-sig')[BASE].copy(); d['filename']=d.filename.astype(str).map(lambda x:Path(x).name); d['confidence']=1.0; return d[MAIN]
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train-fused',type=Path,required=True); p.add_argument('--val-gt',type=Path,required=True); p.add_argument('--test-gt',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(a.train_fused,encoding='utf-8-sig'); miss=set(MAIN)-set(d.columns)
    if miss: raise ValueError(f'Missing columns: {sorted(miss)}')
    d[MAIN].to_csv(a.output_dir/'train_A2.csv',index=False,encoding='utf-8-sig'); d.to_csv(a.output_dir/'train_A2_audit.csv',index=False,encoding='utf-8-sig'); expert(a.val_gt).to_csv(a.output_dir/'val_A2_expert_only.csv',index=False,encoding='utf-8-sig'); expert(a.test_gt).to_csv(a.output_dir/'test_A2_expert_only.csv',index=False,encoding='utf-8-sig')
    source=d['source'].value_counts().to_dict() if 'source' in d else {}
    rows=[{'item':'final_train_boxes','value':len(d)}]+[{'item':f'source_{k}','value':v} for k,v in source.items()]
    pd.DataFrame(rows).to_csv(a.output_dir/'A2_summary.csv',index=False,encoding='utf-8-sig')
if __name__=='__main__': main()
