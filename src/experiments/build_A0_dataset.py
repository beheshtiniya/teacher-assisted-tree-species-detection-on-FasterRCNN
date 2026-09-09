
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import pandas as pd
COLS=['filename','class','xmin','ymin','xmax','ymax']
def clean(path:Path)->pd.DataFrame:
    df=pd.read_csv(path,encoding='utf-8-sig'); missing=set(COLS)-set(df.columns)
    if missing: raise ValueError(f'{path}: missing {sorted(missing)}')
    out=df[COLS].copy(); out['filename']=out['filename'].astype(str).map(lambda x:Path(x).name)
    out['class']=pd.to_numeric(out['class'],errors='raise').astype(int)
    for c in COLS[2:]: out[c]=pd.to_numeric(out[c],errors='raise')
    if not set(out['class']).issubset({1,2,3,4}): raise ValueError('Class outside 1..4')
    out['confidence']=1.0; return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--val',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    counts={}
    for split,path in [('train',a.train),('val',a.val),('test',a.test)]:
        df=clean(path); df.to_csv(a.output_dir/f'{split}_A0.csv',index=False,encoding='utf-8-sig'); counts[split]=len(df)
    pd.DataFrame([{'item':f'{k}_boxes','count':v} for k,v in counts.items()]).to_csv(a.output_dir/'A0_summary.csv',index=False,encoding='utf-8-sig')
if __name__=='__main__': main()
