
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
BASE=['filename','class','xmin','ymin','xmax','ymax']; MAIN=BASE+['confidence']
def read(path:Path,conf:bool):
    df=pd.read_csv(path,encoding='utf-8-sig'); need=set(BASE+(['confidence'] if conf else [])); miss=need-set(df.columns)
    if miss: raise ValueError(f'{path}: missing {sorted(miss)}')
    df=df.copy(); df['filename']=df['filename'].astype(str).map(lambda x:Path(x).name); df['class']=pd.to_numeric(df['class'],errors='raise').astype(int)
    for c in BASE[2:]: df[c]=pd.to_numeric(df[c],errors='raise')
    if conf: df['confidence']=pd.to_numeric(df['confidence'],errors='raise')
    else: df['confidence']=1.0
    return df

def main():
    p=argparse.ArgumentParser(); p.add_argument('--name',required=True); p.add_argument('--train-gt',type=Path,required=True); p.add_argument('--val-gt',type=Path,required=True); p.add_argument('--test-gt',type=Path,required=True); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--minimum-confidence',type=float,default=0.25); a=p.parse_args()
    gt=read(a.train_gt,False); pred=read(a.predictions,True); pred=pred[pred.confidence>=a.minimum_confidence].copy()
    pred=pred.drop_duplicates(subset=BASE).sort_values(['filename','confidence'],ascending=[True,False])
    train=pd.concat([gt[MAIN],pred[MAIN]],ignore_index=True); a.output_dir.mkdir(parents=True,exist_ok=True)
    train.to_csv(a.output_dir/f'train_{a.name}.csv',index=False,encoding='utf-8-sig')
    read(a.val_gt,False)[MAIN].to_csv(a.output_dir/f'val_{a.name}_expert_only.csv',index=False,encoding='utf-8-sig')
    read(a.test_gt,False)[MAIN].to_csv(a.output_dir/f'test_{a.name}_expert_only.csv',index=False,encoding='utf-8-sig')
    audit=pd.concat([gt.assign(source='ground_truth')[MAIN+['source']],pred.assign(source='model_prediction')[MAIN+['source']]],ignore_index=True)
    audit.to_csv(a.output_dir/f'train_{a.name}_audit.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'item':'human_gt_boxes','count':len(gt)},{'item':'accepted_prediction_boxes','count':len(pred)},{'item':'final_train_boxes','count':len(train)}]).to_csv(a.output_dir/f'{a.name}_summary.csv',index=False,encoding='utf-8-sig')
if __name__=='__main__': main()
