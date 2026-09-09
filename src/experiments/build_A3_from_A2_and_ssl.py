
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd
BASE=['filename','class','xmin','ymin','xmax','ymax']; MAIN=BASE+['confidence']
def iou(box,boxes):
    if len(boxes)==0:return np.zeros(0)
    xx1=np.maximum(box[0],boxes[:,0]); yy1=np.maximum(box[1],boxes[:,1]); xx2=np.minimum(box[2],boxes[:,2]); yy2=np.minimum(box[3],boxes[:,3]); inter=np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1); a=max(0,(box[2]-box[0])*(box[3]-box[1])); b=np.maximum(0,(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])); return inter/np.maximum(a+b-inter,1e-12)
def read(path):
    d=pd.read_csv(path,encoding='utf-8-sig'); miss=set(MAIN)-set(d.columns)
    if miss: raise ValueError(f'{path}: missing {sorted(miss)}')
    d=d.copy(); d.filename=d.filename.astype(str).map(lambda x:Path(x).name); return d
def main():
    p=argparse.ArgumentParser(); p.add_argument('--a2-train',type=Path,required=True); p.add_argument('--ssl-predictions',type=Path,required=True); p.add_argument('--val-gt',type=Path,required=True); p.add_argument('--test-gt',type=Path,required=True); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--minimum-confidence',type=float,default=0.50); p.add_argument('--overlap-iou',type=float,default=0.50); a=p.parse_args()
    base=read(a.a2_train); pred=read(a.ssl_predictions); pred=pred[pred.confidence>=a.minimum_confidence].copy(); groups={k:g[BASE[2:]].to_numpy(float) for k,g in base.groupby(base.filename.str.lower())}; keep=[]; maxes=[]
    for _,r in pred.iterrows():
        m=float(iou(r[BASE[2:]].to_numpy(float),groups.get(r.filename.lower(),np.empty((0,4)))).max(initial=0.0)); maxes.append(m); keep.append(m<a.overlap_iou)
    pred['max_iou_with_A2']=maxes; accepted=pred.loc[keep].drop_duplicates(subset=BASE).copy(); final=pd.concat([base[MAIN],accepted[MAIN]],ignore_index=True); a.output_dir.mkdir(parents=True,exist_ok=True)
    final.to_csv(a.output_dir/'train_A3.csv',index=False,encoding='utf-8-sig'); accepted.to_csv(a.output_dir/'accepted_ssl_predictions.csv',index=False,encoding='utf-8-sig'); pred.loc[[not x for x in keep]].to_csv(a.output_dir/'rejected_ssl_predictions.csv',index=False,encoding='utf-8-sig')
    for split,path in [('val',a.val_gt),('test',a.test_gt)]:
        d=pd.read_csv(path,encoding='utf-8-sig')[BASE].copy(); d['confidence']=1.0; d[MAIN].to_csv(a.output_dir/f'{split}_A3_expert_only.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'item':'A2_base_boxes','count':len(base)},{'item':'ssl_candidates_after_confidence','count':len(pred)},{'item':'accepted_ssl_novel_boxes','count':len(accepted)},{'item':'final_A3_boxes','count':len(final)}]).to_csv(a.output_dir/'A3_summary.csv',index=False,encoding='utf-8-sig')
if __name__=='__main__': main()
