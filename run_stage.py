
from __future__ import annotations
import argparse, json, os, subprocess, sys, datetime as dt
from pathlib import Path
from src.common.config import load_config
from src.common.process import run

ROOT=Path(__file__).resolve().parent

def choose_retina_checkpoint(summary:Path, output:Path)->Path:
    import pandas as pd
    d=pd.read_csv(summary); metric='best_val_map_50'; row=d.loc[d[metric].astype(float).idxmax()]; run_id=int(row['run']); ck=output/f'run_{run_id:02d}'/'best_model.pth'
    if not ck.is_file(): raise FileNotFoundError(ck)
    return ck

def choose_warm_checkpoint(repo:Path)->Path:
    w=repo/'runs'/'mydata_full_unbalanced'/'mydata_dual_teacher_ssl_warmstart_burn0_staged70'/'weights'
    for n in ('best_ema.pt','best.pt'):
        p=w/n
        if p.is_file(): return p
    raise FileNotFoundError(f'No best_ema.pt or best.pt in {w}')

def cmd_stage(n:int,c,dry:bool):
    py=c.python; s=c.settings; labels=c.labels; out=c.output_root; repo=c.repo
    dataset=out/'datasets'/'efficienttree_640_full_unbalanced'; et_cfg=repo/'configs'/'ssod'/'custom'; sup_yaml=et_cfg/'mydata_full_supervised_4class_staged70.yaml'; ssl_yaml=et_cfg/'mydata_full_ssl_4class_staged70.yaml'; sup_run=repo/'runs'/'mydata_full_unbalanced'/'mydata_full_supervised_4class_staged70'; sup_best=sup_run/'weights'/'best.pt'
    et_sup_pred=out/'predictions'/'efficientteacher_supervised'; retina_model=out/'models'/'retinanet'; retina_pred=out/'predictions'/'retinanet'; fusion=out/'fusion'/'supervised_et_retinanet'; warm_labels=out/'datasets'/'warmstart_labels'; warm_dataset=out/'datasets'/'efficienttree_640_warmstart'; warm_yaml=et_cfg/'mydata_full_ssl_warmstart_burn0_staged70.yaml'; warm_run_name='mydata_dual_teacher_ssl_warmstart_burn0_staged70'; warm_pred=out/'predictions'/'efficientteacher_warmstart_ssl'; exps=out/'experiments'; fr=out/'models'/'fasterrcnn'
    if n==1: run([py,ROOT/'src/pipeline/preflight.py'],dry_run=dry)
    elif n==2: run([py,ROOT/'src/data/prepare_mydata_640.py','--root',c.data_root,'--images-dir',c.images,'--labels-dir',labels,'--output-dir',dataset,'--size',s['image_size_et'],'--overwrite'],dry_run=dry)
    elif n==3: run([py,ROOT/'src/efficienttree/make_configs.py','--repo',repo,'--dataset-dir',dataset,'--batch-size','4','--workers','0','--img-size',s['image_size_et'],'--epochs',s['et_epochs'],'--burn-epochs','1','--class-names',*c.raw['classes']],dry_run=dry)
    elif n==4: run([py,repo/'train.py','--cfg',et_cfg/'mydata_smoke_ssl_4class.yaml','epochs','70','name','mydata_smoke_ssl_4class','exist_ok','True'],cwd=repo,env={'EFFICIENTTREE_STOP_EPOCH':'2'},dry_run=dry)
    elif n==5:
        import yaml
        src=et_cfg/'mydata_smoke_ssl_4class.yaml'; dst=et_cfg/'mydata_smoke_ssl_4class_burn10.yaml'
        if not dry:
            d=yaml.safe_load(src.read_text(encoding='utf-8-sig')); d.setdefault('hyp',{})['burn_epochs']=10; d['name']='mydata_smoke_ssl_4class_burn10'; d['resume']=False; d['exist_ok']=True; dst.write_text(yaml.safe_dump(d,sort_keys=False,allow_unicode=True),encoding='utf-8')
        run([py,repo/'train.py','--cfg',dst,'epochs','70','name','mydata_smoke_ssl_4class_burn10','exist_ok','True'],cwd=repo,env={'EFFICIENTTREE_STOP_EPOCH':'12'},dry_run=dry)
    elif n==6: run([py,repo/'train.py','--cfg',sup_yaml,'epochs',s['et_epochs'],'name','mydata_full_supervised_4class_staged70','exist_ok','True'],cwd=repo,env={'EFFICIENTTREE_STOP_EPOCH':str(s['et_epochs'])},dry_run=dry)
    elif n==7: run([py,ROOT/'src/experiments/build_A0_dataset.py','--train',labels/'train_labels.csv','--val',labels/'val_labels.csv','--test',labels/'test_labels.csv','--output-dir',exps/'A0'],dry_run=dry)
    elif n==8:
        run([py,ROOT/'src/efficienttree/predict_all_splits.py','--python',py,'--repo',repo,'--dataset',dataset,'--labels-dir',labels,'--unlabeled-list',labels/'unlabeled_images.txt','--weights',sup_best,'--output',et_sup_pred,'--tag','supervised','--confidence',s['confidence'],'--gt-iou',s['gt_iou']],dry_run=dry)
        run([py,ROOT/'src/experiments/build_expert_plus_predictions.py','--name','A1','--train-gt',labels/'train_labels.csv','--val-gt',labels/'val_labels.csv','--test-gt',labels/'test_labels.csv','--predictions',et_sup_pred/'train_missing_object_candidates.csv','--output-dir',exps/'A1','--minimum-confidence',s['confidence']],dry_run=dry)
    elif n==9: run([py,ROOT/'src/retinanet/train_retinanet.py','--images-dir',c.images,'--train-csv',labels/'train_labels.csv','--val-csv',labels/'val_labels.csv','--test-csv',labels/'test_labels.csv','--output-dir',retina_model,'--num-runs',s['retina_runs'],'--max-epochs',s['retina_epochs']],dry_run=dry)
    elif n==10:
        ck=retina_model/'run_01'/'best_model.pth' if dry else choose_retina_checkpoint(retina_model/'all_runs_summary.csv',retina_model)
        run([py,ROOT/'src/retinanet/predict_retinanet.py','--checkpoint',ck,'--images-dir',c.images,'--train-csv',labels/'train_labels.csv','--val-csv',labels/'val_labels.csv','--test-csv',labels/'test_labels.csv','--unlabeled-list',labels/'unlabeled_images.txt','--output-root',retina_pred,'--confidence',s['confidence'],'--nms-iou',s['gt_iou'],'--gt-iou',s['gt_iou']],dry_run=dry)
    elif n==11:
        run([py,ROOT/'src/fusion/dual_teacher_fusion.py','--gt-train',labels/'train_labels.csv','--gt-val',labels/'val_labels.csv','--gt-test',labels/'test_labels.csv','--et-train',et_sup_pred/'train_gt_plus_predictions.csv','--et-val',et_sup_pred/'val_gt_plus_predictions.csv','--et-test',et_sup_pred/'test_gt_plus_predictions.csv','--retina-train',retina_pred/'combined'/'train_gt_plus_filtered_predictions.csv','--retina-val',retina_pred/'combined'/'val_gt_plus_filtered_predictions.csv','--retina-test',retina_pred/'combined'/'test_gt_plus_filtered_predictions.csv','--et-unlabeled',et_sup_pred/'unlabeled_predictions_256.csv','--retina-unlabeled',retina_pred/'predictions'/'unlabeled_predictions_confidence_ge_0p25_from_initial_all_predictions.csv','--output-dir',fusion,'--min-et-confidence',s['confidence'],'--min-retina-confidence',s['confidence'],'--agreement-iou',s['agreement_iou'],'--single-model-confidence',s['single_model_confidence'],'--gt-iou',s['gt_iou']],dry_run=dry)
        run([py,ROOT/'src/experiments/build_A2_from_fusion.py','--train-fused',fusion/'combined'/'train_gt_plus_fused_predictions.csv','--val-gt',labels/'val_labels.csv','--test-gt',labels/'test_labels.csv','--output-dir',exps/'A2'],dry_run=dry)
    elif n==12:
        run([py,ROOT/'src/fusion/build_warmstart_labels.py','--train-gt',labels/'train_labels.csv','--val-gt',labels/'val_labels.csv','--test-gt',labels/'test_labels.csv','--unlabeled-list',labels/'unlabeled_images.txt','--train-fused',fusion/'predictions'/'train_fused_predictions.csv','--unlabeled-fused',fusion/'predictions'/'unlabeled_fused_predictions.csv','--output-dir',warm_labels,'--minimum-confidence',s['warmstart_confidence']],dry_run=dry)
        run([py,ROOT/'src/data/prepare_mydata_640.py','--root',c.data_root,'--images-dir',c.images,'--labels-dir',warm_labels,'--train-csv',warm_labels/'train_labels.csv','--val-csv',warm_labels/'val_labels.csv','--test-csv',warm_labels/'test_labels.csv','--unlabeled-list',warm_labels/'unlabeled_images.txt','--output-dir',warm_dataset,'--size',s['image_size_et'],'--overwrite'],dry_run=dry)
        run([py,ROOT/'src/efficienttree/make_warmstart_config.py','--source-yaml',ssl_yaml,'--dataset',warm_dataset,'--weights',sup_best,'--output-yaml',warm_yaml,'--run-name',warm_run_name,'--burn-epochs','0'],dry_run=dry)
    elif n==13: run([py,repo/'train.py','--cfg',warm_yaml,'epochs',s['et_epochs'],'name',warm_run_name,'exist_ok','True'],cwd=repo,env={'EFFICIENTTREE_STOP_EPOCH':str(s['et_epochs'])},dry_run=dry)
    elif n==14:
        ck=repo/'runs'/'mydata_full_unbalanced'/warm_run_name/'weights'/'best_ema.pt' if dry else choose_warm_checkpoint(repo)
        run([py,ROOT/'src/efficienttree/predict_all_splits.py','--python',py,'--repo',repo,'--dataset',dataset,'--labels-dir',labels,'--unlabeled-list',labels/'unlabeled_images.txt','--weights',ck,'--output',warm_pred,'--tag','warmstart_ssl','--confidence',s['confidence'],'--gt-iou',s['gt_iou']],dry_run=dry)
        run([py,ROOT/'src/experiments/build_A3_from_A2_and_ssl.py','--a2-train',exps/'A2'/'train_A2.csv','--ssl-predictions',warm_pred/'train_missing_object_candidates.csv','--val-gt',labels/'val_labels.csv','--test-gt',labels/'test_labels.csv','--output-dir',exps/'A3','--minimum-confidence',s['warmstart_confidence'],'--overlap-iou',s['gt_iou']],dry_run=dry)
    elif n==15:
        mapping={'A0':('train_A0.csv','val_A0.csv','test_A0.csv'),'A1':('train_A1.csv','val_A1_expert_only.csv','test_A1_expert_only.csv'),'A2':('train_A2.csv','val_A2_expert_only.csv','test_A2_expert_only.csv'),'A3':('train_A3.csv','val_A3_expert_only.csv','test_A3_expert_only.csv')}
        for exp,(tr,va,te) in mapping.items(): run([py,ROOT/'src/fasterrcnn/train_fasterrcnn.py','--images-dir',c.images,'--train-csv',exps/exp/tr,'--val-csv',exps/exp/va,'--test-csv',exps/exp/te,'--output-dir',fr/exp,'--num-runs',s['fasterrcnn_runs'],'--max-epochs',s['fasterrcnn_epochs']],dry_run=dry)
    elif n==16: run([py,ROOT/'src/reports/summarize_results.py','--root',fr,'--output',out/'reports'/'fasterrcnn_A0_A3_mean_std.csv'],dry_run=dry)
    else: raise ValueError(f'Unknown stage {n}')

def main():
    p=argparse.ArgumentParser(); p.add_argument('stage',type=int); p.add_argument('--to',type=int); p.add_argument('--dry-run',action='store_true'); p.add_argument('--force',action='store_true'); a=p.parse_args(); c=load_config(); end=a.to or a.stage
    for n in range(a.stage,end+1):
        state=c.out('stage_state',f'{n:02d}.done.json')
        if state.is_file() and not a.force and not a.dry_run: print(f'SKIP stage {n:02d}: already completed. Use --force to rerun.'); continue
        print('\n'+'='*80+f'\nSTAGE {n:02d}\n'+'='*80); cmd_stage(n,c,a.dry_run)
        if not a.dry_run: state.parent.mkdir(parents=True,exist_ok=True); state.write_text(json.dumps({'stage':n,'completed_at':dt.datetime.now().isoformat(timespec='seconds')},indent=2),encoding='utf-8')
if __name__=='__main__':
    try: main()
    except Exception as e: print(f'\nFAILED: {e}',file=sys.stderr); raise SystemExit(1)
