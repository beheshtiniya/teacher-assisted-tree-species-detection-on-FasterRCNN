from __future__ import annotations
import argparse, json, os, subprocess, sys, datetime as dt
from pathlib import Path
from src.common.config import load_config
from src.common.process import run


ROOT = Path(__file__).resolve().parent


def choose_retina_checkpoint(summary:Path, output:Path)->Path:
    import pandas as pd
    d=pd.read_csv(summary)
    metric='best_val_map_50'
    row=d.loc[d[metric].astype(float).idxmax()]
    run_id=int(row['run'])
    ck=output/f'run_{run_id:02d}'/'best_model.pth'
    if not ck.is_file():
        raise FileNotFoundError(ck)
    return ck


def choose_warm_checkpoint(repo:Path)->Path:
    w=repo/'runs'/'mydata_full_unbalanced'/'mydata_dual_teacher_ssl_warmstart_burn0_staged70'/'weights'
    for n in ('best_ema.pt','best.pt'):
        p=w/n
        if p.is_file():
            return p
    raise FileNotFoundError(f'No best_ema.pt or best.pt in {w}')


def cmd_stage(n:int,c,dry:bool):

    py=c.python
    s=c.settings
    labels=c.labels
    out=c.output_root
    repo=c.repo

    dataset=out/'datasets'/'efficienttree_640_full_unbalanced'

    et_cfg=repo/'configs'/'ssod'/'custom'

    sup_yaml=et_cfg/'mydata_full_supervised_4class_staged70.yaml'
    ssl_yaml=et_cfg/'mydata_full_ssl_4class_staged70.yaml'

    sup_run=repo/'runs'/'mydata_full_unbalanced'/'mydata_full_supervised_4class_staged70'
    sup_best=sup_run/'weights'/'best.pt'

    et_sup_pred=out/'predictions'/'efficientteacher_supervised'

    retina_model=out/'models'/'retinanet'
    retina_pred=out/'predictions'/'retinanet'

    fusion=out/'fusion'/'exp4_staged_fusion'

    # اصلاح مهم:
    # Warmstart مستقیم با Dataset3
    warm_dataset=dataset

    warm_yaml=et_cfg/'mydata_full_ssl_warmstart_burn0_staged70.yaml'
    warm_run_name='mydata_dual_teacher_ssl_warmstart_burn0_staged70'

    warm_pred=out/'predictions'/'efficientteacher_warmstart_ssl'

    exps=out/'experiments'
    fr=out/'models'/'fasterrcnn'
        if n==1:
        run(
            [
                py,
                ROOT/'src/pipeline/preflight.py'
            ],
            dry_run=dry
        )


    elif n==2:
        run(
            [
                py,
                ROOT/'src/data/prepare_mydata_640.py',
                '--root',
                c.data_root,
                '--images-dir',
                c.images,
                '--labels-dir',
                labels,
                '--output-dir',
                dataset,
                '--size',
                s['image_size_et'],
                '--overwrite'
            ],
            dry_run=dry
        )


    elif n==3:
        run(
            [
                py,
                ROOT/'src/efficienttree/make_configs.py',
                '--repo',
                repo,
                '--dataset-dir',
                dataset,
                '--batch-size',
                '4',
                '--workers',
                '0',
                '--img-size',
                s['image_size_et'],
                '--epochs',
                s['et_epochs'],
                '--burn-epochs',
                '1',
                '--class-names',
                *c.raw['classes']
            ],
            dry_run=dry
        )


    elif n==4:
        run(
            [
                py,
                repo/'train.py',
                '--cfg',
                et_cfg/'mydata_smoke_ssl_4class.yaml',
                'epochs',
                '70',
                'name',
                'mydata_smoke_ssl_4class',
                'exist_ok',
                'True'
            ],
            cwd=repo,
            env={'EFFICIENTTREE_STOP_EPOCH':'2'},
            dry_run=dry
        )


    elif n==5:
        import yaml

        src=et_cfg/'mydata_smoke_ssl_4class.yaml'
        dst=et_cfg/'mydata_smoke_ssl_4class_burn10.yaml'

        if not dry:
            d=yaml.safe_load(
                src.read_text(encoding='utf-8-sig')
            )

            d.setdefault('hyp',{})['burn_epochs']=10
            d['name']='mydata_smoke_ssl_4class_burn10'
            d['resume']=False
            d['exist_ok']=True

            dst.write_text(
                yaml.safe_dump(
                    d,
                    sort_keys=False,
                    allow_unicode=True
                ),
                encoding='utf-8'
            )

        run(
            [
                py,
                repo/'train.py',
                '--cfg',
                dst,
                'epochs',
                '70',
                'name',
                'mydata_smoke_ssl_4class_burn10',
                'exist_ok',
                'True'
            ],
            cwd=repo,
            env={'EFFICIENTTREE_STOP_EPOCH':'12'},
            dry_run=dry
        )


    elif n==6:
        run(
            [
                py,
                repo/'train.py',
                '--cfg',
                sup_yaml,
                'epochs',
                s['et_epochs'],
                'name',
                'mydata_full_supervised_4class_staged70',
                'exist_ok',
                'True'
            ],
            cwd=repo,
            env={
                'EFFICIENTTREE_STOP_EPOCH':
                str(s['et_epochs'])
            },
            dry_run=dry
        )


    elif n==7:
        run(
            [
                py,
                ROOT/'src/experiments/build_A0_dataset.py',
                '--train',
                labels/'train_labels.csv',
                '--val',
                labels/'val_labels.csv',
                '--test',
                labels/'test_labels.csv',
                '--output-dir',
                exps/'A0'
            ],
            dry_run=dry
        )


    elif n==8:
        run(
            [
                py,
                ROOT/'src/efficienttree/predict_all_splits.py',
                '--python',
                py,
                '--repo',
                repo,
                '--dataset',
                dataset,
                '--labels-dir',
                labels,
                '--unlabeled-list',
                labels/'unlabeled_images.txt',
                '--weights',
                sup_best,
                '--output',
                et_sup_pred,
                '--tag',
                'supervised',
                '--confidence',
                s['confidence'],
                '--gt-iou',
                s['gt_iou']
            ],
            dry_run=dry
        )
            elif n==9:
        run(
            [
                py,
                ROOT/'src/retinanet/train_retinanet.py',
                '--images-dir',
                c.images,
                '--train-csv',
                labels/'train_labels.csv',
                '--val-csv',
                labels/'val_labels.csv',
                '--test-csv',
                labels/'test_labels.csv',
                '--output-dir',
                retina_model,
                '--num-runs',
                s['retina_runs'],
                '--max-epochs',
                s['retina_epochs']
            ],
            dry_run=dry
        )


    elif n==10:

        ck = (
            retina_model/'run_01'/'best_model.pth'
            if dry
            else choose_retina_checkpoint(
                retina_model/'all_runs_summary.csv',
                retina_model
            )
        )

        run(
            [
                py,
                ROOT/'src/retinanet/predict_retinanet.py',
                '--checkpoint',
                ck,
                '--images-dir',
                c.images,
                '--train-csv',
                labels/'train_labels.csv',
                '--val-csv',
                labels/'val_labels.csv',
                '--test-csv',
                labels/'test_labels.csv',
                '--unlabeled-list',
                labels/'unlabeled_images.txt',
                '--output-root',
                retina_pred,
                '--confidence',
                s['confidence'],
                '--nms-iou',
                s['gt_iou'],
                '--gt-iou',
                s['gt_iou']
            ],
            dry_run=dry
        )


    # =====================================================
    # STAGE 11
    # NEW EXP4 STAGED FUSION
    #
    # RetinaNet + ET supervised
    #       |
    #       GT IoU filtering
    #       |
    #       NMS
    #
    #       +
    #
    # ET SSL warmstart
    #       |
    #       GT IoU filtering
    #       |
    #       NMS
    #
    #       +
    #
    # GT
    #
    # output:
    # train_final_fused.csv
    # =====================================================

    elif n==11:

        run(
            [
                py,
                ROOT/'src/fusion/staged_fusion_exp4.py',

                '--gt-train',
                labels/'train_labels.csv',

                '--retina-train',
                retina_pred/'predictions'/
                'predictions_train_after_confidence_filter.csv',

                '--et-supervised-train',
                et_sup_pred/'train_predictions_256.csv',

                '--et-ssl-train',
                warm_pred/'train_predictions_256.csv',

                '--output-dir',
                exps/'exp4_D3_best_train_fusion',

                '--confidence',
                s['confidence'],

                '--gt-iou',
                s['gt_iou'],

                '--nms-iou',
                s['gt_iou']

            ],
            dry_run=dry
        )


    # =====================================================
    # STAGE 12
    # WARMSTART SSL DIRECTLY ON DATASET3
    # NO fusion labels
    # =====================================================

    elif n==12:

        run(
            [
                py,
                ROOT/'src/efficienttree/make_warmstart_config.py',

                '--source-yaml',
                ssl_yaml,

                '--dataset',
                dataset,

                '--weights',
                sup_best,

                '--output-yaml',
                warm_yaml,

                '--run-name',
                warm_run_name,

                '--burn-epochs',
                '0'

            ],
            dry_run=dry
        )


    elif n==13:

        run(
            [
                py,
                repo/'train.py',

                '--cfg',
                warm_yaml,

                'epochs',
                s['et_epochs'],

                'name',
                warm_run_name,

                'exist_ok',
                'True'

            ],
            cwd=repo,

            env={
                'EFFICIENTTREE_STOP_EPOCH':
                str(s['et_epochs'])
            },

            dry_run=dry
        )
            elif n==9:
        run(
            [
                py,
                ROOT/'src/retinanet/train_retinanet.py',
                '--images-dir',
                c.images,
                '--train-csv',
                labels/'train_labels.csv',
                '--val-csv',
                labels/'val_labels.csv',
                '--test-csv',
                labels/'test_labels.csv',
                '--output-dir',
                retina_model,
                '--num-runs',
                s['retina_runs'],
                '--max-epochs',
                s['retina_epochs']
            ],
            dry_run=dry
        )


    elif n==10:

        ck = (
            retina_model/'run_01'/'best_model.pth'
            if dry
            else choose_retina_checkpoint(
                retina_model/'all_runs_summary.csv',
                retina_model
            )
        )

        run(
            [
                py,
                ROOT/'src/retinanet/predict_retinanet.py',
                '--checkpoint',
                ck,
                '--images-dir',
                c.images,
                '--train-csv',
                labels/'train_labels.csv',
                '--val-csv',
                labels/'val_labels.csv',
                '--test-csv',
                labels/'test_labels.csv',
                '--unlabeled-list',
                labels/'unlabeled_images.txt',
                '--output-root',
                retina_pred,
                '--confidence',
                s['confidence'],
                '--nms-iou',
                s['gt_iou'],
                '--gt-iou',
                s['gt_iou']
            ],
            dry_run=dry
        )


    # =====================================================
    # STAGE 11
    # NEW EXP4 STAGED FUSION
    #
    # RetinaNet + ET supervised
    #       |
    #       GT IoU filtering
    #       |
    #       NMS
    #
    #       +
    #
    # ET SSL warmstart
    #       |
    #       GT IoU filtering
    #       |
    #       NMS
    #
    #       +
    #
    # GT
    #
    # output:
    # train_final_fused.csv
    # =====================================================

    elif n==11:

        run(
            [
                py,
                ROOT/'src/fusion/staged_fusion_exp4.py',

                '--gt-train',
                labels/'train_labels.csv',

                '--retina-train',
                retina_pred/'predictions'/
                'predictions_train_after_confidence_filter.csv',

                '--et-supervised-train',
                et_sup_pred/'train_predictions_256.csv',

                '--et-ssl-train',
                warm_pred/'train_predictions_256.csv',

                '--output-dir',
                exps/'exp4_D3_best_train_fusion',

                '--confidence',
                s['confidence'],

                '--gt-iou',
                s['gt_iou'],

                '--nms-iou',
                s['gt_iou']

            ],
            dry_run=dry
        )


    # =====================================================
    # STAGE 12
    # WARMSTART SSL DIRECTLY ON DATASET3
    # NO fusion labels
    # =====================================================

    elif n==12:

        run(
            [
                py,
                ROOT/'src/efficienttree/make_warmstart_config.py',

                '--source-yaml',
                ssl_yaml,

                '--dataset',
                dataset,

                '--weights',
                sup_best,

                '--output-yaml',
                warm_yaml,

                '--run-name',
                warm_run_name,

                '--burn-epochs',
                '0'

            ],
            dry_run=dry
        )


    elif n==13:

        run(
            [
                py,
                repo/'train.py',

                '--cfg',
                warm_yaml,

                'epochs',
                s['et_epochs'],

                'name',
                warm_run_name,

                'exist_ok',
                'True'

            ],
            cwd=repo,

            env={
                'EFFICIENTTREE_STOP_EPOCH':
                str(s['et_epochs'])
            },

            dry_run=dry
        )
        