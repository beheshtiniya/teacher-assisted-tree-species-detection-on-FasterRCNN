
from __future__ import annotations
import argparse, shutil
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.common.process import run


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument('--python', required=True)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--labels-dir', type=Path, required=True)
    p.add_argument('--unlabeled-list', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tag', required=True)
    p.add_argument('--confidence', type=float, default=0.25)
    p.add_argument('--gt-iou', type=float, default=0.50)
    p.add_argument('--device', default='0')
    p.add_argument('--dry-run', action='store_true')
    args=p.parse_args()
    root=Path(__file__).resolve().parents[2]
    prep=root/'src/efficienttree/prepare_image_folder.py'
    prep_un=root/'src/efficienttree/prepare_unlabeled_images.py'
    export=root/'src/efficienttree/export_predictions.py'
    scale=root/'src/efficienttree/scale_predictions_640_to_256.py'
    filt=root/'src/efficienttree/filter_overlap_and_merge.py'
    validate=root/'src/validation/validate_detection_csv.py'
    summarize=root/'src/efficienttree/summarize_unlabeled_predictions.py'
    detect=args.repo/'detect.py'
    if not args.weights.is_file(): raise FileNotFoundError(args.weights)
    args.output.mkdir(parents=True, exist_ok=True)
    sources=args.output/'sources_640'; detector=args.output/'detector_runs'
    for split in ('train','val','test'):
        source=sources/split
        run([args.python,prep,'--test-list',args.dataset/'splits'/f'{split}.txt','--output-dir',source],dry_run=args.dry_run)
        run_name=f'{args.tag}_{split}_conf025'
        run([args.python,detect,'--weights',args.weights,'--source',source,'--img-size','640','--conf-thres',args.confidence,'--device',args.device,'--save-txt','--save-conf','--nosave','--project',detector,'--name',run_name,'--exist-ok'],cwd=args.repo,dry_run=args.dry_run)
        raw640=args.output/f'{split}_predictions_640.csv'; raw256=args.output/f'{split}_predictions_256.csv'
        combined=args.output/f'{split}_gt_plus_predictions.csv'; candidates=args.output/f'{split}_missing_object_candidates.csv'; removed=args.output/f'{split}_removed_gt_overlap.csv'
        run([args.python,export,'--source-dir',source,'--labels-dir',detector/run_name/'labels','--output-dir',args.output/'annotated'/split,'--csv',raw640,'--min-conf',args.confidence],dry_run=args.dry_run)
        run([args.python,scale,'--input',raw640,'--output',raw256],dry_run=args.dry_run)
        run([args.python,filt,'--ground-truth',args.labels_dir/f'{split}_labels.csv','--predictions',raw256,'--combined',combined,'--candidates',candidates,'--removed',removed,'--iou',args.gt_iou,'--split',split],dry_run=args.dry_run)
        run([args.python,validate,'--csv',combined,'--split',split],dry_run=args.dry_run)
    source=sources/'unlabeled'
    run([args.python,prep_un,'--list',args.unlabeled_list,'--images-root',args.dataset,'--output-dir',source],dry_run=args.dry_run)
    run_name=f'{args.tag}_unlabeled_conf025'
    run([args.python,detect,'--weights',args.weights,'--source',source,'--img-size','640','--conf-thres',args.confidence,'--device',args.device,'--save-txt','--save-conf','--nosave','--project',detector,'--name',run_name,'--exist-ok'],cwd=args.repo,dry_run=args.dry_run)
    raw640=args.output/'unlabeled_predictions_640.csv'; raw256=args.output/'unlabeled_predictions_256.csv'
    run([args.python,export,'--source-dir',source,'--labels-dir',detector/run_name/'labels','--output-dir',args.output/'annotated'/'unlabeled','--csv',raw640,'--min-conf',args.confidence],dry_run=args.dry_run)
    run([args.python,scale,'--input',raw640,'--output',raw256],dry_run=args.dry_run)
    run([args.python,summarize,'--images-dir',source,'--predictions',raw256,'--summary',args.output/'unlabeled_image_summary.csv','--no-detection-list',args.output/'unlabeled_without_prediction.txt'],dry_run=args.dry_run)
    return 0
if __name__=='__main__': raise SystemExit(main())
