from __future__ import annotations
import argparse, fnmatch, shutil, datetime as dt
from pathlib import Path
PATTERNS=[
'end_to_end_pipeline_config_*.json','end_to_end_pipeline_controller_v3*.py','resume_after_et_run_retina_fusion_csvs.py','run_post_training_predictions_and_csvs.py','run_warmstart_final_only.py','check_dual_teacher_pipeline_paths*.py','check_end_to_end_no_training*','RUN_POST_TRAINING_*','RUN_RESUME_*','RUN_WARMSTART_*','run_end_to_end_pipeline_v3*','README_FA(*','README_end_to_end_controller*','README_epochs45*','dual_teacher_fusion*.py','article2_retina2*','article2_predictions*','fasterrcnn*.py','build_train_ef_ssl_conf_gt_0p5*.py','01_prepare_test_images(*','02_export_predictions(*','03_scale_predictions_640_to_256(*','04_filter_overlap_and_merge(*','05_validate_final_csv(*','06_run_train_val_test_predict_filter_merge(*','*.zip']
EXCLUDE={'Teacher_Assisted_Tree_Detection','EfficientTree-master','legacy_archive'}
def main():
 p=argparse.ArgumentParser(description='Safely move legacy duplicate files into a timestamped archive. Dry-run is the default.'); p.add_argument('--source',type=Path,required=True); p.add_argument('--apply',action='store_true'); a=p.parse_args(); src=a.source.resolve(); dest=src/'legacy_archive'/dt.datetime.now().strftime('%Y%m%d_%H%M%S'); selected=[]
 for item in src.iterdir():
  if item.name in EXCLUDE or item.is_dir(): continue
  if any(fnmatch.fnmatch(item.name,pat) for pat in PATTERNS): selected.append(item)
 print(f'Source: {src}\nArchive: {dest}\nFiles: {len(selected)}')
 for f in selected: print(('MOVE ' if a.apply else 'DRY  ')+f.name)
 if a.apply:
  dest.mkdir(parents=True,exist_ok=True)
  for f in selected: shutil.move(str(f),str(dest/f.name))
if __name__=='__main__': main()
