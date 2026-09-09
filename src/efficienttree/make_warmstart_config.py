
from __future__ import annotations
import argparse, copy
from pathlib import Path
import yaml

def main():
    p=argparse.ArgumentParser(); p.add_argument('--source-yaml',type=Path,required=True); p.add_argument('--dataset',type=Path,required=True); p.add_argument('--weights',type=Path,required=True); p.add_argument('--output-yaml',type=Path,required=True); p.add_argument('--run-name',required=True); p.add_argument('--burn-epochs',type=int,default=0); a=p.parse_args()
    c=yaml.safe_load(a.source_yaml.read_text(encoding='utf-8-sig')); ds=c.setdefault('Dataset',{}); ds.update(train=str((a.dataset/'splits/train.txt').resolve()),val=str((a.dataset/'splits/val.txt').resolve()),test=str((a.dataset/'splits/test.txt').resolve()),target=str((a.dataset/'splits/target_unlabeled.txt').resolve())); c['weights']=str(a.weights.resolve()); c['resume']=False;c['exist_ok']=True;c['name']=a.run_name;c.setdefault('hyp',{})['burn_epochs']=a.burn_epochs; a.output_yaml.parent.mkdir(parents=True,exist_ok=True); a.output_yaml.write_text(yaml.safe_dump(c,sort_keys=False,allow_unicode=True),encoding='utf-8')
if __name__=='__main__': main()
