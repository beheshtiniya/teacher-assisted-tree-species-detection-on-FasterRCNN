
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args(); rows=[]
    for exp in ('A0','A1','A2','A3'):
        f=a.root/exp/'all_runs_mean_std.csv'
        if f.is_file():
            d=pd.read_csv(f); d.insert(0,'experiment',exp); rows.append(d)
    if not rows: raise FileNotFoundError('No Faster R-CNN result summaries found.')
    a.output.parent.mkdir(parents=True,exist_ok=True); pd.concat(rows,ignore_index=True).to_csv(a.output,index=False,encoding='utf-8-sig'); print(a.output)
if __name__=='__main__': main()
