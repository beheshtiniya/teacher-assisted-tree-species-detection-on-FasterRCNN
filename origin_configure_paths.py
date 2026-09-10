
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description='Create the local path configuration.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--python', dest='python_executable', required=True)
    parser.add_argument('--efficienttree-repo', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--no-create', action='store_true')
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    data_root = args.data_root.expanduser().resolve()
    repo = (args.efficienttree_repo or (project_root / 'EfficientTree-master')).expanduser().resolve()
    output_root = (args.output_root or (data_root / 'tatd_outputs')).expanduser().resolve()
    py_value = args.python_executable.strip()
    if py_value.lower() not in {'python', 'python3', 'py'}:
        py_path = Path(py_value).expanduser().resolve()
        if not py_path.is_file():
            raise FileNotFoundError(f'Python executable not found: {py_path}')
        py_value = str(py_path)

    required = {
        'images': data_root / 'images',
        'labels': data_root / 'labels',
        'train_labels.csv': data_root / 'labels' / 'train_labels.csv',
        'val_labels.csv': data_root / 'labels' / 'val_labels.csv',
        'test_labels.csv': data_root / 'labels' / 'test_labels.csv',
        'unlabeled_images.txt': data_root / 'labels' / 'unlabeled_images.txt',
        'EfficientTree-master': repo,
        'EfficientTree train.py': repo / 'train.py',
        'EfficientTree detect.py': repo / 'detect.py',
    }
    missing = [f'{name}: {path}' for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError('Required project inputs are missing:\n' + '\n'.join(missing))

    if not args.no_create:
        output_root.mkdir(parents=True, exist_ok=True)
        for name in ['datasets','models','predictions','fusion','experiments','reports','stage_state']:
            (output_root/name).mkdir(parents=True, exist_ok=True)

    config = {
        'schema_version': 1,
        'project_root': project_root.as_posix(),
        'data_root': data_root.as_posix(),
        'python_executable': py_value,
        'efficienttree_repo': repo.as_posix(),
        'output_root': output_root.as_posix(),
        'classes': ['class1','class2','class3','class4'],
        'settings': {
            'image_size_et': 640,
            'image_size_csv': 256,
            'confidence': 0.25,
            'gt_iou': 0.50,
            'agreement_iou': 0.50,
            'single_model_confidence': 0.80,
            'warmstart_confidence': 0.50,
            'et_epochs': 45,
            'retina_runs': 3,
            'retina_epochs': 80,
            'fasterrcnn_runs': 5,
            'fasterrcnn_epochs': 20,
        }
    }
    out = project_root/'config'/'paths.local.json'
    out.write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('='*72)
    print('PATH CONFIGURATION CREATED')
    print('='*72)
    print(f'Config:          {out}')
    print(f'Project root:    {project_root}')
    print(f'Data root:       {data_root}')
    print(f'EfficientTree:   {repo}')
    print(f'Output root:     {output_root}')
    print(f'Python:          {py_value}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
