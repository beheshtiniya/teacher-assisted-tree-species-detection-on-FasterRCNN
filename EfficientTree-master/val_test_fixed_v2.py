# EfficientTeacher
"""
Validate a trained YOLOv5 detection model on a detection dataset

Usage:
    $ python val.py --weights yolov5s.pt --data coco128.yaml --img 640

Usage - formats:
    $ python val.py --weights yolov5s.pt                 # PyTorch
                              yolov5s.torchscript        # TorchScript
                              yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                              yolov5s_openvino_model     # OpenVINO
                              yolov5s.engine             # TensorRT
                              yolov5s.mlmodel            # CoreML (macOS-only)
                              yolov5s_saved_model        # TensorFlow SavedModel
                              yolov5s.pb                 # TensorFlow GraphDef
                              yolov5s.tflite             # TensorFlow Lite
                              yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                              yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from tqdm import tqdm



FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
#ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# from models.backbone.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.metrics import ap_per_class, ConfusionMatrix, oks_iou,box_iou
from utils.general import coco80_to_coco91_class, check_dataset, check_img_size, check_requirements, \
    check_suffix, check_yaml, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, \
    increment_path, colorstr, print_args, non_max_suppression_lmk_and_bbox, scale_coords_landmarks
from utils.plots import plot_images
from utils.torch_utils import select_device, time_sync
from utils.callbacks import Callbacks
from utils.profile import profile
from utils.torch_utils import is_parallel
import copy
from configs.defaults import get_cfg

from utils.metrics import NMEMeter
from utils.detect_multi_backend import DetectMultiBackend


def resolve_weights_paths(weights):
    """
    Resolve the actual EfficientTree best.pt checkpoint.

    The training output directory can differ depending on whether the run name
    was appended once or twice. When --weights is omitted or the supplied path
    no longer exists, search the local runs directory and select the newest
    best.pt belonging to EfficientTree_tree_ssl.
    """
    supplied = []
    if weights:
        supplied = [Path(item).expanduser() for item in weights]

    existing = [path.resolve() for path in supplied if path.is_file()]
    missing = [path for path in supplied if not path.is_file()]

    if existing:
        if missing:
            print("WARNING: Some supplied checkpoint paths do not exist:")
            for path in missing:
                print(f"  - {path}")
        return [str(path) for path in existing]

    search_roots = [
        ROOT / "runs" / "EfficientTree_tree_ssl",
        ROOT / "runs",
    ]

    discovered = []
    seen = set()
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for candidate in search_root.rglob("best.pt"):
            try:
                resolved = candidate.resolve()
            except OSError:
                resolved = candidate
            key = str(resolved).lower()
            if key not in seen and candidate.is_file():
                seen.add(key)
                discovered.append(resolved)

    if not discovered:
        supplied_text = "\n".join(f"  - {path}" for path in supplied) or "  - none"
        raise FileNotFoundError(
            "No existing best.pt checkpoint was found.\n"
            "Supplied paths:\n"
            f"{supplied_text}\n\n"
            "Search location:\n"
            f"  {ROOT / 'runs'}\n\n"
            "Use PowerShell to locate it:\n"
            "  Get-ChildItem -Path .\\runs -Filter best.pt -Recurse -File | "
            "Select-Object FullName, LastWriteTime"
        )

    preferred = [
        path for path in discovered
        if "efficienttree_tree_ssl" in str(path).lower()
        and path.parent.name.lower() == "weights"
    ]
    candidates = preferred or discovered
    candidates.sort(
        key=lambda path: path.stat().st_mtime if path.exists() else 0,
        reverse=True,
    )

    selected = candidates[0]

    print("\nCheckpoint auto-discovery:")
    if supplied:
        print("Configured checkpoint was not found:")
        for path in supplied:
            print(f"  - {path}")

    print(f"Selected checkpoint: {selected}")

    if len(candidates) > 1:
        print("Other best.pt candidates:")
        for path in candidates[1:10]:
            print(f"  - {path}")
        print(
            "The newest EfficientTree_tree_ssl candidate was selected. "
            "Use --weights with an explicit path if another checkpoint is intended."
        )

    return [str(selected)]


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})



def process_batch_oks(detections, labels, iouv, num_points):
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    correct_class = labels[:, 0:1] == detections[:, 5]
    ious = oks_iou(labels, detections, num_points)
    ious = torch.from_numpy(ious).to(iouv.device)

    for i in range(len(iouv)):
        x = torch.where((ious >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), ious[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct

def process_batch_old(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    # print('process batch:', detections.shape)
    # iou = box_iou(labels[:, 1:], xywh2xyxy(poly2hbb(detections[:, -9:-1])))
    x = torch.where((iou >= iouv[0]) & (labels[:, 0:1] == detections[:, 5]))  # IoU above threshold and classes match
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct

def process_batch(detections, labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        detections (Array[N, 6]), x1, y1, x2, y2, conf, class
        labels (Array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def describe_output_structure(value, depth=0):
    """Return a compact description of nested model outputs for diagnostics."""
    indent = "  " * depth
    if torch.is_tensor(value):
        return f"{indent}Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device})"
    if isinstance(value, dict):
        lines = [f"{indent}dict"]
        for key, nested in value.items():
            lines.append(f"{indent}  [{key!r}]")
            lines.append(describe_output_structure(nested, depth + 2))
        return "\n".join(lines)
    if isinstance(value, (list, tuple)):
        lines = [f"{indent}{type(value).__name__}(len={len(value)})"]
        for index, nested in enumerate(value):
            lines.append(f"{indent}  [{index}]")
            lines.append(describe_output_structure(nested, depth + 2))
        return "\n".join(lines)
    return f"{indent}{type(value).__name__}: {value!r}"


def extract_inference_tensor(outputs, batch_size, nc):
    """
    Extract the decoded detection tensor expected by EfficientTree NMS.

    Expected shape for this 4-class dataset is [batch, predictions, 5 + nc].
    The repository's standalone backend can return nested lists/tuples, while
    the original val.py unwraps only one level.
    """
    exact = []
    fallback = []

    def walk(value, path="output"):
        if torch.is_tensor(value):
            if value.ndim == 3 and int(value.shape[0]) == int(batch_size):
                item = (path, value)
                if int(value.shape[-1]) == int(nc + 5):
                    exact.append(item)
                elif int(value.shape[-1]) >= 6:
                    fallback.append(item)
            return
        if isinstance(value, dict):
            for key, nested in value.items():
                walk(nested, f"{path}[{key!r}]")
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                walk(nested, f"{path}[{index}]")

    walk(outputs)
    candidates = exact or fallback
    if not candidates:
        raise RuntimeError(
            "No NMS-compatible 3-D prediction tensor was found.\n"
            + describe_output_structure(outputs)
        )

    # The decoded output normally has the largest number of candidate boxes.
    candidates.sort(key=lambda item: (int(item[1].shape[1]), int(item[1].shape[2])), reverse=True)
    selected_path, selected = candidates[0]

    if int(selected.shape[-1]) != int(nc + 5):
        raise RuntimeError(
            f"Selected tensor {selected_path} has shape {tuple(selected.shape)}, "
            f"but NMS expects last dimension {nc + 5} for nc={nc}.\n"
            + describe_output_structure(outputs)
        )

    return selected, selected_path


def process_confusion_matrix_safe(confusion_matrix, detections, labels, nc):
    """Update the repository ConfusionMatrix, including empty-prediction cases."""
    if labels is None:
        labels = torch.zeros((0, 5), device=detections.device if detections is not None else 'cpu')

    if detections is None or len(detections) == 0:
        try:
            confusion_matrix.process_batch(None, labels)
        except Exception:
            # Compatibility fallback for older repository variants.
            for gt_class in labels[:, 0].int().tolist():
                confusion_matrix.matrix[nc, gt_class] += 1
        return

    if len(labels) == 0:
        try:
            confusion_matrix.process_batch(detections, labels)
        except Exception:
            for pred_class in detections[:, 5].int().tolist():
                confusion_matrix.matrix[pred_class, nc] += 1
        return

    confusion_matrix.process_batch(detections, labels)


def scalar_at(value, index, default=0.0):
    """Safely read scalar/list/ndarray metric values."""
    try:
        array = np.asarray(value)
        if array.ndim == 0:
            return float(array.item())
        return float(array[index])
    except Exception:
        return float(default)


def save_final_test_reports(
    save_dir,
    weights,
    split_path,
    seen,
    nt,
    names,
    nc,
    mp,
    mr,
    map50,
    map5095,
    p,
    r,
    f1,
    ap50,
    ap5095_per_class,
    ap_class,
    cls_thr,
    confusion_matrix,
    conf_thres,
    iou_thres,
    cm_conf_thres,
    cm_iou_thres,
    speed_ms,
):
    """Save final TEST metrics and raw/normalized confusion-matrix reports."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    name_map = names if isinstance(names, dict) else {i: str(v) for i, v in enumerate(names)}
    ap_index = {int(class_id): i for i, class_id in enumerate(np.asarray(ap_class).tolist())}
    nt_array = np.asarray(nt, dtype=float).reshape(-1)

    rows = []
    for class_id in range(nc):
        metric_index = ap_index.get(class_id)
        row = {
            'class_id': class_id,
            'class_name': str(name_map.get(class_id, f'class{class_id}')),
            'labels': int(nt_array[class_id]) if class_id < len(nt_array) else 0,
            'precision': scalar_at(p, metric_index) if metric_index is not None else 0.0,
            'recall': scalar_at(r, metric_index) if metric_index is not None else 0.0,
            'f1': scalar_at(f1, metric_index) if metric_index is not None else 0.0,
            'ap50': scalar_at(ap50, metric_index) if metric_index is not None else 0.0,
            'ap50_95': scalar_at(ap5095_per_class, metric_index) if metric_index is not None else 0.0,
            'best_conf_threshold': scalar_at(cls_thr, metric_index) if metric_index is not None else 0.0,
        }
        rows.append(row)

    per_class_path = save_dir / 'test_metrics_per_class.csv'
    with per_class_path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    matrix_official = np.asarray(confusion_matrix.matrix, dtype=np.int64)
    # EfficientTree/YOLO stores rows=Predicted and columns=True. The report requested
    # by the user is also saved transposed as rows=Ground Truth and columns=Predicted.
    matrix_gt_pred = matrix_official.T
    labels = [str(name_map.get(i, f'class{i}')) for i in range(nc)] + ['background']

    def write_matrix_csv(path, matrix):
        with Path(path).open('w', newline='', encoding='utf-8-sig') as handle:
            writer = csv.writer(handle)
            writer.writerow(['Ground Truth / Predicted', *labels])
            for label, row in zip(labels, matrix):
                writer.writerow([label, *[float(x) if isinstance(x, np.floating) else int(x) for x in row]])

    write_matrix_csv(save_dir / 'confusion_matrix_counts_gt_rows_pred_cols.csv', matrix_gt_pred)
    write_matrix_csv(save_dir / 'confusion_matrix_counts_official_pred_rows_true_cols.csv', matrix_official)

    row_sums = matrix_gt_pred.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix_gt_pred.astype(np.float64),
        row_sums,
        out=np.zeros_like(matrix_gt_pred, dtype=np.float64),
        where=row_sums != 0,
    )
    with (save_dir / 'confusion_matrix_normalized_gt_rows_pred_cols.csv').open(
        'w', newline='', encoding='utf-8-sig'
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(['Ground Truth / Predicted', *labels])
        for label, row in zip(labels, normalized):
            writer.writerow([label, *[f'{float(x):.8f}' for x in row]])

    summary = {
        'evaluation_split': 'test',
        'test_path': str(split_path),
        'checkpoint': str(weights[0] if isinstance(weights, list) and len(weights) == 1 else weights),
        'images': int(seen),
        'labels': int(np.asarray(nt).sum()),
        'num_classes': int(nc),
        'class_names': [str(name_map.get(i, f'class{i}')) for i in range(nc)],
        'precision_mean': float(mp),
        'recall_mean': float(mr),
        'map50': float(map50),
        'map50_95': float(map5095),
        'ap_iou_thresholds': [round(float(x), 2) for x in np.linspace(0.50, 0.95, 10)],
        'nms_conf_threshold': float(conf_thres),
        'nms_iou_threshold': float(iou_thres),
        'confusion_matrix_conf_threshold': float(cm_conf_thres),
        'confusion_matrix_iou_threshold': float(cm_iou_thres),
        'confusion_matrix_orientation_requested_csv': 'rows=Ground Truth, columns=Predicted',
        'confusion_matrix_orientation_repository': 'rows=Predicted, columns=True',
        'speed_ms_per_image': {
            'preprocess': float(speed_ms[0]),
            'inference': float(speed_ms[1]),
            'nms': float(speed_ms[2]),
        },
        'per_class_csv': str(per_class_path),
    }
    with (save_dir / 'test_metrics_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    with (save_dir / 'TEST_SPLIT_CONFIRMATION.txt').open('w', encoding='utf-8') as handle:
        handle.write('FINAL EVALUATION SPLIT: TEST\n')
        handle.write(f'Test list: {split_path}\n')
        handle.write(f'Images evaluated: {seen}\n')
        handle.write(f'Labels evaluated: {int(np.asarray(nt).sum())}\n')
        handle.write('Validation data was not used for these final metrics.\n')

    return summary, rows


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # all, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project=ROOT / 'runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        callbacks=Callbacks(),
        compute_loss=None,
        model_post=None,
        eval_num = -1,
        cfg = None,
        val_ssod = False,
        num_points = 0,
        val_kp = False,
        val_dp1000 = False,
        dnn=False,  # use OpenCV DNN for ONNX inference
        names = {},
        test_path=None,
        cm_conf_thres=0.25,
        cm_iou_thres=0.45
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by all.py
        device = next(model.parameters()).device  # get model device
        pt, jit, engine = True, False, False
        half &= device.type != 'cpu'  # half precision only supported on CUDA
        model.half() if half else model.float()
    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        # Resolve the real best.pt first because EfficientTree run folders may
        # contain the experiment name once or twice.
        weights = resolve_weights_paths(weights)
        print(f"Using checkpoint for FINAL TEST evaluation: {weights[0]}")
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, pt, jit, engine, is_magicmind = model.stride, model.pt, model.jit, model.engine, model.magicmind
        gs = max(int(model.stride), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        half = model.fp16  # FP16 supported on limited backends with CUDA
        if engine:
            batch_size = model.batch_size
        else:
            device = model.device
            if not (pt or jit):
                batch_size = 1  # export.py models default to batch-size 1

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
        if cfg not in (None, ''):
            val_cfg = get_cfg()
            val_cfg.merge_from_file(cfg)
            data = {}
            data['val'] = str(val_cfg.Dataset.val)
            cfg_test = None
            try:
                cfg_test = str(val_cfg.Dataset.test)
            except Exception:
                cfg_test = None
            data['test'] = str(test_path) if test_path else cfg_test
            data['nc'] = int(val_cfg.Dataset.nc)
            data['names'] = list(val_cfg.Dataset.names)
            val_kp = bool(val_cfg.Dataset.val_kp)
        else:
            data = check_dataset(data)  # check
            val_cfg = None
            if test_path:
                data['test'] = str(test_path)

        if task == 'test' and not data.get('test'):
            raise KeyError(
                "No TEST path was resolved. Pass --test-path or define Dataset.test in the config."
            )

        # Data
        if val_ssod:
            pass
        elif pt: #only pt we print FLOPs and PARAMS
            model_profile = copy.deepcopy(model)
            flops, params =  profile(model_profile.module if is_parallel(model_profile) else model_profile, (torch.ones((1, 3, imgsz, imgsz)).to(device),1), clever=True)
            print("Flops {} Params {}".format(flops, params))
        else:
            pass


    # Configure
    model.eval()
    selected_split_path = data.get(task) or data.get('val')
    is_coco = isinstance(selected_split_path, str) and selected_split_path.endswith('val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        # if device.type != 'cpu':
        #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        model.warmup(imgsz=(1 if pt else batch_size, 3, imgsz, imgsz))  # warmup
        pad = 0.0 if task == 'speed' else 0.5
        task = task if task in ('all', 'val', 'test') else 'test'
        if task != 'test':
            raise ValueError("This patched script is reserved for final TEST evaluation. Use --task test.")
        selected_split_path = str(data['test'])
        print('=' * 80)
        print('FINAL EVALUATION SPLIT: TEST')
        print(f'Test list: {selected_split_path}')
        print('=' * 80)
        dataloader = create_dataloader(selected_split_path, imgsz, batch_size, gs, single_cls, pad=pad, rect=True, cfg=val_cfg,
                                       prefix=colorstr('test: '))[0]

    seen = 0
    try:
        confusion_matrix = ConfusionMatrix(nc=nc, conf=cm_conf_thres, iou_thres=cm_iou_thres)
    except TypeError:
        confusion_matrix = ConfusionMatrix(nc=nc)
    try:
        names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    except:
        names = names
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    dt, p, r, f1, mp, mr, map50, map = [0.0, 0.0, 0.0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []

    ##model_postprocess
    # from models.head.yolov5_head import Detect
    # model_post = Detect(cfg)
    # model_post.training = False
    # model_post.eval()

    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        # 默认-1 则全部测试
        if batch_i==eval_num:
            break
        t1 = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if pt:
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
        elif is_magicmind: #for cambricon no need to divide 255.0 
            pass
        else:
            img /= 255.0   
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t2 = time_sync()
        dt[0] += t2 - t1

        # Run model
        if val_ssod:
            outputs, sup_feats = model(img, augment=augment) 
        else:
            outputs = model(img, augment=augment)  # inference and training outputs
        if model_post is not None:
            model_post.cur_imgsize = img.shape[2:]
            if is_magicmind:
                outputs = model_post.post_process_v2(outputs)
            else:
                outputs = model_post.post_process(outputs)
        dt[1] += time_sync() - t2

        # Robustly unwrap EfficientTree's nested standalone output.
        out, output_path = extract_inference_tensor(outputs, batch_size=nb, nc=nc)
        if batch_i == 0:
            print(f"NMS prediction tensor: {output_path}, shape={tuple(out.shape)}")
        # Run NMS
        # if num_points == 4:
            # targets[:, 2:] *= torch.Tensor([width, height, width, height, width, height, width, height, width, height, width, height]).to(device)  # to pixels
        # if num_points == 8:
        if val_kp:
            targets[:, 2:2+2*(2 + num_points)] *= torch.Tensor([width, height] * (2 + num_points)).to(device)  # to pixels
        else:
            targets[:, 2:6] *= torch.Tensor([width, height] * 2).to(device)  # to pixels
        # targets[:, 2:] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:6] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t3 = time_sync()
        if num_points > 0:
            out = non_max_suppression_lmk_and_bbox(out, conf_thres, iou_thres, labels=lb, agnostic=single_cls, num_points=num_points)
        else:
            out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        dt[2] += time_sync() - t3
        # print(time_sync() - t3)

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            # Convert labels to native image coordinates once, including images with no predictions.
            if nl:
                if num_points > 0 and val_kp:
                    labelsn = labels
                else:
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)
            else:
                labelsn = torch.zeros((0, 5), device=targets.device)

            if len(pred) == 0:
                if plots:
                    process_confusion_matrix_safe(confusion_matrix, None, labelsn, nc)
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred
            if num_points > 0:
                scale_coords_landmarks(img[si].shape[1:], predn[:, -1 - num_points * 2:-1], shape, num_points, shapes[si][1])

            # Evaluate
            if nl:
                if num_points > 0 and val_kp:
                    scale_coords_landmarks(img[si].shape[1:], labels[:, 5:5+num_points * 2], shape, num_points, shapes[si][1])
                    correct = process_batch_oks(predn, labels, iouv, num_points)
                else:
                    correct = process_batch(predn, labelsn, iouv)
            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=pred.device)

            if plots:
                process_confusion_matrix_safe(confusion_matrix, predn, labelsn, nc)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            callbacks.run('on_val_image_end', pred, predn, path, names, img[si])


        # Plot images
        if plots and batch_i < 3:
            f = save_dir / f'test_batch{batch_i}_labels.jpg'  # labels
            if val_kp:
                Thread(target=plot_images, args=(img, targets, paths, f, num_points, names), daemon=True).start()
            else:
                Thread(target=plot_images, args=(img, targets, paths, f, 0, names), daemon=True).start()
            # Thread(target=plot_images_keypoints, args=(img, targets, paths, f, names), daemon=True).start()
            # f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
            # Thread(target=plot_images, args=(img, output_to_target_keypoints(out), paths, f, names), daemon=True).start()
            # Thread(target=plot_images, args=(img, out, paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class, cls_thr = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        ap50 = ap[:, 0]
        ap5095_per_class = ap.mean(1)
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap5095_per_class.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        p = np.zeros(0, dtype=float)
        r = np.zeros(0, dtype=float)
        f1 = np.zeros(0, dtype=float)
        ap50 = np.zeros(0, dtype=float)
        ap5095_per_class = np.zeros(0, dtype=float)
        ap_class = np.zeros(0, dtype=int)
        cls_thr = np.zeros(0, dtype=float)
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc) if len(stats) else np.zeros(nc, dtype=int)
    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap5095_per_class[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        callbacks.run('on_val_end')

    # Final TEST reports (all values below are computed on the TEST split only).
    summary, per_class_rows = save_final_test_reports(
        save_dir=save_dir,
        weights=weights,
        split_path=selected_split_path,
        seen=seen,
        nt=nt,
        names=names,
        nc=nc,
        mp=mp,
        mr=mr,
        map50=map50,
        map5095=map,
        p=p,
        r=r,
        f1=f1,
        ap50=ap50,
        ap5095_per_class=ap5095_per_class,
        ap_class=ap_class,
        cls_thr=cls_thr,
        confusion_matrix=confusion_matrix,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        cm_conf_thres=cm_conf_thres,
        cm_iou_thres=cm_iou_thres,
        speed_ms=t,
    )
    print('\nFinal TEST summary:')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        # anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        anno_json = str('/mnt/bowen/exp/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {colorstr('bold', save_dir)}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap5095_per_class[i]
    if val_ssod:
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t, cls_thr
    else:
        return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser(
        description='Final EfficientTree evaluation on the held-out TEST split.'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='',
        help='Optional detection dataset YAML. The explicit --test-path takes precedence.',
    )
    parser.add_argument(
        '--cfg',
        type=str,
        default=r'configs\ssod\custom\yolov8s_tree_ssod_masoumeh.yaml',
        help='Training config used to reconstruct dataset/model settings.',
    )
    parser.add_argument(
        '--test-path',
        type=str,
        default=r'E:\FASTRCNN\teacher_student\efficienttree_dataset\test.txt',
        help='Held-out TEST image-list file. All final metrics use this split.',
    )
    parser.add_argument(
        '--weights',
        nargs='+',
        type=str,
        default=None,
        help=(
            'Optional explicit best.pt path. When omitted, the newest '
            'EfficientTree_tree_ssl best.pt under ./runs is discovered automatically.'
        ),
    )
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.001, help='NMS confidence used for AP/mAP evaluation')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--cm-conf-thres', type=float, default=0.25, help='Repository confusion-matrix confidence threshold')
    parser.add_argument('--cm-iou-thres', type=float, default=0.45, help='Repository confusion-matrix matching IoU threshold')
    parser.add_argument('--task', choices=['test'], default='test')
    parser.add_argument('--device', default='0')
    parser.add_argument('--single-cls', action='store_true')
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--save-txt', action='store_true')
    parser.add_argument('--save-hybrid', action='store_true')
    parser.add_argument('--save-conf', action='store_true')
    parser.add_argument('--save-json', action='store_true')
    parser.add_argument(
        '--project',
        default=r'E:\FASTRCNN\FASTRCNN\EfficientTree-master\runs\EfficientTree_tree_ssl',
    )
    parser.add_argument('--name', default='best_model_final_test')
    parser.add_argument('--exist-ok', action='store_true', default=True)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--val-ssod', action='store_true')
    parser.add_argument('--num-points', type=int, default=0)
    parser.add_argument('--val-dp1000', action='store_true')
    parser.add_argument('--dnn', action='store_true')
    opt = parser.parse_args()
    opt.save_txt |= opt.save_hybrid
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    set_logging()

    requirements_file = ROOT / 'requirements.txt'
    if requirements_file.is_file():
        check_requirements(exclude=('tensorboard', 'thop'))
    else:
        print(
            f"Note: {requirements_file} was not found; "
            "dependency re-check was skipped."
        )

    if opt.task in ('all', 'val', 'test'):  # run normally
        run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
