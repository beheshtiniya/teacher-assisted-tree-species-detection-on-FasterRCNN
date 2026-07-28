import argparse
import os
import sys
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


# =========================================================
# 0) Global constants / config
# =========================================================
CLASS_IDS = [1, 2, 3, 4]
CLASS_NAMES = ["background", "class1", "class2", "class3", "class4"]

CONFIG = {
    "num_runs": 3,
    "max_epochs": 80,
    "patience": 15,
    "batch_size": 4,   # If GPU memory allows, you can increase this to 8.
    "base_seed": 42,
    "lr": 0.0025,
    "weight_decay": 5e-4,
    "iou_threshold": 0.5,
    "score_threshold_start": 0.05,
    "threshold_candidates": [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
}


# =========================================================
# 1) Reproducibility
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================================================
# 2) Dataset
# =========================================================
class TreeDetectionDataset(Dataset):
    def __init__(self, img_dir, csv_file, transforms=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file).copy()
        self.transforms = transforms

        required_cols = ["filename", "class", "xmin", "ymin", "xmax", "ymax"]
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Column '{col}' not found in {csv_file}")

        self.df["filename"] = self.df["filename"].astype(str)
        self.df["class"] = self.df["class"].astype(int)

        for col in ["xmin", "ymin", "xmax", "ymax"]:
            self.df[col] = pd.to_numeric(self.df[col], errors="raise")

        valid_classes = {1, 2, 3, 4}
        found_classes = set(self.df["class"].unique().tolist())
        if not found_classes.issubset(valid_classes):
            raise ValueError(
                f"Invalid classes found in {csv_file}: {found_classes - valid_classes}"
            )

        self.image_names = sorted(self.df["filename"].drop_duplicates().tolist())
        self.grouped = self.df.groupby("filename")

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        filename = self.image_names[idx]
        img_path = os.path.join(self.img_dir, filename)

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {e}")

        w, h = img.size
        rows = self.grouped.get_group(filename).copy()

        boxes_np = rows[["xmin", "ymin", "xmax", "ymax"]].values.astype(np.float32)

        boxes_np[:, 0] = np.clip(boxes_np[:, 0], 0, w)
        boxes_np[:, 2] = np.clip(boxes_np[:, 2], 0, w)
        boxes_np[:, 1] = np.clip(boxes_np[:, 1], 0, h)
        boxes_np[:, 3] = np.clip(boxes_np[:, 3], 0, h)

        valid = (boxes_np[:, 2] > boxes_np[:, 0]) & (boxes_np[:, 3] > boxes_np[:, 1])
        boxes_np = boxes_np[valid]
        labels_np = rows["class"].values.astype(np.int64)[valid]

        if len(boxes_np) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes_np, dtype=torch.float32)
            labels = torch.tensor(labels_np, dtype=torch.int64)
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img = self.transforms(img)
        else:
            img = F.to_tensor(img)

        return img, target


# =========================================================
# 3) Utilities & Diagnostic dataset check
# =========================================================
def collate_fn(batch):
    return tuple(zip(*batch))


def validate_splits(train_csv, val_csv, test_csv):
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_names = set(train_df["filename"].astype(str).unique())
    val_names = set(val_df["filename"].astype(str).unique())
    test_names = set(test_df["filename"].astype(str).unique())

    tv = train_names & val_names
    tt = train_names & test_names
    vt = val_names & test_names

    print(f"train ∩ val = {len(tv)}")
    print(f"train ∩ test = {len(tt)}")
    print(f"val ∩ test = {len(vt)}")

    assert len(tv) == 0, "Leakage found between train and val"
    assert len(tt) == 0, "Leakage found between train and test"
    assert len(vt) == 0, "Leakage found between val and test"


def inspect_dataset(dataset, split_name):
    labels = dataset.df["class"].value_counts().sort_index()
    label_pct = dataset.df["class"].value_counts(normalize=True).sort_index() * 100.0

    invalid_count = 0
    clipped_count = 0
    total_boxes = 0
    edge_touch_count = 0
    area_values = []

    for filename in dataset.image_names:
        rows = dataset.grouped.get_group(filename)
        image_path = os.path.join(dataset.img_dir, filename)

        with Image.open(image_path) as image:
            width, height = image.size

        boxes = rows[["xmin", "ymin", "xmax", "ymax"]].to_numpy(dtype=np.float32)

        total_boxes += len(boxes)

        invalid = (
            (boxes[:, 2] <= boxes[:, 0]) |
            (boxes[:, 3] <= boxes[:, 1])
        )
        invalid_count += int(invalid.sum())

        outside = (
            (boxes[:, 0] < 0) |
            (boxes[:, 1] < 0) |
            (boxes[:, 2] > width) |
            (boxes[:, 3] > height)
        )
        clipped_count += int(outside.sum())

        edge_touch = (
            (boxes[:, 0] <= 0) |
            (boxes[:, 1] <= 0) |
            (boxes[:, 2] >= width) |
            (boxes[:, 3] >= height)
        )
        edge_touch_count += int(edge_touch.sum())

        valid_boxes = boxes[~invalid]
        if len(valid_boxes) > 0:
            areas = (valid_boxes[:, 2] - valid_boxes[:, 0]) * (valid_boxes[:, 3] - valid_boxes[:, 1])
            area_values.extend(areas.tolist())

    boxes_per_image = dataset.df.groupby("filename").size()
    aug_mask = dataset.df["filename"].str.contains(r"_aug|__aug", regex=True, na=False)

    print(f"\nDataset check: {split_name}")
    print(f"Images: {len(dataset)}")
    print(f"Boxes: {total_boxes}")
    print(f"Mean boxes/image: {boxes_per_image.mean():.3f}")
    print(f"Median boxes/image: {boxes_per_image.median():.3f}")
    print(f"Invalid boxes: {invalid_count}")
    print(f"Boxes requiring clipping: {clipped_count}")
    print(f"Boxes touching image border: {edge_touch_count}")
    print(f"Augmented filename rows: {int(aug_mask.sum())}")
    print("Class distribution:")
    print(labels)
    print("Class percentage:")
    print(label_pct.round(4))

    if len(area_values) > 0:
        area_values = np.array(area_values, dtype=np.float32)
        print(
            "Area stats | "
            f"mean={area_values.mean():.2f}, "
            f"median={np.median(area_values):.2f}, "
            f"std={area_values.std():.2f}, "
            f"min={area_values.min():.2f}, "
            f"max={area_values.max():.2f}"
        )


# =========================================================
# 4) Model Definition
# =========================================================
def get_model(num_classes=5):
    model = retinanet_resnet50_fpn(weights="DEFAULT")

    in_channels = model.backbone.out_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    return model


# =========================================================
# 5) Plotting
# =========================================================
def save_lr_curve(history_df, run_dir, run_idx):
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["lr"], marker="o", linewidth=2, color="blue")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title(f"Learning Rate Curve - Run {run_idx:02d}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "lr_curve.png"), dpi=300)
    plt.close()


def save_loss_curve(history_df, run_dir, run_idx):
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], marker="o", linewidth=2, color="red")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title(f"Train Loss Curve - Run {run_idx:02d}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "train_loss_curve.png"), dpi=300)
    plt.close()


def save_val_map_curve(history_df, run_dir, run_idx):
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["val_map_50"], marker="o", linewidth=2, color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Validation mAP@0.50")
    plt.title(f"Validation mAP@0.50 Curve - Run {run_idx:02d}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "val_map_curve.png"), dpi=300)
    plt.close()


def save_confusion_matrix_heatmap(conf_mat, run_dir):
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.title("Confusion Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix_heatmap.png"), dpi=300)
    plt.close()


# =========================================================
# 6) IoU and Confusion Matrix
# =========================================================
def compute_iou_matrix(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter

    return inter / torch.clamp(union, min=1e-6)


def update_confusion_matrix(
    conf_mat,
    gt_boxes,
    gt_labels,
    pred_boxes,
    pred_labels,
    pred_scores,
    iou_threshold=0.5,
    score_threshold=0.5,
):
    if pred_boxes.numel() > 0:
        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

    num_gt = gt_boxes.shape[0]
    num_pred = pred_boxes.shape[0]

    matched_gt_indices = set()
    matched_pred_indices = set()

    if num_gt > 0 and num_pred > 0:
        iou_mat = compute_iou_matrix(gt_boxes, pred_boxes)
        candidate_matches = []

        for i in range(num_gt):
            for j in range(num_pred):
                iou_val = float(iou_mat[i, j].item())
                if iou_val >= iou_threshold:
                    candidate_matches.append((iou_val, i, j))

        candidate_matches.sort(key=lambda x: x[0], reverse=True)

        for _, gt_idx, pred_idx in candidate_matches:
            if gt_idx in matched_gt_indices or pred_idx in matched_pred_indices:
                continue

            gt_cls = int(gt_labels[gt_idx].item())
            pred_cls = int(pred_labels[pred_idx].item())

            conf_mat[gt_cls, pred_cls] += 1
            matched_gt_indices.add(gt_idx)
            matched_pred_indices.add(pred_idx)

    for gt_idx in range(num_gt):
        if gt_idx not in matched_gt_indices:
            gt_cls = int(gt_labels[gt_idx].item())
            conf_mat[gt_cls, 0] += 1

    for pred_idx in range(num_pred):
        if pred_idx not in matched_pred_indices:
            pred_cls = int(pred_labels[pred_idx].item())
            conf_mat[0, pred_cls] += 1

    return conf_mat


def confusion_summary(conf_mat):
    foreground = conf_mat[1:, 1:]

    tp = int(np.trace(foreground))
    fn = int(conf_mat[1:, 0].sum())
    fp = int(conf_mat[0, 1:].sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def confusion_summary_per_class(conf_mat):
    rows = []

    for cls_id in CLASS_IDS:
        tp = int(conf_mat[cls_id, cls_id])
        fp = int(conf_mat[:, cls_id].sum() - tp)
        fn = int(conf_mat[cls_id, :].sum() - tp)
        support = int(conf_mat[cls_id, :].sum())

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        rows.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES[cls_id],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    return pd.DataFrame(rows)


def compute_macro_weighted_metrics(per_class_df):
    macro_precision = float(per_class_df["precision"].mean())
    macro_recall = float(per_class_df["recall"].mean())
    macro_f1 = float(per_class_df["f1"].mean())

    weights = per_class_df["support"].to_numpy(dtype=np.float64)
    if weights.sum() > 0:
        weights = weights / weights.sum()
        weighted_precision = float(np.sum(per_class_df["precision"].to_numpy(dtype=np.float64) * weights))
        weighted_recall = float(np.sum(per_class_df["recall"].to_numpy(dtype=np.float64) * weights))
        weighted_f1 = float(np.sum(per_class_df["f1"].to_numpy(dtype=np.float64) * weights))
    else:
        weighted_precision = 0.0
        weighted_recall = 0.0
        weighted_f1 = 0.0

    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


def extract_per_class_ap(metrics_dict):
    rows = []

    classes = metrics_dict.get("classes", [])
    map_per_class = metrics_dict.get("map_per_class", [])
    mar_100_per_class = metrics_dict.get("mar_100_per_class", [])

    class_to_ap = {}
    class_to_mar100 = {}

    if isinstance(classes, list) and isinstance(map_per_class, list):
        for cls_id, ap_value in zip(classes, map_per_class):
            class_to_ap[int(cls_id)] = float(ap_value)

    if isinstance(classes, list) and isinstance(mar_100_per_class, list):
        for cls_id, mar_value in zip(classes, mar_100_per_class):
            class_to_mar100[int(cls_id)] = float(mar_value)

    for cls_id in CLASS_IDS:
        rows.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES[cls_id],
            "ap": class_to_ap.get(cls_id, np.nan),
            "mar_100": class_to_mar100.get(cls_id, np.nan),
        })

    return pd.DataFrame(rows)


# =========================================================
# 7) Evaluation & Threshold Tuning
# =========================================================
@torch.no_grad()
def evaluate_map(model, data_loader, device):
    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        max_detection_thresholds=[1, 10, 300],
    )

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        preds = []
        gts = []

        for output, target in zip(outputs, targets):
            preds.append({
                "boxes": output["boxes"].detach().cpu(),
                "scores": output["scores"].detach().cpu(),
                "labels": output["labels"].detach().cpu(),
            })

            gts.append({
                "boxes": target["boxes"].detach().cpu(),
                "labels": target["labels"].detach().cpu(),
            })

        metric.update(preds, gts)

    results = metric.compute()
    clean_results = {}

    for k, v in results.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                clean_results[k] = float(v.item())
            else:
                clean_results[k] = v.cpu().tolist()
        else:
            clean_results[k] = v

    return clean_results


@torch.no_grad()
def evaluate_map_and_confusion(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5):
    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
        max_detection_thresholds=[1, 10, 300],
    )

    conf_mat = np.zeros((5, 5), dtype=np.int64)

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        preds = []
        gts = []

        for output, target in zip(outputs, targets):
            pred_boxes = output["boxes"].detach().cpu()
            pred_scores = output["scores"].detach().cpu()
            pred_labels = output["labels"].detach().cpu()

            gt_boxes = target["boxes"].detach().cpu()
            gt_labels = target["labels"].detach().cpu()

            preds.append({
                "boxes": pred_boxes,
                "scores": pred_scores,
                "labels": pred_labels,
            })

            gts.append({
                "boxes": gt_boxes,
                "labels": gt_labels,
            })

            conf_mat = update_confusion_matrix(
                conf_mat=conf_mat,
                gt_boxes=gt_boxes,
                gt_labels=gt_labels,
                pred_boxes=pred_boxes,
                pred_labels=pred_labels,
                pred_scores=pred_scores,
                iou_threshold=iou_threshold,
                score_threshold=score_threshold,
            )

        metric.update(preds, gts)

    results = metric.compute()
    clean_results = {}

    for k, v in results.items():
        if torch.is_tensor(v):
            if v.numel() == 1:
                clean_results[k] = float(v.item())
            else:
                clean_results[k] = v.cpu().tolist()
        else:
            clean_results[k] = v

    return clean_results, conf_mat


@torch.no_grad()
def select_best_threshold(model, val_loader, device, iou_threshold=0.5):
    thresholds = CONFIG["threshold_candidates"]

    best_thr = 0.05
    best_f1 = -1.0
    best_summary = None

    print("\n--- Tuning score threshold on validation set ---")

    model.eval()

    for thr in thresholds:
        conf_mat = np.zeros((5, 5), dtype=np.int64)

        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                conf_mat = update_confusion_matrix(
                    conf_mat=conf_mat,
                    gt_boxes=target["boxes"].detach().cpu(),
                    gt_labels=target["labels"].detach().cpu(),
                    pred_boxes=output["boxes"].detach().cpu(),
                    pred_labels=output["labels"].detach().cpu(),
                    pred_scores=output["scores"].detach().cpu(),
                    iou_threshold=iou_threshold,
                    score_threshold=thr,
                )

        summary = confusion_summary(conf_mat)

        print(
            f"Threshold: {thr:.2f} | "
            f"TP: {summary['tp']}, FP: {summary['fp']}, FN: {summary['fn']} | "
            f"Precision: {summary['precision']:.3f}, "
            f"Recall: {summary['recall']:.3f}, "
            f"F1: {summary['f1']:.3f}"
        )

        if summary["f1"] > best_f1:
            best_f1 = summary["f1"]
            best_thr = thr
            best_summary = summary

    print(
        f"Selected threshold: {best_thr:.2f} | "
        f"Precision: {best_summary['precision']:.3f}, "
        f"Recall: {best_summary['recall']:.3f}, "
        f"F1: {best_summary['f1']:.3f}"
    )

    return best_thr


# =========================================================
# 8) Train One Epoch
# =========================================================
def train_one_epoch(model, optimizer, data_loader, device, pbar, log_every=20):
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if not torch.isfinite(losses):
            loss_details = {
                name: float(value.detach().cpu())
                for name, value in loss_dict.items()
            }
            raise RuntimeError(
                f"Non-finite loss at batch {batch_idx}: {loss_details}"
            )

        optimizer.zero_grad(set_to_none=True)
        losses.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        loss_value = float(losses.item())
        running_loss += loss_value

        pbar.update(1)

        if batch_idx % log_every == 0 or batch_idx == len(data_loader):
            avg_loss = running_loss / batch_idx
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "avg": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.6f}",
            })

    return running_loss / len(data_loader)


# =========================================================
# 9) Early Stopping
# =========================================================
class EarlyStopping:
    def __init__(self, patience=12, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.counter = 0
        self.should_stop = False

    def step(self, current_score):
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.counter = 0
            improved = True
        else:
            self.counter += 1
            improved = False
            if self.counter >= self.patience:
                self.should_stop = True

        return improved


# =========================================================
# 10) Main
# =========================================================
def parse_args():
    parser=argparse.ArgumentParser(description="Train RetinaNet independently on expert labels.")
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    return parser.parse_args()

def main():
    args=parse_args()
    img_dir=args.images_dir; train_csv=args.train_csv; val_csv=args.val_csv; test_csv=args.test_csv; output_dir=args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    CONFIG.update(num_runs=args.num_runs,max_epochs=args.max_epochs,patience=args.patience,batch_size=args.batch_size,base_seed=args.base_seed,lr=args.lr,weight_decay=args.weight_decay)
    validate_splits(train_csv, val_csv, test_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    num_runs=CONFIG["num_runs"]; max_epochs=CONFIG["max_epochs"]; batch_size=CONFIG["batch_size"]; base_seed=CONFIG["base_seed"]; patience=CONFIG["patience"]
    iou_threshold=CONFIG["iou_threshold"]; initial_score_threshold=CONFIG["score_threshold_start"]

    print("\nUsing configuration:")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    all_runs_results = []

    for run_idx in range(1, num_runs + 1):
        run_seed = base_seed + run_idx - 1

        print("\n" + "=" * 70)
        print(f"Starting Run {run_idx}/{num_runs} | Seed = {run_seed}")
        print("=" * 70)

        set_seed(run_seed)

        run_dir = os.path.join(output_dir, f"run_{run_idx:02d}")
        os.makedirs(run_dir, exist_ok=True)

        train_dataset = TreeDetectionDataset(img_dir=img_dir, csv_file=train_csv)
        val_dataset = TreeDetectionDataset(img_dir=img_dir, csv_file=val_csv)
        test_dataset = TreeDetectionDataset(img_dir=img_dir, csv_file=test_csv)

        if run_idx == 1:
            inspect_dataset(train_dataset, "train")
            inspect_dataset(val_dataset, "validation")
            inspect_dataset(test_dataset, "test")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_fn
        )

        model = get_model(num_classes=5)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.SGD(
            params,
            lr=CONFIG["lr"],
            momentum=0.9,
            weight_decay=CONFIG["weight_decay"],
            nesterov=True,
        )

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            threshold=1e-4,
            min_lr=1e-6,
        )

        early_stopper = EarlyStopping(patience=patience)

        best_map_50 = -1.0
        best_epoch = -1
        stopped_early = False
        epochs_completed = 0

        best_model_path = os.path.join(run_dir, "best_model.pth")
        history = []

        for epoch in range(1, max_epochs + 1):
            with tqdm(
                total=len(train_loader),
                desc=f"Run {run_idx:02d} | Epoch {epoch:02d}",
                file=sys.stdout,
                ascii=True,
                dynamic_ncols=False
            ) as pbar:

                train_loss = train_one_epoch(
                    model=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    device=device,
                    pbar=pbar
                )

                val_metrics = evaluate_map(
                    model=model,
                    data_loader=val_loader,
                    device=device,
                )

                val_map = val_metrics.get("map", 0.0)
                val_map_50 = val_metrics.get("map_50", 0.0)
                val_map_75 = val_metrics.get("map_75", 0.0)
                current_lr = optimizer.param_groups[0]["lr"]

                pbar.set_postfix({
                    "loss": f"{train_loss:.4f}",
                    "mAP50": f"{val_map_50:.4f}",
                    "lr": f"{current_lr:.6f}",
                })

            selection_score = val_map_50
            improved = early_stopper.step(selection_score)

            row = {
                "run": run_idx,
                "seed": run_seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_map": val_map,
                "val_map_50": val_map_50,
                "val_map_75": val_map_75,
                "lr": current_lr,
                "best_map_50_so_far": max(best_map_50, val_map_50),
                "earlystop_counter": early_stopper.counter,
                "is_new_best": int(improved),
            }
            history.append(row)

            if improved:
                best_map_50 = selection_score
                best_epoch = epoch

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_map": val_map,
                        "val_map_50": val_map_50,
                        "val_map_75": val_map_75,
                    },
                    best_model_path,
                )

                print(
                    f"  -> [Epoch {epoch:02d}] Improved! "
                    f"mAP50: {val_map_50:.4f}, mAP: {val_map:.4f}. Model saved."
                )
            else:
                if epoch % 5 == 0 or early_stopper.counter > 2:
                    print(
                        f"  -> [Epoch {epoch:02d}] No improvement "
                        f"({early_stopper.counter}/{patience})"
                    )

            lr_scheduler.step(val_map_50)
            epochs_completed = epoch

            if early_stopper.should_stop:
                stopped_early = True
                print(f"  -> Early stopping triggered at epoch {epoch}")
                break

        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)

        save_lr_curve(history_df, run_dir, run_idx)
        save_loss_curve(history_df, run_dir, run_idx)
        save_val_map_curve(history_df, run_dir, run_idx)

        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        print(
            f"Loaded best model from epoch {checkpoint['epoch']} | "
            f"val mAP50={checkpoint['val_map_50']:.4f} | "
            f"val mAP={checkpoint['val_map']:.4f}"
        )

        best_score_threshold = select_best_threshold(
            model=model,
            val_loader=val_loader,
            device=device,
            iou_threshold=iou_threshold,
        )

        val_metrics_final, val_conf_mat = evaluate_map_and_confusion(
            model=model,
            data_loader=val_loader,
            device=device,
            iou_threshold=iou_threshold,
            score_threshold=best_score_threshold,
        )

        test_metrics, conf_mat = evaluate_map_and_confusion(
            model=model,
            data_loader=test_loader,
            device=device,
            iou_threshold=iou_threshold,
            score_threshold=best_score_threshold,
        )

        initial_test_metrics, initial_conf_mat = evaluate_map_and_confusion(
            model=model,
            data_loader=test_loader,
            device=device,
            iou_threshold=iou_threshold,
            score_threshold=initial_score_threshold,
        )

        val_summary = confusion_summary(val_conf_mat)
        test_summary = confusion_summary(conf_mat)
        initial_test_summary = confusion_summary(initial_conf_mat)

        val_per_class_df = confusion_summary_per_class(val_conf_mat)
        test_per_class_df = confusion_summary_per_class(conf_mat)
        initial_test_per_class_df = confusion_summary_per_class(initial_conf_mat)

        val_macro_weighted = compute_macro_weighted_metrics(val_per_class_df)
        test_macro_weighted = compute_macro_weighted_metrics(test_per_class_df)
        initial_test_macro_weighted = compute_macro_weighted_metrics(initial_test_per_class_df)

        val_ap_df = extract_per_class_ap(val_metrics_final)
        test_ap_df = extract_per_class_ap(test_metrics)
        initial_test_ap_df = extract_per_class_ap(initial_test_metrics)

        val_per_class_df = val_per_class_df.merge(
            val_ap_df, on=["class_id", "class_name"], how="left"
        )
        test_per_class_df = test_per_class_df.merge(
            test_ap_df, on=["class_id", "class_name"], how="left"
        )
        initial_test_per_class_df = initial_test_per_class_df.merge(
            initial_test_ap_df, on=["class_id", "class_name"], how="left"
        )

        test_metrics_row = {
            "run": run_idx,
            "seed": run_seed,
            "best_epoch": best_epoch,
            "epochs_completed": epochs_completed,
            "stopped_early": int(stopped_early),
            "best_val_map_50": best_map_50,
            "selected_threshold": best_score_threshold,
            "val_precision_at_selected_thr": val_summary["precision"],
            "val_recall_at_selected_thr": val_summary["recall"],
            "val_f1_at_selected_thr": val_summary["f1"],
            "val_macro_precision_at_selected_thr": val_macro_weighted["macro_precision"],
            "val_macro_recall_at_selected_thr": val_macro_weighted["macro_recall"],
            "val_macro_f1_at_selected_thr": val_macro_weighted["macro_f1"],
            "val_weighted_precision_at_selected_thr": val_macro_weighted["weighted_precision"],
            "val_weighted_recall_at_selected_thr": val_macro_weighted["weighted_recall"],
            "val_weighted_f1_at_selected_thr": val_macro_weighted["weighted_f1"],
            "test_precision_at_selected_thr": test_summary["precision"],
            "test_recall_at_selected_thr": test_summary["recall"],
            "test_f1_at_selected_thr": test_summary["f1"],
            "test_macro_precision_at_selected_thr": test_macro_weighted["macro_precision"],
            "test_macro_recall_at_selected_thr": test_macro_weighted["macro_recall"],
            "test_macro_f1_at_selected_thr": test_macro_weighted["macro_f1"],
            "test_weighted_precision_at_selected_thr": test_macro_weighted["weighted_precision"],
            "test_weighted_recall_at_selected_thr": test_macro_weighted["weighted_recall"],
            "test_weighted_f1_at_selected_thr": test_macro_weighted["weighted_f1"],
            "test_precision_at_0p05": initial_test_summary["precision"],
            "test_recall_at_0p05": initial_test_summary["recall"],
            "test_f1_at_0p05": initial_test_summary["f1"],
            "test_macro_precision_at_0p05": initial_test_macro_weighted["macro_precision"],
            "test_macro_recall_at_0p05": initial_test_macro_weighted["macro_recall"],
            "test_macro_f1_at_0p05": initial_test_macro_weighted["macro_f1"],
            "test_weighted_precision_at_0p05": initial_test_macro_weighted["weighted_precision"],
            "test_weighted_recall_at_0p05": initial_test_macro_weighted["weighted_recall"],
            "test_weighted_f1_at_0p05": initial_test_macro_weighted["weighted_f1"],
        }

        for k, v in val_metrics_final.items():
            if not isinstance(v, list):
                test_metrics_row[f"val_{k}"] = v

        for k, v in test_metrics.items():
            if not isinstance(v, list):
                test_metrics_row[f"test_{k}"] = v

        pd.DataFrame([test_metrics_row]).to_csv(
            os.path.join(run_dir, "test_metrics.csv"),
            index=False
        )

        conf_df = pd.DataFrame(conf_mat, index=CLASS_NAMES, columns=CLASS_NAMES)
        conf_df.to_csv(os.path.join(run_dir, "confusion_matrix.csv"))

        val_conf_df = pd.DataFrame(val_conf_mat, index=CLASS_NAMES, columns=CLASS_NAMES)
        val_conf_df.to_csv(os.path.join(run_dir, "val_confusion_matrix.csv"))

        initial_conf_df = pd.DataFrame(initial_conf_mat, index=CLASS_NAMES, columns=CLASS_NAMES)
        initial_conf_df.to_csv(os.path.join(run_dir, "test_confusion_matrix_thr_0p05.csv"))

        val_per_class_df.to_csv(os.path.join(run_dir, "val_per_class_metrics.csv"), index=False)
        test_per_class_df.to_csv(os.path.join(run_dir, "test_per_class_metrics.csv"), index=False)
        initial_test_per_class_df.to_csv(
            os.path.join(run_dir, "test_per_class_metrics_thr_0p05.csv"),
            index=False
        )

        save_confusion_matrix_heatmap(conf_mat, run_dir)

        print(
            f"Run {run_idx:02d} finished. "
            f"Best validation mAP50: {best_map_50:.4f} at epoch {best_epoch}"
        )
        print(
            f"Selected threshold = {best_score_threshold:.2f} | "
            f"Test Precision = {test_summary['precision']:.4f} | "
            f"Test Recall = {test_summary['recall']:.4f} | "
            f"Test F1 = {test_summary['f1']:.4f} | "
            f"Test Macro F1 = {test_macro_weighted['macro_f1']:.4f} | "
            f"Test Weighted F1 = {test_macro_weighted['weighted_f1']:.4f}"
        )

        print("\nTest per-class metrics at selected threshold:")
        print(test_per_class_df[["class_id", "precision", "recall", "f1", "support", "ap", "mar_100"]])

        all_runs_results.append(test_metrics_row)

    all_runs_df = pd.DataFrame(all_runs_results)
    all_runs_df.to_csv(os.path.join(output_dir, "all_runs_summary.csv"), index=False)

    metric_cols = [
        col for col in all_runs_df.columns
        if col.startswith("test_") or col == "best_val_map_50"
    ]

    mean_std_rows = []
    for c in metric_cols:
        mean_std_rows.append({
            "metric": c,
            "mean": all_runs_df[c].mean(),
            "std": all_runs_df[c].std()
        })

    mean_std_df = pd.DataFrame(mean_std_rows)
    mean_std_df.to_csv(os.path.join(output_dir, "all_runs_mean_std.csv"), index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"All results saved to: {output_dir}")
    print("=" * 70)
    print("Summary of all runs:")
    print(all_runs_df)
    print("\nMean and Standard Deviation across runs:")
    print(mean_std_df)


if __name__ == "__main__":
    main()
