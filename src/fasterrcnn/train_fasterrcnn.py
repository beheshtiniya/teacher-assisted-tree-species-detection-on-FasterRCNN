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
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchmetrics.detection.mean_ap import MeanAveragePrecision


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
            raise ValueError(f"Invalid classes found in {csv_file}: {found_classes - valid_classes}")

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
# 3) Utilities
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


def get_model(num_classes=5):
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


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
    plt.plot(history_df["epoch"], history_df["val_map"], marker="o", linewidth=2, color="green")
    plt.xlabel("Epoch")
    plt.ylabel("Validation mAP")
    plt.title(f"Validation mAP Curve - Run {run_idx:02d}")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "val_map_curve.png"), dpi=300)
    plt.close()


def save_confusion_matrix_heatmap(conf_mat, run_dir):
    class_names = [f"class{i}" for i in range(5)]

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted Class")
    plt.ylabel("Ground Truth Class")
    plt.title("Confusion Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "confusion_matrix_heatmap.png"), dpi=300)
    plt.close()


# =========================================================
# 4) IoU and Confusion Matrix
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


def update_confusion_matrix(conf_mat, gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores,
                            iou_threshold=0.5, score_threshold=0.5):
    if pred_boxes.numel() > 0:
        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

    num_gt = gt_boxes.shape[0]
    num_pred = pred_boxes.shape[0]

    matched_gt = set()
    matched_pred = set()

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
            if gt_idx in matched_gt or pred_idx in matched_pred:
                continue

            gt_cls = int(gt_labels[gt_idx].item())
            pred_cls = int(pred_labels[pred_idx].item())
            conf_mat[gt_cls, pred_cls] += 1

            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)

    for gt_idx in range(num_gt):
        if gt_idx not in matched_gt:
            gt_cls = int(gt_labels[gt_idx].item())
            conf_mat[gt_cls, 0] += 1

    for pred_idx in range(num_pred):
        if pred_idx not in matched_pred:
            pred_cls = int(pred_labels[pred_idx].item())
            conf_mat[0, pred_cls] += 1

    return conf_mat


# =========================================================
# 5) Evaluation
# =========================================================
@torch.no_grad()
def evaluate_map(model, data_loader, device):
    """نسخه سبک بدون نوار پیشرفت داخلی برای جلوگیری از چندخطی شدن"""
    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True
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
    """استفاده در پایان هر Run برای تست نهایی"""
    model.eval()

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True
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
                score_threshold=score_threshold
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


# =========================================================
# 6) Train One Epoch
# =========================================================
def train_one_epoch(model, optimizer, data_loader, device, pbar, log_every=20):
    """
    استفاده از pbar پاس داده شده برای جلوگیری از ایجاد خط جدید.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader, start=1):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_value = losses.item()
        running_loss += loss_value

        # آپدیت نوار پیشرفت موجود
        pbar.update(1)

        # آپدیت اطلاعات متنی نوار در فواصل مشخص
        if batch_idx % log_every == 0 or batch_idx == len(data_loader):
            avg_loss = running_loss / batch_idx
            current_lr = optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{loss_value:.4f}",
                "avg": f"{avg_loss:.4f}",
                "lr": f"{current_lr:.5f}"
            })

    return running_loss / len(data_loader)


# =========================================================
# 7) Early Stopping
# =========================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
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
# 8) Main
# =========================================================
def parse_args():
    p=argparse.ArgumentParser(description='Train Faster R-CNN with repeated seeded runs.')
    p.add_argument('--images-dir',required=True); p.add_argument('--train-csv',required=True); p.add_argument('--val-csv',required=True); p.add_argument('--test-csv',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--num-runs',type=int,default=5); p.add_argument('--max-epochs',type=int,default=20); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--workers',type=int,default=0); p.add_argument('--base-seed',type=int,default=42); p.add_argument('--patience',type=int,default=5); p.add_argument('--min-delta',type=float,default=1e-4); p.add_argument('--lr',type=float,default=0.005); p.add_argument('--weight-decay',type=float,default=0.0005); p.add_argument('--iou-threshold',type=float,default=0.5); p.add_argument('--score-threshold',type=float,default=0.5)
    return p.parse_args()

def main():
    args=parse_args(); img_dir=args.images_dir; train_csv=args.train_csv; val_csv=args.val_csv; test_csv=args.test_csv; output_dir=args.output_dir; os.makedirs(output_dir,exist_ok=True)
    validate_splits(train_csv,val_csv,test_csv); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); print('Device:',device)
    num_runs=args.num_runs; max_epochs=args.max_epochs; batch_size=args.batch_size; base_seed=args.base_seed; patience=args.patience; min_delta=args.min_delta; iou_threshold=args.iou_threshold; score_threshold=args.score_threshold
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

        # نکته: num_workers=0 برای جلوگیری از تداخل کنسول در ویندوز الزامی است
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.workers,
                                  collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                 collate_fn=collate_fn)

        model = get_model(num_classes=5)
        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

        best_map = -1.0
        best_epoch = -1
        stopped_early = False
        epochs_completed = 0

        best_model_path = os.path.join(run_dir, "best_model.pth")
        history = []

        for epoch in range(1, max_epochs + 1):
            # ایجاد یک نوار برای کل اپک
            with tqdm(total=len(train_loader),
                      desc=f"Run {run_idx:02d} | Epoch {epoch:02d}",
                      file=sys.stdout,
                      ascii=True,
                      dynamic_ncols=False) as pbar:

                train_loss = train_one_epoch(
                    model=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    device=device,
                    pbar=pbar
                )

                val_metrics = evaluate_map(model=model, data_loader=val_loader, device=device)

                val_map = val_metrics.get("map", 0.0)
                val_map_50 = val_metrics.get("map_50", 0.0)
                val_map_75 = val_metrics.get("map_75", 0.0)
                current_lr = optimizer.param_groups[0]["lr"]

                # نمایش نتایج در همان خط نوار پیشرفت
                pbar.set_postfix(loss=f"{train_loss:.3f}", mAP=f"{val_map:.3f}", lr=f"{current_lr:.5f}")

            # چاپ خلاصه اپک بعد از بسته شدن نوار (خارج از حلقه بچ)
            improved = early_stopper.step(val_map)

            row = {
                "run": run_idx, "seed": run_seed, "epoch": epoch,
                "train_loss": train_loss, "val_map": val_map,
                "val_map_50": val_map_50, "val_map_75": val_map_75,
                "lr": current_lr, "best_map_so_far": max(best_map, val_map),
                "earlystop_counter": early_stopper.counter, "is_best_epoch": int(improved)
            }
            history.append(row)

            if improved:
                best_map = val_map
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> [Epoch {epoch:02d}] Improved! mAP: {val_map:.4f}. Model saved.")
            else:
                if epoch % 5 == 0 or early_stopper.counter > 2:  # کاهش تعداد چاپ‌های اضافی
                    print(f"  -> [Epoch {epoch:02d}] No improvement ({early_stopper.counter}/{patience})")

            lr_scheduler.step()
            epochs_completed = epoch

            if early_stopper.should_stop:
                stopped_early = True
                print(f"  -> Early stopping triggered at epoch {epoch}")
                break

        # ذخیره‌سازی نتایج Run
        history_df = pd.DataFrame(history)
        history_df.to_csv(os.path.join(run_dir, "history.csv"), index=False)
        save_lr_curve(history_df, run_dir, run_idx)
        save_loss_curve(history_df, run_dir, run_idx)
        save_val_map_curve(history_df, run_dir, run_idx)

        # تست نهایی روی بهترین مدل این Run
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_metrics, conf_mat = evaluate_map_and_confusion(model, test_loader, device, iou_threshold, score_threshold)

        test_metrics_row = {
            "run": run_idx, "seed": run_seed, "best_epoch": best_epoch,
            "epochs_completed": epochs_completed, "stopped_early": int(stopped_early),
            "best_val_map": best_map,
        }
        for k, v in test_metrics.items():
            if not isinstance(v, list): test_metrics_row[f"test_{k}"] = v

        pd.DataFrame([test_metrics_row]).to_csv(os.path.join(run_dir, "test_metrics.csv"), index=False)

        class_names = [f"class{i}" for i in range(5)]
        conf_df = pd.DataFrame(conf_mat, index=class_names, columns=class_names)
        conf_df.to_csv(os.path.join(run_dir, "confusion_matrix.csv"))
        save_confusion_matrix_heatmap(conf_mat, run_dir)

        print(f"Run {run_idx:02d} finished. Best mAP: {best_map:.4f} at epoch {best_epoch}")
        all_runs_results.append(test_metrics_row)

    # گزارش نهایی کلیه Runها
    all_runs_df = pd.DataFrame(all_runs_results)
    all_runs_df.to_csv(os.path.join(output_dir, "all_runs_summary.csv"), index=False)

    metric_cols = [col for col in all_runs_df.columns if col.startswith("test_") or col == "best_val_map"]
    mean_std_rows = [{"metric": c, "mean": all_runs_df[c].mean(), "std": all_runs_df[c].std()} for c in metric_cols]

    mean_std_df = pd.DataFrame(mean_std_rows)
    mean_std_df.to_csv(os.path.join(output_dir, "all_runs_mean_std.csv"), index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print(f"Summary saved to: {output_dir}")
    print("=" * 70)
    print(all_runs_df)


if __name__ == "__main__":
    main()
