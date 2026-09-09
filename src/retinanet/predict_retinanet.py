import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.ops import nms


# ============================================================
# Configuration
# ============================================================
RUN_DIR = Path(".")
CHECKPOINT_PATH = RUN_DIR / "best_model.pth"

IMAGES_DIR = Path(".")


LABEL_FILES = {
    # RetinaNet is evaluated against the same original split files used for training.
    "train": Path("."),
    "val": Path("."),
    "test": Path("."),
}

UNLABELED_IMAGES_LIST = Path(".")

# Separate export for high-confidence predictions on unlabeled images.
UNLABELED_HIGH_CONF_THRESHOLD = 0.75

PREDICTIONS_OUTPUT_DIR = Path(".")
COMBINED_OUTPUT_DIR = Path(".")

NUM_CLASSES = 5  # Same head size used during training
VALID_CLASS_IDS = {1, 2, 3, 4}
BATCH_SIZE = 4
NUM_WORKERS = 0  # Recommended on Windows

# Step 1: remove very weak predictions.
# Only predictions with confidence >= this value continue to NMS.
PRED_SCORE_THRESHOLD = 0.25

# Step 2: class-agnostic NMS.
# When two predictions overlap above this IoU, keep only the higher-score box,
# even when their predicted classes are different.
CLASS_AGNOSTIC_NMS_IOU_THRESHOLD = 0.50

# Step 3: remove a prediction when its IoU with ANY GT box is greater
# than this value. Class is deliberately ignored during this filtering.
IOU_REMOVE_THRESHOLD = 0.50

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}

PREDICTION_COLUMNS = [
    "filename",
    "class",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
    "confidence",
]


# ============================================================
# Utility functions
# ============================================================
def natural_key(value):
    """Natural sorting key, e.g. 2.tif before 10.tif."""
    text = str(value)
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    ]


def filename_key(value):
    """Case-insensitive basename used for matching images and CSV rows."""
    return Path(str(value).strip()).name.lower()


def threshold_token(value):
    """Convert 0.75 to 0p75 for deterministic output filenames."""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def resolve_table_path(path_without_required_extension):
    """
    Accept an exact path or automatically try .csv, .xlsx and .xls.
    """
    path = Path(path_without_required_extension)

    if path.is_file():
        return path

    for suffix in (".csv", ".xlsx", ".xls"):
        candidate = Path(str(path) + suffix)
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Label file not found: {path}\n"
        f"Also tried: {path}.csv, {path}.xlsx, {path}.xls"
    )


def read_table(path):
    """Read CSV or Excel."""
    path = resolve_table_path(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df.columns = [str(col).strip() for col in df.columns]
    return df, path


def find_column(df, candidates):
    """Find a column using case-insensitive aliases."""
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def standardize_gt_dataframe(df, source_path):
    """
    Standardize the input label table while preserving label confidence.

    Source rule:
    - confidence == 1.0: original human ground truth
    - confidence < 1.0: input pseudo-label (only for backward compatibility)

    For backward compatibility, when the input file has no confidence column,
    every input box is treated as human ground truth with confidence 1.0.
    """
    aliases = {
        "filename": ["filename", "file_name", "image", "image_name", "img_name"],
        "class": ["class", "class_id", "label", "category_id"],
        "xmin": ["xmin", "x_min", "left"],
        "ymin": ["ymin", "y_min", "top"],
        "xmax": ["xmax", "x_max", "right"],
        "ymax": ["ymax", "y_max", "bottom"],
    }

    rename_map = {}

    for standard_name, candidates in aliases.items():
        found = find_column(df, candidates)
        if found is None:
            raise ValueError(
                f"Required column '{standard_name}' was not found in:\n"
                f"{source_path}\n"
                f"Available columns: {list(df.columns)}"
            )
        rename_map[found] = standard_name

    confidence_column = find_column(
        df,
        ["confidence", "conf", "score", "probability"],
    )
    if confidence_column is not None:
        rename_map[confidence_column] = "confidence"

    standardized = df.rename(columns=rename_map)
    selected_columns = [
        "filename", "class", "xmin", "ymin", "xmax", "ymax"
    ]
    if "confidence" in standardized.columns:
        selected_columns.append("confidence")

    gt = standardized[selected_columns].copy()

    if "confidence" not in gt.columns:
        gt["confidence"] = 1.0

    gt["filename"] = gt["filename"].astype(str).map(
        lambda x: Path(x.strip()).name
    )

    gt["class"] = pd.to_numeric(gt["class"], errors="coerce")
    for col in ["xmin", "ymin", "xmax", "ymax", "confidence"]:
        gt[col] = pd.to_numeric(gt[col], errors="coerce")

    invalid_numeric = gt[
        ["class", "xmin", "ymin", "xmax", "ymax", "confidence"]
    ].isna().any(axis=1)

    if invalid_numeric.any():
        print(
            f"[WARNING] Dropping {int(invalid_numeric.sum())} rows with "
            f"non-numeric label values from {source_path}"
        )
        gt = gt.loc[~invalid_numeric].copy()

    valid_confidence = gt["confidence"].between(0.0, 1.0, inclusive="both")
    if (~valid_confidence).any():
        raise ValueError(
            f"Found {int((~valid_confidence).sum())} input labels with "
            f"confidence outside [0, 1] in:\n{source_path}"
        )

    gt["class"] = gt["class"].astype(int)

    valid_box = (
        (gt["xmax"] > gt["xmin"]) &
        (gt["ymax"] > gt["ymin"])
    )

    if (~valid_box).any():
        print(
            f"[WARNING] Dropping {int((~valid_box).sum())} invalid label boxes "
            f"from {source_path}"
        )
        gt = gt.loc[valid_box].copy()

    gt["source"] = np.where(
        np.isclose(gt["confidence"].to_numpy(dtype=float), 1.0),
        "ground_truth",
        "input_pseudo_label",
    )

    gt["_key"] = gt["filename"].map(filename_key)
    return gt.reset_index(drop=True)



def read_unlabeled_image_names(txt_path):
    """
    Read image filenames from TXT.
    Empty lines, comments and an optional filename header are ignored.
    """
    txt_path = Path(txt_path)

    if not txt_path.is_file():
        raise FileNotFoundError(
            f"Unlabeled image list not found:\n{txt_path}"
        )

    names = []
    seen = set()

    with open(txt_path, "r", encoding="utf-8-sig") as file:
        for raw_line in file:
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            value = line.split(",")[0].strip().strip('"').strip("'")

            if value.lower() in {
                "filename",
                "file_name",
                "image",
                "image_name",
            }:
                continue

            name = Path(value).name
            key = filename_key(name)

            if key and key not in seen:
                seen.add(key)
                names.append(name)

    if not names:
        raise RuntimeError(
            f"No image names were found in:\n{txt_path}"
        )

    return names

def make_empty_predictions_dataframe():
    return pd.DataFrame(columns=PREDICTION_COLUMNS)


def save_csv(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ============================================================
# Model
# ============================================================
def get_model(num_classes=5):
    """
    Build exactly the RetinaNet architecture used in training,
    without downloading pretrained weights during inference.
    """
    try:
        model = retinanet_resnet50_fpn(
            weights=None,
            weights_backbone=None,
        )
    except TypeError:
        # Compatibility with older torchvision versions.
        model = retinanet_resnet50_fpn(
            pretrained=False,
            pretrained_backbone=False,
        )

    in_channels = model.backbone.out_channels
    num_anchors = model.anchor_generator.num_anchors_per_location()[0]

    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    return model


def clean_state_dict_keys(state_dict):
    """Remove common wrappers such as module. from checkpoint keys."""
    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod.", "model."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def load_trained_model(checkpoint_path, device):
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    model = get_model(num_classes=NUM_CLASSES)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    state_dict = clean_state_dict_keys(state_dict)
    model.load_state_dict(state_dict, strict=True)

    model.to(device)
    model.eval()

    if isinstance(checkpoint, dict):
        epoch = checkpoint.get("epoch", "unknown")
        val_map_50 = checkpoint.get("val_map_50", "unknown")
        print(
            f"Loaded checkpoint | epoch={epoch} | "
            f"val_map_50={val_map_50}"
        )

    return model


# ============================================================
# Image inference dataset
# ============================================================
class InferenceImageDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                width, height = image.size
                tensor = F.to_tensor(image)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to open image: {image_path}\n{exc}"
            ) from exc

        return tensor, image_path.name, (width, height)


def inference_collate_fn(batch):
    images, filenames, sizes = zip(*batch)
    return list(images), list(filenames), list(sizes)


def collect_image_paths(images_dir):
    if not images_dir.is_dir():
        raise NotADirectoryError(
            f"Images directory not found:\n{images_dir}"
        )

    image_paths = [
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_paths = sorted(
        image_paths,
        key=lambda p: natural_key(p.name),
    )

    if not image_paths:
        raise RuntimeError(
            f"No supported images were found in:\n{images_dir}"
        )

    lower_names = [path.name.lower() for path in image_paths]
    duplicates = sorted({
        name for name in lower_names
        if lower_names.count(name) > 1
    })

    if duplicates:
        raise RuntimeError(
            "Duplicate image basenames were found:\n"
            + "\n".join(duplicates[:20])
        )

    return image_paths


# ============================================================
# Prediction generation
# ============================================================
@torch.inference_mode()
def generate_predictions(
    model,
    image_paths,
    device,
    score_threshold,
):
    dataset = InferenceImageDataset(image_paths)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=inference_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    rows = []

    for images, filenames, sizes in tqdm(
        loader,
        desc="Generating predictions",
        total=len(loader),
    ):
        images = [
            image.to(device, non_blocking=True)
            for image in images
        ]

        outputs = model(images)

        for output, filename, (width, height) in zip(
            outputs,
            filenames,
            sizes,
        ):
            boxes = output["boxes"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()

            keep = (scores >= score_threshold) & np.isin(
                labels,
                list(VALID_CLASS_IDS),
            )
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]

            if len(boxes) == 0:
                continue

            boxes[:, 0] = np.clip(boxes[:, 0], 0, width)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, width)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, height)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, height)

            valid = (
                (boxes[:, 2] > boxes[:, 0]) &
                (boxes[:, 3] > boxes[:, 1])
            )

            boxes = boxes[valid]
            labels = labels[valid]
            scores = scores[valid]

            for box, label, score in zip(
                boxes,
                labels,
                scores,
            ):
                rows.append({
                    "filename": filename,
                    "class": int(label),
                    "xmin": float(box[0]),
                    "ymin": float(box[1]),
                    "xmax": float(box[2]),
                    "ymax": float(box[3]),
                    "confidence": float(score),
                })

    if not rows:
        return make_empty_predictions_dataframe()

    predictions = pd.DataFrame(
        rows,
        columns=PREDICTION_COLUMNS,
    )

    predictions = predictions.sort_values(
        by=["filename", "confidence"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return predictions


# ============================================================
# IoU filtering
# ============================================================
def pairwise_iou_numpy(boxes1, boxes2):
    """
    boxes1: N x 4
    boxes2: M x 4
    returns: N x M
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros(
            (len(boxes1), len(boxes2)),
            dtype=np.float32,
        )

    boxes1 = boxes1.astype(np.float32)
    boxes2 = boxes2.astype(np.float32)

    left_top = np.maximum(
        boxes1[:, None, :2],
        boxes2[None, :, :2],
    )
    right_bottom = np.minimum(
        boxes1[:, None, 2:],
        boxes2[None, :, 2:],
    )

    wh = np.clip(
        right_bottom - left_top,
        a_min=0,
        a_max=None,
    )
    intersection = wh[:, :, 0] * wh[:, :, 1]

    area1 = (
        (boxes1[:, 2] - boxes1[:, 0]) *
        (boxes1[:, 3] - boxes1[:, 1])
    )
    area2 = (
        (boxes2[:, 2] - boxes2[:, 0]) *
        (boxes2[:, 3] - boxes2[:, 1])
    )

    union = (
        area1[:, None]
        + area2[None, :]
        - intersection
    )

    return intersection / np.clip(union, 1e-7, None)



def apply_class_agnostic_nms(
    predictions,
    iou_threshold,
):
    """
    Apply NMS independently inside each image while ignoring class labels.

    For overlapping boxes, torchvision.ops.nms keeps the prediction with
    the highest confidence score. Predictions from different classes can
    suppress each other because labels are not passed to NMS.
    """
    pred = predictions.copy()

    if pred.empty:
        removed = pred.copy()
        return pred, removed

    pred["_key"] = pred["filename"].map(filename_key)

    kept_parts = []
    removed_parts = []

    for _, group in pred.groupby("_key", sort=False):
        group = group.copy().reset_index(drop=True)

        boxes = torch.as_tensor(
            group[["xmin", "ymin", "xmax", "ymax"]].to_numpy(
                dtype=np.float32
            ),
            dtype=torch.float32,
        )
        scores = torch.as_tensor(
            group["confidence"].to_numpy(dtype=np.float32),
            dtype=torch.float32,
        )

        keep_indices = nms(
            boxes=boxes,
            scores=scores,
            iou_threshold=iou_threshold,
        ).cpu().numpy()

        keep_mask = np.zeros(len(group), dtype=bool)
        keep_mask[keep_indices] = True

        if keep_mask.any():
            kept_parts.append(group.loc[keep_mask].copy())

        if (~keep_mask).any():
            removed_group = group.loc[~keep_mask].copy()
            removed_group["removal_reason"] = (
                "class_agnostic_nms_lower_score"
            )
            removed_parts.append(removed_group)

    if kept_parts:
        kept = pd.concat(kept_parts, ignore_index=True)
    else:
        kept = pd.DataFrame(columns=list(predictions.columns) + ["_key"])

    if removed_parts:
        removed = pd.concat(removed_parts, ignore_index=True)
    else:
        removed = pd.DataFrame(
            columns=list(predictions.columns)
            + ["_key", "removal_reason"]
        )

    kept = kept.drop(columns=["_key"], errors="ignore")
    removed = removed.drop(columns=["_key"], errors="ignore")

    kept = kept.sort_values(
        by=["filename", "confidence"],
        ascending=[True, False],
    ).reset_index(drop=True)

    removed = removed.sort_values(
        by=["filename", "confidence"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return kept, removed

def filter_predictions_against_gt(
    predictions,
    gt,
    iou_threshold,
):
    """
    Remove every prediction whose maximum IoU with any GT box
    from the same image is strictly greater than iou_threshold.

    Matching is class-independent.
    """
    pred = predictions.copy()

    if pred.empty:
        pred["max_iou_with_gt"] = pd.Series(dtype=float)
        removed = pred.copy()
        return pred, removed

    pred["_key"] = pred["filename"].map(filename_key)

    gt_boxes_by_image = {
        key: group[
            ["xmin", "ymin", "xmax", "ymax"]
        ].to_numpy(dtype=np.float32)
        for key, group in gt.groupby("_key")
    }

    kept_parts = []
    removed_parts = []

    for key, group in pred.groupby("_key", sort=False):
        group = group.copy()

        gt_boxes = gt_boxes_by_image.get(key)

        if gt_boxes is None or len(gt_boxes) == 0:
            group["max_iou_with_gt"] = 0.0
            kept_parts.append(group)
            continue

        pred_boxes = group[
            ["xmin", "ymin", "xmax", "ymax"]
        ].to_numpy(dtype=np.float32)

        iou_matrix = pairwise_iou_numpy(
            pred_boxes,
            gt_boxes,
        )
        max_iou = iou_matrix.max(axis=1)
        group["max_iou_with_gt"] = max_iou

        # User requested removal only when IoU is greater than 0.5.
        remove_mask = max_iou > iou_threshold

        if (~remove_mask).any():
            kept_parts.append(
                group.loc[~remove_mask].copy()
            )

        if remove_mask.any():
            removed_parts.append(
                group.loc[remove_mask].copy()
            )

    output_columns = (
        PREDICTION_COLUMNS
        + ["max_iou_with_gt"]
    )

    if kept_parts:
        kept = pd.concat(
            kept_parts,
            ignore_index=True,
        )
    else:
        kept = pd.DataFrame(columns=output_columns + ["_key"])

    if removed_parts:
        removed = pd.concat(
            removed_parts,
            ignore_index=True,
        )
    else:
        removed = pd.DataFrame(columns=output_columns + ["_key"])

    kept = kept.drop(columns=["_key"], errors="ignore")
    removed = removed.drop(columns=["_key"], errors="ignore")

    kept = kept.reindex(columns=output_columns)
    removed = removed.reindex(columns=output_columns)

    return kept, removed


def build_combined_gt_and_predictions(gt, kept_predictions):
    # Preserve the original confidence and provenance of every input label.
    # Human GT: confidence=1, source=ground_truth
    # Optional input pseudo-label: original confidence,
    # source=input_pseudo_label
    gt_output = gt[
        [
            "filename",
            "class",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "confidence",
            "source",
        ]
    ].copy()
    gt_output["max_iou_with_gt"] = np.nan

    # New detections produced by this RetinaNet checkpoint keep their own score.
    pred_output = kept_predictions.copy()
    pred_output["source"] = "retinanet_prediction"

    columns = [
        "filename",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "confidence",
        "source",
        "max_iou_with_gt",
    ]

    combined = pd.concat(
        [
            gt_output[columns],
            pred_output[columns],
        ],
        ignore_index=True,
    )

    combined["_source_order"] = combined["source"].map({
        "ground_truth": 0,
        "input_pseudo_label": 1,
        "retinanet_prediction": 2,
    }).fillna(3)

    combined = combined.sort_values(
        by=["filename", "_source_order", "confidence"],
        ascending=[True, True, False],
    )

    return combined.drop(
        columns=["_source_order"],
    ).reset_index(drop=True)


# ============================================================
# Main
# ============================================================
def parse_args():
    p=argparse.ArgumentParser(description='Run RetinaNet prediction, class-agnostic NMS, and GT-overlap filtering.')
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--images-dir', type=Path, required=True)
    p.add_argument('--train-csv', type=Path, required=True)
    p.add_argument('--val-csv', type=Path, required=True)
    p.add_argument('--test-csv', type=Path, required=True)
    p.add_argument('--unlabeled-list', type=Path, required=True)
    p.add_argument('--output-root', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--confidence', type=float, default=0.25)
    p.add_argument('--nms-iou', type=float, default=0.50)
    p.add_argument('--gt-iou', type=float, default=0.50)
    p.add_argument('--unlabeled-high-confidence', type=float, default=0.75)
    return p.parse_args()

def main():
    global RUN_DIR, CHECKPOINT_PATH, IMAGES_DIR, LABEL_FILES, UNLABELED_IMAGES_LIST
    global PREDICTIONS_OUTPUT_DIR, COMBINED_OUTPUT_DIR, BATCH_SIZE, NUM_WORKERS
    global PRED_SCORE_THRESHOLD, CLASS_AGNOSTIC_NMS_IOU_THRESHOLD, IOU_REMOVE_THRESHOLD
    global UNLABELED_HIGH_CONF_THRESHOLD
    args=parse_args()
    CHECKPOINT_PATH=args.checkpoint; RUN_DIR=CHECKPOINT_PATH.parent.parent; IMAGES_DIR=args.images_dir
    LABEL_FILES={'train':args.train_csv,'val':args.val_csv,'test':args.test_csv}
    UNLABELED_IMAGES_LIST=args.unlabeled_list
    PREDICTIONS_OUTPUT_DIR=args.output_root/'predictions'; COMBINED_OUTPUT_DIR=args.output_root/'combined'
    BATCH_SIZE=args.batch_size; NUM_WORKERS=args.workers; PRED_SCORE_THRESHOLD=args.confidence
    CLASS_AGNOSTIC_NMS_IOU_THRESHOLD=args.nms_iou; IOU_REMOVE_THRESHOLD=args.gt_iou
    UNLABELED_HIGH_CONF_THRESHOLD=args.unlabeled_high_confidence

    PREDICTIONS_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    COMBINED_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 72)
    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Images: {IMAGES_DIR}")
    print(
        f"Minimum prediction confidence: "
        f"{PRED_SCORE_THRESHOLD}"
    )
    print(
        f"Class-agnostic NMS threshold: "
        f"IoU > {CLASS_AGNOSTIC_NMS_IOU_THRESHOLD}"
    )
    print(
        f"GT-overlap removal threshold: "
        f"IoU > {IOU_REMOVE_THRESHOLD}"
    )
    print("=" * 72)

    image_paths = collect_image_paths(IMAGES_DIR)
    print(f"Found {len(image_paths)} images.")

    model = load_trained_model(
        CHECKPOINT_PATH,
        device,
    )

    # --------------------------------------------------------
    # 1) Generate predictions and remove very weak detections
    # --------------------------------------------------------
    confidence_filtered_predictions = generate_predictions(
        model=model,
        image_paths=image_paths,
        device=device,
        score_threshold=PRED_SCORE_THRESHOLD,
    )

    confidence_path = (
        PREDICTIONS_OUTPUT_DIR
        / "predictions_all_images_after_confidence_filter.csv"
    )
    save_csv(
        confidence_filtered_predictions,
        confidence_path,
    )

    # This is the first complete prediction file used as the source
    # for extracting rows listed in unlabeled_images.txt.
    initial_all_predictions_path = (
        PREDICTIONS_OUTPUT_DIR
        / "predictions_all_images.csv"
    )
    save_csv(
        confidence_filtered_predictions,
        initial_all_predictions_path,
    )

    print(
        f"\nFirst all-image prediction file "
        f"(before NMS and GT filtering):\n"
        f"{initial_all_predictions_path}"
    )
    print(
        f"Predictions after confidence filtering:\n"
        f"{confidence_path}"
    )
    print(
        f"Boxes after confidence filtering: "
        f"{len(confidence_filtered_predictions)}"
    )

    # --------------------------------------------------------
    # 2) Remove duplicate predictions using class-agnostic NMS
    # --------------------------------------------------------
    nms_predictions, nms_removed_predictions = (
        apply_class_agnostic_nms(
            predictions=confidence_filtered_predictions,
            iou_threshold=CLASS_AGNOSTIC_NMS_IOU_THRESHOLD,
        )
    )

    nms_kept_path = (
        PREDICTIONS_OUTPUT_DIR
        / "predictions_all_images_after_class_agnostic_nms.csv"
    )
    nms_removed_path = (
        PREDICTIONS_OUTPUT_DIR
        / "predictions_removed_by_class_agnostic_nms.csv"
    )

    save_csv(nms_predictions, nms_kept_path)
    save_csv(nms_removed_predictions, nms_removed_path)

    print(
        f"\nPredictions after class-agnostic NMS:\n"
        f"{nms_kept_path}"
    )
    print(
        f"Boxes removed by class-agnostic NMS: "
        f"{len(nms_removed_predictions)}"
    )
    print(
        f"Boxes remaining after NMS: "
        f"{len(nms_predictions)}"
    )

    # --------------------------------------------------------
    # 2.1) Extract unlabeled rows AFTER class-agnostic NMS
    #      Source: predictions_all_images_after_class_agnostic_nms.csv
    #      No GT filtering is applied to unlabeled images.
    # --------------------------------------------------------
    unlabeled_names = read_unlabeled_image_names(
        UNLABELED_IMAGES_LIST
    )
    unlabeled_keys = {
        filename_key(name)
        for name in unlabeled_names
    }

    unlabeled_nms_predictions_with_key = nms_predictions.copy()
    unlabeled_nms_predictions_with_key["_key"] = (
        unlabeled_nms_predictions_with_key["filename"].map(filename_key)
    )

    # All NMS-cleaned predictions for images in unlabeled_images.txt.
    unlabeled_all_after_nms = (
        unlabeled_nms_predictions_with_key[
            unlabeled_nms_predictions_with_key["_key"].isin(
                unlabeled_keys
            )
        ]
        .drop(columns=["_key"])
        .sort_values(
            by=["filename", "confidence"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    # Unlabeled images have no imported GT or EfficientTree labels.
    # Their exported boxes therefore come only from RetinaNet.
    unlabeled_all_after_nms["source"] = "retinanet_prediction"

    unlabeled_all_path = (
        PREDICTIONS_OUTPUT_DIR
        / "unlabeled_predictions_from_initial_all_predictions.csv"
    )
    save_csv(
        unlabeled_all_after_nms,
        unlabeled_all_path,
    )

    # High-confidence subset taken after NMS; no GT filtering is applied.
    unlabeled_high_conf = (
        unlabeled_all_after_nms[
            unlabeled_all_after_nms["confidence"]
            >= UNLABELED_HIGH_CONF_THRESHOLD
        ]
        .sort_values(
            by=["filename", "confidence"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    high_conf_token = threshold_token(UNLABELED_HIGH_CONF_THRESHOLD)
    unlabeled_predictions_path = (
        PREDICTIONS_OUTPUT_DIR
        / (
            f"unlabeled_predictions_confidence_ge_{high_conf_token}_"
            "from_initial_all_predictions.csv"
        )
    )
    save_csv(
        unlabeled_high_conf,
        unlabeled_predictions_path,
    )

    high_conf_image_names = sorted(
        unlabeled_high_conf["filename"]
        .drop_duplicates()
        .astype(str)
        .tolist(),
        key=natural_key,
    )

    high_conf_images_path = (
        PREDICTIONS_OUTPUT_DIR
        / f"unlabeled_images_with_prediction_confidence_ge_{high_conf_token}.txt"
    )
    with open(
        high_conf_images_path,
        "w",
        encoding="utf-8",
    ) as file:
        for filename in high_conf_image_names:
            file.write(f"{filename}\n")

    available_keys = {
        filename_key(path.name)
        for path in image_paths
    }
    missing_names = [
        name
        for name in unlabeled_names
        if filename_key(name) not in available_keys
    ]

    missing_names_path = (
        PREDICTIONS_OUTPUT_DIR
        / "unlabeled_images_missing_from_images_folder.txt"
    )
    with open(
        missing_names_path,
        "w",
        encoding="utf-8",
    ) as file:
        for filename in missing_names:
            file.write(f"{filename}\n")

    print(
        f"\nUnlabeled image names read: "
        f"{len(unlabeled_names)}"
    )
    print(
        f"All unlabeled prediction rows extracted after "
        f"class-agnostic NMS: "
        f"{len(unlabeled_all_after_nms)}"
    )
    print(
        f"All unlabeled prediction CSV:\n"
        f"{unlabeled_all_path}"
    )
    print(
        f"High-confidence unlabeled predictions "
        f"(confidence >= {UNLABELED_HIGH_CONF_THRESHOLD}): "
        f"{len(unlabeled_high_conf)}"
    )
    print(
        f"Unlabeled images having at least one "
        f"high-confidence prediction: "
        f"{len(high_conf_image_names)}"
    )
    print(
        f"High-confidence unlabeled prediction CSV "
        f"(after class-agnostic NMS):\n"
        f"{unlabeled_predictions_path}"
    )
    print(
        f"High-confidence unlabeled image TXT:\n"
        f"{high_conf_images_path}"
    )
    print(
        f"Listed unlabeled images missing from image folder: "
        f"{len(missing_names)}"
    )

    confidence_with_key = confidence_filtered_predictions.copy()
    confidence_with_key["_key"] = (
        confidence_with_key["filename"].map(filename_key)
    )

    nms_with_key = nms_predictions.copy()
    nms_with_key["_key"] = (
        nms_with_key["filename"].map(filename_key)
    )

    nms_removed_with_key = nms_removed_predictions.copy()
    nms_removed_with_key["_key"] = (
        nms_removed_with_key["filename"].map(filename_key)
    )

    split_gt = {}
    all_split_keys = set()
    summary_rows = []

    # --------------------------------------------------------
    # 3) Read GT files and save NMS-cleaned predictions by split
    # --------------------------------------------------------
    for split_name, label_base_path in LABEL_FILES.items():
        raw_gt, resolved_label_path = read_table(
            label_base_path
        )
        gt = standardize_gt_dataframe(
            raw_gt,
            resolved_label_path,
        )

        split_gt[split_name] = gt
        split_keys = set(gt["_key"].unique())
        all_split_keys.update(split_keys)

        split_after_confidence = (
            confidence_with_key[
                confidence_with_key["_key"].isin(split_keys)
            ]
            .drop(columns=["_key"])
            .reset_index(drop=True)
        )

        split_after_nms = (
            nms_with_key[
                nms_with_key["_key"].isin(split_keys)
            ]
            .drop(columns=["_key"])
            .reset_index(drop=True)
        )

        split_removed_by_nms = (
            nms_removed_with_key[
                nms_removed_with_key["_key"].isin(split_keys)
            ]
            .drop(columns=["_key"])
            .reset_index(drop=True)
        )

        save_csv(
            split_after_confidence,
            PREDICTIONS_OUTPUT_DIR
            / f"predictions_{split_name}_after_confidence_filter.csv",
        )
        save_csv(
            split_after_nms,
            PREDICTIONS_OUTPUT_DIR
            / f"predictions_{split_name}_after_class_agnostic_nms.csv",
        )
        save_csv(
            split_removed_by_nms,
            PREDICTIONS_OUTPUT_DIR
            / f"predictions_{split_name}_removed_by_nms.csv",
        )

        print(
            f"\n{split_name.upper()} labels: "
            f"{resolved_label_path}"
        )

    # Save NMS-cleaned predictions for images outside train/val/test.
    other_predictions = (
        nms_with_key[
            ~nms_with_key["_key"].isin(all_split_keys)
        ]
        .drop(columns=["_key"])
        .reset_index(drop=True)
    )

    save_csv(
        other_predictions,
        PREDICTIONS_OUTPUT_DIR
        / "predictions_images_outside_train_val_test.csv",
    )

    # --------------------------------------------------------
    # 4) Remove NMS-cleaned predictions overlapping GT
    # 5) Combine remaining predictions with GT
    # --------------------------------------------------------
    for split_name, gt in split_gt.items():
        split_keys = set(gt["_key"].unique())

        split_after_confidence = (
            confidence_with_key[
                confidence_with_key["_key"].isin(split_keys)
            ]
            .drop(columns=["_key"])
            .reset_index(drop=True)
        )

        split_after_nms = (
            nms_with_key[
                nms_with_key["_key"].isin(split_keys)
            ]
            .drop(columns=["_key"])
            .reset_index(drop=True)
        )

        split_removed_by_nms = (
            nms_removed_with_key[
                nms_removed_with_key["_key"].isin(split_keys)
            ]
            .drop(columns=["_key"])
            .reset_index(drop=True)
        )

        kept_predictions, removed_predictions = (
            filter_predictions_against_gt(
                predictions=split_after_nms,
                gt=gt,
                iou_threshold=IOU_REMOVE_THRESHOLD,
            )
        )

        kept_path = (
            COMBINED_OUTPUT_DIR
            / f"{split_name}_predictions_final_candidates.csv"
        )
        removed_path = (
            COMBINED_OUTPUT_DIR
            / f"{split_name}_predictions_removed_by_gt_iou.csv"
        )

        save_csv(kept_predictions, kept_path)
        save_csv(removed_predictions, removed_path)

        combined = build_combined_gt_and_predictions(
            gt=gt,
            kept_predictions=kept_predictions,
        )

        combined_path = (
            COMBINED_OUTPUT_DIR
            / f"{split_name}_gt_plus_filtered_predictions.csv"
        )
        save_csv(combined, combined_path)

        summary_rows.append({
            "split": split_name,
            "images_in_gt": int(gt["_key"].nunique()),
            "gt_boxes": int(len(gt)),
            "predictions_after_confidence_filter": int(
                len(split_after_confidence)
            ),
            "removed_by_class_agnostic_nms": int(
                len(split_removed_by_nms)
            ),
            "predictions_after_nms": int(
                len(split_after_nms)
            ),
            "removed_by_gt_iou_gt_0p50": int(
                len(removed_predictions)
            ),
            "final_prediction_candidates": int(
                len(kept_predictions)
            ),
            "combined_rows_gt_plus_predictions": int(
                len(combined)
            ),
            "minimum_prediction_confidence": (
                PRED_SCORE_THRESHOLD
            ),
            "class_agnostic_nms_iou_threshold": (
                CLASS_AGNOSTIC_NMS_IOU_THRESHOLD
            ),
            "gt_iou_remove_threshold": (
                IOU_REMOVE_THRESHOLD
            ),
        })

        print("\n" + "-" * 72)
        print(f"Split: {split_name}")
        print(f"GT boxes: {len(gt)}")
        print(
            f"After confidence filter: "
            f"{len(split_after_confidence)}"
        )
        print(
            f"Removed by class-agnostic NMS: "
            f"{len(split_removed_by_nms)}"
        )
        print(
            f"After class-agnostic NMS: "
            f"{len(split_after_nms)}"
        )
        print(
            f"Removed because IoU with GT > "
            f"{IOU_REMOVE_THRESHOLD}: "
            f"{len(removed_predictions)}"
        )
        print(
            f"Final prediction candidates: "
            f"{len(kept_predictions)}"
        )
        print(
            f"Final GT + prediction file:\n"
            f"{combined_path}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path = (
        COMBINED_OUTPUT_DIR
        / "filtering_summary.csv"
    )
    save_csv(summary_df, summary_path)

    settings_path = (
        COMBINED_OUTPUT_DIR
        / "prediction_settings.txt"
    )

    with open(
        settings_path,
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            f"checkpoint={CHECKPOINT_PATH}\n"
            f"images_dir={IMAGES_DIR}\n"
            f"minimum_prediction_confidence="
            f"{PRED_SCORE_THRESHOLD}\n"
            f"unlabeled_images_list="
            f"{UNLABELED_IMAGES_LIST}\n"
            f"unlabeled_high_confidence_threshold="
            f"{UNLABELED_HIGH_CONF_THRESHOLD}\n"
            f"unlabeled_export_source="
            f"predictions_all_images_after_class_agnostic_nms.csv\n"
            f"class_agnostic_nms_iou_threshold="
            f"{CLASS_AGNOSTIC_NMS_IOU_THRESHOLD}\n"
            f"gt_iou_remove_threshold="
            f"{IOU_REMOVE_THRESHOLD}\n"
            f"nms_class_agnostic=True\n"
            f"nms_keep_rule=higher_confidence_box\n"
            f"input_label_source_rule=confidence_1_ground_truth_else_input_pseudo_label\n"
            f"unlabeled_box_source=retinanet_prediction_only\n"
            f"gt_remove_rule=max_iou_with_any_gt>"
            f"{IOU_REMOVE_THRESHOLD}\n"
            f"class_ignored_during_gt_filtering=True\n"
        )

    print("\n" + "=" * 72)
    print("PROCESS COMPLETE")
    print(
        f"Prediction audit files:\n"
        f"{PREDICTIONS_OUTPUT_DIR}"
    )
    print(
        f"Final split files:\n"
        f"{COMBINED_OUTPUT_DIR}"
    )
    print(f"Summary:\n{summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
