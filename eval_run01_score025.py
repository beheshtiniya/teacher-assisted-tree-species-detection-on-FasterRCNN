import os
import importlib.util
import pandas as pd
import torch
from torch.utils.data import DataLoader

TRAIN_SCRIPT = r"E:\FASTRCNN\FASTRCNN\Teacher_Assisted_Tree_Detection\src\fasterrcnn\train_fasterrcnn.py"
IMAGES_DIR = r"E:\FASTRCNN\teacher_student\images"
TEST_CSV = r"E:\FASTRCNN\teacher_student\labels\article2_hemogen_mydataset\test_labels.csv"
RUN_DIR = r"E:\FASTRCNN\teacher_student\tatd_outputs\models\fasterrcnn\exp4_D3_A3_native_native\run_01"
BEST_MODEL = os.path.join(RUN_DIR, "best_model.pth")

IOU_THRESHOLD = 0.50
SCORE_THRESHOLD = 0.25
BATCH_SIZE = 4
WORKERS = 0

def load_train_module(path):
    spec = importlib.util.spec_from_file_location("frcnn_train_module", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def main():
    if not os.path.isfile(BEST_MODEL):
        raise FileNotFoundError(f"best_model.pth not found:\n{BEST_MODEL}")
    if not os.path.isfile(TEST_CSV):
        raise FileNotFoundError(f"test CSV not found:\n{TEST_CSV}")

    fr = load_train_module(TRAIN_SCRIPT)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Checkpoint:", BEST_MODEL)
    print("Test CSV:", TEST_CSV)
    print("IoU threshold:", IOU_THRESHOLD)
    print("Score threshold:", SCORE_THRESHOLD)

    test_dataset = fr.TreeDetectionDataset(img_dir=IMAGES_DIR, csv_file=TEST_CSV)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKERS,
        collate_fn=fr.collate_fn,
    )

    print("Test images:", len(test_dataset))

    model = fr.get_model(num_classes=5)
    state = torch.load(BEST_MODEL, map_location=device)
    model.load_state_dict(state)
    model.to(device)

    test_metrics, conf_mat = fr.evaluate_map_and_confusion(
        model=model,
        data_loader=test_loader,
        device=device,
        iou_threshold=IOU_THRESHOLD,
        score_threshold=SCORE_THRESHOLD,
    )

    class_names = [f"class{i}" for i in range(5)]
    conf_df = pd.DataFrame(conf_mat, index=class_names, columns=class_names)

    rows = []
    for c in range(1, 5):
        tp = float(conf_mat[c, c])

        # Match the metric table you showed:
        # precision excludes background-row unmatched predictions.
        precision_den = float(conf_mat[1:5, c].sum())
        recall_den = float(conf_mat[c, :].sum())

        precision = tp / precision_den if precision_den > 0 else 0.0
        recall = tp / recall_den if recall_den > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({
            "class": f"class{c}",
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

    metrics_df = pd.DataFrame(rows)
    macro_p = float(metrics_df["precision"].mean())
    macro_r = float(metrics_df["recall"].mean())
    macro_f1 = float(metrics_df["f1_score"].mean())

    correct = float(sum(conf_mat[c, c] for c in range(1, 5)))
    gt_total = float(conf_mat[1:5, :].sum())
    accuracy = correct / gt_total if gt_total > 0 else 0.0

    out_conf = os.path.join(RUN_DIR, "confusion_matrix_score025.csv")
    out_metrics = os.path.join(RUN_DIR, "metrics_score025.csv")
    out_map = os.path.join(RUN_DIR, "test_map_score025.csv")

    conf_df.to_csv(out_conf)
    metrics_df.to_csv(out_metrics, index=False)

    scalar_map = {k: v for k, v in test_metrics.items() if not isinstance(v, list)}
    pd.DataFrame([scalar_map]).to_csv(out_map, index=False)

    print("\n=== CONFUSION MATRIX @ score >= 0.25 ===")
    print(conf_df.to_string())

    print("\n=== CLASS METRICS (same style as your table) ===")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== MACRO / ACCURACY ===")
    print(f"Macro Precision = {macro_p:.4f}")
    print(f"Macro Recall    = {macro_r:.4f}")
    print(f"Macro F1        = {macro_f1:.4f}")
    print(f"Accuracy        = {accuracy:.4f}")

    print("\nSaved:")
    print(out_conf)
    print(out_metrics)
    print(out_map)

if __name__ == "__main__":
    main()
