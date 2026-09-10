from pathlib import Path
import json
import pandas as pd

A2_CSV = Path(
    r"E:\FASTRCNN\teacher_student\tatd_outputs\experiments"
    r"\exp4_D3_A2_native\train_A2_native.csv"
)

LABELED_PSEUDO_CSV = Path(
    r"E:\FASTRCNN\teacher_student\tatd_outputs\experiments"
    r"\exp4_D3_A3_native_native\A3_native_native_labeled_predictions.csv"
)

OUT_DIR = Path(
    r"E:\FASTRCNN\teacher_student\tatd_outputs\experiments"
    r"\exp4_D3_A3_native_native_without_unlabeled"
)

OUT_CSV = OUT_DIR / "train_A3_native_native_without_unlabeled.csv"
OUT_SUMMARY = OUT_DIR / "summary.json"

TRAIN_COLS = ["filename", "class", "xmin", "ymin", "xmax", "ymax"]

def main():
    if not A2_CSV.is_file():
        raise FileNotFoundError(A2_CSV)
    if not LABELED_PSEUDO_CSV.is_file():
        raise FileNotFoundError(LABELED_PSEUDO_CSV)

    a2 = pd.read_csv(A2_CSV)
    lp = pd.read_csv(LABELED_PSEUDO_CSV)

    required_a2 = TRAIN_COLS + ["source"]
    for c in required_a2:
        if c not in a2.columns:
            raise ValueError(f"A2 missing column: {c}")

    for c in TRAIN_COLS:
        if c not in lp.columns:
            raise ValueError(f"Labeled pseudo CSV missing column: {c}")

    expert = a2[a2["source"].astype(str).str.lower() == "expert"].copy()

    if len(expert) != 6078:
        raise RuntimeError(f"Expected 6078 expert GT rows, found {len(expert)}")

    if len(lp) != 7450:
        raise RuntimeError(f"Expected 7450 accepted labeled pseudo rows, found {len(lp)}")

    final = pd.concat(
        [expert[TRAIN_COLS], lp[TRAIN_COLS]],
        ignore_index=True
    )

    if len(final) != 13528:
        raise RuntimeError(f"Expected 13528 final rows, found {len(final)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final.to_csv(OUT_CSV, index=False)

    class_counts = {
        str(int(k)): int(v)
        for k, v in final["class"].value_counts().sort_index().to_dict().items()
    }

    summary = {
        "configuration": "A3_native,native WITHOUT unlabeled",
        "expert_gt_rows": int(len(expert)),
        "accepted_labeled_pseudo_rows": int(len(lp)),
        "final_rows": int(len(final)),
        "unique_images": int(final["filename"].nunique()),
        "class_counts": class_counts,
        "output_csv": str(OUT_CSV),
    }

    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== WITHOUT-UNLABELED BUILD COMPLETE ===")
    print("Expert GT                 :", len(expert))
    print("Labeled pseudo            :", len(lp))
    print("FINAL rows                :", len(final))
    print("FINAL unique images       :", final["filename"].nunique())
    print("Class counts:")
    for k, v in class_counts.items():
        print(f"  C{k}: {v}")
    print("\nCSV:")
    print(OUT_CSV)
    print("\nSummary:")
    print(OUT_SUMMARY)

if __name__ == "__main__":
    main()
