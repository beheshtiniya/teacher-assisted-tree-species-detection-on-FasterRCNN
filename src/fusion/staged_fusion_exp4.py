from pathlib import Path
import argparse
import pandas as pd


# =====================================================
# IOU
# =====================================================

def IoU(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2-x1) * max(0, y2-y1)

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = area1 + area2 - inter

    if union <= 0:
        return 0

    return inter / union



# =====================================================
# REMOVE OVERLAP WITH GT
# =====================================================

def filter_by_gt(pred_df, gt_df, threshold):

    keep = []
    removed = 0

    gt_groups = dict(tuple(gt_df.groupby("filename")))


    for _, p in pred_df.iterrows():

        filename = p["filename"]

        pred_box = [
            p["xmin"],
            p["ymin"],
            p["xmax"],
            p["ymax"]
        ]

        max_iou = 0


        if filename in gt_groups:

            for _, g in gt_groups[filename].iterrows():

                gt_box = [
                    g["xmin"],
                    g["ymin"],
                    g["xmax"],
                    g["ymax"]
                ]

                max_iou = max(
                    max_iou,
                    IoU(pred_box, gt_box)
                )


        if max_iou > threshold:
            removed += 1
        else:
            keep.append(p)


    return pd.DataFrame(keep), removed



# =====================================================
# CLASS AGNOSTIC NMS
# =====================================================

def class_agnostic_nms(df, threshold):

    result = []


    for filename, group in df.groupby("filename"):

        boxes = group.sort_values(
            "confidence",
            ascending=False
        ).to_dict("records")


        while boxes:

            best = boxes.pop(0)

            result.append(best)


            remain = []


            b1 = [
                best["xmin"],
                best["ymin"],
                best["xmax"],
                best["ymax"]
            ]


            for b in boxes:

                b2 = [
                    b["xmin"],
                    b["ymin"],
                    b["xmax"],
                    b["ymax"]
                ]


                if IoU(b1, b2) <= threshold:
                    remain.append(b)


            boxes = remain


    return pd.DataFrame(result)



# =====================================================
# MAIN
# =====================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument("--gt-train", required=True)

    parser.add_argument("--retina-train", required=True)

    parser.add_argument("--et-supervised-train", required=True)

    parser.add_argument("--et-ssl-train", required=True)

    parser.add_argument("--output-dir", required=True)


    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25
    )

    parser.add_argument(
        "--gt-iou",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5
    )


    args = parser.parse_args()


    OUT_DIR = Path(args.output_dir)
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------
    # Load
    # -----------------------------

    gt = pd.read_csv(args.gt_train)

    retina = pd.read_csv(args.retina_train)

    et_sup = pd.read_csv(args.et_supervised_train)

    et_ssl = pd.read_csv(args.et_ssl_train)



    retina["source"] = "retinanet"
    et_sup["source"] = "et_supervised"
    et_ssl["source"] = "et_ssl"



    print("\nINPUT")
    print("----------------")
    print("GT:", len(gt))
    print("Retina:", len(retina))
    print("ET supervised:", len(et_sup))
    print("ET SSL:", len(et_ssl))



    # =================================================
    # STAGE 1
    # Retina + ET supervised
    # =================================================


    retina = retina[
        retina["confidence"] >= args.confidence
    ]

    et_sup = et_sup[
        et_sup["confidence"] >= args.confidence
    ]


    stage1_input = pd.concat(
        [
            retina,
            et_sup
        ],
        ignore_index=True
    )


    stage1_filtered, removed1 = filter_by_gt(
        stage1_input,
        gt,
        args.gt_iou
    )


    stage1_final = class_agnostic_nms(
        stage1_filtered,
        args.nms_iou
    )


    stage1_final.to_csv(
        OUT_DIR/"stage1_retina_et_supervised.csv",
        index=False
    )


    print("\nSTAGE 1")
    print("----------------")
    print("Before:", len(stage1_input))
    print("GT removed:", removed1)
    print("After NMS:", len(stage1_final))



    # =================================================
    # STAGE 2
    # + ET SSL
    # =================================================


    et_ssl = et_ssl[
        et_ssl["confidence"] >= args.confidence
    ]


    ssl_filtered, removed_ssl = filter_by_gt(
        et_ssl,
        gt,
        args.gt_iou
    )


    stage2_input = pd.concat(
        [
            stage1_final,
            ssl_filtered
        ],
        ignore_index=True
    )


    final_predictions = class_agnostic_nms(
        stage2_input,
        args.nms_iou
    )


    final_predictions.to_csv(
        OUT_DIR/"final_exp4_D3_predictions.csv",
        index=False
    )


    print("\nFINAL FUSION")
    print("----------------")
    print("Stage1:", len(stage1_final))
    print("SSL removed:", removed_ssl)
    print("Before NMS:", len(stage2_input))
    print("Final:", len(final_predictions))


    print(
        final_predictions["source"].value_counts()
    )



    # =================================================
    # FINAL Faster R-CNN DATASET
    # =================================================


    gt_final = gt.copy()

    gt_final["confidence"] = 1.0
    gt_final["source"] = "expert"


    train_final = pd.concat(
        [
            gt_final,
            final_predictions
        ],
        ignore_index=True
    )


    train_final.to_csv(
        OUT_DIR/"train_final_fused.csv",
        index=False
    )


    print("\nFINAL DATASET")
    print("----------------")
    print("GT:", len(gt_final))
    print("Predictions:", len(final_predictions))
    print("Total:", len(train_final))



if __name__ == "__main__":
    main()