import pandas as pd
from pathlib import Path


# =====================================================
# PATHS
# =====================================================

GT_CSV = r"E:\FASTRCNN\teacher_student\labels\article2_hemogen_mydataset\train_labels.csv"


# RetinaNet RAW predictions after confidence filtering
RETINA_CSV = r"E:\FASTRCNN\teacher_student\tatd_outputs\predictions\retinanet\predictions\predictions_train_after_confidence_filter.csv"


# EfficientTree supervised D3
ET_SUP_CSV = r"E:\FASTRCNN\teacher_student\tatd_outputs\predictions\efficientteacher_supervised\train_predictions_256.csv"


# EfficientTree SSL D3
ET_SSL_CSV = r"E:\FASTRCNN\teacher_student\tatd_outputs\predictions\efficientteacher_ssl_D3_ep20\train_predictions_256.csv"



OUT_DIR = Path(
    r"E:\FASTRCNN\teacher_student\tatd_outputs\experiments\exp4_stage_fusion"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CONF_THRESHOLD = 0.25
GT_IOU_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.50



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

def filter_by_gt(pred_df, gt_df):

    keep=[]
    removed=0


    gt_groups = dict(tuple(gt_df.groupby("filename")))


    for _,p in pred_df.iterrows():

        filename=p["filename"]

        pred_box=[
            p["xmin"],
            p["ymin"],
            p["xmax"],
            p["ymax"]
        ]


        max_iou=0


        if filename in gt_groups:

            for _,g in gt_groups[filename].iterrows():

                gt_box=[
                    g["xmin"],
                    g["ymin"],
                    g["xmax"],
                    g["ymax"]
                ]


                max_iou=max(
                    max_iou,
                    IoU(pred_box,gt_box)
                )


        if max_iou > GT_IOU_THRESHOLD:
            removed += 1

        else:
            keep.append(p)


    return pd.DataFrame(keep), removed



# =====================================================
# CLASS AGNOSTIC NMS
# =====================================================

def class_agnostic_nms(df):

    result=[]


    for filename,group in df.groupby("filename"):


        boxes = group.sort_values(
            "confidence",
            ascending=False
        ).to_dict("records")


        while boxes:

            best=boxes.pop(0)

            result.append(best)


            remain=[]


            b1=[
                best["xmin"],
                best["ymin"],
                best["xmax"],
                best["ymax"]
            ]


            for b in boxes:

                b2=[
                    b["xmin"],
                    b["ymin"],
                    b["xmax"],
                    b["ymax"]
                ]


                if IoU(b1,b2) <= NMS_IOU_THRESHOLD:
                    remain.append(b)


            boxes=remain


    return pd.DataFrame(result)



# =====================================================
# LOAD
# =====================================================

gt=pd.read_csv(GT_CSV)

retina=pd.read_csv(RETINA_CSV)

et_sup=pd.read_csv(ET_SUP_CSV)

et_ssl=pd.read_csv(ET_SSL_CSV)



retina["source"]="retinanet"
et_sup["source"]="et_supervised"
et_ssl["source"]="et_ssl"



print("\nINPUT BOXES")
print("----------------")
print("GT:",len(gt))
print("Retina:",len(retina))
print("ET supervised:",len(et_sup))
print("ET SSL:",len(et_ssl))



# =====================================================
# STAGE 1
# RetinaNet + ET supervised
# =====================================================


retina = retina[
    retina["confidence"] >= CONF_THRESHOLD
]


et_sup = et_sup[
    et_sup["confidence"] >= CONF_THRESHOLD
]


stage1_input=pd.concat(
    [
        retina,
        et_sup
    ],
    ignore_index=True
)


stage1_filtered, removed_stage1 = filter_by_gt(
    stage1_input,
    gt
)


stage1_final = class_agnostic_nms(
    stage1_filtered
)



stage1_final.to_csv(
    OUT_DIR/"stage1_retina_et_supervised.csv",
    index=False
)



print("\n======================")
print("STAGE 1")
print("======================")

print("Before GT filtering:",len(stage1_input))
print("Removed by GT:",removed_stage1)
print("After NMS:",len(stage1_final))

print(
    stage1_final["source"].value_counts()
)



# =====================================================
# STAGE 2
# Add SSL
# =====================================================


et_ssl = et_ssl[
    et_ssl["confidence"] >= CONF_THRESHOLD
]


ssl_filtered, removed_ssl = filter_by_gt(
    et_ssl,
    gt
)


stage2_input=pd.concat(
    [
        stage1_final,
        ssl_filtered
    ],
    ignore_index=True
)


final_predictions = class_agnostic_nms(
    stage2_input
)



final_predictions.to_csv(
    OUT_DIR/"final_exp4_D3_predictions.csv",
    index=False
)



print("\n======================")
print("FINAL FUSION")
print("======================")

print("Stage1 boxes:",len(stage1_final))

print("SSL removed by GT:",removed_ssl)

print("Before final NMS:",len(stage2_input))

print("Final predictions:",len(final_predictions))


print("\nSOURCE CONTRIBUTION")
print(
    final_predictions["source"].value_counts()
)


print("\nCLASS DISTRIBUTION")
print(
    final_predictions["class"].value_counts().sort_index()
)



# =====================================================
# FINAL DATASET FOR Faster R-CNN
# =====================================================


gt_final=gt.copy()

gt_final["confidence"]=1.0
gt_final["source"]="expert"



train_final=pd.concat(
    [
        gt_final,
        final_predictions
    ],
    ignore_index=True
)



train_final.to_csv(
    OUT_DIR/"train_final_fused_with_GT.csv",
    index=False
)



print("\n======================")
print("FINAL DATASET")
print("======================")

print("Expert GT boxes:",len(gt_final))

print("Added predictions:",len(final_predictions))

print("Total training boxes:",len(train_final))