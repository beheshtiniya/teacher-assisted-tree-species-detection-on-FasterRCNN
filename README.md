
# Teacher-Assisted Tree Species Detection

This repository contains the software, configuration files, and
reproducibility pipeline used for tree-species object detection in
heterogeneous forests using aerial RGB imagery.

The study combines supervised and semi-supervised object-detection
workflows, including EfficientTree/EfficientTeacher, RetinaNet,
multi-source prediction fusion, warm-start Teacher--Student
semi-supervised learning, annotation expansion, and downstream
Faster R-CNN evaluation.

The repository supports the experimental workflow reported across four
experiments, including the development of the Best A3 annotation-expansion
strategy in Experiment 3 and its transfer to Dataset 3 in Experiment 4.

> **Important:** The image dataset is not included in this software
> repository and must be downloaded separately from Zenodo:
> https://doi.org/10.5281/zenodo.21385214

---

## Associated data and reproducibility resources

### Main aerial-image dataset

The aerial RGB imagery, expert annotations, unlabeled imagery, and
Dataset 1--2 resources are publicly available at:

**Dataset for Tree Species Detection in Heterogeneous Forests Using
Aerial RGB Imagery**

**DOI:**  
https://doi.org/10.5281/zenodo.21385214

The original pre-annotated dataset introduced in the source study is
available at:

https://doi.org/10.5281/zenodo.7528566

### Dataset 2 reconstruction resources

Resources used to reconstruct Dataset 2 are archived at:

https://zenodo.org/records/22013937

### Dataset 3 partitioning resources

The image-level MILP partitioning resources used to construct Dataset 3
are archived at:

https://zenodo.org/records/22013323

### Best A3 results from Experiments 3 and 4

The annotation-expanded Best A3 datasets, downstream Faster R-CNN
results, and reproducibility materials associated with Experiments 3
and 4 are available at:

https://zenodo.org/uploads/22063524

This archive contains materials associated with:

- Best A3 obtained in Experiment 3 using Dataset 2;
- transfer of the same Best A3 pathway to Dataset 3 in Experiment 4;
- repeated Faster R-CNN evaluation;
- run-level evaluation outputs;
- annotation-expanded training data; and
- Faster R-CNN reproducibility resources.

### Complete software archive

The complete tree-species detection and multi-source pseudo-label
refinement software archive is available at:

https://doi.org/10.5281/zenodo.21639908

### Confidence-threshold pseudo-labeling implementation

The confidence-threshold pseudo-labeling implementation used in the
earlier experimental stage is available separately at:

https://github.com/beheshtiniya/pseudo-labeling-confidence-thresholds-tree-species-identification

---

## Experimental overview

The study was organized into four main experiments.

### Experiment 1 -- Sensitivity to dataset construction

Experiment 1 evaluated supervised Faster R-CNN performance under three
alternative dataset constructions derived from the labeled image
collection.

Datasets 1, 2, and 3 use different image allocations and test
partitions. Their results are therefore interpreted as a
dataset-partition sensitivity analysis rather than as a controlled
ranking of partitioning strategies.

### Experiment 2 -- Single-model pseudo-label expansion

Experiment 2 investigated whether predictions from a trained detector
could be used to complete or refine training annotations.

The experiment included confidence-threshold analysis and examined how
pseudo-label selection affected downstream Faster R-CNN performance.

The corresponding implementation is available in the separate
confidence-threshold repository listed above.

### Experiment 3 -- Multi-source annotation refinement and Best A3

Experiment 3 was performed using Dataset 2 and progressively constructed
the annotation-expanded datasets A0--A3.

The principal stages were:

1. train supervised EfficientTree;
2. train RetinaNet;
3. generate predictions from the complementary detectors;
4. construct the native-confidence A2 annotation set through
   multi-source prediction fusion;
5. retrain EfficientTree using the expanded A2 labeled dataset;
6. use the selected A2-trained EfficientTree checkpoint to warm-start
   Teacher--Student semi-supervised learning;
7. use the A2 expanded annotations as the labeled/source data and the
   additional unlabeled image collection as the target data during SSL;
8. generate SSL-enhanced predictions;
9. perform the final A3 refinement; and
10. train Faster R-CNN on the resulting annotation-expanded dataset.

The strongest Experiment 3 configuration was:

`A3_native,native`

or equivalently:

`A3_{\mathrm{native},\mathrm{native}}`

This configuration is referred to throughout the study as **Best A3**.

Conceptually, the selected pathway was:

```text
Dataset 2 expert annotations
        |
        +--> RetinaNet
        |
        +--> supervised EfficientTree
                    |
                    v
          native-confidence fusion
                    |
                    v
               A2_native
                    |
                    v
         retrain EfficientTree
                    |
                    v
          selected checkpoint
                    |
                    v
     warm-start Teacher--Student SSL
       labeled/source = A2_native
       target = unlabeled images
                    |
                    v
        SSL-enhanced EfficientTree
                    |
                    v
            final A3 refinement
                    |
                    v
         A3_native,native
                    |
                    v
             Faster R-CNN
````

---

## Experiment 4 -- Transfer of Best A3 to Dataset 3

Experiment 4 tested whether the Best A3 pathway identified in
Experiment 3 remained effective after transfer to Dataset 3.

The upstream annotation-expansion procedure was reproduced under the
Dataset 3 construction rather than replacing it with a different
pseudo-labeling strategy.

The main pathway was:

```text
Dataset 3 expert annotations
        |
        +--> RetinaNet
        |
        +--> supervised EfficientTree
                    |
                    v
          native-confidence fusion
                    |
                    v
               A2_native
                    |
                    v
         retrain EfficientTree
                    |
                    v
          selected checkpoint
                    |
                    v
     warm-start Teacher--Student SSL
       labeled/source = A2_native
       target = unlabeled images
                    |
                    v
        SSL-enhanced EfficientTree
                    |
                    v
            final A3 refinement
                    |
                    v
         A3_native,native
```

Two downstream Faster R-CNN configurations were then evaluated.

### Best A3 without final unlabeled inclusion

The final Faster R-CNN training set contained:

* the original Dataset 3 expert annotations; and
* accepted Best A3 pseudo-labels generated for the labeled training
  images.

Accepted pseudo-labels originating from the additional unlabeled-image
collection were not added to the final Faster R-CNN training set.

### Best A3 with final unlabeled inclusion

The same upstream Best A3 pathway was used, but accepted predictions from
the additional unlabeled-image collection were also added to the final
Faster R-CNN training set.

> **Important:** Both Experiment 4 configurations use the unlabeled-image
> collection during the upstream Teacher--Student SSL stage.
> The difference between the two configurations concerns only whether
> accepted predictions from those unlabeled images are included in the
> final Faster R-CNN training dataset.

This distinction is important for correctly reproducing Experiment 4.

---

## Experimental datasets

The annotation-expansion workflow uses the following dataset notation:

* **A0:** expert annotations only.
* **A1:** expert annotations plus accepted predictions from an initial
  auxiliary detector configuration.
* **A2:** expert annotations plus accepted multi-source predictions from
  EfficientTree and RetinaNet.
* **A2_native:** the native-confidence A2 configuration selected as the
  source for the subsequent Best A3 pathway.
* **A3:** annotation-expanded data obtained after the warm-start
  Teacher--Student refinement stage.
* **A3_native,native:** the Best A3 configuration identified in
  Experiment 3 and subsequently transferred to Dataset 3 in Experiment 4.

Expert annotations have priority throughout the expansion process and
are not replaced by model predictions.

Validation and test annotations remain separated from the training
annotation-expansion pipeline.

---

## Prediction fusion and annotation preservation

For the main Best A3 workflow, predictions are filtered before being
added to the expert annotations.

The principal reproducibility settings are:

```text
Prediction confidence threshold:        0.25
GT-overlap IoU threshold:                0.50
Prediction duplicate-removal IoU:        0.50
Faster R-CNN evaluation score threshold: 0.25
Detection-matching IoU threshold:        0.50
```

For labeled images, predictions overlapping an expert annotation by more
than IoU 0.50 are removed before pseudo-label fusion.

Remaining predictions are processed using class-agnostic greedy
duplicate removal. A lower-confidence prediction is suppressed only
when its IoU with a retained prediction exceeds 0.50.

The ground-truth-overlap filtering stage is not applied to the
additional unlabeled-image collection because expert boxes are not
available for those images.

These settings describe the Best A3 reproducibility pathway and should
not be confused with the separate confidence-threshold analyses performed
in Experiment 2.

---

## Faster R-CNN training and evaluation

The principal downstream comparisons use Faster R-CNN with a
ResNet-50-FPN backbone.

The main repeated-training configuration is:

```text
Independent runs:            5
Random seeds:                42--46
Maximum epochs:              20
Batch size:                  4
Optimizer:                   SGD
Initial learning rate:       0.005
Momentum:                    0.9
Weight decay:                0.0005
Learning-rate scheduler:     StepLR
Scheduler step size:         5 epochs
Scheduler gamma:             0.1
Early-stopping patience:     5
Minimum improvement:         0.0001
Checkpoint selection:        validation mAP
Confusion-matrix threshold:  0.25
Detection-matching IoU:      0.50
```

Each run is trained independently.

The best checkpoint for each run is selected using validation mAP.
The test partition is evaluated only after model training and checkpoint
selection.

The 0.25 score threshold applies to confusion-matrix-based evaluation.
mAP computation uses the detector outputs independently of this
confusion-matrix score threshold.

Run-level outputs may include:

* selected model checkpoints;
* validation statistics;
* test metrics;
* confusion matrices;
* class-level precision, recall, and F1-score;
* macro-averaged metrics; and
* run summaries used to calculate mean and sample standard deviation.

---

## Expected project layout

After extracting the software and downloading the dataset, a typical
directory structure is:

```text
Teacher_Assisted_Tree_Detection/
├── EfficientTree-master/
├── config/
├── docs/
├── src/
│   └── fasterrcnn/
├── stages/
├── tools/
├── configure_paths.cmd
├── configure_paths.py
├── preflight.cmd
├── run_stage.cmd
├── run_all.cmd
├── requirements.txt
└── environment.yml

DATA_ROOT/
├── images/
└── labels/
    ├── train_labels.csv
    ├── val_labels.csv
    ├── test_labels.csv
    └── unlabeled_images.txt
```

The dataset may be stored anywhere on the user's computer and does not
need to be copied into the software repository.

Generated predictions, annotation-expanded datasets, model checkpoints,
and experiment outputs should be written to separate output directories.

---

## Path configuration

This repository does not require public releases to contain
machine-specific absolute paths.

Users can configure their local dataset and Python paths with:

```text
configure_paths.cmd "D:\path\to\dataset" "C:\path\to\python.exe"
```

This creates:

```text
config/paths.local.json
```

The generated `paths.local.json` contains machine-specific local paths
and should not be committed to GitHub or included in a public software
archive.

A public template is provided in:

```text
config/paths.example.json
```

---

## Pipeline organization

The reproducibility workflow is divided into stages so that individual
components can be executed and inspected separately.

```text
01  Preflight checks
02  Prepare the EfficientTree dataset
03  Generate EfficientTree configuration files
04  Run initial smoke tests
05  Run pseudo-label diagnostics
06  Train supervised EfficientTree
07  Build the expert-only A0 dataset
08  Generate supervised EfficientTree predictions
09  Train RetinaNet
10  Generate RetinaNet predictions
11  Fuse predictions and construct A2_native
12  Retrain EfficientTree using A2_native
13  Warm-start Teacher--Student SSL from the A2-trained checkpoint
14  Generate SSL-enhanced predictions
15  Construct A3_native,native
16  Train and evaluate Faster R-CNN
17  Summarize repeated-run results
```

Depending on the experiment, the dataset configuration supplied to these
stages determines whether the Experiment 3/Dataset 2 or
Experiment 4/Dataset 3 pathway is reproduced.

A single stage can be run with:

```text
run_stage.cmd 02
```

The complete configured workflow can be run with:

```text
run_all.cmd
```

Users should inspect local configuration files, run the preflight checks,
and verify input/output paths before beginning computationally expensive
training.

---

## Coordinate and class conventions

The original annotation CSV files use:

```text
filename,class,xmin,ymin,xmax,ymax
```

Class identifiers in the original CSV annotations are:

```text
1, 2, 3, 4
```

EfficientTree uses YOLO-style zero-based class identifiers:

```text
0, 1, 2, 3
```

The preparation scripts perform the required class mapping.

The original annotation coordinate system uses 256 x 256 pixel imagery.
EfficientTree data are prepared at 640 x 640 pixels when required, and
predictions are converted back to the original coordinate system for
annotation fusion and downstream processing.

---

## EfficientTree / EfficientTeacher implementation

The EfficientTree component used in this study was based on the public
implementation released by Hou et al.:

[https://github.com/houbr233/EfficientTree](https://github.com/houbr233/EfficientTree)

The present repository contains the study-specific configuration,
data-preparation, prediction-processing, fusion, and downstream
reproducibility workflow required to integrate EfficientTree into the
tree-species annotation-expansion experiments.

---

## Software requirements

The main Python dependencies are listed in:

```text
requirements.txt
environment.yml
```

GPU training requires:

* a compatible NVIDIA GPU;
* a compatible CUDA installation;
* PyTorch with CUDA support; and
* sufficient storage for prepared datasets, predictions, checkpoints,
  and repeated-run outputs.

Exact CUDA and PyTorch builds may depend on the user's hardware and
software environment.

---

## Important reproducibility notes

* Original images and expert annotations should remain unchanged.
* Generated annotation-expanded datasets should be written to separate
  output directories.
* Expert annotations have priority over model-generated annotations.
* Validation and test annotations must remain separate from the training
  annotation-expansion process.
* Validation results should be used for model/checkpoint selection.
* Held-out test results should not be used to tune pseudo-label
  thresholds.
* Machine-specific path files should not be committed to the public
  repository.
* Intermediate prediction files and audit tables should be retained when
  reproducing the full annotation-expansion pathway.
* The Experiment 4 `without final unlabeled inclusion` condition still
  uses unlabeled imagery during the upstream Teacher--Student SSL stage.
* Experiment 3 and Experiment 4 should use separate output directories
  to prevent accidental reuse or overwriting of intermediate artifacts.

---

## Reproducibility scope

Different public resources support different levels of reproduction.

### Downstream Best A3 Faster R-CNN reproduction

Users interested in reproducing only the final Best A3 Faster R-CNN
training and evaluation can use the released Best A3 annotation-expanded
datasets together with the Faster R-CNN training code.

The corresponding Experiment 3 and Experiment 4 resources are available
at:

[https://zenodo.org/uploads/22063524](https://zenodo.org/uploads/22063524)

### Complete Best A3 annotation-generation reproduction

Users wishing to regenerate Best A3 from the original expert annotations
must additionally reproduce the upstream stages:

```text
RetinaNet
  +
supervised EfficientTree
  ↓
A2_native fusion
  ↓
EfficientTree retraining on A2_native
  ↓
warm-start Teacher--Student SSL
  ↓
SSL-enhanced predictions
  ↓
A3_native,native
```

The source-code repository and complete software archive provide the
corresponding implementation resources.

---

## Data and code availability

The software, data, and dataset-partitioning resources used in this study
are publicly available through the following repositories and archives.

**Main GitHub repository**

[https://github.com/beheshtiniya/teacher-assisted-tree-species-detection-on-FasterRCNN](https://github.com/beheshtiniya/teacher-assisted-tree-species-detection-on-FasterRCNN)

**Confidence-threshold pseudo-labeling implementation**

[https://github.com/beheshtiniya/pseudo-labeling-confidence-thresholds-tree-species-identification](https://github.com/beheshtiniya/pseudo-labeling-confidence-thresholds-tree-species-identification)

**Complete multi-source refinement software archive**

[https://doi.org/10.5281/zenodo.21639908](https://doi.org/10.5281/zenodo.21639908)

**Aerial imagery, expert annotations, unlabeled imagery, and Dataset 1--2 resources**

[https://doi.org/10.5281/zenodo.21385214](https://doi.org/10.5281/zenodo.21385214)

**Original pre-annotated dataset**

[https://doi.org/10.5281/zenodo.7528566](https://doi.org/10.5281/zenodo.7528566)

**Dataset 2 reconstruction resources**

[https://zenodo.org/records/22013937](https://zenodo.org/records/22013937)

**Dataset 3 image-level MILP partitioning resources**

[https://zenodo.org/records/22013323](https://zenodo.org/records/22013323)

**Best A3 Experiment 3 and Experiment 4 results and reproducibility resources**

[https://zenodo.org/uploads/22063524](https://zenodo.org/uploads/22063524)

**EfficientTree source implementation**

[https://github.com/houbr233/EfficientTree](https://github.com/houbr233/EfficientTree)

---

## Dataset citation

Please cite the associated dataset when using the imagery or expert
annotations:

> Dataset for Tree Species Detection in Heterogeneous Forests Using
> Aerial RGB Imagery. Zenodo.
> [https://doi.org/10.5281/zenodo.21385214](https://doi.org/10.5281/zenodo.21385214)

Users relying on Dataset 2 or Dataset 3 reconstruction should also cite
the corresponding reconstruction or partitioning archive where
appropriate.

---

## Software citation

Please cite the archived Zenodo software record associated with the
version of the software used in your study:

[https://doi.org/10.5281/zenodo.21639908](https://doi.org/10.5281/zenodo.21639908)

When using the released Best A3 Experiment 3 or Experiment 4 datasets,
results, or downstream Faster R-CNN reproducibility materials, please
also cite the corresponding Best A3 archive:

[https://zenodo.org/uploads/22063524](https://zenodo.org/uploads/22063524)

---

## License

The software license is provided in the `LICENSE` file.

Datasets and external software components may be distributed under
separate licenses. Users should consult the corresponding Zenodo records
and upstream repositories before redistributing or reusing those
materials.

```

چند تغییر مهمی که نسبت به README قبلی اعمال کردم: Experiment 3 دیگر صرفاً «fusion + SSL» معرفی نشده و مسیر واقعی `A2_native → retrain ET → warm-start SSL → A3_native,native` مشخص شده؛ Experiment 4 نیز صریحاً به‌عنوان **انتقال Best A3 از Dataset 2 به Dataset 3** معرفی شده است. همچنین تفاوت `with/without unlabeled` طوری نوشته شده که کسی تصور نکند حالت without-unlabeled در مرحله SSL از تصاویر بدون برچسب استفاده نکرده است.

یک نکته هم درباره لینک `https://zenodo.org/uploads/22063524`: چون این آدرس را خودت برای رکورد نتایج دادی، در متن دقیقاً همان را گذاشتم. وقتی رکورد Zenodo نهایی و منتشر شد، اگر آدرس آن به فرم `https://zenodo.org/records/...` یا DOI تبدیل شد، **بهتر است همه این موارد را با URL/DOI نهایی رکورد جایگزین کنی**؛ برای citation پایدارتر است.
```
