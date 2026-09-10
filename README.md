# Teacher-Assisted Tree Species Detection

This repository contains the software, configuration files, processing
utilities, and reproducibility pipeline developed for tree-species object
detection in heterogeneous dense forests using high-resolution aerial RGB
imagery.

The study follows a data-centric object-detection framework in which
Faster R-CNN is retained as the common downstream detector. The complete
workflow combines supervised detection, pseudo-label generation,
confidence-based selection, multi-source prediction fusion, EfficientTree
retraining, warm-start Teacher–Student semi-supervised learning (SSL),
annotation expansion, and repeated downstream Faster R-CNN evaluation.

Four progressive experiments are supported. These experiments examine
dataset-construction sensitivity, single-model pseudo-labeling,
multi-source annotation refinement, development of the **Best A3**
configuration on Dataset 2, and transfer of the same refinement pathway
to Dataset 3.

> **Important:** The aerial imagery and associated datasets are not
> included directly in this GitHub repository. They must be downloaded
> separately from the Zenodo resources listed below.

---

## Associated Data and Reproducibility Resources

### Original Pre-Annotated Dataset

The original pre-annotated tree-species dataset used as the source
collection in this study is publicly available at:

https://doi.org/10.5281/zenodo.7528566

The released labeled collection used in the present experiments contains
4,438 aerial RGB image patches and 10,128 expert-provided bounding boxes
for four tree species.

Datasets 1, 2, and 3 used in this study are alternative
train–validation–test constructions derived from this same labeled
collection. They are not independently collected datasets.

---

### Dataset 1 and Additional Unlabeled Imagery

Dataset 1, including the labeled aerial RGB imagery, expert annotations,
and the associated additional unlabeled-image collection, is available at:

**Dataset for Tree Species Detection in Heterogeneous Forests Using
Aerial RGB Imagery**

https://doi.org/10.5281/zenodo.21385214

---

### Dataset 2 Reconstruction Resources

Resources used to reconstruct the Dataset 2 train–validation–test
configuration are archived at:

https://zenodo.org/records/22013937

---

### Dataset 3 Partitioning Resources

The image-level mixed-integer linear programming (MILP) resources used to
construct Dataset 3 are archived at:

https://zenodo.org/records/22013323

Dataset 3 uses a fixed image-level 60/20/20 train–validation–test
construction with class-composition balancing performed through MILP.

---

### Best A3 Reproducibility Package

The annotation-expanded datasets, downstream Faster R-CNN results, and
reproducibility materials associated with the selected Best A3 pathway in
Experiments 3 and 4 are available at:

https://zenodo.org/records/22063524

The archive includes materials associated with:

- Best A3 developed on Dataset 2 in Experiment 3;
- transfer of the same Best A3 pathway to Dataset 3 in Experiment 4;
- annotation-expanded training data;
- repeated downstream Faster R-CNN evaluations;
- run-level evaluation outputs; and
- Faster R-CNN training and evaluation resources used for reproducibility.

---

### Complete Software Archive

A complete archived release of the tree-species detection and multi-source
pseudo-label refinement software is available at:

https://doi.org/10.5281/zenodo.21639908

---

### Confidence-Threshold Pseudo-Labeling Repository

The confidence-threshold pseudo-labeling implementation used in
Experiment 2 is maintained separately at:

https://github.com/beheshtiniya/pseudo-labeling-confidence-thresholds-tree-species-identification

---

### EfficientTree Source Implementation

The EfficientTree component used in this study was based on the public
implementation released by Hou et al.:

https://github.com/houbr233/EfficientTree

The present repository contains the study-specific data preparation,
configuration, prediction processing, fusion, SSL integration, and
downstream reproducibility workflow required to use EfficientTree within
the tree-species annotation-expansion experiments.

---

# Experimental Overview

The study is organized into four progressive experiments. Faster R-CNN
serves as the common downstream detector in the main comparisons, allowing
changes in detection performance to be interpreted primarily in relation
to dataset construction and training-annotation refinement rather than a
change in the final detector architecture.

---

## Experiment 1 — Sensitivity to Dataset Construction

Experiment 1 evaluates supervised Faster R-CNN under three alternative
train–validation–test constructions derived from the same complete
labeled collection.

Datasets 1, 2, and 3 differ in image allocation, subset size, class
composition, and test-set composition. Cross-dataset differences are
therefore interpreted as **dataset-construction sensitivity** rather than
as a controlled ranking of partitioning strategies.

Resources:

- Dataset 1 and additional unlabeled imagery:  
  https://doi.org/10.5281/zenodo.21385214

- Dataset 2 reconstruction resources:  
  https://zenodo.org/records/22013937

- Dataset 3 MILP partitioning resources:  
  https://zenodo.org/records/22013323

- Original pre-annotated source dataset:  
  https://doi.org/10.5281/zenodo.7528566

---

## Experiment 2 — Single-Model Pseudo-Label Expansion

Experiment 2 investigates whether predictions from a trained Faster R-CNN
detector can be used to complete or expand training annotations.

Two preliminary studies are included:

1. annotation completion on pre-existing labeled training images; and
2. confidence-threshold analysis using a separate unlabeled image pool.

The second study evaluates multiple prediction-confidence thresholds and
examines how pseudo-label selection affects downstream Faster R-CNN
performance.

The corresponding implementation is available at:

https://github.com/beheshtiniya/pseudo-labeling-confidence-thresholds-tree-species-identification

The threshold settings used in Experiment 2 are specific to that
experiment and should not be confused with the pseudo-label and
Faster R-CNN evaluation thresholds used in Experiments 3 and 4.

---

## Experiment 3 — Multi-Source Annotation Refinement and Best A3

Experiment 3 is performed on Dataset 2 and progressively constructs the
annotation-expanded configurations A0–A3.

The principal workflow is:

1. train supervised EfficientTree;
2. train RetinaNet;
3. generate predictions from both auxiliary detectors;
4. construct the native-confidence A2 annotation set through multi-source
   prediction fusion;
5. retrain EfficientTree using the expanded A2 labeled dataset;
6. select the A2-trained EfficientTree checkpoint;
7. use the selected checkpoint to warm-start Teacher–Student SSL;
8. use A2 expanded annotations as labeled/source data and the additional
   unlabeled-image collection as target data during SSL;
9. generate SSL-enhanced EfficientTree predictions;
10. perform the final A3 refinement; and
11. train and evaluate Faster R-CNN on the resulting expanded annotation
    set.

The selected Experiment 3 configuration is:


A3_native,native

or equivalently:

A3_{native,native}

This configuration is referred to throughout the study as Best A3.

Conceptually, the selected pathway is:

Dataset 2 expert annotations
        |
        +------------------> RetinaNet
        |
        +------------------> supervised EfficientTree
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
                 warm-start Teacher–Student SSL
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

The corresponding Best A3 reproducibility package is available at:

https://zenodo.org/records/22063524

Experiment 4 — Transfer of Best A3 to Dataset 3

Experiment 4 evaluates whether the Best A3 pathway identified in
Experiment 3 remains effective after transfer to Dataset 3.

The upstream annotation-expansion procedure is reproduced under the
Dataset 3 construction rather than replaced by a different
pseudo-labeling strategy.

The principal pathway is:

Dataset 3 expert annotations
        |
        +------------------> RetinaNet
        |
        +------------------> supervised EfficientTree
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
                 warm-start Teacher–Student SSL
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

Two downstream Faster R-CNN configurations are evaluated.

Best A3 Without Final Unlabeled Inclusion

The final Faster R-CNN training set contains:

the original Dataset 3 expert annotations; and
accepted Best A3 model-generated annotations for the labeled Dataset 3
training images.

Accepted pseudo-labels originating from the additional unlabeled-image
collection are not added to the final Faster R-CNN training set.

Importantly, the additional unlabeled imagery is still used during the
upstream Teacher–Student SSL stage.

Best A3 With Final Unlabeled Inclusion

This configuration uses the same upstream Best A3 pathway, but accepted
SSL-enhanced EfficientTree predictions generated on the additional
unlabeled-image collection are also incorporated into the final
Faster R-CNN training set.

Important: Both Experiment 4 configurations use the additional
unlabeled-image collection during the upstream Teacher–Student SSL
stage. The difference concerns only whether accepted predictions
originating from those unlabeled images are included in the final
Faster R-CNN training dataset.

The corresponding reproducibility resources are available at:

https://zenodo.org/records/22063524

Experimental Dataset Notation

The annotation-expansion workflow uses the following notation:

A0: expert annotations only.
A1: expert annotations plus accepted predictions from one auxiliary
prediction source.
A1–ET: expert annotations plus accepted supervised EfficientTree
predictions.
A1–RetinaNet: expert annotations plus accepted RetinaNet
predictions.
A2: expert annotations plus accepted predictions obtained through
multi-source EfficientTree and RetinaNet fusion.
A2_native: the native-confidence A2 configuration used as the
warm-start source for the selected Best A3 pathway.
A3: annotation-expanded data obtained after warm-start
Teacher–Student refinement.
A3_native,native: the selected Best A3 configuration developed on
Dataset 2 and transferred to Dataset 3.

Expert annotations have priority throughout the main annotation-expansion
workflow and are not replaced by model-generated predictions.

Validation and test annotations remain separate from the training
annotation-expansion process.

Prediction Fusion and Annotation Preservation

For the main Best A3 workflow, model-generated predictions are filtered
before being combined with the expert annotations.

The principal native-confidence annotation-refinement settings are:

Candidate prediction confidence threshold:    0.25
GT-overlap IoU threshold:                      0.50
Prediction duplicate-removal IoU threshold:   0.50
Detection-matching IoU threshold:              0.50

The 0.25 candidate-prediction threshold refers to pseudo-label
generation and annotation fusion. It is distinct from the downstream
Faster R-CNN confusion-matrix evaluation threshold.

For labeled images, predictions are compared class-agnostically with the
available expert annotations. Predictions whose overlap with an expert
annotation satisfies

IoU > 0.50

are removed.

An overlap exactly equal to 0.50 is retained.

After ground-truth-aware filtering, the remaining predictions are pooled
image-wise and processed using greedy class-agnostic duplicate removal.

A lower-confidence prediction is suppressed only when its IoU with an
already retained prediction satisfies

IoU > 0.50

An IoU exactly equal to 0.50 is retained.

For additional unlabeled images, ground-truth-aware filtering is not
applied because expert bounding boxes are unavailable. These predictions
undergo confidence filtering and prediction-to-prediction duplicate
removal instead.

Faster R-CNN Training

The main downstream experiments use Faster R-CNN with a pretrained
ResNet-50-FPN backbone and a prediction head configured for four tree
species plus background.

The principal repeated-training configuration is:

Independent runs:             5
Random seeds:                 42–46
Maximum epochs:               20
Batch size:                   4
Optimizer:                    SGD
Initial learning rate:        0.005
Momentum:                     0.9
Weight decay:                 0.0005
Learning-rate scheduler:      StepLR
Scheduler step size:          5 epochs
Scheduler gamma:              0.1
Early-stopping patience:      5
Minimum improvement:          0.0001
Checkpoint selection:         validation mAP
Detection-matching IoU:       0.50

Each Faster R-CNN configuration is initialized and trained independently.
Weights and optimizer states are not transferred between downstream
configurations.

Validation mAP is computed during training, and the checkpoint with the
highest validation mAP is retained for each run. The held-out test set is
evaluated only after training and checkpoint selection.

Faster R-CNN Evaluation Thresholds

The confusion-matrix score threshold is experiment-specific and should not
be confused with the 0.25 confidence threshold used for candidate
pseudo-label selection in the native-confidence annotation-refinement
pipeline.

Experiment 3 Main Evaluation
Confusion-matrix score threshold: 0.50
Detection-matching IoU:           0.50

Best A3 was additionally examined using a threshold-sensitivity analysis
at:

0.25
0.50

The score threshold affects confusion-matrix-based Precision, Recall,
F1-score, and Accuracy. It does not affect model optimization,
validation-mAP-based checkpoint selection, or mAP computation.

Experiment 4 Evaluation
Confusion-matrix score threshold: 0.25
Detection-matching IoU:           0.50

mAP computation is performed independently of the confusion-matrix score
threshold.

Running the Repository
1. Configure Local Paths

Before running the pipeline, configure the local dataset location and
Python executable.

On Windows:

configure_paths.cmd "D:\path\to\dataset" "C:\path\to\python.exe"

The corresponding Python implementation is:

configure_paths.py

This creates a local configuration file such as:

config/paths.local.json

Machine-specific path files should remain local and should not be
committed to the public repository.

A public template is provided as:

config/paths.example.json
2. Install the Environment

A Conda environment can be created using:

conda env create -f environment.yml

Alternatively, dependencies can be installed into an existing compatible
Python environment using:

pip install -r requirements.txt

GPU-based training requires a compatible NVIDIA GPU, CUDA installation,
and CUDA-enabled PyTorch build.

3. Run Preflight Checks

Before starting model training or pseudo-label generation, run:

preflight.cmd

The preflight stage should be used to verify the local Python
environment, configured paths, input datasets, and required files before
computationally expensive stages are started.

4. Run an Individual Pipeline Stage

The main Windows stage launcher is:

run_stage.cmd <stage_number>

For example:

run_stage.cmd 02

The underlying Python stage runner is:

run_stage.py

The reproducibility workflow is organized as follows:

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
13  Warm-start Teacher–Student SSL from the A2-trained checkpoint
14  Generate SSL-enhanced predictions
15  Construct A3_native,native
16  Train and evaluate Faster R-CNN
17  Summarize repeated-run results

The selected dataset configuration determines whether the Dataset 2 /
Experiment 3 or Dataset 3 / Experiment 4 pathway is reproduced.

5. Run the Complete Configured Pipeline

After path configuration and successful preflight checks, the complete
configured workflow can be launched with:

run_all.cmd

Because the full pipeline includes computationally expensive detector
training and repeated evaluations, users should verify all local paths,
dataset selections, and output directories before running the complete
workflow.

Important Experiment 4 Files

Several repository files are specifically associated with reproduction or
evaluation of Experiment 4.

run_experiment4_best_full_v2.py

This script is associated with execution of the complete transferred
Best A3 workflow on Dataset 3.

Its workflow includes the Experiment 4 sequence from auxiliary prediction
generation and A2 construction through EfficientTree retraining,
warm-start Teacher–Student SSL, final A3 construction, and downstream
Faster R-CNN processing.

Users reproducing the complete Experiment 4 workflow should ensure that
the Dataset 3 paths and Experiment 4-specific output directories are
configured before execution.

run_stage_exp4.py

This is an Experiment 4-specific stage runner.

It is intended for stage-by-stage execution or resumption of the
Experiment 4 workflow rather than execution of the general Dataset 2
pipeline.

Experiment 3 and Experiment 4 should use separate output directories to
avoid accidental reuse or overwriting of intermediate predictions,
annotations, and checkpoints.

exp4_on_dataset3_run_stage.py

This script supports stage-oriented execution of the Experiment 4
workflow under the Dataset 3 configuration.

It should be treated as an Experiment 4 / Dataset 3 workflow utility
rather than the general repository entry point.

Users should verify Dataset 3 input paths and Experiment 4 output paths
before running it.

build_A3_native_native_without_unlabeled_D3.py

This script constructs the Dataset 3 annotation-expanded training
configuration corresponding to:

A3_native,native — without final unlabeled inclusion

The resulting final Faster R-CNN training configuration contains:

Dataset 3 expert annotations; and
accepted Best A3 predictions associated with the labeled Dataset 3
training images.

Accepted predictions originating from the additional unlabeled-image
collection are not included in this final Faster R-CNN training set.

This does not mean that unlabeled imagery is excluded from the
complete Best A3 pipeline. The unlabeled collection is still used
during upstream warm-start Teacher–Student SSL.

eval_run01_score025.py

This is an evaluation utility associated with evaluation of a trained
Faster R-CNN run at a confusion-matrix score threshold of 0.25.

It should not be interpreted as a training script.

The score threshold affects confusion-matrix-based Precision, Recall,
F1-score, and Accuracy. It does not define the pseudo-label fusion
threshold and does not alter mAP computation.

Experiment 3 uses 0.50 for its principal reported confusion-matrix
evaluation and additionally examines 0.25 in the threshold-sensitivity
analysis. Experiment 4 uses the corresponding 0.25 evaluation setting
for the Dataset 3 runs.

Supporting Repository Files
environment.yml

Conda environment specification used to define the software environment.

requirements.txt

Python dependency list for installation into an existing compatible
environment.

configure_paths.py

Python implementation of the local path-configuration utility.

configure_paths.cmd

Recommended Windows wrapper for configuring the dataset and Python paths.

origin_configure_paths.py

An additional/original path-configuration helper retained in the
repository.

For normal public reproduction, users should generally use the current
configuration entry points:

configure_paths.cmd
configure_paths.py

unless specific reproduction instructions require the original helper.

preflight.cmd

Runs preflight checks before the main computational pipeline.

run_stage.cmd

Windows launcher for running an individual pipeline stage.

run_stage.py

Python stage-runner implementation used by the staged pipeline.

run_all.cmd

Runs the complete configured pipeline.

pseudo_labeling_results.xlsx

A results workbook associated with pseudo-labeling analyses. This file is
not an executable component of the pipeline.

train_explain.md

Training-related documentation intended for reference rather than direct
execution.

Expected Project Layout

A typical local structure is:

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
├── run_stage.py
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

The dataset may be stored anywhere on the user's computer and does not
need to be copied into the software repository.

Generated predictions, expanded annotations, model checkpoints, and
experiment outputs should be written to separate output directories.

Coordinate and Class Conventions

The original annotation CSV files use:

filename,class,xmin,ymin,xmax,ymax

The original class identifiers are:

1  Norway spruce   (Picea abies)
2  Silver fir      (Abies alba)
3  Scots pine      (Pinus sylvestris)
4  European beech  (Fagus sylvatica)

EfficientTree uses YOLO-style zero-based class identifiers:

0, 1, 2, 3

The preparation scripts perform the required class mapping.

The original detection patches use a 256 × 256 pixel coordinate system.

EfficientTree data may be prepared at 640 × 640 pixels when required,
and predictions are converted back to the original coordinate system
before annotation fusion and downstream Faster R-CNN processing.

Recommended Execution Order

For a new installation, the recommended workflow is:

1. Download the required dataset and reproducibility resources.
2. Create or activate the required Python environment.
3. Configure local paths with configure_paths.cmd.
4. Run preflight.cmd.
5. Verify dataset and output paths.
6. Test individual stages using run_stage.cmd.
7. Run the complete workflow with run_all.cmd when the staged checks pass.

For Experiment 4, use the Experiment 4-specific runner or stage utilities
after completing the common repository configuration.

Important Reproducibility Notes
Original images and expert annotations should remain unchanged.
Expert annotations have priority over model-generated predictions.
Generated annotation-expanded datasets should be written to separate
output directories.
Validation and test annotations must remain separate from training
annotation expansion.
Validation results should be used for model and checkpoint selection.
Held-out test results should not be used to tune pseudo-label
thresholds.
Machine-specific path files should not be committed to the public
repository.
Intermediate prediction files and audit outputs should be retained when
reproducing the complete refinement pathway.
Experiment 3 and Experiment 4 should use separate output directories.
The Experiment 4 without final unlabeled inclusion configuration
still uses unlabeled imagery during upstream Teacher–Student SSL.
The difference between the Experiment 4 with- and without-final-unlabeled
configurations concerns only the final Faster R-CNN training dataset.
Cross-dataset comparisons should be interpreted descriptively because
Datasets 1, 2, and 3 contain different validation and test image
allocations.
Direct performance comparisons are most appropriate between
configurations evaluated on the same fixed dataset-specific test
partition.
Reproducibility Scope
Downstream Best A3 Faster R-CNN Reproduction

Users interested only in reproducing the final Best A3 Faster R-CNN
training and evaluation can use the released Best A3 annotation-expanded
training data together with the Faster R-CNN training and evaluation code.

Resources:

https://zenodo.org/records/22063524

Complete Best A3 Annotation-Generation Reproduction

Users wishing to regenerate Best A3 from the original expert annotations
must reproduce the complete upstream annotation-refinement pipeline:

RetinaNet
   +
supervised EfficientTree
   |
   v
A2_native fusion
   |
   v
EfficientTree retraining on A2_native
   |
   v
warm-start Teacher–Student SSL
   |
   v
SSL-enhanced EfficientTree predictions
   |
   v
final A3 refinement
   |
   v
A3_native,native

The main GitHub repository and the complete Zenodo software archive
provide the corresponding implementation resources.

Data and Code Availability

Main GitHub repository

https://github.com/beheshtiniya/teacher-assisted-tree-species-detection-on-FasterRCNN

Complete software archive

https://doi.org/10.5281/zenodo.21639908

Confidence-threshold pseudo-labeling implementation

https://github.com/beheshtiniya/pseudo-labeling-confidence-thresholds-tree-species-identification

Dataset 1 and additional unlabeled imagery

https://doi.org/10.5281/zenodo.21385214

Original pre-annotated dataset

https://doi.org/10.5281/zenodo.7528566

Dataset 2 reconstruction resources

https://zenodo.org/records/22013937

Dataset 3 MILP partitioning resources

https://zenodo.org/records/22013323

Best A3 Experiment 3 and Experiment 4 reproducibility resources

https://zenodo.org/records/22063524

EfficientTree source implementation

https://github.com/houbr233/EfficientTree

Dataset Citation

When using the associated imagery or expert annotations, please cite the
corresponding dataset record:

Dataset for Tree Species Detection in Heterogeneous Forests Using Aerial
RGB Imagery. Zenodo.
https://doi.org/10.5281/zenodo.21385214

The original pre-annotated source dataset is available at:

https://doi.org/10.5281/zenodo.7528566

Users relying on Dataset 2 or Dataset 3 reconstruction resources should
also cite the corresponding archive where appropriate:

Dataset 2:

https://zenodo.org/records/22013937

Dataset 3:

https://zenodo.org/records/22013323

Software Citation

Please cite the archived Zenodo software record corresponding to the
software release used in the study:

https://doi.org/10.5281/zenodo.21639908

When using the released Best A3 Experiment 3 or Experiment 4 datasets,
run-level results, or downstream Faster R-CNN reproducibility materials,
please also cite:

https://zenodo.org/records/22063524

License

The software license is provided in the LICENSE file.

Datasets and external software components may be distributed under
separate licenses. Users should consult the corresponding Zenodo records
and upstream repositories before redistributing or reusing those
materials.
