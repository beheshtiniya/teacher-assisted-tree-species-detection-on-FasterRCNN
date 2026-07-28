
# Teacher-Assisted Tree Species Detection

This repository contains the software, configuration files, and reproducibility
pipeline used for tree-species object detection in heterogeneous forests using
aerial RGB imagery.

The project combines supervised and semi-supervised object-detection workflows,
including EfficientTree/EfficientTeacher, RetinaNet, dual-teacher prediction
fusion, warm-start semi-supervised learning, and Faster R-CNN evaluation.

> **Important:** The dataset is not included in this software archive and must
> be downloaded separately from:
> https://doi.org/10.5281/zenodo.21385214

---

## Associated dataset

The aerial RGB imagery and expert annotations are publicly available on Zenodo:

**Dataset for Tree Species Detection in Heterogeneous Forests Using Aerial RGB Imagery**

**DOI:** https://doi.org/10.5281/zenodo.21385214

Users must download the dataset separately before running the pipeline.

---

## Expected project layout

After extracting the software and downloading the dataset, the recommended
directory structure is:

```text
Teacher_Assisted_Tree_Detection/
├── EfficientTree-master/
├── config/
├── docs/
├── src/
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

The dataset may be stored anywhere on the user's computer. It does not need to
be copied into the software repository.

---

## Path configuration

This repository does not contain machine-specific absolute paths.

Users configure their local dataset and Python paths by running:

```bat
configure_paths.cmd "D:\path\to\dataset" "C:\path\to\python.exe"
```

This command creates:

```text
config/paths.local.json
```

The generated `paths.local.json` file contains local computer paths and should
not be committed to GitHub or included in a Zenodo software release.

A public configuration template is provided in:

```text
config/paths.example.json
```

---

## Pipeline organization

The workflow is divided into numbered stages so that each part can be executed
and checked independently.

```text
01  Preflight checks
02  Prepare the 640×640 EfficientTree dataset
03  Generate EfficientTree configuration files
04  Run the initial smoke test
05  Run the pseudo-label diagnostic smoke test
06  Train supervised EfficientTree
07  Build the expert-only A0 dataset
08  Generate supervised EfficientTree predictions and build A1
09  Train RetinaNet
10  Generate RetinaNet predictions
11  Fuse model predictions and build A2
12  Build the dual-teacher warm-start dataset
13  Train warm-start semi-supervised EfficientTree
14  Generate SSL predictions and build A3
15  Train and evaluate Faster R-CNN
16  Summarize the final results
```

A single stage can be run with:

```bat
run_stage.cmd 02
```

The complete workflow can be run with:

```bat
run_all.cmd
```

Users should first inspect each stage, verify the configuration, and run the
preflight checks before starting computationally expensive training.

---

## Experimental datasets

The software organizes the experimental training datasets as follows:

- **A0:** expert annotations only.
- **A1:** expert annotations plus accepted predictions from the supervised
  EfficientTree model.
- **A2:** expert annotations plus accepted fused predictions from
  EfficientTree and RetinaNet.
- **A3:** the final teacher-assisted dataset incorporating accepted predictions
  from the warm-start semi-supervised EfficientTree workflow.

Validation and test annotations remain separated from training data. They must
not be automatically promoted to training pseudo-labels.

---

## Coordinate and class conventions

The original annotation CSV files use:

```text
filename,class,xmin,ymin,xmax,ymax
```

Class identifiers in the original CSV files are:

```text
1, 2, 3, 4
```

EfficientTree uses YOLO-style class identifiers:

```text
0, 1, 2, 3
```

The data-preparation scripts perform this mapping automatically.

The original annotation coordinate system is 256×256 pixels. EfficientTree
images are prepared at 640×640 pixels, and model predictions are converted back
to the original coordinate system when required.

---

## Main filtering parameters

The default reproducibility configuration uses:

```text
EfficientTree minimum confidence: 0.25
RetinaNet minimum confidence: 0.25
Model-agreement IoU: 0.50
Single-model minimum confidence: 0.80
NMS IoU: 0.50
GT-overlap filtering IoU: 0.50
Warm-start pseudo-label minimum confidence: 0.50
```

These parameters should not be changed after examining the held-out test
results.

---

## Software requirements

The main Python dependencies are listed in:

```text
requirements.txt
environment.yml
```

GPU training requires a compatible NVIDIA GPU, CUDA installation, PyTorch
installation, and sufficient storage for prepared images, model checkpoints,
and experiment outputs.

Exact CUDA and PyTorch versions may depend on the user's hardware.

---

## Important reproducibility notes

- The original images and expert annotations must remain unchanged.
- Generated datasets and predictions should be written to separate output
  directories.
- Validation and test data must not be used to tune thresholds after final
  evaluation.
- Model checkpoints must be selected using validation results only.
- Machine-specific files such as `config/paths.local.json`, caches, checkpoints,
  and generated outputs are not part of the public source-code archive.
- Users should retain audit CSV files generated by the pipeline.

---

## Dataset citation

Please cite the associated dataset when using the imagery or annotations:

> Dataset for Tree Species Detection in Heterogeneous Forests Using Aerial RGB
> Imagery. Zenodo. https://doi.org/10.5281/zenodo.21385214

---

## Software citation

Please cite the archived Zenodo software record associated with the specific
version of this repository. The software DOI should be added here after the
software record is published.

---

## License

The software license is provided in the `LICENSE` file.

The dataset may have a separate license. Users must consult the Zenodo dataset
record before downloading, redistributing, or reusing the data.

---
**Dataset for Tree Species Detection in Heterogeneous Forests Using Aerial RGB Imagery**

**DOI:** https://doi.org/10.5281/zenodo.21385214





# Repeated Training and Evaluation Pipeline using Faster R-CNN

This script implements a repeated training pipeline for object detection using **Faster R-CNN** on a tree species dataset. The goal is to evaluate the stability and consistency of model performance across multiple independent runs.

---

## 🎯 Purpose

To train the same model architecture multiple times from scratch, allowing comparison of performance metrics across runs. This setup helps analyze the robustness of the training process and the effects of random initialization and data shuffling.

---

## 🔁 Overview of Training Loop

- The script performs **10 independent training runs**.
- Each run trains a new Faster R-CNN model using the same training and validation datasets.
- The best-performing model (based on validation loss) is saved separately for each run.
- After each training session, a test-time evaluation is triggered by calling a separate script: `test_evaluate.py`.

---

## 🧠 Key Features

| Feature | Description |
|--------|-------------|
| **Model** | `Faster R-CNN with ResNet-50 FPN` backbone |
| **Loss Function** | Multi-component object detection loss (inherent in torchvision's Faster R-CNN) |
| **Optimizer** | Stochastic Gradient Descent (SGD) with momentum and weight decay |
| **Early Stopping** | Stops training after 3 epochs without improvement in validation loss |
| **Repeatability** | Training is repeated 10 times (`run_1` to `run_10`) to assess model stability |
| **Checkpointing** | Each run saves the best model as `checkpoints/run_X/best_model.pth` |

---

## 📁 Input Requirements

- `train_labels.csv`: Ground-truth annotations for training images
- `val_labels.csv`: Ground-truth annotations for validation images
- `images_rename/`: Directory containing all input images (referenced by filename)
- `test_evaluate.py`: Script to evaluate each saved model on a test set

---

📂 Dataset Access
The weakly labeled and original ground-truth datasets used in this evaluation are publicly available at the following link:

🔗 https://yun.ir/9b88b8


## 📤 Output

- **Model Checkpoints**: Saved in `checkpoints/run_X/` (10 folders for 10 runs)
- **Printed Logs**: Training loss and validation loss per epoch
- **Test Evaluation**: Each best model is automatically passed to `test_evaluate.py` for performance analysis

---

## 🔧 Configuration

- Batch size: 1
- Max epochs per run: 10
- Early stopping patience: 3 epochs
- Number of classes: Automatically inferred from training labels

---

## 📈 Use Case

This script is ideal for:
- Evaluating model generalizability across different random seeds
- Selecting the most consistent or highest-performing model
- Feeding all models into a later ensemble or comparative analysis

---

## 📝 Note

Make sure `test_evaluate.py` is properly configured to load the model checkpoints and evaluate them accordingly. The dataset CSV files must include the following columns:  
`filename`, `xmin`, `ymin`, `xmax`, `ymax`, `class`

---

## 📄 License

MIT License
