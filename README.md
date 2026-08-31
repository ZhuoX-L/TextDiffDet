# TextDiffDet

**TextDiffDet: A Pre-diagnostic Clinical Text-Guided Diffusion Model for Limb Fracture Localization in CT Images**

This repository provides the research implementation of TextDiffDet.

TextDiffDet is designed for  CT slice-level fracture localization. The framework mainly consists of the following components:

- **Clinical Semantic Encoding Module (CSEM)**: employs a BERT encoder and gated cross-attention mechanism;
- **Spatial Refinement Module (SRM)**: applied to Stages 4 and 5 of the ResNet backbone;
- **Mamba Fracture Long-range Modeling Module (MFLM)**: applied to Stages 4 and 5 of the ResNet backbone;
- **DiffusionDet detection head**: progressively refines bounding box proposals through iterative denoising.

> This repository contains code for scientific research purposes only. It is not a medical device and must not be used for clinical diagnosis.

## Project Structure

```text
configs/                    Experiment configuration files
diffusiondet/               TextDiffDet and diffusion-based detection code
detectron2/                 Bundled Detectron2 fork with SRM/MFLM backbone modifications
Text/                       Lightweight BERT implementation used by CSEM
dataset.py                  COCO dataset registration
train_net.py                Main training entry point
tools/                      Vocabulary construction and extended evaluation tools
```

## Dataset Format

The **LimbFrac-CT** dataset should first be split at the **patient/case level** and then organized in COCO format:

```text
datasets/limbfrac_ct/
  train/images/
  train/annotations.json
  val/images/
  val/annotations.json
```

The dataset must be split by **patient/case**, rather than by randomly splitting individual 2D CT slices. All CT slices belonging to the same patient or case must be assigned to the same dataset subset. Slices from the same patient must not appear simultaneously in the training and validation/test sets.

Therefore, the **patient/case-level split must be completed before exporting the 2D CT slices**, in order to prevent data leakage caused by images from the same patient appearing in different dataset subsets.

Each image record may optionally contain a pre-diagnostic `report` field.

Do **not** use the final radiological diagnosis or any textual information generated after the diagnostic process has been completed.

A custom dataset root directory can be specified using:

```bash
export TEXTDIFFDET_DATA_ROOT=/path/to/your/coco_dataset
```

## Installation

Create a Conda environment with Python 3.10:

```bash
conda create -n textdiffdet python=3.10 -y
```

Activate the environment:

```bash
conda activate textdiffdet
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Install the bundled Detectron2 package:

```bash
pip install -e ./detectron2
```

Install the BERT implementation used by the project:

```bash
pip install -e ./Text
```

The complete installation procedure is:

```bash
conda create -n textdiffdet python=3.10 -y
conda activate textdiffdet
pip install -r requirements.txt
pip install -e ./detectron2
pip install -e ./Text
```

Download a **ResNet-50 ImageNet pretrained weight file** compatible with Detectron2 and save it as:

```text
weights/R-50.pkl
```

By default, the configuration enables the **token-level CSEM (Clinical Semantic Encoding Module)** used in the paper.

To reproduce checkpoints from an earlier version that used **pooled-CLS** features, override the following configuration:

```text
TEXT.USE_TOKEN_LEVEL False
```

Do not directly evaluate an old checkpoint with token-level mode enabled, as the model architecture may not match the historical checkpoint weights.

## Building the Text Vocabulary

Run the following command to construct the medical text vocabulary from the training-set annotation file:

```bash
python tools/build_vocab.py \
  --annotations "$TEXTDIFFDET_DATA_ROOT/train/annotations.json" \
  --output assets/medical_vocab.pkl
```

where:

- `--annotations`: path to the COCO-format annotation file of the training set;
- `--output`: path used to save the generated medical vocabulary.

The generated vocabulary file will be saved as:

```text
assets/medical_vocab.pkl
```

## Model Training

Run the following command to start training:

```bash
python train_net.py --num-gpus 1 --config-file configs/diffdet.bone_R50.yaml
```

where:

- `--num-gpus 1`: use one GPU for training;
- `--config-file configs/diffdet.bone_R50.yaml`: specify the model configuration file.

## Model Evaluation

Evaluate a trained model using:

```bash
python train_net.py --num-gpus 1 --eval-only \
  --config-file configs/diffdet.bone_R50.yaml \
  MODEL.WEIGHTS /path/to/model.pth
```

Replace:

```text
/path/to/model.pth
```

with the path to the actual trained model checkpoint.

### Evaluation Metrics

In COCO-style evaluation, `mAP` refers to **mAP@0.50:0.95**.

AP is calculated at the following IoU thresholds:

```text
0.50, 0.55, 0.60, 0.65, 0.70,
0.75, 0.80, 0.85, 0.90, 0.95
```

The final mAP is obtained by averaging AP across these IoU thresholds.

In addition to the overall mAP, the following metrics should also be reported:

- **AP50**: average precision at an IoU threshold of 0.50;
- **AP75**: average precision at an IoU threshold of 0.75.

Extended evaluation can also be performed using:

```text
tools/evaluate_extended.py
```

which includes:

- **F1 score** at a fixed confidence threshold;
- **IoU (Intersection over Union)**;
- **Normalized Center Error**;
- **Area Error**;
- **Object-scale stratified evaluation** based on target-size tertiles calculated from the training set.

## Paper–Code Alignment

For a detailed description of how the model architecture presented in the paper corresponds to the implementation in this repository, please refer to:

```text
PAPER_CODE_ALIGNMENT.md
```

The public configuration files use the same module names as those adopted in the paper:

- **CSEM**: Clinical Semantic Encoding Module;
- **SRM**: Spatial Refinement Module;
- **MFLM**: Mamba Fracture Long-range Modeling Module.

## License

The license in the project root directory is inherited from the original **DiffusionDet** repository.

Third-party components included in this project retain their respective original license notices.
