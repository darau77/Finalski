# Building Surface Defect Classification

A consolidated final project for **AMO 3208 Machine Learning Algorithms**, Astana IT University, 2026–2027. **Uranov Daryn, Tazhibayev Marsel and Akhmetov Kazbek — SE-2403.**

The validation-selected **frozen ImageNet MobileNetV3 Small + logistic head** achieves **86.64% test accuracy and 0.8643 macro F1** on the shared 584-image test subset. Further CNN tuning improves accuracy from 81.16% to 82.19%. This repository combines the original study, C3, CNN tuning and pretrained transfer into one report, presentation and application.

## Final deliverables

- [Report PDF](docs/Final_Report.pdf) · [Editable Word report](docs/Final_Report.docx)
- [PowerPoint with speaker notes](docs/Final_Presentation.pptx) · [Slide PDF](docs/Final_Presentation.pdf)
- [Requirements and evidence map](docs/CRITERIA_MATRIX.md)
- [Equal speaking plan and defense guide](docs/DEFENSE.md)

These are the authoritative consolidated documents. Experiment folders retain source evidence; their historical results do not supersede the final comparison below.

## Start the demo

Use Python 3.12. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r project/requirements.txt
python -m streamlit run project/app.py
```

On Windows, activate with `.venv\Scripts\activate`. Select a bundled example and click **Classify**, or upload JPG, PNG or WebP. MobileNet is the default. Enable **supporting experiments** to access all twelve saved configurations. No retraining or full dataset is needed for inference. Package installation requires internet; inference stays local. Uploads are not saved.

Optional: open `demo/index.html` in a desktop browser for a self-contained visualization of the original CNN C2 and ViT, including feature maps and attention. It is an architecture explorer, not the MobileNet demo. GitHub's HTML viewer shows source; download the file to run it.

## Five main models on the same test subset

| Model | Accuracy | Macro F1 | Correct / 584 | Training basis |
|---|---:|---:|---:|---|
| MobileNetV3 transfer | **86.64%** | **0.8643** | **506** | Frozen ImageNet features + fitted logistic head |
| Tuned custom CNN | 82.19% | 0.8200 | 480 | C2 continuation with validation-selected smoothing recipe |
| Compact ViT | 71.06% | 0.7055 | 415 | Trained from scratch |
| Tuned Random Forest | 76.37% | 0.7612 | 446 | Handcrafted features and five-fold CV |
| Logistic regression | 62.50% | 0.6203 | 365 | Handcrafted features and five-fold CV |

Supporting configurations are the historical decision tree, KNN, untuned Random Forest, CNN C2, CNN C3, and two other CNN continuation trials. C1's 77.91% is archived evidence: its checkpoint was not supplied and it is not counted among the twelve runnable configurations.

The MobileNet head selects C = 0.01 by validation macro F1 (0.8874). CNN tuning selects label smoothing 0.05 at epoch 9 by validation macro F1 (0.7797). Settings are frozen before the new test evaluations. The original C2 scored 474/584; MobileNet adds 32 correct predictions, or 5.48 percentage points. This is a single-seed observed comparison, not a significance claim.

## Data and evaluation

The supplied original copy has 3,965 RGB images. Cleaning removes 61 redundant same-label files and 11 conflicting-label files, leaving 3,893 unique pixel hashes. The fixed seed-42 split is 2,725 train, 584 validation and 584 test. Classical CV combines train and validation into 3,309 development images; CNN/ViT and MobileNet head selection retain the separate validation subset.

The original dataset source is [BD3](https://github.com/Praveenkottari/BD3-Dataset), described by [Kottari and Arjunan (2024)](https://doi.org/10.1145/3671127.3698789). The supplied local folder counts differ from the public README for three labels; the report documents this rather than relabelling images. The full submission archive includes the supplied originals; Git history omits image data. Dataset authors retain their original rights; no new dataset license is asserted.

`data_preparation/` retains the stored-augmentation generator and metadata for 17,517 files. That expansion was **not used** in any model result reported here. MobileNet uses a separate original-plus-horizontal-flip feature procedure on training images only.

## Reproduce and verify

Place `BD3_original_dataset` inside `project/` or set `BD3_DATASET`. Run the executed classical, CNN and ViT notebooks from `project/`; use a copy because retraining changes artifacts. Jupyter/VS Code notebook support may need separate installation.

From the repository root:

```bash
python project/verify_final.py --dataset /absolute/path/BD3_original_dataset
python scripts/audit_data.py --dataset /absolute/path/BD3_original_dataset
python scripts/test_app.py
python experiments/tuning_v1/verify.py
```

The first command reruns all twelve configurations over all 584 test images; labels match saved predictions and neural probabilities match within tolerance. Without `--dataset`, it checks stored evidence only. Data auditing recomputes original pixel hashes and checks augmentation metadata grouping; derivative image bytes are not re-audited. The unified app passes twelve real bundled-image classification paths through Streamlit AppTest. This does not certify a manual browser/file-upload rehearsal.

C3 reproduction:

```bash
python experiments/run_cnn_config3.py --project project --data /absolute/path/BD3_original_dataset --output experiments/new_c3_run
```

CNN tuning and frozen MobileNet reproduction:

```bash
python experiments/tuning_v1/run.py --data /absolute/path/BD3_original_dataset --mobile-weights experiments/tuning_v1/source_weights/mobilenet_v3_small-047dcff4.pth --output experiments/new_tuning_run
```

These scripts require empty output directories. The official MobileNet source checkpoint is included and its expected hash prefix is verified. The comparison is frozen-feature transfer, not end-to-end MobileNet fine-tuning.

## Published 99% models

EfficientNet-B0 and DenseNet201 have published binary-crack accuracies of 99.11% and 99.32% in [Su and Wang (2020)](https://doi.org/10.1155/2020/7240129). Those are not seven-class BD3 results. Official pretrained-weight downloads failed here due to DNS resolution; neither has a local measured score. The final report separates these external benchmarks from our results.

A runner is ready when networking or local weights are available:

```bash
python experiments/tuning_v1/run_pretrained.py --data /absolute/path/BD3_original_dataset --output experiments/new_external_run --weights-dir /absolute/path/weights
```

Omit `--weights-dir` to download official ImageNet weights. This is a new seven-class frozen-feature experiment, not a reproduction of the authors' binary-crack checkpoints. See `experiments/tuning_v1/external_attempt/status.json` for actual failures.

## Repository map

| Path | Contents |
|---|---|
| `project/` | Unified app and shared inference, model definitions, executed notebooks, registry and original checkpoints |
| `project/results/` | Main metrics, split, exclusions and figures |
| `experiments/config3_results/` | C3 frozen protocol, checkpoint, history and predictions |
| `experiments/tuning_v1/` | Tuning/transfer source, original pretrained weights, results and verification |
| `data_preparation/` | Stored augmentation source and manifests |
| `demo/` | Offline C2 CNN/ViT architecture explorer |
| `verification/` | Full checkpoint, app, data and document checks |
| `docs/` | Consolidated report, presentation, criteria map and defense guide |

## Equal responsibilities

| Member | One-third responsibility area |
|---|---|
| Uranov Daryn | CNN, ViT and transfer experiments, neural evaluation and error analysis |
| Tazhibayev Marsel | Data, features, classical models, CV and tuning |
| Akhmetov Kazbek | Integration, literature, report, slides and demonstration |

All members review the full study. This is the agreed responsibility allocation, not a fabricated contribution history.

## Limits and release status

The holdout has been reused; building IDs are unavailable; near-duplicate scenes may remain; protocols and training budgets differ; pretrained MobileNet uses external ImageNet data. Probabilities are uncalibrated and unrelated images still receive a known class. Predictions do not determine structural safety.

The local Git repository and archives are complete. GitHub publication is pending valid authentication. Native PowerPoint opening, a live browser/upload rehearsal, team defense and Moodle submission remain external actions. CI configuration is included; no remote CI result is claimed.
