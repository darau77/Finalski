# Building Surface Defect Classification

Final project for AMO 3208 Machine Learning Algorithms, Astana IT University, 2026–2027. Team: **Uranov Daryn, Tazhibayev Marsel, Akhmetov Kazbek (SE-2403)**.

Five primary model families classify wall photographs into algae, major crack, minor crack, peeling, plain, spalling and stain. The validation-selected CNN C2 achieves **81.16% test accuracy and 0.8085 macro F1**. The updated endterm CNN C3 is integrated as a supporting training-configuration experiment. Results describe one reused local test subset; they do not establish performance on unseen buildings.

## Open the deliverables

- [Final report PDF](docs/Final_Report.pdf)
- [Editable Word report](docs/Final_Report.docx)
- [Presentation with speaker notes](docs/Final_Presentation.pptx)
- [Presentation PDF preview](docs/Final_Presentation.pdf) (static slide images; use PPTX for editing)
- [Rubric coverage and outstanding checks](docs/CRITERIA_MATRIX.md)
- [Defense guide](docs/DEFENSE.md)

## Demo in one minute

**No installation:** download or clone the repository, then open `demo/index.html` in a desktop browser. It embeds the CNN C2 and ViT weights, examples, feature maps and attention. GitHub's file viewer displays HTML source; download it to run the demo.

**All model families:** use Python 3.12, then run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r project/requirements.txt
python -m streamlit run project/app.py
```

On Windows, activate using `.venv\Scripts\activate`. Select a bundled example and click **Classify**, or upload a JPG/PNG/WebP. Enable **Show supporting experiments** to access C3, the untuned forest and the historical tree. No full dataset or retraining is needed to demonstrate inference. Installation requires internet; inference stays local.

The five primary families are logistic regression, Random Forest, KNN, CNN and ViT. Tuning and C1/C2/C3 are configurations, not new algorithm families. The historical decision tree is supplementary, not part of the five-family final-project scope.

## Results on the same 584 images

| Primary method | Accuracy | Macro F1 | Training source |
|---|---:|---:|---|
| Logistic regression | 62.50% | 0.6203 | `project/notebook.ipynb` |
| Random Forest tuned | 76.37% | 0.7612 | `project/notebook.ipynb` |
| KNN | 65.07% | 0.6375 | `project/notebook.ipynb` |
| CNN C2 cosine | 81.16% | 0.8085 | `project/cnn_training.ipynb` |
| Small ViT | 71.06% | 0.7055 | `project/vit_training.ipynb` |

| CNN recipe | Validation F1 | Test accuracy | Evidence status |
|---|---:|---:|---|
| C1 original | 0.7594 | 77.91% | Archived metrics/history; checkpoint unavailable |
| C2 cosine | 0.7774 | 81.16% | Checkpoint and predictions verified |
| C3 constant | 0.7694 | 76.03% | Updated endterm checkpoint and predictions verified |

C2 is selected by validation macro F1 within the CNN comparison. The final test subset is excluded from fitting and epoch selection, but has been reused across project iterations. Classical models use 3,309 development images with five-fold CV; neural models use 2,725 train and 584 validation images. This is not a controlled architecture-only comparison.

## Repository map

| Path | Purpose |
|---|---|
| `project/` | Model definitions, executed notebooks, Streamlit app and dependencies |
| `project/models/` | Eight runnable saved model configurations and registry |
| `project/results/` | Predictions, histories, metrics, original split and exclusions |
| `experiments/` | C3 training source, frozen protocol, selected checkpoint and evidence |
| `data_preparation/` | Assignment 2 augmentation generator and source-linked manifests |
| `demo/` | Self-contained browser observatory and rebuild source |
| `scripts/` | Data audit and application checks |
| `verification/` | Machine-readable results of completed checks |
| `docs/` | Report, presentation, requirement map and defense guide |

## Reproduce the experiments

Obtain the original images from [BD3](https://github.com/Praveenkottari/BD3-Dataset), citing Kottari and Arjunan (2024), [DOI](https://doi.org/10.1145/3671127.3698789). The separate full submission archive includes the supplied original copy; images are omitted from Git history to keep the repository manageable. Preserve the local labels: the public README's first three counts differ from the supplied folders. The report discloses this discrepancy.

Place `BD3_original_dataset` inside `project/`, or set `BD3_DATASET` to its absolute path. Run notebooks from `project/` in this order: `notebook.ipynb`, `cnn_training.ipynb`, `vit_training.ipynb`. Use VS Code's notebook support or install Jupyter separately. Outputs are already retained. Retraining changes checkpoints and results: use a copy and regenerate tables/documents if scores change.

Run C3 separately from the repository root, using a new output directory:

```bash
python experiments/run_cnn_config3.py --project project --data /absolute/path/BD3_original_dataset --output experiments/new_c3_run
```

The C3 script refuses a non-empty output directory. It freezes its protocol before training and selected checkpoint before test loading. C1's checkpoint is not available, so its score is explicitly archived evidence.

The stored augmentation extension contains 17,517 manifest records: 16,349 train, 584 validation and 584 test. It is **not used in the reported model experiments**. `data_preparation/README.txt` documents generation; reproduce image derivatives from the originals when needed. No accuracy benefit from this extension is claimed.

## Verification

```bash
python project/verify_final.py
python project/verify_final.py --dataset /absolute/path/BD3_original_dataset
python scripts/audit_data.py --dataset /absolute/path/BD3_original_dataset
python scripts/test_app.py
```

The first command checks stored metrics and test membership. The second additionally reruns all eight checkpoints on all 584 test images. It writes `verification/checkpoints.json`; use `--output` to keep a separate audit. The audit command recomputes source pixel hashes and checks augmentation-manifest grouping. The app test exercises actual bundled-image classification for every available model through Streamlit AppTest. It does not prove interactive browser rendering or manual file-upload behavior.

All eight checkpoint prediction sets and eight application paths passed. The report's twelve formulas are native editable Word equations, not screenshots. Report pages and slide renders were visually reviewed. See the requirement matrix for the explicit boundary of each check.

Browser automation cannot open local files in this environment, and local web serving is restricted. A live browser rehearsal remains a team action. C1 cannot be checkpoint-verified because its original weights are missing. These limitations are preserved rather than marked as passed.

Rebuild the offline explorer after changing C2/ViT weights:

```bash
python demo/build_comparison.py --project project
```

## Equal responsibility allocation

| Member | One-third responsibility area |
|---|---|
| Uranov Daryn | CNN/ViT implementation, training configurations, neural evaluation and error analysis |
| Tazhibayev Marsel | Data cleaning, feature engineering, classical models, CV and tuning |
| Akhmetov Kazbek | Application integration, literature, report, presentation and demonstration |

All members must review and understand the full study. The allocation is a responsibility plan, not a fabricated historical contribution or commit log. Live defense requires actual participation.

## Scope and data handling

Uploads are processed locally and are not saved. The classifier always outputs one known label, including for unrelated images. Probability scores are not calibrated guarantees. Attention is an internal computation, not a causal explanation. Predictions do not determine crack depth, structural safety or repair urgency. Data retains its original authorship and source attribution; no new license over third-party images is asserted.
