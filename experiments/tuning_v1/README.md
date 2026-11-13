# Validation-guided tuning and pretrained comparison

These experiments are integrated into the authoritative [final report](../../docs/Final_Report.pdf) and [presentation](../../docs/Final_Presentation.pptx). `PUBLISHED_COMPARISON.md` records external benchmark context.

From the repository root, using the project Python environment:

```bash
python experiments/tuning_v1/run.py --data /path/to/BD3_original_dataset --mobile-weights experiments/tuning_v1/source_weights/mobilenet_v3_small-047dcff4.pth --output experiments/tuning_v1/new_results
python -m streamlit run project/app.py
```

The demo uses the delivered `results/` snapshot. Every training invocation requires an empty output directory, so original evidence is preserved. The original ImageNet MobileNet checkpoint is included under `source_weights/` with attribution to torchvision. The CNN trials initialize from `project/models/cnn.pt`. The MobileNet source checkpoint must have the official hash prefix; the delivered stripped backbone can run inference but cannot replace the original 1,000-class source checkpoint when reproducing training.

## Additional 99%-paper architectures

```bash
python experiments/tuning_v1/run_pretrained.py --data /path/to/BD3_original_dataset --output experiments/tuning_v1/new_pretrained_results
```

The script downloads official ImageNet EfficientNet-B0 and DenseNet201 weights, extracts frozen features, tunes a seven-class logistic head using validation macro F1, and evaluates after selection. It does not reproduce the papers' binary heads or full fine-tuning protocols. Network download is necessary unless `--weights-dir /path/to/weights` provides these files:

- `efficientnet_b0_rwightman-7f5810bc.pth`
- `densenet201-c1103571.pth`

Official URLs and the observed DNS failures are in `external_attempt/status.json`. An unavailable model receives no measured accuracy. Do not replace missing measurements with a literature score.

## Evidence

- `results/protocol.json`: settings fixed before training and checkpoint/source hashes.
- `results/*_history.csv`: every CNN training epoch.
- `results/cnn_validation.json` and `mobile_validation.json`: complete searched configurations.
- `results/selection_frozen_before_test.json`: selections made before test-image loading.
- `results/comparison.csv`: measured shared-test metrics.
- `results/*_predictions.csv`, `*_probabilities.npy`, `*_class_report.json`: individual predictions and per-class findings.
- `results/*.pt`, `mobilenet_head.joblib`: runnable model artifacts.

This is a single-seed, reused-holdout experiment. Further validation tuning also increases validation overfitting risk. External published experiments use different tasks/splits and do not certify 99% accuracy here.
