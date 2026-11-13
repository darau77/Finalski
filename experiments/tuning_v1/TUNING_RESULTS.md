# Tuning and pretrained comparison: measured results

Selection used validation macro F1 before test-image loading. All five measured configurations share the same 584 test images.

| Configuration | Validation F1 | Test accuracy | Test macro F1 | Correct / 584 |
|---|---:|---:|---:|---:|
| CNN C2 baseline | 0.7774 | 81.16% | 0.8085 | 474 |
| CNN: lower LR | 0.7778 | 81.68% | 0.8148 | 477 |
| CNN: label smoothing | 0.7797 | 82.19% | 0.8200 | 480 |
| CNN: smaller step | 0.7743 | 81.68% | 0.8142 | 477 |
| ImageNet MobileNetV3 + LR | 0.8874 | 86.64% | 0.8643 | 506 |

Validation-selected configuration: **ImageNet MobileNetV3 + LR**. It scores **86.64% test accuracy** and **0.8643 macro F1**, a change of **+5.48 percentage points** from CNN C2. This is an observed single-run comparison, not a significance claim.

Three CNN trials each ran 15 additional epochs from the same C2 checkpoint. MobileNet uses a frozen ImageNet feature extractor plus a validation-tuned logistic head, with original and horizontally flipped training images. It is not a fully fine-tuned backbone.

EfficientNet-B0 and DenseNet201 could not be executed: their official pretrained weights failed to download because DNS resolution is unavailable. Their literature scores (99.11% and 99.32% on binary crack detection) are external references only. See [PUBLISHED_COMPARISON.md](PUBLISHED_COMPARISON.md) for sources and task differences.

The old final report remains a snapshot of the earlier eight configurations. This addendum supplies the new experiment; do not present its results as if they were already included in that report. All original artifacts are preserved.
