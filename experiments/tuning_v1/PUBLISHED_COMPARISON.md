# What the published “99%” results mean

These are external literature benchmarks, not test results measured by this repository. An ImageNet checkpoint is an initialization for transfer learning; it is not the authors' crack-trained checkpoint and does not inherit their reported accuracy.

| Published approach | Reported result | Evaluation context | Relationship to this project |
|---|---:|---|---|
| EfficientNet-B0 transfer learning, Su & Wang (2020), Table 2 | 99.11% accuracy; 97.37% on their different-dataset evaluation | Binary crack/noncrack photographs | Suitable architecture to test; neither score is seven-class BD3 accuracy |
| DenseNet201, Su & Wang (2020), Table 2 | 99.32% accuracy; 97.49% on their different-dataset evaluation | Same binary crack comparison | Larger architecture to test; paper-trained weights were not supplied |
| VGG-based transfer learning, Yang et al. (2020) | 99.83% CCIC; 99.72% BCD; 97.07% SDNET | Different binary crack datasets | Shows how the same approach's reported accuracy changes by dataset |
| Pretrained ViT patch16, BD3 authors' README benchmark table | 98.79% F1 on augmented BD3; 93.23% F1 on original BD3 | Their 60/20/20 splits and augmentation pipeline | Closest domain reference, but different splits/preprocessing; F1 is not accuracy |

The BD3 README abstract says original-data ViT F1 is 0.9342, whereas its benchmark table gives 0.9323. This comparison uses the explicitly labelled table value and records the inconsistency. The published augmented split cannot be treated as equivalent to this repository's source-grouped, deduplicated 2,725/584/584 partition.

## Executed local experiment

Three further-training CNN recipes start from the preserved validation-selected C2 checkpoint. Each trains for 15 epochs, changing learning rate, AdamW weight decay and label smoothing. Epochs and trial winners are chosen only by validation macro F1. The original C2 remains a candidate: a newly trained model is not automatically considered better.

A locally cached torchvision ImageNet MobileNetV3 Small checkpoint has the expected official SHA-256 filename prefix. Its original 1,000-class classifier is removed. Frozen 576-dimensional features from the original and horizontally flipped training photographs train a standardized multinomial logistic head. C in {0.01, 0.1, 1, 10} is chosen by validation macro F1. This is frozen-feature transfer learning, not full backbone fine-tuning and not a “99% model” claim.

The EfficientNet-B0 and DenseNet201 downloads were attempted from official torchvision URLs, but this execution environment cannot resolve the download host. `external_attempt/status.json` records the actual failures. No local scores are invented for those models. `run_pretrained.py` implements the same frozen-feature, validation-selected benchmark and accepts a local weights directory for offline use.

## Sources

- Su, C., & Wang, W. (2020). Concrete cracks detection using convolutional neural network based on transfer learning. *Mathematical Problems in Engineering*, Article 7240129. https://doi.org/10.1155/2020/7240129 (Table 2; original publisher: https://onlinelibrary.wiley.com/doi/10.1155/2020/7240129).
- Yang, Q., Shi, W., Chen, J., & Lin, W. (2020). Deep convolution neural network-based transfer learning method for civil infrastructure crack detection. *Automation in Construction, 116*, 103199. https://doi.org/10.1016/j.autcon.2020.103199
- Kottari, P., & Arjunan, P. (2024). BD3 benchmark repository, comparison table. https://github.com/Praveenkottari/BD3-Dataset
- Torchvision pretrained weights and transformations: https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html ; https://docs.pytorch.org/vision/main/models/generated/torchvision.models.densenet201.html ; https://docs.pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v3_small.html

Accessed 13 November 2026. No paper's whole training pipeline is claimed as reproduced. The existing holdout has already been reused across project stages, and repeated validation tuning can also overfit validation. Independent building-level evaluation remains needed.
