# Defense guide

Use `Final_Presentation.pptx` for delivery; the PDF is a portable visual backup. Rehearse for approximately 12 minutes, four minutes per person. References (slides 20–23) are backup material.

| Speaker | Slides | Four-minute focus |
|---|---|---|
| Akhmetov Kazbek | 1–3, then 15–18 | Research question and literature; external benchmarks, limits, demonstration and conclusions |
| Tazhibayev Marsel | 4–7 | Data audit, training-only augmentation, 646 features, five-fold CV and forest tuning |
| Uranov Daryn | 8–14 | CNN/ViT, C1–C3, three continuation trials, frozen MobileNet transfer and measured errors |

Slide 19 states the equal responsibility allocation. Everyone should explain the split, macro F1 and limitations. Allocation describes responsibility, not an invented work history.

## Demonstration

From the repository root after installing `project/requirements.txt`:

```bash
python -m streamlit run project/app.py
```

1. Keep MobileNet selected, choose a bundled image and click Classify.
2. Explain the predicted category and uncalibrated class scores.
3. Choose the tuned CNN and classify the same example.
4. Enable supporting experiments to show the retained configurations if asked.
5. Explain that an individual example illustrates inference; the reported accuracy uses all 584 test images.

The default five-model menu and seven supporting configurations use one shared inference module. Automated checks cover all twelve bundled-image classification paths. Rehearse an actual browser upload before class; that interaction was not verified here. The optional `demo/index.html` explains only the original CNN C2 and ViT internals.

## Questions to prepare

**What is the best result?** Frozen ImageNet MobileNetV3 Small with a fitted logistic head: 506/584 correct, 86.64% accuracy and 0.8643 macro F1.

**Did you fine-tune MobileNet?** No. Its backbone is frozen. We fit a StandardScaler and logistic classifier on 576-dimensional features. Validation chooses C = 0.01 from four values.

**What improved in your CNN?** Three 15-epoch continuations start from C2. Validation selects smoothing 0.05, learning rate 0.0001 and weight decay 0.001. Test accuracy rises from 474 to 480 correct, or 81.16% to 82.19%. Multiple settings changed; smoothing alone is not proven causal.

**Why not 99%?** Published EfficientNet/DenseNet results use binary-crack tasks and different data. Their weights could not be downloaded here; no local score is claimed for them. Our seven-class measured maximum is 86.64%.

**How did you prevent leakage?** Exact pixel duplicates and conflicting-label groups are excluded before splitting. Training augmentation stays in training. Settings use CV/validation, not test fitting. Building IDs are unavailable, so same-building and near-duplicate leakage cannot be ruled out.

**What is the split?** 2,725 train, 584 validation, 584 test after cleaning to 3,893 images. Classical CV uses the combined 3,309 development images; neural and transfer selection retain validation separately.

**Why macro F1?** It averages each class's F1 equally. Accuracy measures the overall fraction correct; these answer different questions.

**Why did ViT lose?** This compact scratch model has 273,127 parameters and limited training data. The experiment compares particular recipes, not every transformer against every CNN.

**What does the augmentation experiment prove?** It preserves a source-linked expanded dataset. That stored expansion was not used in reported training; no accuracy benefit is claimed.

**What are the weakest classes?** MobileNet's major-crack F1 is 0.7765 and peeling F1 is 0.8108. Minor crack and stain improve over C2, while algae declines.

**Is the model ready for building safety decisions?** No. It categorizes visible surfaces; it does not estimate structural integrity, severity or repairs. External building-level testing and calibrated uncertainty remain future work.

**What is reproducible?** Saved checkpoints, histories, splits, per-image predictions, tuning selections and scripts. All twelve current checkpoints reproduce recorded test predictions. C1 lacks a supplied checkpoint and remains archived evidence.
