# Defense guide

Aim for equal speaking time, approximately four minutes per member. Slide numbering follows the final 20-slide deck; references are backup slides 18–20.

| Presenter | Slides | Explain |
|---|---|---|
| Kazbek | 1–3, 15–16 | Problem, research rationale, live app, limitations and conclusions |
| Marsel | 4–7 | Split, augmentation boundary, 646 features, CV and tuning |
| Daryn | 8–14 | CNN, ViT, C1/C2/C3, results, errors and stability |
| All | 17 | Equal responsibility allocation and questions |

Daryn has more short technical slides; Kazbek includes the demonstration. Practice to keep total time equal.

## Demo script

1. Start `python -m streamlit run project/app.py`.
2. Select a bundled wall image and CNN C2; click Classify.
3. Explain the highest-scoring class and the probability table without claiming calibration.
4. Select the tuned Random Forest, classify the same example, and compare predictions.
5. Enable supporting experiments, select CNN C3, and explain the schedule comparison.
6. Open `demo/index.html` separately. Compare CNN feature maps with ViT patches and attention.
7. End with the recorded test comparison and limitations. A single demo image is not a benchmark.

## Questions each member should answer

- **Why five model families?** They cover linear feature-based prediction, tree ensembles, neighbor-based prediction, convolution and attention. CNN recipes and forest tuning are configurations, not additional families.
- **Why macro F1?** It weights each category equally and reveals weaknesses masked by aggregate accuracy.
- **What prevents leakage?** Exact pixel duplicates are removed before splitting; scaling stays inside CV folds; augmentation derivatives stay with training sources. Same-building near-duplicates remain unresolved.
- **Why different split sizes for classical and neural models?** Classical CV uses train plus validation as a development set; neural checkpoint selection uses a fixed validation set. This prevents architecture-only causal conclusions.
- **Why C2?** Highest validation macro F1 among the three CNN recipes, 0.7774. The test score is reported after that choice.
- **What does C3 add?** A measured constant-rate comparison against C2 with the same prescribed starting rate, batch size and epoch budget. One seed limits generalization.
- **Did the stored augmentation improve accuracy?** This has not been measured. Its generation and provenance are supplied separately from the reported model runs.
- **Can you reproduce C1?** Its history and scores are retained, but its checkpoint is absent. C2, C3 and the other included checkpoints are rerun-verified.
- **Why does CNN beat ViT here?** It is an observed result for these configurations; local texture, data size and training differences are possible explanations, not isolated causal findings.
- **Can this assess building safety?** No. It classifies visual categories and has no structural measurements, localization or unknown-image rejection.

Before defense, each person should run the app and explain one correct and one incorrect test prediction from the saved CSV files.
