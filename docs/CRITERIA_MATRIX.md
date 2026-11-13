# Requirements and evidence map

This maps the supplied final-project structure, 2026–2027 methodical guidelines, sample report and updated stage reports to the consolidated submission. It is an evidence checklist, not a promised grade.

| Requirement | Final evidence | Status / boundary |
|---|---|---|
| Word and PDF academic report | `Final_Report.docx`, `Final_Report.pdf` | Complete; 21 pages |
| Topic, abstract, introduction, purpose and objectives | Report title and Sections 1–2 | Complete; five objectives |
| Recent literature review | Section 2 | Twelve studies from 2022–2025; older foundational/benchmark sources separate |
| 12–18 references in APA style | Report references | Seventeen references |
| Data source, types, cleaning and split | Section 3, split and exclusion manifests | Complete; local/public class-count discrepancy disclosed |
| Feature selection/preprocessing | Sections 3–4, executed notebook | Explicit handcrafted 646-feature design; no false automatic-selection claim |
| Mathematical methods | Sections 4–5 | Twelve native Word equation objects; no equation screenshots |
| Compare at least three methods; 3-person team with 3–5 main models | Sections 4–6; five main models | MobileNet, tuned CNN, ViT, tuned forest and logistic regression; seven supporting configurations |
| Cross-validation and tuning | Sections 4–6, CV results, frozen experiment records | Five-fold classical CV, forest search, three CNN trials and four MobileNet head settings |
| Quantitative results and visual analysis | Sections 6–7 | Metrics, confusion matrix, class analysis and training figures |
| Link conclusions to objectives | Section 9 | Five numbered conclusions |
| Working application and repository | `project/app.py`, `project/inference.py`, registry | One unified demo; all twelve AppTest classification paths pass |
| Model verification | `verification/checkpoints.json` | All twelve checkpoints rerun on all 584 test images; C1 checkpoint unavailable and excluded |
| Three project stages integrated | Section 3 stage table; experiment folders | Midterm, augmentation preparation, C3 and final tuning retained |
| Equal responsibility and defense participation | Section 9, `DEFENSE.md`, slide 19 | One-third allocation; actual participation requires the team |
| Presentation and readable figures | `Final_Presentation.pptx`, slide PDF | 23 slides, speaker notes, two editable native charts; structural/layout checks pass |
| Published high-accuracy comparisons | Section 8 | External 99% figures separated from local measurements; EfficientNet/DenseNet downloads failed |
| Submission and external execution | README release status | GitHub publication needs valid authentication; native PowerPoint opening, live browser upload, Moodle submission and oral defense remain external |

## Evaluation boundaries

The supplied rubric's listed weights total 90 although its printed total is 100. No missing criterion or ten points has been invented. Instructor assessment, plagiarism checks and oral performance cannot be certified by artifact inspection.

The holdout is reused, protocols differ, ImageNet adds external training data, and only one neural seed is available. Stored augmentation is documented but unused in reported runs. These limits are stated consistently in the report and slides.
