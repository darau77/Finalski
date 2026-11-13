# Final-project requirement and evidence map

Reviewed against **Final project structure and assesment (1) (2) (1).pdf** and **Methodical Guidlines for Final Project_2026-2027.pdf**, with the supplied example report and dataset-source slides. Updated endterm report, Assignment 2, C3 source supplement and original project files are incorporated. The dataset-source slides list suggestions; they do not require a real-time feed for this image project.

“Provided” means the specified artifact content is present. “Verified” names a completed check. Neither label guarantees an instructor's grade or substitutes for live defense.

| Requirement | Evidence | Status |
|---|---|---|
| Maximum three members; three-person project uses 3–5 models and an application | Report pp. 1, 14; README; five primary families: LR, RF, KNN, CNN, ViT | Provided; historical tree and alternative settings explicitly supplementary |
| Clear topic, abstract, keywords | Report p. 1 | Provided |
| Introduction: problem, purpose, objectives, research object, theoretical/methodological basis | Report p. 2 | Provided |
| Literature analysis of 10–12 recent sources from the last 3–5 years | Report pp. 3–4: 12 studies from 2022–2025 | Provided; domain and methodology studies synthesized, with limits on transfer |
| APA bibliography of 12–18 sources | Report pp. 16–17: 13 references; 12 recent plus foundational ViT (2021) | Provided |
| Data origin, size, structure and variable types | Report p. 5; split/exclusion CSVs; source link in README | Provided; local/public label-count discrepancy disclosed |
| Cleaning, normalization, feature selection and synthetic-data purpose where relevant | Report pp. 5–6; 646 selected descriptors; train-only transformations | Provided; exact source pixels independently checked |
| Training, validation and test protocol | Report pp. 7, 9–10; notebooks and C3 protocol | Provided; reused holdout, unequal development protocols and lack of building IDs disclosed |
| Mathematical formulations made with an equation editor | Report pp. 7–9: 12 native editable OMML equation blocks | Verified as native Word mathematics; rendered visually |
| At least three preprocessing/ML methods compared | Five primary families; Table 3; slide 12 | Provided |
| Comparison, validation and model selection | Five-fold classical CV; eight-setting RF search; neural validation checkpoint selection | Stored results checked; original full training not rerun |
| Numeric results, tables, charts and labelled figures | Report pp. 10–12; three figures and nine captioned tables; slides 11–14 | Verified against saved evidence; two editable native slide charts |
| Discussion and interpretation | Report p. 13 | Provided; no unsupported significance, causality or safety claims |
| Numbered conclusions corresponding to objectives; recommendations | Report p. 14; recommendations pp. 12–14 | Provided: five matching numbered conclusions |
| Working application and available source code | project/app.py; demo/index.html; notebooks; eight saved checkpoints | All eight Streamlit classification paths passed AppTest; browser interaction and file upload still require manual rehearsal |
| Saved-model result reproducibility | verification/checkpoints.json | Verified: all eight configurations reproduce all 584 test labels; neural probabilities checked within tolerance |
| Data integrity | verification/data_audit.json | Verified: 3,893 actual original pixel hashes; 17,517 augmentation metadata records. Derivative image bytes not re-audited |
| Updated C3 experiment included | experiments/config3_results/; report p. 10; slides 10–11 | Checkpoint, selected epoch, frozen hash and source protocol audited |
| Word and PDF report; PDF for Moodle | Final_Report.docx and Final_Report.pdf | Provided; 17 rendered pages reviewed. Moodle submission is not performed |
| Presentation covers objective, methods, findings, application | Final_Presentation.pptx; DEFENSE.md | Provided: 20 slides, speaker notes and equal speaking plan |
| All members actively participate and explain their work | Report p. 14; slide 17; DEFENSE.md | Responsibility plan provided; actual participation and understanding must be demonstrated by the team |
| Originality and contribution | Source and data attribution; explicit historical evidence boundaries | No fabricated contribution history or plagiarism certification; instructor assessment remains external |
| A repository the team can show | Local Git repository, README, code, checkpoints, CI workflow and docs | Local package complete; remote publication blocked by invalid GitHub authentication |

## Boundaries that must stay visible

- C1 has archived metrics and history, but its weights were not supplied. It is not counted among the eight checkpoint-verified configurations.
- Assignment 2's stored augmentation is a prepared extension. None of the reported model runs uses its 16,349 training files; its effect on accuracy remains untested.
- The original label-count mismatch and potential same-building overlap cannot be resolved from these files. The study reports these as limitations, not as passed controls.
- The guide's listed weights (25 + 10 + 20 + 15 + 20) sum to 90 although its total says 100. No missing ten-point criterion is invented.
- Native PowerPoint opening, a live browser/file-upload demo, team defense, remote CI execution and Moodle upload are not certified by package checks.

## Before presenting

Follow DEFENSE.md on the actual presentation laptop. Open the PDF and PPTX; run a bundled-image prediction, one upload, model switching and the browser explorer. Each member explains their assigned section and the limitations. Publish the prepared Git repository after restoring GitHub authentication. These are explicit remaining actions, not hidden claims of 100% completion.
