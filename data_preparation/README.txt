FINAL REPOSITORY NOTE
This folder retains the generator and metadata only. Archived dataset/*.csv and
dataset/*.json are stored directly here; generated image and preview folders are
not included. Regenerate them with the script using the original images.
No reported model was trained on these stored variants.

BD3 augmented training dataset

This package adds up to five controlled variants of each training photograph.
It preserves the existing class labels and training, validation, and test split.
The variants are transformed copies, not newly collected photographs. They do
not increase the number of independent scenes or guarantee better predictions.

DATASET COUNTS

The cleaned source dataset has 3,893 photographs: 2,725 training, 584 validation,
and 584 test images. The script attempts five variants per training photograph.
Variants identical to an original or another retained image are excluded.
This run retained 13,624 variants. One blur variant was identical to its plain
source photograph and was skipped.

Training:    16,349 files = 2,725 originals + 13,624 augmented variants
Validation:     584 original files
Test:           584 original files
Total:       17,517 files from 3,893 unique source photographs

The counts are recorded in dataset/summary.json and dataset/class_counts.csv.

The seven labels are algae, major_crack, minor_crack, peeling, plain, spalling,
and stain. Every original training image is retained. Skipped duplicate variants
can cause small changes in the training class proportions.

IMAGE VARIANTS

Each training photograph has its original and up to five retained variants.
The six possible versions are also the columns in the overview preview:

original          Original photograph, copied without changing its file bytes
rotate_minus08    Rotation of -8 degrees, followed by a centre crop and resize
rotate_plus08     Rotation of +8 degrees, followed by a centre crop and resize
flip_horizontal   Horizontal mirror image
blur_mild         Mild Gaussian blur, radius 0.6 pixels
lighting          Small, reproducible brightness and contrast changes

Images remain 512 by 512 pixels and are stored as JPEG files. Rotation uses a
centred crop of about 445 by 445 pixels, then resizes it to 512 by 512, producing
a mild zoom. This avoids empty rotation corners and removes the source image's
outer border. Some features near the edges can be cropped, so review these
variants before training; full preservation of edge defects is not guaranteed.

Validation and test files are copied unchanged. Transform parameters are
recorded for each output in dataset/manifest.csv. The lighting variant applies
brightness of 0.9 or 1.1 and contrast of 0.95 or 1.05. These choices are seeded
from each source image's
pixel hash, so they remain the same when the processing order changes.
Labels are inherited from the source photographs;
the augmented labels have not been independently re-annotated or validated.
Visual checks cover samples rather than every generated image. Review the
previews, especially minor_crack_detail.jpg, before training.

PACKAGE CONTENTS

dataset/train/<class>/       Training originals and augmented variants
dataset/validation/<class>/  Unchanged validation photographs
dataset/test/<class>/        Unchanged test photographs
generate_dataset.py         Reproducible generation script
dataset/manifest.csv        Output records, original-source grouping, and hashes
dataset/source_split_manifest.csv  Original cleaned-image list and fixed splits
dataset/summary.json        Generation settings and aggregate counts
dataset/class_counts.csv    Counts by class and split
dataset/skipped_variants.csv  Excluded no-op or exact-duplicate variants
dataset/previews/augmentation_overview.jpg
dataset/previews/minor_crack_detail.jpg

The manifest records source_id, source_path, split, augmentation, parameters,
output_pixel_hash, and file_sha256, alongside output paths and class labels.
source_id links every augmented image to its original photograph. Keep this
grouping when changing the training workflow.

USING THIS DATASET

For the existing CNN or ViT setup, load dataset/train for training,
dataset/validation for validation and checkpoint selection, and dataset/test
only for the final evaluation. Do not randomly split all generated files together:
an original and its variants must never appear in different evaluation splits.

The old notebooks are not automatically updated to use this folder. No models
have been retrained, and existing model scores describe the earlier dataset.
If online augmentation remains enabled in a neural training notebook, consider
the combined transformation strength before also using these stored variants.

The classical notebook uses ordinary StratifiedKFold. Do not simply append the
augmented images to its input table and keep that procedure: related versions
of the same photograph could appear in both training and validation folds.

For classical cross-validation:
1. Create each fold using only original source photographs.
2. Add augmented variants only for sources assigned to that fold's training set.
3. Evaluate on the unchanged original photographs in the held-out fold.
4. Keep the fixed test set outside cross-validation and model selection.

Use the same source_id to track originals and their variants. Group-based
splitting can enforce this separation, but validation scores should still be
calculated on original photographs for comparison with the earlier results.
The package contains variants only for the original training split; validation
sources remain unaugmented if they are later included in a classical development
set. Any further augmentation must follow the same fold-first rule.

REPRODUCING THE DATASET

Requirements: Python with Pillow installed.

    python -m pip install pillow

From this package folder, run the following command with your actual paths.
Choose a separate output folder to keep this generated dataset intact.

    python generate_dataset.py --dataset-root "/path/to/BD3_original_dataset" --split-manifest "/path/to/midterm_project/results/split_manifest.csv" --output-dir "/path/to/new_augmented_dataset/dataset"

The source dataset root contains the original train/<class>/ image paths.
The original split manifest supplies the cleaned image list and fixed splits.
Keep both inputs available; this script does not download the original dataset.
The --output-dir is the data directory: it will contain train, validation, test,
the generated metadata files, and previews. That directory must not exist yet.

SOURCE AND REPORTING

Original dataset citation, as recorded in the existing project report:
Kottari, P., and Arjunan, P. (2024). BD3: Building Defects Detection Dataset for
Benchmarking Computer Vision Techniques for Automated Defect Identification.
DOI: 10.1145/3671127.3698789
Dataset repository: https://github.com/Praveenkottari/BD3-Dataset

In the assignment report, describe the 3,893 cleaned original photographs and
13,624 retained generated training variants separately. Explain that augmentation
simulates small changes in orientation, focus, and lighting. Report any effect
on model performance only after retraining and evaluating on the unchanged
original validation and test images.
