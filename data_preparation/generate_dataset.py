"""Build deterministic BD3 augmentations after the original train/val/test split.

Requires Pillow. Originals are copied byte for byte. Only training
images receive derivatives. This script does not train or alter any model.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps


VARIANTS = (
    "rotate_minus08", "rotate_plus08", "flip_horizontal", "blur_mild", "lighting"
)
FIELDS = (
    "path", "label", "split", "source_id", "source_path", "augmentation",
    "parameters", "width", "height", "output_pixel_hash", "file_sha256"
)


def pixel_hash(image):
    # The same convention used by the supplied project's split manifest.
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def rotation(image, degrees):
    # A conservative central square removes exposed rotation corners and source
    # frame edges. Avoid mirrored padding, which can reflect black frame lines
    # into the image and make them look like defects.
    width, height = image.size
    radians = math.radians(abs(degrees))
    side = math.floor(min(width, height) / (math.cos(radians) + math.sin(radians))) - 8
    left, top = (width - side) // 2, (height - side) // 2
    turned = image.rotate(degrees, resample=Image.Resampling.BICUBIC, expand=False)
    cropped = turned.crop((left, top, left + side, top + side))
    return cropped.resize(image.size, Image.Resampling.LANCZOS), side


def augment(image, kind, source_id):
    if kind.startswith("rotate_"):
        degrees = -8 if kind == "rotate_minus08" else 8
        changed, crop_side = rotation(image, degrees)
        return changed, {"degrees_counterclockwise": degrees,
                         "central_crop_pixels": crop_side,
                         "resized_back_to_original_dimensions": True}
    if kind == "flip_horizontal":
        return ImageOps.mirror(image), {"axis": "horizontal"}
    if kind == "blur_mild":
        return image.filter(ImageFilter.GaussianBlur(radius=0.6)), {"gaussian_radius_px": 0.6}
    if kind == "lighting":
        # Derived from the source content, independent of processing order.
        selector = int(source_id[:8], 16)
        brightness = 0.9 if selector % 2 == 0 else 1.1
        contrast = 0.95 if (selector // 2) % 2 == 0 else 1.05
        changed = ImageEnhance.Brightness(image).enhance(brightness)
        return ImageEnhance.Contrast(changed).enhance(contrast), {
            "brightness_factor": brightness, "contrast_factor": contrast
        }
    raise ValueError(kind)


def read_sources(dataset_root, manifest):
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    required = {"path", "label", "pixel_hash", "width", "height", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("The source manifest is empty or missing required columns.")
    seen_paths, seen_hashes = set(), set()
    for row in rows:
        rel = Path(row["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe source path: {rel}")
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"Unexpected split: {row['split']}")
        if Path(row["label"]).name != row["label"] or row["label"] in {".", ".."}:
            raise ValueError("Invalid class name.")
        if row["path"] in seen_paths or row["pixel_hash"] in seen_hashes:
            raise ValueError("Source manifest contains duplicate paths or pixel hashes.")
        seen_paths.add(row["path"])
        seen_hashes.add(row["pixel_hash"])
        if not (dataset_root / rel).is_file():
            raise FileNotFoundError(dataset_root / rel)
    return rows


def process_source(row, dataset_root, output_dir):
    source = dataset_root / row["path"]
    with Image.open(source) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB")
    source_id = pixel_hash(image)
    if source_id != row["pixel_hash"]:
        raise ValueError(f"Source content differs from the original manifest: {source}")
    if image.size != (int(row["width"]), int(row["height"])):
        raise ValueError(f"Source dimensions differ from the manifest: {source}")
    destination_dir = output_dir / row["split"] / row["label"]
    destination_dir.mkdir(parents=True, exist_ok=True)
    original = destination_dir / source.name
    if original.exists():
        raise FileExistsError(original)
    shutil.copyfile(source, original)
    source_bytes_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if hashlib.sha256(original.read_bytes()).hexdigest() != source_bytes_hash:
        raise ValueError(f"Original copy failed byte verification: {original}")

    def record(path, kind, parameters, decoded, file_digest):
        return {
            "path": path.relative_to(output_dir).as_posix(), "label": row["label"],
            "split": row["split"], "source_id": source_id, "source_path": row["path"],
            "augmentation": kind, "parameters": json.dumps(parameters, sort_keys=True),
            "width": decoded.width, "height": decoded.height,
            "output_pixel_hash": pixel_hash(decoded), "file_sha256": file_digest,
        }

    records = [record(original, "original", {}, image, source_bytes_hash)]
    skipped = []
    if row["split"] == "train":
        for kind in VARIANTS:
            changed, parameters = augment(image, kind, source_id)
            target = destination_dir / f"{source.stem}__aug_{kind}.jpg"
            if target.exists():
                raise FileExistsError(target)
            changed.save(target, "JPEG", quality=95, subsampling=0)
            # Check the actual saved image, including JPEG decoding.
            with Image.open(target) as saved:
                saved.load()
                if saved.mode != "RGB" or saved.size != image.size:
                    raise ValueError(f"Invalid saved variant: {target}")
                if pixel_hash(saved) == source_id:
                    skipped.append({"source_path": row["path"], "source_id": source_id,
                                    "augmentation": kind, "reason": "identical_to_own_original"})
                else:
                    parameters.update({"jpeg_quality": 95, "jpeg_subsampling": 0})
                    records.append(record(target, kind, parameters, saved,
                                          hashlib.sha256(target.read_bytes()).hexdigest()))
            if skipped and skipped[-1]["augmentation"] == kind:
                target.unlink()
    return records, skipped


def font(size):
    return ImageFont.load_default(size=size)


def make_preview(records, output_dir):
    by_source = defaultdict(dict)
    for row in records:
        if row["split"] == "train":
            by_source[row["source_id"]][row["augmentation"]] = row
    classes = sorted({r["label"] for r in records})
    kinds = ("original",) + VARIANTS
    selected = []
    for label in classes:
        families = [v for v in by_source.values() if v["original"]["label"] == label]
        selected.append(next((v for v in families if all(k in v for k in kinds)), families[0]))
    headings = ("Original", "Rotate -8 degrees", "Rotate +8 degrees", "Horizontal flip",
                "Mild blur", "Lighting change")
    cell, gap, left, top, row_label = 224, 10, 130, 88, 10
    sheet = Image.new("RGB", (left + 6 * (cell + gap), top + 7 * (cell + row_label + gap)), "#f7f8fa")
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 14), "BD3 training image augmentation", fill="#17202a", font=font(26))
    draw.text((18, 46), "One training source per class. Validation and test originals are unchanged.",
              fill="#45505c", font=font(17))
    for j, title in enumerate(headings):
        draw.text((left + j * (cell + gap), top - 23), title, fill="#17202a", font=font(15))
    for i, variants in enumerate(selected):
        y = top + i * (cell + row_label + gap)
        label = variants["original"]["label"].replace("_", " ").title()
        draw.text((12, y + cell // 2 - 10), label, fill="#17202a", font=font(17))
        for j, kind in enumerate(kinds):
            if kind not in variants:
                draw.text((left + j * (cell + gap) + 8, y + 90),
                          "Duplicate omitted", fill="#45505c", font=font(17))
                continue
            with Image.open(output_dir / variants[kind]["path"]) as im:
                sheet.paste(im.resize((cell, cell), Image.Resampling.LANCZOS),
                            (left + j * (cell + gap), y))
    preview_dir = output_dir / "previews"
    preview_dir.mkdir()
    sheet.save(preview_dir / "augmentation_overview.jpg", quality=95)
    minor = selected[classes.index("minor_crack")]
    detail = Image.new("RGB", (3 * 512, 2 * 550 + 55), "#f7f8fa")
    draw = ImageDraw.Draw(detail)
    draw.text((15, 12), "Minor crack: full resolution augmentation check", fill="#17202a", font=font(27))
    for j, kind in enumerate(kinds):
        x, y = (j % 3) * 512, 55 + (j // 3) * 550
        draw.text((x + 12, y), headings[j], fill="#17202a", font=font(22))
        if kind not in minor:
            draw.text((x + 12, y + 220), "Duplicate omitted", fill="#45505c", font=font(24))
            continue
        with Image.open(output_dir / minor[kind]["path"]) as im:
            detail.paste(im, (x, y + 32))
    detail.save(preview_dir / "minor_crack_detail.jpg", quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="A new destination; existing directories are never overwritten.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    rows = read_sources(dataset_root, args.split_manifest)
    if output_dir.exists():
        parser.error("--output-dir already exists. Use a new destination.")
    # Detect filename collisions before writing any image.
    planned = set()
    for row in rows:
        name = Path(row["path"])
        names = [name.name]
        if row["split"] == "train":
            names += [f"{name.stem}__aug_{kind}.jpg" for kind in VARIANTS]
        for filename in names:
            rel = (row["split"], row["label"], filename)
            if rel in planned:
                raise ValueError(f"Output filename collision: {rel}")
            planned.add(rel)
    output_dir.mkdir(parents=True)
    (output_dir / "BUILD_INCOMPLETE.txt").write_text(
        "Build is incomplete. Do not use this dataset until summary.json is present.\n"
    )
    records, skipped = [], []
    print(f"Checking {len(rows):,} originals and generating five variants per training source...", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        batches = pool.map(lambda r: process_source(r, dataset_root, output_dir), rows)
        for index, (batch, omissions) in enumerate(batches, 1):
            records.extend(batch)
            skipped.extend(omissions)
            if index % 250 == 0 or index == len(rows):
                print(f"Verified sources {index:,}/{len(rows):,}; output images {len(records):,}", flush=True)

    # Reserve hashes of all originals first, including validation/test. Remove
    # duplicate variants instead of moving or modifying an original source.
    seen_output_hashes = {r["output_pixel_hash"] for r in records if r["augmentation"] == "original"}
    distinct_records = []
    for row in records:
        if row["augmentation"] == "original":
            distinct_records.append(row)
        elif row["output_pixel_hash"] in seen_output_hashes:
            (output_dir / row["path"]).unlink()
            skipped.append({"source_path": row["source_path"], "source_id": row["source_id"],
                            "augmentation": row["augmentation"], "reason": "identical_to_another_output"})
        else:
            seen_output_hashes.add(row["output_pixel_hash"])
            distinct_records.append(row)
    records = distinct_records
    source_splits = defaultdict(set)
    hashes = defaultdict(list)
    for row in records:
        source_splits[row["source_id"]].add(row["split"])
        hashes[row["output_pixel_hash"]].append(row)
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise ValueError("An original source or derivative appears in multiple splits.")
    if any(len({r["split"] for r in group}) > 1 for group in hashes.values()):
        raise ValueError("Exact decoded duplicate across output splits.")
    duplicate_groups = [group for group in hashes.values() if len(group) > 1]
    if duplicate_groups:
        raise ValueError(f"Unexpected exact duplicates in output: {len(duplicate_groups)} groups.")
    expected = Counter(r["split"] for r in rows)
    expected["train"] *= 1 + len(VARIANTS)
    expected["train"] -= len(skipped)
    actual = Counter(r["split"] for r in records)
    if actual != expected or len(planned) != len(records) + len(skipped):
        raise ValueError("Output counts do not match the planned augmentation.")

    with (output_dir / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    shutil.copyfile(args.split_manifest, output_dir / "source_split_manifest.csv")
    with (output_dir / "skipped_variants.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source_path", "source_id", "augmentation", "reason"))
        writer.writeheader()
        writer.writerows(skipped)
    classes = sorted({r["label"] for r in rows})
    counts = Counter((r["label"], r["split"]) for r in records)
    originals = Counter((r["label"], r["split"]) for r in rows)
    class_rows = []
    for label in classes:
        train_originals = originals[label, "train"]
        class_rows.append({"class": label, "training_originals": train_originals,
                           "training_variants": counts[label, "train"] - train_originals,
                           "training_total": counts[label, "train"],
                           "validation": counts[label, "validation"], "test": counts[label, "test"]})
    with (output_dir / "class_counts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=class_rows[0].keys())
        writer.writeheader()
        writer.writerows(class_rows)
    make_preview(records, output_dir)
    summary = {
        "status": "complete", "source_dataset": str(dataset_root),
        "source_manifest": str(args.split_manifest.resolve()), "classes": classes,
        "unique_original_images": len(rows),
        "training_originals": sum(r["split"] == "train" for r in rows),
        "new_training_variants": sum(r["augmentation"] != "original" for r in records),
        "split_counts": dict(actual), "total_images": len(records),
        "attempted_variants_per_training_source": len(VARIANTS), "variants": VARIANTS,
        "skipped_duplicate_variants": len(skipped),
        "skipped_reasons": dict(Counter(r["reason"] for r in skipped)),
        "class_counts": class_rows,
        "checks": {"all_source_pixel_hashes_match_existing_manifest": True,
                   "all_original_copies_byte_identical": True,
                   "all_saved_variants_decoded_and_size_checked": True,
                   "no_source_group_crosses_splits": True,
                   "no_exact_decoded_output_duplicates": True,
                   "only_training_sources_augmented": True},
        "evaluation_note": "Derived images are not independent new photographs. Keep the fixed validation/test sets. For cross-validation, split originals into folds first, train using derivatives of training-fold sources only, and score original held-out-fold images only.",
        "model_training_performed": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output_dir / "BUILD_INCOMPLETE.txt").unlink()
    print(json.dumps({"status": "complete", "split_counts": dict(actual),
                      "new_variants": summary["new_training_variants"],
                      "total_images": len(records)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
