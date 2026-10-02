"""
One-time dataset cleanup: rename every image/label file to just its
unique Roboflow hash + extension, stripping out the original
(often non-English) source filename entirely.

Why: Roboflow dataset filenames look like:
    some-cyrillic-or-greek-name_jpg.rf.<32-char-hash>.jpg
The <hash> part is already a unique identifier for each image, shared
between the image and its matching label file. Everything before
".rf." is just leftover from the original source filename and isn't
needed for training -- and non-English characters in it have been
causing repeated encoding corruption when moving between Windows and
WSL filesystems.

After running this, every file becomes:
    <hash>.jpg   (in images/)
    <hash>.txt   (in labels/)
which is guaranteed ASCII-safe and keeps image/label pairs matched
correctly (since Roboflow uses the same hash for both).

Safe to run multiple times -- already-renamed files are skipped.
"""

import os
import re
from pathlib import Path

HASH_PATTERN = re.compile(r"\.rf\.([0-9a-fA-F]{32})", re.IGNORECASE)


def sanitize_folder(images_dir: Path, labels_dir: Path):
    renamed = 0
    skipped = 0
    no_match = []

    for img_path in sorted(images_dir.iterdir()):
        if not img_path.is_file():
            continue

        name = img_path.name
        match = HASH_PATTERN.search(name)
        if not match:
            # Already renamed (no ".rf.<hash>" pattern left), or unexpected format
            no_match.append(name)
            skipped += 1
            continue

        file_hash = match.group(1).lower()
        ext = img_path.suffix  # e.g. '.jpg'
        new_img_name = f"{file_hash}{ext}"
        new_img_path = images_dir / new_img_name

        # Find the matching label file (same original base name, .txt extension)
        label_name = img_path.stem + ".txt"
        # img_path.stem strips only the LAST extension; since filenames have
        # patterns like "name_jpg.rf.hash.jpg", stem removes the final ".jpg"
        old_label_path = labels_dir / label_name
        new_label_path = labels_dir / f"{file_hash}.txt"

        if not old_label_path.exists():
            print(f"  WARNING: no matching label for {name} (expected {label_name}) -- skipping this pair")
            skipped += 1
            continue

        img_path.rename(new_img_path)
        old_label_path.rename(new_label_path)
        renamed += 1

    return renamed, skipped, no_match


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sanitize dataset filenames (strip non-ASCII names, keep hash)")
    parser.add_argument("--dataset", required=True, help="Path to dataset root (containing train/valid/test)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    total_renamed = 0
    total_skipped = 0

    for split in ["train", "valid", "test"]:
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"

        if not images_dir.exists():
            print(f"Skipping '{split}' -- folder not found at {images_dir}")
            continue

        print(f"\nProcessing '{split}'...")
        renamed, skipped, no_match = sanitize_folder(images_dir, labels_dir)
        total_renamed += renamed
        total_skipped += skipped
        print(f"  Renamed: {renamed}, Skipped/already-clean: {skipped}")
        if no_match:
            print(f"  {len(no_match)} files had no '.rf.<hash>' pattern (likely already renamed)")

    print(f"\nDone. Total renamed: {total_renamed}, Total skipped: {total_skipped}")
    print("All filenames are now ASCII-safe hash-based names.")
    print("Re-run training now -- the encoding issue should be gone for good.")
