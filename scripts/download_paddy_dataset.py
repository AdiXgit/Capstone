#!/usr/bin/env python3
"""Download a public paddy/rice leaf disease image dataset and lay it out for
Ultralytics YOLOv8 classification training.

Output layout (what scripts/train_yolo_classifier.py expects):

    data/paddy_disease/
    ├── train/<class>/*.jpg
    └── val/<class>/*.jpg          (20% stratified holdout)

Source
------
Uses `kagglehub`, which downloads Kaggle datasets given a Kaggle API token.
Get a token once: kaggle.com → Account → "Create New API Token" → save
kaggle.json to ~/.kaggle/kaggle.json  (or export KAGGLE_USERNAME / KAGGLE_KEY).

Default dataset: `nirmalsankalana/rice-leaf-disease-image` (folder-per-class).
Override with --slug for any other class-folder dataset, e.g.
`dedeikhsandwisaputra/rice-leafs-disease-dataset`.

Usage
-----
    python scripts/download_paddy_dataset.py
    python scripts/download_paddy_dataset.py --slug <owner/dataset> --val-split 0.2
"""
import argparse
import os
import random
import shutil
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT_DIR = os.path.join(REPO, "data", "paddy_disease")
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_class_dirs(root: str) -> dict[str, list[str]]:
    """Walk `root` and collect image files grouped by their parent folder name.

    Works whether the dataset ships as <root>/<class>/*.jpg or nests the class
    folders one level deeper (a common Kaggle quirk).
    """
    classes: dict[str, list[str]] = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        imgs = [f for f in filenames if f.lower().endswith(IMG_EXT)]
        if not imgs:
            continue
        cls = os.path.basename(dirpath.rstrip("/"))
        classes.setdefault(cls, [])
        classes[cls].extend(os.path.join(dirpath, f) for f in imgs)
    return classes


def build_split(classes: dict[str, list[str]], val_split: float, seed: int = 42):
    random.seed(seed)
    if os.path.exists(OUT_DIR):
        print(f"Removing existing {OUT_DIR}")
        shutil.rmtree(OUT_DIR)

    total_train = total_val = 0
    for cls, files in sorted(classes.items()):
        files = sorted(files)
        random.shuffle(files)
        n_val = max(1, int(len(files) * val_split)) if len(files) > 1 else 0
        val_files, train_files = files[:n_val], files[n_val:]

        for split, split_files in (("train", train_files), ("val", val_files)):
            dst_dir = os.path.join(OUT_DIR, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for i, src in enumerate(split_files):
                ext = os.path.splitext(src)[1].lower()
                shutil.copy2(src, os.path.join(dst_dir, f"{cls}_{i:05d}{ext}"))

        total_train += len(train_files)
        total_val += len(val_files)
        print(f"  {cls:32s} train={len(train_files):5d}  val={len(val_files):5d}")

    print(f"\nDone. {len(classes)} classes → {total_train} train / {total_val} val images")
    print(f"Output: {OUT_DIR}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", default="nirmalsankalana/rice-leaf-disease-image",
                    help="Kaggle dataset slug (owner/dataset) with folder-per-class images")
    ap.add_argument("--val-split", type=float, default=0.2, help="Validation fraction (default 0.2)")
    ap.add_argument("--from-path", default=None,
                    help="Skip download; build the split from an already-extracted dataset directory")
    args = ap.parse_args()

    if args.from_path:
        src_root = args.from_path
    else:
        try:
            import kagglehub
        except ImportError:
            sys.exit("kagglehub not installed. Run: pip install -r requirements-vision.txt")
        print(f"Downloading Kaggle dataset: {args.slug}")
        try:
            src_root = kagglehub.dataset_download(args.slug)
        except Exception as e:
            sys.exit(
                f"\nDownload failed: {e}\n\n"
                "Most likely a missing Kaggle token. Fix with either:\n"
                "  • Save kaggle.json to ~/.kaggle/kaggle.json (chmod 600), or\n"
                "  • export KAGGLE_USERNAME=... KAGGLE_KEY=...\n"
                "Then re-run. Or download any rice-disease dataset manually and pass\n"
                "  --from-path /path/to/extracted/dataset\n"
            )
        print(f"Downloaded to: {src_root}")

    classes = find_class_dirs(src_root)
    if not classes:
        sys.exit(f"No class folders with images found under {src_root}")
    build_split(classes, args.val_split)


if __name__ == "__main__":
    main()
