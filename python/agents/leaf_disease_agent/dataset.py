"""Leaf-strip dataset for the rice leaf disease images.

Two preprocessing facts drive this module:

1. Images are long leaf strips (~3.4:1), so a square crop sampled along the
   strip is both correct framing and effectively free augmentation.
2. Every Bacterial-leaf-blight image is exactly 3081x897 while the other two
   classes have mixed resolutions. Absolute resolution therefore leaks the
   label, so images are height-normalised before any crop is taken and the
   original pixel dimensions never reach the model.
"""
import os
import glob

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

STRIP_HEIGHT = 256
CROP_SIZE = 224

CLASS_NAMES = ["Bacterial leaf blight", "Brown spot", "Leaf smut"]


def discover_images(root):
    """Return (paths, labels) for the class folders under root."""
    paths, labels = [], []
    for idx, name in enumerate(CLASS_NAMES):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing class folder: {folder}")
        found = sorted(
            p for p in glob.glob(os.path.join(folder, "*"))
            if p.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not found:
            raise FileNotFoundError(f"No images in {folder}")
        paths.extend(found)
        labels.extend([idx] * len(found))
    return paths, labels


def _normalise_height(img):
    """Resize so height == STRIP_HEIGHT, preserving aspect ratio."""
    w, h = img.size
    new_w = max(CROP_SIZE, round(w * STRIP_HEIGHT / h))
    return img.resize((new_w, STRIP_HEIGHT), Image.BILINEAR)


def train_transform():
    return transforms.Compose([
        transforms.RandomCrop(CROP_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=12, translate=(0.04, 0.04), shear=6),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def load_strip(path):
    """Open an image and height-normalise it. Shared by training and serving."""
    with Image.open(path) as img:
        return _normalise_height(img.convert("RGB"))


def tta_crops(strip, n_crops=5):
    """Evenly spaced square crops along a height-normalised strip.

    A single centre crop sees only ~25% of a 3.4:1 leaf, so averaging logits
    over crops is how a whole leaf gets classified.
    """
    width = strip.size[0]
    span = max(0, width - CROP_SIZE)
    if n_crops == 1 or span == 0:
        offsets = [span // 2]
    else:
        offsets = [round(i * span / (n_crops - 1)) for i in range(n_crops)]

    # Square, vertically centred: training uses RandomCrop(CROP_SIZE), so
    # cropping the full STRIP_HEIGHT here would feed the model a taller input
    # than it ever saw while training.
    top = (STRIP_HEIGHT - CROP_SIZE) // 2
    tf = eval_transform()
    return torch.stack([
        tf(strip.crop((x, top, x + CROP_SIZE, top + CROP_SIZE))) for x in offsets
    ])


class LeafCropDataset(Dataset):
    """Yields one randomly-cropped square per access.

    `repeats` controls how many crops per epoch are drawn from each image,
    which is how 40 images per class turn into a usable number of gradient
    steps without re-reading JPEGs from disk.
    """

    def __init__(self, paths, labels, repeats=1):
        self.paths = list(paths)
        self.labels = list(labels)
        self.repeats = repeats
        self.transform = train_transform()
        self._strips = [load_strip(p) for p in self.paths]

    def __len__(self):
        return len(self.paths) * self.repeats

    def __getitem__(self, i):
        idx = i % len(self.paths)
        # int() not the raw numpy scalar: numpy defaults to int32 on Windows,
        # which collates to a tensor CrossEntropyLoss rejects.
        return self.transform(self._strips[idx]), int(self.labels[idx])
