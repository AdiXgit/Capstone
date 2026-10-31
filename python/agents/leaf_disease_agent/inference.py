"""Serving-side wrapper around the trained leaf-disease checkpoint."""
import io
import os

import torch

from PIL import Image

# _normalise_height is imported rather than reimplemented: serving must apply
# the exact same height normalisation as training, and two copies would
# silently diverge.
from dataset import CLASS_NAMES, _normalise_height, tta_crops
from model import build_model

BASE = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT = os.environ.get(
    "LEAF_DISEASE_CHECKPOINT", os.path.join(BASE, "models", "leaf_disease_cnn.pt")
)

# Below this averaged probability the crop is reported as inconclusive rather
# than as a diagnosis. A wrong confident diagnosis costs a farmer a spray.
MIN_CONFIDENCE = float(os.environ.get("LEAF_DISEASE_MIN_CONFIDENCE", "0.55"))


class LeafDiseaseClassifier:
    def __init__(self, checkpoint=CHECKPOINT, n_crops=5):
        if not os.path.exists(checkpoint):
            raise FileNotFoundError(
                f"No checkpoint at {checkpoint}. Run train.py first."
            )
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.class_names = ckpt.get("class_names", CLASS_NAMES)
        self.cv_accuracy = ckpt.get("cv_accuracy")
        self.n_crops = n_crops

        # Built as "scratch" so torchvision does not fetch ImageNet weights we
        # are about to overwrite; the architecture is identical either way.
        net, _, _ = build_model("scratch", arch=ckpt.get("arch", "resnet18"),
                                num_classes=len(self.class_names))
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        self.net = net

    @torch.no_grad()
    def predict(self, image_bytes):
        with Image.open(io.BytesIO(image_bytes)) as img:
            strip = _normalise_height(img.convert("RGB"))

        crops = tta_crops(strip, n_crops=self.n_crops)
        probs = torch.softmax(self.net(crops), dim=1).mean(0)

        conf, idx = float(probs.max()), int(probs.argmax())
        return {
            "disease": self.class_names[idx],
            "confidence": round(conf, 4),
            "inconclusive": conf < MIN_CONFIDENCE,
            "probabilities": {
                name: round(float(p), 4) for name, p in zip(self.class_names, probs)
            },
            "crops_averaged": crops.shape[0],
            "model_cv_accuracy": self.cv_accuracy,
        }
