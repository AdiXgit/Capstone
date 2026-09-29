"""YOLOv8 image-based paddy leaf disease detection.

This is the *image* half of the Crop Health agent. The tabular half (data_store.py)
scores stress from NDVI/LAI/DAS; this module classifies a photo of a leaf into a
paddy disease using an Ultralytics YOLOv8 classification model, then maps the
prediction to a KVK treatment (disease_treatment.py).

Design notes
------------
* Heavy deps (ultralytics/torch) are imported lazily so the rest of the agent and
  the REST gateway keep running even when the vision stack is not installed. Hitting
  the detector without the deps returns a clear, actionable error instead of a crash.
* Model resolution order:
    1. $YOLO_MODEL_PATH if set and present
    2. models/paddy_disease_cls.pt  (produced by scripts/train_yolo_classifier.py)
    3. yolov8n-cls.pt               (pretrained; Ultralytics auto-downloads it)
  Only (1)/(2) know paddy classes. With (3) the pipeline works end-to-end but the
  labels are generic ImageNet classes — `is_finetuned` is False and predictions are
  flagged as not paddy-specific so the UI can say so honestly.
"""
from __future__ import annotations

import os
import io
import time
import logging

import numpy as np

log = logging.getLogger("CROP_HEALTH.disease_detection")

# Leaf-validity gate (RGB vegetation index). A classifier always returns *some*
# class with high confidence, even for a face — so before trusting a disease
# prediction we check the image actually contains foliage. We use an NDVI-style
# index computed from RGB (true NDVI needs a near-infrared band a phone lacks):
#   ExG (Excess Green) = 2g - r - b on normalised channels; green_fraction is the
# share of green-dominant pixels. Calibrated on the val set + non-leaf images:
#   faces/rooms ≈ 0.000–0.002; healthy/brown-spot/leaf-blast leaves ≈ 0.10 median.
LEAF_REJECT = 0.006   # below this → treated as "not a leaf" (face crop was 0.0015)
LEAF_CAUTION = 0.025  # below this → classify but warn "low vegetation, verify"


def vegetation_index(rgb_image) -> dict:
    """RGB vegetation metrics used as the leaf-validity gate. Returns green_fraction
    (0-1), mean ExG, and mean VARI (an NDVI-like ratio) over vegetation pixels."""
    arr = np.asarray(rgb_image.resize((256, 256))).astype(np.float32)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]
    total = R + G + B + 1e-6
    r, g, b = R / total, G / total, B / total
    exg = 2 * g - r - b
    veg = (exg > 0.02) & (G >= R)
    green_fraction = float(veg.mean())
    denom = G + R - B
    vari = np.where(np.abs(denom) > 1e-3, (G - R) / denom, 0.0)
    vari = np.clip(vari, -1.0, 1.0)
    mean_vari = float(vari[veg].mean()) if veg.any() else 0.0
    return {
        "green_fraction": round(green_fraction, 4),
        "mean_exg": round(float(exg.mean()), 4),
        "mean_vari": round(mean_vari, 4),
    }

_BASE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_BASE, "../../.."))
_FINETUNED_PATH = os.path.join(_REPO, "models", "paddy_disease_cls.pt")
_PRETRAINED = "yolov8n-cls.pt"

try:
    from disease_treatment import lookup as _lookup_treatment
except ImportError:  # pragma: no cover - allows import from other cwd
    from .disease_treatment import lookup as _lookup_treatment


class DetectorUnavailable(RuntimeError):
    """Raised when the vision stack (ultralytics/torch) is not installed."""


class PaddyDiseaseDetector:
    """Lazy-loading wrapper around an Ultralytics YOLOv8-cls model."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or os.environ.get("YOLO_MODEL_PATH")
        if not self.model_path:
            self.model_path = _FINETUNED_PATH if os.path.exists(_FINETUNED_PATH) else _PRETRAINED
        self.is_finetuned = self.model_path not in (_PRETRAINED,) and os.path.exists(self.model_path)
        self._model = None  # lazy

    # ── model loading ────────────────────────────────────────────

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO  # heavy: torch + ultralytics
        except Exception as e:  # ImportError or torch load issues
            raise DetectorUnavailable(
                "Vision dependencies not installed. Run "
                "`pip install -r requirements-vision.txt` to enable image disease detection. "
                f"(import error: {e})"
            ) from e
        t0 = time.time()
        log.info("Loading YOLO model: %s (finetuned=%s)", self.model_path, self.is_finetuned)
        self._model = YOLO(self.model_path)
        log.info("YOLO model loaded in %.2fs", time.time() - t0)

    @property
    def available(self) -> bool:
        """True when the vision stack can be imported (does not load the model)."""
        try:
            import ultralytics  # noqa: F401
            return True
        except Exception:
            return False

    # ── inference ────────────────────────────────────────────────

    def predict(self, image_bytes: bytes, top_k: int = 3) -> dict:
        """Classify one leaf image.

        Returns a dict with the top prediction, top-k list, the mapped KVK
        treatment and provenance. Raises DetectorUnavailable if deps are missing,
        or ValueError if the bytes are not a readable image.
        """
        try:
            from PIL import Image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise ValueError(f"Could not read image: {e}") from e

        # ── Leaf-validity gate (before trusting the classifier) ──
        veg = vegetation_index(image)
        gf = veg["green_fraction"]
        if gf < LEAF_REJECT:
            return {
                "is_leaf": False,
                "vegetation_index": gf,
                "vegetation": veg,
                "predicted_disease": None,
                "confidence": None,
                "predictions": [],
                "message": (f"No paddy foliage detected (vegetation index {gf*100:.1f}% "
                            f"< {LEAF_REJECT*100:.1f}%). Point the camera at a paddy leaf so it "
                            f"fills the frame, in good light."),
                "model": {"path": os.path.basename(self.model_path),
                          "is_finetuned": self.is_finetuned,
                          "note": "Input rejected by vegetation gate — not classified."},
                "latency_ms": 0.0,
            }

        self._ensure_loaded()
        t0 = time.time()
        results = self._model.predict(image, verbose=False)
        latency_ms = round((time.time() - t0) * 1000, 1)

        r = results[0]
        names = r.names  # {idx: label}
        probs = r.probs
        # top-k indices, highest first
        k = min(top_k, len(names))
        top_idx = probs.top5[:k] if hasattr(probs, "top5") else [int(probs.top1)]
        conf_all = probs.data.tolist() if hasattr(probs, "data") else None

        predictions = []
        for idx in top_idx:
            raw = names[int(idx)]
            conf = float(conf_all[int(idx)]) if conf_all else float(probs.top1conf)
            info = _lookup_treatment(raw)
            predictions.append({
                "raw_label": raw,
                "disease": info["label"],
                "confidence": round(conf, 4),
                "severity": info["severity"],
                "treatment": info["treatment"],
                "is_paddy_condition": info["matched"],
            })

        top = predictions[0]
        low_veg = gf < LEAF_CAUTION
        return {
            "is_leaf": True,
            "vegetation_index": gf,
            "vegetation": veg,
            "low_vegetation": low_veg,
            "vegetation_note": (
                f"Vegetation index {gf*100:.1f}% — low; verify this is a clear leaf shot."
                if low_veg else f"Vegetation index {gf*100:.1f}% — leaf confirmed."),
            "predicted_disease": top["disease"],
            "confidence": top["confidence"],
            "severity": top["severity"],
            "treatment": top["treatment"],
            "is_paddy_condition": top["is_paddy_condition"],
            "predictions": predictions,
            "model": {
                "path": os.path.basename(self.model_path),
                "is_finetuned": self.is_finetuned,
                "note": ("Fine-tuned on the paddy disease dataset."
                         if self.is_finetuned else
                         "Base pretrained model — labels are generic. Run "
                         "scripts/train_yolo_classifier.py for paddy-specific accuracy."),
            },
            "latency_ms": latency_ms,
        }


    def explain(self, image_bytes: bytes, grid: int = 8, work: int = 224) -> dict:
        """Occlusion saliency: slide a gray patch over the image and measure how much
        the top class's confidence drops — high drop = region the model relies on.
        Returns a base64 PNG heat-overlay. Model-agnostic (no Grad-CAM hooks), so it
        stays robust across YOLO versions. ~grid*grid forward passes."""
        self._ensure_loaded()
        from PIL import Image
        import base64
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        base = self._model.predict(img, verbose=False)[0]
        top_idx = int(base.probs.top1)
        top_name = base.names[top_idx]
        base_conf = float(base.probs.data[top_idx])

        w, h = img.size
        arr = np.asarray(img.resize((work, work))).copy()
        cell = work // grid
        heat = np.zeros((grid, grid), dtype=np.float32)
        for gy in range(grid):
            for gx in range(grid):
                occ = arr.copy()
                occ[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell] = 128
                p = self._model.predict(Image.fromarray(occ), verbose=False)[0]
                heat[gy, gx] = max(0.0, base_conf - float(p.probs.data[top_idx]))

        if heat.max() > 0:
            heat /= heat.max()
        heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        heat_np = np.asarray(heat_img).astype(np.float32) / 255.0
        overlay = np.asarray(img).astype(np.float32)
        red = np.zeros_like(overlay)
        red[..., 0] = 255.0
        alpha = (heat_np * 0.55)[..., None]
        blended = (overlay * (1 - alpha) + red * alpha).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(blended).save(buf, "PNG")
        info = _lookup_treatment(top_name)
        return {
            "top_class": info["label"],
            "grid": grid,
            "overlay_base64": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
        }


# Process-wide singleton so the model is loaded once and reused.
_DETECTOR: PaddyDiseaseDetector | None = None


def get_detector() -> PaddyDiseaseDetector:
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = PaddyDiseaseDetector()
    return _DETECTOR
