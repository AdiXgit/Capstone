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

# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — botanical gate
# ─────────────────────────────────────────────────────────────────────────────
# A closed-set softmax classifier ALWAYS returns some class, and it is often
# *maximally* confident on inputs it has never seen. Measured on this model:
# a synthetic RGB-keyboard image scores top1=1.000 with entropy=0.003. So
# confidence can never be the only line of defence — the image itself has to be
# checked for plant material first.
#
# The old gate used a single cue (green_fraction from Excess-Green) with a 0.006
# floor. That is trivially defeated by anything green: an RGB-lit gaming keyboard
# measured green_fraction=0.381 — 63x over the threshold — and was labelled
# "Neck Blast, HIGH, 62%". The gate below uses four complementary cues, each
# calibrated against 400 held-out val leaves and a negative set (neon keyboards,
# flat green surfaces, dark rooms, skin, sky, UI screenshots).
#
# Cues and why each one is needed:
#   foliage_fraction  share of pixels whose HUE is in the botanical yellow-green
#                     -> green band with *natural* saturation. Excludes neon LEDs
#                     (saturation ~1.0), which plain Excess-Green happily counts.
#   brightness        mean HSV value. Real leaf photos are daylight shots
#                     (val p1 = 0.576); the keyboard scene was 0.173.
#   hue_dispersion    circular dispersion of hue over saturated pixels. Foliage is
#                     unimodal in hue (val median 0.010); RGB lighting sprays the
#                     whole hue circle (keyboard 0.739).
#   texture           mean abs. luminance gradient. Foliage has vein/lesion detail
#                     (val p1 = 0.0033); a flat painted wall or colour wash is 0.0.
#
# Thresholds sit below the 1st percentile of real leaves, so the false-reject rate
# on genuine paddy photos stays ~1-2% while every negative above is rejected.
# Neck-blast panicle shots can be legitimately brown (foliage_fraction ~0.0), so
# low greenness alone is NOT a hard reject — it only rejects when the image also
# lacks plant-like structure. Re-derive with scripts/calibrate_gate.py.
MIN_BRIGHTNESS = 0.40   # darker than any real val leaf (p1=0.576); dark desk=0.17
MAX_BRIGHTNESS = 0.97   # blown-out / pure white frame
MAX_HUE_DISPERSION = 0.65  # val p99=0.493; keyboard=0.739, neon rainbow=0.722
MIN_TEXTURE = 0.0020    # val p1=0.0033; flat colour fields = 0.0000
MIN_FOLIAGE = 0.015     # only enforced together with the structure check below
MIN_CHROMA = 0.020      # val p1=0.033 (all classes); a B/W UI screenshot = 0.000
MIN_PLANT_SCORE = 0.45  # composite score required to run the classifier

LEAF_CAUTION = 0.025    # foliage below this → classify but flag "low vegetation"


def vegetation_index(rgb_image) -> dict:
    """Multi-cue botanical metrics backing the plant gate.

    Returns the four calibrated cues plus the legacy Excess-Green figures, so
    callers (and the Model Performance page) can show why an image was rejected.
    """
    arr = np.asarray(rgb_image.convert("RGB").resize((256, 256))).astype(np.float32)
    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]

    # Legacy Excess-Green / VARI (kept for continuity of the reported numbers).
    total = R + G + B + 1e-6
    r, g, b = R / total, G / total, B / total
    exg = 2 * g - r - b
    veg = (exg > 0.02) & (G >= R)
    green_fraction = float(veg.mean())
    denom = G + R - B
    # Guard the divisor itself — np.where still evaluates both branches, so a raw
    # (G-R)/denom warns (and yields inf/nan) wherever denom is 0.
    safe = np.where(np.abs(denom) > 1e-3, denom, 1.0)
    vari = np.where(np.abs(denom) > 1e-3, (G - R) / safe, 0.0)
    vari = np.clip(vari, -1.0, 1.0)
    mean_vari = float(vari[veg].mean()) if veg.any() else 0.0

    # HSV without a PIL round-trip (vectorised, matches colorsys conventions).
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    chroma = mx - mn + 1e-6
    value = mx / 255.0
    sat = np.where(mx > 1e-6, chroma / (mx + 1e-6), 0.0)
    hue = np.zeros_like(mx)
    m = mx == R
    hue[m] = ((G - B)[m] / chroma[m]) % 6
    m = mx == G
    hue[m] = ((B - R)[m] / chroma[m]) + 2
    m = mx == B
    hue[m] = ((R - G)[m] / chroma[m]) + 4
    hue *= 60.0

    # Botanical hue band with natural saturation — excludes neon/LED greens.
    foliage = (hue >= 50) & (hue <= 160) & (sat > 0.12) & (sat < 0.95) \
        & (value > 0.10) & (value < 0.98)
    foliage_fraction = float(foliage.mean())

    # Circular dispersion of hue over colourful pixels: 0 = one hue, 1 = rainbow.
    colourful = sat > 0.25
    if int(colourful.sum()) > 50:
        h_rad = np.deg2rad(hue[colourful])
        resultant = float(np.hypot(np.cos(h_rad).mean(), np.sin(h_rad).mean()))
        hue_dispersion = float(np.clip(1.0 - resultant, 0.0, 1.0))
    else:
        hue_dispersion = 0.0

    # Mean absolute luminance gradient — flat synthetic fields score ~0.
    lum = (0.299 * R + 0.587 * G + 0.114 * B) / 255.0
    texture = float((np.abs(np.diff(lum, axis=1)).mean()
                     + np.abs(np.diff(lum, axis=0)).mean()) / 2.0)

    # Share of pixels carrying real colour. Plant matter is always chromatic —
    # even a brown neck-blast panicle. A greyscale screenshot or document scans 0.
    chromatic_fraction = float((sat > 0.15).mean())

    return {
        "chromatic_fraction": round(chromatic_fraction, 4),
        "green_fraction": round(green_fraction, 4),
        "mean_exg": round(float(exg.mean()), 4),
        "mean_vari": round(mean_vari, 4),
        "foliage_fraction": round(foliage_fraction, 4),
        "brightness": round(float(value.mean()), 4),
        "hue_dispersion": round(hue_dispersion, 4),
        "texture": round(texture, 5),
    }


def plant_gate(veg: dict) -> dict:
    """Decide whether an image plausibly shows a paddy plant.

    Returns {is_plant, plant_score, reasons}. `reasons` lists every cue that
    failed, so the UI can tell the farmer what to fix rather than just "rejected".
    """
    reasons = []
    ff = veg["foliage_fraction"]
    bright = veg["brightness"]
    disp = veg["hue_dispersion"]
    tex = veg["texture"]

    # ── Hard rejects: the image is not assessable, whatever its colour ──
    if bright < MIN_BRIGHTNESS:
        reasons.append(
            f"too dark to assess (brightness {bright:.2f} < {MIN_BRIGHTNESS:.2f}) — "
            "shoot the leaf in daylight")
    elif bright > MAX_BRIGHTNESS:
        reasons.append(f"over-exposed (brightness {bright:.2f}) — avoid direct glare")
    if disp > MAX_HUE_DISPERSION:
        reasons.append(
            f"artificial multi-coloured lighting (hue spread {disp:.2f} > "
            f"{MAX_HUE_DISPERSION:.2f}) — plants are one colour family")
    if tex < MIN_TEXTURE:
        reasons.append(
            f"flat surface with no leaf texture (detail {tex:.4f} < {MIN_TEXTURE:.4f})")
    if veg["chromatic_fraction"] < MIN_CHROMA:
        reasons.append(
            f"no colour information (chromatic pixels {veg['chromatic_fraction']*100:.1f}% "
            f"< {MIN_CHROMA*100:.1f}%) — looks like a screenshot or document, not a plant")
    # Low greenness is only fatal when the frame also lacks plant-like structure.
    # Neck-blast panicles are legitimately brown, so they survive this check via
    # their texture; a brown wall or a skin close-up does not.
    if ff < MIN_FOLIAGE and tex < 0.006:
        reasons.append(
            f"no plant material found (foliage {ff*100:.1f}% < {MIN_FOLIAGE*100:.1f}%)")

    # ── Composite score: graded confidence that this is a plant ──
    # Each cue maps to 0-1 and is averaged; used for the UI read-out and as a
    # final soft threshold so marginal frames are declined rather than guessed.
    s_fol = min(1.0, ff / 0.08)                                   # 8% foliage = full marks
    s_bright = 1.0 if 0.55 <= bright <= 0.95 else max(0.0, 1.0 - abs(bright - 0.75) / 0.40)
    s_disp = max(0.0, 1.0 - disp / MAX_HUE_DISPERSION)
    s_tex = min(1.0, tex / 0.006)
    plant_score = float(np.clip(
        0.35 * s_fol + 0.20 * s_bright + 0.25 * s_disp + 0.20 * s_tex, 0.0, 1.0))

    if not reasons and plant_score < MIN_PLANT_SCORE:
        reasons.append(
            f"does not look like a paddy plant (plant score {plant_score:.2f} < "
            f"{MIN_PLANT_SCORE:.2f})")

    return {
        "is_plant": not reasons,
        "plant_score": round(plant_score, 3),
        "reasons": reasons,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — prediction confidence gate
# ─────────────────────────────────────────────────────────────────────────────
# The model is a 4-class closed-set softmax (Brown Spot / Healthy / Leaf Blast /
# Neck Blast). It has no "unknown" class, so probabilities always sum to 1 and an
# unfamiliar image is forced into one of the four buckets. Neck Blast is the worst
# offender: it scores top1=1.000 on *every* val image (precision=recall=1.00), which
# means it is trivially separable and therefore acts as the sink for anything
# out-of-distribution — which is exactly where the keyboard photo landed.
#
# So a single top-1 number is not evidence. We require three things to agree:
#   top1     absolute probability of the winning class
#   margin   top1 - top2. A diffuse 0.62/0.20/0.14 spread is a guess; a real call
#            is 0.97/0.02/0.01.
#   entropy  Shannon entropy normalised by log(n_classes). High = the model is
#            spreading its bets.
#
# Thresholds from 400 held-out val leaves:
#   top1    p1=0.473  p5=0.582  p10=0.716  med=0.982
#   margin  p1=0.049  p5=0.253  p10=0.459  med=0.967
#   entropy p95=0.551 p99=0.714
# CONFIRMED keeps the bulk of genuine predictions; PROVISIONAL is shown but
# flagged and severity-capped; below that we abstain rather than name a disease.
# The rejected tail is ~5% of real leaves — images a farmer should simply retake,
# which is a better outcome than a confident wrong fungicide dose.
CONF_CONFIRMED = 0.80
MARGIN_CONFIRMED = 0.45
ENTROPY_CONFIRMED = 0.35

CONF_MIN = 0.55      # below → abstain
MARGIN_MIN = 0.20    # below → abstain (the keyboard scored 0.416 but failed entropy)
ENTROPY_MAX = 0.70   # above → abstain (keyboard = 0.729)


def _distribution_stats(probs: np.ndarray) -> dict:
    """top1 / margin / normalised entropy for a probability vector."""
    p = np.asarray(probs, dtype=np.float64)
    p = p / (p.sum() + 1e-12)
    order = np.sort(p)[::-1]
    top1 = float(order[0])
    margin = float(order[0] - order[1]) if p.size > 1 else 1.0
    entropy = float(-(p * np.log(p + 1e-12)).sum() / np.log(max(p.size, 2)))
    return {"top1": top1, "margin": margin, "entropy": entropy}


def grade_confidence(stats: dict) -> dict:
    """Map distribution stats to CONFIRMED / PROVISIONAL / ABSTAIN with a reason."""
    t, m, e = stats["top1"], stats["margin"], stats["entropy"]
    if t >= CONF_CONFIRMED and m >= MARGIN_CONFIRMED and e <= ENTROPY_CONFIRMED:
        return {"level": "CONFIRMED", "accepted": True,
                "note": "Strong, well-separated prediction."}
    if t >= CONF_MIN and m >= MARGIN_MIN and e <= ENTROPY_MAX:
        return {"level": "PROVISIONAL", "accepted": True,
                "note": (f"Moderate confidence ({t*100:.0f}%, margin {m*100:.0f}pp) — "
                         "treat as indicative and confirm with a second photo or your KVK.")}
    failed = []
    if t < CONF_MIN:
        failed.append(f"top confidence {t*100:.0f}% < {CONF_MIN*100:.0f}%")
    if m < MARGIN_MIN:
        failed.append(f"margin over the runner-up {m*100:.0f}pp < {MARGIN_MIN*100:.0f}pp")
    if e > ENTROPY_MAX:
        failed.append(f"prediction spread across classes (entropy {e:.2f} > {ENTROPY_MAX:.2f})")
    return {"level": "ABSTAIN", "accepted": False,
            "note": "Not confident enough to name a disease: " + "; ".join(failed) + "."}


# Severity is a property of the disease, not of this photo. A 60% guess must not
# render as a red HIGH badge, so a provisional call is capped one notch down.
_SEVERITY_ORDER = ["NONE", "LOW", "MEDIUM", "HIGH"]


def _cap_severity(severity: str, level: str) -> str:
    if level != "PROVISIONAL" or severity not in _SEVERITY_ORDER:
        return severity
    i = _SEVERITY_ORDER.index(severity)
    return _SEVERITY_ORDER[max(0, i - 1)]


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

        # ── Layer 1: botanical gate (before the classifier ever runs) ──
        veg = vegetation_index(image)
        gate = plant_gate(veg)
        gf = veg["foliage_fraction"]
        if not gate["is_plant"]:
            return {
                "is_leaf": False,
                "vegetation_index": gf,
                "vegetation": veg,
                "plant_score": gate["plant_score"],
                "reject_reasons": gate["reasons"],
                "confidence_level": "REJECTED",
                "predicted_disease": None,
                "confidence": None,
                "severity": None,
                "predictions": [],
                "message": ("This doesn't look like a paddy plant, so it was not classified: "
                            + "; ".join(gate["reasons"])
                            + ". Fill the frame with a paddy leaf in daylight and retry."),
                "model": {"path": os.path.basename(self.model_path),
                          "is_finetuned": self.is_finetuned,
                          "note": "Input rejected by the plant gate — classifier not run."},
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

        # ── Layer 2: confidence gate on the softmax distribution ──
        stats = _distribution_stats(conf_all) if conf_all else {
            "top1": float(probs.top1conf), "margin": 1.0, "entropy": 0.0}
        grade = grade_confidence(stats)

        top = predictions[0]
        low_veg = gf < LEAF_CAUTION
        common = {
            "is_leaf": True,
            "vegetation_index": gf,
            "vegetation": veg,
            "plant_score": gate["plant_score"],
            "low_vegetation": low_veg,
            "vegetation_note": (
                f"Foliage {gf*100:.1f}% — low; verify this is a clear leaf shot."
                if low_veg else
                f"Plant confirmed (foliage {gf*100:.1f}%, plant score {gate['plant_score']:.2f})."),
            "confidence_level": grade["level"],
            "confidence_note": grade["note"],
            "certainty": {
                "top1": round(stats["top1"], 4),
                "margin": round(stats["margin"], 4),
                "entropy": round(stats["entropy"], 4),
                "thresholds": {
                    "confirmed": {"top1": CONF_CONFIRMED, "margin": MARGIN_CONFIRMED,
                                  "entropy": ENTROPY_CONFIRMED},
                    "minimum": {"top1": CONF_MIN, "margin": MARGIN_MIN,
                                "entropy": ENTROPY_MAX},
                },
            },
            "predictions": predictions,
        }

        # Not confident enough to name a disease — return the ranking, no verdict.
        if not grade["accepted"]:
            return {
                **common,
                "predicted_disease": None,
                "confidence": round(stats["top1"], 4),
                "severity": None,
                "treatment": None,
                "is_paddy_condition": None,
                "message": ("Inconclusive — " + grade["note"] + " Retake the photo with a "
                            "single leaf filling the frame in even daylight, or consult your KVK."),
                "model": {
                    "path": os.path.basename(self.model_path),
                    "is_finetuned": self.is_finetuned,
                    "note": "Prediction withheld by the confidence gate.",
                },
                "latency_ms": latency_ms,
            }

        return {
            **common,
            "predicted_disease": top["disease"],
            "confidence": top["confidence"],
            "severity": _cap_severity(top["severity"], grade["level"]),
            "reported_severity": top["severity"],
            "treatment": top["treatment"],
            "is_paddy_condition": top["is_paddy_condition"],
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
