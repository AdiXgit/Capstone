"""Maps a YOLO image-classification label to a canonical paddy disease and its
KVK (Krishi Vigyan Kendra) treatment.

The image detector (disease_detection.py) predicts a *class name* — whatever the
fine-tuned dataset used as its folder names (e.g. "bacterial_leaf_blight",
"brown_spot", "blast", "normal"). Different public paddy datasets spell these
slightly differently, so we normalise the raw label to a token key and look up a
single canonical entry. This keeps the vision path speaking the same treatment
vocabulary as the tabular crop-health agent (treatment_map.py).
"""
import re

# Canonical paddy leaf conditions -> KVK advisory. Keys are normalised tokens
# (lowercase, non-alphanumerics collapsed to "_"). Aliases map dataset-specific
# spellings onto a canonical key.
DISEASE_INFO = {
    "bacterial_leaf_blight": {
        "label": "Bacterial Leaf Blight",
        "severity": "HIGH",
        "treatment": "Copper oxychloride 50WP @ 3g/L. Drain the field, stop nitrogen top-dressing, "
                     "and avoid clip-transplanting. Use resistant varieties next season.",
    },
    "bacterial_leaf_streak": {
        "label": "Bacterial Leaf Streak",
        "severity": "MEDIUM",
        "treatment": "Copper-based bactericide spray. Reduce nitrogen, improve field drainage, "
                     "and remove weed hosts along bunds.",
    },
    "bacterial_panicle_blight": {
        "label": "Bacterial Panicle Blight",
        "severity": "HIGH",
        "treatment": "No curative spray is reliable — manage with balanced nutrition, avoid excess "
                     "nitrogen at booting, and grow tolerant varieties. Confirm with KVK.",
    },
    "blast": {
        "label": "Rice Blast",
        "severity": "HIGH",
        "treatment": "Tricyclazole 75WP @ 0.6g/L or Tebuconazole 50WG @ 1g/L. Spray immediately at "
                     "first lesions; repeat before neck-blast at boot stage. Avoid excess nitrogen.",
    },
    "leaf_blast": {
        "label": "Leaf Blast",
        "severity": "HIGH",
        "treatment": "Tricyclazole 75WP @ 0.6g/L or Tebuconazole 50WG @ 1g/L. Spray at first "
                     "spindle-shaped lesions and avoid excess nitrogen; repeat before boot stage.",
    },
    "neck_blast": {
        "label": "Neck Blast",
        "severity": "HIGH",
        "treatment": "Tricyclazole 75WP @ 0.6g/L at boot/heading — neck blast attacks the panicle "
                     "base and causes whiteheads. Spray preventively before flowering; avoid excess nitrogen.",
    },
    "brown_spot": {
        "label": "Brown Spot",
        "severity": "MEDIUM",
        "treatment": "Mancozeb 75WP @ 2g/L. Often a nutrient-stress disease — correct potassium and "
                     "soil fertility, and treat seed with Carbendazim before sowing.",
    },
    "dead_heart": {
        "label": "Dead Heart (Stem Borer damage)",
        "severity": "HIGH",
        "treatment": "Stem borer damage. Cartap hydrochloride 4G @ 25kg/ha or Chlorpyriphos 20EC "
                     "@ 2.5ml/L. Clip and destroy affected tillers; use pheromone traps.",
    },
    "downy_mildew": {
        "label": "Downy Mildew",
        "severity": "MEDIUM",
        "treatment": "Metalaxyl-based fungicide spray. Improve drainage and avoid dense planting; "
                     "rogue out severely infected plants.",
    },
    "hispa": {
        "label": "Rice Hispa",
        "severity": "MEDIUM",
        "treatment": "Hispa beetle. Chlorpyriphos 20EC @ 2ml/L or Quinalphos 25EC. Clip and burn "
                     "scraped leaf tips; avoid over-fertilising with nitrogen.",
    },
    "leaf_scald": {
        "label": "Leaf Scald",
        "severity": "MEDIUM",
        "treatment": "Propiconazole 25EC @ 1ml/L. Balance nitrogen and remove infected residue "
                     "after harvest.",
    },
    "leaf_smut": {
        "label": "Leaf Smut",
        "severity": "LOW",
        "treatment": "Usually minor. Propiconazole or Mancozeb if widespread. Manage residue and "
                     "avoid excess nitrogen.",
    },
    "narrow_brown_spot": {
        "label": "Narrow Brown Leaf Spot",
        "severity": "LOW",
        "treatment": "Propiconazole 25EC @ 1ml/L if severe. Correct potassium deficiency and use "
                     "resistant varieties.",
    },
    "sheath_blight": {
        "label": "Sheath Blight",
        "severity": "HIGH",
        "treatment": "Hexaconazole 5EC @ 2ml/L or Validamycin 3L @ 2ml/L directed at the sheath. "
                     "Avoid dense planting and excess nitrogen; improve air movement.",
    },
    "tungro": {
        "label": "Tungro (viral)",
        "severity": "HIGH",
        "treatment": "Viral disease spread by green leafhopper — no cure. Control the vector "
                     "(Imidacloprid seed treatment), remove infected plants, and plant tolerant "
                     "varieties. Contact KVK for the area advisory.",
    },
    "normal": {
        "label": "Healthy Leaf",
        "severity": "NONE",
        "treatment": "No disease detected in the image. Maintain current practices and continue "
                     "weekly scouting.",
    },
}

# Dataset-specific spellings -> canonical key above.
ALIASES = {
    "healthy": "normal",
    "healthy_leaf": "normal",
    "bacterialblight": "bacterial_leaf_blight",
    "bacterial_blight": "bacterial_leaf_blight",
    "blb": "bacterial_leaf_blight",
    "rice_blast": "blast",
    "brownspot": "brown_spot",
    "brown_spots": "brown_spot",
    "leaf_hispa": "hispa",
    "rice_hispa": "hispa",
    "stem_borer": "dead_heart",
    "deadheart": "dead_heart",
    "sheathblight": "sheath_blight",
    "false_smut": "leaf_smut",
}


def normalize_label(raw: str) -> str:
    """Collapse a raw class label to a lookup token: lowercase, alnum + '_'."""
    token = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")
    return ALIASES.get(token, token)


def lookup(raw_label: str) -> dict:
    """Return {label, severity, treatment, matched} for a raw class label.

    `matched` is False when the label is not a paddy condition we know (e.g. the
    base ImageNet model returned "banana" because no fine-tuned model is loaded).
    """
    key = normalize_label(raw_label)
    if key in DISEASE_INFO:
        info = dict(DISEASE_INFO[key])
        info["matched"] = True
        return info
    return {
        "label": str(raw_label).replace("_", " ").title(),
        "severity": "UNKNOWN",
        "treatment": "This label is not a known paddy leaf condition. Fine-tune the model on the "
                     "paddy dataset (scripts/train_yolo_classifier.py) for real disease treatment.",
        "matched": False,
    }
