"""KVK (Krishi Vigyan Kendra) treatment recommendations.

Recommendations are keyed by (stress_level, growth_stage) so the agent can
return a single, stage-aware action for whatever the classifier predicts.
"""

STRESS_LEVELS = ["Healthy", "Mild", "Moderate", "Severe"]

# Pests/diseases most likely at each stage (Karnataka paddy, agronomic literature).
PEST_BY_STAGE = {
    "Germination":   ["Army Worm"],
    "Vegetative":    ["Stem Borer", "Leaf Folder", "Gall Midge"],
    "Flowering":     ["Rice Bug", "Brown Planthopper", "Stem Borer"],
    "Grain_Filling": ["Rice Bug", "Brown Planthopper", "Rats"],
    "Maturity":      ["Rats", "Birds"],
}
DISEASE_BY_STAGE = {
    "Germination":   ["Damping Off", "Seed Rot"],
    "Vegetative":    ["Bacterial Leaf Blight", "Brown Spot", "Blast"],
    "Flowering":     ["Blast (Neck)", "False Smut", "Sheath Blight"],
    "Grain_Filling": ["False Smut", "Grain Discolouration", "Sheath Blight"],
    "Maturity":      ["Grain Discolouration"],
}

PEST_DISEASE_TREATMENT = {
    "Blast":                 "Tricyclazole 75WP @ 0.6g/L or Tebuconazole 50WG @ 1g/L",
    "Blast (Neck)":          "Tricyclazole 75WP @ 0.6g/L — spray at boot stage immediately",
    "Sheath Blight":         "Hexaconazole 5EC @ 2ml/L or Validamycin 3L @ 2ml/L",
    "False Smut":            "Propiconazole 25EC @ 1ml/L at booting stage",
    "Bacterial Leaf Blight": "Copper oxychloride 50WP @ 3g/L — reduce nitrogen input",
    "Brown Spot":            "Mancozeb 75WP @ 2g/L — check soil fertility",
    "Stem Borer":            "Chlorpyriphos 20EC @ 2.5ml/L or Cartap 4G @ 25kg/ha",
    "Leaf Folder":           "Monocrotophos 36SL @ 1.5ml/L — spray on rolled leaves",
    "Brown Planthopper":     "Buprofezin 25SC @ 1ml/L — do NOT use pyrethroids",
    "Gall Midge":            "Carbofuran 3G @ 25kg/ha at tillering",
    "Rice Bug":              "Malathion 50EC @ 2ml/L — spray early morning",
}

STAGE_ADVISORIES = {
    "Sowing":        "Puddle and level field. Treat seeds with Carbendazim 2g/kg. Maintain 2–5cm water.",
    "Germination":   "Maintain thin water layer. Apply Butachlor 1.5L/ha pre-emergence. Scout for army worm.",
    "Vegetative":    "Apply top-dress nitrogen (25% N). Control weeds. Scout stem borer and leaf folder weekly.",
    "Flowering":     "CRITICAL: Do NOT spray pesticides during flowering. Maintain water. Watch for neck blast.",
    "Grain_Filling": "Maintain moisture. Spray for Rice Bug if sighted. Protect from birds.",
    "Maturity":      "Drain field 10–15 days before harvest. Check grain moisture < 20%.",
    "Harvest":       "Harvest when 85% grains are straw-coloured. Thresh within 24h.",
}


def _likely_treatment_line(stage):
    pests = PEST_BY_STAGE.get(stage, [])
    diseases = DISEASE_BY_STAGE.get(stage, [])
    parts = []
    for p in pests[:1]:
        if p in PEST_DISEASE_TREATMENT:
            parts.append(f"For {p}: {PEST_DISEASE_TREATMENT[p]}.")
    for d in diseases[:1]:
        if d in PEST_DISEASE_TREATMENT:
            parts.append(f"For {d}: {PEST_DISEASE_TREATMENT[d]}.")
    return " ".join(parts)


def _build_treatment_map():
    """Explicit (stress_level, growth_stage) -> KVK recommendation matrix."""
    mapping = {}
    for stage, advisory in STAGE_ADVISORIES.items():
        likely = _likely_treatment_line(stage)
        mapping[("Healthy", stage)] = (
            f"Crop performing at or above stage benchmark. Maintain current practices. {advisory}"
        )
        mapping[("Mild", stage)] = (
            f"Slightly below expected NDVI for {stage.replace('_', ' ')}. "
            f"Check irrigation and top-dress nutrients. {advisory}"
        )
        mapping[("Moderate", stage)] = (
            f"Below expected performance for {stage.replace('_', ' ')}. "
            f"Inspect for pest/nutrient stress. {likely or 'Scout field and confirm cause before treating.'}"
        )
        mapping[("Severe", stage)] = (
            f"URGENT: NDVI significantly below stage benchmark for {stage.replace('_', ' ')}. "
            f"{likely or 'Inspect for blast, Brown Planthopper, or nutrient deficiency.'} "
            "Contact local KVK before applying any treatment."
        )
    return mapping


TREATMENT_MAP = _build_treatment_map()


def get_treatment(stress_level, stage):
    return TREATMENT_MAP.get(
        (stress_level, stage),
        f"Scout field for {stage.replace('_', ' ')}-stage issues and consult local KVK.",
    )


def crop_status_from_stress(stress_level):
    return {"Healthy": "HEALTHY", "Mild": "HEALTHY", "Moderate": "STRESSED", "Severe": "DISEASED"}.get(
        stress_level, "UNKNOWN"
    )
