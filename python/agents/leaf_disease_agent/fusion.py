"""Fuses an image diagnosis with the rule-based pest risk forecast.

The two signals answer different questions and are deliberately kept separate
until the end:

    the Go rulebook  -- "do temperature, humidity and growth stage *favour*
                        an outbreak?"  (predictive, no observation of the field)
    the image CNN    -- "is a disease *present* on this leaf right now?"
                        (observational, no knowledge of conditions)

So a confident detection raises risk regardless of what the rules predicted,
and agreement between the two is treated as independent corroboration.
"""
from treatment_map import SEVERITY_ORDER, lookup

RISK_LEVELS = ("Low", "Medium", "High")


def _max_risk(a, b):
    if a not in SEVERITY_ORDER:
        return b
    if b not in SEVERITY_ORDER:
        return a
    return a if SEVERITY_ORDER[a] >= SEVERITY_ORDER[b] else b


def fuse(cnn_result, rule_risk=None, rule_pests=None, rule_advisory=""):
    """Combine an image prediction with optional rule-based risk.

    rule_risk is "Low"/"Medium"/"High" from the pest risk agent, or None when
    that agent was unreachable or no weather context was supplied.
    """
    rule_pests = rule_pests or []
    rules_available = rule_risk in SEVERITY_ORDER

    if cnn_result["inconclusive"]:
        risk = rule_risk if rules_available else "Unknown"
        parts = [
            f"Leaf photo was inconclusive (best guess {cnn_result['disease']} at "
            f"{cnn_result['confidence']:.0%} confidence, below the "
            f"reporting threshold). No diagnosis is being made from it."
        ]
        if rules_available:
            parts.append(
                f"Weather-and-stage rules alone put pest risk at {rule_risk}. {rule_advisory}".strip()
            )
        else:
            parts.append("No weather context available either, so no risk level can be given.")
        parts.append(
            "Retake the photo filling the frame with a single leaf in even daylight."
        )
        return {
            "overall_risk": risk,
            "basis": "rules_only" if rules_available else "none",
            "detected_disease": None,
            "detection_confidence": cnn_result["confidence"],
            "corroborated": False,
            "confidence": 0.60 if rules_available else 0.30,
            "advisory": " ".join(parts),
            "treatment": None,
            "rule_pests": rule_pests,
        }

    disease = cnn_result["disease"]
    entry = lookup(disease)
    observed_risk = entry["severity"]

    # Agreement between an independent forecast and an independent observation
    # is worth more than either alone -- but only when both are actually
    # elevated. The rulebook covers insect pests, so "Medium pest pressure"
    # plus a cosmetic leaf disease is two unrelated minor findings, not
    # corroboration; escalating that to High would tell a farmer to act on a
    # disease whose own advisory says no treatment is warranted.
    rules_elevated = rules_available and SEVERITY_ORDER.get(rule_risk, 0) >= 2
    corroborated = rules_elevated and SEVERITY_ORDER[observed_risk] >= 2
    risk = _max_risk(observed_risk, rule_risk if rules_available else None)
    if corroborated:
        risk = "High"

    model_acc = cnn_result.get("model_cv_accuracy") or 0.80
    confidence = 0.5 * cnn_result["confidence"] + 0.5 * model_acc
    if corroborated:
        confidence = min(0.95, confidence + 0.05)

    parts = [
        f"{disease} detected on the submitted leaf at {cnn_result['confidence']:.0%} "
        f"confidence ({entry['pathogen']}). Expected impact: {entry['yield_impact']}.",
        entry["action"],
    ]
    if rules_available:
        if corroborated:
            pest_note = f" Rules had flagged {', '.join(rule_pests)}." if rule_pests else ""
            parts.append(
                f"Weather-and-stage rules independently rated pest pressure {rule_risk}, "
                f"which corroborates the image.{pest_note} Treat this as High risk and "
                f"scout the whole field, not just the sampled plant."
            )
        elif rules_elevated:
            pest_note = f" ({', '.join(rule_pests)})" if rule_pests else ""
            parts.append(
                f"Separately, weather-and-stage rules rate insect pest pressure "
                f"{rule_risk}{pest_note}. That is a different concern from the minor "
                f"leaf disease found here, and it is what drives the {risk} overall "
                f"rating -- not the leaf diagnosis."
            )
        else:
            parts.append(
                f"Weather-and-stage rules rated background pest pressure {rule_risk}, "
                f"so conditions are not currently driving spread -- but the disease is "
                f"already present, so act on the observation."
            )
    else:
        parts.append(
            "No weather context was supplied, so this is based on the leaf image alone; "
            "add district weather to also get outbreak-risk forecasting."
        )

    return {
        "overall_risk": risk,
        "basis": "image+rules" if rules_available else "image_only",
        "detected_disease": disease,
        "detection_confidence": cnn_result["confidence"],
        "corroborated": corroborated,
        "confidence": round(confidence, 4),
        "advisory": " ".join(parts),
        "treatment": {
            "pathogen": entry["pathogen"],
            "severity": observed_risk,
            "signs": entry["signs"],
            "action": entry["action"],
            "etl": entry["etl"],
        },
        "rule_pests": rule_pests,
    }
