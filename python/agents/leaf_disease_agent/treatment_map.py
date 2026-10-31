"""Advisory content for the three diseases the image classifier can detect.

Note on provenance: the Go pest rulebook (go/agents/pest_risk/pest_rules.go)
covers insect pests and blast/sheath blight, and models *none* of these three
leaf diseases, so this map is additive rather than a duplicate of it.

Economic Threshold Levels are only quoted where a threshold is meaningful.
Bacterial leaf blight and leaf smut have no actionable ETL in NIPHM's Rice IPM
package -- the former is managed culturally and prophylactically because
curative bactericides are unreliable, the latter is not economically
significant. Those entries say so instead of carrying an invented number.
"""

SEVERITY_ORDER = {"Low": 1, "Medium": 2, "High": 3}

TREATMENTS = {
    "Bacterial leaf blight": {
        "pathogen": "Xanthomonas oryzae pv. oryzae (bacterial)",
        "severity": "High",
        "signs": (
            "Water-soaked lesions starting at leaf tips and margins, enlarging into "
            "yellow to straw-coloured stripes along the margin. Bacterial ooze droplets "
            "visible in early morning; seedlings may wilt entirely (kresek)."
        ),
        "action": (
            "Stop all nitrogen top-dressing immediately and drain the field -- standing "
            "water plus high nitrogen is what drives spread. Spray Copper oxychloride "
            "3g/L. Avoid moving through the crop when foliage is wet. Sow a resistant "
            "variety next season; bactericides alone will not rescue a severe infection."
        ),
        "etl": "No actionable ETL -- bacterial disease, managed culturally and prophylactically",
        "yield_impact": "20-50% loss in susceptible varieties under severe infection",
    },
    "Brown spot": {
        "pathogen": "Bipolaris oryzae (fungal)",
        "severity": "Medium",
        "signs": (
            "Oval, sesame-seed shaped brown lesions with a grey or whitish centre and "
            "dark brown margin, scattered across the whole leaf blade rather than "
            "concentrated at the margins."
        ),
        "action": (
            "Spray Mancozeb 2.5g/L or Propiconazole 1ml/L. Treat the cause as well as "
            "the symptom: brown spot tracks potassium and silicon deficiency, so request "
            "a soil test and correct K rather than repeat-spraying a nutrient problem."
        ),
        "etl": "Indicative only: ~5-10% leaf area affected (not a NIPHM-verified figure)",
        "yield_impact": "Typically 5-15% loss; higher on nutrient-deficient soils",
    },
    "Leaf smut": {
        "pathogen": "Entyloma oryzae (fungal)",
        "severity": "Low",
        "signs": (
            "Small, angular to linear, slightly raised black spots on both leaf surfaces, "
            "sometimes coalescing. Leaf tips may dry out as the crop matures."
        ),
        "action": (
            "Usually no treatment is warranted -- leaf smut rarely causes economic loss. "
            "Avoid excess nitrogen. If infection is heavy late in the season, "
            "Propiconazole 1ml/L is sufficient."
        ),
        "etl": "Not economically significant -- no ETL defined",
        "yield_impact": "Negligible in most seasons",
    },
}


def lookup(disease):
    """Return the advisory entry for a detected disease class."""
    try:
        return TREATMENTS[disease]
    except KeyError:
        raise ValueError(f"Unknown disease {disease!r}; known: {sorted(TREATMENTS)}") from None
