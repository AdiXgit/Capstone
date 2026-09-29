#!/usr/bin/env python3
"""Re-derive the detection gate thresholds from the val set.

The constants in disease_detection.py (MIN_BRIGHTNESS, MAX_HUE_DISPERSION,
MIN_TEXTURE, MIN_CHROMA, and the confidence tiers) are not guesses — each sits
just outside the 1st percentile of real held-out paddy photos, so genuine leaves
survive while out-of-distribution frames are refused. Re-run this after
retraining or after swapping the dataset, and update the constants if the
percentiles have moved.

    python scripts/calibrate_gate.py                 # percentiles + a pass/fail audit
    python scripts/calibrate_gate.py --negatives DIR # add your own negative images

Anything in --negatives should be a non-paddy photo; the script reports how many
of them still receive a disease label (the number you want at zero).
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "python", "agents", "crop_health_agent"))

CLASSES = ["Brown_Spot", "Healthy", "Leaf_Blast", "Neck_Blast"]
CUES = ["foliage_fraction", "brightness", "hue_dispersion", "texture", "chromatic_fraction"]


def _val_files(split_dir: str) -> dict[str, list[str]]:
    return {c: sorted(glob.glob(os.path.join(split_dir, c, "*"))) for c in CLASSES}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", default=os.path.join(REPO, "data", "paddy_disease", "val"))
    ap.add_argument("--negatives", default=None, help="directory of non-paddy images")
    ap.add_argument("--limit", type=int, default=0, help="cap images per class (0 = all)")
    args = ap.parse_args()

    from PIL import Image
    from disease_detection import (
        vegetation_index, plant_gate, PaddyDiseaseDetector,
        MIN_BRIGHTNESS, MAX_HUE_DISPERSION, MIN_TEXTURE, MIN_CHROMA,
    )

    if not os.path.isdir(args.val):
        print(f"val split not found: {args.val}\n"
              f"Fetch it with scripts/download_paddy_dataset.py first.")
        return 1

    files = _val_files(args.val)
    if args.limit:
        files = {c: fs[: args.limit] for c, fs in files.items()}

    # ── Cue percentiles on genuine leaves ────────────────────────
    print("Cue distribution over real val leaves (thresholds must sit below p1)\n")
    rows = {c: [] for c in CUES}
    per_class: dict[str, list[dict]] = {}
    for c, fs in files.items():
        vals = [vegetation_index(Image.open(f).convert("RGB")) for f in fs]
        per_class[c] = vals
        for cue in CUES:
            rows[cue] += [v[cue] for v in vals]

    print(f"{'cue':20s} {'p1':>9s} {'p5':>9s} {'median':>9s} {'p95':>9s}   threshold")
    limits = {
        "foliage_fraction": "(soft, paired with texture)",
        "brightness": f">= {MIN_BRIGHTNESS}",
        "hue_dispersion": f"<= {MAX_HUE_DISPERSION}",
        "texture": f">= {MIN_TEXTURE}",
        "chromatic_fraction": f">= {MIN_CHROMA}",
    }
    for cue in CUES:
        a = np.array(rows[cue], dtype=float)
        q = np.percentile(a, [1, 5, 50, 95])
        print(f"{cue:20s} {q[0]:9.4f} {q[1]:9.4f} {q[2]:9.4f} {q[3]:9.4f}   {limits[cue]}")

    # ── False-reject audit: how many real leaves does the gate refuse? ──
    print("\nBotanical gate on real leaves (these are false rejects)\n")
    total = rejected = 0
    for c, vals in per_class.items():
        rej = sum(0 if plant_gate(v)["is_plant"] else 1 for v in vals)
        total += len(vals)
        rejected += rej
        print(f"  {c:12s} {rej:3d}/{len(vals):3d} rejected ({rej/max(1,len(vals))*100:.1f}%)")
    print(f"  {'TOTAL':12s} {rejected:3d}/{total:3d} rejected ({rejected/max(1,total)*100:.1f}%)")

    # ── End-to-end tiers + accuracy on what actually gets served ──
    det = PaddyDiseaseDetector()
    if not det.is_finetuned:
        print("\nNo fine-tuned model found — skipping the confidence audit.")
        return 0

    from disease_treatment import lookup
    expected = {c: lookup(c)["label"] for c in CLASSES}

    print("\nEnd-to-end tiers (confidence gate applied)\n")
    tiers: dict[str, int] = {}
    served = correct = ungated_correct = n = 0
    for c, fs in files.items():
        for f in fs:
            r = det.predict(open(f, "rb").read())
            lvl = "REJECTED" if r["is_leaf"] is False else r["confidence_level"]
            tiers[lvl] = tiers.get(lvl, 0) + 1
            n += 1
            if r["predictions"] and r["predictions"][0]["disease"] == expected[c]:
                ungated_correct += 1
            if r.get("predicted_disease"):
                served += 1
                correct += r["predicted_disease"] == expected[c]
    for lvl in ("CONFIRMED", "PROVISIONAL", "ABSTAIN", "REJECTED"):
        print(f"  {lvl:12s} {tiers.get(lvl, 0):4d}")
    print(f"\n  answered            {served}/{n} ({served/max(1,n)*100:.1f}%)")
    print(f"  ungated top-1 acc   {ungated_correct/max(1,n):.4f}")
    print(f"  accuracy on served  {correct/max(1,served):.4f}")

    # ── Negatives: the number that matters is zero ───────────────
    if args.negatives:
        negs = [f for f in sorted(glob.glob(os.path.join(args.negatives, "*")))
                if not os.path.isdir(f)]
        print(f"\nNegatives from {args.negatives} ({len(negs)} images)\n")
        leaked = 0
        for f in negs:
            r = det.predict(open(f, "rb").read())
            dis = r.get("predicted_disease")
            leaked += dis is not None
            why = "; ".join(r.get("reject_reasons") or [r.get("confidence_note", "")])
            print(f"  {os.path.basename(f):24s} "
                  f"{'LEAKED -> ' + dis if dis else 'no label':24s} {why[:60]}")
        print(f"\n  leaking a disease label: {leaked}/{len(negs)}  (target: 0)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
