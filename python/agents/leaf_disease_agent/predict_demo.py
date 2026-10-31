"""Run the full diagnosis path on a local image, without gRPC.

Useful for two things: checking a trained checkpoint actually works before
deploying it, and demonstrating the fusion logic without standing up the Go
pest risk agent.

Usage:
    python predict_demo.py path/to/leaf.jpg
    python predict_demo.py path/to/leaf.jpg --rule-risk Medium --rule-pests "Stem Borer"
    python predict_demo.py --sample            # one image from each class
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CLASS_NAMES
from fusion import fuse
from inference import LeafDiseaseClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE, "../../../data/rice+leaf+diseases")


def diagnose(clf, path, rule_risk, rule_pests, truth=None):
    with open(path, "rb") as f:
        cnn = clf.predict(f.read())

    result = fuse(cnn, rule_risk=rule_risk, rule_pests=rule_pests,
                  rule_advisory="Monitor fields every 3 days.")

    print(f"\n--- {os.path.basename(path)} ---")
    if truth:
        mark = "OK " if cnn["disease"] == truth and not cnn["inconclusive"] else "MISS"
        print(f"  truth        : {truth}  [{mark}]")
    print(f"  detected     : {cnn['disease']} ({cnn['confidence']:.1%})"
          f"{'  INCONCLUSIVE' if cnn['inconclusive'] else ''}")
    print(f"  probabilities: " + ", ".join(
        f"{k}={v:.1%}" for k, v in cnn["probabilities"].items()))
    print(f"  overall risk : {result['overall_risk']}  (basis={result['basis']}, "
          f"corroborated={result['corroborated']}, conf={result['confidence']:.2f})")
    print(f"  advisory     : {result['advisory']}")
    return cnn, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="path to a leaf image")
    ap.add_argument("--sample", action="store_true",
                    help="diagnose one image from each class folder")
    ap.add_argument("--rule-risk", default=None, choices=["Low", "Medium", "High"],
                    help="simulate the Go pest agent's weather-based risk")
    ap.add_argument("--rule-pests", default="", help="comma-separated pest names")
    args = ap.parse_args()

    if not args.image and not args.sample:
        ap.error("give an image path or --sample")

    rule_pests = [p.strip() for p in args.rule_pests.split(",") if p.strip()]
    clf = LeafDiseaseClassifier()
    print(f"checkpoint cv_accuracy={clf.cv_accuracy}  classes={clf.class_names}")

    if args.sample:
        for cls in CLASS_NAMES:
            found = sorted(glob.glob(os.path.join(DATA_ROOT, cls, "*")))
            if found:
                diagnose(clf, found[0], args.rule_risk, rule_pests, truth=cls)
    else:
        diagnose(clf, args.image, args.rule_risk, rule_pests)


if __name__ == "__main__":
    main()
