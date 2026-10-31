"""Checkpoint-backed tests. Skipped until train.py has produced a model."""
import io
import os
import sys
import glob

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CLASS_NAMES
from inference import CHECKPOINT, LeafDiseaseClassifier

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE, "../../../data/rice+leaf+diseases")

pytestmark = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT),
    reason="no trained checkpoint; run train.py first",
)


@pytest.fixture(scope="module")
def clf():
    return LeafDiseaseClassifier()


def _bytes_for(cls, index=0):
    path = sorted(glob.glob(os.path.join(DATA_ROOT, cls, "*")))[index]
    with open(path, "rb") as f:
        return f.read()


def test_prediction_shape_and_normalisation(clf):
    out = clf.predict(_bytes_for(CLASS_NAMES[0]))
    assert out["disease"] in CLASS_NAMES
    assert 0.0 <= out["confidence"] <= 1.0
    assert sum(out["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert out["crops_averaged"] >= 1


def test_beats_chance_on_a_sample_from_each_class(clf):
    """Not an accuracy measurement -- that is what cross-validation is for.

    This only catches a checkpoint that loaded with scrambled or mismatched
    weights, which would score at or below chance.
    """
    correct = 0
    total = 0
    for cls in CLASS_NAMES:
        for i in range(5):
            if clf.predict(_bytes_for(cls, i))["disease"] == cls:
                correct += 1
            total += 1
    assert correct / total > 0.34, f"only {correct}/{total} correct -- chance is 1/3"


def test_handles_arbitrary_aspect_ratios(clf):
    """Farmer photos will not be 3.4:1 leaf strips like the training set."""
    for size in [(224, 224), (1000, 200), (300, 900)]:
        buf = io.BytesIO()
        Image.new("RGB", size, (40, 110, 50)).save(buf, format="JPEG")
        assert clf.predict(buf.getvalue())["disease"] in CLASS_NAMES


def test_rejects_non_image_bytes(clf):
    with pytest.raises(Exception):
        clf.predict(b"this is not an image")
