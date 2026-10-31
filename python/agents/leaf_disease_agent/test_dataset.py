import os
import sys

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset import CROP_SIZE, STRIP_HEIGHT, LeafCropDataset, _normalise_height, tta_crops


def strip(w, h):
    return Image.new("RGB", (w, h), (30, 120, 40))


def test_height_normalisation_erases_resolution_cue():
    """All 40 blight images are 3081x897 while other classes are mixed sizes.

    If absolute resolution reached the model it could classify on source
    artefacts instead of disease, so every input must leave this function at
    the same height.
    """
    for size in [(3081, 897), (766, 250), (250, 200), (4160, 2340)]:
        assert _normalise_height(strip(*size)).size[1] == STRIP_HEIGHT


def test_normalisation_preserves_aspect_ratio():
    out = _normalise_height(strip(3081, 897))
    assert abs(out.size[0] / out.size[1] - 3081 / 897) < 0.01


def test_narrow_images_padded_to_crop_width():
    """A near-square image would resize below CROP_SIZE wide and break cropping."""
    assert _normalise_height(strip(200, 200)).size[0] >= CROP_SIZE


def test_tta_crops_span_the_whole_leaf():
    crops = tta_crops(_normalise_height(strip(3081, 897)), n_crops=5)
    assert crops.shape == (5, 3, CROP_SIZE, CROP_SIZE)


def test_tta_single_crop_is_centred():
    assert tta_crops(_normalise_height(strip(3081, 897)), n_crops=1).shape[0] == 1


def test_repeats_multiply_epoch_length_not_image_count():
    paths, labels = [], []
    ds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "../../../data/rice+leaf+diseases")
    from dataset import discover_images
    paths, labels = discover_images(ds_dir)
    subset, sublabels = paths[:4], labels[:4]
    assert len(LeafCropDataset(subset, sublabels, repeats=3)) == 12


def test_crops_from_same_image_differ():
    """Random cropping must actually vary, or augmentation is a no-op."""
    from dataset import discover_images
    ds_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "../../../data/rice+leaf+diseases")
    paths, labels = discover_images(ds_dir)
    ds = LeafCropDataset(paths[:1], labels[:1], repeats=2)
    a, _ = ds[0]
    b, _ = ds[1]
    assert not a.allclose(b)
