"""Fast unit tests for shared utilities. Run with: pytest"""
import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
from PIL import Image  # noqa: E402

from core.utils import EarlyStopping, get_transforms  # noqa: E402
from core.models import FoodClassifier  # noqa: E402


def test_transforms_output_shape():
    train_tf, val_tf = get_transforms(img_size=224)
    img = Image.new("RGB", (500, 300), color=(120, 80, 40))
    assert val_tf(img).shape == (3, 224, 224)
    assert train_tf(img).shape == (3, 224, 224)


def test_early_stopping_triggers_after_patience(tmp_path):
    model = torch.nn.Linear(2, 2)
    stopper = EarlyStopping(patience=2, path=str(tmp_path / "ckpt" / "best.pth"))
    stopper(1.0, model)          # first loss -> checkpoint saved
    assert os.path.exists(stopper.path)
    stopper(0.5, model)          # improvement resets counter
    assert stopper.counter == 0
    stopper(0.6, model)          # worse
    assert not stopper.early_stop
    stopper(0.7, model)          # worse again -> patience reached
    assert stopper.early_stop


def test_unknown_architecture_rejected():
    with pytest.raises(ValueError):
        FoodClassifier(architecture="resnet50")


def test_vegan_taxonomy_is_valid_json():
    with open(os.path.join(ROOT, "core", "vegan_taxonomy.json"), encoding="utf-8") as f:
        taxonomy = json.load(f)
    assert isinstance(taxonomy, dict) and len(taxonomy) > 0
