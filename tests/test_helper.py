import random
import torch
import numpy as np
from PIL import Image

import pytest

from src import helper
from src.helper import (
    str2bool,
    set_seed,
    _build_model_factory,
    _make_paired_transform,
    get_cv_pids,
    load_model,
    get_train_test_pids,
)


def test_pids_split():
    train_pids, test_pids = get_train_test_pids()
    assert len(train_pids) != 0
    assert len(test_pids) != 0


def test_cv_split():
    train_pids, _ = get_train_test_pids()
    cv = get_cv_pids(train_pids)
    assert len(cv) == 5
    for fold in cv:
        assert len(fold) == 2
        assert len(fold[0]) != 0
        assert len(fold[1]) != 0


def test_str2bool_true_false():
    assert str2bool(True) is True
    assert str2bool("true") is True
    assert str2bool("1") is True
    assert str2bool("false") is False
    assert str2bool("0") is False
    with pytest.raises(Exception):
        str2bool("notabool")


def test_set_seed_reproducible():
    set_seed(123)
    a = random.random()
    set_seed(123)
    b = random.random()
    assert a == b
    # numpy
    set_seed(999)
    x = np.random.rand(3)
    set_seed(999)
    y = np.random.rand(3)
    assert np.allclose(x, y)


def test_build_model_factory_returns_factory():
    f = _build_model_factory("unet")
    assert callable(f)
    m = f(num_classes=3, in_channels=1)
    assert hasattr(m, "parameters")
    f2 = _build_model_factory("unetpp")
    assert callable(f2)


def test_make_paired_transform_basic():
    tf = _make_paired_transform(size=64, aug=False)
    img = Image.new("L", (200, 200), color=128)
    mask = Image.new("L", (200, 200), color=0)
    it, mt = tf(img, mask)
    assert isinstance(it, torch.Tensor)
    assert isinstance(mt, torch.Tensor)
    assert it.shape[1] == 64 and it.shape[2] == 64


def test_get_train_test_and_cv(tmp_path):
    # create fake images to infer pids
    d = tmp_path / "images"
    d.mkdir()
    names = [
        "1_0.jpg",
        "2_0.jpg",
        "3_0.jpg",
        "4_0.jpg",
        "5_0.jpg",
        "6_0.jpg",
        "7_0.jpg",
    ]
    for n in names:
        (d / n).write_text("x")

    train, test = get_train_test_pids(image_path=str(d), seed=42)
    assert isinstance(train, list) and isinstance(test, list)
    assert len(train) + len(test) == len(set([int(n.split("_")[0]) for n in names]))

    cv = get_cv_pids((train, test), cv=3, seed=42)
    assert len(cv) == 3
    for tr, va in cv:
        assert isinstance(tr, list) and isinstance(va, list)


def test_load_model_monkeypatch(tmp_path, monkeypatch):
    # Create a fake checkpoint with state_dict
    ckpt = {"state_dict": {"module.conv.weight": torch.randn(1, 1, 1, 1)}}
    p = tmp_path / "m.pth"
    torch.save(ckpt, str(p))

    # Monkeypatch UNet to accept the signature used in helper.load_model
    class Dummy:
        def __init__(self, in_channels=1, out_channels=5):
            self._m = torch.nn.Conv2d(in_channels, out_channels, 1)

        def load_state_dict(self, st, strict=True):
            return

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(helper, "UNet", Dummy)
    m = load_model(str(p), in_channels=1, num_classes=5, device="cpu")
    assert m is not None
