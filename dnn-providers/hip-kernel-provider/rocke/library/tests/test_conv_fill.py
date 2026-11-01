# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""conv_fill: the forward benchmarks' --fill data and --kernels-file lists."""

import pytest

from benchmarks.common import conv_fill


@pytest.fixture(autouse=True)
def _uniform_after():
    yield
    conv_fill.set_mode("uniform")


def test_modes_and_floor():
    assert conv_fill.mode() == "uniform" and conv_fill.ref_floor() == 1.0
    conv_fill.set_mode("miopen")
    assert conv_fill.ref_floor() < 1e-20
    with pytest.raises(ValueError):
        conv_fill.set_mode("normal")


def test_fill_ranges():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    x = conv_fill.fill_x(torch.empty(4096), generator=g)
    assert x.min() >= -1.0 and x.max() <= 1.0 and x.min() < 0
    conv_fill.set_mode("miopen")
    x = conv_fill.fill_x(torch.empty(4096), generator=g)
    assert x.min() >= 0.0 and x.max() <= 0.01
    w = conv_fill.fill_w(torch.empty(4096), "fp16", generator=g)
    assert w.min() >= -0.01 / 3 and w.max() <= 0.005
    w = conv_fill.fill_w(torch.empty(4096), "bf16", generator=g)
    assert w.min() >= -0.005 and w.max() <= 0.005


def test_kernels_file(tmp_path):
    assert conv_fill.load_kernels_file(None) is None
    f = tmp_path / "k.txt"
    f.write_text("fwd-direct_fwd_wmma-a\n\ncase1\tfwd-direct_fwd_wmma-b\n")
    m = conv_fill.load_kernels_file(str(f))
    # A label is kept with and without its direction prefix.
    assert conv_fill.kernels_for(m, "case2") == {
        "fwd-direct_fwd_wmma-a",
        "direct_fwd_wmma-a",
    }
    assert "direct_fwd_wmma-b" in conv_fill.kernels_for(m, "case1")
    assert "direct_fwd_wmma-a" in conv_fill.kernels_for(m, "case1")
