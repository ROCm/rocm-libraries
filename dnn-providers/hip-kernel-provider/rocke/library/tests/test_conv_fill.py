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


def test_ref_floor_is_forward_only():
    """--fill only sets the forward inputs: dgrad / wgrad keep the floor 1."""
    conv_fill.set_mode("miopen")
    assert conv_fill.ref_floor("fwd") < 1e-20
    assert conv_fill.ref_floor("dgrad") == conv_fill.ref_floor("wgrad") == 1.0


def test_kernels_file(tmp_path):
    assert conv_fill.load_kernels_file(None) is None
    f = tmp_path / "k.txt"
    f.write_text(
        "fwd-direct_fwd_wmma-a\n\ncase1\tdirect_fwd_wmma-b\n"
        "case1\twgrad-direct_wgrad-c\n"
    )
    m = conv_fill.load_kernels_file(str(f))
    # A prefixed label is that direction's only.
    assert conv_fill.kernels_for(m, "case2") == {"fwd-direct_fwd_wmma-a"}
    # A bare label is the kernel in every direction.
    assert conv_fill.kernels_for(m, "case1") == {
        "fwd-direct_fwd_wmma-a",
        "fwd-direct_fwd_wmma-b",
        "dgrad-direct_fwd_wmma-b",
        "wgrad-direct_fwd_wmma-b",
        "wgrad-direct_wgrad-c",
    }


@pytest.mark.parametrize("direct", [True, False])
def test_kernels_file_bare_label_matches_identity_label(tmp_path, direct):
    """The cache runners keep a plan iff ``identity.label()`` is in
    ``kernels_for``: a label written with or without its direction prefix
    selects it, a prefixed one of another direction does not."""
    from kernels.conv import aot_jobs as ks
    from kernels.conv import direct_aot_jobs as dks
    from rocke.core.arch import ArchTarget

    if direct:
        ident = dks._identity(
            "gfx950",
            64,
            "direct_grouped",
            "direct_fwd",
            dks.DirectCaps(KH=3, PAD=1, stride=1, cpg=16, kpg=16),
            dict(block_q=16, block_groups=2, double_buffer=True),
            "fp16",
        )
    else:
        target = ArchTarget.from_gfx("gfx950")
        geos = ks._geometry_list(target, "bf16", "dgrad")
        job = next(
            iter(
                ks._direction_jobs(
                    "dgrad",
                    "gfx950",
                    "bf16",
                    target,
                    ks.CACHE_WGRAD_SPLIT_KS,
                    64,
                    geometries=geos[:1],
                    llvm_flavor="llvm22",
                )
            )
        )
        ident = job.identity
    label = ident.label()
    direction, bare = label.split("-", 1)
    assert direction in conv_fill.DIRECTIONS
    other = next(d for d in conv_fill.DIRECTIONS if d != direction)
    for entry, kept in ((label, True), (bare, True), (f"{other}-{bare}", False)):
        f = tmp_path / "k.txt"
        f.write_text(entry + "\n")
        m = conv_fill.load_kernels_file(str(f))
        assert (label in conv_fill.kernels_for(m, "any")) is kept, entry


class _FakeTensor:
    def __init__(self, shape, dtype=None, value=None):
        self.shape, self.dtype, self.value, self.range = shape, dtype, value, None

    def uniform_(self, lo, hi, generator=None):
        self.range = (lo, hi)
        return self


class _FakeTorch:
    float16, bfloat16, float32 = "f16", "bf16", "f32"

    @staticmethod
    def empty(*shape, dtype=None):
        return _FakeTensor(shape, dtype)

    @staticmethod
    def full(shape, value):
        return _FakeTensor(tuple(shape), value=value)


@pytest.mark.parametrize("mode", conv_fill.FILL_CHOICES)
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_fwd_operands_follow_fill(monkeypatch, mode, dtype):
    """A forward benchmark's x / w follow --fill (U(-1, 1) by default)."""
    import sys

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    conv_fill.set_mode(mode)
    x, w = conv_fill.fwd_operands((2, 5, 7, 8), (16, 3, 1, 1), dtype, "f16")
    assert x.shape == (2, 5, 7, 8) and w.shape == (16, 3, 1, 1)
    assert x.dtype == w.dtype == "f16"
    if mode == "uniform":
        assert x.range == w.range == (-1.0, 1.0)
    else:
        assert x.range == (0.0, 0.01)
        lo = -0.01 / 3 if dtype == "fp16" else -0.005
        assert w.range == pytest.approx((lo, 0.005))
    x, _ = conv_fill.fwd_operands((1,), (1,), dtype)
    assert x.dtype is None  # torch's default (fp32)


@pytest.mark.parametrize("mode", conv_fill.FILL_CHOICES)
def test_implicit_fwd_sweep_inputs_follow_fill(monkeypatch, mode):
    """The implicit-GEMM fwd sweep's A / B follow --fill; --debug-init wins."""
    import sys
    import types

    from benchmarks.common import benchmark_implicit_gemm_conv as bi

    monkeypatch.setitem(sys.modules, "torch", _FakeTorch)
    conv_fill.set_mode(mode)
    args = types.SimpleNamespace(debug_init=None)
    a, b = bi._fwd_sweep_inputs((2, 5, 7, 8), (16, 3, 3, 8), "fp16", args)
    assert (a.shape, b.shape) == ((2, 5, 7, 8), (16, 3, 3, 8))
    if mode == "uniform":
        assert a.range == b.range == (-1.0, 1.0)
    else:
        assert a.range == (0.0, 0.01)
        assert b.range == pytest.approx((-0.01 / 3, 0.005))
    args.debug_init = 1.0
    a, b = bi._fwd_sweep_inputs((2, 5, 7, 8), (16, 3, 3, 8), "fp16", args)
    assert a.range is None and b.range is None and a.value == b.value == 1.0


def _fills_through_conv_fill(fn, helper):
    import inspect

    src = inspect.getsource(fn)
    assert "uniform_(" not in src and helper in src, fn.__name__
    assert "ref_floor=conv_fill.ref_floor()" in src, fn.__name__


def test_implicit_fwd_sweep_fills_through_conv_fill():
    """The implicit-GEMM fwd sweep draws no U(-1, 1) operands of its own and
    verifies with the --fill floor."""
    from benchmarks.common import benchmark_implicit_gemm_conv as bi

    _fills_through_conv_fill(bi._run_sweep, "_fwd_sweep_inputs(")


def test_direct_fwd_sweeps_fill_through_conv_fill():
    """Neither direct fwd sweep draws U(-1, 1) operands of its own; each
    verifies with the --fill floor."""
    pytest.importorskip("torch")  # the module imports the torch reference
    from benchmarks.common import benchmark_direct_conv as bd

    for fn in (bd._run_depthwise_sweep, bd._run_sweep, bd._run_nongrouped_sweep):
        _fills_through_conv_fill(fn, "conv_fill.fwd_operands(")


def test_uniform_inputs_unchanged():
    """--fill uniform draws exactly what the sweeps drew before: x then w,
    U(-1, 1), from the same seed."""
    torch = pytest.importorskip("torch")

    torch.manual_seed(42)
    x, w = conv_fill.fwd_operands((2, 5, 7, 8), (16, 3, 1, 4), "bf16", torch.bfloat16)
    torch.manual_seed(42)
    x0 = torch.empty(2, 5, 7, 8, dtype=torch.bfloat16).uniform_(-1.0, 1.0)
    w0 = torch.empty(16, 3, 1, 4, dtype=torch.bfloat16).uniform_(-1.0, 1.0)
    assert torch.equal(x, x0) and torch.equal(w, w0)
