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
