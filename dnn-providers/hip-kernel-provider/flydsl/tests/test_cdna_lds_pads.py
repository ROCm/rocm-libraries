# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The generic CDNA kernel's LDS padding at head_dim 256, and the tiles that rely on it.

On gfx942's and gfx90a's 64 KiB of LDS the padded K and transposed-V tiles of a 256-wide
head do not fit when the whole K tile is staged at once, so those objects drop both pads
(`_GENERIC_LDS_PADS`) and their V^T reads conflict. A tile that stages K in column chunks
(`k_chunk`) or V in column slices (`dv_split`) fits with the pads, and keeps them. The plain
head_dim 256 prefill objects on those two arches are built that way. Generator-level only:
no object, descriptor or GPU is read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
from generators import _sdpa_cdna_build as build  # noqa: E402
from generators._instances import CdnaTier, _cdna_instance, instances_for  # noqa: E402

UNPADDED = {"k_pad": 0, "vt_pad": 0}
SMALL_LDS_ARCHES = ("gfx942", "gfx90a")


def _pads(arch: str, dtype: str = "bf16", head_dim: int = 256, **tier) -> dict:
    instance = _cdna_instance(
        dtype, head_dim, 0, 0, CdnaTier("generic", 128, **tier), 100, 0, 1
    )
    return build._lds_pads(build._as_built("sdpa_cdna", instance), arch)


@pytest.mark.parametrize("arch", SMALL_LDS_ARCHES)
def test_a_whole_k_tile_at_256_drops_both_pads(arch: str) -> None:
    assert _pads(arch) == UNPADDED
    assert _pads(arch, dtype="f16") == UNPADDED


@pytest.mark.parametrize("arch", SMALL_LDS_ARCHES)
@pytest.mark.parametrize(
    "tier",
    [
        {"k_chunk": 128, "kv_gpfetch": 0},
        {"k_chunk": 64, "kv_gpfetch": 0},
        {"dv_split": 2},
        {"dv_split": 2, "k_chunk": 128, "kv_gpfetch": 0},
    ],
)
def test_a_chunked_k_or_split_v_tile_keeps_the_pads(arch: str, tier: dict) -> None:
    assert _pads(arch, **tier) == {}


def test_only_the_64_kib_parts_at_256_change_the_pads() -> None:
    assert _pads("gfx950") == {}
    assert _pads("gfx942", head_dim=128) == {}
    assert _pads("gfx942", dtype="f32") == {}


@pytest.mark.parametrize("arch", SMALL_LDS_ARCHES)
def test_the_plain_256_prefill_objects_stage_k_in_chunks_with_the_pads(
    arch: str,
) -> None:
    plain = [
        i
        for i in instances_for("sdpa_cdna", arch)
        if i.knobs["head_dim"] == 256
        and i.knobs["head_dim_v"] == 256
        and i.dtype in ("bf16", "f16")
        and not any(
            i.knobs[k] for k in ("qk_mask", "has_bias", "has_sink", "has_dropout")
        )
    ]
    assert plain, arch
    for instance in plain:
        assert instance.knobs["k_chunk"] == 128, instance.name
        assert instance.knobs["kv_gpfetch"] == 0, instance.name
        assert (
            build._lds_pads(build._as_built("sdpa_cdna", instance), arch) == {}
        ), instance.name
    # Each class, causal and not, in both dtypes; and the non-causal key-aligned build.
    classes = {(i.dtype, i.knobs["causal"], i.knobs["kv_align"]) for i in plain}
    assert classes == {
        (dtype, causal, align)
        for dtype in ("bf16", "f16")
        for causal, align in ((0, 0), (0, 64), (1, 0))
    }, arch


@pytest.mark.parametrize("arch", SMALL_LDS_ARCHES)
def test_the_256_dropout_objects_keep_the_whole_k_tile(arch: str) -> None:
    dropout = [
        i
        for i in instances_for("sdpa_cdna", arch)
        if i.knobs["head_dim"] == 256
        and i.knobs["has_dropout"]
        and not i.knobs["qk_mask"]
        and i.dtype != "f32"
    ]
    assert dropout, arch
    for instance in dropout:
        assert instance.knobs["k_chunk"] == 0, instance.name
        assert build._lds_pads(build._as_built("sdpa_cdna", instance), arch) == (
            UNPADDED
        ), instance.name
