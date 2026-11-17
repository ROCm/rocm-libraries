# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""gfx950's dual-wave kernel at head_dim 256 (vendored-kernel modification 43).

AOTriton's geometry for 256: 4 waves of 32 rows, so a 128-row tile, with each wave issuing
two KV DMAs per 64-column band, since the KV tile's 8 LDS lines no longer fall one to a
wave. The rows and the launch block are generator-level; the traits and the DMA map need
the kernel module, so FlyDSL, and skip without it. No object or GPU is read.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
sys.path.insert(0, str(FLYDSL / "kernels_src"))
from generators import _sdpa_cdna_build as build  # noqa: E402
from generators._instances import sdpa_cdna_dualwave_instances  # noqa: E402

WARP = 64


def _dualwave(arch: str = "gfx950"):
    return sdpa_cdna_dualwave_instances(arch)


def test_head_dim_256_ships_on_the_128_row_tile_one_wave_per_simd() -> None:
    wide = [i for i in _dualwave() if i.knobs["head_dim"] == 256]
    assert {(i.dtype, i.knobs["causal"]) for i in wide} == {
        (dtype, causal) for dtype in ("bf16", "f16") for causal in (0, 1)
    }
    for instance in wide:
        assert instance.knobs["block_m"] == 128, instance.name
        assert instance.knobs["waves_per_eu"] == 1, instance.name
        assert instance.knobs["has_bias"] == 0, instance.name


def test_the_other_head_dims_keep_the_256_row_tile() -> None:
    narrow = [i for i in _dualwave() if i.knobs["head_dim"] != 256]
    assert narrow
    for instance in narrow:
        assert instance.knobs["head_dim"] in (64, 128), instance.name
        assert instance.knobs["block_m"] == 256, instance.name
        assert instance.knobs["waves_per_eu"] == 0, instance.name


@pytest.mark.parametrize("arch", ["gfx942", "gfx90a"])
def test_only_gfx950_runs_the_dual_wave_kernel(arch: str) -> None:
    assert _dualwave(arch) == []


@pytest.mark.parametrize("block_m, threads", [(256, 512), (128, 256)])
def test_a_wave_of_64_per_32_rows(block_m: int, threads: int) -> None:
    assert build._BLOCK_THREADS["dualwave"](block_m) == threads


# --- the kernel's traits and KV DMA map (FlyDSL) ---------------------------------


@pytest.fixture(scope="module")
def utils():
    pytest.importorskip("flydsl")
    from kernels.attention_cdna import flash_attn_utils  # noqa: PLC0415

    return flash_attn_utils


def _traits(utils, head_dim: int, **kwargs):
    return utils._make_dualwave_swp_traits(32, 8, head_dim, **kwargs)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
def test_eight_waves_are_upstreams_traits(utils, head_dim: int) -> None:
    default = _traits(utils, head_dim)
    assert _traits(utils, head_dim, num_waves=8) == default
    assert (default.NUM_WAVES, default.BLOCK_M, default.BLOCK_SIZE) == (8, 256, 512)
    assert not default.DS_READ_TR_OP


def test_four_waves_change_the_wave_geometry_and_nothing_else(utils) -> None:
    eight, four = _traits(utils, 256), _traits(utils, 256, num_waves=4)
    differ = {
        f.name
        for f in dataclasses.fields(four)
        if getattr(four, f.name) != getattr(eight, f.name)
    }
    assert differ == {"NUM_WAVES", "BLOCK_M", "BLOCK_SIZE", "DS_READ_TR_OP"}
    assert (four.NUM_WAVES, four.BLOCK_M, four.BLOCK_SIZE) == (4, 128, 256)
    assert four.DS_READ_TR_OP
    # Two 64-row double buffers of K and V: past half of gfx950's 160 KiB.
    assert four.LDS_KV_TOTAL_SIZE * four.BF16_BYTES == 136192


@pytest.mark.parametrize("num_waves", [6, 16])
def test_other_wave_counts_are_refused(utils, num_waves: int) -> None:
    with pytest.raises(ValueError):
        _traits(utils, 256, num_waves=num_waves)


def _dma_map(utils, traits):
    """Every (wave, lane, DMA) of one K tile: the LDS (line, element) it writes and the
    KV (token, column) it reads, the KV rows at a pitch of the head dim."""
    pitch = traits.HEAD_DIM
    writes = {}
    for wave in range(traits.NUM_WAVES):
        for lane in range(WARP):
            n_in_warp, d_bucket = lane // traits.VEC_KV, lane % traits.VEC_KV
            for d in range(utils._kv_dma_count(traits)):
                line = wave + utils._kv_dma_line(traits, d)
                elem = utils._linear_kv_src_elem(
                    traits, 0, d, n_in_warp, wave, d_bucket, pitch
                )
                key = (line, lane * traits.VEC_KV)
                assert key not in writes, (wave, lane, d)
                writes[key] = divmod(elem, pitch)
    return writes


@pytest.mark.parametrize("head_dim, num_waves", [(64, 8), (128, 8), (256, 8), (256, 4)])
def test_the_kv_dma_fills_the_tile_the_reads_expect(
    utils, head_dim: int, num_waves: int
) -> None:
    """Each band of a KV tile is SMEM_N_RPT lines of 512 elements, and the reads take
    token t, column c from line t % N_RPT + (c // 64) * N_RPT at element
    (t // N_RPT) * 64 + c % 64 (`_k_lds_read_base_per_lane`, `_swizzled_ks_offset`). The
    DMAs must write every slot once, each with the token and column the reads take there.
    """
    traits = _traits(utils, head_dim, num_waves=num_waves)
    n_rpt, granule = traits.SMEM_N_RPT, traits.D_128B_SIZE
    writes = _dma_map(utils, traits)
    expected = {}
    for token in range(traits.BLOCK_N):
        for col in range(0, head_dim, traits.VEC_KV):
            line = token % n_rpt + (col // granule) * n_rpt
            expected[(line, (token // n_rpt) * granule + col % granule)] = (token, col)
    assert writes == expected


@pytest.mark.parametrize("num_waves, count", [(8, 4), (4, 8)])
def test_every_wait_counts_each_wave_s_dmas(utils, num_waves: int, count: int) -> None:
    assert utils._kv_dma_count(_traits(utils, 256, num_waves=num_waves)) == count
