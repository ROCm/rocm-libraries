# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatch-to-launch wiring for the gfx950 GDN prefill split path.

GDN prefill is the chunkwise KDA kernel in GDN mode: two split halves
(``chunk_prep`` + ``chunk_scan``), selected by an explicit algorithm pin, with
the scan's ``value_splits`` chosen from a batch-heads-banded tuning table. No
GPU here -- this asserts the request maps onto the right spec, tile and grid.
"""

from __future__ import annotations

import pytest

from dispatch.gdn import (
    GDN_PREFILL_REGISTRY,
    GdnPrefillRequest,
    dispatch_gdn_prefill,
    gdn_prefill_candidates,
)
from dispatch.gdn.prefill_gfx950 import value_splits_for

ARCH = "gfx950"


def _req(**kw) -> GdnPrefillRequest:
    base = dict(
        batch=8,
        seqlen=1024,
        arch=ARCH,
        num_k_heads=8,
        num_v_heads=8,  # BH = 8*8 = 64 by default
        head_k_dim=128,
        head_v_dim=128,
    )
    base.update(kw)
    return GdnPrefillRequest(**base)


def test_registry_has_exactly_the_two_split_halves():
    names = {c.name for c in gdn_prefill_candidates()}
    assert names == {
        "gdn_prefill_gfx950_chunk_prep",
        "gdn_prefill_gfx950_chunk_scan",
    }


def test_prep_carries_the_raw_gdn_gate():
    result = dispatch_gdn_prefill(_req(algorithm="chunk_prep"))
    assert result.candidate.name == "gdn_prefill_gfx950_chunk_prep"
    spec = result.spec
    assert spec.gate_kind == "gdn"
    assert spec.kv_group == 1  # num_v_heads == num_k_heads -> MHA
    assert spec.raw_inputs
    assert spec.fuse_gate
    assert spec.fuse_qk_l2norm
    assert spec.fuse_beta_sigmoid
    assert spec.has_dt_bias
    # Raw prep keeps the 256-thread builder regardless of the scan's split.
    assert spec.tile.block_size == 256


def test_prep_kv_group_tracks_gqa_ratio():
    result = dispatch_gdn_prefill(
        _req(num_k_heads=8, num_v_heads=32, algorithm="chunk_prep")
    )
    assert result.spec.kv_group == 4


@pytest.mark.parametrize(
    "num_v_heads,batch_heads,value_splits,block_size",
    [
        (8, 64, 8, 64),  # BH=64  -> vs8 (measured optimum)
        (16, 128, 2, 128),  # BH=128 -> vs2 (near-parity)
        (32, 256, 1, 256),  # BH=256 -> vs1 (natural parallelism)
    ],
)
def test_scan_value_splits_table(num_v_heads, batch_heads, value_splits, block_size):
    result = dispatch_gdn_prefill(
        _req(num_k_heads=8, num_v_heads=num_v_heads, algorithm="chunk_scan")
    )
    assert result.candidate.name == "gdn_prefill_gfx950_chunk_scan"
    assert result.request.batch_heads == batch_heads
    spec = result.spec
    assert spec.value_splits == value_splits
    assert spec.tile.block_size == block_size
    assert spec.token_major_io


def test_scan_vs8_uses_m16_atom():
    spec = dispatch_gdn_prefill(
        _req(num_k_heads=8, num_v_heads=8, algorithm="chunk_scan")
    ).spec
    assert spec.value_splits == 8
    assert spec.tile.scan_atom_m == 16


def test_value_splits_bands():
    assert value_splits_for(1) == 8
    assert value_splits_for(64) == 8
    assert value_splits_for(65) == 2
    assert value_splits_for(128) == 2
    assert value_splits_for(129) == 1
    assert value_splits_for(4096) == 1


def test_auto_has_no_fused_default():
    with pytest.raises(ValueError, match="no fused default"):
        dispatch_gdn_prefill(_req())  # algorithm/spec_id both "auto"


def test_spec_id_pin_bypasses_auto_guard_and_selects_scan():
    result = dispatch_gdn_prefill(_req(spec_id="gfx950_gdn_chunk_scan"))
    assert result.candidate.name == "gdn_prefill_gfx950_chunk_scan"


def test_seqlen_must_tile_chunk():
    with pytest.raises(ValueError):
        dispatch_gdn_prefill(_req(seqlen=1000, algorithm="chunk_scan"))


def test_gqa_ratio_must_divide():
    with pytest.raises(ValueError):
        dispatch_gdn_prefill(
            _req(num_k_heads=8, num_v_heads=12, algorithm="chunk_prep")
        )


def test_bf16_only():
    with pytest.raises(ValueError):
        dispatch_gdn_prefill(_req(dtype="f16", algorithm="chunk_scan"))


def test_wrong_arch_rejected():
    with pytest.raises(ValueError):
        dispatch_gdn_prefill(_req(arch="gfx942", algorithm="chunk_scan"))


def test_launch_geometry_is_present_and_consistent():
    result = dispatch_gdn_prefill(_req(algorithm="chunk_scan"))
    assert len(result.grid) == 3 and all(x > 0 for x in result.grid)
    assert result.block[0] == result.spec.tile.block_size
    assert len(result.signature) > 0


def test_prep_grid_scales_with_chunks():
    result = dispatch_gdn_prefill(_req(seqlen=1024, algorithm="chunk_prep"))
    # BH * num_chunks workgroups; num_chunks = 1024 / 32 = 32, BH = 64.
    assert result.request.num_chunks == 32
    assert len(result.grid) == 3 and all(x > 0 for x in result.grid)
