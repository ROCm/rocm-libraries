# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for gfx942 split/fused chunkwise KDA."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROCKE = Path(__file__).resolve().parents[1]
_BUILDER = _ROCKE / "builders" / "gfx942" / "kda"
sys.path.insert(0, str(_BUILDER))


def _gpu_ready() -> bool:
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        props = torch.cuda.get_device_properties(0)
        arch = getattr(props, "gcnArchName", "")
        if not arch:
            arch = torch.cuda.get_arch_list()[0]
        return "gfx942" in arch
    except Exception:  # noqa: BLE001
        return False


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not _gpu_ready(), reason="needs a gfx942 GPU with ROCm torch"),
]

GATES = (-0.1, -0.5, -2.0, -5.0)
TOL = 3e-2


@pytest.mark.parametrize("gate_low", GATES)
def test_prep_tiles_match_float64_oracle(gate_low):
    import kda_chunk_prep as prep
    from kernels.gfx942.kda_chunkwise import KdaChunkPrepSpec

    assert prep.check(
        KdaChunkPrepSpec(),
        num_tiles=128,
        gate_low=gate_low,
        tol=2e-2,
        verbose=False,
    )


@pytest.mark.parametrize("gate_low", GATES)
def test_split_path_matches_token_serial(gate_low):
    import kda_chunk_split as split
    from kernels.gfx942.kda_chunkwise import KdaChunkScanSpec

    worst = split.check(
        KdaChunkScanSpec(),
        B=2,
        H=4,
        T=256,
        gate_low=gate_low,
        logical_head_v=128,
        verbose=False,
    )
    assert worst <= TOL


@pytest.mark.parametrize("gate_low", GATES)
def test_fused_path_matches_token_serial(gate_low):
    import kda_chunk_fused as fused
    from kernels.gfx942.kda_chunkwise import KdaChunkFusedSpec

    worst = fused.check(
        KdaChunkFusedSpec(),
        B=2,
        H=4,
        T=256,
        gate_low=gate_low,
        logical_head_v=128,
        verbose=False,
    )
    assert worst <= TOL


# head_k sets the group count of the intra-chunk cumulative-decay fold:
# block_size / head_k threads share a channel, so 128 folds two row-groups and
# 64 folds four. Only from three groups up does the fold read a group total out
# of a row the fold itself rewrites, so 64 is the width that exercises the
# hazard and the reason the default 128 everywhere else never caught it.
FOLD_HEAD_KS = (64, 96, 128)

# The fold failed intermittently -- roughly one launch in ten on a single
# workgroup -- so a single check would pass most of the time. B2/H4/T256 is 32
# workgroups, which makes one bad fold anywhere a failure, and the repeats are
# what turn "usually passes" into a real gate.
FOLD_REPEATS = 4


@pytest.mark.parametrize("head_k", FOLD_HEAD_KS)
def test_fused_grouped_cumsum_fold_is_stable(head_k):
    import kda_chunk_fused as fused
    from kernels.gfx942.kda_chunkwise import KdaChunkFusedSpec

    spec = KdaChunkFusedSpec(head_k=head_k)
    for attempt in range(FOLD_REPEATS):
        worst = fused.check(
            spec,
            B=2,
            H=4,
            T=256,
            gate_low=-0.5,
            logical_head_v=128,
            verbose=False,
        )
        assert worst <= TOL, (
            f"head_k={head_k} attempt {attempt}: worst={worst:.3e} > {TOL:.1e}"
        )


def test_split_and_fused_agree_bitwise():
    import torch

    import kda_chunk_fused as fused
    import kda_chunk_split as split
    from kernels.gfx942.kda_chunkwise import (
        KdaChunkFusedSpec,
        KdaChunkScanSpec,
    )

    B, H, T, DK, DV = 2, 4, 256, 128, 128
    q, k, v, g, beta = fused.make_inputs(B, H, T, DK, DV)
    out_s, state_s = split.launch_packed(KdaChunkScanSpec(), q, k, v, g, beta)
    out_f, state_f = fused.launch_packed(KdaChunkFusedSpec(), q, k, v, g, beta)
    torch.cuda.synchronize()
    assert torch.equal(out_s, out_f)
    assert torch.equal(state_s, state_f)


def test_fused_input_prefetch_is_bitwise_identical():
    import torch

    import kda_chunk_fused as fused
    from kernels.gfx942.kda_chunkwise import KdaChunkFusedSpec

    B, H, T, DK, DV = 2, 4, 256, 128, 128
    q, k, v, g, beta = fused.make_inputs(B, H, T, DK, DV)
    base = KdaChunkFusedSpec(prefetch_inputs=False)
    prefetched = KdaChunkFusedSpec(prefetch_inputs=True)
    out_b, state_b = fused.launch_packed(base, q, k, v, g, beta)
    out_p, state_p = fused.launch_packed(prefetched, q, k, v, g, beta)
    torch.cuda.synchronize()
    assert torch.equal(out_b, out_p)
    assert torch.equal(state_b, state_p)
