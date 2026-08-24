# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-GPU numeric lane for the chunkwise KDA family on gfx950.

Two references, deliberately independent of each other and of the kernels:

- The six per-chunk tiles are checked against a float64 torch oracle that
  builds them from the pairwise exponent *difference*, with no midpoint
  factoring -- so it never forms the overflowing ``Gamma_i / Gamma_j`` the
  kernel is designed around, and agreement is a check of that factoring rather
  than a restatement of it.
- Both full forward paths are checked against a token-serial float64 walk of
  the gated delta rule. That oracle is not chunked at all, so agreement tests
  the chunkwise factorization itself.

The gate range is swept to -5, the reference ``gate_lower_bound``: a 32-token chunk
accumulates up to 160 nats there, which is the regime that saturates the
factored exponents and the only one where the clamping actually matters.

Every test is marked ``gpu`` and skipped off a gfx950. Select with
``run_all.py --gpu`` (or ``pytest -m gpu``); the default CPU lane excludes it.
"""

from __future__ import annotations

import os
import sys

import pytest

# The host builders live under library/builders and import each other by module
# name, the way they do when run as scripts.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "builders/gfx950/kda",
    ),
)


def _gpu_ready():
    """True only on a gfx950 box with ROCm torch.

    Gate on ``gcnArchName`` (the ISA target), NOT the marketing name.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName.lower()
    return "gfx950" in arch


requires_gfx950_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx950 GPU with ROCm torch"
)

pytestmark = [pytest.mark.gpu, requires_gfx950_gpu]

# -0.1 is the typical regime; -5.0 is the reference gate lower bound and is what
# saturates the factored exponents.
GATES = [-0.1, -0.5, -2.0, -5.0]
# bf16 operands and an fp32 accumulator against a float64 oracle. The chunkwise
# path also carries a triangular solve, so the tolerance is a bf16 tolerance
# with room for its conditioning, not a tight fp32 one.
TOL = 3e-2


@pytest.mark.parametrize("gate_low", GATES)
def test_prep_tiles_match_float64_oracle(gate_low):
    """All six per-chunk tiles, over enough chunks to cover every code path."""
    import kda_chunk_prep as prep

    from kernels.gfx950.kda_chunkwise import KdaChunkPrepSpec

    spec = KdaChunkPrepSpec()
    assert prep.check(spec, 128, gate_low=gate_low, tol=2e-2, verbose=False)


@pytest.mark.parametrize("gate_low", GATES)
def test_split_path_matches_token_serial(gate_low):
    """rocke prep + rocke scan, against the token-serial recurrence."""
    import kda_chunk_split as split

    from kernels.gfx950.kda_chunkwise import KdaChunkScanSpec

    worst = split.check(KdaChunkScanSpec(), 2, 4, 256, gate_low=gate_low, verbose=False)
    assert worst <= TOL, f"gate {gate_low}: worst rel {worst:.3e}"


@pytest.mark.parametrize("gate_low", GATES)
def test_fused_path_matches_token_serial(gate_low):
    """One kernel, tiles never leaving LDS, against the same oracle.

    The two paths run the same emitted scan body over the same tile math, so
    this is the check that routing the tiles through HBM instead of LDS did not
    change the arithmetic.
    """
    import kda_chunk_fused as fused

    from kernels.gfx950.kda_chunkwise import KdaChunkFusedSpec

    worst = fused.check(
        KdaChunkFusedSpec(), 2, 4, 256, gate_low=gate_low, verbose=False
    )
    assert worst <= TOL, f"gate {gate_low}: worst rel {worst:.3e}"


def test_split_and_fused_agree_bitwise():
    """The two paths share one emitted scan body, so they must agree exactly.

    Not merely within tolerance: same math, same order, same rounding. A
    difference here means the tile round trip through HBM lost something the
    LDS path kept, which no tolerance-based check would localize.
    """
    import torch

    import kda_chunk_fused as fused
    import kda_chunk_split as split

    from kernels.gfx950.kda_chunkwise import KdaChunkFusedSpec, KdaChunkScanSpec

    B, H, T = 2, 4, 256
    spec_s = KdaChunkScanSpec()
    q, k, v, g, beta = fused.make_inputs(B, H, T, spec_s.head_k, spec_s.head_v)
    o_s, ht_s = split.launch_packed(spec_s, q, k, v, g, beta)
    o_f, ht_f = fused.launch_packed(KdaChunkFusedSpec(), q, k, v, g, beta)
    torch.cuda.synchronize()
    assert torch.equal(o_s, o_f), "outputs diverge between the split and fused paths"
    assert torch.equal(ht_s, ht_f), "final states diverge"


def test_scan_rejects_a_spec_it_cannot_emit():
    """The admission rule is part of the contract, not a convenience.

    Checked on the GPU lane too because this is the guard that stops a spec
    whose lane mapping has no valid emission from reaching the launcher.
    """
    from kernels.gfx950.kda_chunkwise import KdaChunkScanSpec

    import kda_chunk_split as split

    with pytest.raises(ValueError, match="unsupported spec"):
        split.make_launcher(KdaChunkScanSpec(head_v=64))
