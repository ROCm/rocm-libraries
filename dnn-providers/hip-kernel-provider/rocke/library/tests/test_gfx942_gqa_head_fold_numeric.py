# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On-GPU numeric guardrail for the gfx942 GQA head-fold (D128 sliding-window).

The head-fold packs the ``num_queries_per_kv`` query heads that share a KV head
into one workgroup's 128-row M-tile (row m -> token m//4, head m%4), so the paged
K/V is loaded once per KV head instead of once per query head. It rewrites the
device-side ``(token, head)`` index math, the folded head index ``kv_head*4 + m%4``
and the launch grid. Its structure/selection is covered by the CPU emit test
(``test_gfx942_gqa_head_fold.py``); this file guards the one thing a host test
cannot: that the folded kernel still computes the *right numbers*.

A spec/emit test cannot catch a numeric regression in the emitted kernel -- that
is exactly what let #9198 (the D128 ring slot-reuse bug) ship. The guardrail below
launches the production kernel at unit-variance (randn) magnitude -- NOT the
uniform_(-0.1, 0.1) range whose near-uniform softmax hides such bugs -- and asserts
max_abs against an fp32 windowed paged-attention oracle. If a future edit flips the
(token, head) packing, gets the folded head index or grid wrong, or drops the
window mask, this fails.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

from kernels import UnifiedAttentionProblem
from kernels.common import attention_unified as au


def _load_windowed_harness():
    """Load the gfx950 prefill parity harness (its ``ref_paged_attn`` supports
    sliding windows and ``make_inputs`` builds randn paged K/V). The reference is
    pure torch (device-agnostic); only the rocke kernel it launches is arch-
    specific, and on a gfx942 device the production dispatcher selects the fold.
    """
    here = Path(__file__).resolve()
    # rocke/library/tests -> rocke/library/builders/gfx950/attention/prefill
    harness = (
        here.parents[1]
        / "builders/gfx950/attention/prefill/parity_unified_attention.py"
    )
    if not harness.exists():
        return None
    spec = importlib.util.spec_from_file_location(
        "parity_unified_attention_win", harness
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses in the harness can resolve __module__.
    sys.modules["parity_unified_attention_win"] = mod
    spec.loader.exec_module(mod)
    return mod


def _gpu_ready():
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    if not torch.cuda.is_available():
        return False
    name = torch.cuda.get_device_name(0).lower()
    # Substring match against the runtime device name; kept lowercase and generic
    # so it accepts every gfx942 part this cohort can run on.
    return "mi300" in name or "gfx942" in name


requires_gfx942_gpu = pytest.mark.skipif(
    not _gpu_ready(), reason="needs a gfx942 GPU with ROCm torch"
)


def _fold_problem(dtype="bf16", sq=4096, hq=32, hk=8, d=128, bs=32, window=4096):
    return UnifiedAttentionProblem(
        total_q=sq,
        num_seqs=1,
        num_query_heads=hq,
        num_kv_heads=hk,
        head_size=d,
        block_size=bs,
        max_seqlen_q=sq,
        max_seqlen_k=sq,
        dtype=dtype,
        sliding_window=window,
    )


# ---------------------------------------------------------------------------
# Routing anchor (no GPU): the numeric cohort below must build the fold kernel,
# so a future cohort change that stops routing D128-SWA-GQA-4:1-bf16 to the fold
# turns this file's numeric guardrail into a no-op -- and this assert red first.
# ---------------------------------------------------------------------------


@pytest.fixture
def gfx942():
    old_arch = au._RESOLVED_ATTENTION_ARCH
    au._RESOLVED_ATTENTION_ARCH = "gfx942"
    try:
        yield
    finally:
        au._RESOLVED_ATTENTION_ARCH = old_arch
        au._2D_LAUNCH_META.clear()


@pytest.mark.parametrize("bs", [16, 32])
def test_numeric_cohort_routes_to_fold(gfx942, bs):
    from kernels.common.attention_unified import gfx942_gqa_fold_eligible
    from kernels.gfx942.attention_tiled_2d import build_gfx942_4warp_gqa
    from rocke.core.ir_print import print_ir

    p = _fold_problem(bs=bs)
    assert gfx942_gqa_fold_eligible(
        p.head_size,
        p.num_queries_per_kv,
        p.sliding_window,
        p.dtype,
        p.block_size,
    ), "D128 SWA GQA-4:1 bf16 must be fold-eligible"
    spec = au._tiled_spec_from_problem(p)
    ir = print_ir(build_gfx942_4warp_gqa(spec, arch="gfx942"))
    assert re.search(r"_4wgqa_fold\b", ir), "cohort must build the fold kernel"


# ---------------------------------------------------------------------------
# On-GPU numeric guardrail -- the check a spec/emit test cannot make
# ---------------------------------------------------------------------------


@requires_gfx942_gpu
@pytest.mark.parametrize("bs", [16, 32])
@pytest.mark.parametrize("sq,window", [(2048, 512), (8192, 4096)])
@pytest.mark.parametrize("hq,hk", [(32, 8)])
def test_fold_numeric_vs_fp32_windowed_oracle(sq, hq, hk, window, bs):
    """Launch the production fold kernel at randn magnitude and assert max_abs
    against the fp32 windowed paged-attn oracle.

    The window is deliberately < seqlen so keys older than ``window`` are
    actually clipped -- a window >= kv_len degenerates to full causal and would
    pass even with a broken SWA lower bound. These cases exercise the window
    boundary the fold cohort is named for (an off-by-one in the drop-out-of-
    window mask fails here, not just a packing bug).

    ``bs`` matches the block sizes the routing anchor above asserts are
    fold-eligible, so everything the predicate ships is numerically checked.
    The two are NOT redundant: BPT = BN // BS is the number of paged KV blocks
    consumed per 32-key tile, so bs=32 walks one block-table entry per tile and
    bs=16 walks two -- a different gather, and a stride/index error in the
    second entry is invisible at bs=32.
    """
    import torch

    assert window < sq, "window must be < seqlen to exercise SWA masking"

    H = _load_windowed_harness()
    if H is None:
        pytest.skip("windowed parity harness not present in this checkout")

    s = H.Scenario(
        name=f"d128swa_fold_{hq}_{hk}_S{sq}_w{window}_bs{bs}",
        seq_lens=[(sq, sq)],
        num_query_heads=hq,
        num_kv_heads=hk,
        head_size=128,
        block_size=bs,  # fold requires block_size <= 32
        dtype=torch.bfloat16,
        sliding_window=window,
    )
    data = H.make_inputs(s, seed=0)
    ref = H.run_reference(s, data)
    out, _ = H.run_unified("rocke", s, data)
    m = H.compare(ref, out)
    # 4e-2 is the repo-wide bf16 attention gate (ALGORITHM.md §10, the gfx942
    # attention README, parity_unified_attention.py, both dense numeric tests).
    # The fold repacks which M-tile row holds which (token, head) pair; it does
    # not change the MFMA accumulation order or the fp32 accumulator, so it must
    # not need more headroom than the unfolded kernel.
    #
    # Measured on gfx942 (ROCm 7.13): max_abs = 0.01562 on ALL four
    # cases -- 2.6x inside this gate. That value is exactly 2^-6, i.e. one bf16
    # ulp for outputs in [2, 4), so the residual is the bf16 rounding of the
    # stored output, not accumulation drift. There is no mechanism here that
    # would need the 6e-2 this file originally asserted.
    assert m["max_abs"] <= 4e-2, (
        f"fold D128 SWA {hq}/{hk} S{sq} w{window} bs{bs}: max_abs "
        f"{m['max_abs']:.4f} > 4e-2 (fold numeric regression?)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
