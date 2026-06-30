# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Self-contained per-level reproduction driver for the FP8 fused-MoE MEGA-kernel.

Regenerates **each of the 12 documented optimization levels** (cumulative
best->) of the fp8 fused-MoE mega-kernel: for every level it materializes the
kernel (via a spec/build/grid **flag-config** on the production
``moe_fused_mega_fp8.py`` *or* by file-path-loading a curated
``levels/level_NN_<name>.py`` **snapshot**), runs the HARDENED parity gate
(T1 / T8 / skewed-E0), times a launch-only perf measurement (T1 + T8, warm
best-of-N), and prints one table row of the kernel's OWN numeric perf.

This is the clean, skinny-decode-style entry point: ``reproduce_levels.py`` is
the single, fully standalone driver that turns the campaign audit trail into a
reproducible ledger. It depends on NOTHING under ``/tmp`` and ships NO external
hand-tuned-asm comparison -- only the kernel's numeric T1/T8 ms. All the parity
machinery and the launch-only timing closure are inlined below, so the script
needs neither ``parity_fp8.py`` nor ``perf_fp8.py`` present.

The 12 documented levels (cumulative best, T1 ms, from the README
step table):

    L0  vec-load (baseline)                 1.83 -> 0.872  snapshot
    L1  tile_m 32->16 (kill padded-M)       0.871-> 0.472  flag (tile_m) on snap
    L2  fuse 3-pass quant -> 1              0.472-> 0.337  snapshot
    L3  down software-pipeline              0.337-> 0.333  snapshot
    L4  m_tile_base correctness fix         0.333-> 0.331  snapshot (ALWAYS-ON)
    L5  gate+up SW-pipeline + wave IL       0.331-> 0.291  snapshot
    L6  epilogue drain hoist               0.291-> 0.280  sub-diff of L5
    L7  K=128 hero atom                      0.280-> 0.182  flag (gate_up_k/down_k)
    L8  direct-to-LDS gate+up               0.182-> 0.161  flag (use_dtla)
    L9  iglp_opt(1) cadence                  0.161-> 0.157  flag (sched_cadence)
    L10 active de-padded grid                0.152-> 0.131  flag (harness grid)
    L11 persistent kernel + XCD (NEUTRAL)    0.131         flag (persistent) OFF-default

Run from ``dnn-providers/hip-kernel-provider/rocKE/Python`` with ``PYTHONPATH=$(pwd)``::

    PYTHONPATH=$(pwd) python \
        -m rocke.examples.gfx950.fused_mega_moe.reproduce_levels

    # subset / toggles
    ... reproduce_levels --levels 0,1,7,11
    ... reproduce_levels --levels 0-9 --no-perf
"""

from __future__ import annotations

import argparse
import gc
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
LEVELS_DIR = HERE / "levels"
ROOT = HERE.resolve().parents[5]  # .../composablekernel
sys.path.insert(0, str(ROOT / "Python"))

import torch  # noqa: E402

from rocke.examples.gfx950.moe.fused_moe_e2e_perf import (  # noqa: E402
    Scenario,
    make_inputs,
    time_callable_ms,
    _compare,
)
from rocke.helpers.compile import compile_kernel  # noqa: E402
from rocke.runtime.launcher import (  # noqa: E402
    KernelLauncher,
    LaunchConfig,
)
from rocke.examples.gfx950.fused_mega_moe.levels._build_by_path import (  # noqa: E402
    load_level_module,
)

# Production kernel (the FINAL best; all levers default-on).
from rocke.instances.common import moe_fused_mega_fp8 as PROD  # noqa: E402


# ---------------------------------------------------------------------------
# Self-contained constants + helpers (inlined; asm-free).
#   Timing knobs + parity tolerance + fp8 quant/reference machinery, copied
#   verbatim from the legacy parity_fp8 / perf_fp8 harnesses so this driver is
#   fully standalone (no import from those modules, no /tmp dependency).
# ---------------------------------------------------------------------------

# --- timing knobs (was perf_fp8) ---
WARMUP = 25
ATTEMPTS = 50
OUTER_RUNS = 5

# --- fp8 quant constants + parity tolerance (was parity_fp8) ---
GROUP_K = 128
FP8_MAX = 448.0
AMAX_FLOOR = 1e-6
TOL = 1.5e-2
TILE_M = 32  # FusedMegaKernelSpecFp8.tile_m default for the hardened spec


def _warm_gpu() -> None:
    """~30 untimed 4096x4096 matmuls to bring the GPU to a warm clock state."""
    x = torch.randn(4096, 4096, device="cuda")
    for _ in range(30):
        x = x @ x
    torch.cuda.synchronize()


def _isolate_lane() -> None:
    torch.cuda.synchronize()
    try:
        from rocke.runtime.launcher import synchronize_and_release

        synchronize_and_release()
    except Exception:
        pass
    gc.collect()


def _quant_fp8(x_f32: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize ``x_f32 / scale`` to fp8 e4m3 (RNE + saturating clamp)."""
    q = x_f32 / scale
    q = torch.clamp(q, -FP8_MAX, FP8_MAX)
    return q.to(torch.float8_e4m3fn)


def _blockwise_scale(w: torch.Tensor, out_axis: int, k_axis: int) -> torch.Tensor:
    """Per-128-block amax/448 scale for a 2D weight, blocking BOTH axes by 128.

    Returns a ``[ceil(out/128), ceil(k/128)]`` f32 scale tensor.
    """
    out_dim, Kdim = w.shape
    nob = (out_dim + GROUP_K - 1) // GROUP_K
    nkb = (Kdim + GROUP_K - 1) // GROUP_K
    wb = w.reshape(nob, GROUP_K, nkb, GROUP_K).abs()
    amax = wb.amax(dim=(1, 3))  # [nob, nkb]
    amax = torch.clamp(amax, min=AMAX_FLOOR)
    return (amax / FP8_MAX).float()


def _apply_skew_routing(inputs, s: Scenario, e0: int) -> None:
    """Force a skewed routing that piles ALL tokens onto expert ``e0``.

    For the HARDENED at-scale parity case: every token's top-1 pick is ``e0``;
    any remaining topk columns get distinct ``(e0+k) % experts`` ids so the
    softmax weights are well-defined. This packs expert ``e0``'s slot with
    > tile_m=16 distinct, magnitude-significant tokens, exercising the
    down-GEMM ``mi > 0`` atom -- the path the balanced T1/T8 routing never hits.
    """
    T = s.tokens
    K = s.topk
    E = s.experts
    device = inputs.X.device
    ids = torch.empty(T, K, dtype=torch.int32, device=device)
    ids[:, 0] = e0
    for k in range(1, K):
        ids[:, k] = (e0 + k) % E
    g = torch.Generator(device=device)
    g.manual_seed(20260603)
    vals = torch.randn(T, K, dtype=torch.float32, device=device, generator=g)
    weights = torch.softmax(vals, dim=-1)
    inputs.topk_ids_pre = ids
    inputs.topk_weights_pre = weights


def build_static_padded_inputs(inputs, s: Scenario, tile_m: int = TILE_M):
    """Reconstruct the production static-offset (de-padded/compacted) layout.

    Returns the tensors the mega-kernel consumes plus the compacted-layout
    descriptors (``expert_base`` / ``blocks_per_expert`` / ``counts``). Each
    ACTIVE expert contributes only ceil(count[e]/tile_m) blocks packed to the
    FRONT; ``num_m_blocks = sum_e ceil(count[e]/tile_m)``.
    """
    TILE_M = tile_m
    device = inputs.X.device
    E = s.experts
    H = s.hidden

    top_ids = inputs.topk_ids_pre  # (T, K) i32
    top_weights = inputs.topk_weights_pre  # (T, K) f32

    counts = [int((top_ids == e).sum()) for e in range(E)]
    blocks_per_expert = [(c + TILE_M - 1) // TILE_M for c in counts]
    num_m_blocks = sum(blocks_per_expert)
    if num_m_blocks == 0:
        num_m_blocks = 1

    total_padded = num_m_blocks * TILE_M

    grouped_input_padded = torch.zeros(
        total_padded, H, dtype=inputs.X.dtype, device=device
    )
    sorted_token_ids_padded = torch.full(
        (total_padded,), -1, dtype=torch.int32, device=device
    )
    sorted_weights_padded = torch.zeros(
        total_padded, dtype=torch.float32, device=device
    )
    block_expert_cpu = torch.full((num_m_blocks,), -1, dtype=torch.int32, device=device)

    expert_base = [-1] * E

    blk = 0
    for e in range(E):
        be = blocks_per_expert[e]
        if be == 0:
            continue
        mask = top_ids == e  # (T, K)
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        ce = int(token_idx.numel())
        poff = blk * TILE_M
        expert_base[e] = poff
        block_expert_cpu[blk : blk + be] = e
        grouped_input_padded[poff : poff + ce] = inputs.X[token_idx]
        sorted_token_ids_padded[poff : poff + ce] = token_idx.to(torch.int32)
        sorted_weights_padded[poff : poff + ce] = top_weights[token_idx, slot_idx].to(
            torch.float32
        )
        blk += be

    block_expert_ids = block_expert_cpu

    return {
        "grouped_input_padded": grouped_input_padded,
        "sorted_token_ids_padded": sorted_token_ids_padded,
        "sorted_weights_padded": sorted_weights_padded,
        "block_expert_ids": block_expert_ids,
        "slot_size": TILE_M,
        "total_padded": total_padded,
        "num_m_blocks": num_m_blocks,
        "blocks_per_expert": blocks_per_expert,
        "expert_base": expert_base,
        "counts": counts,
        "tile_m": TILE_M,
    }


def make_fp8_inputs(s: Scenario, *, seed: int = 11939):
    """Build the f32 masters + routing, then quantize to fp8 with block scales."""
    inputs = make_inputs(s, seed=seed)
    skew = getattr(s, "_skew_expert", None)
    if skew is not None:
        _apply_skew_routing(inputs, s, int(skew))
    device = inputs.X.device
    E, ni, H = inputs.W_gate.shape  # W_gate (E, I, H), contraction H
    nHb = H // GROUP_K
    nIb = ni // GROUP_K

    Wg_f32 = inputs.W_gate.float()  # (E, I, H)
    Wu_f32 = inputs.W_up.float()
    Wd_f32 = inputs.W_down.float()  # (E, H, I), contraction I

    Wg_q = torch.empty(E, ni, H, dtype=torch.float8_e4m3fn, device=device)
    Wu_q = torch.empty(E, ni, H, dtype=torch.float8_e4m3fn, device=device)
    gate_scale = torch.empty(E, nHb, nIb, dtype=torch.float32, device=device)
    up_scale = torch.empty(E, nHb, nIb, dtype=torch.float32, device=device)
    for e in range(E):
        sg = _blockwise_scale(Wg_f32[e], out_axis=0, k_axis=1)  # [nIb, nHb]
        su = _blockwise_scale(Wu_f32[e], out_axis=0, k_axis=1)
        gate_scale[e] = sg.T.contiguous()
        up_scale[e] = su.T.contiguous()
        sg_full = sg.repeat_interleave(GROUP_K, 0).repeat_interleave(GROUP_K, 1)
        su_full = su.repeat_interleave(GROUP_K, 0).repeat_interleave(GROUP_K, 1)
        Wg_q[e] = _quant_fp8(Wg_f32[e], sg_full)
        Wu_q[e] = _quant_fp8(Wu_f32[e], su_full)

    Wd_q = torch.empty(E, H, ni, dtype=torch.float8_e4m3fn, device=device)  # (E, H, I)
    down_scale = torch.empty(E, nIb, nHb, dtype=torch.float32, device=device)
    for e in range(E):
        sd = _blockwise_scale(Wd_f32[e], out_axis=0, k_axis=1)  # [nHb, nIb]
        down_scale[e] = sd.T.contiguous()
        sd_full = sd.repeat_interleave(GROUP_K, 0).repeat_interleave(GROUP_K, 1)
        Wd_q[e] = _quant_fp8(Wd_f32[e], sd_full)

    return {
        "inputs": inputs,
        "X_f32": inputs.X.float(),  # (T, H)
        "Wg_q": Wg_q,
        "Wu_q": Wu_q,
        "Wd_q": Wd_q,
        "gate_scale": gate_scale,  # [E, nHb, nIb]
        "up_scale": up_scale,  # [E, nHb, nIb]
        "down_scale": down_scale,  # [E, nIb, nHb]
        "nHb": nHb,
        "nIb": nIb,
    }


def _build_padded_activation(fp8in, padded, s: Scenario):
    """Build the padded fp8 ``A`` + ``AScale`` in the sorted layout.

    ONE scale per m-block per kg (amax over the valid rows), broadcast to every
    row of the block incl. padding rows, matching the kernel's per-lane fold.
    Returns ``(A_q, AScale)``.
    """
    inputs = fp8in["inputs"]
    device = inputs.X.device
    nHb = fp8in["nHb"]
    H = s.hidden
    E = s.experts
    total_padded = padded["total_padded"]
    TILE_M = padded["tile_m"]
    expert_base = padded["expert_base"]
    blocks_per_expert = padded["blocks_per_expert"]

    A_q = torch.zeros(total_padded, H, dtype=torch.float8_e4m3fn, device=device)
    AScale = torch.full(
        (total_padded, nHb), AMAX_FLOOR / FP8_MAX, dtype=torch.float32, device=device
    )

    top_ids = inputs.topk_ids_pre
    Xf = fp8in["X_f32"]
    for e in range(E):
        m = top_ids == e
        token_idx, _ = m.nonzero(as_tuple=True)
        ce = int(token_idx.numel())
        if ce == 0:
            continue
        base = expert_base[e]
        be = blocks_per_expert[e]
        X_sub = Xf[token_idx]  # (ce, H)
        xb = X_sub.reshape(ce, nHb, GROUP_K).abs()
        blk_amax = torch.clamp(xb.amax(dim=(0, 2)), min=AMAX_FLOOR)  # (nHb,)
        blk_scale = (blk_amax / FP8_MAX).float()  # (nHb,)
        blk_scale_full = blk_scale.repeat_interleave(GROUP_K)  # (H,)
        Xq = _quant_fp8(X_sub, blk_scale_full.unsqueeze(0))  # (ce, H)
        A_q[base : base + ce] = Xq
        AScale[base : base + be * TILE_M] = blk_scale.unsqueeze(0)
    return A_q, AScale


def torch_fused_moe_fp8_reference(
    fp8in, padded, A_q, AScale, s: Scenario, *, tile_m: int
):
    """f32-accumulator oracle gathering the SAME padded fp8 A/AScale the kernel
    consumes; the dynamic Hidden quant is done PER ``tile_m``-row m-block.
    """
    inputs = fp8in["inputs"]
    nIb = fp8in["nIb"]
    top_ids = inputs.topk_ids_pre
    top_weights = inputs.topk_weights_pre
    T, H = inputs.X.shape
    device = inputs.X.device
    expert_base = padded["expert_base"]

    Y = torch.zeros(T, H, dtype=torch.float32, device=device)

    for e in range(s.experts):
        mask = top_ids == e
        if not mask.any():
            continue
        token_idx, slot_idx = mask.nonzero(as_tuple=True)
        if token_idx.numel() == 0:
            continue
        count = token_idx.numel()
        base = expert_base[e]

        Xq = A_q[base : base + count]  # (count, H) fp8
        a_scale = AScale[base : base + count]  # (count, nHb)
        a_scale_full = a_scale.repeat_interleave(GROUP_K, dim=1)  # (count, H)
        Xq_f32 = Xq.float() * a_scale_full  # dequant

        sg = fp8in["gate_scale"][e]  # [nHb, nIb] (kg, n_blk)
        su = fp8in["up_scale"][e]
        sg_full = sg.T.repeat_interleave(GROUP_K, 0).repeat_interleave(
            GROUP_K, 1
        )  # (I, H)
        su_full = su.T.repeat_interleave(GROUP_K, 0).repeat_interleave(GROUP_K, 1)
        Wg_dq = fp8in["Wg_q"][e].float() * sg_full  # (I, H)
        Wu_dq = fp8in["Wu_q"][e].float() * su_full
        gate = Xq_f32 @ Wg_dq.T  # (count, I)
        up = Xq_f32 @ Wu_dq.T
        hidden = torch.nn.functional.silu(gate) * up  # (count, I) f32

        sd = fp8in["down_scale"][e]  # [nIb, nHb] (inter_blk, h_out_blk)
        sd_full = sd.T.repeat_interleave(GROUP_K, 0).repeat_interleave(
            GROUP_K, 1
        )  # (H, I)
        Wd_dq = fp8in["Wd_q"][e].float() * sd_full  # (H, I)

        out = torch.empty(count, H, dtype=torch.float32, device=device)
        for blk_start in range(0, count, tile_m):
            blk_end = min(blk_start + tile_m, count)
            h_blk = hidden[blk_start:blk_end]  # (rows, I)
            rows = h_blk.shape[0]
            hb = h_blk.reshape(rows, nIb, GROUP_K).abs()
            h_amax = torch.clamp(hb.amax(dim=(0, 2)), min=AMAX_FLOOR)  # (nIb,)
            h_scale = (h_amax / FP8_MAX).float()
            h_scale_full = h_scale.repeat_interleave(GROUP_K)  # (I,)
            Hq = _quant_fp8(h_blk, h_scale_full.unsqueeze(0))
            Hq_f32 = Hq.float() * h_scale_full.unsqueeze(0)
            out[blk_start:blk_end] = Hq_f32 @ Wd_dq.T  # (rows, H)

        w = top_weights[token_idx, slot_idx].unsqueeze(-1)
        Y.index_add_(0, token_idx, w * out)

    return Y


# ---------------------------------------------------------------------------
# Scenarios (canonical decode; same as parity_fp8 / perf_fp8)
# ---------------------------------------------------------------------------


def _scn_t1() -> Scenario:
    return Scenario(
        name="decode_T1_E8_K2_H4096_I7168",
        tokens=1,
        experts=8,
        topk=2,
        hidden=4096,
        intermediate=7168,
    )


def _scn_t8() -> Scenario:
    return Scenario(
        name="decode_T8_E8_K2_H4096_I7168",
        tokens=8,
        experts=8,
        topk=2,
        hidden=4096,
        intermediate=7168,
    )


def _scn_hard() -> Scenario:
    s = Scenario(
        name="hard_skewE0_T64_E8_K1_H4096_I7168",
        tokens=64,
        experts=8,
        topk=1,
        hidden=4096,
        intermediate=7168,
    )
    object.__setattr__(s, "_skew_expert", 0)
    return s


# ---------------------------------------------------------------------------
# Documented level catalog (cumulative best->)
# ---------------------------------------------------------------------------


@dataclass
class LevelDef:
    idx: int
    name: str
    lever: str
    mechanism: str  # "snapshot" | "flag"
    before: float  # documented T1 ms BEFORE this lever
    after: float  # documented cumulative-best T1 ms AFTER
    snapshot: Optional[str] = None  # levels/level_NN_*.py filename
    tile_m: int = 16
    gate_up_k: int = 128
    down_k: int = 128
    use_dtla: bool = True
    sched_cadence: Optional[str] = None  # None == iglp1 default
    grid_mode: str = "active"  # "static" | "active" | "persistent"
    persistent: bool = False
    hardened_required: bool = True  # L0-L3 predate the L4 m_tile_base fix:
    #   the skewE0 (>tile_m) hardened case is EXPECTED to fail there (the bug is
    #   live). For those levels the skewE0 FAIL is documented, not a blocker; the
    #   canonical T1/T8 parity still gates perf. L4+ require skewE0 PASS.
    note: str = ""


# Notes on grid_mode for snapshot levels (L0-L6): the snapshots emit the
# pre-active-grid kernel; their own ``moe_fused_mega_fp8_grid`` already returns
# the de-padded ``(grid.x, num_m_blocks)`` since ``build_static_padded_inputs``
# packs active blocks to the front. To honestly reproduce the PRE-L10 "static
# (28,8)" launch we force a STATIC pad (one full slot per expert) for L0-L9 and
# only switch to the de-padded ACTIVE grid at L10 (the lever under test). L11
# adds the persistent relinearization on top of active.
LEVELS = [
    LevelDef(
        0,
        "level_00_baseline",
        "vec-load (baseline)",
        "snapshot",
        1.83,
        0.872,
        snapshot="level_00_baseline.py",
        tile_m=32,
        grid_mode="static",
        hardened_required=False,
        note="byte->vec delta; baseline IS pre-vec/pre-tile_m state; "
        "pre-L4 fix so skewE0 hardened parity is EXPECTED-FAIL",
    ),
    LevelDef(
        1,
        "level_01_tilem16",
        "tile_m 32->16 (kill padded-M)",
        "snapshot",
        0.871,
        0.472,
        snapshot="level_01_tilem16.py",
        tile_m=16,
        grid_mode="static",
        hardened_required=False,
        note="tile_m=16 pre-fuse snapshot-2 base (the tile_m lever is also "
        "a clean production flag; snapshot keeps the progression "
        "monotone). pre-L4: skewE0 hardened is EXPECTED-FAIL",
    ),
    LevelDef(
        2,
        "level_02_fusequant",
        "fuse 3-pass quant->1",
        "snapshot",
        0.472,
        0.337,
        snapshot="level_02_fusequant.py",
        tile_m=16,
        grid_mode="static",
        hardened_required=False,
        note="pre-L4: skewE0 hardened is EXPECTED-FAIL",
    ),
    LevelDef(
        3,
        "level_03_down_pipeline",
        "down SW-pipeline",
        "snapshot",
        0.337,
        0.333,
        snapshot="level_03_down_pipeline.py",
        tile_m=16,
        grid_mode="static",
        hardened_required=False,
        note="pre-L4: skewE0 hardened is EXPECTED-FAIL",
    ),
    LevelDef(
        4,
        "level_04_mtile_fix",
        "m_tile_base fix (ALWAYS-ON)",
        "snapshot",
        0.333,
        0.331,
        snapshot="level_04_mtile_fix.py",
        tile_m=16,
        grid_mode="static",
        note="correctness fix; gated by the HARDENED skewE0 parity row",
    ),
    LevelDef(
        5,
        "level_05_gateup_interleave",
        "gate+up SW-pipeline + wave-pair MFMA interleave",
        "snapshot",
        0.331,
        0.291,
        snapshot="level_05_gateup_interleave.py",
        tile_m=16,
        grid_mode="static",
    ),
    LevelDef(
        6,
        "level_06_drain_hoist",
        "epilogue drain hoist",
        "sub-diff(L5)",
        0.291,
        0.280,
        snapshot="level_05_gateup_interleave.py",
        tile_m=16,
        grid_mode="static",
        note="shares md5 with L5; SortedWeights hoist folded into L5 snap",
    ),
    LevelDef(
        7,
        "level_07_k128",
        "K=128 hero atom",
        "snapshot",
        0.280,
        0.182,
        snapshot="level_07_k128.py",
        tile_m=16,
        grid_mode="snapshot_native",
        note="K128 pre-DTLA snapshot (the production K128+no-DTLA flag "
        "combo does not compile; documented snapshot fallback in "
        "docs/REPRO_PLAN Sec.2/Sec.4). This curated snapshot carries its "
        "own persistent-XCD ABI (num_cta/total_tiles/grid_x), so it "
        "launches on its native grid (de-padded buffers)",
    ),
    LevelDef(
        8,
        "level_08_dtla",
        "direct-to-LDS gate+up",
        "flag",
        0.182,
        0.161,
        gate_up_k=128,
        down_k=128,
        use_dtla=True,
        sched_cadence="none",
        grid_mode="static",
        note="flag use_dtla False->True on production kernel; iglp OFF "
        "(sched_cadence=none) so the L8->L9 iglp delta is isolated",
    ),
    LevelDef(
        9,
        "level_09_iglp",
        "iglp_opt(1) cadence",
        "flag",
        0.161,
        0.157,
        sched_cadence="iglp1",
        grid_mode="static",
        note="flag sched_cadence none->iglp1 on production kernel",
    ),
    LevelDef(
        10,
        "level_10_active_grid",
        "active de-padded grid",
        "flag",
        0.152,
        0.131,
        sched_cadence="iglp1",
        grid_mode="active",
        note="harness-side active grid.y = sum_e ceil(count_e/tile_m)",
    ),
    LevelDef(
        11,
        "level_11_persistent",
        "persistent kernel + XCD (NEUTRAL)",
        "flag",
        0.131,
        0.131,
        sched_cadence="iglp1",
        grid_mode="persistent",
        persistent=True,
        note="documented OFF-by-default; correct but neutral at this grid",
    ),
]


# ---------------------------------------------------------------------------
# Module + launcher materialization (flag-config OR snapshot-by-path)
# ---------------------------------------------------------------------------


def _materialize_module(lvl: LevelDef):
    """Return ``(module, spec)`` for a level.

    flag levels build from the production module; snapshot levels load the
    curated ``levels/level_NN_*.py`` by path (production file untouched).
    """
    if lvl.mechanism in ("snapshot", "sub-diff(L5)"):
        path = LEVELS_DIR / lvl.snapshot
        # Use a name under a REAL package (rocke.instances.common.*) so the
        # snapshot's relative imports (`from ...core.ir import ...`) resolve.
        # Suffix with the level idx so L5 and L6 (same snapshot file) load as
        # distinct modules.
        mod = load_level_module(
            path,
            mod_name=f"rocke.instances.common._repro_level_{lvl.idx}_{path.stem}",
        )
        # Snapshot specs are self-contained; only tile_m is a knob they all carry.
        spec = mod.FusedMegaKernelSpecFp8(
            name=f"rocke_fused_moe_mega_fp8_L{lvl.idx}", tile_m=lvl.tile_m
        )
        return mod, spec
    # flag mechanism -> production module.
    spec = PROD.FusedMegaKernelSpecFp8(
        name=f"rocke_fused_moe_mega_fp8_L{lvl.idx}",
        tile_m=lvl.tile_m,
        gate_up_k=lvl.gate_up_k,
        down_k=lvl.down_k,
        use_dtla=lvl.use_dtla,
        sched_cadence=lvl.sched_cadence,
    )
    return PROD, spec


def _make_launcher(mod, spec, *, persistent: bool):
    # Snapshot builders predate the ``persistent`` kwarg; only the production
    # builder accepts it. Pass it only when actually requesting the persistent
    # ABI (L11) -- which is always a production (flag) level.
    if persistent:
        kd = mod.build_moe_fused_mega_gemm_fp8(spec, arch="gfx950", persistent=True)
    else:
        kd = mod.build_moe_fused_mega_gemm_fp8(spec, arch="gfx950")
    art = compile_kernel(kd, arch="gfx950", capture_ir_text=False)
    try:
        sig = mod.moe_fused_mega_fp8_signature(spec, persistent=persistent)
    except TypeError:
        sig = mod.moe_fused_mega_fp8_signature(spec)
    return KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=sig,
        cache_key=("moe_fused_mega_fp8_repro", spec.kernel_name(), persistent),
    ), len(art.hsaco)


# ---------------------------------------------------------------------------
# Grid construction (static / active / persistent)
# ---------------------------------------------------------------------------


def _pad_to_static(padded: dict, s: Scenario) -> dict:
    """Extend the de-padded layout to the PRE-L10 static (one-block-per-expert).

    ``build_static_padded_inputs`` packs ONLY active-expert blocks to the front
    (``num_m_blocks = sum_e ceil(count_e/tile_m)`` -> 2 at T1). The PRE-L10
    launch paid for one block PER expert regardless of occupancy -> grid.y = E
    (the documented "T1 8 blocks" state). We reproduce that by APPENDING
    ``E - num_active_blocks_worth`` empty padding blocks (BlockExpertIds=-1,
    token-id=-1) so grid.y = E and the launch is in-bounds. Active experts'
    rows/scales are untouched (their ``expert_base`` is unchanged) so parity is
    identical; the extra blocks are skipped by the kernel's -1 sentinel.

    Returns a NEW dict (shallow copy) with grown ``block_expert_ids`` /
    ``sorted_token_ids_padded`` / ``sorted_weights_padded`` and the buffers
    re-sized; ``A_q``/``AScale`` are grown by the caller to match total_padded.
    """
    import torch as _t

    tile_m = padded["tile_m"]
    E = s.experts
    cur_blocks = padded["num_m_blocks"]
    static_blocks = max(cur_blocks, E)
    if static_blocks == cur_blocks:
        return padded  # already >= E blocks (T8 etc.)
    extra = static_blocks - cur_blocks
    dev = padded["block_expert_ids"].device
    new_total = static_blocks * tile_m

    bei = _t.full((static_blocks,), -1, dtype=_t.int32, device=dev)
    bei[:cur_blocks] = padded["block_expert_ids"]
    sti = _t.full((new_total,), -1, dtype=_t.int32, device=dev)
    sti[: padded["total_padded"]] = padded["sorted_token_ids_padded"]
    sw = _t.zeros(new_total, dtype=_t.float32, device=dev)
    sw[: padded["total_padded"]] = padded["sorted_weights_padded"]

    out = dict(padded)
    out["block_expert_ids"] = bei
    out["sorted_token_ids_padded"] = sti
    out["sorted_weights_padded"] = sw
    out["num_m_blocks"] = static_blocks
    out["total_padded"] = new_total
    out["_extra_pad_rows"] = extra * tile_m
    return out


def _grid_for(mod, lvl: LevelDef, s: Scenario, spec, padded):
    """Return ``(grid, persistent_scalars_or_None)``.

    static  -> (grid.x, E, 1)             pre-L10 padded launch (padded buffers)
    active  -> (grid.x, num_m_blocks, 1)  de-padded (L10)
    persistent -> ((P,1,1), grid_x, total_work, P)  (L11)
    """
    inter = s.intermediate
    nmb = padded["num_m_blocks"]
    if lvl.grid_mode == "persistent":
        # Production persistent ABI (L11): (P,1,1) + grid_x/total_work/P.
        (grid, grid_x, total_work, P) = mod.moe_fused_mega_fp8_persistent_grid(
            nmb, inter, spec
        )
        return grid, {"grid_x": grid_x, "total_work": total_work, "P": P}
    if lvl.grid_mode == "snapshot_native":
        # The snapshot owns its grid + ABI scalars (e.g. L7's persistent-XCD
        # num_cta/total_tiles/grid_x). Launch on its native grid.
        grid = mod.moe_fused_mega_fp8_grid(nmb, inter, spec)
        num_cta, total_tiles, grid_x = mod.moe_fused_mega_fp8_persistent_params(
            nmb, inter, spec
        )
        return grid, {"num_cta": num_cta, "total_tiles": total_tiles, "grid_x": grid_x}
    # static + active both launch (grid.x, num_m_blocks); for static the caller
    # has already grown ``padded`` to E blocks via ``_pad_to_static``.
    return mod.moe_fused_mega_fp8_grid(nmb, inter, spec), None


# ---------------------------------------------------------------------------
# Value packing (shared with parity_fp8 / perf_fp8) + run
# ---------------------------------------------------------------------------


def _mega_values(fp8in, s, padded, A_q, AScale, Y_f32, persist_scalars):
    H = s.hidden
    inter = s.intermediate
    nHb = fp8in["nHb"]
    nIb = fp8in["nIb"]
    v = {
        "A": A_q,
        "WGate": fp8in["Wg_q"],
        "WUp": fp8in["Wu_q"],
        "WDown": fp8in["Wd_q"],
        "AScale": AScale,
        "WGateScale": fp8in["gate_scale"],
        "WUpScale": fp8in["up_scale"],
        "WDownScale": fp8in["down_scale"],
        "SortedTokenIds": padded["sorted_token_ids_padded"],
        "SortedWeights": padded["sorted_weights_padded"],
        "BlockExpertIds": padded["block_expert_ids"],
        "Y": Y_f32,
        "M": padded["total_padded"],
        "N": inter,
        "K": H,
        "H_out": H,
        "stride_a": H,
        "stride_b_gate": inter * H,
        "stride_b_up": inter * H,
        "stride_b_down": H * inter,
        "stride_a_scale": nHb,
        "stride_gate_scale": nIb,
        "stride_up_scale": nIb,
        "stride_down_scale": nHb,
        "stride_gate_scale_e": nHb * nIb,
        "stride_up_scale_e": nHb * nIb,
        "stride_down_scale_e": nIb * nHb,
        "slot_size": padded["slot_size"],
        "tokens": s.tokens,
    }
    if persist_scalars:
        v.update(persist_scalars)
    return v


def _prepare(mod, lvl, s, spec, launcher):
    """Build padded inputs ONCE; return packing closure inputs.

    The torch reference consumes the DE-PADDED ``padded`` + ``A_q``/``AScale``
    (``padded_ref``) so parity is computed on the active layout regardless of
    grid mode. For the static grid the kernel launch consumes a buffer-grown
    copy (extra empty -1 blocks) so grid.y = E is in-bounds.
    """
    fp8in = make_fp8_inputs(s)
    device = fp8in["inputs"].X.device
    padded = build_static_padded_inputs(fp8in["inputs"], s, tile_m=spec.tile_m)
    A_q, AScale = _build_padded_activation(fp8in, padded, s)
    Y_f32 = torch.zeros(s.tokens, s.hidden, dtype=torch.float32, device=device)

    padded_ref = padded  # for the reference (de-padded, active layout)
    if lvl.grid_mode == "static":
        padded = _pad_to_static(padded, s)
        extra_rows = padded.get("_extra_pad_rows", 0)
        if extra_rows:
            nHb = fp8in["nHb"]
            A_pad = torch.zeros(extra_rows, s.hidden, dtype=A_q.dtype, device=device)
            A_q = torch.cat([A_q, A_pad], dim=0)
            AScale_pad = torch.full(
                (extra_rows, nHb), 1e-6 / 448.0, dtype=torch.float32, device=device
            )
            AScale = torch.cat([AScale, AScale_pad], dim=0)

    grid, persist = _grid_for(mod, lvl, s, spec, padded)
    block = (spec.block_size, 1, 1)
    cfg = LaunchConfig(stream=0, grid=grid, block=block)
    return fp8in, padded, padded_ref, A_q, AScale, Y_f32, cfg, grid, persist


def _run_parity(mod, lvl, s, spec, launcher) -> tuple[float, float, str]:
    """Hardened parity for one scenario: returns (max_abs, rel, status)."""
    fp8in, padded, padded_ref, A_q, AScale, Y_f32, cfg, grid, persist = _prepare(
        mod, lvl, s, spec, launcher
    )
    _isolate_lane()
    Y_f32.zero_()
    launcher(_mega_values(fp8in, s, padded, A_q, AScale, Y_f32, persist), config=cfg)
    torch.cuda.synchronize()
    Y_kernel = Y_f32.clone()
    Y_ref = torch_fused_moe_fp8_reference(
        fp8in, padded_ref, A_q, AScale, s, tile_m=spec.tile_m
    )
    mx, mn, rl = _compare(Y_kernel, Y_ref)
    _isolate_lane()
    status = "PASS" if rl < TOL else "FAIL"
    return mx, rl, status


def _time_level(mod, lvl, s, spec, launcher) -> tuple[float, float, tuple]:
    """Launch-only best/median ms (warm best-of-N) for one scenario."""
    fp8in, padded, padded_ref, A_q, AScale, Y_f32, cfg, grid, persist = _prepare(
        mod, lvl, s, spec, launcher
    )
    vals = _mega_values(fp8in, s, padded, A_q, AScale, Y_f32, persist)

    def call():
        Y_f32.zero_()
        launcher(vals, config=cfg)
        _ = Y_f32.to(torch.bfloat16)

    _isolate_lane()
    runs = [
        time_callable_ms(call, warmup=WARMUP, attempts=ATTEMPTS)
        for _ in range(OUTER_RUNS)
    ]
    _isolate_lane()
    return min(runs), statistics.median(runs), grid


# ---------------------------------------------------------------------------
# Per-level driver
# ---------------------------------------------------------------------------


@dataclass
class LevelResult:
    idx: int
    name: str
    lever: str
    mechanism: str
    t1_ms: Optional[float]
    t8_ms: Optional[float]
    parity: str
    hardened: str  # skewE0 status: PASS / FAIL / EXPECTED-FAIL
    before: float
    after: float
    grid: str
    hsaco_bytes: Optional[int]
    note: str


def run_level(lvl: LevelDef, *, do_perf: bool) -> LevelResult:
    print(f"\n=== L{lvl.idx} {lvl.lever}  [{lvl.mechanism}] ===")
    mod, spec = _materialize_module(lvl)
    try:
        launcher, nbytes = _make_launcher(mod, spec, persistent=lvl.persistent)
    except Exception as exc:  # noqa: BLE001
        print(f"  BUILD FAILED: {exc!r}")
        return LevelResult(
            lvl.idx,
            lvl.name,
            lvl.lever,
            lvl.mechanism,
            None,
            None,
            "BUILD_FAIL",
            "-",
            lvl.before,
            lvl.after,
            lvl.grid_mode,
            None,
            lvl.note,
        )
    print(f"  built {spec.kernel_name()}  hsaco={nbytes}B  grid={lvl.grid_mode}")

    # --- HARDENED parity: T1, T8, skewed-E0 (never loosened) ---
    s1, s8, sh = _scn_t1(), _scn_t8(), _scn_hard()
    # The skewed-E0 hardened case must run on a tile_m=32 spec (mfmas_m_down=2).
    if lvl.mechanism in ("snapshot", "sub-diff(L5)"):
        spec_hard = mod.FusedMegaKernelSpecFp8(
            name=f"{spec.kernel_name()}_m32", tile_m=32
        )
    else:
        spec_hard = PROD.FusedMegaKernelSpecFp8(
            name=f"{spec.kernel_name()}_m32",
            tile_m=32,
            gate_up_k=lvl.gate_up_k,
            down_k=lvl.down_k,
            use_dtla=lvl.use_dtla,
            sched_cadence=lvl.sched_cadence,
        )
    launcher_hard, _ = _make_launcher(mod, spec_hard, persistent=lvl.persistent)

    mx1, rl1, st1 = _run_parity(mod, lvl, s1, spec, launcher)
    print(f"  parity T1   rel={rl1:.3e} max={mx1:.3e} {st1}")
    mx8, rl8, st8 = _run_parity(mod, lvl, s8, spec, launcher)
    print(f"  parity T8   rel={rl8:.3e} max={mx8:.3e} {st8}")
    mxh, rlh, sth = _run_parity(mod, lvl, sh, spec_hard, launcher_hard)
    if lvl.hardened_required:
        print(f"  parity skewE0 rel={rlh:.3e} max={mxh:.3e} {sth}  (required)")
        parity = (
            "PASS" if (st1 == "PASS" and st8 == "PASS" and sth == "PASS") else "FAIL"
        )
    else:
        # pre-L4: the >tile_m m_tile_base bug is live, so skewE0 is EXPECTED to
        # fail. It is reported as EXPECTED-FAIL (the L4 row proves the fix), and
        # does NOT gate this level's perf -- canonical T1/T8 parity does.
        exp = "EXPECTED-FAIL" if sth == "FAIL" else "PASS(!)"
        print(
            f"  parity skewE0 rel={rlh:.3e} max={mxh:.3e} {sth}  "
            f"({exp}; pre-L4 fix, not a gate)"
        )
        parity = "PASS" if (st1 == "PASS" and st8 == "PASS") else "FAIL"

    t1 = t8 = None
    grid_str = lvl.grid_mode
    if do_perf and parity == "PASS":
        b1, m1, g1 = _time_level(mod, lvl, s1, spec, launcher)
        b8, m8, g8 = _time_level(mod, lvl, s8, spec, launcher)
        t1, t8 = b1, b8
        grid_str = f"{lvl.grid_mode}{g1}"
        print(f"  perf T1 best={b1:.6f} med={m1:.6f}  T8 best={b8:.6f} med={m8:.6f}")
    elif parity != "PASS":
        print("  perf SKIPPED (parity FAIL)")

    if lvl.hardened_required:
        hardened = sth
    else:
        hardened = "EXPECTED-FAIL" if sth == "FAIL" else "PASS"

    return LevelResult(
        lvl.idx,
        lvl.name,
        lvl.lever,
        lvl.mechanism,
        t1,
        t8,
        parity,
        hardened,
        lvl.before,
        lvl.after,
        grid_str,
        nbytes,
        lvl.note,
    )


# ---------------------------------------------------------------------------
# Table + markdown emit
# ---------------------------------------------------------------------------


def _parse_levels_arg(arg: str) -> list[int]:
    if arg.strip().lower() == "all":
        return list(range(len(LEVELS)))
    out: list[int] = []
    for tok in arg.split(","):
        tok = tok.strip()
        if "-" in tok:
            a, b = tok.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif tok:
            out.append(int(tok))
    return sorted(set(out))


def _fmt(v, spec="{:.4f}"):
    return spec.format(v) if v is not None else "-"


def render_table(results: list[LevelResult]) -> str:
    """Numeric-only per-level ledger: L, lever, mechanism, T1, T8, parity, skewE0.

    No ratio-to-asm column and no external benchmark -- the kernel's own numbers.
    """
    lines = []
    lines.append("| L | lever | mech | T1 ms | T8 ms | parity | skewE0 |")
    lines.append("|--:|---|---|--:|--:|:--:|:--:|")
    for r in results:
        lines.append(
            f"| {r.idx} | {r.lever} | {r.mechanism} | {_fmt(r.t1_ms)} | "
            f"{_fmt(r.t8_ms)} | {r.parity} | {r.hardened} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--levels", default="all")
    ap.add_argument("--perf", dest="perf", action="store_true", default=True)
    ap.add_argument("--no-perf", dest="perf", action="store_false")
    ap.add_argument(
        "--out",
        default=None,
        help="optional path to save the markdown ledger; "
        "by default the table is only printed (it is also "
        "embedded in README.md)",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA / ROCm not available; aborting.", file=sys.stderr)
        return 1
    dev = torch.cuda.get_device_name(0)
    print(f"device: {dev}")

    sel = _parse_levels_arg(args.levels)
    chosen = [LEVELS[i] for i in sel if 0 <= i < len(LEVELS)]

    if args.perf:
        _warm_gpu()

    results: list[LevelResult] = []
    for lvl in chosen:
        results.append(run_level(lvl, do_perf=args.perf))

    table = render_table(results)
    print("\n" + table)

    # --- relative-gain / ordering check ---
    perf_rows = [r for r in results if r.t1_ms is not None]
    final = next((r for r in results if r.idx == 11), None)
    final9 = next((r for r in results if r.idx == 9), None)
    ordering_ok = True
    ord_notes = []
    # The documented chain monotonically improves T1 through L9 (L10/L11 are
    # grid-mode levers measured on the production kernel and may be ~neutral).
    snap_seq = [r for r in perf_rows if r.idx in (0, 2, 3, 5, 7, 8, 9)]
    for a, b in zip(snap_seq, snap_seq[1:]):
        if b.t1_ms > a.t1_ms * 1.15:  # allow thermal jitter, demand the trend
            ordering_ok = False
            ord_notes.append(
                f"L{a.idx}->L{b.idx} regressed {a.t1_ms:.4f}->{b.t1_ms:.4f}"
            )

    final_t1 = (
        final.t1_ms if final and final.t1_ms else (final9.t1_ms if final9 else None)
    )

    if args.out:
        md = _render_md(dev, results, table, ordering_ok, ord_notes, final_t1)
        Path(args.out).write_text(md)
        print(f"\nwrote {args.out}")
    print(f"ordering_ok={ordering_ok} {('; '.join(ord_notes)) if ord_notes else ''}")
    return 0


def _render_md(dev, results, table, ordering_ok, ord_notes, final_t1) -> str:
    parity_pass = sum(1 for r in results if r.parity == "PASS")
    lines = [
        "# REPRO_RESULTS — regenerated per-level ledger",
        "",
        f"Device: `{dev}`.  Generated by `reproduce_levels.py` "
        "(launch-only, warm best-of-N, GPU serial; kernel's own numeric perf).",
        "",
        f"- levels run: **{len(results)}**",
        f"- parity PASS: **{parity_pass}/{len(results)}**",
        f"- reproduced FINAL T1: **{_fmt(final_t1)}** ms (documented 0.124–0.157 band)",
        f"- ordering/relative-gain trend holds: **{ordering_ok}**"
        + (f"  ({'; '.join(ord_notes)})" if ord_notes else ""),
        "",
        "## Reproduced step table",
        "",
        table,
        "",
        "## Per-level provenance / notes",
        "",
        "| L | mechanism | source | grid | hsaco B | note |",
        "|--:|---|---|---|--:|---|",
    ]
    for r in results:
        src = (
            r.name
            if r.mechanism in ("snapshot", "sub-diff(L5)")
            else "production+flags"
        )
        lines.append(
            f"| {r.idx} | {r.mechanism} | {src} | {r.grid} | "
            f"{_fmt(r.hsaco_bytes, '{:d}')} | {r.note} |"
        )
    lines += [
        "",
        "## Method (hard constraints)",
        "",
        "- Launch-only timing: timed closure = `Y.zero_(); launch; Y.to(bf16)`; "
        "static padded inputs built ONCE outside the timed region.",
        f"- Warm >= {WARMUP}, attempts {ATTEMPTS}, best-of-{OUTER_RUNS} outer "
        "runs; `_isolate_lane()` between phases; GPU serial + warm.",
        "- HARDENED parity: T1 / T8 / skewed-E0 (tile_m=32), "
        f"gate rel < {TOL} — never loosened.",
        "- Numeric perf only: the kernel's own launch-only T1/T8 ms (no "
        "external hand-tuned-asm comparison, no `/tmp` dependency).",
        "- Snapshot levels load `levels/level_NN_*.py` by file path "
        "(production kernel untouched); flag levels build the production "
        "`moe_fused_mega_fp8.py` with spec/build flags "
        "(`tile_m`, `gate_up_k`, `down_k`, `use_dtla`, `sched_cadence`, "
        "`persistent`) + harness grid mode (static/active/persistent).",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
