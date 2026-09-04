# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dispatcher-driven fused-MoE benchmark (fp8 e4m3 block scale).

Every kernel this harness runs comes from :func:`dispatch_moe_plan`; nothing
here hand-builds a spec. That is the point: the token count is a *selection*
knob for this family, so sweeping it exercises dispatch, candidate
registration, the builders behind them and the two-launch plan protocol end to
end. A routing regression -- a band boundary that moves, a candidate that stops
being registered, a stage-2 spec that drifts out of agreement with its stage 1
-- shows up here as a changed kernel name, a changed launch count, or a
correctness failure, rather than as a number that is quietly 30% off.

The default token set spans everything the fp8 MoE family routes differently:

* ``T=1``   -- the fused band: ONE launch, the whole layer in one kernel;
* ``T=32``  -- the split band: TWO launches (gate/up + requantize, then the
  down GEMM over H_out);
* ``T=64``  -- the same split band, i.e. a band-interior control: it must
  select the same kernels as ``T=32``;
* ``T=512`` -- the next split band, which differs only in ``tile_m``.

Correctness is checked per shape against a numpy f32 model of exactly what the
kernel consumes (the same quantised operands, the same per-tile intermediate
requantisation). That model is O(T) per token, so it is only run over every
token up to ``ORACLE_MAX_TOKENS``; above that a random sample of tokens is
verified exactly instead, and the whole output is still checked for
finiteness. The report says which was used -- ``oracle/all`` or
``oracle/sample``.

Run it (system python, torch-free -- see the note below)::

    export ROCKE=<rocke>/platform/python
    PYTHONPATH=$ROCKE python3 -m rocke.benchmark.moe.fused_mega_fp8_dispatch

Show the routing decisions without touching the GPU::

    PYTHONPATH=$ROCKE python3 -m rocke.benchmark.moe.fused_mega_fp8_dispatch \
        --plan-only

A quick smoke run (short warmup; see ``--warmup``) plus a JSON record::

    PYTHONPATH=$ROCKE python3 -m rocke.benchmark.moe.fused_mega_fp8_dispatch \
        --tokens 1,32 --warmup 20 --iters 10 --check sample \
        --json /tmp/moe_dispatch.json

Pick a GPU with ``HIP_VISIBLE_DEVICES``; the harness uses the default device.

torch-free on purpose: importing torch before the first comgr compile resolves
comgr against torch's bundled LLVM, which changes codegen and can make lowering
pathologically slow. :func:`main` asserts torch is absent, exactly as the
example harness under ``examples/gfx950/fused_mega_moe`` does.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from ...dispatch.families.moe import MoeRequest, dispatch_moe_plan, moe_launch_grid

SCHEMA = "ck.dsl.benchmark.moe.fused_mega_fp8_dispatch/v1"

#: Block-scale group width, on both weight axes and on the activation.
GROUP_K = 128
FP8_MAX = 448.0
AMAX_FLOOR = 1e-6

#: Above this token count the per-token oracle stops being run over every
#: token. It is a numpy f32 GEMM chain per expert block and grows with T, so at
#: prefill it costs minutes against a measurement of microseconds.
ORACLE_MAX_TOKENS = 64

#: Qwen3-30B-A3B MoE geometry, and the token counts that span the family's
#: routing: fused band, split band (twice, to pin the band interior), and the
#: wider-tile_m split band.
DEFAULT_TOKENS = (1, 32, 64, 512)


# ---------------------------------------------------------------------------
# fp8 e4m3 (OCP "fn" flavour: no inf, 0x7f/0xff are NaN, max magnitude 448)
# ---------------------------------------------------------------------------
#
# Encoded against a 256-entry table rather than through ``ml_dtypes``, which
# rocke does not depend on.


def _e4m3_value_table() -> np.ndarray:
    codes = np.arange(256, dtype=np.uint16)
    sign, exp, man = (codes >> 7) & 1, (codes >> 3) & 0xF, codes & 0x7
    sub = (man / 8.0) * (2.0**-6)
    nrm = (1.0 + man / 8.0) * np.power(2.0, exp.astype(np.float64) - 7.0)
    val = np.where(exp == 0, sub, nrm)
    val = np.where(sign == 1, -val, val)
    val[(exp == 0xF) & (man == 0x7)] = np.nan
    return val.astype(np.float32)


_E4M3_TABLE = _e4m3_value_table()
_FINITE = np.flatnonzero(~np.isnan(_E4M3_TABLE))
_ORDER = np.argsort(_E4M3_TABLE[_FINITE], kind="stable")
_SORTED_VALS = _E4M3_TABLE[_FINITE][_ORDER].astype(np.float32)
_SORTED_CODES = _FINITE[_ORDER].astype(np.uint8)


def quantize_e4m3(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest, saturating quantisation of f32 -> e4m3 code bytes."""
    xf = np.clip(np.asarray(x, dtype=np.float32), -FP8_MAX, FP8_MAX)
    idx = np.clip(np.searchsorted(_SORTED_VALS, xf), 1, _SORTED_VALS.size - 1)
    lo, hi = _SORTED_VALS[idx - 1], _SORTED_VALS[idx]
    return _SORTED_CODES[np.where((hi - xf) <= (xf - lo), idx, idx - 1)]


def dequantize_e4m3(codes: np.ndarray) -> np.ndarray:
    return _E4M3_TABLE[codes]


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def _block_scale_2d(w: np.ndarray) -> np.ndarray:
    """Per-(128x128)-block amax/448 scale, shape ``[out/128, k/128]``."""
    nob, nkb = w.shape[0] // GROUP_K, w.shape[1] // GROUP_K
    amax = np.abs(w.reshape(nob, GROUP_K, nkb, GROUP_K)).max(axis=(1, 3))
    return (np.maximum(amax, AMAX_FLOOR) / FP8_MAX).astype(np.float32)


def _expand(scale: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(scale, GROUP_K, axis=0), GROUP_K, axis=1)


# ---------------------------------------------------------------------------
# Shapes and records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoeShape:
    """One MoE problem. Only ``num_tokens`` varies across the default set."""

    num_tokens: int
    hidden: int = 2048
    intermediate: int = 768
    num_experts: int = 128
    top_k: int = 8

    def request(self, *, arch: str, dtype: str) -> MoeRequest:
        return MoeRequest(
            num_tokens=self.num_tokens,
            hidden=self.hidden,
            intermediate=self.intermediate,
            num_experts=self.num_experts,
            top_k=self.top_k,
            arch=arch,
            dtype=dtype,
        )

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StageRecord:
    """One dispatched launch: who was selected, and the spec fields that prove
    which route it is.

    ``tile_m``/``coop_b_lds``/``static_inter_scale``/``grid`` are carried rather
    than left to the kernel name because they are what actually distinguishes
    the bands; the name only reflects them by convention.
    """

    candidate: str
    spec_id: str
    kernel_name: str
    spec_hash: str
    compile_key: str
    tile_m: int
    warp_m: int
    warp_n: int
    tile_n_inter: int
    tile_n_down: int
    coop_b_lds: bool
    static_inter_scale: bool
    grid: tuple[int, int, int]
    grid_dispatch: tuple[int, int, int]
    block: tuple[int, int, int]
    build_s: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ShapeRecord:
    shape: MoeShape
    route: str
    launches: int
    stages: tuple[StageRecord, ...] = ()
    explanation: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    calibrated: bool = False
    us: Optional[float] = None
    check: str = "none"
    checked_tokens: int = 0
    rel: Optional[float] = None
    verdict: str = "n/a"
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.verdict in ("pass", "n/a")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["shape"] = self.shape.as_dict()
        d["stages"] = [s.as_dict() for s in self.stages]
        return d


# ---------------------------------------------------------------------------
# Host data
# ---------------------------------------------------------------------------


class ExpertWeights:
    """Quantised expert weights: a function of ``(E, I, H)`` and the seed only.

    Built once per run and shared by every token count, because this is the
    expensive part. Each expert is generated and quantised on its own so the
    f32 master never exists for the whole stack -- at 128 experts that is the
    difference between a few hundred MB and several GB of host memory.
    """

    def __init__(self, shape: MoeShape, *, seed: int) -> None:
        E, inter, hidden = shape.num_experts, shape.intermediate, shape.hidden
        self.nHb = hidden // GROUP_K
        self.nIb = inter // GROUP_K
        rng = np.random.default_rng(seed)

        self.Wg_q = np.empty((E, inter, hidden), dtype=np.uint8)
        self.Wu_q = np.empty((E, inter, hidden), dtype=np.uint8)
        self.Wd_q = np.empty((E, hidden, inter), dtype=np.uint8)
        self.gate_scale = np.empty((E, self.nHb, self.nIb), dtype=np.float32)
        self.up_scale = np.empty((E, self.nHb, self.nIb), dtype=np.float32)
        self.down_scale = np.empty((E, self.nIb, self.nHb), dtype=np.float32)

        for e in range(E):
            wg = (rng.standard_normal((inter, hidden)) * 0.05).astype(np.float32)
            wu = (rng.standard_normal((inter, hidden)) * 0.05).astype(np.float32)
            wd = (rng.standard_normal((hidden, inter)) * 0.05).astype(np.float32)
            sg, su, sd = (
                _block_scale_2d(wg),
                _block_scale_2d(wu),
                _block_scale_2d(wd),
            )
            self.gate_scale[e], self.up_scale[e], self.down_scale[e] = (
                sg.T,
                su.T,
                sd.T,
            )
            self.Wg_q[e] = quantize_e4m3(wg / _expand(sg))
            self.Wu_q[e] = quantize_e4m3(wu / _expand(su))
            self.Wd_q[e] = quantize_e4m3(wd / _expand(sd))

    def expert_f32(self, e: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dequantised ``(Wg, Wu, Wd)`` for one expert, for the oracle."""
        return (
            dequantize_e4m3(self.Wg_q[e]) * _expand(self.gate_scale[e].T),
            dequantize_e4m3(self.Wu_q[e]) * _expand(self.up_scale[e].T),
            dequantize_e4m3(self.Wd_q[e]) * _expand(self.down_scale[e].T),
        )


class Routing:
    """Activations and top-k routing for one token count."""

    def __init__(self, shape: MoeShape, *, seed: int) -> None:
        rng = np.random.default_rng(seed + shape.num_tokens)
        T, E = shape.num_tokens, shape.num_experts
        self.X = (rng.standard_normal((T, shape.hidden)) * 0.1).astype(np.float32)
        logits = rng.standard_normal((T, E)).astype(np.float32)
        ids = np.argsort(-logits, axis=-1, kind="stable")[:, : shape.top_k]
        vals = np.take_along_axis(logits, ids, axis=-1)
        ex = np.exp(vals - vals.max(axis=-1, keepdims=True))
        self.topk_ids = ids.astype(np.int32)
        self.topk_weights = (ex / ex.sum(axis=-1, keepdims=True)).astype(np.float32)


class TokenLayout:
    """Sorted/padded activation layout for one ``tile_m``.

    ``tile_m`` comes from the dispatched spec, so this is rebuilt whenever the
    selected band changes it -- the intermediate's requantisation amax is taken
    over a ``tile_m``-row tile, which makes the layout part of the kernel's
    numerics rather than a free host-side choice.
    """

    def __init__(self, routing: Routing, shape: MoeShape, tile_m: int) -> None:
        self.shape = shape
        self.tile_m = tile_m
        self.routing = routing
        E, T = shape.num_experts, shape.num_tokens

        self.counts = [int((routing.topk_ids == e).sum()) for e in range(E)]
        self.blocks_per_expert = [(c + tile_m - 1) // tile_m for c in self.counts]
        self.num_m_blocks = max(sum(self.blocks_per_expert), 1)
        self.total_padded = self.num_m_blocks * tile_m

        self.sorted_token_ids = np.full(self.total_padded, -1, dtype=np.int32)
        self.sorted_weights = np.zeros(self.total_padded, dtype=np.float32)
        self.block_expert_ids = np.full(self.num_m_blocks, -1, dtype=np.int32)
        self.expert_base = [-1] * E
        # Padded row that carries token ``t``'s ``slot``-th expert, so a
        # per-token check can find the exact tile the kernel put it in.
        self.slot_row = np.full((T, shape.top_k), -1, dtype=np.int64)

        blk = 0
        for e in range(E):
            be = self.blocks_per_expert[e]
            if be == 0:
                continue
            tok, slot = np.nonzero(routing.topk_ids == e)
            base = blk * tile_m
            self.expert_base[e] = base
            self.block_expert_ids[blk : blk + be] = e
            self.sorted_token_ids[base : base + tok.size] = tok.astype(np.int32)
            self.sorted_weights[base : base + tok.size] = routing.topk_weights[
                tok, slot
            ]
            self.slot_row[tok, slot] = base + np.arange(tok.size)
            blk += be

        self._build_activation()

    def _build_activation(self) -> None:
        """One activation scale per expert per K-group, broadcast over its
        blocks -- including onto the padding rows, which is what the kernel's
        per-lane dequant fold assumes.
        """
        s, tm = self.shape, self.tile_m
        nHb = s.hidden // GROUP_K
        self.A_q = np.zeros((self.total_padded, s.hidden), dtype=np.uint8)
        self.AScale = np.full(
            (self.total_padded, nHb), AMAX_FLOOR / FP8_MAX, dtype=np.float32
        )
        for e in range(s.num_experts):
            if self.blocks_per_expert[e] == 0:
                continue
            tok, _ = np.nonzero(self.routing.topk_ids == e)
            base = self.expert_base[e]
            sub = self.routing.X[tok]
            amax = np.maximum(
                np.abs(sub.reshape(tok.size, nHb, GROUP_K)).max(axis=(0, 2)),
                AMAX_FLOOR,
            )
            scale = (amax / FP8_MAX).astype(np.float32)
            self.A_q[base : base + tok.size] = quantize_e4m3(
                sub / np.repeat(scale, GROUP_K)[None, :]
            )
            self.AScale[base : base + self.blocks_per_expert[e] * tm] = scale

    def active_experts(self) -> int:
        return sum(1 for c in self.counts if c > 0)


def oracle_tokens(
    weights: ExpertWeights,
    layout: TokenLayout,
    tokens: Sequence[int],
    *,
    hidden_group_k: int = GROUP_K,
) -> np.ndarray:
    """f32 model of the kernel, evaluated for ``tokens`` only.

    Consumes exactly the operands the kernel consumes (the quantised
    activation, the quantised weights, the same block scales), so a mismatch is
    the kernel's arithmetic and not a quantisation difference.

    Restricting it to a token subset is what makes a large-T check affordable:
    a token's output depends on the ``tile_m``-row tiles it sits in, one per
    routed expert, because the intermediate's fp8 scale is a per-tile amax. So
    the work is ``len(tokens) * top_k`` tiles instead of every tile in the
    problem -- and it stays an exact check on the tokens it covers, not a
    self-consistency comparison against another kernel.
    """
    s = layout.shape
    tm = layout.tile_m
    index = {int(t): i for i, t in enumerate(tokens)}
    out = np.zeros((len(index), s.hidden), dtype=np.float32)

    # (expert, block) -> the (padded row, token, slot) triples wanted from it.
    tiles: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for t in index:
        for slot in range(s.top_k):
            e = int(layout.routing.topk_ids[t, slot])
            row = int(layout.slot_row[t, slot])
            blk = (row - layout.expert_base[e]) // tm
            tiles.setdefault((e, blk), []).append((row, t, slot))

    nhb = s.intermediate // hidden_group_k
    for e in sorted({k[0] for k in tiles}):
        Wg, Wu, Wd = weights.expert_f32(e)
        base_e = layout.expert_base[e]
        end_e = base_e + layout.counts[e]
        for (expert, blk), wants in tiles.items():
            if expert != e:
                continue
            r0 = base_e + blk * tm
            r1 = min(r0 + tm, end_e)
            Xdq = dequantize_e4m3(layout.A_q[r0:r1]) * np.repeat(
                layout.AScale[r0:r1], GROUP_K, axis=1
            )
            hidden = _silu(Xdq @ Wg.T) * (Xdq @ Wu.T)
            amax = np.maximum(
                np.abs(hidden.reshape(r1 - r0, nhb, hidden_group_k)).max(axis=(0, 2)),
                AMAX_FLOOR,
            )
            hs = np.repeat((amax / FP8_MAX).astype(np.float32), hidden_group_k)[None, :]
            requantized = dequantize_e4m3(quantize_e4m3(hidden / hs)) * hs
            tile_out = requantized @ Wd.T
            for row, t, slot in wants:
                weight = layout.routing.topk_weights[t, slot]
                out[index[t]] += weight * tile_out[row - r0]
    return out


# ---------------------------------------------------------------------------
# Device side (rocke's ctypes-only HIP runtime; no torch)
# ---------------------------------------------------------------------------


class DeviceWeights:
    """The expert weight stack on device, uploaded once per swizzle layout.

    ~600 MB at the default geometry, and independent of the token count, so it
    is hoisted out of the per-shape work. The swizzle flags change how the
    kernel *addresses* these buffers rather than what they contain, so feeding
    a swizzled kernel row-major weights is silently wrong rather than an error
    -- hence they are part of this object's identity and asserted against every
    spec launched over it.
    """

    def __init__(
        self, weights: ExpertWeights, *, swizzle_gu: bool, swizzle_down: bool
    ) -> None:
        from ...instances.common.moe_fused_mega_fp8 import swizzle_b_fp8_weights

        self.swizzle_gu = swizzle_gu
        self.swizzle_down = swizzle_down
        self.nHb, self.nIb = weights.nHb, weights.nIb
        self._keep: list = []

        def upload_w(arr: np.ndarray, swizzle: bool):
            return _upload(swizzle_b_fp8_weights(arr) if swizzle else arr, self._keep)

        self.values = {
            "WGate": upload_w(weights.Wg_q, swizzle_gu),
            "WUp": upload_w(weights.Wu_q, swizzle_gu),
            "WDown": upload_w(weights.Wd_q, swizzle_down),
            "WGateScale": _upload(weights.gate_scale, self._keep),
            "WUpScale": _upload(weights.up_scale, self._keep),
            "WDownScale": _upload(weights.down_scale, self._keep),
        }


def _upload(arr: np.ndarray, keep: list):
    from ...runtime.host_buffers import as_u8_buffer
    from ...runtime.launcher import DeviceMem

    arr = np.ascontiguousarray(arr)
    mem = DeviceMem(arr.nbytes)
    _runtime().memcpy_h2d(mem.ptr(), as_u8_buffer(arr), arr.nbytes)
    keep += [arr, mem]
    return mem


#: One HIP runtime for the whole process. ``Runtime`` owns per-stream lifetime
#: buckets, so a fresh instance per upload would scatter that bookkeeping.
_RUNTIME = None


def _runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from ...runtime.hip_module import Runtime

        _RUNTIME = Runtime()
    return _RUNTIME


class DeviceProblem:
    """Per-shape device buffers, bound to an already-uploaded weight stack."""

    def __init__(self, layout: TokenLayout, dev_weights: DeviceWeights) -> None:
        from ...runtime.launcher import DeviceMem

        self.rt = _runtime()
        self._keep: list = []
        s, w = layout.shape, dev_weights
        hidden, inter = s.hidden, s.intermediate

        self.y_shape = (s.num_tokens, hidden)
        self.y_nbytes = s.num_tokens * hidden * 4
        self.Y = DeviceMem(self.y_nbytes)
        # Partial-fusion staging: written by stage 1, read straight back by
        # stage 2, so it is allocated up front rather than between the two
        # launches -- its placement relative to the ~600 MB weight stack is on
        # the critical path and moves the measurement if it lands after it.
        self.Inter = DeviceMem(layout.total_padded * inter)
        self.InterScale = DeviceMem(layout.num_m_blocks * (inter // GROUP_K) * 4)
        self._keep += [self.Y, self.Inter, self.InterScale]

        # A superset of every launch's ABI, keyed by parameter name: each
        # launcher packs only the names its own signature lists.
        self.values = {
            "A": _upload(layout.A_q, self._keep),
            "Inter": self.Inter,
            "InterScale": self.InterScale,
            "AScale": _upload(layout.AScale, self._keep),
            "SortedTokenIds": _upload(layout.sorted_token_ids, self._keep),
            "SortedWeights": _upload(layout.sorted_weights, self._keep),
            "BlockExpertIds": _upload(layout.block_expert_ids, self._keep),
            "Y": self.Y,
            "M": layout.total_padded,
            "N": inter,
            "K": hidden,
            "H_out": hidden,
            "stride_a": hidden,
            "stride_b_gate": inter * hidden,
            "stride_b_up": inter * hidden,
            "stride_b_down": hidden * inter,
            "stride_a_scale": w.nHb,
            "stride_gate_scale": w.nIb,
            "stride_up_scale": w.nIb,
            "stride_down_scale": w.nHb,
            "stride_gate_scale_e": w.nHb * w.nIb,
            "stride_up_scale_e": w.nIb * w.nHb,
            "stride_down_scale_e": w.nIb * w.nHb,
            "slot_size": layout.tile_m,
            "tokens": s.num_tokens,
            **w.values,
        }

    def zero_y(self) -> None:
        self.rt.memset(self.Y.ptr(), 0, self.y_nbytes)

    def read_y(self) -> np.ndarray:
        out = np.zeros(self.y_shape, dtype=np.float32)
        buf = (ctypes.c_uint8 * out.nbytes).from_buffer(out)
        self.rt.memcpy_d2h(buf, self.Y.ptr(), out.nbytes)
        del buf
        return out


# ---------------------------------------------------------------------------
# Dispatch -> build -> launch
# ---------------------------------------------------------------------------


def _stage_record(result, *, grid, build_s: float) -> StageRecord:
    spec = result.spec
    return StageRecord(
        candidate=result.candidate.name,
        spec_id=result.candidate.spec_id,
        kernel_name=spec.kernel_name(),
        spec_hash=result.kernel_id.spec_hash,
        compile_key=result.kernel_id.compile_key,
        tile_m=int(spec.tile_m),
        warp_m=int(spec.warp_m),
        warp_n=int(spec.warp_n),
        tile_n_inter=int(spec.tile_n_inter),
        tile_n_down=int(spec.tile_n_down),
        coop_b_lds=bool(spec.coop_b_lds),
        static_inter_scale=bool(spec.static_inter_scale),
        grid=tuple(int(x) for x in grid),
        grid_dispatch=tuple(int(x) for x in result.grid),
        block=(int(spec.block_size), 1, 1),
        build_s=round(build_s, 3),
    )


def plan_stage_grids(plan, layout: TokenLayout) -> list[tuple[int, int, int]]:
    """The grid each dispatched launch is actually run with.

    ``result.grid`` can only carry a worst-case block count, because the exact
    one needs the routing histogram and a selection cannot wait for one. Once
    the routing exists, :func:`moe_launch_grid` restates the same geometry
    against the real count, which is what gets launched here.

    Launching the bound instead is not an option: the fused kernel guards on
    the ``-1`` empty-block marker, but the split down GEMM does not, so its
    surplus blocks would rebase the weight pointers by ``-1`` experts and read
    from before the buffer.
    """
    return [
        tuple(int(x) for x in moe_launch_grid(result, layout.num_m_blocks))
        for result in plan
    ]


def _dispatched_block_count(result) -> int:
    """The block count baked into ``result.grid``.

    Which axis carries it is the launch's business -- the prologue puts the
    blocks on ``x``, the GEMMs on ``y`` -- so this probes for the axis rather
    than assuming one.
    """
    probe = 1 << 20
    return int(result.grid[moe_launch_grid(result, probe).index(probe)])


def grid_bound_warnings(plan, layout: TokenLayout) -> tuple[str, ...]:
    """Launches whose real block count exceeds what the dispatcher reported.

    This should never fire: the reported count is a bound. It fires if that
    bound ever regresses back into an estimate -- which it once was, spreading
    the slots evenly across the experts and so computing the AVERAGE. Nothing
    on the device is sized from that number here, so the run stays valid, but a
    caller that took ``result.grid`` verbatim would launch fewer blocks than
    the routing has and the tokens in the missing blocks would simply not be
    computed. Silently wrong rather than slow, so it is worth saying out loud.
    """
    return tuple(
        f"{result.candidate.name}: routing needs {layout.num_m_blocks} token "
        f"blocks, dispatched grid bound is {_dispatched_block_count(result)}"
        for result in plan
        if layout.num_m_blocks > _dispatched_block_count(result)
    )


def _compile_cached(result, *, arch: str, cache: dict, spec=None, tag: str = ""):
    """Compile one dispatched selection, keyed on the dispatcher's compile key.

    ``compile_key`` is arch + ABI + spec hash with the problem deliberately
    excluded, so two token counts that land in the same band share one compile
    -- which is exactly the T=32/T=64 pair in the default set.
    """
    from ...helpers.compile import compile_kernel

    key = (result.kernel_id.compile_key, tag)
    if key not in cache:
        spec = result.spec if spec is None else spec
        t = time.time()
        kernel = result.candidate.built(spec, arch)
        cache[key] = (
            compile_kernel(kernel, arch=arch, capture_ir_text=False),
            time.time() - t,
        )
    return cache[key]


def _launch_plan(plan, dev: DeviceProblem, artifacts, grids) -> Callable[[], None]:
    """One callable that issues every launch of the plan, in plan order.

    Each launcher is bound to the signature the dispatcher reported for that
    selection, so the argument ABI comes from the same place the kernel did.
    ``dev.values`` is a superset keyed by parameter name and the launcher packs
    only what its own signature asks for, which is what lets stage 1 and
    stage 2 -- whose ABIs share names but not membership -- take one dict.
    """
    from ...runtime.launcher import KernelLauncher, LaunchConfig

    calls = []
    for result, artifact, grid in zip(plan, artifacts, grids):
        launcher = KernelLauncher(
            hsaco=artifact.hsaco,
            kernel_name=artifact.kernel_name,
            signature=result.signature,
            cache_key=("moe_bench", result.candidate.name, result.spec.kernel_name()),
        )
        config = LaunchConfig(stream=0, grid=grid, block=result.block)
        calls.append((launcher, config))

    values = dev.values

    def launch() -> None:
        for launcher, config in calls:
            launcher(values, config=config)

    return launch


def _calibrate_inter_scale(plan, dev, grids, *, arch, cache) -> None:
    """Honour the ``static_inter_scale`` precondition the dispatcher declares.

    That spec reads the intermediate's fp8 scale instead of deriving it from
    the tile amax, so an uninitialised ``InterScale`` yields NaN at full speed:
    no launch error, no suspicious latency. Populating it from the dynamic form
    of the *same* kernel on the *same* input is also the strictest choice for a
    correctness check -- both then quantize with the same divisor, so any
    remaining difference is arithmetic rather than a different quantisation.
    """
    from ...runtime.launcher import KernelLauncher, LaunchConfig

    candidate = plan[0].candidate
    spec = plan[0].spec
    cal_spec = replace(spec, name=f"{spec.name}_cal", static_inter_scale=False)
    artifact, _ = _compile_cached(
        plan[0], arch=arch, cache=cache, spec=cal_spec, tag="cal"
    )
    launcher = KernelLauncher(
        hsaco=artifact.hsaco,
        kernel_name=artifact.kernel_name,
        # Same candidate, so the same ABI: dropping the static scale changes
        # where the scale comes from, not the argument list.
        signature=candidate.signature(cal_spec),
        cache_key=("moe_bench_cal", cal_spec.kernel_name()),
    )
    launcher(
        dev.values,
        config=LaunchConfig(stream=0, grid=grids[0], block=(cal_spec.block_size, 1, 1)),
    )
    dev.rt.sync()


def check_tokens(shape: MoeShape, *, mode: str, sample: int, seed: int):
    """Which tokens to verify, and the label that says so in the report."""
    T = shape.num_tokens
    if mode == "none":
        return np.empty(0, dtype=np.int64), "none"
    if mode == "oracle" or (mode == "auto" and T <= ORACLE_MAX_TOKENS):
        return np.arange(T, dtype=np.int64), "oracle/all"
    n = min(sample, T)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(T, size=n, replace=False)), f"oracle/sample({n})"


# ---------------------------------------------------------------------------
# Per-shape driver
# ---------------------------------------------------------------------------


def run_shape(
    shape: MoeShape,
    weights: ExpertWeights,
    *,
    arch: str,
    dtype: str,
    args,
    compile_cache: dict,
    weight_cache: dict,
    log: Callable[[str], None],
) -> ShapeRecord:
    """Dispatch one shape, run whatever came back, check it and time it."""
    plan = dispatch_moe_plan(shape.request(arch=arch, dtype=dtype))
    split = len(plan) == 2
    route = "split" if split else "fused"
    spec1 = plan[0].spec

    if split and plan[0].spec.tile_m != plan[1].spec.tile_m:
        # Stage 2 reads stage 1's intermediate with stage 1's row blocking, so
        # this is a wrong answer rather than a slow one. Cheap to assert here.
        return ShapeRecord(
            shape=shape,
            route=route,
            launches=len(plan),
            explanation=tuple(plan[0].explanation),
            verdict="FAIL",
            note=(
                f"plan disagrees on tile_m: stage1={plan[0].spec.tile_m} "
                f"stage2={plan[1].spec.tile_m}"
            ),
        )

    layout = TokenLayout(Routing(shape, seed=args.seed), shape, spec1.tile_m)
    grids = plan_stage_grids(plan, layout)
    warnings = grid_bound_warnings(plan, layout)
    log(
        f"  T={shape.num_tokens:<5} {route:<5} "
        f"{shape.num_tokens * shape.top_k} slots -> {layout.num_m_blocks} blocks, "
        f"{layout.active_experts()}/{shape.num_experts} experts active"
    )
    for warning in warnings:
        log(f"        WARN {warning}")

    if args.plan_only:
        stages = tuple(
            _stage_record(r, grid=g, build_s=0.0) for r, g in zip(plan, grids)
        )
        return ShapeRecord(
            shape=shape,
            route=route,
            launches=len(plan),
            stages=stages,
            explanation=tuple(plan[0].explanation),
            warnings=warnings,
        )

    artifacts, build_s = [], []
    for result in plan:
        artifact, seconds = _compile_cached(result, arch=arch, cache=compile_cache)
        artifacts.append(artifact)
        build_s.append(seconds)
    stages = tuple(
        _stage_record(r, grid=g, build_s=s) for r, g, s in zip(plan, grids, build_s)
    )
    record = ShapeRecord(
        shape=shape,
        route=route,
        launches=len(plan),
        stages=stages,
        explanation=tuple(plan[0].explanation),
        warnings=warnings,
    )

    swizzle = (bool(spec1.swizzle_gu), bool(spec1.swizzle_down))
    for result in plan:
        if (result.spec.swizzle_gu, result.spec.swizzle_down) != swizzle:
            return replace(
                record,
                verdict="FAIL",
                note=(
                    "plan mixes weight layouts: "
                    f"{result.candidate.name} wants "
                    f"({result.spec.swizzle_gu}, {result.spec.swizzle_down})"
                ),
            )
    if swizzle not in weight_cache:
        t = time.time()
        weight_cache[swizzle] = DeviceWeights(
            weights, swizzle_gu=swizzle[0], swizzle_down=swizzle[1]
        )
        log(
            f"        uploaded expert weights swizzle={swizzle} "
            f"in {time.time() - t:.1f}s"
        )
    dev = DeviceProblem(layout, weight_cache[swizzle])
    launch = _launch_plan(plan, dev, artifacts, grids)

    calibrated = bool(spec1.static_inter_scale)
    try:
        if calibrated:
            _calibrate_inter_scale(plan, dev, grids, arch=arch, cache=compile_cache)
        dev.zero_y()
        launch()
        dev.rt.sync()
        Y = dev.read_y()
    except Exception as exc:  # noqa: BLE001 - a launch failure is a result
        return replace(
            record,
            calibrated=calibrated,
            verdict="FAIL",
            note=f"LAUNCH_FAIL {type(exc).__name__}: {str(exc)[:80]}",
        )

    tokens, check = check_tokens(
        shape, mode=args.check, sample=args.sample_tokens, seed=args.seed
    )
    rel: Optional[float] = None
    note = ""
    # Finiteness covers the WHOLE output even when only a sample is verified
    # numerically: the failure mode a static intermediate scale introduces is
    # NaN at full speed, and that would be invisible in an unlucky sample.
    finite = bool(np.isfinite(Y).all())
    if len(tokens):
        ref = oracle_tokens(
            weights, layout, tokens, hidden_group_k=spec1.hidden_group_k
        )
        got = Y[tokens]
        rel = float(np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9))
        ok = finite and rel < args.tol
        if not ok:
            note = (
                f"rel={rel:.3e} tol={args.tol:.1e} finite={finite} "
                f"nan={int(np.isnan(Y).sum())}"
            )
        verdict = "pass" if ok else "FAIL"
    else:
        verdict = "pass" if finite else "FAIL"
        note = "" if finite else f"non-finite output ({int(np.isnan(Y).sum())} NaN)"

    record = replace(
        record,
        calibrated=calibrated,
        check=check,
        checked_tokens=int(len(tokens)),
        rel=rel,
        verdict=verdict,
        note=note,
    )
    if verdict != "pass":
        return record

    from ...runtime.launcher import time_launches

    # Timed last, and never re-read: the epilogue accumulates into Y with
    # atomics and nothing re-zeros it between iterations, so Y is meaningless
    # after this point. The calibration launch stays outside the loop -- it is
    # a precondition of the selection, not part of the layer.
    ms = time_launches(launch, warmup=args.warmup, iters=args.iters)
    return replace(record, us=ms * 1000.0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_HEADER = (
    f"{'tokens':>6} {'route':<6} {'tile_m':>6} {'launch':>6} {'us':>9} "
    f"{'check':<18} {'rel':>10}  verdict"
)


def format_row(record: ShapeRecord) -> str:
    tile_m = record.stages[0].tile_m if record.stages else 0
    us = "-" if record.us is None else f"{record.us:9.1f}"
    rel = "-" if record.rel is None else f"{record.rel:10.3e}"
    line = (
        f"{record.shape.num_tokens:>6} {record.route:<6} {tile_m:>6} "
        f"{record.launches:>6} {us:>9} {record.check:<18} {rel:>10}  "
        f"{record.verdict}"
    )
    if record.note:
        line += f"  {record.note}"
    for stage in record.stages:
        bound = ""
        if tuple(stage.grid) != tuple(stage.grid_dispatch):
            bound = f" (dispatch bound {tuple(stage.grid_dispatch)})"
        line += (
            f"\n{'':>8}{stage.candidate} grid={tuple(stage.grid)}{bound} "
            f"block={tuple(stage.block)} coop={int(stage.coop_b_lds)} "
            f"static_scale={int(stage.static_inter_scale)} "
            f"warp_m={stage.warp_m} warp_n={stage.warp_n} "
            f"spec_hash={stage.spec_hash}"
        )
    for warning in record.warnings:
        line += f"\n{'':>8}WARN {warning}"
    return line


def write_json(path: Path, *, config: dict, records: Sequence[ShapeRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "schema": SCHEMA,
        "config": config,
        "shapes": [r.as_dict() for r in records],
    }
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _default_arch() -> str:
    """The local device where it can be probed, else the codegen default."""
    try:
        from ...runtime.hip_module import get_device_arch

        return get_device_arch() or "gfx950"
    except Exception:  # noqa: BLE001 - no HIP is a normal --plan-only case
        return "gfx950"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dispatcher-driven fused-MoE fp8 benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--tokens",
        default=",".join(str(t) for t in DEFAULT_TOKENS),
        help="comma-separated token counts; the default set spans both routes "
        "and both tile_m bands",
    )
    parser.add_argument("--arch", default=None, help="default: the local device")
    parser.add_argument("--dtype", default="fp8")
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=768)
    parser.add_argument("--experts", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--check",
        default="auto",
        choices=("auto", "oracle", "sample", "none"),
        help="'oracle' verifies every token against the numpy f32 model, "
        "'sample' verifies --sample-tokens of them exactly, 'none' checks only "
        f"that the output is finite. 'auto' is oracle up to T={ORACLE_MAX_TOKENS} "
        "and sample above it",
    )
    parser.add_argument("--sample-tokens", type=int, default=8)
    parser.add_argument("--tol", type=float, default=1.5e-2)
    parser.add_argument(
        "--warmup",
        type=int,
        default=300,
        help="the split route needs a few hundred iterations to reach steady "
        "state; under-warming inflates it by 20-60%% and looks like a routing "
        "difference",
    )
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=11939)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the routing decisions without building or launching",
    )
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    # See the module docstring: torch's bundled LLVM changes what comgr does.
    assert "torch" not in sys.modules, "this harness must stay torch-free"

    arch = args.arch or _default_arch()
    tokens = [int(t) for t in args.tokens.replace(",", " ").split()]
    shapes = [
        MoeShape(
            num_tokens=t,
            hidden=args.hidden,
            intermediate=args.intermediate,
            num_experts=args.experts,
            top_k=args.top_k,
        )
        for t in tokens
    ]

    def log(msg: str) -> None:
        print(msg, flush=True)

    log(
        f"arch={arch} dtype={args.dtype} hidden={args.hidden} "
        f"intermediate={args.intermediate} experts={args.experts} "
        f"top_k={args.top_k} tokens={tokens}"
    )

    weights: Optional[ExpertWeights] = None
    if not args.plan_only:
        t = time.time()
        weights = ExpertWeights(shapes[0], seed=args.seed)
        log(f"generated + quantised {args.experts} experts in {time.time() - t:.1f}s")

    compile_cache: dict = {}
    weight_cache: dict = {}
    records: list[ShapeRecord] = []
    for shape in shapes:
        try:
            records.append(
                run_shape(
                    shape,
                    weights,
                    arch=arch,
                    dtype=args.dtype,
                    args=args,
                    compile_cache=compile_cache,
                    weight_cache=weight_cache,
                    log=log,
                )
            )
        except Exception as exc:  # noqa: BLE001 - a shape that fails is data
            records.append(
                ShapeRecord(
                    shape=shape,
                    route="?",
                    launches=0,
                    verdict="FAIL",
                    note=f"{type(exc).__name__}: {str(exc)[:120]}",
                )
            )

    log("")
    log(_HEADER)
    for record in records:
        log(format_row(record))

    if args.json:
        write_json(
            args.json,
            config={
                "arch": arch,
                "dtype": args.dtype,
                "hidden": args.hidden,
                "intermediate": args.intermediate,
                "num_experts": args.experts,
                "top_k": args.top_k,
                "tokens": tokens,
                "warmup": args.warmup,
                "iters": args.iters,
                "tol": args.tol,
                "check": args.check,
                "sample_tokens": args.sample_tokens,
                "seed": args.seed,
                "plan_only": bool(args.plan_only),
            },
            records=records,
        )
        log(f"wrote {args.json}")
    return 0 if all(r.ok for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
