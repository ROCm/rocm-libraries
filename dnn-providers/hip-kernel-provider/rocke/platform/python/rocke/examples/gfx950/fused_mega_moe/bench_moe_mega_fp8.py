"""Torch-free numpy harness for rocKE's fused-MoE mega kernel (FP8 e4m3).

Why torch-free: importing torch before the first Comgr compile makes
``build_hsaco_from_llvm_ir`` pathological -- the same kernel that compiles in
~0.1s in a clean process never finishes once torch's LLVM is resident in the
process. This harness therefore never imports torch. Host data, quantisation
and the reference run on numpy; device memory, launch and timing go through
rocKE's own ctypes-only HIP runtime.

fp8 e4m3 is encoded by hand against a 256-entry table rather than via
``ml_dtypes``, which this path deliberately does not depend on.

Structure mirrors what actually depends on what: :class:`Weights` is a function
of the shape alone and is the expensive part (128 experts of fp8 quantisation),
so it is built once and cached on disk; :class:`Layout` is a function of
``tile_m`` and is cheap. A lever sweep therefore pays the weight cost once.

Run it with an interpreter that has numpy but NOT torch -- ``main`` asserts as
much, because a torch import anywhere in the process is the failure above and it
shows up as a hang rather than an error::

    PYTHONPATH=<rocke platform python> python3 -u \\
        rocke/examples/gfx950/fused_mega_moe/bench_moe_mega_fp8.py --shape qwen3
    ... --shape qwen3 --sweep                    # every config in sweep_configs()
    ... --shape qwen3 --sweep --configs 'gb_*'   # fnmatch over config labels
"""

from __future__ import annotations

import argparse
import ctypes
import fnmatch
import os
import sys
import time
from dataclasses import (
    dataclass,
    fields as dataclass_fields,
    replace as dataclass_replace,
)
from pathlib import Path

import numpy as np

# The dispatcher's tuning record: the shape these sweeps are run against and
# the knobs they produced. Imported rather than restated so a sweep cannot be
# measuring a geometry the dispatcher does not route to. Plain data -- it pulls
# in neither torch nor the IR layer, so it is safe at module scope here.
from rocke.instances.common.moe_fused_mega_fp8_tuned import TUNED_SHAPE

GROUP_K = 128  # block-scale group width, on both weight axes and the activation
#: Above this token count ``--ref auto`` stops using the numpy oracle. The oracle
#: is a per-expert f32 GEMM chain in numpy and grows with T; at T=4096 it is tens
#: of minutes against a ~4 ms measurement.
ORACLE_MAX_TOKENS = 64
FP8_MAX = 448.0
AMAX_FLOOR = 1e-6
# Generated fp8 expert weights are ~600 MB per shape, so the cache must never
# land inside the source tree. Point ROCKE_MOE_BENCH_CACHE at scratch space.
CACHE_ROOT = Path(
    os.environ.get(
        "ROCKE_MOE_BENCH_CACHE", str(Path(__file__).resolve().parent / ".cache")
    )
)


def log(msg: str) -> None:
    print(msg, flush=True)


class step:
    """Print a phase and its duration, so a stall is always attributable."""

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        log(f"[ .. ] {self.label}")
        self.t = time.time()
        return self

    def __exit__(self, exc_type, *a):
        tag = "FAIL" if exc_type else " ok "
        log(f"[{tag}] {self.label:<44} {time.time() - self.t:7.2f}s")


# ---------------------------------------------------------------------------
# fp8 e4m3 (OCP "fn" flavour: no inf, 0x7f/0xff are NaN, max magnitude 448)
# ---------------------------------------------------------------------------


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


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x, dtype=np.float32))


def block_scale_2d(w: np.ndarray) -> np.ndarray:
    """Per-(128x128)-block amax/448 scale, shape [out/128, k/128]."""
    nob, nkb = w.shape[0] // GROUP_K, w.shape[1] // GROUP_K
    amax = np.abs(w.reshape(nob, GROUP_K, nkb, GROUP_K)).max(axis=(1, 3))
    return (np.maximum(amax, AMAX_FLOOR) / FP8_MAX).astype(np.float32)


def expand(scale: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(scale, GROUP_K, axis=0), GROUP_K, axis=1)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    name: str
    tokens: int
    experts: int
    topk: int
    hidden: int
    intermediate: int


def _tuned(name: str, tokens: int, **overrides) -> Shape:
    """A lab shape stated as its distance from the dispatcher's tuned cohort.

    The dims come from :data:`TUNED_SHAPE` so that this catalogue and the bands
    in ``rocke.dispatch.families.moe`` cannot describe different geometries --
    a sweep run against dims the dispatcher does not route to would produce
    numbers that look like evidence for a band and are not. ``overrides`` is
    how a probe says which single dimension it is moving.
    """
    dims = dict(
        experts=TUNED_SHAPE.num_experts,
        topk=TUNED_SHAPE.top_k,
        hidden=TUNED_SHAPE.hidden,
        intermediate=TUNED_SHAPE.intermediate,
    )
    dims.update(overrides)
    return Shape(name=name, tokens=tokens, **dims)


SHAPES = {
    # Qwen3-30B-A3B decode: hidden and expert count come off the traced operand
    # shapes; top-8 off the incumbent topkGating template arguments. This is the
    # shape the shipped mega-kernel levers were tuned against, and it is the
    # cohort the dispatcher's token bands claim.
    "qwen3": _tuned("qwen3", 32),
    # Same work, but the gate/up weight row stride is no longer a power of two
    # (2176 = 17*128). Probes whether the 2048B stride is aliasing DRAM
    # banks/channels and costing sustained read rate.
    "qwen3_h2176": _tuned("qwen3_h2176", 32, hidden=2176),
    "qwen3_h2560": _tuned("qwen3_h2560", 32, hidden=2560),
    # Prefill. Same model, only the batch grows. These are a DIFFERENT regime,
    # not more of the same: past T~64 every expert is live, so the weight stream
    # stops growing and cost becomes proportional to rows. The mega-kernel is a
    # decode design and does not lead here, so a lever that wins at these shapes
    # is not evidence about decode (and vice versa) -- keep the two separate.
    "qwen3_t256": _tuned("qwen3_t256", 256),
    "qwen3_t1024": _tuned("qwen3_t1024", 1024),
    "qwen3_t4096": _tuned("qwen3_t4096", 4096),
    "canonical": Shape("canonical", 8, 8, 2, 4096, 7168),
    "tiny": Shape("tiny", 8, 8, 2, 1024, 512),
}


# ---------------------------------------------------------------------------
# Host state
# ---------------------------------------------------------------------------

_WEIGHT_ARRAYS = (
    "Wg_q",
    "Wu_q",
    "Wd_q",
    "gate_scale",
    "up_scale",
    "down_scale",
    "X",
    "topk_ids",
    "topk_weights",
)


class Weights:
    """Quantised expert weights + routing. Depends on the shape only.

    Weights are generated and quantised one expert at a time so the f32 master
    never exists for the whole tensor: at 128 experts that is the difference
    between a few hundred MB and several GB of host memory.
    """

    def __init__(self, shape: Shape, *, seed: int = 11939, use_cache: bool = True):
        self.shape = shape
        self.nHb = shape.hidden // GROUP_K
        self.nIb = shape.intermediate // GROUP_K
        cache = CACHE_ROOT / f"{shape.name}_e{shape.experts}_seed{seed}"
        if use_cache and (cache / "topk_ids.npy").exists():
            with step(f"load cached weights ({cache.name})"):
                for nm in _WEIGHT_ARRAYS:
                    setattr(self, nm, np.load(cache / f"{nm}.npy"))
            return

        with step(f"generate + quantise {shape.experts} experts"):
            self._generate(seed)
        if use_cache:
            with step(f"cache weights -> {cache.name}"):
                cache.mkdir(parents=True, exist_ok=True)
                for nm in _WEIGHT_ARRAYS:
                    np.save(cache / f"{nm}.npy", getattr(self, nm))

    def _generate(self, seed: int) -> None:
        s = self.shape
        rng = np.random.default_rng(seed)
        E, I, H = s.experts, s.intermediate, s.hidden

        self.X = (rng.standard_normal((s.tokens, H)) * 0.1).astype(np.float32)
        self.Wg_q = np.empty((E, I, H), dtype=np.uint8)
        self.Wu_q = np.empty((E, I, H), dtype=np.uint8)
        self.Wd_q = np.empty((E, H, I), dtype=np.uint8)
        self.gate_scale = np.empty((E, self.nHb, self.nIb), dtype=np.float32)
        self.up_scale = np.empty((E, self.nHb, self.nIb), dtype=np.float32)
        self.down_scale = np.empty((E, self.nIb, self.nHb), dtype=np.float32)

        for e in range(E):
            wg = (rng.standard_normal((I, H)) * 0.05).astype(np.float32)
            wu = (rng.standard_normal((I, H)) * 0.05).astype(np.float32)
            wd = (rng.standard_normal((H, I)) * 0.05).astype(np.float32)
            sg, su, sd = block_scale_2d(wg), block_scale_2d(wu), block_scale_2d(wd)
            self.gate_scale[e], self.up_scale[e], self.down_scale[e] = sg.T, su.T, sd.T
            self.Wg_q[e] = quantize_e4m3(wg / expand(sg))
            self.Wu_q[e] = quantize_e4m3(wu / expand(su))
            self.Wd_q[e] = quantize_e4m3(wd / expand(sd))

        logits = rng.standard_normal((s.tokens, E)).astype(np.float32)
        ids = np.argsort(-logits, axis=-1, kind="stable")[:, : s.topk]
        vals = np.take_along_axis(logits, ids, axis=-1)
        ex = np.exp(vals - vals.max(axis=-1, keepdims=True))
        self.topk_ids = ids.astype(np.int32)
        self.topk_weights = (ex / ex.sum(axis=-1, keepdims=True)).astype(np.float32)


class Layout:
    """Sorted/padded activation layout for one ``tile_m``, plus the oracle."""

    def __init__(self, w: Weights, tile_m: int, hidden_group_k: int = GROUP_K):
        self.w = w
        self.shape = w.shape
        self.tile_m = tile_m
        self.hidden_group_k = hidden_group_k
        self._build_blocks()
        self._build_activation()

    def _build_blocks(self) -> None:
        s, tm = self.shape, self.tile_m
        self.counts = [int((self.w.topk_ids == e).sum()) for e in range(s.experts)]
        self.blocks_per_expert = [(c + tm - 1) // tm for c in self.counts]
        self.num_m_blocks = max(sum(self.blocks_per_expert), 1)
        self.total_padded = self.num_m_blocks * tm

        self.sorted_token_ids = np.full(self.total_padded, -1, dtype=np.int32)
        self.sorted_weights = np.zeros(self.total_padded, dtype=np.float32)
        self.block_expert_ids = np.full(self.num_m_blocks, -1, dtype=np.int32)
        self.expert_base = [-1] * s.experts

        blk = 0
        for e in range(s.experts):
            be = self.blocks_per_expert[e]
            if be == 0:
                continue
            tok, slot = np.nonzero(self.w.topk_ids == e)
            base = blk * tm
            self.expert_base[e] = base
            self.block_expert_ids[blk : blk + be] = e
            self.sorted_token_ids[base : base + tok.size] = tok.astype(np.int32)
            self.sorted_weights[base : base + tok.size] = self.w.topk_weights[tok, slot]
            blk += be

    def _build_activation(self) -> None:
        """One activation scale per expert per K-group, broadcast over its blocks.

        That broadcast (including onto padding rows) is what the kernel's
        per-lane dequant fold assumes.
        """
        s, tm, w = self.shape, self.tile_m, self.w
        self.A_q = np.zeros((self.total_padded, s.hidden), dtype=np.uint8)
        self.AScale = np.full(
            (self.total_padded, w.nHb), AMAX_FLOOR / FP8_MAX, dtype=np.float32
        )
        for e in range(s.experts):
            if self.blocks_per_expert[e] == 0:
                continue
            tok, _ = np.nonzero(w.topk_ids == e)
            base = self.expert_base[e]
            sub = w.X[tok]
            amax = np.maximum(
                np.abs(sub.reshape(tok.size, w.nHb, GROUP_K)).max(axis=(0, 2)),
                AMAX_FLOOR,
            )
            scale = (amax / FP8_MAX).astype(np.float32)
            self.A_q[base : base + tok.size] = quantize_e4m3(
                sub / np.repeat(scale, GROUP_K)[None, :]
            )
            self.AScale[base : base + self.blocks_per_expert[e] * tm] = scale

    def reference(self) -> np.ndarray:
        """f32 oracle consuming exactly the operands the kernel consumes."""
        s, tm, w = self.shape, self.tile_m, self.w
        hgk = self.hidden_group_k
        Y = np.zeros((s.tokens, s.hidden), dtype=np.float32)
        for e in range(s.experts):
            if self.blocks_per_expert[e] == 0:
                continue
            tok, slot = np.nonzero(w.topk_ids == e)
            n, base = tok.size, self.expert_base[e]

            Xdq = dequantize_e4m3(self.A_q[base : base + n]) * np.repeat(
                self.AScale[base : base + n], GROUP_K, axis=1
            )
            Wg = dequantize_e4m3(w.Wg_q[e]) * expand(w.gate_scale[e].T)
            Wu = dequantize_e4m3(w.Wu_q[e]) * expand(w.up_scale[e].T)
            Wd = dequantize_e4m3(w.Wd_q[e]) * expand(w.down_scale[e].T)

            hidden = silu(Xdq @ Wg.T) * (Xdq @ Wu.T)

            out = np.empty((n, s.hidden), dtype=np.float32)
            # The intermediate's scale-block width is a kernel knob (it is
            # produced and consumed inside the fused kernel), so the reference
            # has to model whatever the config chose, not always 128.
            nhb = s.intermediate // hgk
            for b0 in range(0, n, tm):
                b1 = min(b0 + tm, n)
                blk = hidden[b0:b1]
                amax = np.maximum(
                    np.abs(blk.reshape(b1 - b0, nhb, hgk)).max(axis=(0, 2)),
                    AMAX_FLOOR,
                )
                hs = np.repeat((amax / FP8_MAX).astype(np.float32), hgk)[None, :]
                out[b0:b1] = (dequantize_e4m3(quantize_e4m3(blk / hs)) * hs) @ Wd.T

            np.add.at(Y, tok, w.topk_weights[tok, slot][:, None] * out)
        return Y

    def active_experts(self) -> int:
        return sum(1 for c in self.counts if c > 0)

    def weight_bytes(self, tensors: int = 3) -> float:
        """Expert-weight bytes the launch must stream from HBM.

        Each active expert's gate/up/down slices are read exactly once across
        the grid, so this is also the floor for the shape -- at ~2 tokens per
        expert there is no reuse left to recover.
        """
        s = self.shape
        return float(self.active_experts() * s.intermediate * s.hidden * tensors)


# ---------------------------------------------------------------------------
# Device side (rocKE HIP runtime only -- no torch)
# ---------------------------------------------------------------------------


class DeviceProblem:
    """Device-side buffers for one (layout, weight-layout) pair.

    ``swizzle_gu`` / ``swizzle_down`` upload the corresponding weight tensors
    in the kernel's coalesced tiled layout instead of row-major. They must
    match the spec flags of every kernel run against this DeviceProblem -- a
    mismatch is silently wrong, not an error, which is why the caller keys its
    cache on them.
    """

    def __init__(
        self, layout: Layout, *, swizzle_gu: bool = False, swizzle_down: bool = False
    ):
        from rocke.instances.common.moe_fused_mega_fp8 import swizzle_b_fp8_weights
        from rocke.runtime.host_buffers import as_u8_buffer
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import DeviceMem

        self.rt = Runtime()
        self._keep: list = []
        self.swizzle_gu = swizzle_gu
        self.swizzle_down = swizzle_down
        w, s = layout.w, layout.shape

        def upload(arr: np.ndarray):
            arr = np.ascontiguousarray(arr)
            self._keep.append(arr)
            mem = DeviceMem(arr.nbytes)
            self.rt.memcpy_h2d(mem.ptr(), as_u8_buffer(arr), arr.nbytes)
            self._keep.append(mem)
            return mem

        def upload_w(arr: np.ndarray, swizzle: bool):
            return upload(swizzle_b_fp8_weights(arr) if swizzle else arr)

        H, I, nHb, nIb = s.hidden, s.intermediate, w.nHb, w.nIb
        self.y_shape = (s.tokens, H)
        self.y_nbytes = s.tokens * H * 4
        self.Y = DeviceMem(self.y_nbytes)
        self._keep.append(self.Y)

        # Partial-fusion staging, allocated HERE rather than per-config even
        # though only ``split`` configs read it. Allocating it inside the config
        # loop instead -- after the ~600 MB of weights are already resident --
        # measures 520 us against 412 us for the same binaries, because this
        # buffer is written by stage 1 and read straight back by stage 2, so its
        # placement is on the critical path. One shared allocation up front keeps
        # every split config comparable to a fresh-process run. 1.4 MB.
        self.Inter = DeviceMem(layout.num_m_blocks * layout.tile_m * I)
        self.InterScale = DeviceMem(layout.num_m_blocks * (I // GROUP_K) * 4)
        self._keep += [self.Inter, self.InterScale]

        self.values = {
            "A": upload(layout.A_q),
            "WGate": upload_w(w.Wg_q, swizzle_gu),
            "WUp": upload_w(w.Wu_q, swizzle_gu),
            "WDown": upload_w(w.Wd_q, swizzle_down),
            "AScale": upload(layout.AScale),
            "WGateScale": upload(w.gate_scale),
            "WUpScale": upload(w.up_scale),
            "WDownScale": upload(w.down_scale),
            "SortedTokenIds": upload(layout.sorted_token_ids),
            "SortedWeights": upload(layout.sorted_weights),
            "BlockExpertIds": upload(layout.block_expert_ids),
            "Y": self.Y,
            "M": layout.total_padded,
            "N": I,
            "K": H,
            "H_out": H,
            "stride_a": H,
            "stride_b_gate": I * H,
            "stride_b_up": I * H,
            "stride_b_down": H * I,
            "stride_a_scale": nHb,
            "stride_gate_scale": nIb,
            "stride_up_scale": nIb,
            "stride_down_scale": nHb,
            "stride_gate_scale_e": nHb * nIb,
            "stride_up_scale_e": nHb * nIb,
            "stride_down_scale_e": nIb * nHb,
            "slot_size": layout.tile_m,
            "tokens": s.tokens,
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
# Tuning configurations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    label: str
    tile_m: int = 16
    tile_n_inter: int = 256
    tile_n_down: int = 256
    tile_k_gu: int = 32
    tile_k_down: int = 64
    warp_n: int = 4
    gate_up_k: int = 128
    down_k: int = 128
    use_dtla: bool = True
    sched_cadence: str = "iglp1"
    persistent: bool = False
    use_fused_kloop: bool = True
    dtla_depth: int = 2
    lds_pad: int = 0
    window_group: int = 1
    down_fused_cells: bool = False
    down_depth: int = 2
    down_group: int = 1
    window_sched: str = "barrier"
    hidden_group_k: int = 128
    swizzle_gu: bool = False
    swizzle_down: bool = False
    #: Warps along M. Splits tile_m across warps so tile_m can grow with the
    #: per-warp accumulator count held fixed. Not a latency lever on this shape,
    #: but it is the only way to vary tile_m without also varying register
    #: pressure, so any tile study needs it as a control.
    warp_m: int = 1
    #: PARTIAL FUSION: run the down GEMM as a SECOND launch tiled over H_out
    #: instead of fusing it over the intermediate (reduction) axis. Worth
    #: -15.5 us at the decode shape (412.5 vs 428.0) because it drops the
    #: cross-slice fp32 atomics from 12.6 MB to 2.1 MB.
    split: bool = False
    #: Stage 2's own geometry. It is a separate kernel with about a quarter of
    #: stage 1's register pressure, so it does not want stage 1's shape:
    #: warp_n=4 is worth 11 us over the warp_n=1 stage 1 runs at (423.5 -> 412.5),
    #: which is why these are separate fields rather than inherited.
    down_warp_n: int = 4
    #: Neutral at this shape (412.5 padded vs 412.4 unpadded) -- stage 2 reads the
    #: intermediate straight from HBM rather than restaging it through LDS, so the
    #: bank-conflict padding the fused kernel needs buys nothing here. Kept as a
    #: field, and at 0, because it costs LDS for no measured return.
    down_lds_pad: int = 0
    down_tile_n: int = 128
    #: Output tiles walked per split-down CTA. At 1 every h-tile CTA re-stages
    #: the same intermediate; raising it stages once and reuses.
    down_h_loop: int = 1
    #: Prefetch the next K group's W_down under the current group's MFMAs.
    down_pipeline_k: bool = False
    #: Stage 1 reads the intermediate's fp8 scale instead of deriving it from
    #: the tile amax, collapsing the three-pass epilogue into one pass.
    static_inter_scale: bool = False
    #: Accumulate MFMAs in arch VGPRs, deleting the per-K-group
    #: v_accvgpr_read traffic the block-scale fold otherwise needs.
    mfma_vgpr_form: bool = False
    #: Stage the gate/up weight tile once per CTA in LDS, shared by all waves,
    #: instead of a private per-wave copy. This is what makes wide tile_m cut
    #: weight traffic rather than just shrinking the grid.
    coop_b_lds: bool = False

    def spec(self, PROD, intermediate: int = TUNED_SHAPE.intermediate):
        return PROD.FusedMegaKernelSpecFp8(
            name=f"np_{self.label}",
            tile_m=self.tile_m,
            warp_m=self.warp_m,
            block_size=0,  # re-derive from warp_m * warp_n * wave_size
            tile_n_inter=self.tile_n_inter,
            tile_n_down=self.tile_n_down,
            tile_k_gu=self.tile_k_gu,
            tile_k_down=self.tile_k_down,
            warp_n=self.warp_n,
            gate_up_k=self.gate_up_k,
            down_k=self.down_k,
            use_dtla=self.use_dtla,
            sched_cadence=self.sched_cadence,
            use_fused_kloop=self.use_fused_kloop,
            dtla_depth=self.dtla_depth,
            lds_pad=self.lds_pad,
            window_group=self.window_group,
            down_fused_cells=self.down_fused_cells,
            down_depth=self.down_depth,
            down_group=self.down_group,
            window_sched=self.window_sched,
            hidden_group_k=self.hidden_group_k,
            swizzle_gu=self.swizzle_gu,
            swizzle_down=self.swizzle_down,
            split_inter_max=intermediate,
            static_inter_scale=self.static_inter_scale,
            mfma_vgpr_form=self.mfma_vgpr_form,
            coop_b_lds=self.coop_b_lds,
        )

    def down_spec(self, PROD, intermediate: int):
        """Stage-2 spec for ``split``. Same tile_m (the intermediate's row
        layout is shared) and independent everything else.

        warp_m is capped at 4 here: stage 2 keeps warp_n=4, and
        warp_m*warp_n*64 must stay within the 1024-thread workgroup limit.
        """
        return dataclass_replace(
            self.spec(PROD, intermediate),
            name=f"np_{self.label}_dn",
            warp_m=min(self.warp_m, 4),
            warp_n=self.down_warp_n,
            tile_n_down=self.down_tile_n,
            lds_pad=self.down_lds_pad,
            down_h_loop=self.down_h_loop,
            down_pipeline_k=self.down_pipeline_k,
            block_size=0,
        )


def sweep_configs() -> "list[Config]":
    """Levers worth trying on a 128-expert / top-8 / narrow-intermediate MoE.

    The shipped defaults were tuned at I=7168; here I=768 leaves only three
    N-tiles, so the narrow variants trade per-CTA work for CTA count. The
    K-atom / DTLA / scheduler levers are the documented compute-side wins,
    included to show whether compute-side tuning moves this shape at all.
    """
    cfgs = [Config("default")]
    cfgs += [Config(f"tile_n_inter{v}", tile_n_inter=v) for v in (64, 128)]
    cfgs += [Config(f"tile_n_down{v}", tile_n_down=v) for v in (128, 512)]
    cfgs += [Config(f"warp_n{v}", warp_n=v) for v in (2, 8)]
    cfgs.append(Config("gate_up_k32", gate_up_k=32))
    cfgs.append(Config("down_k32", down_k=32))
    cfgs.append(Config("no_dtla", use_dtla=False, gate_up_k=32))
    cfgs += [Config(f"sched_{c}", sched_cadence=c) for c in ("none", "sgb")]
    cfgs.append(Config("tile_m32", tile_m=32))
    return cfgs + combo_configs() + grid_configs()


def grid_configs() -> "list[Config]":
    """Third pass onward: the accumulated tuning record, oldest first.

    It opens as a cross of the geometry knobs with the DTLA prefetch depth --
    depth interacts with geometry (the useful depth is bounded by the N-cell
    count ``tile_n_inter / warp_n / 32``, and its LDS cost trades against
    CTAs/CU, whose product with the in-flight cell count sets the memory-level
    parallelism), so its peak moves with the tile shape rather than being a
    constant. Everything after the ``gb_`` block below was appended as later
    findings reopened earlier questions: the weight swizzle, partial fusion,
    the prefill wave-count ladder, and the cooperative shared weight tile.

    Read it as a lab notebook, in order. Each block's comment says what it was
    testing and against which bottleneck; a lever rejected in an early block is
    often re-tested in a later one because the bottleneck moved, and that is
    deliberate rather than redundant.
    """
    out: "list[Config]" = []
    for tni, wn in ((64, 1), (128, 1), (128, 2), (256, 2), (256, 4)):
        for tnd in (128, 256):
            for d in (2, 3, 4):
                out.append(
                    Config(
                        f"g_tni{tni}_wn{wn}_tnd{tnd}_d{d}",
                        tile_n_inter=tni,
                        warp_n=wn,
                        tile_n_down=tnd,
                        dtla_depth=d,
                    )
                )

    # Knobs not yet crossed with the leading geometry (tni128/wn1/tnd128/d3).
    def best(label: str, **kw) -> Config:
        base = dict(tile_n_inter=128, warp_n=1, tile_n_down=128, dtla_depth=3)
        base.update(kw)
        return Config(f"gb_{label}", **base)

    out += [
        best("tkgu64", tile_k_gu=64),
        best("tkdown128", tile_k_down=128),
        best("tkdown32", tile_k_down=32),
        best("downk32", down_k=32),
        best("gateupk32", gate_up_k=32),
        best("sched_sgb", sched_cadence="sgb"),
        best("tile_m32", tile_m=32),
        best("tile_m8", tile_m=8),
        best("persistent", persistent=True),
        # No-LDS weight path: gate/up weights go global->VGPR with a rolling
        # ``dtla_depth``-deep window over the ni cells, like the down GEMM and
        # like Triton. Drops BStage_smem entirely, so occupancy should rise on
        # its own rather than being forced.
        *[best(f"nolds_d{d}", use_dtla=False, dtla_depth=d) for d in (2, 3, 4, 6)],
        # Finer wave decomposition: Triton gives each wave a 16x32 output tile
        # (BLOCK_N=128 split over 4 warps) and so has ~4x the waves in flight.
        # Ours hands a whole 16x128 tile to ONE wave, which is why occupancy
        # sits at 6.9 even once LDS stops capping it at 7.
        *[
            Config(f"gb_nolds_tni128_wn4_d{d}", tile_n_inter=128, warp_n=4,
                   tile_n_down=128, use_dtla=False, dtla_depth=d)
            for d in (2, 3)
        ],
        *[
            Config(f"gb_nolds_tni256_wn8_d{d}", tile_n_inter=256, warp_n=8,
                   tile_n_down=256, use_dtla=False, dtla_depth=d)
            for d in (2, 3)
        ],
        Config("gb_nolds_tni128_wn2_d2", tile_n_inter=128, warp_n=2,
               tile_n_down=128, use_dtla=False, dtla_depth=2),
        # Push on the no-LDS depth-3 point.
        *[
            best(f"n3_{lbl}", use_dtla=False, dtla_depth=3, **kw)
            for lbl, kw in (
                ("persistent", dict(persistent=True)),
                ("tnd256", dict(tile_n_down=256)),
                ("tnd512", dict(tile_n_down=512)),
                ("tkgu64", dict(tile_k_gu=64)),
                ("tkdown32", dict(tile_k_down=32)),
                ("tkdown128", dict(tile_k_down=128)),
                ("downk32", dict(down_k=32)),
                ("sgb", dict(sched_cadence="sgb")),
            )
        ],
        # Same no-LDS compute path, but LDS inflated purely as ballast to walk
        # CTAs/CU back down (10752B -> 15 CTAs; +32*pad bytes per step). If
        # throughput RISES as occupancy falls, contention between resident CTAs
        # is costing more than the extra latency hiding buys.
        *[
            best(f"n3_pad{p}", use_dtla=False, dtla_depth=3, lds_pad=p)
            for p in (128, 256, 384, 512)
        ],
        # Scheduling region width for the windowed path, at the tied-best point.
        *[
            best(f"n3b_g{g}", use_dtla=False, dtla_depth=3, lds_pad=384,
                 window_group=g)
            for g in (2, 4, 8)
        ],
        *[
            best(f"n3b_d{d}_g{g}", use_dtla=False, dtla_depth=d, lds_pad=384,
                 window_group=g)
            for d, g in ((2, 2), (4, 2), (4, 4), (6, 4), (8, 8))
        ],
        # Fused down cells: shared LDS A read + windowed W_down prefetch.
        *[
            best(f"dn_d{dd}_g{dg}", use_dtla=False, dtla_depth=2, lds_pad=384,
                 window_group=2, down_fused_cells=True, down_depth=dd,
                 down_group=dg)
            for dd, dg in ((2, 1), (2, 2), (3, 1), (3, 2), (4, 2), (4, 4),
                           (2, 4), (8, 8))
        ],
        # Coalesced weight layout on the leading config (gb_dn_d2_g1). In
        # row-major the 64 lanes of a B fragment read 16 rows 2048B apart, so
        # each 16B chunk instruction touches 16 cache lines instead of 8;
        # the swizzled layout makes a chunk 1024 contiguous bytes. gu/dn are
        # separated because they are different streams (gate+up is ~2/3 of the
        # weight bytes) and each carries its own upload cost.
        *[
            best(f"dn_d2_g1_swz{lbl}", use_dtla=False, dtla_depth=2, lds_pad=384,
                 window_group=2, down_fused_cells=True, down_depth=2,
                 down_group=1, swizzle_gu=gu, swizzle_down=dn)
            for lbl, gu, dn in (("gu", True, False), ("dn", False, True),
                                ("", True, True))
        ],
        # Re-sweep on top of the swizzle. Every knob below was measured as a
        # dead end while the kernel was address-pipe bound; the swizzle cuts
        # TA_BUSY by 14% and moves the stall into TCP_PENDING (waiting on the
        # memory system), so the levers that buy memory-level parallelism --
        # prefetch depth, CTA count, scheduling window -- are worth re-testing
        # against the new bottleneck rather than inherited conclusions.
        *[
            best(f"swz_d{d}", use_dtla=False, dtla_depth=d, lds_pad=384,
                 window_group=2, down_fused_cells=True, down_depth=d,
                 down_group=1, swizzle_gu=True, swizzle_down=True)
            for d in (3, 4, 6, 8)
        ],
        *[
            best(f"swz_g{g}", use_dtla=False, dtla_depth=2, lds_pad=384,
                 window_group=g, down_fused_cells=True, down_depth=2,
                 down_group=g, swizzle_gu=True, swizzle_down=True)
            for g in (1, 4, 8)
        ],
        *[
            best(f"swz_pad{p}", use_dtla=False, dtla_depth=2, lds_pad=p,
                 window_group=2, down_fused_cells=True, down_depth=2,
                 down_group=1, swizzle_gu=True, swizzle_down=True)
            for p in (0, 128, 256)
        ],
        best("swz_sgb", use_dtla=False, dtla_depth=2, lds_pad=384,
             window_group=2, down_fused_cells=True, down_depth=2,
             down_group=1, window_sched="sgb", swizzle_gu=True,
             swizzle_down=True),
        best("swz_persistent", use_dtla=False, dtla_depth=2, lds_pad=384,
             window_group=2, down_fused_cells=True, down_depth=2,
             down_group=1, persistent=True, swizzle_gu=True, swizzle_down=True),
        best("swz_dtla", use_dtla=True, dtla_depth=2, lds_pad=384,
             window_group=2, down_fused_cells=True, down_depth=2,
             down_group=1, swizzle_gu=True, swizzle_down=True),
        # PARTIAL FUSION on the dispatched geometry. `split_best` is the current
        # overall best on this shape: 412.5 us against the fused kernel's 428.0
        # and tuned Triton's 426.1. The `split_dwn*` entries are the stage-2 warp
        # count, the one knob that does not transfer from stage 1 (warp_n 1 -> 4
        # is worth 11 us here, where the fused kernel is flat in warp_n); the
        # `pad384` / `dtn256` entries are controls showing stage 2 is insensitive
        # to LDS padding and to its own N-tile.
        *[
            Config(f"gb_split{tag}", tile_n_inter=128, warp_n=1, tile_n_down=128,
                   use_dtla=False, dtla_depth=2, lds_pad=384, window_group=2,
                   down_fused_cells=True, down_depth=2, down_group=1,
                   swizzle_gu=True, swizzle_down=True, split=True, **kw)
            for tag, kw in (
                ("_best", dict(down_warp_n=4, down_lds_pad=0)),
                ("_dwn1", dict(down_warp_n=1, down_lds_pad=0)),
                ("_dwn2", dict(down_warp_n=2, down_lds_pad=0)),
                ("_pad384", dict(down_warp_n=4, down_lds_pad=384)),
                ("_dtn256", dict(down_warp_n=4, down_lds_pad=0, down_tile_n=256)),
            )
        ],
        # FIX 3 (in-CTA h-tile reuse). Stage 2's H_out axis was purely a grid
        # axis, so at qwen3 prefill the 16 CTAs of a token block each staged the
        # SAME [16, 768] intermediate into their own LDS: 16x redundant staging,
        # and only 12 MFMAs per warp to amortise a ~700-instruction prologue
        # over. down_h_loop=h stages once and walks h tiles.
        *[
            Config(f"gb_split_hl{h}", tile_n_inter=128, warp_n=1, tile_n_down=128,
                   use_dtla=False, dtla_depth=2, lds_pad=384, window_group=2,
                   down_fused_cells=True, down_depth=2, down_group=1,
                   swizzle_gu=True, swizzle_down=True, split=True,
                   down_warp_n=4, down_lds_pad=0, down_h_loop=h)
            for h in (2, 4, 8, 16)
        ],
        # FIX 2 (K-direction software pipeline in stage 2). The down window
        # runs along N, which is 2 cells wide here, so down_depth 2/4/8 all
        # emit identical ISA and the kernel never gets more than 4 loads in
        # flight. down_pipeline_k issues group kg+1's W_down before consuming
        # kg's. Crossed with the h-loop since the two are independent.
        *[
            Config(f"gb_split_pk{'' if h == 1 else f'_hl{h}'}", tile_n_inter=128,
                   warp_n=1, tile_n_down=128, use_dtla=False, dtla_depth=2,
                   lds_pad=384, window_group=2, down_fused_cells=True,
                   down_depth=2, down_group=1, swizzle_gu=True,
                   swizzle_down=True, split=True, down_warp_n=4, down_lds_pad=0,
                   down_h_loop=h, down_pipeline_k=True)
            for h in (1, 4, 16)
        ],
        # ISA FINDING: the block-scale fold reads every accumulator out of
        # AGPRs each K group (GROUP_K == atom.k, so one MFMA per group), which
        # is 64 v_accvgpr_read per gate/up K-loop iteration -- 19% of the body
        # -- and pins the kernel at 280 registers / 1 wave per SIMD. Nothing
        # in the tree ever set -amdgpu-mfma-vgpr-form.
        *[
            Config(f"gb_split_vf{tag}", tile_n_inter=128, warp_n=1,
                   tile_n_down=128, use_dtla=False, dtla_depth=2,
                   down_fused_cells=True, down_depth=2,
                   down_group=1, swizzle_gu=True, swizzle_down=True, split=True,
                   down_warp_n=4, down_lds_pad=0, mfma_vgpr_form=True,
                   **{"lds_pad": 384, "window_group": 2, **kw})
            for tag, kw in (
                ("", {}),
                ("_hl4", dict(down_h_loop=4)),
                ("_ss", dict(static_inter_scale=True)),
                ("_ss_hl4", dict(static_inter_scale=True, down_h_loop=4)),
                ("_wg1", dict(window_group=1)),
                # The K loop waits 16 times per iteration for 34 loads: the
                # fused-cell walk consumes each cell's B right after issuing
                # it, so a wave eats ~16 serialised L2 latencies per K group
                # with under one wave per SIMD to hide them. A deeper window
                # issues more cells before the first wait; it costs in-flight
                # B fragments, which is what mfma_vgpr_form just paid for.
                ("_wg4", dict(window_group=4)),
                ("_wg8", dict(window_group=8)),
                ("_wg4_hl4", dict(window_group=4, down_h_loop=4)),
                ("_wg8_hl4", dict(window_group=8, down_h_loop=4)),
            )
        ],
        # The same lever on the FUSED kernel, which has the identical fold.
        *[
            Config(f"gb_vf{tag}", tile_n_down=128,
                   use_dtla=False, dtla_depth=2, lds_pad=384, window_group=2,
                   down_fused_cells=True, down_depth=2, down_group=1,
                   swizzle_gu=True, swizzle_down=True, mfma_vgpr_form=True,
                   **{"tile_n_inter": 128, "warp_n": 1, **kw})
            for tag, kw in (("", {}), ("_wn2", dict(tile_n_inter=256, warp_n=2)))
        ],
        # FIX 1 (single-pass epilogue). The dynamic amax makes the intermediate
        # scale depend on the whole tile, forcing SiLU -> f32 LDS scratch ->
        # cross-lane/cross-warp amax -> re-read -> convert, barrier-separated.
        # Supplying the scale collapses that to one pass and drops LDS from
        # 22608 to 8192 B. _cal in the name is a reminder that the scale is
        # calibrated from the dynamic kernel on the same input, so parity
        # against it should be exact rather than merely close.
        *[
            Config(f"gb_split_ss{tag}", tile_n_inter=128, warp_n=1,
                   tile_n_down=128, use_dtla=False, dtla_depth=2,
                   window_group=2, down_fused_cells=True, down_depth=2,
                   down_group=1, swizzle_gu=True, swizzle_down=True, split=True,
                   down_warp_n=4, down_lds_pad=0, static_inter_scale=True,
                   **{"lds_pad": 384, **kw})
            for tag, kw in (
                ("", {}),
                ("_hl16", dict(down_h_loop=16)),
                ("_p0", dict(lds_pad=0)),
            )
        ],
        # PREFILL: waves per workgroup WITHOUT the amax coupling.
        #
        # Stage 1 ships warp_n=1, i.e. a 64-thread workgroup -- one wave, which
        # is why prefill runs at 1.95 waves/CU with the memory unit stalled 0.53%
        # of the time. The obvious fix, raising warp_n alone, is already known to
        # lose monotonically (+12.6% at 2, +76.4% at 8), and the reason is the
        # requantize: `warps_per_block = 128 / (tile_n_inter / warp_n)`, so at a
        # fixed tile_n_inter every added warp lands inside the SAME 128-inter
        # scale block and has to join a barrier-separated amax combine.
        #
        # Scaling tile_n_inter WITH warp_n keeps 128 columns per warp, so
        # warps_per_block stays 1: each warp owns a whole scale block, computes
        # its amax alone, and the combine degenerates to a self-read. Per-warp
        # accumulators are unchanged (16x128 f32 = 32 VGPR per GEMM) because the
        # warp's tile does not grow -- only the workgroup's does. This is the one
        # occupancy axis Part 5 did not separate from the coupling.
        # tile_n_down has to be a multiple of warp_n*16 for the same reason (the
        # down warp grid tiles it too), which is why the odd warp counts carry
        # 192 rather than 128. That constraint is exactly what the long-standing
        # `gb_slice1` / `gb_slice2` parity failures were: same idea, tile_n_down
        # left at 128, three or six warps covering 96 of its 128 columns.
        *[
            Config(f"gb_pf{sp}_wn{wn}", tile_n_inter=128 * wn, warp_n=wn,
                   tile_n_down=tnd, use_dtla=False, dtla_depth=2, lds_pad=384,
                   window_group=2, down_fused_cells=True, down_depth=2,
                   down_group=1, swizzle_gu=True, swizzle_down=True,
                   split=(sp == "s"), down_warp_n=4, down_lds_pad=0)
            for sp in ("", "s")
            for wn, tnd in ((1, 128), (2, 128), (3, 192), (4, 128), (6, 192))
        ],
        # The same decoupling reached from the other side. warps_per_block is
        # `hidden_group_k / (tile_n_inter / warp_n)`, so a NARROWER scale group
        # keeps one block per warp at a smaller tile_n_inter. That matters here
        # because tile_n_inter must divide I=768 and tile_n_down must divide
        # H_out=2048, and 768 has a factor of 3 while 2048 does not -- so at
        # hgk=128 the only decoupled warp counts this shape admits are 1 and 2.
        # Dropping hgk buys 4 and 8 waves. It also makes the intermediate's
        # quantization finer, which is a numerical change, not just a tiling one:
        # watch `rel` against the oracle, not just the latency.
        # A narrow scale group forces the legacy 32-wide down atom, because the
        # down K atom has to divide hidden_group_k. That is a second change on
        # top of the wave count, so `gb_pfg128*` repeats hgk=128 with the same
        # down_k=32 to separate the two -- without it, any difference here could
        # just be the atom.
        *[
            Config(f"gb_pfg{hgk}{sp}_wn{wn}", tile_n_inter=hgk * wn, warp_n=wn,
                   tile_n_down=128, hidden_group_k=hgk, down_k=32,
                   use_dtla=False, dtla_depth=2, lds_pad=384, window_group=2,
                   down_fused_cells=True, down_depth=2, down_group=1,
                   swizzle_gu=True, swizzle_down=True,
                   split=(sp == "s"), down_warp_n=4, down_lds_pad=0)
            for sp in ("", "s")
            for hgk, wn in ((128, 1), (128, 2), (64, 2), (64, 4), (32, 4), (32, 8))
        ],
        # Same ladder with the LDS padding off. Adding warps only raises waves
        # per CU if workgroups per CU holds, and stage 1's LDS scales with
        # tile_n_inter -- so the wn2 entries above double LDS per workgroup at
        # the same time as they double waves per workgroup, and the two can
        # cancel. lds_pad=0 halves LDS per row (1412 B -> 644 B), which is the
        # only way to tell "more waves does not help" apart from "waves per CU
        # never moved".
        *[
            Config(f"gb_pfp0{sp}_wn{wn}", tile_n_inter=128 * wn, warp_n=wn,
                   tile_n_down=128, use_dtla=False, dtla_depth=2, lds_pad=0,
                   window_group=2, down_fused_cells=True, down_depth=2,
                   down_group=1, swizzle_gu=True, swizzle_down=True,
                   split=(sp == "s"), down_warp_n=4, down_lds_pad=0)
            for sp in ("", "s")
            for wn in (1, 2)
        ],
        # Control: the SAME wave counts with the coupling left in (tile_n_inter
        # pinned at 128, so warps_per_block = warp_n). If the family above wins
        # and this one does not, the coupling was the limiter rather than the
        # wave count -- that is the whole point of running both.
        *[
            Config(f"gb_pfc_wn{wn}", tile_n_inter=128, warp_n=wn,
                   tile_n_down=128, use_dtla=False, dtla_depth=2, lds_pad=384,
                   window_group=2, down_fused_cells=True, down_depth=2,
                   down_group=1, swizzle_gu=True, swizzle_down=True)
            for wn in (2, 4)
        ],
        # Wide token tiles via warp_m. Each row's weight re-read factor is
        # 1/tile_m, so tile_m=64 cuts weight traffic ~4x; warp_m splits the tile
        # across warps so the per-warp accumulator count stays put (without it
        # tile_m=64 needs 416 VGPR + 160 AGPR and collapses to one wave/SIMD).
        # lds_pad must go to 0 as tile_m grows: stage 1 stages tile_m rows of
        # hidden + f32 scratch + scales, which at pad=384 is 1412 B/row and puts
        # tile_m=64 over the 160 KB LDS budget.
        *[
            Config(f"gb_swz_tm{tm}_wm{wm}", tile_n_inter=128, warp_n=1,
                   tile_n_down=128, use_dtla=False, dtla_depth=2, lds_pad=0,
                   window_group=2, down_fused_cells=True, down_depth=2,
                   down_group=1, swizzle_gu=True, swizzle_down=True,
                   tile_m=tm, warp_m=wm)
            for tm, wm in ((32, 2), (64, 4))
        ],
        *[
            Config(f"gb_split_tm{tm}_wm{wm}", tile_n_inter=128, warp_n=1,
                   tile_n_down=128, use_dtla=False, dtla_depth=2, lds_pad=0,
                   window_group=2, down_fused_cells=True, down_depth=2,
                   down_group=1, swizzle_gu=True, swizzle_down=True, split=True,
                   down_warp_n=4, down_lds_pad=0, tile_m=tm, warp_m=wm)
            for tm, wm in ((32, 2), (64, 4))
        ],
        # Same, but accumulating in VGPRs. Weight traffic is 1/tile_m and the
        # kernel is L2-bandwidth bound on it, so tile_m is the one lever that
        # changes the bytes; what blocked it was the register budget, which
        # mfma_vgpr_form has since cut by ~100 per lane.
        *[
            Config(f"gb_split_tm{tm}_wm{wm}_vf{'_ss' if kw else ''}", tile_n_inter=128, warp_n=1,
                   tile_n_down=128, use_dtla=False, dtla_depth=2, lds_pad=0,
                   window_group=2, down_fused_cells=True, down_depth=2,
                   down_group=1, swizzle_gu=True, swizzle_down=True, split=True,
                   down_warp_n=4, down_lds_pad=0, tile_m=tm, warp_m=wm,
                   mfma_vgpr_form=True, **kw)
            for tm, wm, kw in ((32, 1, {}), (32, 2, {}), (64, 2, {}), (64, 4, {}),
                               (32, 2, dict(static_inter_scale=True)),
                               (64, 4, dict(static_inter_scale=True)))
        ],
        # COOPERATIVE shared weight tile. The tm sweep above found tile_m=64
        # inert (2506 -> 2610 us) because warp_m only splits the ACCUMULATORS:
        # every wave still streamed its own private copy of the weight tile, so
        # the traffic stayed per-wave and the 4x was never collected. Staging the
        # tile once per CTA in LDS makes it per-CTA -- 6.63 GB -> 1.66 GB -- and
        # drops LDS/wave enough for 8 waves/CU (measured: 138 regs, 0 spills,
        # 32768 B, 2 WG/CU) against the 1.7 waves/CU everything else has been
        # stuck at. tm16 is the degenerate single-wave case, kept as a parity
        # vehicle: it adds an LDS round trip with no sharing, so it should be
        # SLOWER, and any numerical divergence there is the staging itself.
        *[
            Config(f"gb_coop_tm{tm}_wm{wm}{tag}", tile_n_inter=128, warp_n=1,
                   tile_n_down=128, use_dtla=False, dtla_depth=2, lds_pad=0,
                   window_group=2, down_fused_cells=True, down_depth=2,
                   down_group=1, swizzle_gu=True, swizzle_down=True, split=True,
                   down_warp_n=4, down_lds_pad=0, tile_m=tm, warp_m=wm,
                   mfma_vgpr_form=True, static_inter_scale=True,
                   coop_b_lds=True, **kw)
            for tm, wm, tag, kw in (
                (16, 1, "", {}),
                (64, 4, "", {}),
                (64, 4, "_hl4", dict(down_h_loop=4)),
                (32, 2, "", {}),
            )
        ],
        # More CTAs: the 64-wide hidden block legalises tile_n_inter=64 (12
        # inter slices, 1332 CTAs). Rejected before because CTA count was not
        # the constraint while the address pipe was saturated.
        *[
            Config(f"gb_swz_h64_d{d}_g{g}", tile_n_inter=64, warp_n=1,
                   tile_n_down=128, use_dtla=False, dtla_depth=d,
                   window_group=g, down_fused_cells=True, down_depth=d,
                   down_group=g, hidden_group_k=64, down_k=32, lds_pad=0,
                   swizzle_gu=True, swizzle_down=False)
            for d, g in ((2, 1), (2, 2), (3, 2))
        ],
        # Finer wave decomposition (Triton hands each wave a 16x32 tile).
        *[
            Config(f"gb_swz_tni{tni}_wn{wn}_d{d}", tile_n_inter=tni, warp_n=wn,
                   tile_n_down=tnd, use_dtla=False, dtla_depth=d,
                   window_group=2, down_fused_cells=True, down_depth=d,
                   down_group=1, lds_pad=384, swizzle_gu=True,
                   swizzle_down=True)
            for tni, wn, tnd, d in ((128, 2, 128, 2), (128, 2, 128, 3),
                                    (256, 4, 256, 2), (256, 2, 256, 2))
        ],
        # Atom-shape probes, all on swizzled weights so the layout is not the
        # variable. gfx950's fp8 catalog is exactly three shapes: 16x16x32 and
        # 32x32x16 (CDNA3 carry-overs, 8B per lane) and the 16x16x128 hero
        # (32B per lane) used by default. 16x16x16 and 32x32x8 are f16-only
        # shapes and have no fp8 form.
        #   - k32: the narrower-K atom, 4x the MFMA and 4x the load
        #     instructions for identical bytes. Weight swizzle does not apply
        #     (its 8B fragment is under the 16B chunk) so gate/up stays
        #     row-major here.
        #   - tile_m32: not an atom change, but the token-tile a 32x32 atom
        #     would FORCE. This shape pads 256 token-slots into 111 blocks, so
        #     M=32 doubles padded rows 1776 -> 3552. Measuring it prices the
        #     32x32 family without porting the whole output/quant path to a
        #     16-float accumulator.
        best("swz_gateupk32", use_dtla=False, dtla_depth=2, lds_pad=384,
             window_group=2, down_fused_cells=True, down_depth=2, down_group=1,
             gate_up_k=32, swizzle_gu=False, swizzle_down=True),
        best("swz_downk32", use_dtla=False, dtla_depth=2, lds_pad=384,
             window_group=2, down_fused_cells=True, down_depth=2, down_group=1,
             down_k=32, swizzle_gu=True, swizzle_down=False),
        best("swz_tile_m32", use_dtla=False, dtla_depth=2, lds_pad=384,
             window_group=2, down_fused_cells=True, down_depth=2, down_group=1,
             tile_m=32, swizzle_gu=True, swizzle_down=True),
        best("swz_tile_m8", use_dtla=False, dtla_depth=2, lds_pad=384,
             window_group=2, down_fused_cells=True, down_depth=2, down_group=1,
             tile_m=8, swizzle_gu=True, swizzle_down=True),
        # Inter-slice collapse: fewer slices -> proportionally fewer atomic
        # accumulations into Y (the doc's second target, 12 MB vs Triton's
        # 1.7 MB), now measured with the weight path no longer dominating.
        *[
            Config(f"gb_swz_slice{s}", tile_n_inter=tn, warp_n=wn,
                   tile_n_down=128, use_dtla=False, dtla_depth=2,
                   window_group=1, down_fused_cells=True, down_depth=2,
                   down_group=1, lds_pad=0, swizzle_gu=True, swizzle_down=True)
            for s, tn, wn in ((3, 256, 2), (2, 384, 3), (1, 768, 6))
        ],
        # Collapse the inter split: each output element is atomically
        # accumulated once per inter slice, so 6 -> 3 -> 2 -> 1 slices cuts the
        # partial-Y atomic traffic proportionally. warp_n is set so each warp
        # still covers exactly one 128-wide scale block (the fuse-quant
        # invariant), at the cost of a bigger f32 staging buffer in LDS.
        *[
            Config(f"gb_slice{s}", tile_n_inter=tn, warp_n=wn, tile_n_down=128,
                   use_dtla=False, dtla_depth=2, window_group=1,
                   down_fused_cells=True, down_depth=2, down_group=1,
                   lds_pad=0)
            for s, tn, wn in ((3, 256, 2), (2, 384, 3), (1, 768, 6))
        ],
        # Isolation probe: fused down emitter at atoms_per_group>1, everything
        # else at the known-good settings.
        best("dk32_fused", use_dtla=False, dtla_depth=2, lds_pad=0,
             window_group=2, down_fused_cells=True, down_depth=2,
             down_group=2, down_k=32),
        # Isolation probe: warp_n=2 with the fused down emitter, hgk still 128.
        Config("gb_w2_fused", tile_n_inter=128, warp_n=2, tile_n_down=128,
               use_dtla=False, dtla_depth=2, window_group=2,
               down_fused_cells=True, down_depth=2, down_group=2, down_k=32),
        # Isolation probe: 64-wide hidden block at the OLD 128-wide inter tile
        # (legal because warp_n=2 makes each warp's N-extent exactly 64).
        Config("gb_h64_iso", tile_n_inter=128, warp_n=2, tile_n_down=128,
               use_dtla=False, dtla_depth=2, window_group=2,
               down_fused_cells=True, down_depth=2, down_group=2,
               hidden_group_k=64, down_k=32, lds_pad=0),
        # 64-wide hidden scale block legalises tile_n_inter=64 -> 12 inter
        # slices, 1332 CTAs instead of 666.
        *[
            Config(f"gb_h64_d{d}_g{g}", tile_n_inter=64, warp_n=1,
                   tile_n_down=128, use_dtla=False, dtla_depth=d,
                   window_group=g, down_fused_cells=True, down_depth=d,
                   down_group=g, hidden_group_k=64, down_k=32, lds_pad=0)
            for d, g in ((2, 1), (2, 2), (3, 2), (4, 2))
        ],
        *[
            Config(f"gb_h64p_d2_g2_p{p}", tile_n_inter=64, warp_n=1,
                   tile_n_down=128, use_dtla=False, dtla_depth=2,
                   window_group=2, down_fused_cells=True, down_depth=2,
                   down_group=2, hidden_group_k=64, down_k=32, lds_pad=p)
            for p in (64, 128, 192, 256)
        ],
        # Softer scheduling fence: more loads in flight per wave.
        *[
            best(f"sg_d{d}_g{g}", use_dtla=False, dtla_depth=d, lds_pad=384,
                 window_group=g, down_fused_cells=True, down_depth=d,
                 down_group=g, window_sched="sgb")
            for d, g in ((2, 1), (2, 2), (3, 2), (4, 2), (4, 4), (6, 4), (8, 8))
        ],
        # Refine around depth 2 / group 2.
        *[
            best(f"n4_d{d}_g{g}_p{p}", use_dtla=False, dtla_depth=d,
                 lds_pad=p, window_group=g)
            for d in (2, 3)
            for g in (1, 2, 3)
            for p in (0, 256, 384, 448)
        ],
        *[
            Config(f"gb_nolds_wn4_d{d}", use_dtla=False, dtla_depth=d)
            for d in (2, 3, 4)
        ],
    ]
    return out


def combo_configs() -> "list[Config]":
    """Second pass: stack the single-lever winners and try the grid modes.

    ``warp_n2`` was the only lever that beat the default in isolation, so the
    combinations are built around it; the persistent grid relinearises the
    (3 x 111) work space, which is the other untested axis.
    """
    return [
        Config("warp_n1", warp_n=1),
        # Single-wave threadgroup: one warp owns a whole 128-inter scale block,
        # so the block amax needs no cross-warp step and the body has no
        # barriers at all. Also the register-pressure proxy for a register-
        # chained gate/up -> down design (mfmas_n = 8 gate + 8 up accumulators).
        Config("tni128_wn1", tile_n_inter=128, warp_n=1),
        Config("tni128_wn1_tnd128", tile_n_inter=128, warp_n=1, tile_n_down=128),
        Config("wn2_tnd128", warp_n=2, tile_n_down=128),
        Config("wn2_tni128", warp_n=2, tile_n_inter=128),
        Config("wn2_sched_none", warp_n=2, sched_cadence="none"),
        Config("wn2_tkgu64", warp_n=2, tile_k_gu=64),
        Config("wn2_tkdown128", warp_n=2, tile_k_down=128),
        # DTLA off but KEEPING the K=128 hero atom. The original "no_dtla"
        # config needlessly pinned gate_up_k=32, conflating two levers and
        # landing on the build-pathological legacy atom. global_load_lds forces
        # 16B/lane fetch granularity; the legacy path issues 32B/lane to VGPRs.
        Config("no_dtla_k128", use_dtla=False),
        Config("wn2_no_dtla_k128", warp_n=2, use_dtla=False),
        # DTLA prefetch depth. mfmas_n is 4 at warp_n=4 and 8 at warp_n=1, so
        # depth can usefully go up to the cell count before it saturates.
        *[Config(f"depth{d}", dtla_depth=d) for d in (3, 4, 6, 8)],
        *[
            Config(f"wn1_depth{d}", tile_n_inter=128, warp_n=1, dtla_depth=d)
            for d in (3, 4, 6, 8)
        ],
        # Around the depth-3 peak.
        Config("wn1_d3_tnd128", tile_n_inter=128, warp_n=1, dtla_depth=3,
               tile_n_down=128),
        Config("wn1_d3_tnd512", tile_n_inter=128, warp_n=1, dtla_depth=3,
               tile_n_down=512),
        Config("wn2_d3", warp_n=2, dtla_depth=3),
        Config("wn2_tni128_d3", tile_n_inter=128, warp_n=2, dtla_depth=3),
        Config("wn1_d3_sched_none", tile_n_inter=128, warp_n=1, dtla_depth=3,
               sched_cadence="none"),
        # LDS row padding to break the bank-0 aliasing of the Hidden buffers.
        *[
            Config(f"best_pad{p}", tile_n_inter=128, warp_n=1, dtla_depth=3,
                   tile_n_down=128, lds_pad=p)
            for p in (16, 32, 48, 64)
        ],
        Config("pad16_d2", tile_n_inter=128, warp_n=1, tile_n_down=128, lds_pad=16),
        Config("percell", use_fused_kloop=False),
        Config("percell_wn2", warp_n=2, use_fused_kloop=False),
        Config("percell_tni128_wn1", tile_n_inter=128, warp_n=1, use_fused_kloop=False),
        Config("persistent", persistent=True),
        Config("wn2_persistent", warp_n=2, persistent=True),
    ]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class build_budget:
    """Abort a config whose lowering explodes instead of hanging the sweep.

    The legacy K=32 atom emits four MFMAs per 128-wide scale group where the
    hero atom emits one, so its unrolled IR is large enough that the Python
    lowerer plus Comgr can run for many minutes.
    """

    def __init__(self, seconds: float):
        self.seconds = seconds

    def __enter__(self):
        import signal

        def _onalarm(signum, frame):
            raise TimeoutError(f"lowering exceeded {self.seconds:.0f}s")

        self.old = signal.signal(signal.SIGALRM, _onalarm)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *a):
        import signal

        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, self.old)
        return False


def build_failure_note(exc: BaseException, budget: float) -> str:
    """A config that will not build is data, not a crash.

    The sweep records why and moves on, so one pathological lowering does not
    cost every result after it.
    """
    if isinstance(exc, TimeoutError):
        return f"BUILD_SLOW (>{budget:.0f}s lowering)"
    return f"BUILD_FAIL {type(exc).__name__}: {str(exc)[:50]}"


def verification_launch(dev: DeviceProblem, launch, row: dict):
    """One untimed launch into a zeroed Y; returns Y, or None after a fault.

    Y must be re-zeroed first because the epilogue accumulates into it
    atomically -- reading it after a timed loop would return N launches summed.
    """
    try:
        dev.zero_y()
        launch()
        dev.rt.sync()
        return dev.read_y()
    except Exception as exc:  # noqa: BLE001
        row["note"] = f"LAUNCH_FAIL {type(exc).__name__}: {str(exc)[:50]}"
        return None


def record_parity(Y: np.ndarray, ref: np.ndarray, args, row: dict) -> bool:
    """Score Y against the reference. False disqualifies the config from timing.

    ``--phase gateup`` never writes Y, so there is nothing to score there.
    """
    if args.phase != "full":
        row["rel"] = float("nan")
        return True
    rel = float(np.abs(Y - ref).max() / (np.abs(ref).max() + 1e-9))
    row["rel"] = rel
    if rel < args.tol and np.isfinite(Y).all():
        return True
    row["note"] = (
        f"PARITY_FAIL rel={rel:.3e} nan={int(np.isnan(Y).sum())} "
        f"ymax={float(np.abs(Y).max()):.4g} "
        f"refmax={float(np.abs(ref).max()):.4g}"
    )
    return False


def config_shape_error(cfg: Config, shape: Shape) -> "str | None":
    """Reject tilings the kernel cannot mask, BEFORE they reach the GPU.

    Neither the inter loop nor the down epilogue masks a partial trailing tile:
    the down store computes its column from ``block_id_x * tile_n_down`` and
    writes unconditionally. A tile_n_down that does not divide H_out therefore
    walks off the end of Y and raises a memory access fault, which aborts the
    whole sweep process and loses every result after it -- far worse than the
    config simply being skipped. The spec's own guard cannot catch this because
    H_out and I are runtime arguments and the spec never sees them.
    """
    checks = [("tile_n_inter", cfg.tile_n_inter, "intermediate", shape.intermediate),
              ("tile_n_down", cfg.tile_n_down, "H_out", shape.hidden)]
    if cfg.split:
        # down_h_loop folds h-tiles into one CTA, so the CTA's whole span --
        # not just one tile -- has to divide H_out or the last CTA runs off the
        # end (the down epilogue does not mask a partial trailing tile).
        checks.append(("down_tile_n*down_h_loop",
                       cfg.down_tile_n * cfg.down_h_loop, "H_out", shape.hidden))
    for name, tile, extent_name, extent in checks:
        if extent % tile:
            return (
                f"SHAPE_SKIP {name}={tile} does not divide {extent_name}={extent} "
                "(no partial-tile masking)"
            )
    return None


def kernel_reference(tile_m: int, hgk: int, layout: Layout, dev: DeviceProblem,
                     args) -> np.ndarray:
    """Parity reference from the dispatched fused kernel instead of numpy.

    The numpy oracle is the real check, but it is O(T) and unusable at prefill,
    so above ``ORACLE_MAX_TOKENS`` every config is compared against the shipped
    fused config run at the same tile_m/hgk. That config is itself oracle-verified
    at decode, and it shares no code with the split path's stage 2, so this still
    catches config-specific breakage -- but it CANNOT catch a fault common to
    every config. Treat a clean 'rel' at prefill as "agrees with the shipped
    kernel", not as "correct".
    """
    ref_cfg = Config(
        "__ref__", tile_m=tile_m, hidden_group_k=hgk, tile_n_inter=128, warp_n=1,
        tile_n_down=128, use_dtla=False, dtla_depth=2, lds_pad=384, window_group=2,
        down_fused_cells=True, down_depth=2, down_group=1,
        swizzle_gu=dev.swizzle_gu, swizzle_down=dev.swizzle_down,
    )
    zero = np.zeros(dev.y_shape, dtype=np.float32)
    # capture_y hands back the verification launch's Y and skips timing; tol is
    # infinite because there is nothing yet to be in parity with.
    probe = argparse.Namespace(**{**vars(args), "phase": "full",
                                  "tol": float("inf"), "capture_y": True})
    row = evaluate(ref_cfg, layout, dev, zero, probe)
    if "_y" not in row:
        raise SystemExit(f"could not build the parity reference: {row['note']}")
    return row["_y"]


def evaluate_split(cfg: Config, layout: Layout, dev: DeviceProblem, ref, args,
                   row: dict) -> dict:
    """Two-launch partial fusion: gate/up + requant, then the down GEMM.

    Stage 1 is the same builder as the fused kernel with ``split_gateup=True``,
    which publishes the requantized intermediate to HBM instead of holding it in
    LDS for a fused stage 2. Stage 2 then tiles over H_out and reduces all of I
    in-block, so no CTA owns a partial and the cross-slice fp32 atomics drop
    from 12.6 MB to 2.1 MB.

    The two stages get independent geometry on purpose (``cfg.down_*``). Stage 2
    has about a quarter of stage 1's register pressure and wants 4 warps where
    stage 1 wants 1, which is worth 11 us of the split's 15 us margin over the
    fused kernel -- so inheriting stage 1's spec would hide most of the win.

    Only worthwhile on a narrow-intermediate, many-expert shape: stage 2's grid
    is ``(H_out / tile_n_down) x num_m_blocks``, so a small expert count or a wide
    intermediate starves it and the fused kernel wins instead. It is correct
    either way, just not always faster -- compare the two on the shape at hand.
    """
    from rocke.helpers.compile import compile_kernel
    from rocke.instances.common import moe_fused_mega_fp8 as PROD
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, time_launches

    inter, h_out = layout.shape.intermediate, layout.shape.hidden
    spec1 = cfg.spec(PROD, inter)
    spec2 = cfg.down_spec(PROD, inter)

    t = time.time()
    try:
        with build_budget(args.build_budget):
            a1 = compile_kernel(
                PROD.build_moe_fused_mega_gemm_fp8(
                    spec1, arch="gfx950", split_gateup=True
                ),
                arch="gfx950", capture_ir_text=False,
            )
            a2 = compile_kernel(
                PROD.build_moe_split_down_fp8(spec2, arch="gfx950"),
                arch="gfx950", capture_ir_text=False,
            )
            # A supplied scale has to come from somewhere. Deriving it from the
            # dynamic kernel on this very input is the strictest version of the
            # comparison: the two kernels then quantize with the SAME divisor,
            # so any output difference is the epilogue restructuring rather
            # than a different quantization, and parity should be exact.
            a_cal = compile_kernel(
                PROD.build_moe_fused_mega_gemm_fp8(
                    dataclass_replace(spec1, name=f"{spec1.name}_cal",
                                      static_inter_scale=False),
                    arch="gfx950", split_gateup=True,
                ),
                arch="gfx950", capture_ir_text=False,
            ) if cfg.static_inter_scale else None
    except Exception as exc:  # noqa: BLE001
        row["note"] = build_failure_note(exc, args.build_budget)
        return row
    row["build_s"] = time.time() - t

    # Shared across configs -- see the allocation note in DeviceProblem. Inter is
    # [num_m_blocks*tile_m, I] fp8; its scales are row-uniform within a 128-inter
    # block, so there is one f32 per (m-block, block), not one per row.
    nmb = layout.num_m_blocks
    Inter, InterScale = dev.Inter, dev.InterScale

    v1 = dict(dev.values, Inter=Inter, InterScale=InterScale)
    v2 = {
        "Inter": Inter,
        "InterScale": InterScale,
        "WDown": dev.values["WDown"],
        "WDownScale": dev.values["WDownScale"],
        "SortedTokenIds": dev.values["SortedTokenIds"],
        "SortedWeights": dev.values["SortedWeights"],
        "BlockExpertIds": dev.values["BlockExpertIds"],
        "Y": dev.values["Y"],
        "N": inter,  # full contraction extent, reduced inside stage 2
        "H_out": h_out,
        "stride_b_down": dev.values["stride_b_down"],
        "stride_down_scale": dev.values["stride_down_scale"],
        "stride_down_scale_e": dev.values["stride_down_scale_e"],
        "tokens": layout.shape.tokens,
    }

    k1 = KernelLauncher(
        hsaco=a1.hsaco, kernel_name=a1.kernel_name,
        signature=PROD.moe_fused_mega_fp8_signature(spec1, split_gateup=True),
        cache_key=("moe_split_gu", spec1.kernel_name()),
    )
    k2 = KernelLauncher(
        hsaco=a2.hsaco, kernel_name=a2.kernel_name,
        signature=PROD.moe_split_down_fp8_signature(spec2),
        cache_key=("moe_split_dn", spec2.kernel_name()),
    )
    lc1 = LaunchConfig(stream=0, grid=PROD.moe_fused_mega_fp8_grid(nmb, inter, spec1),
                       block=(spec1.block_size, 1, 1))
    lc2 = LaunchConfig(stream=0, grid=PROD.moe_split_down_fp8_grid(nmb, h_out, spec2),
                       block=(spec2.block_size, 1, 1))
    row["grid"] = [list(lc1.grid), list(lc2.grid)]

    if a_cal is not None:
        # One calibration launch, outside the timed region: it fills InterScale,
        # which the static kernel then reads instead of computing.
        KernelLauncher(
            hsaco=a_cal.hsaco, kernel_name=a_cal.kernel_name,
            signature=PROD.moe_fused_mega_fp8_signature(spec1, split_gateup=True),
            cache_key=("moe_split_gu_cal", spec1.kernel_name()),
        )(v1, config=lc1)
        dev.rt.sync()

    if args.phase == "gateup":
        launch = lambda: k1(v1, config=lc1)  # noqa: E731
        tensors = 2
    else:
        def launch():
            k1(v1, config=lc1)
            k2(v2, config=lc2)
        tensors = 3

    Y = verification_launch(dev, launch, row)
    if Y is None or not record_parity(Y, ref, args, row):
        return row

    # The two-launch shape needs FAR more warmup than the fused kernel to reach
    # steady state, and the harness default (30) was tuned on the fused one. Under
    # -warmed, a split config reads anywhere from 500 to 690 us instead of 413,
    # and -- worst of all -- the inflation depends on what else is in the sweep,
    # so it looks like a config difference rather than a measurement artifact.
    # At 300 warmup iterations every split config lands on its solo number.
    # Raising `--warmup` past this is still honoured.
    ms = time_launches(launch, warmup=max(args.warmup, 300), iters=args.iters)
    row["us"] = ms * 1000.0
    row["gbs"] = layout.weight_bytes(tensors) / (ms * 1e-3) / 1e9
    return row


def evaluate(cfg: Config, layout: Layout, dev: DeviceProblem, ref, args) -> dict:
    from rocke.helpers.compile import compile_kernel
    from rocke.instances.common import moe_fused_mega_fp8 as PROD
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, time_launches

    row = {"label": cfg.label, "us": None, "note": ""}
    bad = config_shape_error(cfg, layout.shape)
    if bad:
        row["note"] = bad
        return row
    if cfg.split:
        return evaluate_split(cfg, layout, dev, ref, args, row)
    spec = cfg.spec(PROD, layout.shape.intermediate)

    t = time.time()
    try:
        with build_budget(args.build_budget):
            kd = PROD.build_moe_fused_mega_gemm_fp8(
                spec, arch="gfx950", persistent=cfg.persistent
            )
            art = compile_kernel(kd, arch="gfx950", capture_ir_text=False)
    except Exception as exc:  # noqa: BLE001
        row["note"] = build_failure_note(exc, args.build_budget)
        return row
    row["build_s"] = time.time() - t

    inter = layout.shape.intermediate
    values = dev.values
    if cfg.persistent:
        grid, gx, total_work, P = PROD.moe_fused_mega_fp8_persistent_grid(
            layout.num_m_blocks, inter, spec
        )
        values = dict(dev.values)
        values.update({"grid_x": gx, "total_work": total_work, "P": P})
    else:
        grid = PROD.moe_fused_mega_fp8_grid(layout.num_m_blocks, inter, spec)

    lcfg = LaunchConfig(stream=0, grid=grid, block=(spec.block_size, 1, 1))
    launcher = KernelLauncher(
        hsaco=art.hsaco,
        kernel_name=art.kernel_name,
        signature=PROD.moe_fused_mega_fp8_signature(spec, persistent=cfg.persistent),
        cache_key=("moe_mega_np", spec.kernel_name(), cfg.persistent),
    )
    row["grid"] = grid

    # Phase isolation: the down projection is a `for ho in range(0, H_out, ...)`
    # loop over a RUNTIME argument, so H_out=0 makes it zero-trip and the launch
    # measures the gate/up half alone -- the two-kernel structure's first half,
    # with no kernel surgery. Parity is meaningless then (Y is untouched).
    tensors = 3
    if args.phase == "gateup":
        values = dict(values)
        values["H_out"] = 0
        tensors = 2

    Y = verification_launch(dev, lambda: launcher(values, config=lcfg), row)
    if Y is None or not record_parity(Y, ref, args, row):
        return row

    if getattr(args, "capture_y", False):
        row["_y"] = Y
        return row

    ms = time_launches(
        lambda: launcher(values, config=lcfg),
        warmup=args.warmup,
        iters=args.iters,
    )
    row["us"] = ms * 1000.0
    row["gbs"] = layout.weight_bytes(tensors) / (ms * 1e-3) / 1e9
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shape", default="tiny", choices=sorted(SHAPES))
    ap.add_argument("--tile-m", type=int, default=16)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--configs", default="", help="comma-separated sweep labels")
    ap.add_argument("--tol", type=float, default=1.5e-2)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--no-cache", dest="cache", action="store_false", default=True)
    ap.add_argument(
        "--phase",
        default="full",
        choices=("full", "gateup"),
        help="gateup passes H_out=0 to isolate the gate/up half (no parity)",
    )
    ap.add_argument(
        "--build-budget",
        type=float,
        default=90.0,
        help="seconds a single config may spend in lowering before it is skipped",
    )
    ap.add_argument(
        "--ref",
        default="auto",
        choices=("auto", "oracle", "kernel"),
        help="parity reference. 'oracle' is the numpy f32 model, which is O(T) "
        "and impractical past a few hundred tokens; 'kernel' uses the dispatched "
        "fused config's own output, which catches config-specific breakage but "
        "not a fault shared by every config. 'auto' picks oracle up to "
        f"T={ORACLE_MAX_TOKENS} and kernel above it.",
    )
    # The two flags below exist so ``serve`` can measure the kernel the MoE
    # dispatcher selected, rather than a sweep label that merely happens to
    # describe the same configuration today. Passing the spec verbatim means
    # dispatch and the measured lane cannot drift into different kernels.
    ap.add_argument(
        "--spec-json",
        default="",
        help="JSON file of Config fields to run instead of the sweep "
        "(the dispatcher writes the spec it selected here)",
    )
    ap.add_argument(
        "--json",
        dest="json_out",
        default="",
        help="write the result rows here as JSON",
    )
    args = ap.parse_args()

    assert "torch" not in sys.modules, "harness must stay torch-free"
    shape = SHAPES[args.shape]
    log(
        f"shape={shape.name} T={shape.tokens} E={shape.experts} K={shape.topk} "
        f"H={shape.hidden} I={shape.intermediate}"
    )

    with step("import rocke"):
        import rocke.helpers.compile  # noqa: F401
        import rocke.instances.common.moe_fused_mega_fp8  # noqa: F401

    weights = Weights(shape, use_cache=args.cache)

    use_oracle = args.ref == "oracle" or (
        args.ref == "auto" and shape.tokens <= ORACLE_MAX_TOKENS
    )
    if not use_oracle:
        log(
            "       parity reference: dispatched fused kernel (numpy oracle is "
            f"O(T); --ref auto drops it above T={ORACLE_MAX_TOKENS}). 'rel' below "
            "means agreement with the shipped kernel, not correctness."
        )

    if args.spec_json:
        import json as _json

        fields = _json.loads(Path(args.spec_json).read_text(encoding="utf-8"))
        label = str(fields.pop("label", "dispatched"))
        unknown = sorted(set(fields) - {f.name for f in dataclass_fields(Config)})
        if unknown:
            raise SystemExit(f"--spec-json has unknown Config fields: {unknown}")
        configs = [Config(label, **fields)]
    else:
        configs = (
            sweep_configs() if args.sweep else [Config("default", tile_m=args.tile_m)]
        )
    if args.configs:
        want = [c.strip() for c in args.configs.split(",")]
        configs = [
            c for c in configs if any(fnmatch.fnmatch(c.label, w) for w in want)
        ]

    # Layout + oracle depend only on tile_m, so build one per distinct tile_m.
    # The device buffers additionally depend on the weight layout, since a
    # swizzled kernel must be fed swizzled weights.
    layout_cache: dict = {}
    cache: dict = {}

    ref_cache: dict = {}

    def state_for(tile_m: int, hgk: int = GROUP_K, swz: tuple = (False, False)):
        if tile_m not in layout_cache:
            with step(f"layout tile_m={tile_m}"):
                lay = Layout(weights, tile_m)
            log(
                f"       {shape.tokens * shape.topk} token-slots -> "
                f"{lay.num_m_blocks} blocks, {lay.total_padded} padded rows "
                f"({lay.total_padded / (shape.tokens * shape.topk):.1f}x real work), "
                f"{lay.active_experts()}/{shape.experts} experts active"
            )
            layout_cache[tile_m] = lay
        lay = layout_cache[tile_m]
        if (tile_m, swz) not in cache:
            tag = "".join(n for n, on in (("gu", swz[0]), ("dn", swz[1])) if on)
            with step(f"upload tile_m={tile_m}{' swz=' + tag if tag else ''}"):
                cache[(tile_m, swz)] = DeviceProblem(
                    lay, swizzle_gu=swz[0], swizzle_down=swz[1]
                )
        dv = cache[(tile_m, swz)]
        if (tile_m, hgk) not in ref_cache:
            if use_oracle:
                with step(f"numpy oracle tile_m={tile_m} hgk={hgk}"):
                    lay.hidden_group_k = hgk
                    ref_cache[(tile_m, hgk)] = lay.reference()
                    lay.hidden_group_k = GROUP_K
            else:
                # The quantized intermediate's amax is taken over a tile_m-row
                # tile, so the reference is only valid for the tile_m/hgk it was
                # produced at -- hence the same cache key as the oracle.
                with step(f"kernel reference tile_m={tile_m} hgk={hgk}"):
                    ref_cache[(tile_m, hgk)] = kernel_reference(
                        tile_m, hgk, lay, dv, args
                    )
        return lay, dv, ref_cache[(tile_m, hgk)]

    log(f"\n{'config':<18} {'us':>9} {'GB/s':>7} {'rel':>10}  grid / note")
    rows = []
    for cfg in configs:
        lay, dv, rf = state_for(
            cfg.tile_m, cfg.hidden_group_k, (cfg.swizzle_gu, cfg.swizzle_down)
        )
        row = evaluate(cfg, lay, dv, rf, args)
        rows.append(row)
        if row["us"] is None:
            log(f"{row['label']:<18} {'-':>9} {'-':>7} {'-':>10}  {row['note']}")
        else:
            log(
                f"{row['label']:<18} {row['us']:9.1f} {row['gbs']:7.0f} "
                f"{row['rel']:10.3e}  {row['grid']}"
            )

    ok = [r for r in rows if r["us"] is not None]
    if args.json_out:
        import json as _json

        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            _json.dumps(
                {
                    "shape": {
                        "name": shape.name,
                        "tokens": shape.tokens,
                        "experts": shape.experts,
                        "topk": shape.topk,
                        "hidden": shape.hidden,
                        "intermediate": shape.intermediate,
                    },
                    "tol": args.tol,
                    "iters": args.iters,
                    "rows": rows,
                },
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    if ok:
        base = next((r for r in ok if r["label"] == "default"), None)
        ok.sort(key=lambda r: r["us"])
        log("\nranked:")
        for r in ok:
            d = f"   {base['us'] / r['us']:.3f}x vs default" if base else ""
            log(f"  {r['label']:<18} {r['us']:9.1f} us  {r['gbs']:6.0f} GB/s{d}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
