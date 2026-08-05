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

    PYTHONPATH=<rocke platform python> python -u moe_mega_np.py --shape qwen3
    PYTHONPATH=<rocke platform python> python -u moe_mega_np.py --shape qwen3 --sweep
"""

from __future__ import annotations

import argparse
import ctypes
import fnmatch
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GROUP_K = 128  # block-scale group width, on both weight axes and the activation
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


SHAPES = {
    # Qwen3-30B-A3B decode: hidden and expert count come off the traced operand
    # shapes; top-8 off the incumbent topkGating template arguments.
    "qwen3": Shape("qwen3", 32, 128, 8, 2048, 768),
    # The shape the shipped mega-kernel levers were tuned against.
    # Same work, but the gate/up weight row stride is no longer a power of two
    # (2176 = 17*128). Probes whether the 2048B stride is aliasing DRAM
    # banks/channels and costing sustained read rate.
    "qwen3_h2176": Shape("qwen3_h2176", 32, 128, 8, 2176, 768),
    "qwen3_h2560": Shape("qwen3_h2560", 32, 128, 8, 2560, 768),
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
    def __init__(self, layout: Layout):
        from rocke.runtime.host_buffers import as_u8_buffer
        from rocke.runtime.hip_module import Runtime
        from rocke.runtime.launcher import DeviceMem

        self.rt = Runtime()
        self._keep: list = []
        w, s = layout.w, layout.shape

        def upload(arr: np.ndarray):
            arr = np.ascontiguousarray(arr)
            self._keep.append(arr)
            mem = DeviceMem(arr.nbytes)
            self.rt.memcpy_h2d(mem.ptr(), as_u8_buffer(arr), arr.nbytes)
            self._keep.append(mem)
            return mem

        H, I, nHb, nIb = s.hidden, s.intermediate, w.nHb, w.nIb
        self.y_shape = (s.tokens, H)
        self.y_nbytes = s.tokens * H * 4
        self.Y = DeviceMem(self.y_nbytes)
        self._keep.append(self.Y)

        self.values = {
            "A": upload(layout.A_q),
            "WGate": upload(w.Wg_q),
            "WUp": upload(w.Wu_q),
            "WDown": upload(w.Wd_q),
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
    sched_cadence: "str | None" = None
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

    def spec(self, PROD):
        return PROD.FusedMegaKernelSpecFp8(
            name=f"np_{self.label}",
            tile_m=self.tile_m,
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
    """Third pass: cross the geometry knobs with the DTLA prefetch depth.

    Depth was only ever swept at the two geometries that happened to be
    leading, and it interacts with them: the useful depth is bounded by the
    N-cell count (``tile_n_inter / warp_n / 32``), and its LDS cost trades
    against CTAs/CU, whose product with the in-flight cell count is what
    actually sets the memory-level parallelism. So the peak depth is expected
    to move with the tile shape rather than being a constant.
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
        # NOTE: no use_dtla=False variant here -- DTLA off at the K=128 hero
        # atom hangs the Comgr backend (register allocator pathology on the
        # long live ranges), so it can only be paired with gate_up_k=32.
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
        # Legacy per-cell gate/up emitter: weights global->VGPR, one cell live
        # at a time. The down GEMM (which already has this shape) sustains
        # ~1268 GB/s while the DTLA gate/up phase caps at ~937.
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


def evaluate(cfg: Config, layout: Layout, dev: DeviceProblem, ref, args) -> dict:
    from rocke.helpers.compile import compile_kernel
    from rocke.instances.common import moe_fused_mega_fp8 as PROD
    from rocke.runtime.launcher import KernelLauncher, LaunchConfig, time_launches

    row = {"label": cfg.label, "us": None, "note": ""}
    spec = cfg.spec(PROD)

    t = time.time()
    try:
        with build_budget(args.build_budget):
            kd = PROD.build_moe_fused_mega_gemm_fp8(
                spec, arch="gfx950", persistent=cfg.persistent
            )
            art = compile_kernel(kd, arch="gfx950", capture_ir_text=False)
    except TimeoutError:
        row["note"] = f"BUILD_SLOW (>{args.build_budget:.0f}s lowering)"
        return row
    except Exception as exc:  # noqa: BLE001 - a config that won't build is data
        row["note"] = f"BUILD_FAIL {type(exc).__name__}: {str(exc)[:50]}"
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

    try:
        dev.zero_y()
        launcher(values, config=lcfg)
        dev.rt.sync()
        Y = dev.read_y()
    except Exception as exc:  # noqa: BLE001
        row["note"] = f"LAUNCH_FAIL {type(exc).__name__}: {str(exc)[:50]}"
        return row

    if args.phase == "full":
        rel = float(np.abs(Y - ref).max() / (np.abs(ref).max() + 1e-9))
        row["rel"] = rel
        if not (rel < args.tol and np.isfinite(Y).all()):
            row["note"] = (
                f"PARITY_FAIL rel={rel:.3e} nan={int(np.isnan(Y).sum())} "
                f"ymax={float(np.abs(Y).max()):.4g} "
                f"refmax={float(np.abs(ref).max()):.4g}"
            )
            return row
    else:
        row["rel"] = float("nan")

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

    configs = sweep_configs() if args.sweep else [Config("default", tile_m=args.tile_m)]
    if args.configs:
        want = [c.strip() for c in args.configs.split(",")]
        configs = [
            c for c in configs if any(fnmatch.fnmatch(c.label, w) for w in want)
        ]

    # Layout + oracle depend only on tile_m, so build one per distinct tile_m.
    cache: dict = {}

    ref_cache: dict = {}

    def state_for(tile_m: int, hgk: int = GROUP_K):
        if tile_m not in cache:
            with step(f"layout tile_m={tile_m}"):
                lay = Layout(weights, tile_m)
            log(
                f"       {shape.tokens * shape.topk} token-slots -> "
                f"{lay.num_m_blocks} blocks, {lay.total_padded} padded rows "
                f"({lay.total_padded / (shape.tokens * shape.topk):.1f}x real work), "
                f"{lay.active_experts()}/{shape.experts} experts active"
            )
            with step(f"upload tile_m={tile_m}"):
                dv = DeviceProblem(lay)
            cache[tile_m] = (lay, dv)
        lay, dv = cache[tile_m]
        if (tile_m, hgk) not in ref_cache:
            with step(f"numpy oracle tile_m={tile_m} hgk={hgk}"):
                lay.hidden_group_k = hgk
                ref_cache[(tile_m, hgk)] = lay.reference()
                lay.hidden_group_k = GROUP_K
        return lay, dv, ref_cache[(tile_m, hgk)]

    log(f"\n{'config':<18} {'us':>9} {'GB/s':>7} {'rel':>10}  grid / note")
    rows = []
    for cfg in configs:
        lay, dv, rf = state_for(cfg.tile_m, cfg.hidden_group_k)
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
