# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); its sources are listed below.
#
# Provenance carried over from the sweep snapshot:
# Geometry and schedule policy of the VALU fp32 dK/dV kernel (`fmha_bwd_dkdv_f32_gfx11`). The tile pairs and the wave counts are AOTriton's fp32 tiles
# (`BLOCK_M, BLOCK_N` in {16, 32}, `block_m >= block_n`, wave32 `num_warps` {4, 8}); the policy rows are the sweep's measured
# fp32 schedule timings on gfx1101, and the knob record is the one of the sweep's gfx11 build.

"""Tuning policy, knob record and thread maps of the gfx11 fp32 (VALU) dK/dV backward kernel (`fmha_bwd_dkdv_f32_gfx11`).

Imports nothing from FlyDSL. Every index map the kernel emits (`stage_rc`, `pair_mn`, `owner_of`, the LDS offsets of `lds_layout`, the chunking of the head
dim) is defined here in plain integer arithmetic (`//`, `%`, `+`, `*`), so the kernel evaluates it on FlyDSL integers and
`tools/test_fp32_valu_model.py` on numpy arrays: the permanent CPU oracle runs the same maps the object does.

Algorithm (the fp32 VALU skeleton, as AOTriton's `bwd_kernel_dk_dv`). One workgroup of `T = 32 num_waves` threads owns `block_kv` kv rows of one (kv head,
sequence) and loops the Q axis in tiles of `block_q` rows (and, under GQA, over the query heads of its group; masks cut the visited Q tiles). Per loop step:

  1. S = Q K^T and dP = dO V^T: thread slot `i` holds the pair (q row `m`, kv row `n`) = `pair_mn(tid + i T)`; the head dim is walked in `chunks(tile)` slabs
     (width 32, a trailing 16 for rung 48 / 80 / ...) staged through LDS (fp32, pitch `LDS_PITCH`).
  2. P = exp2(sm_scale log2e S - lse log2e), or exp2((sm_scale S + bias - lse) log2e) with a bias, dS = P (dP - Delta): masks / dropout applied here; P (dropped, rescaled) and dS go to LDS.
  3. Owner accumulators (kv row `own_row`, `cols_per_thread` columns of each slab, registers) += over the loop tile, slab by slab: dV += P^T dO, dK += dS^T Q.

dK carries `sm_scale` once at the store. Accumulation order is fixed, so deterministic with no atomics: heads of a GQA group in order, within a head the visited Q
tiles (a causal / windowed KV tile: the tiles every row of which sees every key first, then the leading masked tiles, then the trailing masked tiles, each ascending;
a plain object: ascending), and within a tile the q rows in order.
"""

from dataclasses import dataclass

WARP_SIZE = 32
LDS_LIMIT = 65536
D_CHUNK = 32  # widest head-dim slab; a rung that is not a multiple of 32 ends in one 16-wide slab
LDS_PITCH = (
    D_CHUNK + 1
)  # row pitch (words) of a staged slab: conflict-free for a row-per-lane read
GENERIC_ARCH = "gfx11-generic"
LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256)

# (BLOCK_M, BLOCK_N) tile pairs of the grid, wave counts, `waves_per_eu` lever values.
GEOMETRIES = ((16, 16), (32, 16), (32, 32))
NUM_WAVES = (4, 8)
WAVES_PER_EU = (1, 2, 3, 4)
SCHED_STRATEGIES = (
    "max-ilp",
    "iterative-ilp",
    "iterative-minreg",
    "iterative-maxocc",
    "default",
)
_DEFAULT_SCHED = "max-memory-clause"

# (block_q, block_kv, num_waves) per hd tile: the best measured fp32 geometry of the nearest measured hd (64, 128, 256; ties to the larger), from the dK/dV
# timing measurements: hd 64 m32n16w4 (best at S >= 1024), hd 128 m16n16w4 (spill-free, best at every S), hd 256 m32n16w8 (best at every S).
# A per-row timing screen on the real objects replaces these rows.
POLICY = {
    16: (32, 16, 4),
    32: (32, 16, 4),
    48: (32, 16, 4),
    64: (32, 16, 4),
    80: (32, 16, 4),
    96: (16, 16, 4),
    128: (16, 16, 4),
    160: (16, 16, 4),
    192: (32, 16, 8),
    224: (32, 16, 8),
    256: (32, 16, 8),
}


def chunks(hd):
    """Widths of the head-dim slabs of a tile: 32 each, then a trailing 16."""
    assert hd > 0 and hd % 16 == 0, hd
    return (D_CHUNK,) * (hd // D_CHUNK) + ((16,) if hd % D_CHUNK else ())


def cols_per_thread(width, tpr):
    assert width % tpr == 0, (width, tpr)
    return width // tpr


# ---- thread maps: integer arithmetic only, shared by the kernel (FlyDSL integers) and the model (numpy arrays) ----


def owner_of(tid, tpr):
    """(owner kv row, column part) of a thread: a row is owned by `tpr` consecutive threads, which hold its columns `cpart + j tpr`."""
    return tid // tpr, tid % tpr


def stage_rc(idx, width):
    """(row, column) inside a slab of the `idx`-th staged element (consecutive threads on consecutive columns)."""
    return idx // width, idx % width


def pair_mn(slot, block_kv):
    """(q row, kv row) inside the tile of a pair slot."""
    return slot // block_kv, slot % block_kv


def pair_slot(m, n, block_kv):
    """LDS slot (within the P / dS block) of the pair (m, n)."""
    return m * block_kv + n


@dataclass(frozen=True)
class Geometry:
    block_q: int
    block_kv: int
    num_waves: int
    tile: int

    @property
    def T(self):
        return WARP_SIZE * self.num_waves

    @property
    def tpr(self):
        """Threads per owner row."""
        return self.T // self.block_kv

    @property
    def npair(self):
        """(q, kv) pairs per thread per loop step."""
        return self.block_q * self.block_kv // self.T

    @property
    def chunks(self):
        return chunks(self.tile)

    def acc_per_tensor(self):
        """Accumulators per thread of one of dK / dV: `cols_per_thread` per slab."""
        return sum(cols_per_thread(w, self.tpr) for w in self.chunks)

    def lds(self):
        return lds_layout(self.block_q, self.block_kv)


@dataclass(frozen=True)
class LdsLayout:
    """Word offsets (fp32) of the LDS block: Q, dO slabs (block_q rows), K, V slabs (block_kv rows), P and dS of the pair tile."""

    o_q: int
    o_do: int
    o_k: int
    o_v: int
    o_p: int
    o_ds: int
    total: int  # words


def lds_layout(block_q, block_kv):
    o_q = 0
    o_do = o_q + block_q * LDS_PITCH
    o_k = o_do + block_q * LDS_PITCH
    o_v = o_k + block_kv * LDS_PITCH
    o_p = o_v + block_kv * LDS_PITCH
    o_ds = o_p + block_q * block_kv
    return LdsLayout(o_q, o_do, o_k, o_v, o_p, o_ds, o_ds + block_q * block_kv)


def lds_bytes(block_q, block_kv):
    return 4 * lds_layout(block_q, block_kv).total


@dataclass(frozen=True)
class BwdDkDvKnobs:
    """The resolved configuration of one fp32 dK/dV object (the record `bwd_dkdv_knobs` returns)."""

    num_waves: int
    waves_per_eu: int | None
    block_q: int
    block_kv: int
    sched_strategy: (
        str | None
    )  # None = the compiler's default (Spec spelling `default`)
    dump: bool
    block_dmodel: int  # compiled tile widths (hdim_qk and hdim_vo of an exact object)
    block_dmodel_v: int
    padded_head: bool
    lds_bytes: int
    legal: bool
    illegal_reason: str = ""

    @property
    def geometry(self):
        return Geometry(self.block_q, self.block_kv, self.num_waves, self.block_dmodel)


def bwd_dkdv_knobs(arch, dtype, tile, cls, dropout, padded, hdim_qk, hdim_vo, **pins):
    """Resolve the policy for `tile`, apply `pins` (Spec knob names; only the knobs a Spec sets), derive the rest. `legal` is false for a record that breaks a
    rule the Spec cannot see (`block_q < block_kv`, thread count not divisible, LDS over 65,536 B, `dump`).
    """
    if arch != GENERIC_ARCH:
        raise ValueError(f"arch {arch!r} is not {GENERIC_ARCH}")
    if dtype != "fp32":
        raise ValueError(f"dtype {dtype!r} is not fp32")
    unknown = set(pins) - {
        "num_waves",
        "block_q",
        "block_kv",
        "waves_per_eu",
        "sched_strategy",
        "dump",
    }
    if unknown:
        raise ValueError(f"unknown fp32 dkdv knob(s) {sorted(unknown)}")
    if tile not in POLICY:
        raise ValueError(f"no dK/dV policy for tile {tile}")
    p_bq, p_bkv, p_waves = POLICY[tile]
    bq = int(pins.get("block_q", p_bq))
    bkv = int(pins.get("block_kv", p_bkv))
    waves = int(pins.get("num_waves", p_waves))
    sched = pins.get("sched_strategy", _DEFAULT_SCHED)
    if sched not in SCHED_STRATEGIES and sched != _DEFAULT_SCHED:
        raise ValueError(f"sched_strategy {sched!r}")
    sched = None if sched == "default" else sched
    wpe = int(pins["waves_per_eu"]) if "waves_per_eu" in pins else None

    why = []
    if pins.get("dump"):
        why.append("dump=1: the fp32 family has no dump object")
    if waves not in NUM_WAVES:
        why.append(f"num_waves {waves} is not one of {NUM_WAVES}")
    if (bq, bkv) not in GEOMETRIES:
        why.append(f"(block_q, block_kv) {(bq, bkv)} is not one of {GEOMETRIES}")
    if wpe is not None and wpe not in WAVES_PER_EU:
        why.append(f"waves_per_eu {wpe} is not one of {WAVES_PER_EU}")
    ok_geom = not why
    if ok_geom:
        T = WARP_SIZE * waves
        if T % bkv or (T // bkv) not in (4, 8, 16) or (bq * bkv) % T:
            why.append(f"{waves} waves do not divide the ({bq}, {bkv}) tile")
    lds = lds_bytes(bq, bkv)
    if lds > LDS_LIMIT:
        why.append(f"LDS {lds} B over the {LDS_LIMIT} B cap")
    return BwdDkDvKnobs(
        num_waves=waves,
        waves_per_eu=wpe,
        block_q=bq,
        block_kv=bkv,
        sched_strategy=sched,
        dump=bool(pins.get("dump", 0)),
        block_dmodel=tile,
        block_dmodel_v=hdim_vo if hdim_vo else tile,
        padded_head=bool(padded) or hdim_qk != tile or hdim_vo != tile,
        lds_bytes=lds,
        legal=not why,
        illegal_reason="; ".join(why),
    )
