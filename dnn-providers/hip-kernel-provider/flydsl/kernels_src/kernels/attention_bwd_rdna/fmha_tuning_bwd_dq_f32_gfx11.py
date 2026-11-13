# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""Knob record and schedule policy of the gfx11 fp32 (VALU) backward dQ (and dB) kernel `fmha_bwd_dq_f32_gfx11`.

Entry point, the call `tools/variant_gfx11.build` makes:

    bwd_dq_knobs(arch, dtype, tile, cls, dropout, padded, hdim_qk, hdim_vo, **pins) -> BwdDqF32KnobsGfx11

`pins` are the knobs the Spec sets, spelled as in the fp32 generic knob table (`variant.KNOBS_GENERIC_F32["dq"]`): `num_waves`, `block_m`, `block_n`,
`waves_per_eu`, `store_db` (0), `lpt_tile_order` (0), `dump` (1). An unpinned knob takes the policy below. The record is frozen; `legal` is False (with
`why` naming the rule) for a configuration the Spec cannot see to be wrong, and `variant_gfx11.build` raises `ValueError` for it.

Geometry (AOTriton's fp32 `bwd_kernel_dq`, VALU; no WMMA on RDNA for fp32): one workgroup owns `block_m` query rows of one head and loops the kv axis in
`block_n`-row tiles.

    T        = 32 * num_waves threads
    pairs    = block_m * block_n / T      (q, kv) score pairs per thread (S, dP, P, dS; one writer per pair, so one writer per dB element)
    TPR      = T / block_m                threads per owner row; the dQ accumulators are `tile * block_m / T` registers per thread
    D_CHUNK  = 32 when tile % 32 == 0, else 16     head-dim slab width staged through LDS (pitch D_CHUNK + 1, fp32)

The grid is AOTriton's fp32 tile set: (block_m, block_n) in {(16, 16), (32, 16), (32, 32)} x num_waves in {4, 8} (`block_m >= block_n`).

Schedule policy (`POLICY`): the sweep's fp32 schedule measurement on gfx1101, the geometric mean of the
dQ skeleton's time over S 256..2048 at hd 64 / 128 / 256 (B 2, H 8):

    hd 64   m32n16w4 1.18 ms  (next m32n32w8 1.45)
    hd 128  m16n16w4 2.35 ms  (next m32n16w8 2.65)
    hd 256  m16n16w8 5.35 ms  (next m32n16w8 5.77)

Every other rung takes the row of the nearest measured rung (ties toward the smaller): 16..96 -> hd 64, 128..192 -> hd 128, 224..256 -> hd 256.
These are skeleton timings, not the object's: a per-row timing run on the real objects replaces them.
`waves_per_eu` is unset by default (a lever: 1..4); `lpt_tile_order` is on for causal.
"""

from dataclasses import dataclass

LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256)
LDS_CAP_BYTES = 65536
WARP_SIZE = 32
ELEM_BYTES = 4
NUM_WAVES = (4, 8)
BLOCK_SIZES = (16, 32)

_MEASURED = {
    64: (32, 16, 4),
    128: (16, 16, 4),
    256: (16, 16, 8),
}  # rung -> (block_m, block_n, num_waves), measured fp32 dQ winners
_NEAREST = {
    16: 64,
    32: 64,
    48: 64,
    64: 64,
    80: 64,
    96: 64,
    128: 128,
    160: 128,
    192: 128,
    224: 256,
    256: 256,
}
POLICY = {tile: _MEASURED[_NEAREST[tile]] for tile in LADDER}

FP_MODE = "noninf"  # `fmha.FastMath` mode: keeps the +inf LSE' of empty rows

_PINS = (
    "num_waves",
    "block_m",
    "block_n",
    "waves_per_eu",
    "store_db",
    "lpt_tile_order",
    "dump",
)


@dataclass(frozen=True)
class BwdDqF32KnobsGfx11:
    arch: str
    dtype: str
    tile: int
    num_waves: int
    waves_per_eu: int  # 0 = unset
    block_m: int
    block_n: int
    lpt_tile_order: bool
    store_db: bool
    dump: bool
    fp_mode: str
    BLOCK_M: int  # = block_m (the name the harness reads from every dQ record)
    d_chunk: int
    lds_bytes: int
    legal: bool
    why: str  # the broken rule when `legal` is False, else ""

    @property
    def block_size(self):
        return self.num_waves * WARP_SIZE


def d_chunk(tile):
    """Head-dim slab width: 32, or 16 for a rung that is not a multiple of 32 (16, 48, 80)."""
    return 32 if tile % 32 == 0 else 16


def lds_bytes(tile, block_m, block_n):
    """Workgroup LDS: the Q and dO slabs (`block_m` x (D_CHUNK + 1)), the K and V slabs (`block_n` x (D_CHUNK + 1)) and dS (`block_m` x `block_n`), fp32."""
    dcp = d_chunk(tile) + 1
    return ELEM_BYTES * (2 * block_m * dcp + 2 * block_n * dcp + block_m * block_n)


def bwd_dq_knobs(arch, dtype, tile, cls, dropout, padded, hdim_qk, hdim_vo, **pins):
    """The frozen knob record for one (functional, pins). See the module docstring."""
    bad = set(pins) - set(_PINS)
    if bad:
        raise TypeError(f"unknown dq knob pin(s) {sorted(bad)}; the knobs are {_PINS}")
    why = []
    if not arch.startswith(("gfx11", "gfx12")):
        why.append(f"arch {arch!r} is not a gfx11/gfx12 target")
    if dtype != "fp32":
        why.append(f"dtype {dtype!r} is not fp32")
    if tile not in LADDER:
        why.append(f"tile {tile} is not a rung of {LADDER}")
    if cls not in ("plain", "causal", "bias"):
        why.append(f"class {cls!r}")
    base = POLICY.get(tile, POLICY[64])
    bm = int(pins.get("block_m", base[0]))
    bn = int(pins.get("block_n", base[1]))
    nw = int(pins.get("num_waves", base[2]))
    if bm not in BLOCK_SIZES:
        why.append(f"block_m {bm} is not one of {BLOCK_SIZES}")
    if bn not in BLOCK_SIZES:
        why.append(f"block_n {bn} is not one of {BLOCK_SIZES}")
    if bm < bn:
        why.append(
            f"block_m {bm} < block_n {bn}: the fp32 tiles have block_m >= block_n"
        )
    if nw not in NUM_WAVES:
        why.append(f"num_waves {nw} is not one of {NUM_WAVES}")
    has_bias = cls == "bias"
    store_db = bool(pins.get("store_db", 1 if has_bias else 0))
    if store_db and not has_bias:
        why.append("store_db applies to bias builds only")
    wpe = int(pins.get("waves_per_eu", 0))
    if wpe not in (0, 1, 2, 3, 4):
        why.append(f"waves_per_eu {wpe} is not 1..4")
    lds = lds_bytes(tile, bm, bn) if not why else 0
    if not why and lds > LDS_CAP_BYTES:
        why.append(f"LDS {lds} B over the {LDS_CAP_BYTES} B workgroup cap")
    return BwdDqF32KnobsGfx11(
        arch=arch,
        dtype=dtype,
        tile=tile,
        num_waves=nw,
        waves_per_eu=wpe,
        block_m=bm,
        block_n=bn,
        lpt_tile_order=bool(pins.get("lpt_tile_order", 1)),
        store_db=store_db,
        dump=bool(pins.get("dump", 0)),
        fp_mode=FP_MODE,
        BLOCK_M=bm,
        d_chunk=d_chunk(tile) if tile in LADDER else 0,
        lds_bytes=lds,
        legal=not why,
        why="; ".join(why),
    )
