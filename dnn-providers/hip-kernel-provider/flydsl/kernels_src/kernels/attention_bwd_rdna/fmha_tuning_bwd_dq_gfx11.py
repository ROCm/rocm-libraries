# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); its sources are listed below.
#
# Provenance carried over from the sweep snapshot:
# Adapted from: https://github.com/ROCm/aiter
#   commit:   8253efc4059516f492df597b80ce5311e9fda8bd
#   path:     aiter/ops/flydsl/kernels/flash_attn_func_gfx1201.py
# via the gfx11 forward `attention_rdna/flash_attn_func_rdna.py` (ROCm/rocm-libraries PR #14111 @ d850254d45e).
#
# `lds_bytes` and `_fits_lds` re-express the LDS accounting of the AOTriton gfx1201 backward-dQ kernel
# (`fmha_bwd_dq_gfx1201_kernel.py`, MIT; reference only):
#   Copyright (c) 2024-2025 Advanced Micro Devices, Inc. / AOTriton contributors, MIT License.

"""Knob record and schedule policy of the gfx11 / gfx12 generic-target backward dQ (and dB) kernel.

Entry point, the call `tools/variant_gfx11.build` makes:

    bwd_dq_knobs(arch, dtype, tile, cls, dropout, padded, hdim_qk, hdim_vo, **pins) -> BwdDqKnobsGfx11

`pins` are the knobs the Spec sets, spelled as in `variant.KNOBS_GENERIC["dq"]`: `num_waves`, `block_m`, `block_n`, `waves_per_eu`,
`store_db` (0), `bias_stage` ("direct"), `contraction_shards`, `kt_lds_layout` ("staged"), `lpt_tile_order` (0), `dump` (1). An unpinned knob takes
the policy below; pins take part in deriving the knobs that depend on them (a pinned `contraction_shards` or `num_waves` moves `block_m`).
The record is frozen; `legal` is False (with `why` naming the rule) for a configuration the Spec cannot see to be wrong, and `variant_gfx11.build`
raises `ValueError` for it, so no illegal object is built.

Geometry (a wave owns 16 query rows of one head-dim shard; no role split, unlike dK/dV):

    BLOCK_M  = 16 * num_waves / contraction_shards      query rows per workgroup (`block_m` is the same number as an input)
    slice    = tile / contraction_shards                  head-dim columns one wave holds of Q, dO and the dQ accumulator (a multiple of 16)
    The S and dP partials of the shards of one 16-row tile are summed through LDS (gfx1201's `reduce_s_across_shards`); dQ stores are disjoint.

Schedule policy: the sweep's initial policy (register-pressure skeletons of this kernel's loop), one
row per rung: (shards Qs, num_waves, block_n). Every row is re-measured on the real objects and may change; `num_waves` and `block_n` are gfx1201's measured prior re-expressed for v16 operands, not confirmed for performance.
`waves_per_eu` 2, `kt_lds_layout` scalar and `lpt_tile_order` on are gfx1201's defaults.
"""

from dataclasses import dataclass

LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256)
LDS_CAP_BYTES = 65536
ROWS_PER_WAVE = 16
WARP_SIZE = 32
ELEM_BYTES = 2
LDS_PAD = 4  # K, V and K^T row padding (elements): K_STRIDE = tile + 4, KT_STRIDE = block_n + 4
MAX_WAVES = 16  # a 512-thread workgroup; the register-pressure skeletons stop here

# tile -> (contraction_shards, num_waves, block_n)
POLICY = {
    16: (1, 16, 32),
    32: (1, 8, 32),
    48: (1, 8, 32),
    64: (1, 8, 32),
    80: (1, 8, 32),
    96: (1, 8, 32),
    128: (
        2,
        8,
        16,
    ),  # schedule trial (gfx1101): 0.89x of Qs 1 on plain and causal; bias re-measured after the dB-writer fix
    160: (2, 8, 16),
    192: (2, 16, 16),
    224: (2, 16, 16),
    256: (4, 16, 16),
}
DEFAULT_WAVES_PER_EU = 2
DEFAULT_KT_LDS_LAYOUT = "scalar"
KT_LDS_LAYOUTS = ("scalar", "staged")
BIAS_STAGES = (
    "direct",
)  # per-element global bias loads; the bias is not staged through LDS
FP_MODE = "noninf"  # `fmha.FastMath` mode: keeps the -inf of masked scores and the +inf LSE' of empty rows

_PINS = (
    "num_waves",
    "block_m",
    "block_n",
    "waves_per_eu",
    "store_db",
    "bias_stage",
    "contraction_shards",
    "kt_lds_layout",
    "lpt_tile_order",
    "dump",
    "rne",
)


@dataclass(frozen=True)
class BwdDqKnobsGfx11:
    arch: str
    dtype: str
    tile: int
    num_waves: int
    waves_per_eu: int
    block_m: int
    block_n: int
    contraction_shards: int
    kt_lds_layout: str
    lpt_tile_order: bool
    store_db: bool
    bias_stage: str  # "" (no bias input) or "direct" (per-element bias loads: no alignment or stride contract, no `bias_chunk_contract_ok`)
    dump: bool
    rne: bool  # bf16 dS narrowing: round-to-nearest-even (True) or the legacy round-toward-zero truncation (False)
    fp_mode: str
    BLOCK_M: int  # 16 * num_waves / contraction_shards
    slice: int  # tile / contraction_shards
    lds_bytes: int
    legal: bool
    why: str  # the broken rule when `legal` is False, else ""

    @property
    def block_size(self):
        return self.num_waves * WARP_SIZE

    @property
    def num_s_accs(self):
        return self.block_n // 16


def lds_bytes(tile, num_waves, shards, block_n, kt_lds_layout):
    """Workgroup LDS of the kernel: the K and V tiles (row-major, `tile + 4` per row), the staged K^T tile (`kt_lds_layout == "staged"`), the
    cross-shard S/dP reduction buffer. The reduction buffer holds one private slot of `2 * (block_n / 16) * 8` f32 per
    lane per wave; it lives in the V tile's storage when it fits there (V is dead once GEMM2 is done and K is not), else after the other
    regions. No reduction (one shard) allocates none."""
    k_tile = block_n * (tile + LDS_PAD)
    v_tile = block_n * (tile + LDS_PAD)
    kt_tile = tile * (block_n + LDS_PAD) if kt_lds_layout == "staged" else 0
    red = 0
    if shards > 1:
        red_bytes = num_waves * 2 * (block_n // 16) * 8 * WARP_SIZE * 4
        red = 0 if red_bytes <= v_tile * ELEM_BYTES else red_bytes
    return (k_tile + v_tile + kt_tile) * ELEM_BYTES + red


def _fits_lds(*args, **kwargs):
    return lds_bytes(*args, **kwargs) <= LDS_CAP_BYTES


def reduction_aliases_v(tile, num_waves, shards, block_n):
    """Whether the cross-shard reduction buffer shares the V tile's storage (`shards == 1` has no reduction, trivially True)."""
    return (
        shards == 1
        or num_waves * 2 * (block_n // 16) * 8 * WARP_SIZE * 4
        <= block_n * (tile + LDS_PAD) * ELEM_BYTES
    )


def bwd_dq_knobs(arch, dtype, tile, cls, dropout, padded, hdim_qk, hdim_vo, **pins):
    """The frozen knob record for one (functional, pins). See the module docstring."""
    bad = set(pins) - set(_PINS)
    if bad:
        raise TypeError(f"unknown dq knob pin(s) {sorted(bad)}; the knobs are {_PINS}")
    why = []
    if not arch.startswith(("gfx11", "gfx12")):
        why.append(f"arch {arch!r} is not a gfx11/gfx12 target")
    if dtype not in ("f16", "bf16"):
        why.append(f"dtype {dtype!r} is not f16 or bf16")
    if tile not in LADDER:
        why.append(f"tile {tile} is not a rung of {LADDER}")
    if cls not in ("plain", "causal", "bias"):
        why.append(f"class {cls!r}")
    base = POLICY.get(tile, (1, 8, 32))
    shards = int(pins.get("contraction_shards", base[0]))
    if shards < 1 or shards & (shards - 1):
        why.append(f"contraction_shards {shards} is not a power of two")
        shards = 1
    block_n = int(pins.get("block_n", base[2]))
    if block_n not in (16, 32):
        why.append(f"block_n {block_n} is not 16 or 32")
    if "num_waves" in pins:
        nw = int(pins["num_waves"])
    elif "block_m" in pins:
        nw = int(pins["block_m"]) // ROWS_PER_WAVE * shards
    else:
        nw = base[1]
    if nw < 1 or nw > MAX_WAVES or nw % shards:
        why.append(
            f"num_waves {nw} must be a multiple of contraction_shards {shards} and at most {MAX_WAVES}"
        )
    block_m = ROWS_PER_WAVE * (nw // shards)
    if "block_m" in pins and int(pins["block_m"]) != block_m:
        why.append(
            f"block_m {pins['block_m']} must be 16 * num_waves ({nw}) / contraction_shards ({shards}) = {block_m}"
        )
    if tile % shards or (tile // shards) % 16:
        why.append(
            f"contraction_shards {shards} leaves a {tile / shards:g}-column slice at tile {tile}: it must be a multiple of 16"
        )
    kt = pins.get("kt_lds_layout", DEFAULT_KT_LDS_LAYOUT)
    if kt not in KT_LDS_LAYOUTS:
        why.append(f"kt_lds_layout {kt!r} is not one of {KT_LDS_LAYOUTS}")
    has_bias = cls == "bias"
    stage = pins.get("bias_stage", "direct" if has_bias else "")
    if stage not in ("", *BIAS_STAGES) or (stage and not has_bias):
        why.append(
            f"bias_stage {stage!r} applies to bias builds only and is one of {BIAS_STAGES}"
        )
    store_db = bool(pins.get("store_db", 1 if has_bias else 0))
    if store_db and not has_bias:
        why.append("store_db applies to bias builds only")
    wpe = int(pins.get("waves_per_eu", DEFAULT_WAVES_PER_EU))
    lds = lds_bytes(tile, nw, shards, block_n, kt) if not why else 0
    if not why and lds > LDS_CAP_BYTES:
        why.append(f"LDS {lds} B over the {LDS_CAP_BYTES} B workgroup cap")
    return BwdDqKnobsGfx11(
        arch=arch,
        dtype=dtype,
        tile=tile,
        num_waves=nw,
        waves_per_eu=wpe,
        block_m=block_m,
        block_n=block_n,
        contraction_shards=shards,
        kt_lds_layout=kt,
        lpt_tile_order=bool(pins.get("lpt_tile_order", 1)),
        store_db=store_db,
        bias_stage=stage,
        dump=bool(pins.get("dump", 0)),
        rne=bool(pins.get("rne", 0)),
        fp_mode=FP_MODE,
        BLOCK_M=block_m,
        slice=tile // shards if shards else tile,
        lds_bytes=lds,
        legal=not why,
        why="; ".join(why),
    )
