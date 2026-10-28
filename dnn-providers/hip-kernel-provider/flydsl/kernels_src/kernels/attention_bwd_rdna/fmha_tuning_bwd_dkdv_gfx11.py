# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.
#
# Provenance carried over from the sweep snapshot:
# Re-expressed from AOTriton's `fmha_tuning_bwd_dkdv_gfx1201.py` (`lds_bytes`, `_fits_lds`, `default_block_m` walk-down, the
# `BLOCK_N = 16 * num_waves / team` geometry), which is MIT-licensed:
#
# AOTriton's notice (MIT):
#
# Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The schedule policy is the sweep's initial register-pressure policy (skeleton builds for gfx11-generic), re-measured on the
# objects; the knob record is the one of the sweep's gfx11 build contract.

"""Tuning policy and knob record of the gfx11 / gfx12 dK/dV backward kernel (`fmha_bwd_dkdv_gfx11`).

Imports nothing from FlyDSL. The kernel is the transposed loop of the gfx1201 one with the gfx11 WMMA operand width (v16), so the
register and LDS arithmetic differs from `fmha_tuning_bwd_dkdv_gfx1201`:

* A team of waves owns 16 KV rows: 1 wave unsplit, 2 under `split_head_dim` (the K role makes S and accumulates dK, the V role
  makes dP and accumulates dV), `2 * contraction_shards` when the contraction is sharded as well. `BLOCK_KV = 16 * num_waves / team`.
* Role-owned residency. A K-role wave holds the K pack of its shard (`hd / C` columns) and the dK accumulator of the same columns, a
  V-role wave the V pack and the dV accumulator of its shard. S and dP partials reduce across shards through LDS and P / dS reach both
  roles through LDS (every wave of a team computes the same elementwise P and dS). Per wave `hd / C` columns of pack and accumulator,
  `hd / C` VGPRs (v16: a 16-column pack is 8 VGPRs, a 16-column f32 accumulator 8). The only column constraint is that `hd / C` is a
  multiple of 16; gfx1201's `D_STEPS % TEAM == 0` is not required here (it spreads both accumulators over the whole team).
* LDS: Q and dO row-major, Q^T and dO^T (`transposed_source="tile"`), the per-row LSE and Delta of the Q block, and the team exchange
  (one f32 slot of 32 lanes x 8 values per wave, a second buffer when the Q block holds two 16-row sub-tiles).
"""

from dataclasses import dataclass

# One wave owns one WMMA M tile of KV rows. Not a knob: it is the instruction.
ROWS_PER_WAVE = 16
LDS_LIMIT = 65536
WAVE_LANES = 32
# LDS row padding, in 16-bit elements: the value the register-pressure skeletons were measured with.
LDS_PAD = 4
# LDS element index granularity of every base (a base is a multiple of 8 elements = 16 B).
_ALIGN = 8

GENERIC_ARCHES = ("gfx11-generic", "gfx12-generic")
LADDER = (16, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256)

# (split_head_dim, contraction_shards, num_waves, block_q) per hd tile: the initial register-pressure policy, re-measured at
# every gate. hd 64 is the schedule trial's winner (gfx1101: 0.86x of 16 waves; its 12 B causal-f16
# spill does not show in time); hd 128 keeps one shard pair (two shards measured 1.56x slower).
POLICY = {
    16: (False, 1, 4, 32),
    32: (False, 1, 4, 32),
    48: (False, 1, 4, 16),
    64: (False, 1, 4, 16),
    80: (False, 1, 8, 32),
    96: (True, 1, 8, 16),
    128: (True, 1, 8, 16),
    160: (True, 2, 16, 16),
    192: (True, 2, 16, 16),
    224: (True, 2, 16, 16),
    256: (True, 2, 16, 16),
}

SCHED_STRATEGIES = (
    "max-ilp",
    "iterative-ilp",
    "iterative-minreg",
    "iterative-maxocc",
    "default",
)
_DEFAULT_SCHED = "max-memory-clause"
_TRANSPOSED_SOURCES = ("tile", "derived")
_DTYPES = ("f16", "bf16")


def team_waves(split_head_dim, contraction_shards):
    return 2 * contraction_shards if split_head_dim else 1


def _up(n, m):
    return (n + m - 1) // m * m


@dataclass(frozen=True)
class LdsLayout:
    """Element offsets (16-bit elements) of the dK/dV LDS block, and its extent. Shared by the kernel and `lds_bytes`, so the
    legality check and the allocation cannot drift apart."""

    qrm_base: int
    qrm_stride: int
    qtr_base: int
    qtr_stride: int
    dorm_base: int
    dorm_stride: int
    dotr_base: int
    dotr_stride: int
    rowq_base: int  # LSE (f32 x block_q) then Delta (f32 x block_q), as 2 * block_q f32 = 4 * block_q elements
    pds_base: int  # team exchange; f32 slots, `pds_bufs` buffers of `num_waves` slots of 32 lanes x 8 values
    pds_bufs: int
    pds_slot_f32: int
    total: int  # elements


def lds_layout(block_q, hd, hdv, num_waves, split_head_dim, transposed_source):
    qrm_stride, dorm_stride = hd + LDS_PAD, hdv + LDS_PAD
    qtr_stride = dotr_stride = block_q + LDS_PAD
    keep_tr = transposed_source == "tile"
    qrm_base = 0
    qtr_base = qrm_base + block_q * qrm_stride
    dorm_base = qtr_base + (hd * qtr_stride if keep_tr else 0)
    dotr_base = dorm_base + block_q * dorm_stride
    rowq_base = _up(dotr_base + (hdv * dotr_stride if keep_tr else 0), _ALIGN)
    pds_base = _up(rowq_base + 4 * block_q, _ALIGN)
    pds_bufs = 2 if (split_head_dim and block_q > 16) else 1
    pds_slot_f32 = WAVE_LANES * 8
    pds_elems = pds_bufs * num_waves * pds_slot_f32 * 2 if split_head_dim else 0
    return LdsLayout(
        qrm_base,
        qrm_stride,
        qtr_base,
        qtr_stride,
        dorm_base,
        dorm_stride,
        dotr_base,
        dotr_stride,
        rowq_base,
        pds_base,
        pds_bufs,
        pds_slot_f32,
        pds_base + pds_elems,
    )


def lds_bytes(
    block_q,
    hd,
    hdv,
    num_waves,
    split_head_dim=False,
    transposed_source="tile",
    elem_bytes=2,
):
    """LDS the kernel allocates, in bytes (`LdsLayout.total` times the element size)."""
    return (
        lds_layout(block_q, hd, hdv, num_waves, split_head_dim, transposed_source).total
        * elem_bytes
    )


def _fits_lds(block_q, hd, hdv, num_waves, split_head_dim, transposed_source):
    return (
        lds_bytes(block_q, hd, hdv, num_waves, split_head_dim, transposed_source)
        <= LDS_LIMIT
    )


def default_block_q(want, hd, hdv, num_waves, split_head_dim, transposed_source):
    """Q rows per staging pass: the policy's choice, walked down to 16 when it does not fit LDS (16 is one WMMA tile)."""
    for bq in (want, 16):
        if _fits_lds(bq, hd, hdv, num_waves, split_head_dim, transposed_source):
            return bq
    return 16


@dataclass(frozen=True)
class BwdDkDvKnobs:
    """The resolved configuration of one dK/dV object (the record `bwd_dkdv_knobs` returns)."""

    num_waves: int
    waves_per_eu: int | None
    block_q: int
    split_head_dim: bool
    contraction_shards: int
    transposed_source: str  # "tile" | "derived"
    sched_strategy: (
        str | None
    )  # None = the compiler's default (Spec spelling `default`)
    dump: bool
    rne: bool  # bf16 P / dS narrowing: round-to-nearest-even (True) or the legacy round-toward-zero truncation (False)
    block_dmodel: int  # compiled tile widths (hdim_qk and hdim_vo of an exact object)
    block_dmodel_v: int
    padded_head: bool
    team_waves: int
    block_kv: int  # BLOCK_KV = 16 * num_waves / team_waves
    lds_bytes: int
    legal: bool
    illegal_reason: str = ""


def bwd_dkdv_knobs(arch, dtype, tile, cls, dropout, padded, hdim_qk, hdim_vo, **pins):
    """Resolve the policy for `tile`, apply `pins` (spelled as in `variant.KNOBS_GENERIC`; only the knobs a Spec sets), derive the rest.

    A pinned `split_head_dim` / `contraction_shards` changes the configuration class; an unpinned `num_waves` is then the policy's
    rounded up to a whole number of teams. `block_q` is the policy's walked down until LDS fits. `legal` is false for a record that
    breaks a rule the Spec cannot see (LDS over 65,536 B, `BLOCK_KV` or a column slice not a multiple of 16).
    """
    if arch not in GENERIC_ARCHES:
        raise ValueError(f"arch {arch!r} is not a generic arch")
    if dtype not in _DTYPES:
        raise ValueError(f"dtype {dtype!r} is not one of {_DTYPES}")
    unknown = set(pins) - {
        "num_waves",
        "block_q",
        "waves_per_eu",
        "split_head_dim",
        "contraction_shards",
        "transposed_source",
        "sched_strategy",
        "dump",
        "rne",
    }
    if unknown:
        raise ValueError(f"unknown dkdv knob(s) {sorted(unknown)}")
    if tile not in POLICY:
        raise ValueError(f"no dK/dV policy for tile {tile}")
    p_split, p_shards, p_waves, p_bq = POLICY[tile]

    split = bool(pins["split_head_dim"]) if "split_head_dim" in pins else p_split
    shards = (
        int(pins["contraction_shards"])
        if "contraction_shards" in pins
        else (p_shards if split else 1)
    )
    team = team_waves(split, shards)
    waves = int(pins["num_waves"]) if "num_waves" in pins else _up(p_waves, team)
    tsrc = pins.get("transposed_source", "tile")
    sched = pins.get("sched_strategy", _DEFAULT_SCHED)
    if sched not in SCHED_STRATEGIES and sched != _DEFAULT_SCHED:
        raise ValueError(f"sched_strategy {sched!r}")
    sched = None if sched == "default" else sched
    hd, hdv = hdim_qk, hdim_vo
    bq = (
        int(pins["block_q"])
        if "block_q" in pins
        else default_block_q(p_bq, hd, hdv, waves, split, tsrc)
    )

    why = []
    if tsrc not in _TRANSPOSED_SOURCES:
        why.append(f"transposed_source {tsrc!r}")
    if not split and shards != 1:
        why.append(f"contraction_shards {shards} needs split_head_dim")
    if shards < 1 or shards & (shards - 1):
        why.append(f"contraction_shards {shards} is not a power of two")
    if split and hd != hdv:
        why.append(f"split_head_dim needs hdim_qk == hdim_vo (got {hd}, {hdv})")
    if split and (hd % 16 or (hd // 16) % shards):
        why.append(f"hd {hd} / contraction_shards {shards} is not a multiple of 16")
    if waves < 1 or waves % team:
        why.append(f"num_waves {waves} is not a multiple of the {team}-wave team")
    if bq < 16 or bq % 16:
        why.append(f"block_q {bq} is not a multiple of 16")
    block_kv = (
        ROWS_PER_WAVE * (waves // team) if waves >= 1 and waves % team == 0 else 0
    )
    lds = (
        lds_bytes(
            bq, hd, hdv, waves, split, tsrc if tsrc in _TRANSPOSED_SOURCES else "tile"
        )
        if bq >= 16 and waves >= 1
        else 0
    )
    if lds > LDS_LIMIT:
        why.append(f"LDS {lds} B over the {LDS_LIMIT} B cap")
    return BwdDkDvKnobs(
        num_waves=waves,
        waves_per_eu=(int(pins["waves_per_eu"]) if "waves_per_eu" in pins else None),
        block_q=bq,
        split_head_dim=split,
        contraction_shards=shards,
        transposed_source=tsrc,
        sched_strategy=sched,
        dump=bool(pins.get("dump", 0)),
        rne=bool(pins.get("rne", 0)),
        block_dmodel=tile,
        block_dmodel_v=hdim_vo if hdim_vo else tile,
        padded_head=bool(padded) or hd != tile or hdv != tile,
        team_waves=team,
        block_kv=block_kv,
        lds_bytes=lds,
        legal=not why,
        illegal_reason="; ".join(why),
    )
