# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""Tuning policy of the fp32 dK/dV backward kernel (gfx950): the per-(tile, geometry) schedule table.

The fp32 twin of `fmha_tuning_bwd_dkdv_gfx950` (same exported names and signatures, `dtype_str="fp32"`), for the native-f32-MFMA family of
`fmha_f32_common_gfx950`. What the 16-bit policy resolves from a measured table of `(num_waves, waves_per_eu, dkv_shards, mfma_rows, block_q)`
this one resolves from `SCHEDULE_TABLE`, the final schedule table of `RP/parity/gaps/fp32/sp-n.md` (SP-N rule after the SA timing, 2026-10-23):

  - the **geometry** is the Spec pin `block_q` in {16, 32} (the (BLOCK_M, BLOCK_N) = (block_q, 16) tile of the family); unpinned it is 32;
  - the **schedule** of a (tile, geometry) is `per-wave` ('pw': a wave owns 16 KV rows x the full head dim) or `D-split` ('ds': the workgroup owns
    16 KV rows, wave w owns head-dim slice w and the S / dP partials are exchanged through LDS), the one the table assigns;
  - the **wave count** is that schedule's representative variant (highest occupancy, then fewest VGPR+AGPR, then 4 waves over 2).

Neither the schedule nor the wave count is a Spec key (`parity/gaps/fp32/interfaces.md`); a Spec fixes the geometry and the table does the rest.
`mfma_rows` (16), `num_stream_buffers` (1), `dkv_shards` (1) and `head_dim_granule` (16 = one panel) have one legal value; `block_kv` is the KV rows
of a workgroup (16 for a D-split, 16 x waves for per-wave) and `waves_per_eu` is 0 (no hint: SP-N's occupancy + 1 probes reached nothing higher).

This module records the policy; which (tile, schedule, feature) combinations the kernel computes is the kernel module's scope check
(`fmha_bwd_dkdv_f32_gfx950`).
"""

from dataclasses import dataclass, fields, replace

from kernels.attention_bwd_cdna.fmha_f32_common_gfx950 import Family

__all__ = [
    "BwdDkDvInputMetadata",
    "BwdDkDvKnobs",
    "DEFAULT_BLOCK_Q",
    "FP32_LADDER",
    "ScheduleEntry",
    "SCHEDULE_TABLE",
    "bwd_dkdv_knobs",
    "schedule_entry",
    "tile_width_for",
]

# The fp32 ladder: the 16-bit ladder without 384 (hd 257..512 rides t512).
FP32_LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 512)

# Geometry of an unpinned Spec.
DEFAULT_BLOCK_Q = 32


@dataclass(frozen=True)
class ScheduleEntry:
    """One (tile, geometry) row of the final schedule table of sp-n.md.

    `final`: the dK/dV schedule the SA rule assigns ('pw' | 'ds'); `pw_waves` / `ds_waves`: wave count of that schedule's representative variant
    (None: no variant qualifies, t512 per-wave); `dq_waves`: the dQ representative (F.Q).
    """

    tile: int
    block_q: int
    final: str
    pw_waves: int | None
    ds_waves: int
    dq_waves: int

    @property
    def waves(self):
        """Wave count of the assigned dK/dV schedule."""
        return self.pw_waves if self.final == "pw" else self.ds_waves

    @property
    def rep(self):
        """Name of the assigned dK/dV representative in sp-n.md: `t<tile>_<geometry>_<schedule>_w<waves>`."""
        return f"t{self.tile}_{self.block_q}x16_{self.final}_w{self.waves}"


def _row(tile, block_q, final, pw_waves, ds_waves, dq_waves):
    return (tile, block_q), ScheduleEntry(
        tile, block_q, final, pw_waves, ds_waves, dq_waves
    )


# (tile, block_q) -> entry. Columns: final schedule, per-wave rep waves, D-split rep waves, dQ rep waves. Comment: SP-N occupancy rule | SA pw/ds geomean.
SCHEDULE_TABLE = dict(
    [
        _row(32, 32, "ds", 2, 2, 2),  # pw | 1.523
        _row(32, 16, "pw", 2, 4, 4),  # ds | 0.765
        _row(
            64, 32, "pw", 4, 4, 2
        ),  # pw | 0.997 (inside the 3.04 % control floor: the SP-N assignment stands)
        _row(64, 16, "pw", 4, 4, 4),  # ds | 0.913
        _row(96, 32, "ds", 2, 4, 4),  # pw | 1.097
        _row(96, 16, "pw", 4, 4, 4),  # ds | 0.763
        _row(128, 32, "ds", 2, 4, 4),  # ds | 1.342
        _row(128, 16, "pw", 4, 4, 4),  # ds | 0.944
        _row(160, 32, "pw", 4, 4, 4),  # pw | 0.781
        _row(160, 16, "pw", 4, 4, 4),  # ds | 0.812
        _row(192, 32, "pw", 4, 4, 4),  # pw | 0.938
        _row(192, 16, "ds", 2, 4, 4),  # ds | 1.488
        _row(224, 32, "pw", 4, 4, 4),  # pw | 0.587
        _row(224, 16, "pw", 2, 4, 4),  # ds | 0.884
        _row(256, 32, "pw", 4, 4, 4),  # pw | 0.713
        _row(256, 16, "pw", 4, 4, 4),  # ds | 0.777
        _row(512, 32, "ds", None, 4, 4),  # ds (only qualifier)
        _row(512, 16, "ds", None, 4, 4),  # ds (only qualifier)
    ]
)


def schedule_entry(tile, block_q):
    """The table row of (tile, block_q)."""
    try:
        return SCHEDULE_TABLE[(tile, block_q)]
    except KeyError:
        raise ValueError(
            f"no fp32 dK/dV schedule for tile {tile}, block_q {block_q}"
        ) from None


def tile_width_for(head_dim):
    """The smallest fp32 rung that holds `head_dim`."""
    for rung in FP32_LADDER:
        if head_dim <= rung:
            return rung
    raise ValueError(f"head_dim {head_dim} is above the fp32 ladder {FP32_LADDER}")


@dataclass(frozen=True)
class BwdDkDvInputMetadata:
    """What to compute. Set by the caller; never by policy. The 16-bit twin's fields (`dtype_str` is `"fp32"`)."""

    num_heads: int
    head_dim: int
    dtype_str: str = "fp32"
    num_kv_heads: int | None = None
    head_dim_v: int | None = None
    sm_scale: float | None = None
    causal: bool = False
    window: bool = False
    lse_layout_th: bool = False
    dropout: bool = False
    bias: bool = False


@dataclass(frozen=True)
class BwdDkDvKnobs:
    """How to compute it. `None` means "policy decides"; `resolve` fills every field. A pin must equal what the table gives for the (tile, geometry)."""

    block_dmodel: int | None = None
    block_dmodel_v: int | None = None
    padded_head: bool | None = None
    hdim_qk_floor: int | None = None

    num_waves: int | None = None
    block_kv: int | None = None
    block_q: int | None = None
    head_dim_granule: int | None = None

    dkv_shards: int | None = None
    mfma_rows: int | None = None
    num_stream_buffers: int | None = None
    waves_per_eu: int | None = None
    tight_registers: bool | None = None

    strides_constexpr: bool | None = None

    # Derived by `resolve`: the assigned dK/dV schedule of the (tile, geometry).
    schedule: str | None = None

    def merge(self, other):
        """`other`'s set fields win; its `None`s leave this one's alone."""
        if other is None:
            return self
        return replace(
            self,
            **{
                f.name: getattr(other, f.name)
                for f in fields(other)
                if getattr(other, f.name) is not None
            },
        )

    def resolve(self, meta: BwdDkDvInputMetadata) -> "BwdDkDvKnobs":
        """The complete build configuration for `meta`. Idempotent: every derived field is recomputed from `meta` and the pins."""
        return self._checked_scope(meta)._with_widths(meta)._with_schedule()

    def _checked_scope(self, meta):
        if meta.window and not meta.causal:
            raise ValueError(
                "window=True requires causal=True; it is a left bound on top of the causal one"
            )
        if meta.dtype_str != "fp32":
            raise NotImplementedError(
                f"this policy is the fp32 family's, got dtype_str {meta.dtype_str!r}"
            )
        num_kv_heads = (
            meta.num_heads if meta.num_kv_heads is None else meta.num_kv_heads
        )
        if meta.num_heads % num_kv_heads:
            raise ValueError(
                f"num_heads ({meta.num_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
            )
        return self

    def _with_widths(self, meta):
        """Decide `block_dmodel`, `block_dmodel_v`, `padded_head`."""
        block_dmodel = (
            self.block_dmodel
            if self.block_dmodel is not None
            else tile_width_for(meta.head_dim)
        )
        if block_dmodel not in FP32_LADDER:
            raise ValueError(
                f"block_dmodel must be one of the fp32 rungs {FP32_LADDER}, got {block_dmodel}"
            )
        if meta.head_dim > block_dmodel:
            raise ValueError(
                f"head_dim {meta.head_dim} does not fit the pinned block_dmodel {block_dmodel}"
            )
        head_dim_v = meta.head_dim if meta.head_dim_v is None else meta.head_dim_v
        if head_dim_v > block_dmodel:
            raise ValueError(
                f"head_dim_v {head_dim_v} does not fit block_dmodel {block_dmodel}"
            )
        block_dmodel_v = (
            block_dmodel if self.block_dmodel_v is None else self.block_dmodel_v
        )
        padded_head = self.padded_head
        if padded_head is None:
            padded_head = meta.head_dim != block_dmodel or head_dim_v != block_dmodel_v
        return replace(
            self,
            block_dmodel=block_dmodel,
            block_dmodel_v=block_dmodel_v,
            padded_head=bool(padded_head),
        )

    def _with_schedule(self):
        """Resolve geometry, schedule, waves and the derived set from the table; a pin that disagrees with the table is an error."""
        bq = DEFAULT_BLOCK_Q if self.block_q is None else self.block_q
        entry = schedule_entry(self.block_dmodel, bq)
        waves = entry.waves
        derived = dict(
            num_waves=waves,
            block_kv=16 if entry.final == "ds" else 16 * waves,
            block_q=bq,
            head_dim_granule=16,
            dkv_shards=1,
            mfma_rows=16,
            num_stream_buffers=1,
            waves_per_eu=0,
            tight_registers=False,
            schedule=entry.final,
        )
        for name, want in derived.items():
            have = getattr(self, name)
            if have is not None and have != want:
                raise ValueError(
                    f"{name}={have} is not the fp32 table's value {want} for tile {self.block_dmodel}, block_q {bq} ({entry.rep})"
                )
        return replace(self, **derived)

    def family(self):
        """The layout parameters of the resolved variant (`fmha_f32_common_gfx950.Family`)."""
        if self.schedule is None:
            raise ValueError(
                "knobs must be resolved: call `bwd_dkdv_knobs(...).resolve(meta)` first"
            )
        return Family(
            "dkdv", self.block_dmodel, self.block_q, self.schedule, self.num_waves
        )


def bwd_dkdv_knobs(arch="gfx950", **overrides):
    """Knobs for `arch`, with `overrides` pinned. Mirrors `fmha_tuning_bwd_dkdv_gfx950.bwd_dkdv_knobs`."""
    if arch != "gfx950":
        raise ValueError(f"this tuning module is gfx950-only, got {arch!r}")
    return BwdDkDvKnobs(**overrides)
