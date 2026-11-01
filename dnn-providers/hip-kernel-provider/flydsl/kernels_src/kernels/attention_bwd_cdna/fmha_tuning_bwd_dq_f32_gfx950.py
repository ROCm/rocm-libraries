# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Not vendored: written for this provider from the FlyDSL SDPA backward sweep (see README.md); it has no upstream.

"""Tuning policy of the fp32 dQ backward kernel (gfx950): the per-(tile, geometry) wave table.

The fp32 twin of `fmha_tuning_bwd_dq_gfx950` (same exported names and signatures, `dtype_str="fp32"`), for the native-f32-MFMA family of
`fmha_f32_common_gfx950`. Where the 16-bit policy resolves a wave geometry from a measured table, this one resolves it from `DQ_TABLE`, the dQ
column of the fp32 schedule table (shared with the dK/dV policy, `fmha_tuning_bwd_dkdv_f32_gfx950`):

  - the **geometry** is the Spec pin `block_m` in {16, 32} (the (BLOCK_M, BLOCK_N) = (block_m, 16) tile of the family); unpinned it is 32;
  - the **schedule** is the D-split dQ layer at every tile (`DSplitDq`): the workgroup owns `block_m` q rows, wave w owns head-dim slice w of Q,
    dO and dQ, K / V are staged 16 rows at a time, and the S / dP partials are exchanged through LDS;
  - the **wave count** is the dQ representative of the (tile, geometry) (highest occupancy, then fewest VGPR+AGPR, then 4 waves over 2).

Neither the wave count nor the schedule is a Spec key; a Spec fixes the geometry and the table does the rest.
`block_n` (16), `head_dim_granule` (the slice unit: 16, or 32 for the 32x32x2 accumulation tile at `block_m` 32) and `waves_per_eu` (0: no hint, probes at
occupancy + 1 reached nothing higher) have one value per entry.

This module records the policy; which (tile, feature) combinations the kernel computes is the kernel module's scope check (`fmha_bwd_dq_f32_gfx950`).
"""

from dataclasses import dataclass, fields, replace

from kernels.attention_bwd_cdna.fmha_f32_common_gfx950 import Family

__all__ = [
    "BwdDqKnobs",
    "DEFAULT_BLOCK_M",
    "DQ_TABLE",
    "DqEntry",
    "FP32_LADDER",
    "FmhaInputMetadata",
    "bwd_dq_knobs",
    "dq_entry",
    "tile_width_for",
]

# The fp32 ladder: the 16-bit ladder without 384 (hd 257..512 rides t512).
FP32_LADDER = (32, 64, 96, 128, 160, 192, 224, 256, 512)

# Geometry of an unpinned Spec.
DEFAULT_BLOCK_M = 32


@dataclass(frozen=True)
class DqEntry:
    """One (tile, geometry) row of the dQ column of the table: the wave count of the dQ representative."""

    tile: int
    block_m: int
    waves: int

    @property
    def rep(self):
        """Name of the representative: `t<tile>_<geometry>_ds_w<waves>`."""
        return f"t{self.tile}_{self.block_m}x16_ds_w{self.waves}"


def _row(tile, block_m, waves):
    return (tile, block_m), DqEntry(tile, block_m, waves)


# (tile, block_m) -> entry. Waves of the dQ representative.
DQ_TABLE = dict(
    [
        _row(32, 32, 2),
        _row(32, 16, 4),
        _row(64, 32, 2),
        _row(64, 16, 4),
        _row(96, 32, 4),
        _row(96, 16, 4),
        _row(128, 32, 4),
        _row(128, 16, 4),
        _row(160, 32, 4),
        _row(160, 16, 4),
        _row(192, 32, 4),
        _row(192, 16, 4),
        _row(224, 32, 4),
        _row(224, 16, 4),
        _row(256, 32, 4),
        _row(256, 16, 4),
        _row(512, 32, 4),
        _row(512, 16, 4),
    ]
)


def dq_entry(tile, block_m):
    """The table row of (tile, block_m)."""
    try:
        return DQ_TABLE[(tile, block_m)]
    except KeyError:
        raise ValueError(
            f"no fp32 dQ entry for tile {tile}, block_m {block_m}"
        ) from None


def tile_width_for(head_dim):
    """The smallest fp32 rung that holds `head_dim`."""
    for rung in FP32_LADDER:
        if head_dim <= rung:
            return rung
    raise ValueError(f"head_dim {head_dim} is above the fp32 ladder {FP32_LADDER}")


@dataclass(frozen=True)
class FmhaInputMetadata:
    """What to compute. Set by the caller; never by policy. The 16-bit twin's fields (`dtype_str` is `"fp32"`, `causal` defaults to False)."""

    num_heads: int
    head_dim: int
    causal: bool = False
    dtype_str: str = "fp32"
    num_kv_heads: int | None = None
    head_dim_v: int | None = None
    sm_scale: float | None = None
    window: bool = False
    bias: bool = False
    dropout: bool = False


@dataclass(frozen=True)
class BwdDqKnobs:
    """How to compute it. `None` means "policy decides"; `resolve` fills every derived field. A pin must equal what the table gives for the
    (tile, geometry). `hdim_qk_floor` is the head-dim extent the build promises (a head dim above it; 0 = no promise); `store_db` builds the dB
    store of a bias build. `paged`, `num_kv_splits` and `bias_stage` are the 16-bit build axes the Spec layer passes through: only their no-feature
    values exist here."""

    block_dmodel: int | None = None
    padded_head: bool | None = None
    hdim_qk_floor: int | None = None
    paged: bool | None = None
    num_kv_splits: int | None = None
    store_db: bool | None = None
    bias_stage: str | None = None

    num_waves: int | None = None
    block_m: int | None = None
    block_n: int | None = None
    head_dim_granule: int | None = None
    waves_per_eu: int | None = None

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

    def resolve(self, meta: FmhaInputMetadata) -> "BwdDqKnobs":
        """The complete build configuration for `meta`. Idempotent: every derived field is recomputed from `meta` and the pins."""
        return self._checked_scope(meta)._with_widths(meta)._with_geometry()

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
        if self.paged or self.num_kv_splits not in (None, 1):
            raise ValueError("the fp32 dQ family has no paged / split-KV axis")
        if self.bias_stage is not None:
            raise ValueError(
                f"bias_stage={self.bias_stage!r}: the fp32 dQ family stages no bias"
            )
        if self.store_db and not meta.bias:
            raise ValueError(
                "store_db=True needs bias=True: dB is the gradient of the bias input"
            )
        return self

    def _with_widths(self, meta):
        """Decide `block_dmodel` and `padded_head`. `head_dim_v` may sit below the tile (the dQ Spec's `hdv` convention is the rung below)."""
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
        padded_head = self.padded_head
        if padded_head is None:
            padded_head = meta.head_dim != block_dmodel
        # the floor of a pinned tile is 0 unless the Spec names one; a build promises head dims above it
        hdim_qk_floor = 0 if self.hdim_qk_floor is None else self.hdim_qk_floor
        if not 0 <= hdim_qk_floor < block_dmodel:
            raise ValueError(
                f"hdim_qk_floor {hdim_qk_floor} must be in [0, block_dmodel={block_dmodel})"
            )
        if meta.head_dim <= hdim_qk_floor:
            raise ValueError(
                f"head_dim {meta.head_dim} is at or below this build's hdim_qk_floor {hdim_qk_floor}"
            )
        return replace(
            self,
            block_dmodel=block_dmodel,
            padded_head=bool(padded_head),
            hdim_qk_floor=int(hdim_qk_floor),
            store_db=bool(self.store_db),
        )

    def _with_geometry(self):
        """Resolve geometry, waves and the derived set from the table; a pin that disagrees with the table is an error."""
        bm = DEFAULT_BLOCK_M if self.block_m is None else self.block_m
        entry = dq_entry(self.block_dmodel, bm)
        derived = dict(
            num_waves=entry.waves,
            block_m=bm,
            block_n=16,
            head_dim_granule=self.family_of(entry).unit,
            waves_per_eu=0,
        )
        for name, want in derived.items():
            have = getattr(self, name)
            if have is not None and have != want:
                raise ValueError(
                    f"{name}={have} is not the fp32 table's value {want} for tile {self.block_dmodel}, block_m {bm} ({entry.rep})"
                )
        return replace(self, **derived)

    @staticmethod
    def family_of(entry):
        return Family("dq", entry.tile, entry.block_m, "ds", entry.waves)

    def family(self):
        """The layout parameters of the resolved variant (`fmha_f32_common_gfx950.Family`)."""
        if self.num_waves is None:
            raise ValueError(
                "knobs must be resolved: call `bwd_dq_knobs(...).resolve(meta)` first"
            )
        return Family("dq", self.block_dmodel, self.block_m, "ds", self.num_waves)


def bwd_dq_knobs(arch="gfx950", **overrides):
    """Knobs for `arch`, with `overrides` pinned. Mirrors `fmha_tuning_bwd_dq_gfx950.bwd_dq_knobs`."""
    if arch != "gfx950":
        raise ValueError(f"this tuning module is gfx950-only, got {arch!r}")
    return BwdDqKnobs(**overrides)
