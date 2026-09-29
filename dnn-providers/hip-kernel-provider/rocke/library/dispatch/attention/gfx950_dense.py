# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 dense attention candidates (CDNA4, wave64, 32x32 MFMA + dense persistent).

Unified-kernel candidates for gfx950 live in :mod:`.gfx950_unified`.

Each candidate is one frozen (tile x persist x wide-DMA) variant, identified
the same way as a unified tuning geometry: ``spec_id`` names the variant and
``tuning_id`` names one configuration from its knob space (``auto``
is the variant's default spec). The candidates are opt-in: only an explicit
``spec_id`` selects one. Admission is the kernel's own
``supports_attention_dense``; dispatch adds no stricter gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from rocke.dispatch.core import CandidateRegistry, KernelCandidate

from .candidate import make_dense_candidate
from .common import (
    AttentionMaskType,
    AttentionRequest,
    _parse_attention_mask_type,
)

_DENSE_LAYOUT = "default"
# Base-spec values the knob space starts from (and sweeps away from): the
# gfx950 CU count and the shipped occupancy hint.
_DEFAULT_NUM_PERSISTENT = 256
_DEFAULT_WAVES_PER_EU = 2


@dataclass(frozen=True)
class Gfx950DenseVariant:
    """One registered gfx950 dense geometry: tile x persist x wide DMA."""

    variant_id: str
    tile: str
    persistent: bool
    wide_lds_dma: bool

    @property
    def candidate_name(self) -> str:
        return f"attention_gfx950_dense_{self.variant_id}"

    @property
    def spec_id(self) -> str:
        return f"gfx950_dense_{self.variant_id}"


# Frozen catalog. Wide DMA is persist-only.
GFX950_DENSE_VARIANTS: Tuple[Gfx950DenseVariant, ...] = (
    Gfx950DenseVariant("grid_default", "default", False, False),
    Gfx950DenseVariant("persist_default", "default", True, False),
    Gfx950DenseVariant("persist_widedma_default", "default", True, True),
    Gfx950DenseVariant("grid_bm128", "bm128", False, False),
    Gfx950DenseVariant("persist_bm128", "bm128", True, False),
    Gfx950DenseVariant("persist_widedma_bm128", "bm128", True, True),
)
GFX950_DENSE_VARIANT_BY_NAME = {v.candidate_name: v for v in GFX950_DENSE_VARIANTS}
GFX950_DENSE_VARIANT_BY_SPEC_ID = {v.spec_id: v for v in GFX950_DENSE_VARIANTS}


def _ragged_self_attention(
    sq: int,
    sk: int,
    block_m: int,
    block_n: int,
    *,
    moving_bottom_right: bool = False,
) -> bool:
    return (sq == sk or moving_bottom_right) and (
        sq % block_m != 0 or sk % block_n != 0
    )


def _base_spec(req: AttentionRequest, variant: Gfx950DenseVariant):
    """The variant's default ``Gfx950AttentionDenseSpec`` for ``req``.

    Tile, persist, and wide DMA come from the frozen variant; every other
    tuning field is at its shipped default and is varied by the knob space.
    Non-tile-multiple self-attention lengths use the on-chip ragged path.
    """
    from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES
    from kernels.gfx950.attention_dense import (
        GFX950_DENSE_LAYOUTS,
        Gfx950AttentionDenseSpec,
    )

    if req.arch != "gfx950":
        raise ValueError(
            f"gfx950 dense spec factory requires arch='gfx950', got {req.arch!r}"
        )
    sq, sk = int(req.seqlen_q), int(req.seqlen_k)
    geometry = DENSE_TILE_GEOMETRIES[variant.tile]
    layout = GFX950_DENSE_LAYOUTS[_DENSE_LAYOUT]
    bm = int(geometry["block_m"])
    bn = int(geometry["block_n"])
    mask_type = _parse_attention_mask_type(req.mask_type)
    moving_bottom_right = (
        mask_type == AttentionMaskType.BOTTOM_RIGHT_CAUSAL and sq != sk
    )
    # Cross-length ragged attention is valid only when bottom-right supplies the
    # shifted diagonal. Equal-length bottom-right is ordinary causal attention.
    ragged = _ragged_self_attention(
        sq, sk, bm, bn, moving_bottom_right=moving_bottom_right
    )
    return Gfx950AttentionDenseSpec(
        batch=int(req.batch),
        seqlen_q=sq,
        seqlen_kv=sk,
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        head_size=int(req.hdim_q),
        causal=mask_type != AttentionMaskType.NO_MASK,
        dtype=req.dtype.lower(),
        block_m=bm,
        block_n=bn,
        waves_per_eu=_DEFAULT_WAVES_PER_EU,
        lds_v_row_pad=int(layout["lds_v_row_pad"]),
        persistent=variant.persistent,
        num_persistent=_DEFAULT_NUM_PERSISTENT,
        persist_decode="auto",
        ragged=ragged,
        sliding_window=int(req.sliding_window),
        use_sinks=bool(req.use_sinks),
        wide_lds_dma=variant.wide_lds_dma,
        causal_bottom_right=moving_bottom_right,
    )


def _supports(spec, *, arch):
    from kernels.gfx950.attention_dense import supports_attention_dense

    return supports_attention_dense(spec, arch=arch)


def _make_gfx950_attention_dense_candidate(
    variant: Gfx950DenseVariant,
) -> KernelCandidate:
    """One gfx950 dense variant. Opt-in: selected only by its ``spec_id``."""
    return make_dense_candidate(
        arch="gfx950",
        name=variant.candidate_name,
        spec_id=variant.spec_id,
        variant_id=variant.variant_id,
        base_spec=lambda req: _base_spec(req, variant),
        supports=_supports,
        # Only frozen grid variants implement a moving bottom-right diagonal.
        features=frozenset(
            {"causal", "sliding_window", "sinks"}
            | ({"causal_bottom_right"} if not variant.persistent else set())
        ),
    )


def register(route: CandidateRegistry, execution: CandidateRegistry) -> None:
    for variant in GFX950_DENSE_VARIANTS:
        candidate = _make_gfx950_attention_dense_candidate(variant)
        route.register(candidate)
        execution.register(candidate)
