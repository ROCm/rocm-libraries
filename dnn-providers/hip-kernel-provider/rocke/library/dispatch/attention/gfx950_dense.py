# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 dense attention candidates (CDNA4, wave64, 32x32 MFMA + dense persistent).

Unified-kernel candidates for gfx950 live in :mod:`.gfx950_unified`.

The grid and persistent bodies are separate algorithms because they serve
different requests: only the grid body runs a moving bottom-right causal
diagonal. Wide DMA is a variant of the persistent algorithm (it requires the
persistent body). Each candidate is identified the same way as a unified
tuning geometry: ``spec_id`` names it and ``tuning_id`` names one
configuration from its knob space -- the tile (``block_m`` / ``block_n``)
included -- with ``auto`` being the candidate's default spec. The candidates
are opt-in: only an explicit ``algorithm`` + ``spec_id`` selects one.
Admission is the kernel's own ``supports_attention_dense``; dispatch adds no
stricter gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Tuple

from rocke.dispatch.core import CandidateRegistry, KernelCandidate

from .candidate import (
    DENSE_GRID_ALGORITHM,
    DENSE_PERSIST_ALGORITHM,
    make_dense_candidate,
)
from .common import (
    AttentionMaskType,
    AttentionRequest,
    _parse_attention_mask_type,
)

_DENSE_LAYOUT = "default"
_DENSE_TILE = "default"
# Base-spec values the knob space starts from (and sweeps away from): the
# gfx950 CU count and the shipped occupancy hint.
_DEFAULT_NUM_PERSISTENT = 256
_DEFAULT_WAVES_PER_EU = 2
_FEATURES = frozenset({"causal", "sliding_window", "sinks", "lse"})


@dataclass(frozen=True)
class Gfx950DenseVariant:
    """One registered gfx950 dense candidate: a body (algorithm), and wide DMA
    for the persistent one."""

    variant_id: str
    algorithm: str
    persistent: bool
    wide_lds_dma: bool
    features: frozenset

    @property
    def candidate_name(self) -> str:
        return f"attention_gfx950_dense_{self.variant_id}"

    @property
    def spec_id(self) -> str:
        return f"gfx950_dense_{self.variant_id}"


GFX950_DENSE_VARIANTS: Tuple[Gfx950DenseVariant, ...] = (
    Gfx950DenseVariant(
        "grid",
        DENSE_GRID_ALGORITHM,
        persistent=False,
        wide_lds_dma=False,
        features=_FEATURES | {"causal_bottom_right"},
    ),
    Gfx950DenseVariant(
        "persist",
        DENSE_PERSIST_ALGORITHM,
        persistent=True,
        wide_lds_dma=False,
        features=_FEATURES,
    ),
    Gfx950DenseVariant(
        "persist_widedma",
        DENSE_PERSIST_ALGORITHM,
        persistent=True,
        wide_lds_dma=True,
        features=_FEATURES,
    ),
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


def _derived(base, knobs: Mapping[str, object]) -> Mapping[str, object]:
    """``ragged`` for the tile the knobs choose: non-tile-multiple
    self-attention lengths take the on-chip ragged path."""
    block_m = int(knobs.get("block_m", base.block_m))
    block_n = int(knobs.get("block_n", base.block_n))
    ragged = _ragged_self_attention(
        int(base.seqlen_q),
        int(base.seqlen_kv),
        block_m,
        block_n,
        moving_bottom_right=bool(base.causal_bottom_right),
    )
    return {"ragged": ragged}


def _base_spec(req: AttentionRequest, variant: Gfx950DenseVariant):
    """The variant's default ``Gfx950AttentionDenseSpec`` for ``req``.

    Persist and wide DMA come from the variant; the tile and every other
    tuning field start at the shipped default and are varied by the knob space.
    Wide DMA cannot take the ragged path, so where the default tile would be
    ragged it starts at the first registered tile that is not (its ``block_m``
    is then resolved per problem and recorded).
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
    geometry = DENSE_TILE_GEOMETRIES[_DENSE_TILE]
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
    if ragged and variant.wide_lds_dma:
        for tile in DENSE_TILE_GEOMETRIES.values():
            if not _ragged_self_attention(
                sq,
                sk,
                int(tile["block_m"]),
                int(tile["block_n"]),
                moving_bottom_right=moving_bottom_right,
            ):
                bm, bn, ragged = int(tile["block_m"]), int(tile["block_n"]), False
                break
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
        emit_lse=bool(req.emit_lse),
    )


def _supports(spec, *, arch):
    from kernels.gfx950.attention_dense import supports_attention_dense

    return supports_attention_dense(spec, arch=arch)


def _make_gfx950_attention_dense_candidate(
    variant: Gfx950DenseVariant,
) -> KernelCandidate:
    """One gfx950 dense variant. Opt-in: selected only by its ``algorithm`` and
    ``spec_id``."""
    return make_dense_candidate(
        arch="gfx950",
        name=variant.candidate_name,
        spec_id=variant.spec_id,
        variant_id=variant.variant_id,
        algorithm=variant.algorithm,
        base_spec=lambda req: _base_spec(req, variant),
        supports=_supports,
        features=variant.features,
        recorded=frozenset({"block_m"}) if variant.wide_lds_dma else frozenset(),
        derived=_derived,
    )


def register(route: CandidateRegistry, execution: CandidateRegistry) -> None:
    for variant in GFX950_DENSE_VARIANTS:
        candidate = _make_gfx950_attention_dense_candidate(variant)
        route.register(candidate)
        execution.register(candidate)
