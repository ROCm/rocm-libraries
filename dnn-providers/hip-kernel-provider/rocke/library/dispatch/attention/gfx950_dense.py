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
are opt-in: only an explicit ``algorithm`` + ``spec_id`` selects one, except
that an unpinned request for contiguous tensors (``layout='dense'``) with more
than one query row is pinned to the body :func:`dense_auto_order` names (see
:func:`dispatch.attention.dispatch_attention`). Admission is the kernel's own
``supports_attention_dense``; dispatch adds no stricter gate.
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
from .common import AttentionRequest, normalize_band

_DENSE_LAYOUT = "default"
_DENSE_TILE = "default"
# Base-spec values the knob space starts from (and sweeps away from): the
# gfx950 CU count and the shipped occupancy hint.
_DEFAULT_NUM_PERSISTENT = 256
_DEFAULT_WAVES_PER_EU = 2
_FEATURES = frozenset(
    {"causal", "sliding_window", "full_attention", "band", "sinks", "lse"}
)


@dataclass(frozen=True)
class Gfx950DenseVariant:
    """One registered gfx950 dense candidate: a body (algorithm), and wide DMA
    for the persistent one."""

    variant_id: str
    algorithm: str
    persistent: bool
    wide_lds_dma: bool
    features: frozenset
    nonpersist_decode: str = "auto"

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
    # The grid body at the qb_minor block order: the grid the hipDNN kernel ingestor
    # launches.
    Gfx950DenseVariant(
        "qbminor",
        DENSE_GRID_ALGORITHM,
        persistent=False,
        wide_lds_dma=False,
        features=_FEATURES | {"causal_bottom_right"},
        nonpersist_decode="qb_minor",
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


_DENSE_BR_BAND_ERROR = (
    "dense: a window or lookahead with bottom-right alignment and "
    "seqlen_q != seqlen_k is not implemented"
)


def _is_ragged(sq: int, sk: int, block_m: int, block_n: int) -> bool:
    """Lengths that are not tile multiples run the on-chip-padded ragged body (any
    mask: the padded keys / query rows are handled by the key-pad mask and the
    guarded store, not by the diagonal)."""
    return sq % block_m != 0 or sk % block_n != 0


def _dense_band(req: AttentionRequest) -> Tuple[bool, int, bool]:
    """``(causal, right_bound, moving_bottom_right)`` the dense body computes for
    ``req``. Raises ``ValueError`` for a band dense cannot serve.

    ``causal`` False is the unbounded right side; ``right_bound`` is the lookahead
    (0 = plain causal). The dense body anchors its window / lookahead on the
    top-left diagonal; only the plain causal mask has a compile-time shifted
    (bottom-right) variant, so a window or lookahead at bottom-right alignment with
    ``seqlen_q != seqlen_k`` is rejected here instead of being computed top-left.
    """
    band = normalize_band(req, uniform_lengths=True)
    sq, sk = int(req.seqlen_q), int(req.seqlen_k)
    window = int(req.sliding_window)
    causal = band.right_bound is not None
    right_bound = band.right_bound or 0
    if band.top_left or sq == sk:
        return causal, right_bound, False  # the diagonals coincide, or top-left
    # Bottom-right alignment with seqlen_q != seqlen_k.
    if not causal:
        if window > 0:
            raise ValueError(_DENSE_BR_BAND_ERROR)
        return False, 0, False  # full attention: alignment is moot
    if window > 0 or right_bound > 0:
        raise ValueError(_DENSE_BR_BAND_ERROR)
    return True, 0, True  # plain causal, compile-time shifted diagonal


def _default_block_m(req: AttentionRequest, variant: Gfx950DenseVariant) -> int:
    """Query tile of the variant's default spec.

    The shared 256-row tile, except on the wide-DMA body (the only one whose
    ``block_m`` is recorded, so its tuning IDs already name the tile) when that
    tile leaves CTAs idle: while the full-tile work count (query blocks x query
    heads x batch) is at most the persistent CTA count, a 128-row tile doubles
    the work items and halves each causal item's KV loop. Measured on the
    256-CU part at D128, both dtypes, plain causal: the 128-row tile wins every
    shape at or below that count and loses at every larger one.
    """
    from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES

    bm = int(DENSE_TILE_GEOMETRIES[_DENSE_TILE]["block_m"])
    if not variant.wide_lds_dma or int(req.hdim_q) != 128:
        return bm
    work = -(-int(req.seqlen_q) // bm) * int(req.nhead_q) * int(req.batch)
    return bm // 2 if work <= _DEFAULT_NUM_PERSISTENT else bm


def _derived(base, knobs: Mapping[str, object]) -> Mapping[str, object]:
    """``ragged`` for the tile the knobs choose: non-tile-multiple lengths take
    the on-chip ragged path."""
    block_m = int(knobs.get("block_m", base.block_m))
    block_n = int(knobs.get("block_n", base.block_n))
    ragged = _is_ragged(int(base.seqlen_q), int(base.seqlen_kv), block_m, block_n)
    return {"ragged": ragged}


def _base_spec(req: AttentionRequest, variant: Gfx950DenseVariant):
    """The variant's default ``Gfx950AttentionDenseSpec`` for ``req``.

    Persist and wide DMA come from the variant; the tile comes from
    :func:`_default_block_m` and every other tuning field starts at the shipped
    default; the knob space varies them. Wide DMA cannot take the ragged path,
    so where its default tile would be ragged it starts at the first registered
    tile that is not (its ``block_m`` is resolved per problem and recorded).
    """
    from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES
    from kernels.gfx950.attention_dense import (
        GFX950_DENSE_LAYOUTS,
        Gfx950AttentionDenseSpec,
    )

    # Exact compare: ``arch`` is canonical by construction. See gfx942_dense.py.
    if req.arch != "gfx950":
        raise ValueError(
            f"gfx950 dense spec factory requires arch='gfx950', got {req.arch!r}"
        )
    sq, sk = int(req.seqlen_q), int(req.seqlen_k)
    geometry = DENSE_TILE_GEOMETRIES[_DENSE_TILE]
    layout = GFX950_DENSE_LAYOUTS[_DENSE_LAYOUT]
    bn = int(geometry["block_n"])
    causal, right_bound, moving_bottom_right = _dense_band(req)
    bm = _default_block_m(req, variant)
    ragged = _is_ragged(sq, sk, bm, bn)
    if ragged and variant.wide_lds_dma:
        for tile in DENSE_TILE_GEOMETRIES.values():
            if not _is_ragged(sq, sk, int(tile["block_m"]), int(tile["block_n"])):
                bm, bn, ragged = int(tile["block_m"]), int(tile["block_n"]), False
                break
    return Gfx950AttentionDenseSpec(
        batch=int(req.batch),
        seqlen_q=sq,
        seqlen_kv=sk,
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        head_size=int(req.hdim_q),
        causal=causal,
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
        nonpersist_decode=variant.nonpersist_decode,
        causal_bottom_right=moving_bottom_right,
        emit_lse=bool(req.emit_lse),
        right_bound=right_bound,
    )


def dense_auto_order(req: AttentionRequest) -> Tuple[str, ...]:
    """Candidate names, best first, for an unpinned contiguous-layout request.

    The persistent body pays off once the default tile yields at least one
    work item per persistent CTA and the causal diagonal is fixed (the moving
    bottom-right diagonal runs only on the grid body). Wide DMA additionally
    needs a causal D128 problem whose lengths are tile multiples. For plain
    causal (no lookahead) it is also taken below that work count: its default
    tile halves there (:func:`_default_block_m`), which keeps every persistent
    CTA busy. Each later entry is a fallback for when an earlier body refuses
    the request.
    """
    from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES

    try:
        causal, right_bound, moving_bottom_right = _dense_band(req)
    except ValueError:
        return ()
    geometry = DENSE_TILE_GEOMETRIES[_DENSE_TILE]
    bm, bn = int(geometry["block_m"]), int(geometry["block_n"])
    sq, sk = int(req.seqlen_q), int(req.seqlen_k)
    work_items = -(-sq // bm) * int(req.nhead_q) * int(req.batch)
    persistent = not moving_bottom_right and work_items >= _DEFAULT_NUM_PERSISTENT
    wide_ok = (
        not moving_bottom_right
        and int(req.hdim_q) == int(req.hdim_v) == 128
        and causal
        and not int(req.sliding_window)
        and not bool(req.use_sinks)
        and sq % bm == 0
        and sk % bn == 0
    )
    by_id = {v.variant_id: v.candidate_name for v in GFX950_DENSE_VARIANTS}
    if persistent:
        order = (
            ("persist_widedma", "persist", "grid") if wide_ok else ("persist", "grid")
        )
    elif wide_ok and not right_bound:
        order = ("persist_widedma", "grid")
    else:
        order = ("grid",)
    return tuple(by_id[v] for v in order)


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
