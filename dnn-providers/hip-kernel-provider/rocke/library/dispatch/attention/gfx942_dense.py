# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx942 dense attention candidate (CDNA3, wave64).

The standalone ``attention_dense`` prefill kernel runs on the 32x32x8 atom (CDNA3
has no 32x32x16 fp16/bf16 atom, so it doubles the K loop), unlike the unified
``dense_pipe`` flash path on the narrow 16x16x16 atom, which lives in
:mod:`.gfx942_unified`. See ``builders/gfx942/attention/prefill/README.md`` for
why the dense kernel is a per-gfx module rather than an arch branch in the
gfx950 body.

Five candidates, identified like every other tuned candidate: ``spec_id``
names one and ``tuning_id`` names one configuration from its knob space,
geometry included (``auto`` is the default spec). ``gfx942_dense`` is the
tuned default; ``gfx942_dense_qbminor`` is the one-CTA-per-block grid at the
qb_minor block order, the grid the hipDNN kernel ingestor launches (the twin
of gfx950's ``qbminor`` variant). The three ``*_aot`` candidates are the other
grids that ingestor launches, each with its grid fixed by its default spec:
``gfx942_dense_persist_aot`` is the D128 persistent grid at 304 CTAs with its
work decode pinned, and ``gfx942_dense_btminor_aot`` / ``gfx942_dense_hqswz_aot``
are the one-CTA-per-block grid at the bt_hkv_minor / hq_minor_swz block order.
All are opt-in: only an explicit ``spec_id`` selects them. Admission is the
kernel's own ``supports_attention_dense``.
"""

from __future__ import annotations

from rocke.dispatch.core import CandidateRegistry, KernelCandidate

from .candidate import make_dense_candidate
from .dense_rules import dense_paged_fields, dense_paged_tile_fits
from .common import (
    AttentionMaskType,
    AttentionRequest,
    _parse_attention_mask_type,
)

CANDIDATE_NAME = "attention_gfx942_dense"
SPEC_ID = "gfx942_dense"
_VARIANT_ID = "dense"
QBMINOR_CANDIDATE_NAME = "attention_gfx942_dense_qbminor"
QBMINOR_SPEC_ID = "gfx942_dense_qbminor"
_QBMINOR_VARIANT_ID = "qbminor"
# The ingestor's other grids: (candidate name, spec_id, variant_id, persistent,
# nonpersist_decode). None of the names starts with QBMINOR_CANDIDATE_NAME.
PERSIST_AOT_SPEC_ID = "gfx942_dense_persist_aot"
BTMINOR_AOT_SPEC_ID = "gfx942_dense_btminor_aot"
HQSWZ_AOT_SPEC_ID = "gfx942_dense_hqswz_aot"
_AOT_GRIDS = (
    (
        "attention_gfx942_dense_persist_aot",
        PERSIST_AOT_SPEC_ID,
        "persist_aot",
        True,
        "qb_minor",
    ),
    (
        "attention_gfx942_dense_btminor_aot",
        BTMINOR_AOT_SPEC_ID,
        "btminor_aot",
        False,
        "bt_hkv_minor",
    ),
    (
        "attention_gfx942_dense_hqswz_aot",
        HQSWZ_AOT_SPEC_ID,
        "hqswz_aot",
        False,
        "hq_minor_swz",
    ),
)

# block_n (KV tile) the default spec ships; 64 is the resource-efficient peak
# (see AttentionDenseSpec.block_n). The knob space sweeps block_m / block_n.
_DENSE_BLOCK_N = 64
# Persistent-grid CTA count of the default spec: gfx942's largest part has 304
# CUs. The knob space sweeps it.
_GFX942_NUM_PERSISTENT = 304


def _default_block_m(req: AttentionRequest) -> int:
    """Query tile of the default spec.

    The shared 256-row tile unless it leaves CUs idle: when the full-tile work
    count (query blocks x query heads x batch) is below the CU count, a
    128-row tile doubles the work items, and each causal item's KV loop is
    half as long. Measured on the 304-CU part at D128, both dtypes, causal
    without window or sinks: the 128-row tile wins every shape below that
    threshold and loses every shape at or above it. A 64-row tile loses almost
    everywhere. The same under-fill holds at D64 and with sinks, so both take the
    rule, except non-causal sinks on the D128 persistent grid: without the causal
    diagonal no item gets shorter, and the doubled work only adds a second
    persistent round. Window requests keep the shared tile (narrow windows have
    their own geometry below), and so do bottom-right (chunked) requests, whose
    KV split below is sized for it, and paged requests whose pages do not fit the
    128-row tile.
    """
    from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES

    bm = int(DENSE_TILE_GEOMETRIES["default"]["block_m"])
    if int(req.sliding_window) or "causal_bottom_right" in req.features():
        return bm
    causal = _parse_attention_mask_type(req.mask_type) != AttentionMaskType.NO_MASK
    if bool(req.use_sinks) and not causal and int(req.hdim_q) == 128:
        return bm
    num_cus = int(req.num_cus) or _GFX942_NUM_PERSISTENT
    work = -(-int(req.seqlen_q) // bm) * int(req.nhead_q) * int(req.batch)
    if work < num_cus and dense_paged_tile_fits(req, bm // 2, _DENSE_BLOCK_N):
        return bm // 2
    return bm


# Sliding-window geometry. A causal band of width W visits about block_m + W keys
# per query row, so a 256-row block over a narrow band spends most of its tiles on
# masked work. Window widths up to this many keys count as narrow.
_NARROW_WINDOW = 256


def _narrow_window_geometry(head_size, dtype):
    """``(block_m, block_n, window_wave_skip)`` for a narrow sliding-window band.

    128-row query blocks on the one-CTA-per-block grid. D128 halves the KV tile to
    32 keys, which halves its K/V LDS footprint so two CTAs fit per CU, and skips
    the (wave, tile) pairs the band fully masks, whose SIMD slot the co-resident
    CTA then fills; bf16 D64 goes to 64-row blocks (its waves-per-eu 4 build is
    small enough to keep several CTAs resident per CU). Depends only on
    compile-time fields, so one binary still serves every runtime shape.
    """
    if head_size == 128:
        return 128, 32, True
    if dtype == "bf16":
        return 64, _DENSE_BLOCK_N, False
    return 128, _DENSE_BLOCK_N, False


def tuned_base_spec(req: AttentionRequest):
    """The default ``Gfx942AttentionDenseSpec`` for ``req``.

    The gfx942 twin of ``gfx950_dense._base_spec``. The persistent grid turns on
    at D128, and non-tile-multiple self-attention lengths take the on-chip
    ragged path. A chunked bottom-right request whose work items fill their last
    wave poorly splits each item's KV range across CTAs, where the kernel admits
    the split.
    ``waves_per_eu`` comes from the kernel's own per-config policy
    (``_tuned_waves_per_eu``): any value the kernel bakes into its
    ``kernel_name`` must be resolved by the kernel's policy, or the name tag and
    the compiled binary can disagree. Every gfx942-private codegen knob stays at
    the concrete spec's default; the knob space varies them.
    """
    from dataclasses import replace

    # Exact compare, deliberately: ``AttentionRequest.__post_init__`` has already
    # canonicalized ``arch``, so a case- or suffix-insensitive compare here would
    # be defending against an input shape that cannot reach this function.
    if req.arch != "gfx942":
        raise ValueError(
            f"gfx942 dense spec factory requires arch='gfx942', got {req.arch!r}"
        )
    head_size = int(req.hdim_q)
    # The persistent grid measured ahead at D128 for every batch and mask; at
    # D64 the non-persistent grid did. Paged K/V runs on the default grid.
    paged = bool(dense_paged_fields(req))
    spec = _make_spec(
        req,
        _default_block_m(req),
        _DENSE_BLOCK_N,
        persistent=head_size == 128 and not paged,
    )
    if spec.causal_bottom_right:
        from kernels.common.attention_dense_kv_split import (
            max_kv_segments_for_extent,
        )

        from .dense_rules import dense_wave_segments

        cus = int(req.num_cus) if int(req.num_cus) > 0 else _GFX942_NUM_PERSISTENT
        work = -(-spec.seqlen_q // spec.block_m) * int(req.nhead_q) * int(req.batch)
        segments = dense_wave_segments(
            work,
            cus,
            max_segments=max_kv_segments_for_extent(
                req.batch, spec.seqlen_q, req.nhead_q, head_size, max_segments=8
            ),
        )
        # The split runs on the one-CTA-per-block grid. Keep it only where the
        # kernel admits it (check_kv_segments).
        split = replace(spec, persistent=False, kv_segments=segments)
        if segments > 1 and _supports(split, arch="gfx942")[0]:
            spec = split
    if 0 < int(req.sliding_window) <= _NARROW_WINDOW:
        bm, bn, skip = _narrow_window_geometry(head_size, req.dtype.lower())
        # Keep the default geometry for any request the narrow one cannot serve.
        if dense_paged_tile_fits(req, bm, bn):
            narrow = _make_spec(req, bm, bn, persistent=False, window_wave_skip=skip)
            if _supports(narrow, arch="gfx942")[0]:
                spec = narrow
    return spec


def _qbminor_base_spec(req: AttentionRequest):
    """The default spec on the grid the hipDNN kernel ingestor launches.

    One CTA per (query block, query head, batch) at the qb_minor block order
    and one KV segment, whatever the problem: the ingestor's native dispatch
    sizes that grid at run time, so no field may resolve a different grid
    or decode from the build-time shape. The tile is the shared default rather
    than :func:`_default_block_m`, whose work count is the build-time shape's;
    the knob space varies it.
    """
    return ingestor_base_spec(req, persistent=False, nonpersist_decode="qb_minor")


def ingestor_base_spec(
    req: AttentionRequest, *, persistent: bool, nonpersist_decode: str
):
    """The default spec of one grid the hipDNN kernel ingestor launches.

    The shared default tile and one KV segment, with the grid fixed: the
    non-persistent grid at ``nonpersist_decode``, or (``persistent``) the
    persistent grid at the 304-CTA count with ``persist_decode`` pinned to what
    ``auto`` resolves to at the build shape. Pinning names the same binary and
    symbol as ``auto`` there, but the ingestor reads the decode to pass its
    fast-division kernargs, and every runtime shape must keep that decode. The
    persistent grid is D128 only: at D64 the grid measured ahead.
    """
    from dataclasses import replace

    from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES

    if req.arch != "gfx942":
        raise ValueError(
            f"gfx942 dense spec factory requires arch='gfx942', got {req.arch!r}"
        )
    if persistent and int(req.hdim_q) != 128:
        raise ValueError(
            f"the ingestor's gfx942 persistent grid is D128 only, got D{req.hdim_q}"
        )
    bm = int(DENSE_TILE_GEOMETRIES["default"]["block_m"])
    spec = _make_spec(req, bm, _DENSE_BLOCK_N, persistent=persistent)
    spec = replace(spec, nonpersist_decode=nonpersist_decode)
    if persistent:
        spec = replace(spec, persist_decode=spec.resolved_persist_decode)
    return spec


def _make_spec(
    req: AttentionRequest,
    bm: int,
    bn: int,
    *,
    persistent: bool,
    window_wave_skip: bool = False,
):
    """The ``Gfx942AttentionDenseSpec`` for ``req`` at one tile geometry and
    grid, with one KV segment."""
    from kernels.gfx942.attention_dense import (
        Gfx942AttentionDenseSpec,
        _tuned_waves_per_eu,
    )

    sq, sk = int(req.seqlen_q), int(req.seqlen_k)
    head_size = int(req.hdim_q)
    dtype = req.dtype.lower()
    mask_type = _parse_attention_mask_type(req.mask_type)
    # on-chip ragged padding for ragged self-attention lengths (seqlen_q==seqlen_kv,
    # not a block_m/block_n multiple). Cross-attention ragged is left to the
    # validator.
    ragged = (sq == sk) and ((sq % bm != 0) or (sk % bn != 0))
    # Bottom-right causal with seqlen_q != seqlen_k: the same predicate the router
    # matches against ``features`` below, so a band this body cannot shift (a
    # bottom-right window) reaches ``supports_attention_dense`` and is rejected
    # there instead of running top-left.
    bottom_right = "causal_bottom_right" in req.features()
    if bottom_right and mask_type == AttentionMaskType.NO_MASK:
        # The band normalizer maps a cross-length NO_MASK request on this arch to
        # the bottom-right causal band; the dense body serves only explicit
        # causal kinds there.
        raise ValueError(
            "gfx942 dense: cross-length NO_MASK requests are not served by the "
            "dense kernel"
        )
    # Chunked prefill: the unmasked body loop skips the mask on every tile below
    # the shifted diagonal.
    diag_split = bottom_right and not ragged
    return Gfx942AttentionDenseSpec(
        batch=int(req.batch),
        seqlen_q=sq,
        seqlen_kv=sk,
        num_query_heads=int(req.nhead_q),
        num_kv_heads=int(req.nhead_k),
        head_size=head_size,
        causal=mask_type != AttentionMaskType.NO_MASK,
        dtype=dtype,
        block_m=bm,
        block_n=bn,
        persistent=persistent,
        num_persistent=_GFX942_NUM_PERSISTENT,
        persist_decode="auto",
        ragged=ragged,
        sliding_window=int(req.sliding_window),
        use_sinks=bool(req.use_sinks),
        waves_per_eu=_tuned_waves_per_eu(head_size, dtype),
        emit_lse=bool(req.emit_lse),
        causal_bottom_right=bottom_right,
        causal_diag_split=diag_split,
        window_wave_skip=window_wave_skip,
        **dense_paged_fields(req),
    )


def _supports(spec, *, arch):
    from kernels.gfx942.attention_dense import supports_attention_dense

    return supports_attention_dense(spec, arch=arch)


def _make_gfx942_attention_dense_candidate() -> KernelCandidate:
    """Dense flash-attn prefill on gfx942 (bf16/fp16, causal/full).

    Carries the port's P1-P5 levers: the 32x32x8 atom with K-loop doubling,
    conflict-free V (D128 fp16), exp2_fast + fused softmax rescale, per-config
    waves-per-eu and the D64 K-bank-conflict pad, and the persistent grid-stride
    variant. Which config gets which lever, and why, is the table in
    ``builders/gfx942/attention/prefill/README.md``.

    When ``Gfx942AttentionDenseSpec.runtime_shape`` holds (default and persistent
    grids, sliding window included), the body reads ``batch``, ``seqlen_q``,
    ``seqlen_kv``, ``num_query_heads``, and ``num_kv_heads`` from kernel params
    (``spec.runtime_param_fields``). On the persistent grid the work decode's
    fast-division magic/shift pairs follow them (``spec.runtime_kernarg_fields``).
    ``signature`` is ``attention_dense_signature``, which appends those i32s after
    ``scale``, and ``bind_torch`` launches through ``run_attention_dense_torch``,
    which packs them via ``attention_dense_runtime_args``.
    """

    return make_dense_candidate(
        arch="gfx942",
        name=CANDIDATE_NAME,
        spec_id=SPEC_ID,
        variant_id=_VARIANT_ID,
        base_spec=tuned_base_spec,
        supports=_supports,
        # Dense: causal (top-left or bottom-right), sliding-window and sinks (alone
        # or combined), optional LSE output, paged K/V. Head size stays out: D64/D128
        # coverage is ``supports_attention_dense``'s call.
        features=frozenset(
            {
                "causal",
                "causal_bottom_right",
                "sliding_window",
                "sinks",
                "lse",
                "layout_paged",
            }
        ),
        # tuned_base_spec picks these from the dtype, head size, work count and window.
        recorded=frozenset(
            {"persistent", "waves_per_eu", "block_m", "block_n", "window_wave_skip"}
        ),
    )


def _make_gfx942_attention_dense_qbminor_candidate() -> KernelCandidate:
    """The dense candidate pinned to the ingestor's grid (see
    :func:`_qbminor_base_spec`)."""

    return make_dense_candidate(
        arch="gfx942",
        name=QBMINOR_CANDIDATE_NAME,
        spec_id=QBMINOR_SPEC_ID,
        variant_id=_QBMINOR_VARIANT_ID,
        base_spec=_qbminor_base_spec,
        supports=_supports,
        features=frozenset(
            {"causal", "causal_bottom_right", "sliding_window", "sinks", "lse"}
        ),
        recorded=frozenset({"waves_per_eu"}),
    )


def _make_gfx942_attention_dense_aot_candidate(
    name: str, spec_id: str, variant_id: str, persistent: bool, nonpersist_decode: str
) -> KernelCandidate:
    """One more grid the ingestor launches (see :func:`ingestor_base_spec`).

    The persistent grid walks the whole dense knob space bar the CTA count,
    which the dense rules scope out of a base whose decode is pinned. A grid
    at another block order offers its default spec only: the qb_minor
    candidate walks the tile and the other knobs on the default grid.
    """
    from functools import partial

    return make_dense_candidate(
        arch="gfx942",
        name=name,
        spec_id=spec_id,
        variant_id=variant_id,
        base_spec=partial(
            ingestor_base_spec,
            persistent=persistent,
            nonpersist_decode=nonpersist_decode,
        ),
        supports=_supports,
        features=frozenset(
            {"causal", "causal_bottom_right", "sliding_window", "sinks", "lse"}
        ),
        recorded=frozenset({"waves_per_eu"}),
        axis_names=None if persistent else frozenset(),
    )


def register(route: CandidateRegistry, execution: CandidateRegistry) -> None:
    for candidate in (
        _make_gfx942_attention_dense_candidate(),
        _make_gfx942_attention_dense_qbminor_candidate(),
        *(_make_gfx942_attention_dense_aot_candidate(*grid) for grid in _AOT_GRIDS),
    ):
        route.register(candidate)
        execution.register(candidate)
