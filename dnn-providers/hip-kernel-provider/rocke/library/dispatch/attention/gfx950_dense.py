# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""gfx950 dense attention candidates (CDNA4, wave64, 32x32 MFMA + dense persistent).

Unified-kernel candidates for gfx950 live in :mod:`.gfx950_unified`.

Dense prefill is registered as one candidate per frozen (tile x persist x
wide-DMA) combo. ``dispatch_attention`` still picks a single winner; the
auto-policy ranker keeps unpinned serving on the historical best-config path
(default tile, persist once the grid fills, wide DMA on aligned causal D128).
Benchmarking enumerates every registered combo via
``registered_attention_combos``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Sequence, Tuple

from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    KernelCandidate,
    OperatorRequest,
)

from .common import (
    ATTENTION_ABI_VERSION,
    AttentionMaskType,
    AttentionRequest,
    FAMILY,
    _parse_attention_mask_type,
    _request_errors,
    _resolve_dense_waves_per_eu,
)

# Family id shared by every gfx950 dense variant. Pinning this spec_id (with
# algorithm auto or attention_dense) is the same opt-in as naming the
# production candidate; the ranker then picks among the variants that admit.
GFX950_DENSE_FAMILY_SPEC_ID = "gfx950_attention_dense"
_PRODUCTION_VARIANT_ID = "persist_widedma_default"
_DENSE_LAYOUT = "default"
_ON_OFF_AUTO = frozenset({"auto", "on", "off"})
_TILE_PINS = frozenset({"auto", "default", "bm128"})


@dataclass(frozen=True)
class Gfx950DenseVariant:
    """One registered gfx950 dense combo: tile geometry x persist x wide DMA."""

    variant_id: str
    tile: str
    persistent: bool
    wide_lds_dma: bool

    @property
    def candidate_name(self) -> str:
        if self.variant_id == _PRODUCTION_VARIANT_ID:
            return "attention_gfx950_dense"
        return f"attention_gfx950_dense_{self.variant_id}"

    @property
    def spec_id(self) -> str:
        if self.variant_id == _PRODUCTION_VARIANT_ID:
            return GFX950_DENSE_FAMILY_SPEC_ID
        return f"gfx950_attention_dense_{self.variant_id}"


# Frozen catalog. Wide DMA is persist-only; there is no persist_decode axis
# (decode stays ``auto`` inside each persistent spec).
GFX950_DENSE_VARIANTS: Tuple[Gfx950DenseVariant, ...] = (
    Gfx950DenseVariant("grid_default", "default", False, False),
    Gfx950DenseVariant("persist_default", "default", True, False),
    Gfx950DenseVariant("persist_widedma_default", "default", True, True),
    Gfx950DenseVariant("grid_bm128", "bm128", False, False),
    Gfx950DenseVariant("persist_bm128", "bm128", True, False),
    Gfx950DenseVariant("persist_widedma_bm128", "bm128", True, True),
)
GFX950_DENSE_VARIANT_BY_NAME = {v.candidate_name: v for v in GFX950_DENSE_VARIANTS}
GFX950_DENSE_VARIANT_BY_ID = {v.variant_id: v for v in GFX950_DENSE_VARIANTS}


def variant_for_candidate_name(name: str) -> Gfx950DenseVariant:
    try:
        return GFX950_DENSE_VARIANT_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(f"unknown gfx950 dense candidate {name!r}") from exc


def _parse_on_off_auto(value: str, field: str) -> str:
    v = value.strip().lower()
    if v not in _ON_OFF_AUTO:
        raise ValueError(f"{field} must be 'auto'/'on'/'off', got {value!r}")
    return v


def _parse_tile(value: str) -> str:
    v = value.strip().lower()
    if v not in _TILE_PINS:
        raise ValueError(f"dense_tile must be 'auto'/'default'/'bm128', got {value!r}")
    return v


def _pins_match(variant: Gfx950DenseVariant, req: AttentionRequest) -> bool:
    tile = _parse_tile(req.dense_tile)
    if tile != "auto" and variant.tile != tile:
        return False
    persist = _parse_on_off_auto(req.dense_persistent, "dense_persistent")
    if persist == "on" and not variant.persistent:
        return False
    if persist == "off" and variant.persistent:
        return False
    wdma = _parse_on_off_auto(req.dense_wide_lds_dma, "dense_wide_lds_dma")
    if wdma == "on" and not variant.wide_lds_dma:
        return False
    if wdma == "off" and variant.wide_lds_dma:
        return False
    return True


def _lookup_variant(
    tile: str, persistent: bool, wide_lds_dma: bool
) -> Gfx950DenseVariant:
    for variant in GFX950_DENSE_VARIANTS:
        if (
            variant.tile == tile
            and variant.persistent == persistent
            and variant.wide_lds_dma == wide_lds_dma
        ):
            return variant
    raise ValueError(
        "no gfx950 dense variant for "
        f"tile={tile!r} persistent={persistent} wide_lds_dma={wide_lds_dma}"
    )


def _auto_variant(req: AttentionRequest) -> Gfx950DenseVariant:
    """Historical serving/strata policy, expressed as a catalog entry.

    Tile is ``default`` unless pinned. Persist turns on once
    ``nqb * Hq * B >= num_persistent``. Wide DMA follows persist + D128 +
    causal + aligned + no sinks.
    """
    from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES

    tile = _parse_tile(req.dense_tile)
    if tile == "auto":
        tile = "default"
    geometry = DENSE_TILE_GEOMETRIES[tile]
    bm = int(geometry["block_m"])
    bn = int(geometry["block_n"])
    sq, sk = int(req.seqlen_q), int(req.seqlen_k)
    mask_type = _parse_attention_mask_type(req.mask_type)
    moving_bottom_right = (
        mask_type == AttentionMaskType.BOTTOM_RIGHT_CAUSAL and sq != sk
    )
    ragged = _ragged_self_attention(
        sq, sk, bm, bn, moving_bottom_right=moving_bottom_right
    )
    nqb = (sq + bm - 1) // bm
    work = nqb * int(req.nhead_q) * int(req.batch)
    np = int(req.dense_num_persistent)
    persist_mode = _parse_on_off_auto(req.dense_persistent, "dense_persistent")
    if persist_mode == "on":
        if moving_bottom_right:
            raise ValueError(
                "dense_persistent='on' is not supported with moving "
                "bottom-right causal attention"
            )
        persistent = True
    elif persist_mode == "off":
        persistent = False
    else:
        persistent = False if moving_bottom_right else work >= np
    wdma_mode = _parse_on_off_auto(req.dense_wide_lds_dma, "dense_wide_lds_dma")
    wdma_ok = _wide_dma_eligible(
        req, SimpleNamespace(persistent=persistent, ragged=ragged)
    )
    if wdma_mode == "on":
        if not persistent:
            raise ValueError(
                "dense_wide_lds_dma='on' requires a persistent dense variant"
            )
        wide = True
    elif wdma_mode == "off":
        wide = False
    else:
        wide = wdma_ok
    return _lookup_variant(tile, persistent, wide)


def select_dense_variant(req: AttentionRequest) -> Gfx950DenseVariant:
    """Variant ``dense_spec_for_request`` and the gfx950 dense ranker share."""
    preferred = _auto_variant(req)
    matching = [v for v in GFX950_DENSE_VARIANTS if _pins_match(v, req)]
    if not matching:
        raise ValueError("no gfx950 dense variant matches the request pins")
    if preferred in matching:
        return preferred
    for variant in matching:
        if (
            variant.persistent == preferred.persistent
            and variant.wide_lds_dma == preferred.wide_lds_dma
        ):
            return variant
    for variant in matching:
        if variant.persistent == preferred.persistent:
            return variant
    return matching[0]


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


def _wide_dma_eligible(req: AttentionRequest, spec) -> bool:
    mask_type = _parse_attention_mask_type(req.mask_type)
    moving_bottom_right = mask_type == AttentionMaskType.BOTTOM_RIGHT_CAUSAL and int(
        req.seqlen_q
    ) != int(req.seqlen_k)
    return (
        bool(spec.persistent)
        and int(req.hdim_q) == 128
        and int(req.hdim_v) == 128
        and req.dtype.lower() in ("fp16", "bf16")
        and mask_type != AttentionMaskType.NO_MASK
        and int(req.sliding_window) == 0
        and not bool(req.use_sinks)
        and not spec.ragged
        and not moving_bottom_right
    )


def _dense_opted_in(
    req: AttentionRequest, variant: Gfx950DenseVariant
) -> Tuple[bool, str]:
    alg = req.algorithm.strip().lower()
    sid = req.spec_id.strip().lower()
    if alg != "attention_dense" and sid not in (
        variant.spec_id,
        GFX950_DENSE_FAMILY_SPEC_ID,
    ):
        return False, "attention_dense is opt-in (algorithm='attention_dense')"
    if sid not in ("auto", variant.spec_id, GFX950_DENSE_FAMILY_SPEC_ID):
        return False, f"request spec_id {req.spec_id!r} != {variant.spec_id!r}"
    return True, "ok"


def _dense_spec(req: OperatorRequest, variant: Gfx950DenseVariant | None = None):
    """Build the launch-ready ``Gfx950AttentionDenseSpec`` for ``variant``.

    Tile, persist, and wide-DMA come from the frozen variant. ``persist_decode``
    stays on the request (``auto`` selects GQA-local mappings inside the spec).
    Non-tile-multiple self-attention lengths use the on-chip ragged path.
    ``variant=None`` keeps the one-argument call used by existing tests and
    selects the auto-policy variant.
    """
    if variant is None:
        variant = select_dense_variant(req)
    from kernels.common.attention_dense_spec import DENSE_TILE_GEOMETRIES
    from kernels.gfx950.attention_dense import (
        GFX950_DENSE_LAYOUTS,
        Gfx950AttentionDenseSpec,
    )

    assert isinstance(req, AttentionRequest)
    if req.arch != "gfx950":
        raise ValueError(
            f"gfx950 dense spec factory requires arch='gfx950', got {req.arch!r}"
        )
    sq, sk = int(req.seqlen_q), int(req.seqlen_k)
    sw = int(req.sliding_window)
    use_sinks = bool(req.use_sinks)
    geometry = DENSE_TILE_GEOMETRIES[variant.tile]
    layout = GFX950_DENSE_LAYOUTS[_DENSE_LAYOUT]
    bm = int(geometry["block_m"])
    bn = int(geometry["block_n"])
    decode = req.dense_persist_decode.strip().lower()
    mask_type = _parse_attention_mask_type(req.mask_type)
    causal = mask_type != AttentionMaskType.NO_MASK
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
        causal=causal,
        dtype=req.dtype.lower(),
        block_m=bm,
        block_n=bn,
        waves_per_eu=_resolve_dense_waves_per_eu(req, 2),
        lds_v_row_pad=int(layout["lds_v_row_pad"]),
        persistent=variant.persistent,
        num_persistent=int(req.dense_num_persistent),
        persist_decode=decode,
        ragged=ragged,
        sliding_window=sw,
        use_sinks=use_sinks,
        wide_lds_dma=variant.wide_lds_dma,
        causal_bottom_right=moving_bottom_right,
    )


def dense_spec_for_request(req: AttentionRequest):
    """Public builder: the launch-ready dense spec for ``req`` at the variant
    selected by pins + the historical auto policy. Pair with
    ``run_attention_dense_torch`` to execute the dispatched dense candidate."""
    return _dense_spec(req, select_dense_variant(req))


def dense_spec_for_candidate(req: AttentionRequest, candidate: KernelCandidate):
    """Launch-ready dense spec for a specific registered gfx950 dense candidate."""
    return _dense_spec(req, variant_for_candidate_name(candidate.name))


def rank_dense_variants(
    request: OperatorRequest, candidates: Sequence[KernelCandidate]
) -> Tuple[KernelCandidate, ...]:
    """Prefer the auto-policy gfx950 dense variant; leave every other candidate
    in registered ``(priority, name)`` order.

    Static ``(priority, name)`` would always pick ``attention_gfx950_dense``
    (the persist+wide-DMA default-tile name) even for short seq / D=64 / SWA.
    """
    preferred = None
    if isinstance(request, AttentionRequest) and request.arch == "gfx950":
        try:
            preferred = select_dense_variant(request)
        except ValueError:
            preferred = None

    def key(candidate: KernelCandidate) -> Tuple[int, int, str]:
        if preferred is None or candidate.algorithm != "attention_dense":
            return (1, candidate.priority, candidate.name)
        variant = GFX950_DENSE_VARIANT_BY_NAME.get(candidate.name)
        return (0 if variant is preferred else 1, candidate.priority, candidate.name)

    return tuple(sorted(candidates, key=key))


def _make_gfx950_attention_dense_candidate(
    variant: Gfx950DenseVariant,
) -> KernelCandidate:
    """One gfx950 dense combo. OPT-IN ONLY: never selected under
    ``algorithm="auto"`` unless ``spec_id`` names this variant or the family
    id ``gfx950_attention_dense``.
    """
    spec_id = variant.spec_id
    name = variant.candidate_name

    def support(req: OperatorRequest) -> Tuple[bool, str]:
        errors = _request_errors(req)
        if errors:
            return False, "; ".join(errors)
        assert isinstance(req, AttentionRequest)
        try:
            ok, why = _dense_opted_in(req, variant)
            if not ok:
                return False, why
            if not _pins_match(variant, req):
                return False, "request pins exclude this dense variant"
            spec = _dense_spec(req, variant)
        except ValueError as e:
            return False, str(e)
        if variant.wide_lds_dma and not _wide_dma_eligible(req, spec):
            return False, (
                "wide-DMA variant requires aligned causal D128, "
                "no sinks, no sliding_window"
            )
        from kernels.gfx950.attention_dense import supports_attention_dense

        ok, why = supports_attention_dense(spec, arch=req.arch)
        if not ok:
            return False, why
        return True, "ok"

    def select(req: OperatorRequest):
        ok, why = candidate.admits(req)
        if not ok:
            raise ValueError(f"{name} does not support request: {why}")
        return _dense_spec(req, variant)

    def build(spec, arch):
        from kernels.gfx950.attention_dense import build_attention_dense

        return build_attention_dense(spec, arch=arch)

    def signature(spec):
        from kernels.gfx950.attention_dense import attention_dense_signature

        return attention_dense_signature(spec)

    def grid(spec, req):
        from kernels.gfx950.attention_dense import attention_dense_grid

        return attention_dense_grid(spec)

    def block(spec):
        from kernels.gfx950.attention_dense import attention_dense_block

        return attention_dense_block(spec)

    def bind_torch(request, spec, tensors, **kwargs):
        from .bindings import bind_dense_attention_torch

        return bind_dense_attention_torch(request, spec, tensors, **kwargs)

    def _tuning_kwargs(req: AttentionRequest) -> dict:
        from kernels.gfx950.attention_dense import supports_attention_dense

        return dict(
            arch="gfx950",
            supports=supports_attention_dense,
            wpe_pinned=int(req.dense_waves_per_eu) != 0,
        )

    def sweep(req: OperatorRequest):
        if not candidate.admits(req)[0]:
            return ()
        assert isinstance(req, AttentionRequest)
        from .tuning_common import iter_dense_tuning_specs

        return iter_dense_tuning_specs(select(req), **_tuning_kwargs(req))

    def sample(req: OperatorRequest, n: int, seed: int):
        if not candidate.admits(req)[0]:
            return ()
        assert isinstance(req, AttentionRequest)
        from .tuning_common import sample_dense_tuning_specs

        return sample_dense_tuning_specs(
            select(req), n, seed, salt=name, **_tuning_kwargs(req)
        )

    candidate = KernelCandidate(
        name=name,
        family=FAMILY,
        algorithm="attention_dense",
        spec_id=spec_id,
        abi_version=ATTENTION_ABI_VERSION,
        priority=3,
        capability=Capability(
            arches=("gfx950",),
            dtypes=("bf16", "fp16"),
            # Only frozen grid variants implement a moving bottom-right diagonal.
            supports_features=frozenset(
                {"causal", "sliding_window", "sinks"}
                | ({"causal_bottom_right"} if not variant.persistent else set())
            ),
        ),
        _supports=support,
        select_spec=select,
        signature=signature,
        grid=grid,
        block=block,
        sweep_space=sweep,
        sample_space=sample,
        build=build,
        bind_torch=bind_torch,
    )
    return candidate


def register_route(registry: CandidateRegistry) -> None:
    for variant in GFX950_DENSE_VARIANTS:
        registry.register(_make_gfx950_attention_dense_candidate(variant))


def register(registry: CandidateRegistry) -> None:
    register_route(registry)
