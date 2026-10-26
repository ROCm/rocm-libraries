# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Attention / FMHA dispatcher family (path-level selection).

Worked implementation mirroring :mod:`rocke.dispatch.gemm.bf16_rcr`, backed by
:mod:`kernels.common.attention_unified` (the unified tiled FMHA emitter).

This module owns only the assembly: the registry, the entry points, and the
re-exports that make ``dispatch.attention`` one import for callers. What each
candidate *is* lives in the arch module that owns it. See :mod:`.common` for
the scope of the dispatch decision and what it deliberately defers.

Registration is explicit rather than an import side effect, so the registry
contents are a readable list, a test can assemble a registry from a subset of
arch modules, and adding an arch touches exactly one line here.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterator, Mapping, Optional, Sequence, Tuple

from rocke.dispatch.core import (
    CandidateRegistry,
    DispatchResult,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    PinRefused,
    Ranker,
    make_dispatch_result,
    make_kernel_id,
    pin_to_spec,
    spec_identity,
)
from rocke.dispatch.tuning.walk import iter_at_level, sample_count

from . import (
    generic,
    gfx942_dense,
    gfx942_unified,
    gfx950_dense,
    gfx950_unified,
    gfx1151,
    gfx1250,
    strided_decode,
)
from .candidate import (
    DENSE_ALGORITHM,
    DENSE_GRID_ALGORITHM,
    DENSE_PERSIST_ALGORITHM,
)
from .common import (
    ATTENTION_ABI_VERSION,
    ATTENTION_DIM_VOCABULARY,
    ATTENTION_FEATURES,
    UNIFIED_BLOCK_SIZES,
    UNIFIED_HEAD_SIZES,
    AttentionMaskType,
    AttentionRequest,
    AttentionSpec,
    AttentionTuningSpec,
    FAMILY,
    _device_num_cus,
    _problem,
    _request_errors,
    _resolve_num_cus,
    _selector_matches,
    canonical_arch,
)

_FAMILY = FAMILY

ATTENTION_ROUTE_REGISTRY = CandidateRegistry(
    _FAMILY, dim_vocabulary=ATTENTION_DIM_VOCABULARY
)
ATTENTION_EXECUTION_REGISTRY = CandidateRegistry(
    _FAMILY,
    dim_vocabulary=ATTENTION_DIM_VOCABULARY,
    require_build=True,
    require_torch_binding=True,
)
for _module in (
    strided_decode,
    generic,
    gfx942_dense,
    gfx942_unified,
    gfx950_dense,
    gfx950_unified,
    gfx1250,
    gfx1151,
):
    _module.register(ATTENTION_ROUTE_REGISTRY, ATTENTION_EXECUTION_REGISTRY)
# Compatibility alias: production auto-dispatch and candidate listing.
ATTENTION_REGISTRY = ATTENTION_ROUTE_REGISTRY


def attention_candidates() -> Tuple[KernelCandidate, ...]:
    return ATTENTION_REGISTRY.candidates()


def attention_execution_candidates() -> Tuple[KernelCandidate, ...]:
    return ATTENTION_EXECUTION_REGISTRY.candidates()


def _tuning_id_filter(prefix: str):
    if not prefix:
        return None
    return lambda spec: getattr(spec, "tuning_id", None) is None or str(
        spec.tuning_id
    ).startswith(prefix)


def iter_registered_attention_combos(
    req: AttentionRequest,
    *,
    candidate_prefix: str = "",
    tuning_id_prefix: str = "",
    tuning_sample: int = 0,
    seed: int = 0,
    sweep_level: str = "production",
) -> Iterator[Tuple[KernelCandidate, object]]:
    """Yield each executable ``(candidate, spec)`` that can launch ``req``.

    Delegates the opt-in probe and ``sweep_space`` expansion to
    :meth:`CandidateRegistry.iter_combos` on :data:`ATTENTION_EXECUTION_REGISTRY`.
    Routing-only unified path labels are not executable and are omitted.
    ``req.algorithm`` / ``req.spec_id`` still filter when they are not ``auto``.

    ``sweep_level="production"`` (the default) walks each candidate's
    production set and ignores ``tuning_sample``. ``sweep_level="full"`` draws
    ``tuning_sample`` random feasible specs per candidate from every knob, dead
    ends included (0 walks that stream, which is millions of specs per shape on
    the transposed unified paths and on dense).

    The level is in effect only while this stream advances, never while it is
    suspended, so interleaving sweeps at different levels is safe.
    """
    if not isinstance(req, AttentionRequest):
        raise TypeError(f"expected AttentionRequest, got {type(req).__name__}")
    keep = _tuning_id_filter(tuning_id_prefix)
    combos = iter_at_level(
        sweep_level,
        lambda: ATTENTION_EXECUTION_REGISTRY.iter_combos(
            req,
            candidate_prefix=candidate_prefix,
            sample=sample_count(sweep_level, tuning_sample),
            seed=seed,
        ),
    )
    for candidate, spec in combos:
        if keep is None or keep(spec):
            yield candidate, spec


def registered_attention_combos(
    req: AttentionRequest,
    *,
    candidate_prefix: str = "",
    tuning_id_prefix: str = "",
    tuning_sample: int = 0,
    seed: int = 0,
    sweep_level: str = "production",
) -> Tuple[Tuple[KernelCandidate, object], ...]:
    """Materialized :func:`iter_registered_attention_combos`."""
    return tuple(
        iter_registered_attention_combos(
            req,
            candidate_prefix=candidate_prefix,
            tuning_id_prefix=tuning_id_prefix,
            tuning_sample=tuning_sample,
            seed=seed,
            sweep_level=sweep_level,
        )
    )


def attention_dispatch_result(
    req: AttentionRequest, candidate: KernelCandidate, spec: object
) -> DispatchResult:
    """Wrap an already-selected executable ``(candidate, spec)`` as a result.

    The stored request is pinned to ``spec`` (see
    :func:`rocke.dispatch.core.pin_to_spec`), not the caller's original
    ``algorithm='auto'`` request.
    """
    pinned = pin_to_spec(req, candidate, spec)
    return make_dispatch_result(
        pinned,
        candidate,
        spec,
        kernel_id=_kernel_id(pinned, candidate, spec),
        headline=f"sweep {candidate.name} ({candidate.algorithm}) on {req.arch}",
    )


def iter_dispatch_attention_all(
    req: AttentionRequest,
    *,
    candidate_prefix: str = "",
    tuning_id_prefix: str = "",
    tuning_sample: int = 0,
    seed: int = 0,
    sweep_level: str = "production",
) -> Iterator[DispatchResult]:
    """Yield each :func:`attention_dispatch_result` for ``req``."""
    if _request_errors(req, allow_unequal_head_dims=True):
        return
    yield from iter_at_level(
        sweep_level,
        lambda: ATTENTION_EXECUTION_REGISTRY.iter_dispatch_all(
            req,
            kernel_id=_kernel_id,
            candidate_prefix=candidate_prefix,
            spec_filter=_tuning_id_filter(tuning_id_prefix),
            sample=sample_count(sweep_level, tuning_sample),
            seed=seed,
        ),
    )


def _executable_candidate(req: AttentionRequest, spec_id: str) -> KernelCandidate:
    wanted = spec_id.strip().lower()
    for candidate in ATTENTION_EXECUTION_REGISTRY.candidates():
        if candidate.spec_id == wanted:
            return candidate
    raise PinRefused(req, wanted, {})


def attention_tuning_spec(
    req: AttentionRequest,
    spec_id: str,
    tuning_id: str = "auto",
    knobs: Optional[Mapping[str, object]] = None,
) -> AttentionTuningSpec:
    """The spec ``dispatch_attention`` returns for ``req`` pinned to
    ``spec_id`` and one configuration of it: ``knobs`` (rebuilt directly, and
    checked against ``tuning_id`` unless that is ``auto``), else ``tuning_id``
    (``auto``: the candidate's default spec).

    Raises :class:`rocke.dispatch.core.PinRefused` (a ``ValueError``) with the
    candidate's reason when it refuses.
    """
    candidate = _executable_candidate(req, spec_id)
    pinned = replace(
        req,
        algorithm=candidate.algorithm,
        spec_id=candidate.spec_id,
        tuning_id=tuning_id,
        tuning_knobs=dict(knobs or {}),
    )
    ok, why = candidate.admits(pinned)
    if not ok:
        raise PinRefused(pinned, candidate.spec_id, {candidate.name: why})
    return dispatch_attention(pinned).spec


def tuning_spec_with_knobs(
    req: AttentionRequest, spec_id: str, knobs: Mapping[str, object]
) -> AttentionTuningSpec:
    """One explicit point of ``spec_id``'s knob space: its default spec with
    ``knobs`` (kernel-spec fields, ``waves_per_eu`` included) applied, in the
    canonical form a sweep emits, with that sweep's ``tuning_id``.

    Raises ``ValueError`` when the candidate refuses the knobs.
    """
    return attention_tuning_spec(req, spec_id, knobs=knobs)


def _kernel_id(
    req: AttentionRequest, candidate: KernelCandidate, spec: object
) -> KernelId:
    return make_kernel_id(req, candidate, spec, op="attention")


def attention_sweep_space(
    req: OperatorRequest,
    *,
    candidate_prefix: str = "",
    tuning_id_prefix: str = "",
    tuning_sample: int = 0,
    seed: int = 0,
    limit: int = 0,
    sweep_level: str = "production",
) -> Sequence[object]:
    """Every concrete engine/configuration available to a sweep.

    Unlike normal dispatch, this deliberately probes opt-in executable
    candidates and expands ``candidate.sweep_space``. Production
    ``algorithm='auto'`` selection remains on ``ATTENTION_ROUTE_REGISTRY.supported``
    and never sees tuning candidates. ``sweep_level`` selects the curated
    stacks (``production``) or the sampled full knob space (``full``).
    ``limit`` stops after that many specs; the stream is only materialized
    up to what is returned.
    """
    if _request_errors(req, allow_unequal_head_dims=True):
        return ()
    assert isinstance(req, AttentionRequest)
    specs = []
    seen = set()
    for _candidate, spec in iter_registered_attention_combos(
        req,
        candidate_prefix=candidate_prefix,
        tuning_id_prefix=tuning_id_prefix,
        tuning_sample=tuning_sample,
        seed=seed,
        sweep_level=sweep_level,
    ):
        # This API feeds the unified paged-attention harness, whose input ABI
        # requires a 2D/3D spec. Dense and WMMA remain available through
        # registered_attention_combos / dispatch_attention_all.
        if getattr(spec, "path", "") not in ("2d", "3d"):
            continue
        h = spec_identity(spec)
        if h not in seen:
            seen.add(h)
            specs.append(spec)
            if limit and len(specs) >= limit:
                break
    return tuple(specs)


def dispatch_attention_all(
    req: AttentionRequest,
    *,
    candidate_prefix: str = "",
    tuning_id_prefix: str = "",
    tuning_sample: int = 0,
    seed: int = 0,
    sweep_level: str = "production",
) -> Tuple[DispatchResult, ...]:
    """Every eligible attention kernel for ``req``, including opt-in variants.

    Uses :func:`registered_attention_combos`, so every result carries an
    executable ``AttentionTuningSpec``. Production :func:`dispatch_attention`
    is unchanged.
    """
    return tuple(
        iter_dispatch_attention_all(
            req,
            candidate_prefix=candidate_prefix,
            tuning_id_prefix=tuning_id_prefix,
            tuning_sample=tuning_sample,
            seed=seed,
            sweep_level=sweep_level,
        )
    )


def priority_ranker(
    request: OperatorRequest, candidates: Sequence[KernelCandidate]
) -> Sequence[KernelCandidate]:
    """Honor registered ``(priority, name)`` order (identity over ``supported``)."""
    return candidates


def dispatch_attention(
    req: AttentionRequest, *, ranker: Ranker | None = None
) -> DispatchResult:
    """Select the attention kernel for ``req``.

    Unpinned (``algorithm`` / ``spec_id`` left at ``auto``) it returns the
    unified 2D-tiled or 3D split-KV path, a pure function of the problem whose
    CTA geometry the instance builder resolves (see :mod:`.common`). A tuned
    kernel -- a dense candidate or a unified tuning geometry -- is selected only
    by pinning ``algorithm`` + ``spec_id``; ``tuning_id`` (with the knobs
    recorded next to it in ``tuning_knobs``) then names one configuration of
    it (``auto``: its default spec).

    A pin that no longer resolves -- a ``spec_id`` removed from the catalog,
    an unknown id, knobs that do not reproduce the id's ``config_key``
    (changed knobs, or changed defaults under them) -- raises
    :class:`rocke.dispatch.core.PinRefused` with the candidate's reason. It
    never falls back to another kernel or configuration; the caller decides
    whether to re-sweep or retry with ``algorithm="auto"``.

    ``ranker`` is the engine-level selection seam; it defaults to
    :func:`priority_ranker`.
    """
    candidate = ATTENTION_REGISTRY.select(req, ranker=ranker or priority_ranker)
    spec = candidate.select_spec(req)
    request = (
        pin_to_spec(req, candidate, spec)
        if isinstance(spec, AttentionTuningSpec)
        else req
    )
    # Standalone kernels (gfx1250 WMMA) return their builder's spec, which has
    # no `path` -- selecting one *is* the decision, with nothing left to route.
    path = getattr(spec, "path", "")
    selected = f"{path} path" if path else candidate.algorithm
    return make_dispatch_result(
        request,
        candidate,
        spec,
        kernel_id=_kernel_id(request, candidate, spec),
        headline=f"selected {candidate.name} ({selected}) on {req.arch}",
    )


def dispatch_for_arches(
    req: AttentionRequest,
    arches: str | Sequence[str],
    *,
    ranker: Ranker | None = None,
    strict: bool = True,
) -> Dict[str, "DispatchResult | ValueError"]:
    """Run :func:`dispatch_attention` for each of ``arches``.

    Returns a ``{canonical_arch: DispatchResult}`` mapping so the caller can
    compare selections across architectures from a single host.

    .. note:: **The arch is taken from the request, never from the device** --
       that part is unconditional. One host read remains: with ``req.num_cus ==
       0`` the CU count is auto-resolved, and on a box whose arch matches the
       target arch that resolution reads the live device (see
       :func:`~.common._resolve_num_cus`). Since ``num_cus`` drives 2D/3D routing
       and ``num_segments``, the same ``(req, arch)`` pair can select a different
       kernel on a gfx942 box, a gfx950 box, and a CPU box. **Pass an explicit
       ``num_cus`` (or ``target_ctas``) to make the comparison reproducible and
       fully host-independent.**

    Example::

        results = dispatch_for_arches(req, ["gfx942", "gfx950"])
        print(results["gfx942"].candidate.name)
        print(results["gfx950"].candidate.name)

    Each arch is canonicalized via :func:`canonical_arch` (strips suffixes such
    as ``:sramecc+``, normalizes case) before dispatch, so ``"GFX950"`` and
    ``"gfx950:sramecc+"`` both appear in the output under the key ``"gfx950"``.
    Duplicate arches after canonicalization are dispatched only once, and
    insertion order is preserved so the output reads in the caller's order.

    Args:
        req:    Base :class:`AttentionRequest`. Its ``arch`` field is overridden
                for each target; all other fields are preserved.
        arches: A sequence of arch names (``["gfx942", "gfx950"]``) or, for CLI
                and config callers, one comma-separated string
                (``"gfx942,gfx950"``). Splitting happens only for the ``str``
                form -- a sequence is used as given, so an arch name is never
                silently cut in half.
        ranker: Optional ranker forwarded to :func:`dispatch_attention`.
        strict: ``True`` (default) propagates the first dispatch failure.
                ``False`` maps each failing arch to its exception instead, so
                one unsupported arch does not discard the results for the
                others -- the point of a cross-arch comparison. Values are then
                ``DispatchResult | ValueError``; test with ``isinstance``.
                Only ``ValueError`` (the dispatcher's "unsupported" answer,
                including ``PinRefused``) is collected: any other exception is a
                bug on the dispatch path and propagates under both settings
                rather than being filed as an unsupported arch.

    Raises:
        ValueError: if ``arches`` is empty, or an entry does not canonicalize.
                    This is a malformed argument, not a per-arch dispatch
                    failure, so ``strict=False`` does not suppress it.
    """
    if isinstance(arches, str):
        # CLI boundary only: --arches gfx942,gfx950. A Sequence[str] is already
        # the structured form and must not be re-split.
        requested = arches.split(",")
    else:
        requested = list(arches)
        # canonical_arch() truncates a target ID at its first comma, so an entry
        # like "gfx942,gfx950" would quietly canonicalize to "gfx942" and the
        # second arch would vanish from the results. Reject it instead.
        joined = [a for a in requested if isinstance(a, str) and "," in a]
        if joined:
            raise ValueError(
                f"comma-joined entries in a sequence of arches: {joined!r}. "
                "Pass the whole thing as one string, or split it yourself"
            )
    if not requested:
        raise ValueError("arches is empty: nothing to dispatch for")

    results: Dict[str, "DispatchResult | ValueError"] = {}
    for raw in requested:
        arch = canonical_arch(raw)
        if not arch:
            raise ValueError(f"invalid arch {raw!r} in arches {arches!r}")
        if arch in results:
            continue  # deduplicate after canonicalization
        try:
            results[arch] = dispatch_attention(replace(req, arch=arch), ranker=ranker)
        except ValueError as exc:
            # ``ValueError`` ONLY, deliberately -- it is the dispatcher's "this
            # arch cannot run this request" answer (``PinRefused`` subclasses it).
            # A bare ``except Exception`` would also swallow ``TypeError`` /
            # ``AttributeError``, i.e. a bug in the dispatch path, and report it
            # in the results mapping as though the arch were merely unsupported.
            # That is the failure mode that let the missing-arch ``TypeError`` in
            # ``gfx942_unified`` read as an unsupported arch instead of a crash.
            if strict:
                raise
            results[arch] = exc
    return results


__all__ = [
    "DENSE_ALGORITHM",
    "DENSE_GRID_ALGORITHM",
    "DENSE_PERSIST_ALGORITHM",
    "ATTENTION_ABI_VERSION",
    "ATTENTION_DIM_VOCABULARY",
    "ATTENTION_EXECUTION_REGISTRY",
    "ATTENTION_FEATURES",
    "ATTENTION_REGISTRY",
    "ATTENTION_ROUTE_REGISTRY",
    "UNIFIED_BLOCK_SIZES",
    "UNIFIED_HEAD_SIZES",
    "AttentionMaskType",
    "AttentionRequest",
    "AttentionSpec",
    "AttentionTuningSpec",
    "attention_candidates",
    "attention_execution_candidates",
    "attention_dispatch_result",
    "attention_sweep_space",
    "attention_tuning_spec",
    "dispatch_attention",
    "dispatch_for_arches",
    "dispatch_attention_all",
    "iter_dispatch_attention_all",
    "iter_registered_attention_combos",
    "priority_ranker",
    "registered_attention_combos",
    "tuning_spec_with_knobs",
]
