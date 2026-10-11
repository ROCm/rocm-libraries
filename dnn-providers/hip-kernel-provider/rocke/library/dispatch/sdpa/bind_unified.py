# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""SDPA binding for the unified attention candidates (ST2 / U-03).

Narrows the SDPA view of the existing unified dispatch.  The existing
dispatch/attention/ candidates are not edited; this module applies
SDPA-specific predicates on top of the registry's own verdict.

Rule added in U-03:
  fp16 d=256  -> declined (scalar fallback; fails tolerance).  ATT-34.

bf16 d=256 is served on gfx950 by the dedicated ``attention_gfx950_d256``
(prefill) and ``attention_d256_decode`` candidates, and on gfx942 by the
``_d256_gfx942_fast`` cohort inside the generic unified kernel -- gfx942 has
no separate d256 candidate, so a generic ``attention_unified_2d`` verdict is
the correct answer there, not a fallback.

Arches with no dedicated candidate (gfx1151, gfx1201, ...) ride the two
generic path candidates, which route them to the arch-neutral scalar kernel.
That kernel serves fp16 and bf16 alike -- ``supports_native_unified_attention``
admits ``dtype in {fp16, bf16}`` with no arch axis -- so this binding adds no
per-arch dtype rule for them. The gfx1151 WMMA FMHA kernels under
``builders/gfx1151/`` are a standalone optimization case study and are not
registered as dispatch candidates; if RD-01 (ST8) wires one in, the fp16-only
gap becomes real and the rule expressing it belongs with that change.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from rocke.core.arch import known_arches

from dispatch.attention import (
    ATTENTION_ROUTE_REGISTRY,
    UNIFIED_HEAD_SIZES,
    AttentionRequest,
)

if TYPE_CHECKING:
    from .common import SdpaRequest

# Candidate support in this registry is shape-dependent: the specialized
# candidates are cohort-gated, so a degenerate request reaches only the generic
# fallback and hides every fast path. SdpaRequest carries no shape fields (ST9
# owns that schema), so probe one representative prefill shape and one
# representative decode shape instead, and admit if either is served.
#
# The head counts are load-bearing, not filler: at nhead_q=1 a 2048x2048
# request routes to the 3D decode path, which drops attention_gfx950_d256 from
# the verdict. 32/8 is a realistic GQA ratio and routes prefill to 2D.
#
# TODO(ST9/DV-01): these constants exist only because the request has no shape.
# When SdpaRequest gains batch/nhead/seqlen fields, _probe_request must read
# them off the request and this probe loop must collapse to a single real
# shape -- two representative shapes cannot prove support for all shapes, so
# until then a request outside both cohorts can be admitted here and still
# fail downstream.
_PROBE_NHEAD_Q = 32
_PROBE_NHEAD_K = 8
_PROBE_SHAPES = (
    (2048, 2048),  # prefill
    (1, 4096),  # decode
)

# ── SDPA-level decline predicates ─────────────────────────────────────────────

_FP16_D256_REASON = (
    "fp16 d=256 falls back to the scalar kernel, which fails tolerance on some "
    "shapes and is slow enough at long sequence lengths to stall graph capture "
    "(ATT-34); use bf16 at d=256, or fp16 at d in {64, 128}"
)


def _sdpa_unified_rejects(req: "SdpaRequest") -> "tuple[bool, str] | None":
    """Return ``(False, reason)`` if this request cannot be served, else ``None``.

    The single gate shared by :func:`sdpa_unified_admits` and
    :func:`sdpa_unified_candidates`, so the two cannot drift apart on what they
    reject. Order matters and is load-bearing: the request is *validated*
    before any SDPA policy rule runs, so an out-of-vocabulary head_dim or an
    unknown arch reports its own cause rather than borrowing the ATT-34 reason
    from a policy rule that happens to match the same request.

    Rules here are in addition to, not instead of, the per-candidate Capability
    and support-closure checks in dispatch/attention/.
    """
    # 1. Request validity, cheapest first.
    if req.head_dim not in UNIFIED_HEAD_SIZES:
        return False, (
            f"head_dim {req.head_dim} is not a unified head size; "
            f"supported: {sorted(UNIFIED_HEAD_SIZES)}"
        )
    if req.arch not in known_arches():
        return False, f"unknown arch {req.arch!r}"

    # 2. SDPA policy declines, applied only to requests that are otherwise sane.
    #    dtype is already normalized to "fp16" by SdpaRequest.__post_init__.
    if req.dtype == "fp16" and req.head_dim == 256:
        return False, _FP16_D256_REASON

    return None


def _probe_request(
    req: "SdpaRequest", seqlen_q: int, seqlen_k: int
) -> AttentionRequest:
    """Build a representative AttentionRequest at a probe shape.

    TODO(ST9/DV-01): batch and the head counts are stand-ins for fields
    SdpaRequest does not carry yet; take them from ``req`` once it does.
    """
    return AttentionRequest(
        batch=1,
        nhead_q=_PROBE_NHEAD_Q,
        nhead_k=_PROBE_NHEAD_K,
        seqlen_q=seqlen_q,
        seqlen_k=seqlen_k,
        hdim_q=req.head_dim,
        hdim_v=req.head_dim,
        arch=req.arch,
        dtype=req.dtype,
    )


def sdpa_unified_admits(req: "SdpaRequest") -> "tuple[bool, str]":
    """Top-level SDPA admission check for the unified family.

    Called by the SDPA dispatcher (ST9) and directly by tests.
    Returns ``(True, "ok")`` only when:
      - nothing in :func:`_sdpa_unified_rejects` applies, AND
      - at least one unified candidate admits a representative shape.

    Does not select a candidate or produce a spec; that is ST9's job.
    """
    rejected = _sdpa_unified_rejects(req)
    if rejected is not None:
        return rejected

    # ``supported()`` applies each candidate's full admits() closure and hides
    # opt-in candidates from auto selection, so it is the registry's own verdict
    # rather than a re-implementation of it.
    for seqlen_q, seqlen_k in _PROBE_SHAPES:
        if ATTENTION_ROUTE_REGISTRY.supported(_probe_request(req, seqlen_q, seqlen_k)):
            return True, "ok"

    # Backstop, not a live branch today: the two generic path candidates
    # declare known_arches() x UNIFIED_DTYPES x UNIFIED_HEAD_SIZES, and one of
    # the two probes is always 2D and the other 3D, so anything past the reject
    # gate is currently served by at least one of them. It goes live as soon as
    # ST9 adds an axis the generic candidates can refuse (sinks, sliding
    # window, fp8), so it stays.
    return False, (
        f"no unified candidate covers arch={req.arch!r} "
        f"dtype={req.dtype!r} d={req.head_dim}"
    )


def sdpa_unified_candidates(req: "SdpaRequest") -> "tuple[str, ...]":
    """Names of the unified candidates that serve ``req``, for tests and triage.

    Deduped across the probe shapes, in registry order. Empty whenever
    :func:`sdpa_unified_admits` would return ``False`` -- the two share
    :func:`_sdpa_unified_rejects`, so an invalid request yields no names here
    instead of being handed to the registry as a probe shape.
    """
    if _sdpa_unified_rejects(req) is not None:
        return ()
    names: list[str] = []
    for seqlen_q, seqlen_k in _PROBE_SHAPES:
        for candidate in ATTENTION_ROUTE_REGISTRY.supported(
            _probe_request(req, seqlen_q, seqlen_k)
        ):
            if candidate.name not in names:
                names.append(candidate.name)
    return tuple(names)
