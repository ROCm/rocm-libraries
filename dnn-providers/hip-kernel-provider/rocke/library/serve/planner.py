# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Turn a wire request into a real dispatch decision.

This is the stage that answers "would rocKE serve this shape, and with what?"
and it is deliberately the only stage that has to work everywhere. Planning runs
the production registry -- the same ``dispatch_attention`` a client calls -- so a
plan is not a summary of what rocKE might do, it is the decision itself,
reproducible on a laptop for an arch that is not attached.

That property comes from the dispatcher, not from this module: selection keys off
``req.arch`` rather than the running device, so the plan for gfx950 is the same
whether or not a gfx950 is present. It is what lets the caller learn a candidate
is unservable without paying for a GPU node to find out.

A shape rocKE declines is a normal outcome, so a rejection is returned as data
with the registry's own accumulated reasons attached, never raised.
"""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any

from dispatch.attention import AttentionRequest, dispatch_attention
from rocke.dispatch.families.moe import MoeRequest, dispatch_moe
from kernels.common.attention_unified import UnifiedAttentionProblem

from .protocol import MoeShapeEntry, ShapeEntry

_REQUEST_FIELDS = frozenset(f.name for f in fields(AttentionRequest))
_PROBLEM_FIELDS = frozenset(f.name for f in fields(UnifiedAttentionProblem))
_MOE_REQUEST_FIELDS = frozenset(f.name for f in fields(MoeRequest))

#: serve's wire spelling -> the dispatcher's field name. The wire contract names
#: the token count ``tokens``; the dispatcher names it ``num_tokens``. Both are
#: fixed by different audiences -- one by callers who already emit this JSON,
#: the other by the dim vocabulary every family shares -- so the difference is
#: translated at the boundary rather than renamed on either side.
_MOE_WIRE_ALIASES = {
    "tokens": "num_tokens",
    "experts": "num_experts",
    "topk": "top_k",
}


def build_attention_request(entry: ShapeEntry, *, arch: str) -> AttentionRequest:
    """Build the dispatch-view request, with ``arch`` authoritative.

    The arch on the envelope wins over anything in the entry: the envelope is
    what the caller resolved for the machine the work is destined for, while a
    per-entry arch is at best a copy of it.
    """
    payload = {k: v for k, v in entry.attention_request.items() if k in _REQUEST_FIELDS}
    payload["arch"] = arch
    return AttentionRequest(**payload)


def build_problem(entry: ShapeEntry, *, num_sms: int) -> UnifiedAttentionProblem:
    """Build the runtime-view problem, preserving the observed ``total_q``."""
    payload = {
        k: v for k, v in entry.problem.items() if k in _PROBLEM_FIELDS and v is not None
    }
    payload["num_sms"] = int(num_sms)
    return UnifiedAttentionProblem(**payload)


def plan_entry(entry: ShapeEntry, *, arch: str) -> dict[str, Any]:
    """Plan one shape. Never raises for an unservable shape."""
    base: dict[str, Any] = {
        "signature": entry.signature,
        "call_count": entry.call_count,
        "softmax_scale": entry.softmax_scale,
        "ragged": entry.ragged,
        "shape_provenance": entry.shape_provenance,
    }
    try:
        request = build_attention_request(entry, arch=arch)
    except (TypeError, ValueError) as exc:
        return {
            **base,
            "servable": False,
            "reason": f"malformed attention_request: {exc}",
        }

    try:
        problem = build_problem(entry, num_sms=request.num_sms)
    except (TypeError, ValueError) as exc:
        return {**base, "servable": False, "reason": f"malformed problem: {exc}"}

    try:
        decision = dispatch_attention(request)
    except ValueError as exc:
        # The registry accumulates every candidate's rejection reason into the
        # message, which is the most useful thing we can hand back.
        return {**base, "servable": False, "reason": str(exc)}

    spec = decision.spec
    return {
        **base,
        "servable": True,
        "candidate": decision.candidate.name,
        "algorithm": decision.candidate.algorithm,
        "spec_id": decision.candidate.spec_id,
        "path": getattr(spec, "path", ""),
        "kernel_name": spec.kernel_name(),
        "arch": request.arch,
        # The dispatcher owns kernel identity, so export its own dict rather
        # than restating a subset here: the field set is stable across
        # dispatcher revisions, while the derived key properties are not.
        "kernel_id": {
            **decision.kernel_id.as_dict(),
            "cache_key": decision.kernel_id.cache_key,
        },
        "explanation": list(decision.explanation),
        "problem": asdict(problem),
    }


def plan_all(entries: tuple[ShapeEntry, ...], *, arch: str) -> list[dict[str, Any]]:
    return [plan_entry(entry, arch=arch) for entry in entries]


# --------------------------------------------------------------------------
# Fused MoE
# --------------------------------------------------------------------------


def build_moe_request(entry: MoeShapeEntry, *, arch: str) -> MoeRequest:
    """Build the dispatch-view MoE request, with ``arch`` authoritative.

    Same rule as attention: the envelope's arch is what the caller resolved for
    the destination machine, so it wins over any per-entry copy.
    """
    payload = {
        _MOE_WIRE_ALIASES.get(k, k): v
        for k, v in entry.moe_request.items()
        if _MOE_WIRE_ALIASES.get(k, k) in _MOE_REQUEST_FIELDS
    }
    payload["arch"] = arch
    return MoeRequest(**payload)


def _weight_layout(spec) -> str:
    """How the plan's consumer must have laid the expert weights out.

    The dispatcher's spec states this as two per-GEMM booleans rather than one
    string, because the two GEMMs could in principle disagree. The plan reports
    the layout the caller has to *supply*, which is a single answer, so a split
    is reported as such instead of being rounded to whichever GEMM is read
    first -- a plan that quietly claimed "row_major" for a swizzled gate/up
    would be a wrong-weights run with no error.
    """
    def layout(swizzled: bool) -> str:
        return "swizzled" if swizzled else "row_major"

    gu = bool(getattr(spec, "swizzle_gu", False))
    down = bool(getattr(spec, "swizzle_down", False))
    if gu != down:
        return f"gate_up={layout(gu)},down={layout(down)}"
    return layout(gu)


def plan_moe_entry(entry: MoeShapeEntry, *, arch: str) -> dict[str, Any]:
    """Plan one MoE layer. Never raises for an unservable shape.

    Declining is the expected answer for most shapes here -- the mega-kernel
    claims only the cohort it was tuned on -- so a rejection carries the
    registry's own reasons rather than an exception.
    """
    base: dict[str, Any] = {
        "signature": entry.signature,
        "call_count": entry.call_count,
        "active_experts": entry.active_experts,
        "shape_provenance": entry.shape_provenance,
    }
    try:
        request = build_moe_request(entry, arch=arch)
    except (TypeError, ValueError) as exc:
        return {**base, "servable": False, "reason": f"malformed moe_request: {exc}"}

    try:
        decision = dispatch_moe(request)
    except ValueError as exc:
        return {**base, "servable": False, "reason": str(exc)}

    spec = decision.spec
    return {
        **base,
        "servable": True,
        "candidate": decision.candidate.name,
        "algorithm": decision.candidate.algorithm,
        "spec_id": decision.candidate.spec_id,
        "weight_layout": _weight_layout(spec),
        "kernel_name": spec.kernel_name(),
        "arch": request.arch,
        "grid": list(decision.grid),
        "block": list(decision.block),
        "kernel_id": {
            **decision.kernel_id.as_dict(),
            "cache_key": decision.kernel_id.cache_key,
        },
        "explanation": list(decision.explanation),
        "problem": dict(entry.problem),
        "spec": asdict(spec),
    }


def plan_moe_all(
    entries: tuple[MoeShapeEntry, ...], *, arch: str
) -> list[dict[str, Any]]:
    return [plan_moe_entry(entry, arch=arch) for entry in entries]


#: ``op`` -> the planner that serves it. Keeps ``__main__`` from growing a
#: branch per operator.
PLANNERS = {"attention": plan_all, "moe": plan_moe_all}
