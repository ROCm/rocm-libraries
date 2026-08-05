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
from kernels.common.attention_unified import UnifiedAttentionProblem

from .protocol import ShapeEntry

_REQUEST_FIELDS = frozenset(f.name for f in fields(AttentionRequest))
_PROBLEM_FIELDS = frozenset(f.name for f in fields(UnifiedAttentionProblem))


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
