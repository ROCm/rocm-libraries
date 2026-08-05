# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Wire format for the ``rocke-serve`` JSON subprocess contract.

The caller is an external kernel-optimization orchestrator that has already
resolved a *complete* attention problem -- one it observed in a serving process,
not one it inferred from tensor geometry alone. It hands that over as JSON,
rocKE plans/builds/measures, and the answer comes back as JSON.

This module is the whole of the format and deliberately imports neither
``kernels`` nor ``dispatch``: parsing and validating a request must not need a
GPU, a comgr, or even the rest of the library, so schema handling stays testable
on its own.

Why the caller sends two views of the same shape
------------------------------------------------
Each entry in ``requests`` carries both an ``attention_request`` and a
``problem``, and they disagree on purpose. ``attention_request`` is the
*dispatch* view, whose ``total_q`` is implicitly ``batch * seqlen_q`` (see
``dispatch.attention.common._problem``). ``problem`` is the *runtime* view and
carries the ``total_q`` actually observed. For a ragged batch -- the normal case
in continuous batching, where sequences in one launch have different query
lengths -- the observed total is strictly less than the padded product. Planning
on the padded upper bound is correct because that is what the kernel must be
able to cover; measuring on it would overstate the work. Keeping both means each
stage reads the view it is entitled to instead of one lossy compromise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REQUEST_SCHEMA = "hyperloom.rocke.serve_request/v1"
RESULT_SCHEMA = "hyperloom.rocke.serve_result/v1"

#: Fields of ``kernels.common.attention_unified.UnifiedAttentionProblem`` that
#: the caller is allowed to set. ``num_sms`` and the codegen knobs
#: (``waves_per_eu``, ``compile_backend``) are deliberately absent: those are
#: rocKE's to choose from the target, not the caller's to pin.
PROBLEM_FIELDS = (
    "total_q",
    "num_seqs",
    "num_query_heads",
    "num_kv_heads",
    "head_size",
    "block_size",
    "max_seqlen_q",
    "max_seqlen_k",
    "dtype",
    "q_dtype",
    "sliding_window",
    "softcap",
    "use_sinks",
    "use_alibi",
    "use_qq_bias",
    "use_fp8",
    "num_kv_blocks",
)

_REQUIRED_PROBLEM_FIELDS = (
    "num_seqs",
    "num_query_heads",
    "num_kv_heads",
    "head_size",
    "block_size",
    "max_seqlen_q",
    "max_seqlen_k",
    "dtype",
)


class ProtocolError(ValueError):
    """A request that cannot be interpreted at all.

    Distinct from a request rocKE understands but declines to serve: that is a
    per-entry rejection carried in the result, not an exception.
    """


@dataclass(frozen=True)
class ShapeEntry:
    """One dispatchable attention shape, with both views the caller sent."""

    attention_request: dict[str, Any]
    problem: dict[str, Any]
    call_count: int = 0
    softmax_scale: float = 0.0
    ragged: bool = False
    observed_total_q: int = 0
    request_total_q: int = 0
    shape_provenance: str = ""
    estimated_fields: tuple[str, ...] = ()

    @property
    def signature(self) -> str:
        """Compact identity used in logs and as the per-shape artifact name."""
        p = self.problem
        return (
            f"d{p.get('head_size')}_b{p.get('block_size')}"
            f"_h{p.get('num_query_heads')}kv{p.get('num_kv_heads')}"
            f"_q{p.get('max_seqlen_q')}_k{p.get('max_seqlen_k')}"
            f"_ns{p.get('num_seqs')}_tq{p.get('total_q')}"
            f"_{p.get('dtype')}"
        )

    @classmethod
    def from_dict(cls, raw: Any, *, index: int) -> "ShapeEntry":
        where = f"requests[{index}]"
        if not isinstance(raw, dict):
            raise ProtocolError(f"{where} must be an object, got {type(raw).__name__}")
        request = raw.get("attention_request")
        problem = raw.get("problem")
        for name, value in (("attention_request", request), ("problem", problem)):
            if not isinstance(value, dict) or not value:
                raise ProtocolError(f"{where}.{name} must be a non-empty object")
        missing = [f for f in _REQUIRED_PROBLEM_FIELDS if problem.get(f) in (None, "")]
        if missing:
            raise ProtocolError(f"{where}.problem is missing {missing}")
        return cls(
            attention_request=dict(request),
            problem={k: problem[k] for k in PROBLEM_FIELDS if k in problem},
            call_count=int(raw.get("call_count") or 0),
            softmax_scale=float(raw.get("softmax_scale") or 0.0),
            ragged=bool(raw.get("ragged")),
            observed_total_q=int(raw.get("observed_total_q") or 0),
            request_total_q=int(raw.get("request_total_q") or 0),
            shape_provenance=str(raw.get("shape_provenance") or ""),
            estimated_fields=tuple(raw.get("estimated_fields") or ()),
        )


@dataclass(frozen=True)
class ServeRequest:
    """A parsed, structurally valid ``rocke-serve`` request."""

    arch: str
    entries: tuple[ShapeEntry, ...]
    op: str = "attention"
    llvm_flavor: str = ""
    profile: dict[str, Any] = field(default_factory=dict)
    advisory: bool = False
    output_dir: str = ""
    budget_s: int = 1800
    num_gpus: int = 1
    kernel: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Any) -> "ServeRequest":
        if not isinstance(raw, dict):
            raise ProtocolError(f"request must be an object, got {type(raw).__name__}")
        schema = str(raw.get("schema") or "")
        if schema != REQUEST_SCHEMA:
            raise ProtocolError(
                f"unsupported schema {schema!r}; expected {REQUEST_SCHEMA!r}"
            )
        op = str(raw.get("op") or "attention")
        if op != "attention":
            raise ProtocolError(
                f"unsupported op {op!r}; rocke-serve serves attention only"
            )
        arch = str(raw.get("arch") or "").strip().lower()
        if not arch.startswith("gfx"):
            raise ProtocolError(f"arch {arch!r} is not a gfx target")
        raw_entries = raw.get("requests")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ProtocolError("requests must be a non-empty list")
        entries = tuple(
            ShapeEntry.from_dict(entry, index=i) for i, entry in enumerate(raw_entries)
        )
        return cls(
            arch=arch,
            entries=entries,
            op=op,
            llvm_flavor=str(raw.get("llvm_flavor") or ""),
            profile=dict(raw.get("profile") or {}),
            advisory=bool(raw.get("advisory")),
            output_dir=str(raw.get("output_dir") or ""),
            budget_s=int(raw.get("budget_s") or 1800),
            num_gpus=int(raw.get("num_gpus") or 1),
            kernel=dict(raw.get("kernel") or {}),
        )


def make_result(
    *,
    status: str,
    report: str = "",
    micro_speedup: float | None = None,
    correctness_passed: bool | None = None,
    artifact_path: str = "",
    plans: list[dict[str, Any]] | None = None,
    measurements: list[dict[str, Any]] | None = None,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    """Build the result object.

    ``micro_speedup`` and ``correctness_passed`` are ``None`` rather than
    neutral defaults whenever the corresponding lane did not run. The caller
    promotes a kernel on this evidence, so "not measured" must not arrive
    looking like "measured, and it broke even" or "measured, and it passed".
    """
    return {
        "schema": RESULT_SCHEMA,
        "status": status,
        "micro_speedup": micro_speedup,
        "correctness_passed": correctness_passed,
        "artifact_path": artifact_path,
        "report": report,
        "plans": plans or [],
        "measurements": measurements or [],
        "reasons": reasons or [],
    }
