# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Legality analysis for normalized fusion graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .fusion_ir import FusionGraph, FusionOp


@dataclass(frozen=True)
class LegalResult:
    ok: bool
    reasons: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()

    @classmethod
    def success(cls, warnings=()) -> "LegalResult":
        return cls(ok=True, warnings=tuple(warnings))

    @classmethod
    def failure(cls, *reasons: str, warnings=()) -> "LegalResult":
        return cls(ok=False, reasons=tuple(reasons), warnings=tuple(warnings))


class FusionLegalizer:
    """Small deterministic legality pass.

    The pass is deliberately conservative: unsupported dtype/layout/op
    combinations are rejected with reasons rather than silently falling
    back to unsafe codegen. Each failure carries a human-readable
    reason -- the :class:`LegalResult` is the contract :func:`explain_fn`
    uses to answer "why didn't this fuse?"
    """

    supported_dtypes = {"fp16", "f16", "bf16", "bfloat16"}
    supported_ops = {
        "input",
        "matmul",
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "relu",
        "gelu",
        "silu",
        "clamp",
        "cast",
        "sum",
        "mean",
        "max",
        "min",
        "output",
    }

    def check_graph(self, graph: FusionGraph) -> LegalResult:
        reasons: List[str] = []
        warnings: List[str] = []
        for t in graph.tensors.values():
            if t.dtype not in self.supported_dtypes:
                reasons.append(f"tensor {t.name}: unsupported dtype {t.dtype!r}")
            if t.layout not in ("contiguous", "broadcast", "unknown"):
                reasons.append(f"tensor {t.name}: unsupported layout {t.layout!r}")
        # Aliasing: an output that is also marked as an input is a
        # potential overlapping-write hazard. Surface it as a warning
        # so the user can opt into explicit ``out=`` semantics.
        for t in graph.tensors.values():
            if t.is_input and t.is_output:
                warnings.append(
                    f"tensor {t.name}: aliases input and output; "
                    "fusion will assume non-overlapping semantics"
                )
        for op in graph.ops.values():
            reasons.extend(self._check_op(graph, op))
        if reasons:
            return LegalResult.failure(*reasons, warnings=warnings)
        return LegalResult.success(warnings=warnings)

    def _check_op(self, graph: FusionGraph, op: FusionOp) -> List[str]:
        reasons: List[str] = []
        if op.kind not in self.supported_ops:
            reasons.append(f"op {op.name}: unsupported kind {op.kind!r}")
        if op.side_effects:
            reasons.append(f"op {op.name}: side-effecting ops cannot be fused")
        if op.kind == "matmul":
            if len(op.inputs) != 2:
                reasons.append(f"op {op.name}: matmul expects 2 inputs")
            else:
                a = graph.tensors[op.inputs[0]]
                b = graph.tensors[op.inputs[1]]
                if len(a.shape) != 2 or len(b.shape) != 2:
                    reasons.append(f"op {op.name}: matmul requires 2D tensors")
                elif (
                    a.shape[1] is not None
                    and b.shape[0] is not None
                    and a.shape[1] != b.shape[0]
                ):
                    reasons.append(
                        f"op {op.name}: matmul K mismatch {a.shape[1]} vs {b.shape[0]}"
                    )
                if a.dtype != b.dtype:
                    reasons.append(
                        f"op {op.name}: mixed dtype matmul {a.dtype}/{b.dtype}"
                    )
        if op.kind in ("add", "sub", "mul", "div"):
            if len(op.inputs) != 2:
                reasons.append(f"op {op.name}: {op.kind} expects 2 inputs")
            else:
                # Broadcast validation: the only broadcast we currently
                # support is per-N bias (1D operand whose extent matches
                # the 2D operand's last dim). Everything else must match
                # full shape.
                a = graph.tensors[op.inputs[0]]
                b = graph.tensors[op.inputs[1]]
                if a.shape and b.shape and a.shape != b.shape:
                    bigger, smaller = (a, b) if len(a.shape) >= len(b.shape) else (b, a)
                    if len(smaller.shape) == 1 and bigger.shape[-1] != smaller.shape[0]:
                        reasons.append(
                            f"op {op.name}: 1D broadcast operand {smaller.name} has "
                            f"extent {smaller.shape[0]} != bigger.shape[-1]={bigger.shape[-1]}"
                        )
                    elif len(smaller.shape) != 1 and len(smaller.shape) != len(
                        bigger.shape
                    ):
                        reasons.append(
                            f"op {op.name}: incompatible ranks ({a.shape} vs {b.shape})"
                        )
        if op.kind == "clamp":
            if "min" not in op.attrs and "max" not in op.attrs:
                reasons.append(f"op {op.name}: clamp needs min and/or max")
        if op.kind in ("sum", "mean", "max"):
            if len(op.inputs) != 1:
                reasons.append(f"op {op.name}: reduction expects 1 input")
            elif len(graph.tensors[op.inputs[0]].shape) < 2:
                reasons.append(f"op {op.name}: reduction requires 2D+ input")
        return reasons
