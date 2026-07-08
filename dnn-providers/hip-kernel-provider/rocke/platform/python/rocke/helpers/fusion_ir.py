# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Normalized fusion graph IR.

This is intentionally smaller than FX and higher-level than CK DSL's
kernel SSA IR. FX is a frontend; ``FusionGraph`` is the planner IR:
typed tensors, normalized ops, region grouping, and a top-level plan
that can be legalized, scheduled, lowered, autotuned and launched.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Shape = Tuple[Optional[int], ...]
Strides = Tuple[Optional[int], ...]


@dataclass(frozen=True)
class FusionTensor:
    """A logical tensor value in the fusion planner."""

    name: str
    shape: Shape
    dtype: str
    layout: str = "contiguous"
    strides: Strides = ()
    producer: Optional[str] = None
    users: Tuple[str, ...] = ()
    is_input: bool = False
    is_output: bool = False

    def with_user(self, op_name: str) -> "FusionTensor":
        if op_name in self.users:
            return self
        return FusionTensor(
            name=self.name,
            shape=self.shape,
            dtype=self.dtype,
            layout=self.layout,
            strides=self.strides,
            producer=self.producer,
            users=tuple([*self.users, op_name]),
            is_input=self.is_input,
            is_output=self.is_output,
        )


@dataclass(frozen=True)
class FusionOp:
    """A normalized graph operation."""

    name: str
    kind: str
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    attrs: Dict[str, Any] = field(default_factory=dict)
    side_effects: bool = False


@dataclass(frozen=True)
class FusionGraph:
    """A normalized DAG of :class:`FusionTensor` and :class:`FusionOp`."""

    tensors: Dict[str, FusionTensor]
    ops: Dict[str, FusionOp]
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]

    def topological_ops(self) -> List[FusionOp]:
        # FX capture already produces topological order; all builders
        # preserve insertion order in ``ops``.
        return list(self.ops.values())

    def graph_hash(self) -> str:
        payload = {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "tensors": {
                k: {
                    "shape": v.shape,
                    "dtype": v.dtype,
                    "layout": v.layout,
                    "producer": v.producer,
                    "users": v.users,
                    "is_input": v.is_input,
                    "is_output": v.is_output,
                }
                for k, v in sorted(self.tensors.items())
            },
            "ops": {
                k: {
                    "kind": v.kind,
                    "inputs": v.inputs,
                    "outputs": v.outputs,
                    "attrs": v.attrs,
                    "side_effects": v.side_effects,
                }
                for k, v in sorted(self.ops.items())
            },
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()[:16]


@dataclass(frozen=True)
class FusionRegion:
    """A scheduled region that will lower to one kernel or pipeline stage."""

    name: str
    kind: str
    op_names: Tuple[str, ...]
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    attrs: Dict[str, Any] = field(default_factory=dict)
    lowerer: Optional[str] = None


@dataclass(frozen=True)
class FusionPlan:
    """Top-level fusion plan after scheduling."""

    graph: FusionGraph
    regions: Tuple[FusionRegion, ...]
    workspaces: Tuple[str, ...] = ()
    explanation: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "graph_hash": self.graph.graph_hash(),
            "regions": [
                {
                    "name": r.name,
                    "kind": r.kind,
                    "op_names": r.op_names,
                    "inputs": r.inputs,
                    "outputs": r.outputs,
                    "attrs": r.attrs,
                    "lowerer": r.lowerer,
                }
                for r in self.regions
            ],
            "workspaces": self.workspaces,
            "explanation": self.explanation,
        }


def build_graph(
    *,
    tensors: Iterable[FusionTensor],
    ops: Iterable[FusionOp],
    inputs: Sequence[str],
    outputs: Sequence[str],
) -> FusionGraph:
    """Construct a graph and fill tensor producer/user links."""

    tmap = {t.name: t for t in tensors}
    omap = {o.name: o for o in ops}
    for op in omap.values():
        for out in op.outputs:
            t = tmap[out]
            tmap[out] = FusionTensor(
                name=t.name,
                shape=t.shape,
                dtype=t.dtype,
                layout=t.layout,
                strides=t.strides,
                producer=op.name,
                users=t.users,
                is_input=t.is_input,
                is_output=t.is_output,
            )
        for inp in op.inputs:
            if inp in tmap:
                tmap[inp] = tmap[inp].with_user(op.name)
    return FusionGraph(
        tensors=tmap,
        ops=omap,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
    )
