# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Greedy fusion-region scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from .fusion_ir import FusionGraph, FusionPlan, FusionRegion


@dataclass(frozen=True)
class RegionCost:
    bytes_saved: int
    extra_ops_per_element: int
    kernels_saved: int
    estimated_vgprs: int

    def score(self) -> int:
        return (
            self.bytes_saved
            + 4096 * self.kernels_saved
            - 64 * self.extra_ops_per_element
        )


class GreedyFusionScheduler:
    """Simple deterministic scheduler.

    Fuses a matmul and its single-use pointwise consumers into one
    ``gemm_epilogue`` region; recognises row reductions as standalone
    ``rowwise_reduction`` regions; emits ``elementwise`` regions for
    other single pointwise ops. Anything outside this coverage becomes
    a ``fallback`` region. This is not an industrial solver; it is the
    first explainable regioning layer that later lowerers consume.

    The cost model in :class:`RegionCost` is used to compute (and
    record) a deterministic score per region. The scheduler itself is
    still greedy -- it always picks fusion when legal -- but the cost
    annotations are written into the region's ``attrs`` so downstream
    tooling can inspect and explain the choice.
    """

    pointwise_kinds = {"add", "mul", "relu", "gelu", "silu", "clamp"}
    reduction_kinds = {"sum", "mean", "max"}

    def schedule(self, graph: FusionGraph) -> FusionPlan:
        regions: List[FusionRegion] = []
        explanation: List[str] = []
        consumed = set()

        for op in graph.topological_ops():
            if op.name in consumed:
                continue
            if op.kind == "matmul":
                region_ops = [op.name]
                outputs = list(op.outputs)
                cur_out = op.outputs[0]
                while True:
                    users = graph.tensors[cur_out].users
                    if len(users) != 1:
                        break
                    nxt = graph.ops[users[0]]
                    if nxt.kind not in self.pointwise_kinds:
                        break
                    region_ops.append(nxt.name)
                    consumed.add(nxt.name)
                    cur_out = nxt.outputs[0]
                    outputs = list(nxt.outputs)
                consumed.add(op.name)
                lowerer = "gemm_epilogue" if len(region_ops) > 1 else "gemm"
                cost = self._estimate_cost(graph, tuple(region_ops))
                # Collect all inputs (op.inputs has only the matmul's
                # operands; epilogue ops pull in biases/residuals).
                all_inputs: List[str] = list(op.inputs)
                seen = set(all_inputs)
                for name in region_ops[1:]:
                    for inp in graph.ops[name].inputs:
                        if inp == cur_out:
                            continue
                        # Replace the per-op intermediate with the
                        # external placeholder it consumed.
                        if inp not in seen and graph.tensors[inp].is_input:
                            all_inputs.append(inp)
                            seen.add(inp)
                explanation.append(
                    f"region {len(regions)}: fused {region_ops} as {lowerer} "
                    f"(score={cost.score()})"
                )
                regions.append(
                    FusionRegion(
                        name=f"region{len(regions)}",
                        kind=lowerer,
                        op_names=tuple(region_ops),
                        inputs=tuple(all_inputs),
                        outputs=tuple(outputs),
                        attrs={"cost": cost.__dict__},
                        lowerer=lowerer,
                    )
                )
            elif op.kind in self.reduction_kinds:
                consumed.add(op.name)
                explanation.append(
                    f"region {len(regions)}: rowwise reduction {op.name} ({op.kind})"
                )
                regions.append(
                    FusionRegion(
                        name=f"region{len(regions)}",
                        kind="rowwise_reduction",
                        op_names=(op.name,),
                        inputs=op.inputs,
                        outputs=op.outputs,
                        attrs={"reduce_op": op.kind},
                        lowerer="rowwise_reduction",
                    )
                )
            elif op.kind in self.pointwise_kinds:
                consumed.add(op.name)
                explanation.append(
                    f"region {len(regions)}: pointwise fallback for {op.name}"
                )
                regions.append(
                    FusionRegion(
                        name=f"region{len(regions)}",
                        kind="elementwise",
                        op_names=(op.name,),
                        inputs=op.inputs,
                        outputs=op.outputs,
                        lowerer="elementwise",
                    )
                )
            else:
                consumed.add(op.name)
                explanation.append(
                    f"region {len(regions)}: unsupported fallback for {op.name}"
                )
                regions.append(
                    FusionRegion(
                        name=f"region{len(regions)}",
                        kind="fallback",
                        op_names=(op.name,),
                        inputs=op.inputs,
                        outputs=op.outputs,
                        lowerer=None,
                    )
                )

        return FusionPlan(
            graph=graph, regions=tuple(regions), explanation=tuple(explanation)
        )

    def _estimate_cost(
        self, graph: FusionGraph, op_names: Tuple[str, ...]
    ) -> RegionCost:
        # Conservative, explainable placeholders. Bytes saved assumes
        # each fused pointwise op avoids one read+write of the GEMM output.
        matmul = graph.ops[op_names[0]]
        out = graph.tensors[matmul.outputs[0]]
        elems = 1
        for d in out.shape:
            elems *= int(d or 1)
        bytes_per = 2 if out.dtype in ("fp16", "f16", "bf16", "bfloat16") else 4
        fused_pointwise = max(0, len(op_names) - 1)
        return RegionCost(
            bytes_saved=elems * bytes_per * 2 * fused_pointwise,
            extra_ops_per_element=fused_pointwise,
            kernels_saved=fused_pointwise,
            estimated_vgprs=32 + 2 * fused_pointwise,
        )
