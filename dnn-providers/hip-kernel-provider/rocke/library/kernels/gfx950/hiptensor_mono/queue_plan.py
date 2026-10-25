# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Task-queue monokernel plans for a set of independent hipTensor contractions."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from rocke.instances.common.gemm_universal import UniversalGemmSpec

from .queue_contraction import Contraction, contraction_ops
from .task_queue import QueueOp, QueuePlan

__all__ = ["contractions_plan", "tensor_names"]


def tensor_names(c: Contraction) -> Tuple[str, str, str]:
    return f"{c.name}_A", f"{c.name}_B", f"{c.name}_D"


def contractions_plan(
    name: str,
    cs: Sequence[Contraction],
    *,
    block_size: int = 512,
    elems_per_ticket: int = 1 << 16,
    gemm_spec: Optional[UniversalGemmSpec] = None,
    order: str = "phase",
    gemm_policy: str = "unify",
    gemm_only: bool = False,
    permute_body: str = "gather",
    fuse_out: str = "off",
    arch: str = "gfx950",
) -> Tuple[QueuePlan, Dict[str, int]]:
    """Plan + scratch tensor sizes (elements).

    ``order``: ``phase`` = every input permute, then every GEMM (grouped by
    body), then every output permute (tickets of independent ops interleave
    across contractions; GEMMs sharing a body are contiguous). ``op`` = one
    contraction after the other (only valid when no two contractions share a
    GEMM body, or with one contraction).

    ``gemm_policy``: ``pick`` = each contraction's own dispatched spec (one
    inlined body per distinct spec: LDS adds up); ``unify`` = per K class
    (K % 64 == 0 or not) the spec of the class's largest GEMM, so the plan
    holds at most two GEMM bodies.

    ``gemm_only``: only the GEMM ops (inputs already permuted, output in E):
    the GEMM-only part of the comparison.
    """
    spec_by_class = None
    if gemm_policy == "unify" and gemm_spec is None and len(cs) > 1:
        from .queue_contraction import gemm_spec_for

        best: Dict[bool, Tuple[int, object]] = {}
        for c in cs:
            M, N, K = c.MNK
            k64 = K % 64 == 0
            fl = M * N * K
            if k64 not in best or fl > best[k64][0]:
                best[k64] = (
                    fl,
                    gemm_spec_for(N, M, K, block_size=block_size, arch=arch),
                )
        spec_by_class = {k: v[1] for k, v in best.items()}
    elif gemm_policy not in ("unify", "pick"):
        raise ValueError(gemm_policy)
    pre: List[QueueOp] = []
    gem: List[QueueOp] = []
    post: List[QueueOp] = []
    scratch: Dict[str, int] = {}
    tensors: List[str] = []
    per_c = []
    for c in cs:
        a, b, d = tensor_names(c)
        tensors += [a, b, d]
        p0, g, p1, sc = contraction_ops(
            c,
            a=a,
            b=b,
            d=d,
            block_size=block_size,
            elems_per_ticket=elems_per_ticket,
            gemm_spec=(
                spec_by_class[c.MNK[2] % 64 == 0] if spec_by_class else gemm_spec
            ),
            permute_body=permute_body,
            fuse_out=fuse_out,
            arch=arch,
        )
        if gemm_only:
            g = [type(x)(**{**x.__dict__, "deps": ()}) for x in g]
            p0, p1 = [], []
        per_c.append((p0, g, p1))
        pre += p0
        gem += g
        post += p1
        scratch.update(sc)
    tensors += list(scratch)
    if order == "phase":
        # contiguous shared bodies: stable sort GEMMs by body key, biggest first
        keys: List[str] = []
        for g in gem:
            if g.body_key not in keys:
                keys.append(g.body_key)
        gem = [g for k in keys for g in gem if g.body_key == k]
        ops = pre + gem + post
    elif order == "op":
        ops = [o for p0, g, p1 in per_c for o in (p0 + g + p1)]
    else:
        raise ValueError(order)
    plan = QueuePlan(name=name, ops=ops, tensors=tensors, block_size=block_size)
    plan.validate()
    return plan, scratch
