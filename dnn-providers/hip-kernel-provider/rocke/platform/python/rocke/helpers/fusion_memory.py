# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Workspace planning for fusion plans.

This module bridges :class:`FusionPlan` to the runtime
:class:`WorkspacePool`. Conceptually:

* :class:`WorkspacePlanner` runs liveness analysis over the
  scheduled regions, producing one :class:`WorkspaceAllocation`
  per cross-region intermediate. The allocations are slot-coloured
  with the smallest set of physical slots that respects liveness,
  so distinct intermediates that don't overlap can share an
  allocation.

* :func:`materialize_plan` turns those allocations into a list of
  :class:`WorkspaceSpec` and (optionally) reserves the slots on a
  caller-supplied :class:`WorkspacePool`. The returned dict maps
  tensor names to live workspace tensors so the executor can pass
  them to each region's launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .fusion_ir import FusionPlan


__all__ = [
    "WorkspaceAllocation",
    "WorkspacePlanner",
    "materialize_plan",
]


@dataclass(frozen=True)
class WorkspaceAllocation:
    r"""One workspace-resident intermediate after liveness analysis.

    Attributes
    ----------
    tensor_name
        The :class:`FusionTensor` name the allocation backs.
    first_region, last_region
        Inclusive index range over :attr:`FusionPlan.regions` over
        which the slot must be live.
    shape, dtype
        Allocation shape (concrete ints) and dtype string. Shape uses
        ``1`` for any ``None`` dim in the tensor descriptor; this
        keeps the planner explicit while letting callers pass an
        override at runtime if they need a dynamic dim.
    slot_name
        Physical slot name. After colouring, distinct
        ``tensor_name``\ s may share a ``slot_name`` if their
        lifetimes don't overlap.
    """

    tensor_name: str
    first_region: int
    last_region: int
    shape: Tuple[int, ...]
    dtype: str
    slot_name: str

    def overlaps(self, other: "WorkspaceAllocation") -> bool:
        return not (
            self.last_region < other.first_region
            or other.last_region < self.first_region
        )


class WorkspacePlanner:
    """Liveness-based workspace planner with slot colouring.

    After running the planner you get one :class:`WorkspaceAllocation`
    per intermediate, grouped by physical slot. The default colouring
    is greedy and deterministic: walk allocations in
    ``(first_region, -nbytes)`` order, place each in the smallest
    existing slot that fits, else mint a new slot.

    Use :func:`materialize_plan` to bind the allocations to a
    :class:`WorkspacePool`; the planner itself is dependency-free
    (no torch / no runtime) so it can be exercised by unit tests
    in pure Python.
    """

    def plan(self, plan: FusionPlan) -> Tuple[WorkspaceAllocation, ...]:
        graph = plan.graph
        region_index_by_op: Dict[str, int] = {}
        for i, region in enumerate(plan.regions):
            for op_name in region.op_names:
                region_index_by_op[op_name] = i

        raw: List[WorkspaceAllocation] = []
        for tensor in graph.tensors.values():
            if tensor.is_input or tensor.is_output:
                continue
            if tensor.producer is None:
                continue
            first = region_index_by_op.get(tensor.producer, 0)
            user_regions = [
                region_index_by_op[u] for u in tensor.users if u in region_index_by_op
            ]
            last = max(user_regions) if user_regions else first
            if first == last:
                # Fully internal to one region; no global workspace.
                continue
            shape = tuple(int(d or 1) for d in tensor.shape)
            raw.append(
                WorkspaceAllocation(
                    tensor_name=tensor.name,
                    first_region=first,
                    last_region=last,
                    shape=shape,
                    dtype=tensor.dtype,
                    slot_name=f"ws_{tensor.name}",
                )
            )
        return tuple(self._colour(raw))

    def _colour(self, allocs: List[WorkspaceAllocation]) -> List[WorkspaceAllocation]:
        """Greedy slot colouring.

        Two allocations can share a physical slot iff their
        ``first_region..last_region`` ranges are disjoint AND they
        share dtype + shape. We sort by ``first_region`` and try to
        reuse an existing released slot; otherwise mint a new slot.
        """
        if not allocs:
            return []
        slots: List[List[WorkspaceAllocation]] = []  # slot -> allocs sharing the slot
        for a in sorted(allocs, key=lambda x: (x.first_region, x.tensor_name)):
            placed = False
            for slot in slots:
                if slot[-1].last_region < a.first_region and (
                    slot[-1].shape == a.shape and slot[-1].dtype == a.dtype
                ):
                    slot.append(a)
                    placed = True
                    break
            if not placed:
                slots.append([a])
        out: List[WorkspaceAllocation] = []
        for slot_idx, slot in enumerate(slots):
            shared_name = f"ws_slot{slot_idx}"
            for a in slot:
                out.append(
                    WorkspaceAllocation(
                        tensor_name=a.tensor_name,
                        first_region=a.first_region,
                        last_region=a.last_region,
                        shape=a.shape,
                        dtype=a.dtype,
                        slot_name=shared_name,
                    )
                )
        return out


def _dtype_to_torch(dtype_str: str):
    """Translate the graph's dtype string to a torch dtype.

    Used by :func:`materialize_plan` when the caller asks the planner
    to actually allocate tensors. We don't import torch at module
    load to keep the planner usable in pure-python tests.
    """
    import torch as _t

    s = dtype_str.lower()
    if s in ("fp16", "f16", "float16", "half"):
        return _t.float16
    if s in ("bf16", "bfloat16"):
        return _t.bfloat16
    if s in ("fp32", "f32", "float32", "float"):
        return _t.float32
    if s in ("i32", "int32"):
        return _t.int32
    raise ValueError(f"unsupported workspace dtype {dtype_str!r}")


def materialize_plan(
    plan: FusionPlan,
    *,
    pool: Any,
    device: Any,
    planner: Optional[WorkspacePlanner] = None,
) -> Tuple[Mapping[str, Any], Tuple[WorkspaceAllocation, ...]]:
    """Allocate every cross-region intermediate on ``pool``.

    Returns a tuple ``(name_to_tensor, allocations)``. ``name_to_tensor``
    maps the original :class:`FusionTensor` names (not the colourer's
    ``ws_slot*`` names) to the workspace tensors so the executor can
    look them up by name without knowing about the colouring.

    A single physical slot is reused across allocations that share it
    -- the underlying :class:`WorkspacePool.get` call hands back a
    view into the shared tensor when the requested shape fits.
    """

    planner = planner or WorkspacePlanner()
    allocs = planner.plan(plan)
    by_slot: Dict[str, List[WorkspaceAllocation]] = {}
    for a in allocs:
        by_slot.setdefault(a.slot_name, []).append(a)
    name_to_tensor: Dict[str, Any] = {}
    for slot_name, slot_allocs in by_slot.items():
        # Use the largest shape of all sharers as the slot capacity.
        max_numel = 0
        max_shape = ()
        for a in slot_allocs:
            n = 1
            for s in a.shape:
                n *= s
            if n >= max_numel:
                max_numel = n
                max_shape = a.shape
        dtype = _dtype_to_torch(slot_allocs[0].dtype)
        slot_tensor = pool.get(slot_name, max_shape, dtype=dtype, device=device)
        for a in slot_allocs:
            n = 1
            for s in a.shape:
                n *= s
            if a.shape == max_shape:
                name_to_tensor[a.tensor_name] = slot_tensor
            else:
                # Re-shape a subview of the slot tensor; this matches
                # WorkspacePool's "smaller shape returns a view" contract.
                name_to_tensor[a.tensor_name] = slot_tensor.flatten()[:n].view(a.shape)
    return name_to_tensor, allocs
