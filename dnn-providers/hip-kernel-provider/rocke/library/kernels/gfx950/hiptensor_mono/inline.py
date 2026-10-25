# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Inline a finished rocKE ``KernelDef`` body into another kernel's builder.

The monokernel composes *existing, tuned* kernel bodies (the dispatched
universal-GEMM ring bodies, permute, elementwise, reduction) as per-task
device code of one persistent kernel. Instead of re-authoring each body as a
device function, the composer builds the stand-alone kernel as usual and then
copies its IR into the monokernel's current region:

* every value the body defines gets a fresh, instance-unique name (results,
  ``scf.for`` induction variables and loop-carried values, LDS allocations);
* the body's kernel parameters are bound to caller values (monokernel
  parameters, or constants / runtime-selected scalars);
* ``gpu.block_id`` reads are replaced by caller-provided task-local tile
  coordinates (the monokernel's ticket decode);
* the trailing ``cf.return`` is dropped (an early return anywhere else is
  refused: it would end the persistent loop).

Only existing IR ops are emitted, so a composed kernel lowers identically
through both engines (the C++ engine consumes the serialized IR).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Mapping, Optional

from rocke.core.ir import IRBuilder, KernelDef, Op, Region, Value

__all__ = ["inline_kernel_body", "body_op_census", "InlineError"]

# ops that read launch geometry other than the block id; a body using them
# cannot be relocated into a task without a mapping (none of the composed
# families use them).
_GEOMETRY_OPS = frozenset(
    {
        "gpu.grid_dim",
        "gpu.block_dim",
        "gpu.num_workgroups",
        "gpu.cluster_id",
        "gpu.cluster_workgroup_id",
        "gpu.cluster_workgroup_flat_id",
    }
)


class InlineError(ValueError):
    pass


def body_op_census(kdef: KernelDef) -> Dict[str, int]:
    out: Dict[str, int] = {}

    def walk(reg: Region) -> None:
        for op in reg.ops:
            out[op.name] = out.get(op.name, 0) + 1
            for r in op.regions:
                walk(r)

    walk(kdef.body)
    return out


def inline_kernel_body(
    b: IRBuilder,
    kdef: KernelDef,
    *,
    params: Mapping[str, Value],
    block_ids: Mapping[str, Value],
    prefix: str,
    store_rewrite: Optional[Mapping[str, Callable[[IRBuilder, Value], Value]]] = None,
) -> None:
    """Copy ``kdef``'s body into ``b``'s current region.

    ``params``: kernel-parameter name -> caller value (every parameter the body
    uses must be bound; type must match). ``block_ids``: axis (``x``/``y``/``z``)
    -> caller i32 value. ``prefix``: unique per inlined instance (value names).
    ``store_rewrite``: param name -> ``f(b, idx) -> idx'``: every
    ``memref.global_store_typed`` through that pointer param stores at
    ``f(idx)`` instead (ops emitted right before the store, in its region);
    any other use of the param is refused (e.g. the monokernel's output
    permute fused into the GEMM epilogue).
    """
    store_rewrite = dict(store_rewrite or {})
    rw_names = {"%" + k for k in store_rewrite}
    pvals: Dict[str, Value] = {}
    for p in kdef.params:
        if p.name in params:
            v = params[p.name]
            if v.type.name != p.type.name:
                raise InlineError(
                    f"{kdef.name}: param {p.name} is {p.type.name}, bound to {v.type.name}"
                )
            pvals["%" + p.name] = v
    vmap: Dict[int, Value] = {}
    names: Dict[str, str] = {}

    def fresh(old: Value) -> Value:
        nv = Value(name=f"%{prefix}_{old.name.lstrip('%')}", type=old.type)
        vmap[id(old)] = nv
        names[old.name] = nv.name
        return nv

    def operand(v: Value) -> Value:
        got = vmap.get(id(v))
        if got is not None:
            return got
        if v.op is None and v.name in pvals:
            return pvals[v.name]
        if v.op is None:
            raise InlineError(f"{kdef.name}: unbound parameter {v.name}")
        raise InlineError(f"{kdef.name}: use of {v.name} before its definition")

    def map_attr(a: Any) -> Any:
        if isinstance(a, Value):
            return operand(a)
        if isinstance(a, dict):
            return {k: map_attr(x) for k, x in a.items()}
        if isinstance(a, list):
            return [map_attr(x) for x in a]
        if isinstance(a, tuple):
            return tuple(map_attr(x) for x in a)
        return a

    def clone_region(src: Region, dst: Region, top: bool) -> None:
        n = len(src.ops)
        for i, op in enumerate(src.ops):
            if op.name == "cf.return":
                if top and i == n - 1:
                    continue
                raise InlineError(f"{kdef.name}: early return cannot be inlined")
            if op.name in _GEOMETRY_OPS:
                raise InlineError(f"{kdef.name}: {op.name} has no task mapping")
            if op.name == "gpu.block_id":
                axis = op.attrs.get("axis", "x")
                if axis not in block_ids:
                    raise InlineError(f"{kdef.name}: no block id for axis {axis}")
                vmap[id(op.result)] = block_ids[axis]
                continue
            new_ops = [operand(v) for v in op.operands]
            rw = [v.name for v in op.operands if v.op is None and v.name in rw_names]
            if rw:
                if (
                    op.name != "memref.global_store_typed"
                    or op.operands[0].name != rw[0]
                    or len(rw) != 1
                ):
                    raise InlineError(
                        f"{kdef.name}: {op.name} uses rewritten param {rw[0]}"
                    )
                b.push_region(dst)
                try:
                    new_ops[1] = store_rewrite[rw[0][1:]](b, new_ops[1])
                finally:
                    b.pop_region()
            new_attrs = map_attr(dict(op.attrs))
            new_op = Op(
                name=op.name,
                operands=new_ops,
                results=[],
                attrs=new_attrs,
                regions=[],
                loc=op.loc,
            )
            if op.name == "scf.for":
                # induction variable + loop-carried values are named in attrs
                # and referenced by body ops as Value objects whose .op is the
                # for op itself; find them by walking the body's operands.
                iv_name = op.attrs["iv"]
                carried = {m["name"] for m in op.attrs.get("iter_args", [])}
                for v in _region_free_defs(op, {iv_name} | carried):
                    nv = fresh(v)
                    nv.op = new_op
                new_op.attrs["iv"] = names.get(
                    iv_name, f"%{prefix}_{iv_name.lstrip('%')}"
                )
                if "iter_args" in op.attrs:
                    new_op.attrs["iter_args"] = [
                        dict(
                            m,
                            name=names.get(
                                m["name"], f"%{prefix}_{m['name'].lstrip('%')}"
                            ),
                        )
                        for m in op.attrs["iter_args"]
                    ]
            for r in op.results:
                nv = fresh(r)
                nv.op = new_op
                new_op.results.append(nv)
            for r in op.regions:
                nr = Region(r.label)
                clone_region(r, nr, top=False)
                new_op.regions.append(nr)
            dst.ops.append(new_op)

    clone_region(kdef.body, b._region_stack[-1], top=True)


def _region_free_defs(for_op: Op, wanted_names) -> list:
    """Value objects named in ``wanted_names`` whose defining op is ``for_op``
    (the iv and the loop-carried values), found among the operands of the body."""
    found: Dict[str, Value] = {}

    def walk(reg: Region) -> None:
        for op in reg.ops:
            for v in op.operands:
                if v.op is for_op and v.name in wanted_names and v.name not in found:
                    found[v.name] = v
            for r in op.regions:
                walk(r)

    for r in for_op.regions:
        walk(r)
    # names never used in the body still need a fresh name in the attrs; the
    # caller falls back to the prefixed name for those.
    return list(found.values())
