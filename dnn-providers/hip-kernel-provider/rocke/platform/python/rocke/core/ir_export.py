# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Portable IR exporter (schema ``rocke.ir/v1``).

Serializes an already-built :class:`~rocke.core.ir.KernelDef` graph into a
structured, dependency-free dict (JSON-able) that the C++ engine can import and
lower without embedding CPython.

This is the offline half of the portable-IR path described in
``dsl_docs/architecture/portable_ir_schema.md``: Python remains the kernel
authoring surface; the *result* of running a builder (the SSA graph) is
exported here and re-driven through the C IRBuilder by
``rocke_import_kernel_from_json`` (``cpp/portable_ir/ir_import_json.cpp``).

Design:
- No source parsing, no transpilation: we walk the `KernelDef` dataclasses.
- SSA values are referenced by their existing names (``%vN`` / ``%A`` / ``%k0``)
  so operands resolve by id on import.
- Types are serialized structurally (scalars as their canonical name string,
  composites as objects) mirroring the C ``rocke/ir.h`` type model.
- Attrs are *typed* (``{"t": <kind>, "v": <value>}``) so the importer never has
  to infer a type from JSON syntax.
- ``loc`` (source span) is intentionally dropped: it never reaches the lowered
  ``.ll`` (verified by the C-vs-Python parity harness), so omitting it keeps the
  artifact small and host-path-free while staying byte-identical after lowering.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from .ir import KernelDef, Op, Param, PtrType, Region, SmemType, Type, VectorType, Value

SCHEMA = "rocke.ir/v1"


# ------------------------------------------------------------------- types


def _type_to_json(t: Type) -> Any:
    """Scalar -> canonical name string; composite -> structured object."""
    if isinstance(t, VectorType):
        return {"kind": "vector", "elem": _type_to_json(t.elem), "count": int(t.count)}
    if isinstance(t, PtrType):
        return {"kind": "ptr", "pointee": _type_to_json(t.pointee), "space": t.space}
    if isinstance(t, SmemType):
        return {
            "kind": "smem",
            "elem": _type_to_json(t.elem),
            "shape": [int(x) for x in t.shape],
        }
    # Plain scalar Type: just the canonical name ("i32", "f16", ...).
    return t.name


# ------------------------------------------------------------------- attrs


def _attr_to_json(value: Any) -> Dict[str, Any]:
    """Typed attribute value.

    Kinds mirror the C ``rocke_attr_kind_t``: i (int64), f (double), b (bool),
    s (string), l (list of nested attr-maps).
    """
    # bool must precede int (bool is an int subclass in Python).
    if isinstance(value, bool):
        return {"t": "b", "v": value}
    if isinstance(value, int):
        return {"t": "i", "v": int(value)}
    if isinstance(value, float):
        return {"t": "f", "v": float(value)}
    if isinstance(value, str):
        return {"t": "s", "v": value}
    if isinstance(value, (list, tuple)):
        items: List[Any] = []
        for item in value:
            if isinstance(item, dict):
                # Nested attr-map (e.g. scf.for iter_args metadata).
                items.append({k: _attr_to_json(v) for k, v in item.items()})
            else:
                # A bare scalar in a list/tuple is wrapped as a single-entry
                # map under "_" so the C list-of-maps model can hold it.
                items.append({"_": _attr_to_json(item)})
        return {"t": "l", "v": items}
    raise TypeError(
        f"ir_export: unsupported attr value type {type(value).__name__!r}: {value!r}"
    )


def _attrs_to_json(attrs: Dict[str, Any]) -> Dict[str, Any]:
    # Sorted for deterministic, diff-stable artifacts.
    return {k: _attr_to_json(v) for k, v in sorted(attrs.items())}


# --------------------------------------------------------------- ops/regions


def _op_to_json(op: Op) -> Dict[str, Any]:
    return {
        "opcode": op.name,
        "operands": [v.name for v in op.operands],
        "results": [{"id": r.name, "type": _type_to_json(r.type)} for r in op.results],
        "attrs": _attrs_to_json(op.attrs),
        "regions": [_region_to_json(r) for r in op.regions],
    }


def _region_to_json(region: Region) -> Dict[str, Any]:
    return {"label": region.label, "ops": [_op_to_json(op) for op in region.ops]}


def _param_to_json(p: Param) -> Dict[str, Any]:
    return {
        "name": p.name,
        "type": _type_to_json(p.type),
        "attrs": dict(sorted(p.attrs.items())),
    }


# ------------------------------------------------------------------- public


def export_kernel_ir(
    kernel: KernelDef,
    *,
    target_hint: str | None = None,
    llvm_flavor_hint: str | None = None,
) -> Dict[str, Any]:
    """Return the portable-IR dict for ``kernel`` (schema ``rocke.ir/v1``)."""
    opcodes = sorted(_collect_opcodes(kernel.body))
    payload: Dict[str, Any] = {
        "schema": SCHEMA,
        "producer": {"name": "rocke_python", "version": "0.1"},
        "requires": {"min_rocke_ir": 1, "opcodes": opcodes},
        "kernel": {
            "name": kernel.name,
            "attrs": _attrs_to_json(kernel.attrs),
            "params": [_param_to_json(p) for p in kernel.params],
            "body": _region_to_json(kernel.body),
        },
    }
    target: Dict[str, Any] = {}
    if target_hint is not None:
        target["arch_hint"] = target_hint
    if llvm_flavor_hint is not None:
        target["llvm_flavor_hint"] = llvm_flavor_hint
    if target:
        payload["target"] = target
    return payload


def export_kernel_ir_json(
    kernel: KernelDef, *, indent: int | None = 2, **kw: Any
) -> str:
    """Serialize :func:`export_kernel_ir` to a JSON string."""
    return json.dumps(export_kernel_ir(kernel, **kw), indent=indent, sort_keys=False)


def _collect_opcodes(region: Region, acc: set | None = None) -> set:
    if acc is None:
        acc = set()
    for op in region.ops:
        acc.add(op.name)
        for sub in op.regions:
            _collect_opcodes(sub, acc)
    return acc
