# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""MLIR-style textual printer for the CK DSL IR.

The output is human-readable and stable: it is consumed by tests (string
fixtures) and dropped into kernel manifests as a `kernel.ir` field for
debugging.
"""

from __future__ import annotations

from typing import List

from .ir import KernelDef, Op


def _format_operand(v) -> str:
    return v.name


def _format_attrs(attrs) -> str:
    if not attrs:
        return ""
    inner = ", ".join(f"{k} = {_attr_value(v)}" for k, v in sorted(attrs.items()))
    return f" {{{inner}}}"


def _attr_value(v) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def _format_results(results) -> str:
    if not results:
        return ""
    return ", ".join(r.name for r in results) + " = "


def _format_types(results) -> str:
    if not results:
        return ""
    return " : " + ", ".join(r.type.name for r in results)


def _print_op(op: Op, indent: int) -> List[str]:
    pad = " " * indent
    head = f"{pad}{_format_results(op.results)}{op.name}"
    args = ""
    if op.operands:
        args = " " + ", ".join(_format_operand(v) for v in op.operands)
    line = f"{head}{args}{_format_attrs(op.attrs)}{_format_types(op.results)}"
    lines = [line]
    for region in op.regions:
        lines.append(f"{pad}  region {region.label!r} {{")
        for sub in region.ops:
            lines.extend(_print_op(sub, indent + 4))
        lines.append(f"{pad}  }}")
    return lines


def print_ir(kernel: KernelDef) -> str:
    param_str = ", ".join(f"%{p.name}: {p.type.name}" for p in kernel.params)
    lines = [f"kernel @{kernel.name}({param_str}) {{"]
    for op in kernel.body.ops:
        lines.extend(_print_op(op, 2))
    lines.append("}")
    return "\n".join(lines)
