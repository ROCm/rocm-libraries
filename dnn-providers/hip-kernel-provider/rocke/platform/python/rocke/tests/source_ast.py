# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Read names and literals out of a Python AST: the shared half of the source
checks' finders (``rocke.tests.torch_uses``, ``rocke.tests.toolchain_pins``)."""

from __future__ import annotations

import ast

__all__ = ["dotted_name", "string_constant"]


def dotted_name(node: ast.AST) -> str:
    """The dotted name an attribute chain spells (``os.environ.get``). A chain
    that does not start at a name gives its attributes only (``f().a.b`` gives
    ``a.b``), and a node that is neither gives ``""``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def string_constant(node: ast.AST | None) -> str | None:
    """The value of a string literal, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None
