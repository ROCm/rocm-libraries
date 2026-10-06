# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Parse the narrow rocGDB text boundary used for stopped-wave values.

rocGDB evaluates a vector DWARF variable for the focus lane, while its raw
``$vN`` register values contain every lane.  Its Python API does not currently
expose the resolved DWARF location pieces, so this module isolates parsing of
``info address`` output.  No rendering or GDB process interaction belongs here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from rocke.core.debug_manifest import (
    DEBUG_DESCRIPTION_SCHEMA,
    DEBUG_MANIFEST_SCHEMA,
)


@dataclass(frozen=True)
class RegisterPiece:
    expression: str
    byte_size: int


_RANGE = re.compile(
    r"Range\s+(0x[0-9a-fA-F]+)-(0x[0-9a-fA-F]+):\s*(.*?)"
    r"(?=\n\s*Range\s|\n\s*\.\s*$)",
    re.DOTALL | re.MULTILINE,
)
_PIECE = re.compile(r"a variable in (\$[vs][0-9]+) \[([1-9][0-9]*)-byte piece\]")
_SINGLE_REGISTER = re.compile(r"is a variable in (\$[vs][0-9]+)\.?\s*$")
_SYMBOL_ADDRESS = re.compile(r"is at (0x[0-9a-fA-F]+)\b")
_PC_SYMBOL = re.compile(r"^([^\s+]+)(?: \+ [0-9]+)? in section ", re.MULTILINE)


def symbol_address(info_address: str) -> int:
    """Extract one minimal-symbol address from rocGDB ``info address``."""
    match = _SYMBOL_ADDRESS.search(info_address)
    if match is None:
        raise ValueError("rocGDB did not report a symbol address")
    return int(match.group(1), 16)


def kernel_symbol(info_symbol: str) -> str:
    """Extract the containing GPU symbol from rocGDB ``info symbol $pc``."""
    match = _PC_SYMBOL.search(info_symbol)
    if match is None:
        raise ValueError("rocGDB could not resolve the stopped PC to a kernel")
    return match.group(1)


def register_pieces(info_address: str, pc: int) -> list[RegisterPiece]:
    """Return the register pieces valid at ``pc`` from ``info address`` text."""
    ranges = list(_RANGE.finditer(info_address))
    if ranges:
        bodies = [
            match.group(3)
            for match in ranges
            if int(match.group(1), 16) <= pc < int(match.group(2), 16)
        ]
        if len(bodies) != 1:
            raise ValueError(
                f"debug value has {len(bodies)} location ranges at PC 0x{pc:x}"
            )
        pieces = [
            RegisterPiece(expression, int(byte_size))
            for expression, byte_size in _PIECE.findall(bodies[0])
        ]
        if not pieces:
            raise ValueError("debug value location has no AMDGPU register pieces")
        return pieces

    single = _SINGLE_REGISTER.search(info_address)
    if single is not None:
        return [RegisterPiece(single.group(1), 4)]
    raise ValueError("debug value has no AMDGPU register location")


def select_debug_names(
    description: dict[str, Any], names: Sequence[str]
) -> tuple[dict[str, Any], list[str]]:
    """Validate an embedded description and apply no-name selection rules."""
    if description.get("schema") != DEBUG_DESCRIPTION_SCHEMA:
        raise ValueError(
            f"unsupported embedded debug schema {description.get('schema')!r}"
        )
    described = description.get("values")
    if not isinstance(described, list) or not described:
        raise ValueError("embedded debug description has no values")
    by_name = {value.get("dwarf", {}).get("name"): value for value in described}
    if any(not isinstance(name, str) or not name for name in by_name):
        raise ValueError("embedded debug description has an invalid DWARF name")
    if not names:
        if len(by_name) != 1:
            raise ValueError(
                "multiple debug values are available; choose one of: "
                + ", ".join(sorted(by_name))
            )
        return by_name, list(by_name)
    selected_names = list(names)
    for name in selected_names:
        if name not in by_name:
            raise ValueError(
                f"unknown debug value {name!r}; choose one of: "
                + ", ".join(sorted(by_name))
            )
    return by_name, selected_names


def bind_debug_description(
    description: dict[str, Any],
    names: Sequence[str],
    *,
    pc: int,
    location_text: Callable[[str], str],
) -> tuple[dict[str, Any], list[str]]:
    """Resolve selected DWARF names to current-PC full-wave register bindings."""
    by_name, selected_names = select_debug_names(description, names)

    bound = []
    for name in selected_names:
        value = by_name[name]
        logical = value["logical"]
        if logical.get("dtype") != "f32":
            raise ValueError(
                "automatic rocGDB register binding currently supports f32 "
                f"logical values, got {logical.get('dtype')!r}"
            )
        pieces = register_pieces(location_text(name), pc)
        if any(piece.byte_size != 4 for piece in pieces):
            raise ValueError(f"debug value {name!r} has a non-32-bit register piece")
        fragment_length = logical.get("layout", {}).get("fragment_length")
        if len(pieces) != fragment_length:
            raise ValueError(
                f"debug value {name!r} has {len(pieces)} register pieces, "
                f"but its layout requires {fragment_length}"
            )
        bound.append(
            {
                "logical": logical,
                "binding": {
                    "kind": "amdgpu_registers",
                    "storage_dtype": "f32",
                    "locations": [piece.expression for piece in pieces],
                },
            }
        )
    return {"schema": DEBUG_MANIFEST_SCHEMA, "values": bound}, selected_names
