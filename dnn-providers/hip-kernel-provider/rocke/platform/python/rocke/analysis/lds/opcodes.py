# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Normalized LDS opcode metadata used by conflict profiles."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "OpcodeSpec",
    "UnsupportedLdsOpcodeError",
    "get_opcode_spec",
    "normalize_opcode",
    "supported_opcodes",
]


class UnsupportedLdsOpcodeError(ValueError):
    """Raised when an opcode is outside the expert's explicit public scope."""


@dataclass(frozen=True)
class OpcodeSpec:
    """Canonical opcode facts shared by normalization and prediction.

    Profiles use this value to check opcode support and build a normalized request.
    ``access_width_bytes`` is the width of one logical LDS access in bytes.
    ``alignment_bytes`` narrows prediction to an explicitly supported address
    alignment; when absent, the full access width is used. It is not necessarily
    the instruction's minimum hardware alignment.
    """

    opcode: str
    direction: str
    access_width_bytes: int
    alignment_bytes: int | None = None


_OPCODE_SPECS = {
    spec.opcode: spec
    for spec in (
        OpcodeSpec("ds_read_b32", "read", 4),
        OpcodeSpec("ds_read_b64", "read", 8),
        OpcodeSpec("ds_read_b128", "read", 16),
        OpcodeSpec("ds_write_b32", "write", 4),
        OpcodeSpec("ds_write_b64", "write", 8),
        OpcodeSpec("ds_write_b128", "write", 16),
        OpcodeSpec("ds_load_b32", "read", 4),
        OpcodeSpec("ds_load_b64", "read", 8),
        OpcodeSpec("ds_load_b96", "read", 12, 16),
        OpcodeSpec("ds_load_b128", "read", 16),
        OpcodeSpec("ds_store_b32", "write", 4),
        OpcodeSpec("ds_store_b64", "write", 8),
        OpcodeSpec("ds_store_b96", "write", 12, 16),
        OpcodeSpec("ds_store_b128", "write", 16),
        OpcodeSpec("ds_load_u8", "read", 1, 4),
        OpcodeSpec("ds_load_i8", "read", 1, 4),
        OpcodeSpec("ds_load_u16", "read", 2, 4),
        OpcodeSpec("ds_load_i16", "read", 2, 4),
        OpcodeSpec("ds_store_b8", "write", 1, 4),
        OpcodeSpec("ds_store_b16", "write", 2, 4),
    )
}


def normalize_opcode(opcode: str) -> str:
    """Return the canonical spelling for a supported LDS opcode."""

    if not isinstance(opcode, str):
        raise TypeError("opcode must be a string")
    normalized = opcode.strip().lower()
    if normalized not in _OPCODE_SPECS:
        choices = ", ".join(_OPCODE_SPECS)
        raise UnsupportedLdsOpcodeError(
            f"unsupported LDS opcode {opcode!r}; supported opcodes: {choices}"
        )
    return normalized


def get_opcode_spec(opcode: str) -> OpcodeSpec:
    """Return immutable metadata for *opcode* after normalization."""

    return _OPCODE_SPECS[normalize_opcode(opcode)]


def supported_opcodes() -> tuple[str, ...]:
    """Return canonical supported opcode names in stable order."""

    return tuple(_OPCODE_SPECS)
