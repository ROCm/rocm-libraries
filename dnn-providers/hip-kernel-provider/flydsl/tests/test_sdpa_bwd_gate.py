# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The per-kernel gate of the combined (merged + dBias reduce) objects.

A two-kernel ELF fixture stands in for a ``bias`` merged object: its two functions are read by ELF
symbol (``analyze.function_bytes``), and ``sdpa_bwd_gate.pin_gate`` holds the merged kernel's
bytes and resource row and the reduce kernel's resource row to the pins'. Pure Python: no
llvm-objdump, no GPU.
"""

from __future__ import annotations

import hashlib
import struct
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
sys.path.insert(0, str(FLYDSL / "tools"))
import analyze  # noqa: E402
import sdpa_bwd_gate  # noqa: E402

MERGED = "fmha_bwd_merged_gfx950_kernel_0"
REDUCE = "fmha_bwd_dbias_reduce_gfx950_kernel_1"
TEXT_ADDR = 0x2100


def _elf(functions: dict[str, bytes]) -> bytes:
    """A minimal ELF64 whose `.text` holds @p functions (256-byte aligned, as the toolchain places kernels) and `.dynsym` names them."""
    text, names, symbols = b"", b"\0", [bytes(24)]
    for name, code in functions.items():
        text += bytes(-len(text) % 256)
        symbols.append(
            struct.pack(
                "<IBBHQQ", len(names), 0x12, 0, 1, TEXT_ADDR + len(text), len(code)
            )
        )
        names += name.encode() + b"\0"
        text += code
    shstr = b"\0.text\0.dynsym\0.dynstr\0.shstrtab\0"
    body, offsets = b"", {}
    for key, data in (
        (".text", text),
        (".dynsym", b"".join(symbols)),
        (".dynstr", names),
        (".shstrtab", shstr),
    ):
        offsets[key] = (64 + len(body), len(data))
        body += data
    shoff = 64 + len(body)

    def header(name, kind, addr, key):
        off, size = offsets[key]
        return struct.pack(
            "<IIQQQQIIQQ", shstr.index(name), kind, 0, addr, off, size, 0, 0, 1, 0
        )

    sections = (
        bytes(64)
        + header(b".text", 1, TEXT_ADDR, ".text")
        + header(b".dynsym", 11, 0, ".dynsym")
        + header(b".dynstr", 3, 0, ".dynstr")
        + header(b".shstrtab", 3, 0, ".shstrtab")
    )
    ident = b"\x7fELF" + bytes([2, 1, 1, 0]) + bytes(8)
    elf_header = ident + struct.pack(
        "<HHIQQQIHHHHHH", 3, 224, 1, 0, 0, shoff, 0, 64, 0, 0, 64, 5, 4
    )
    return elf_header + body + sections


ROW = {
    "vgpr": 256,
    "agpr": 88,
    "vgpr_total": 344,
    "sgpr": 106,
    "sgpr_spill": 151,
    "vgpr_spill": 0,
    "scratch_bytes": 0,
    "lds_bytes": 70656,
    "occupancy": 1,
}
REDUCE_ROW = {**ROW, "vgpr": 81, "agpr": 0, "vgpr_total": 84, "sgpr": 44}


def _facts(blob: bytes, rows: dict[str, dict]) -> dict[str, dict]:
    """The per-kernel facts the gate derives from an object's functions, with the resource rows @p rows."""
    return {
        name: {
            "function_sha256": hashlib.sha256(
                analyze.function_bytes(blob, name)
            ).hexdigest(),
            "resources": dict(rows[name]),
        }
        for name in rows
    }


MERGED_CODE = bytes(range(251)) * 8
REDUCE_CODE = b"\x07" * 100
PIN = _elf({MERGED: MERGED_CODE})
# The reduce pin's row is the CSV's: strings.
MERGED_PIN = {
    "hash": "33d3a5c694cc",
    "function_sha256": _facts(PIN, {MERGED: ROW})[MERGED]["function_sha256"],
    "resources": {k: str(v) for k, v in ROW.items()},
}
REDUCE_PIN = {
    "hash": "bb2f7c351824",
    "resources": {k: str(v) for k, v in REDUCE_ROW.items()},
}


def _verdict(blob: bytes, merged_row=ROW, reduce_row=REDUCE_ROW) -> dict:
    kernels = _facts(blob, {MERGED: merged_row, REDUCE: reduce_row})
    return sdpa_bwd_gate.pin_gate(kernels, MERGED, REDUCE, MERGED_PIN, REDUCE_PIN)


def test_functions_are_read_by_symbol() -> None:
    blob = _elf({MERGED: MERGED_CODE, REDUCE: REDUCE_CODE})
    assert analyze.function_bytes(blob, MERGED) == MERGED_CODE
    assert analyze.function_bytes(blob, REDUCE) == REDUCE_CODE


def test_the_combined_object_equal_to_the_pins_passes() -> None:
    verdict = _verdict(_elf({MERGED: MERGED_CODE, REDUCE: REDUCE_CODE}))
    assert verdict["verdict"] == "ok" and verdict["problems"] == []


def test_the_reduce_bytes_are_not_compared() -> None:
    """The reduce kernel differs from its pin by a few bytes under the merged kernel's llvm_options; only its row is held."""
    verdict = _verdict(_elf({MERGED: MERGED_CODE, REDUCE: b"\x09" * 96}))
    assert verdict["verdict"] == "ok"


def test_the_object_around_the_merged_function_may_move() -> None:
    """Where the merged function sits in the object is no part of the gate: its bytes are."""
    verdict = _verdict(_elf({REDUCE: REDUCE_CODE, MERGED: MERGED_CODE}))
    assert verdict["verdict"] == "ok"


def test_a_changed_merged_function_is_red() -> None:
    changed = bytearray(MERGED_CODE)
    changed[1000] ^= 1
    verdict = _verdict(_elf({MERGED: bytes(changed), REDUCE: REDUCE_CODE}))
    assert verdict["verdict"] == "red"
    assert verdict["problems"] == [
        "merged function bytes differ from the pin 33d3a5c694cc"
    ]


def test_a_changed_merged_row_is_red() -> None:
    blob = _elf({MERGED: MERGED_CODE, REDUCE: REDUCE_CODE})
    verdict = _verdict(blob, merged_row={**ROW, "sgpr_spill": 152})
    assert verdict["verdict"] == "red"
    assert verdict["problems"] == [
        "merged resource row differs from the pin 33d3a5c694cc: sgpr_spill 152 != 151"
    ]


def test_a_changed_reduce_row_is_red() -> None:
    blob = _elf({MERGED: MERGED_CODE, REDUCE: REDUCE_CODE})
    verdict = _verdict(blob, reduce_row={**REDUCE_ROW, "vgpr": 103, "vgpr_total": 106})
    assert verdict["verdict"] == "red"
    assert verdict["problems"] == [
        "reduce resource row differs from the pin bb2f7c351824: vgpr 103 != 81, vgpr_total 106 != 84"
    ]


def test_every_failure_is_reported() -> None:
    changed = bytearray(MERGED_CODE)
    changed[0] ^= 1
    blob = _elf({MERGED: bytes(changed), REDUCE: REDUCE_CODE})
    verdict = _verdict(
        blob, merged_row={**ROW, "occupancy": 2}, reduce_row={**REDUCE_ROW, "sgpr": 45}
    )
    assert verdict["verdict"] == "red" and len(verdict["problems"]) == 3
    assert (
        verdict["merged_function_equal"],
        verdict["merged_row_equal"],
        verdict["reduce_row_equal"],
    ) == (False, False, False)


def test_a_two_kernel_object_needs_a_named_kernel() -> None:
    doc = {"amdhsa.kernels": [{".name": MERGED}, {".name": REDUCE}]}
    with pytest.raises(ValueError, match="name one"):
        analyze._kernel(doc, None)
    assert analyze._kernel(doc, REDUCE)[".name"] == REDUCE
    with pytest.raises(ValueError, match="no kernel"):
        analyze._kernel(doc, "other")
    assert (
        analyze._kernel({"amdhsa.kernels": [{".name": MERGED}]}, None)[".name"]
        == MERGED
    )
