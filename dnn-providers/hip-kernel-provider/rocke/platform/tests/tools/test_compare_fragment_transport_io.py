# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The host I/O oracle detects changed writes and refuses unmodelled operations."""

import importlib.util
import sys

import pytest

from rocke.assets import platform_tools_dir
from rocke.core.ir import BF16, I32, IRBuilder, PtrType


def _tool():
    root = platform_tools_dir()
    spec = importlib.util.spec_from_file_location(
        "fragment_io_oracle", root / "compare_fragment_transport_io.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(root))
    return module


def _copy(offset):
    b = IRBuilder("copy")
    a = b.param("A", PtrType(I32, "global"))
    out = b.param("B", PtrType(I32, "global"))
    zero = b.const_i32(0)
    value = b.global_load(a, zero, I32, align=4)
    b.global_store(out, b.const_i32(offset), value, align=4)
    return b.kernel


def test_oracle_detects_a_changed_store_offset():
    tool = _tool()
    memory = {"%A": bytes(range(16)), "%B": bytes(16)}
    pointers = {"%A": ("%A", 0), "%B": ("%B", 0)}
    baseline = tool.run(_copy(0), {}, 0, memory, pointers, 0, 0)
    candidate = tool.run(_copy(1), {}, 0, memory, pointers, 0, 0)
    assert baseline != candidate
    assert baseline[1] == [("%B", 0, "00010203")]


def test_oracle_refuses_unknown_operations_instead_of_skipping_them():
    tool = _tool()
    b = IRBuilder("unmodelled")
    b._op("test.unmodelled", [])
    with pytest.raises(NotImplementedError, match="test.unmodelled"):
        tool.run(b.kernel, {}, 0, {}, {}, 0, 0)


def test_oracle_refuses_unmodelled_comparison_predicate():
    tool = _tool()
    b = IRBuilder("unmodelled_predicate")
    # A legal IR comparison that the intentionally bounded oracle doesn't model.
    b.cmp_le(b.const_i32(1), b.const_i32(2))
    with pytest.raises(NotImplementedError, match=r"^le$"):
        tool.run(b.kernel, {}, 0, {}, {}, 0, 0)


def test_bf16_decode_uses_high_half_of_fp32_bits():
    tool = _tool()
    assert tool.floating(bytes.fromhex("803f"), BF16) == 1.0
