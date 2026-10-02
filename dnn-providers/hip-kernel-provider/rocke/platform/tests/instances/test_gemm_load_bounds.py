# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Evaluate emitted load addresses and LDS contents without a GPU."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from rocke.instances.common import gemm_universal as gemm
from . import test_gfx1250_gemm as specs


def _walk(region, guards=()):
    for op in region.ops:
        yield op, guards
        for index, child in enumerate(op.regions):
            condition = (
                ((op.operands[0], index == 0),)
                if op.name in ("scf.if", "scf.if_else")
                else ()
            )
            yield from _walk(child, guards + condition)


def _evaluate(env, sizes):
    values = {}

    def evaluate(value):
        values[value.name] = value
        return cached(value.name)

    @lru_cache(None)
    def cached(name):
        value = values[name]
        if name in env:
            return env[name]
        if name.startswith("%k0"):
            return env["k0"]
        op = value.op
        if op.name == "arith.constant":
            return op.attrs["value"]
        if op.name == "arith.constant_vec":
            return [op.attrs["fill"]] * op.attrs["vec"]
        if op.name in ("gpu.block_id", "gpu.thread_id"):
            return env[op.name + "." + op.attrs["axis"]]
        if op.name.startswith("memref.global_load"):
            ptr, index = op.operands
            offset = evaluate(index)
            assert 0 <= offset < sizes[ptr.name], (ptr.name, offset, sizes)
            return offset + 1
        args = [evaluate(v) for v in op.operands]
        kind = op.name.split(".")[-1]
        if kind == "cmp":
            a, b = args
            return {"lt": a < b, "le": a <= b, "eq": a == b}[op.attrs["pred"]]
        if kind == "select":
            return args[1] if args[0] else args[2]
        if kind == "extract":
            return args[0][op.attrs["index"]]
        functions = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a // b,
            "mod": lambda a, b: a % b,
            "and": lambda a, b: a & b,
            "smax": max,
            "smin": min,
            "readfirstlane": lambda a: a,
            "pin_sgpr": lambda a: a,
        }
        return functions[kind](*args)

    return evaluate


@pytest.mark.parametrize("k,k_off", [(32, 0), (33, 0), (33, 32), (40, 32), (65, 64)])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_direct_loads_stage_only_valid_elements(k, k_off, dtype):
    spec = specs.TestGfx1250Gemm._dtl_spec(dtype)
    kernel = gemm.build_universal_gemm(spec, arch="gfx1250")
    # Both output dimensions have a tail; every input element is distinguishable.
    m, n = 129, 130
    sizes = {"%A": m * k, "%B": n * k}
    staged = {"%A": {}, "%B": {}}
    smem_sources = {}
    copies = []
    for op, guards in _walk(kernel.body):
        if op.name == "tile.global_load_async_to_lds":
            smem_sources[op.operands[2].name] = op.operands[0].name
            copies.append((op, guards))
        elif op.name == "tile.smem_store":
            copies.append((op, guards))
    async_count = 0
    for lane in range(spec.block_size):
        env = {
            "%M": m,
            "%N": n,
            "%K": k,
            "k0": k_off,
            "gpu.block_id.x": 1,
            "gpu.block_id.y": 1,
            "gpu.block_id.z": 0,
            "gpu.thread_id.x": lane,
        }
        ev = _evaluate(env, sizes)
        for op, guards in copies:
            if not all(bool(ev(cond)) == branch for cond, branch in guards):
                continue
            if op.name == "tile.global_load_async_to_lds":
                ptr, index, smem, row, col = op.operands
                offset, r, c = ev(index), ev(row), ev(col)
                width = op.attrs["width_bytes"] // 2
                assert offset % width == 0
                assert 0 <= offset and offset + width <= sizes[ptr.name]
                for i in range(width):
                    staged[ptr.name][r, c + i] = offset + i + 1
                async_count += 1
            else:
                smem, row, col, value = op.operands
                if smem.name in smem_sources:
                    staged[smem_sources[smem.name]][ev(row), ev(col)] = ev(value)
    for ptr, extent in (("%A", m), ("%B", n)):
        expected = {
            (r, c): (
                ((128 + r) * k + k_off + c + 1)
                if 128 + r < extent and k_off + c < k
                else 0
            )
            for r in range(128)
            for c in range(32)
        }
        assert staged[ptr] == expected
    if k == 32:
        assert async_count > 0  # the aligned path remains available


def test_tdm_descriptor_uses_remaining_rows():
    captured = []
    original = gemm.build_tdm_descriptor_2d

    def capture(builder, **kwargs):
        captured.append(kwargs)
        return original(builder, **kwargs)

    with patch.object(gemm, "build_tdm_descriptor_2d", capture):
        gemm.build_universal_gemm(
            specs.TestGfx1250Gemm._tdm_spec(depth=1), arch="gfx1250"
        )
    env = {
        "%M": 129,
        "%N": 130,
        "%K": 33,
        "k0": 32,
        "gpu.block_id.x": 1,
        "gpu.block_id.y": 1,
        "gpu.block_id.z": 0,
    }
    ev = _evaluate(env, {})
    assert [ev(d["tensor_dim1"]) for d in captured] == [1, 2]
    assert [ev(d["tensor_dim0"]) for d in captured] == [1, 1]
    assert all(d["tile_dim1"] == 128 for d in captured)


def test_gfx1250_load_lowering_golden(monkeypatch):
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm23")
    monkeypatch.setenv("ROCKE_BACKEND", "python")
    expected = json.loads(
        (Path(__file__).parent / "golden" / "gemm_load_bounds_llvm23.json").read_text()
    )
    cases = {
        "direct": specs.TestGfx1250Gemm._dtl_spec(),
        "prefetch": specs.TestGfx1250Gemm._dtl_spec(prefetch=True),
        **{
            f"tdm_depth_{depth}": specs.TestGfx1250Gemm._tdm_spec(depth=depth)
            for depth in range(1, 5)
        },
    }
    actual = {
        name: hashlib.sha256(
            lower_kernel_to_llvm(
                gemm.build_universal_gemm(spec, arch="gfx1250"), arch="gfx1250"
            ).encode()
        ).hexdigest()
        for name, spec in cases.items()
    }
    assert actual == expected
