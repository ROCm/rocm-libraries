# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU interpretation of the actual peer-copy IR, plus ABI/validation coverage.

This evaluates scalar address/mask operations, not a GPU or a network model.
It verifies the builder's logical accesses; HIP mapping/execution is a later gate.
"""

from __future__ import annotations

import hashlib
import json
import operator
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rocke.core.ir import KernelDef, Region
from rocke.core.ir_serialize import canonicalize, serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.verify import verify
from rocke.instances import (
    PeerTileCopyProblem,
    PeerTileCopySpec,
    build_peer_tile_copy,
    is_valid_peer_tile_copy_spec,
    peer_tile_copy_grid,
    peer_tile_copy_signature,
)
from rocke.runtime.packing import pack_args


def _evaluate(
    kernel: KernelDef,
    spec: PeerTileCopySpec,
    problem: PeerTileCopyProblem,
    source,
    destination,
    *,
    extra_tiles: int = 0,
) -> tuple[list[int], list[int]]:
    reads: list[int] = []
    writes: list[int] = []
    arithmetic = {
        "arith.add": operator.add,
        "arith.mul": operator.mul,
        "arith.div": operator.floordiv,
        "arith.mod": operator.mod,
        "arith.and": operator.and_,
    }

    def region_body(
        region: Region, values: dict, tid: int, block: tuple[int, int]
    ) -> None:
        for op in region.ops:
            operands = [values[value.name] for value in op.operands]
            if op.name == "arith.constant":
                result = op.attrs["value"]
            elif op.name == "gpu.thread_id":
                result = tid
            elif op.name == "gpu.block_id":
                result = block[0 if op.attrs["axis"] == "x" else 1]
            elif op.name == "arith.zext":
                result = operands[0]
            elif op.name in arithmetic:
                result = arithmetic[op.name](*operands)
            elif op.name == "arith.cmp":
                result = getattr(operator, op.attrs["pred"])(*operands)
            elif op.name == "scf.if":
                if operands[0]:
                    region_body(op.regions[0], values, tid, block)
                continue
            elif op.name == "memref.global_load_typed":
                result = operands[0][operands[1]]
                reads.append(operands[1])
            elif op.name == "memref.global_store_typed":
                operands[0][operands[1]] = operands[2]
                writes.append(operands[1])
                continue
            else:
                raise AssertionError(f"unmodeled peer-copy op {op.name}")
            values[op.result.name] = result

    grid = peer_tile_copy_grid(problem.rows, problem.cols, spec)
    if grid[0] and grid[1]:
        for block_y in range(grid[1] + extra_tiles):
            for block_x in range(grid[0] + extra_tiles):
                for tid in range(spec.block_size):
                    values = {
                        f"%{key}": value for key, value in problem.values(0, 0).items()
                    }
                    values["%X"], values["%Y"] = source, destination
                    region_body(kernel.body, values, tid, (block_x, block_y))
    return reads, writes


def _expected(
    problem: PeerTileCopyProblem, source, destination
) -> tuple[list[int], list[int]]:
    reads, writes = [], []
    for row in range(problem.rows):
        for col in range(problem.cols):
            sr, sc = problem.src_row + row, problem.src_col + col
            dr, dc = problem.dst_row + row, problem.dst_col + col
            if (
                0 <= sr < problem.src_rows
                and 0 <= sc < problem.src_cols
                and 0 <= dr < problem.dst_rows
                and 0 <= dc < problem.dst_cols
            ):
                src = sr * problem.src_stride_row + sc * problem.src_stride_col
                dst = dr * problem.dst_stride_row + dc * problem.dst_stride_col
                destination[dst] = source[src]
                reads.append(src)
                writes.append(dst)
    return reads, writes


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize(
    "geometry", [(4, 64, 64), (3, 17, 128), (7, 33, 256), (1, 1, 64)]
)
def test_ir_masks_preserve_bits_guards_and_unique_writes(arch, dtype, geometry) -> None:
    spec = PeerTileCopySpec(*geometry, dtype=dtype)
    problem = PeerTileCopyProblem(
        rows=9,
        cols=21,
        src_rows=8,
        src_cols=18,
        dst_rows=10,
        dst_cols=19,
        src_stride_row=47,
        dst_stride_row=63,
        src_stride_col=2,
        dst_stride_col=3,
        src_row=-1,
        src_col=2,
        dst_row=2,
        dst_col=-3,
    )
    # Payloads include signed zero, infinities, and NaN bit patterns; no conversion.
    patterns = np.array(
        [0x0000, 0x8000, 0x7F80, 0x7FFF, 0xFFFF, 0x1234], dtype=np.uint16
    )
    source = np.resize(patterns, 8 * 47 + 32)
    actual = np.full(10 * 63 + 32, 0xABCD, dtype=np.uint16)
    expected = actual.copy()
    expected_reads, expected_writes = _expected(problem, source, expected)
    kernel = build_peer_tile_copy(spec, arch)
    assert not verify(kernel)
    reads, writes = _evaluate(kernel, spec, problem, source, actual, extra_tiles=1)
    np.testing.assert_array_equal(actual, expected)
    assert Counter(reads) == Counter(expected_reads)
    assert Counter(writes) == Counter(expected_writes)
    assert max(Counter(writes).values(), default=0) <= 1


def test_ir_keeps_large_element_offsets_without_i32_truncation() -> None:
    spec = PeerTileCopySpec(tile_rows=2, tile_cols=3)
    problem = PeerTileCopyProblem(2, 3, 2, 3, 2, 3, 1 << 32, 8)
    source = {
        row * (1 << 32) + col: row * 10 + col for row in range(2) for col in range(3)
    }
    actual = {index: 99 for index in range(16)}
    expected = actual.copy()
    _expected(problem, source, expected)
    _evaluate(build_peer_tile_copy(spec), spec, problem, source, actual)
    assert actual == expected


@pytest.mark.parametrize("empty", [(0, 0), (0, 5), (5, 0)])
def test_empty_rectangle_skips_grid_and_accesses(empty) -> None:
    spec = PeerTileCopySpec()
    problem = PeerTileCopyProblem(*empty, 0, 0, 0, 0, 1, 1)
    assert peer_tile_copy_grid(*empty, spec) == (0, 0, 1)
    assert _evaluate(build_peer_tile_copy(spec), spec, problem, {}, {}) == ([], [])


def test_abi_matches_kernel_params_and_pack_args() -> None:
    spec = PeerTileCopySpec()
    signature = peer_tile_copy_signature(spec)
    kernel = build_peer_tile_copy(spec)
    assert [entry["name"] for entry in signature] == [
        param.name for param in kernel.params
    ]
    assert [entry["type"] for entry in signature[2:]] == ["i64"] * 14
    problem = PeerTileCopyProblem(3, 5, 3, 5, 3, 5, 5, 5)
    assert len(pack_args(signature, problem.values(0x1000, 0x2000))) == 16 * 8
    assert "i64" in serialize(kernel)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype, llvm_type", [("f16", "half"), ("bf16", "bfloat")])
def test_lowering_has_i64_global_accesses_and_no_compute_or_transport(
    arch, dtype, llvm_type
) -> None:
    kernel = build_peer_tile_copy(
        PeerTileCopySpec(tile_rows=1, tile_cols=1, dtype=dtype), arch
    )
    llvm = _lower_kernel_to_llvm_python(kernel, arch=arch)
    accesses = [line for line in llvm.splitlines() if "getelementptr" in line]
    assert len(accesses) == 2
    assert all(
        f"getelementptr inbounds {llvm_type}" in line and ", i64 " in line
        for line in accesses
    )
    assert not any(
        token in llvm
        for token in ("mfma", "atomicrmw", "addrspace(3)", "fpext", "fptrunc")
    )
    assert all(op.name != "arith.trunc" for op in kernel.body.ops)


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"dtype": "f32"}, "unsupported dtype"),
        ({"tile_rows": 0}, "positive integers"),
        ({"tile_cols": -1}, "positive integers"),
        ({"tile_rows": True}, "positive integers"),
        ({"block_size": 32}, "block_size"),
        ({"tile_rows": 4097}, "tile area"),
        ({"name": ""}, "ASCII identifier"),
        ({"name": "a" * 256}, "255 characters"),
    ],
)
def test_spec_rejects_invalid_geometry_before_build(change, reason) -> None:
    spec = replace(PeerTileCopySpec(), **change)
    assert not is_valid_peer_tile_copy_spec(spec)[0]
    with pytest.raises(ValueError, match=reason):
        build_peer_tile_copy(spec)


@pytest.mark.parametrize("index", range(10))
def test_peer_family_goldens(index) -> None:
    geometries = [
        (4, 64, 64, "bf16"),
        (3, 17, 64, "f16"),
        (7, 33, 128, "bf16"),
        (9, 65, 256, "f16"),
        (1, 1, 256, "bf16"),
    ]
    spec = PeerTileCopySpec(*geometries[index // 2])
    arch = "gfx942" if index % 2 else "gfx950"
    kernel = build_peer_tile_copy(spec, arch)
    golden = json.loads(
        (Path(__file__).parent / "differential/golden/llvm_gfx_all.json").read_text()
    )
    llvm = _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor="llvm20")
    assert (
        hashlib.sha256(llvm.encode()).hexdigest()
        == golden["ll"]["peer_tile_copy"][str(index)]
    )
    assert (
        hashlib.sha256(canonicalize(kernel).encode()).hexdigest()
        == golden["ir_canonical"]["peer_tile_copy"][str(index)]
    )


def test_unsupported_target_rejected() -> None:
    assert not is_valid_peer_tile_copy_spec(PeerTileCopySpec(), "gfx1151")[0]
    with pytest.raises(ValueError, match="unsupported peer tile copy arch"):
        build_peer_tile_copy(PeerTileCopySpec(), "gfx1151")


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"rows": -1}, "non-negative"),
        ({"rows": True}, "i64 integer"),
        ({"src_stride_col": 0}, "positive"),
        ({"dst_stride_row": 1}, "must not overlap"),
        ({"src_row": (1 << 63) - 1}, "overflows"),
        ({"src_stride_row": (1 << 63) - 1}, "byte span"),
    ],
)
def test_problem_rejects_overflow_and_nonunique_output(change, reason) -> None:
    problem = PeerTileCopyProblem(2, 3, 2, 3, 2, 3, 3, 3)
    with pytest.raises(ValueError, match=reason):
        replace(problem, **change)


def test_grid_rejects_unrepresentable_axes() -> None:
    with pytest.raises(ValueError, match="grid axis"):
        peer_tile_copy_grid((1 << 63) - 1, 1, PeerTileCopySpec())
