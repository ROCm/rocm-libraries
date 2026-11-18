# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host placement/numerical reconstruction and existing-compute reuse contracts.

The host numerical model is not GPU MFMA execution. Compute-body comparison
proves the wrapper reuses the existing builder semantics; hardware is a later gate.
"""

from __future__ import annotations

import copy
import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rocke.core.ir import KernelDef, Region, Value
from rocke.core.ir_serialize import canonicalize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.verify import verify
from rocke.instances.common.mfma_gemm import build_mfma_gemm
from rocke.instances.common.peer_gemm import (
    PeerGemmAllocation,
    PeerGemmSpec,
    build_peer_gemm,
    is_valid_spec,
    make_peer_gemm_tables,
    peer_gemm_grid,
    peer_gemm_signature,
)
from rocke.runtime.packing import pack_args


def _allocation(device, slot, words):
    return PeerGemmAllocation(
        device, (device + 1) * 0x1000000 + slot * 0x10000, words * 2
    )


def _panel_at(pointer, regions, storage, rows, cols):
    for region in regions:
        if region.pointer <= pointer < region.pointer + region.nbytes:
            offset = (pointer - region.pointer) // 2
            return storage[(region.device, region.pointer)][
                offset : offset + rows * cols
            ].reshape(rows, cols)
    raise AssertionError("panel pointer is outside supplied storage")


@pytest.mark.parametrize("sources", [1, 2, 3, 8])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("input_placement", ["gather", "replicated"])
@pytest.mark.parametrize("output_placement", ["local", "scatter"])
def test_table_plan_reconstructs_distributed_matmul(
    sources, dtype, input_placement, output_placement
) -> None:
    spec = PeerGemmSpec(16, 16, 32, sources=sources, dtype=dtype)
    devices = tuple(range(sources))
    # Small integral inputs make the FP32 mathematical reference exactly representable.
    full_a = (np.arange(sources * 16 * 32).reshape(sources * 16, 32) % 3 - 1).astype(
        np.float32
    )
    full_b = (np.arange(32 * sources * 16).reshape(32, sources * 16) % 5 - 2).astype(
        np.float32
    )
    expected = full_a @ full_b
    storage = {}
    a_shards = tuple(_allocation(s, 1, 16 * 32) for s in devices)
    b_shards = tuple(_allocation(r, 2, 32 * 16) for r in devices)
    c_scattered = tuple(_allocation(s, 3, sources * 16 * 16) for s in devices)
    a_replicated = tuple(_allocation(r, 4, sources * 16 * 32) for r in devices)
    c_local = tuple(_allocation(r, 5, sources * 16 * 16) for r in devices)
    for source, region in enumerate(a_shards):
        storage[(region.device, region.pointer)] = (
            full_a[source * 16 : (source + 1) * 16].copy().ravel()
        )
    for rank, region in enumerate(b_shards):
        storage[(region.device, region.pointer)] = (
            full_b[:, rank * 16 : (rank + 1) * 16].copy().ravel()
        )
    for region in a_replicated:
        storage[(region.device, region.pointer)] = full_a.copy().ravel()
    for region in (*c_scattered, *c_local):
        storage[(region.device, region.pointer)] = np.full(
            sources * 16 * 16, np.nan, dtype=np.float32
        )
    writes = []
    for rank in devices:
        inputs = a_shards if input_placement == "gather" else (a_replicated[rank],)
        outputs = (c_local[rank],) if output_placement == "local" else c_scattered
        tables = make_peer_gemm_tables(
            spec,
            rank=rank,
            rank_devices=devices,
            a_buffers=inputs,
            b_buffer=b_shards[rank],
            c_buffers=outputs,
            input_placement=input_placement,
            output_placement=output_placement,
        )
        remote = input_placement == "gather" or output_placement == "scatter"
        expected_paths = tuple(
            (rank, other) for other in devices if remote and other != rank
        )
        assert tables.required_peer_paths == expected_paths
        assert (
            struct.unpack("<" + "Q" * sources, tables.a_table_bytes())
            == tables.a_addresses
        )
        assert (
            struct.unpack("<" + "Q" * sources, tables.c_table_bytes())
            == tables.c_addresses
        )
        b = storage[(b_shards[rank].device, b_shards[rank].pointer)].reshape(32, 16)
        for source in devices:
            a = _panel_at(tables.a_addresses[source], inputs, storage, 16, 32)
            c = _panel_at(tables.c_addresses[source], outputs, storage, 16, 16)
            c[:] = a @ b
            owner = rank if output_placement == "local" else source
            writes.append(
                (
                    owner,
                    tables.c_addresses[source],
                    tables.c_addresses[source] + 16 * 16 * 2,
                )
            )
    for index, (device, lo, hi) in enumerate(writes):
        assert all(
            other_device != device or max(lo, other_lo) >= min(hi, other_hi)
            for other_device, other_lo, other_hi in writes[index + 1 :]
        )
    if output_placement == "local":
        for rank, region in enumerate(c_local):
            actual = storage[(region.device, region.pointer)].reshape(sources * 16, 16)
            np.testing.assert_array_equal(
                actual, expected[:, rank * 16 : (rank + 1) * 16]
            )
    else:
        for source, region in enumerate(c_scattered):
            panels = storage[(region.device, region.pointer)].reshape(sources, 16, 16)
            actual = panels.transpose(1, 0, 2).reshape(16, sources * 16)
            np.testing.assert_array_equal(
                actual, expected[source * 16 : (source + 1) * 16]
            )


def _replace_inputs(region, replacements):
    for op in region.ops:
        op.operands = [replacements.get(value.name, value) for value in op.operands]
        for child in op.regions:
            _replace_inputs(child, replacements)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_compute_body_is_the_existing_mfma_gemm_body(dtype, arch) -> None:
    spec = PeerGemmSpec(32, 32, 64, dtype=dtype)
    peer = build_peer_gemm(spec, arch)
    baseline = build_mfma_gemm(spec.to_mfma_spec(), arch)
    guard = next(op for op in peer.body.ops if op.name == "scf.if")
    selected_a, selected_c = guard.regions[0].ops[:2]
    assert selected_a.name == selected_c.name == "memref.global_load_typed"
    core = copy.deepcopy(Region("entry", guard.regions[0].ops[2:]))
    element_pointer = baseline.params[0].type
    _replace_inputs(
        core,
        {
            selected_a.result.name: Value("%A", element_pointer),
            selected_c.result.name: Value("%C", element_pointer),
        },
    )
    original = copy.deepcopy(Region("entry", baseline.body.ops[:-1]))
    left = KernelDef("compute", baseline.params[:3], core, dict(baseline.attrs))
    right = KernelDef("compute", baseline.params[:3], original, dict(baseline.attrs))
    assert canonicalize(left) == canonicalize(right)


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_one_kernel_contains_pointer_selection_and_mfma(dtype, arch) -> None:
    spec = PeerGemmSpec(32, 16, 32, sources=3, dtype=dtype)
    kernel = build_peer_gemm(spec, arch)
    assert not verify(kernel)
    assert peer_gemm_grid(spec) == (1, 2, 3)
    signature = peer_gemm_signature(spec)
    assert [entry["name"] for entry in signature] == [p.name for p in kernel.params]
    assert (
        len(pack_args(signature, {"A_ptrs": 0x1000, "B": 0x2000, "C_ptrs": 0x3000}))
        == 24
    )
    llvm = _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor="llvm20")
    assert llvm.count("define amdgpu_kernel") == 1
    assert llvm.count("load ptr addrspace(1)") == 2
    assert "llvm.amdgcn.mfma" in llvm
    assert "atomicrmw" not in llvm and "addrspace(3)" not in llvm
    guard = next(op for op in kernel.body.ops if op.name == "scf.if")
    assert all(op.name != "memref.global_load_typed" for op in kernel.body.ops)
    assert (
        guard.regions[0].ops[0].operands[1].name
        == guard.regions[0].ops[1].operands[1].name
    )


@pytest.mark.parametrize(
    "change, reason",
    [
        ({"sources": 0}, "sources"),
        ({"sources": 9}, "sources"),
        ({"sources": True}, "sources"),
        ({"rows_per_source": 0}, "positive i32"),
        ({"cols_per_rank": 17}, "divisible"),
        ({"k": 15}, "divisible"),
        ({"k": True}, "positive i32"),
        ({"dtype": "f32"}, "unsupported dtype"),
        ({"name": "bad-name"}, "ASCII"),
        ({"rows_per_source": 1 << 30}, "span"),
    ],
)
def test_spec_rejects_unsupported_shapes(change, reason) -> None:
    spec = replace(PeerGemmSpec(16, 16, 32), **change)
    assert not is_valid_spec(spec)[0]
    with pytest.raises(ValueError, match=reason):
        build_peer_gemm(spec)


@pytest.mark.parametrize("index", range(12))
def test_peer_gemm_family_goldens(index) -> None:
    configs = [
        (16, 16, 16, 1, "f16"),
        (16, 16, 16, 2, "bf16"),
        (32, 16, 32, 3, "f16"),
        (16, 32, 64, 8, "bf16"),
        (32, 32, 64, 4, "f16"),
        (32, 32, 32, 3, "bf16"),
    ]
    spec = PeerGemmSpec(*configs[index // 2])
    arch = "gfx942" if index % 2 else "gfx950"
    kernel = build_peer_gemm(spec, arch)
    golden = json.loads(
        (Path(__file__).parent / "differential/golden/llvm_gfx_all.json").read_text()
    )
    llvm = _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor="llvm20")
    assert (
        hashlib.sha256(llvm.encode()).hexdigest()
        == golden["ll"]["peer_gemm"][str(index)]
    )
    assert (
        hashlib.sha256(canonicalize(kernel).encode()).hexdigest()
        == golden["ir_canonical"]["peer_gemm"][str(index)]
    )


def test_planner_checks_owners_sizes_aliasing_and_local_table_allocations() -> None:
    spec = PeerGemmSpec(16, 16, 32)
    inputs = (_allocation(0, 1, 16 * 32), _allocation(1, 1, 16 * 32))
    b = _allocation(0, 2, 32 * 16)
    c = _allocation(0, 3, 2 * 16 * 16)
    args = {
        "rank": 0,
        "rank_devices": (0, 1),
        "a_buffers": inputs,
        "b_buffer": b,
        "c_buffers": (c,),
    }
    plan = make_peer_gemm_tables(spec, **args)
    a_table, c_table = _allocation(0, 6, 8), _allocation(0, 7, 8)
    assert plan.kernel_values(a_table, c_table) == {
        "A_ptrs": a_table.pointer,
        "B": b.pointer,
        "C_ptrs": c_table.pointer,
    }
    with pytest.raises(ValueError, match="source count"):
        replace(plan, c_addresses=plan.c_addresses[:-1])
    with pytest.raises(ValueError, match="B must be local"):
        replace(plan, execution_device=1)
    with pytest.raises(ValueError, match="owner"):
        plan.kernel_values(replace(a_table, device=1), c_table)
    with pytest.raises(ValueError, match="overlap"):
        plan.kernel_values(a_table, a_table)
    with pytest.raises(ValueError, match="overlap"):
        make_peer_gemm_tables(
            spec, **{**args, "c_buffers": (replace(c, pointer=b.pointer),)}
        )
    with pytest.raises(ValueError, match="owner"):
        make_peer_gemm_tables(spec, **{**args, "b_buffer": replace(b, device=1)})
    with pytest.raises(ValueError, match="requires at least"):
        make_peer_gemm_tables(spec, **{**args, "c_buffers": (replace(c, nbytes=16),)})
    with pytest.raises(ValueError, match="unique"):
        make_peer_gemm_tables(spec, **{**args, "rank_devices": (0, 0)})
    with pytest.raises(ValueError, match="rank"):
        make_peer_gemm_tables(spec, **{**args, "rank": 2})


@pytest.mark.parametrize(
    "args", [(-1, 0x1000, 16), (0, 0x1001, 16), (0, 0x1000, 0), (0, 1 << 64, 16)]
)
def test_allocation_rejects_invalid_regions(args) -> None:
    with pytest.raises(ValueError):
        PeerGemmAllocation(*args)
