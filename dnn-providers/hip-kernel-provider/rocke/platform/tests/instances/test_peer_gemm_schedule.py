# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only ownership, finite progress, output reconstruction and scheduled IR tests."""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from rocke.core.ir_serialize import canonicalize, serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.verify import verify
from rocke.instances.common import peer_gemm as module
from rocke.instances.common.peer_gemm import (
    PeerGemmScheduleSpec,
    PeerGemmSpec,
    build_peer_gemm,
    build_scheduled_peer_gemm,
    is_valid_schedule_spec,
    peer_gemm_signature,
    scheduled_peer_gemm_cta_work,
    scheduled_peer_gemm_grid,
    scheduled_peer_gemm_tile_coordinate,
)


@pytest.mark.parametrize("sources", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("order", ["source_grouped", "tile_interleaved"])
@pytest.mark.parametrize("cap", [0, 1, 2, 3, 7, 100])
def test_every_tile_has_one_owner_and_bounded_finite_progress(sources, order, cap):
    spec = PeerGemmScheduleSpec(PeerGemmSpec(32, 48, 32, sources), order, cap)
    assert spec.tile_count == sources * 6
    assert scheduled_peer_gemm_grid(spec) == (
        min(cap or sources * 6, sources * 6),
        1,
        1,
    )
    work = [
        list(scheduled_peer_gemm_cta_work(spec, cta)) for cta in range(spec.cta_count)
    ]
    flat = list(itertools.chain.from_iterable(work))
    expected = set(itertools.product(range(sources), range(2), range(3)))
    assert len(flat) == len(set(flat)) == spec.tile_count
    assert set(flat) == expected
    lengths = [len(items) for items in work]
    assert max(lengths) - min(lengths) <= 1
    for cta, items in enumerate(work):
        assert len(items) == len(range(cta, spec.tile_count, spec.cta_count))
        assert items == [
            scheduled_peer_gemm_tile_coordinate(spec, index)
            for index in range(cta, spec.tile_count, spec.cta_count)
        ]


@pytest.mark.parametrize("order", ["source_grouped", "tile_interleaved"])
@pytest.mark.parametrize("cap", [0, 1, 3])
def test_scheduled_tiles_reconstruct_the_same_panel_product(order, cap):
    base = PeerGemmSpec(32, 48, 32, sources=4, dtype="f16")
    spec = PeerGemmScheduleSpec(base, order, cap)
    rng = np.random.default_rng(42)
    a = rng.integers(-3, 4, size=(4, 32, 32)).astype(np.float32) / 8
    b = rng.integers(-4, 5, size=(32, 48)).astype(np.float32) / 8
    c = np.full((4, 32, 48), np.nan, dtype=np.float32)
    writes = np.zeros(c.shape, dtype=np.int32)
    # Reverse CTA order: correctness must not depend on their execution order.
    for cta in reversed(range(spec.cta_count)):
        for source, row, column in scheduled_peer_gemm_cta_work(spec, cta):
            rows, columns = slice(row * 16, (row + 1) * 16), slice(
                column * 16, (column + 1) * 16
            )
            c[source, rows, columns] = a[source, rows] @ b[:, columns]
            writes[source, rows, columns] += 1
    np.testing.assert_array_equal(writes, np.ones_like(writes))
    np.testing.assert_array_equal(c.astype(np.float16), (a @ b).astype(np.float16))


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("dtype", ["f16", "bf16"])
@pytest.mark.parametrize("order", ["source_grouped", "tile_interleaved"])
def test_scheduled_program_reuses_compute_and_has_guarded_grid_stride(
    monkeypatch, arch, dtype, order
):
    base = PeerGemmSpec(32, 48, 32, sources=4, dtype=dtype)
    spec = PeerGemmScheduleSpec(base, order, 3)
    original = module._emit_peer_gemm_compute
    calls = []

    def record(*args):
        calls.append(args[1])
        return original(*args)

    monkeypatch.setattr(module, "_emit_peer_gemm_compute", record)
    legacy = build_peer_gemm(base, arch)
    kernel = build_scheduled_peer_gemm(spec, arch)
    assert calls == [base, base]
    assert verify(kernel) == []
    assert kernel.name != legacy.name
    assert [param.name for param in kernel.params] == ["A_ptrs", "B", "C_ptrs"]
    assert len(peer_gemm_signature(base)) == 3
    guard = next(op for op in kernel.body.ops if op.name == "scf.if")
    loop = next(op for op in guard.regions[0].ops if op.name == "scf.for")
    assert loop.attrs == {"iv": "%work", "iv_type": "i32"}
    assert loop.operands[1].op.attrs["value"] == spec.tile_count
    assert loop.operands[2].op.attrs["value"] == spec.cta_count
    llvm = _lower_kernel_to_llvm_python(kernel, arch=arch)
    assert "llvm.amdgcn.mfma" in llvm
    assert "llvm.amdgcn.workgroup.id.x" in llvm
    assert "llvm.amdgcn.workgroup.id.y" in llvm
    assert "llvm.amdgcn.workgroup.id.z" in llvm
    assert "atomic" not in llvm and "barrier" not in llvm


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"gemm": None}, "gemm must"),
        ({"work_order": "random"}, "work_order"),
        ({"work_order": None}, "work_order"),
        ({"max_ctas": -1}, "max_ctas"),
        ({"max_ctas": True}, "max_ctas"),
        ({"max_ctas": 1.0}, "max_ctas"),
        ({"max_ctas": 1 << 31}, "max_ctas"),
        ({"gemm": PeerGemmSpec(17, 16, 16)}, "divisible"),
    ],
)
def test_invalid_schedule_is_rejected_before_emission(change, reason):
    spec = replace(PeerGemmScheduleSpec(PeerGemmSpec(32, 48, 32)), **change)
    valid, why = is_valid_schedule_spec(spec)
    assert not valid and reason in why
    with pytest.raises(ValueError):
        build_scheduled_peer_gemm(spec)


@pytest.mark.parametrize("index", [-1, 12, True, 1.5])
def test_host_coordinate_oracle_rejects_invalid_indices(index):
    spec = PeerGemmScheduleSpec(PeerGemmSpec(32, 48, 32))
    with pytest.raises(ValueError):
        scheduled_peer_gemm_tile_coordinate(spec, index)


def test_zero_cap_and_excess_cap_are_the_same_effective_program():
    spec = PeerGemmScheduleSpec(PeerGemmSpec(32, 48, 32), max_ctas=0)
    excess = replace(spec, max_ctas=100)
    assert spec.kernel_name() == excess.kernel_name()
    assert serialize(build_scheduled_peer_gemm(spec)) == serialize(
        build_scheduled_peer_gemm(excess)
    )
    with pytest.raises(ValueError):
        list(scheduled_peer_gemm_cta_work(spec, spec.cta_count))


@pytest.mark.parametrize("index", range(72))
def test_scheduled_family_llvm20_and_canonical_goldens(index):
    shapes = [
        (16, 16, 16, 1, "f16"),
        (16, 16, 16, 2, "bf16"),
        (32, 16, 32, 3, "f16"),
        (16, 32, 64, 8, "bf16"),
        (32, 32, 64, 4, "f16"),
        (32, 32, 32, 3, "bf16"),
    ]
    config = index // 2
    spec = PeerGemmScheduleSpec(
        PeerGemmSpec(*shapes[config // 6]),
        ("source_grouped", "tile_interleaved")[(config // 3) % 2],
        (0, 1, 3)[config % 3],
    )
    arch = "gfx942" if index % 2 else "gfx950"
    kernel = build_scheduled_peer_gemm(spec, arch)
    snapshot = json.loads(
        (Path(__file__).parent / "differential/golden/llvm_gfx_all.json").read_text()
    )
    llvm = _lower_kernel_to_llvm_python(kernel, arch=arch, llvm_flavor="llvm20")
    assert (
        hashlib.sha256(llvm.encode()).hexdigest()
        == snapshot["ll"]["peer_gemm_scheduled"][str(index)]
    )
    assert (
        hashlib.sha256(canonicalize(kernel).encode()).hexdigest()
        == snapshot["ir_canonical"]["peer_gemm_scheduled"][str(index)]
    )
