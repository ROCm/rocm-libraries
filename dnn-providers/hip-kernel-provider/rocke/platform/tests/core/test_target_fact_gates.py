# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Target-specific operations are admitted by the target's recorded facts.

Each LLVM-23-only operation (tensor data mover, cluster launch, multicast
loads, data prefetch, split and named barriers, ...) lowers on a target whose
``arch_specs.json`` row records the fact that admits it, and is refused on
every other target. The refusal names the operation, the fact and the target.
Both engines must apply the same rule with the same message, so every case is
lowered with both, for every known target: a target that records the fact must
lower the operation identically in both engines, any other target must be
refused by both.
"""

from __future__ import annotations

import pytest
from rocke.core.arch import ArchTarget, known_arches
from rocke.core.ir import F32, I16, I32, I64, IRBuilder, PtrType
from rocke.tests.engines import (
    assert_both_engines_reject,
    assert_both_engines_reject_hip,
    lower_hip_with_both_engines,
    lower_with_both_engines,
)

# How each fact named in a refusal reads off a target's row.
_FACTS = {
    "memory.has_tdm": lambda t: t.memory.has_tdm,
    "memory.has_cluster_launch": lambda t: t.memory.has_cluster_launch,
    "memory.has_multicast_load": lambda t: t.memory.has_multicast_load,
    "memory.has_scalar_data_prefetch": lambda t: t.memory.has_scalar_data_prefetch,
    "memory.has_global_prefetch": lambda t: t.memory.has_global_prefetch,
    "target_family gfx12_cdna": lambda t: t.target_family == "gfx12_cdna",
}


def _kernel(emit):
    """A kernel holding one operation; ``emit(b, src, dst, flat, lds)`` adds it."""
    b = IRBuilder("one_op")
    src = b.param("src", PtrType(I16, "global"), readonly=True, align=16)
    dst = b.param("dst", PtrType(I32, "global"), align=16)
    flat = b.param("flat", PtrType(F32, "private"))
    lds = b.smem_addr_of(b.smem_alloc(I64, [4], name_hint="stage"))
    result = emit(b, src, dst, flat, lds)
    if result is not None:
        b.global_store(dst, b.const_i32(0), result)
    b.ret()
    return b.kernel


def _descriptors(b):
    d4 = b.zero_vec(I32, 4)
    d8 = b.zero_vec(I32, 8)
    return d4, d8, d4, d4, d8


def _with_cluster_dims(b, src, dst, flat, lds):
    b.set_cluster_dims(2)


def _transposed_load(b, src, dst, flat, lds):
    b.global_load_tr16_b128(src, dtype=I16)


# operation name -> (fact that admits it, emitter)
_CASES = {
    "tensor_load_to_lds": (
        "memory.has_tdm",
        lambda b, s, d, f, l: b.tensor_load_to_lds(*_descriptors(b), cachepolicy=5),
    ),
    "tensor_store_from_lds": (
        "memory.has_tdm",
        lambda b, s, d, f, l: b.tensor_store_from_lds(*_descriptors(b), cachepolicy=6),
    ),
    "s_wait_tensorcnt": ("memory.has_tdm", lambda b, s, d, f, l: b.s_wait_tensorcnt(3)),
    "cluster_dims": ("memory.has_cluster_launch", _with_cluster_dims),
    "cluster_id": (
        "memory.has_cluster_launch",
        lambda b, s, d, f, l: b.cluster_id("x"),
    ),
    "cluster_workgroup_id": (
        "memory.has_cluster_launch",
        lambda b, s, d, f, l: b.cluster_workgroup_id("y"),
    ),
    "cluster_workgroup_max_id": (
        "memory.has_cluster_launch",
        lambda b, s, d, f, l: b.cluster_workgroup_max_id("z"),
    ),
    "cluster_workgroup_flat_id": (
        "memory.has_cluster_launch",
        lambda b, s, d, f, l: b.cluster_workgroup_flat_id(),
    ),
    "cluster_workgroup_max_flat_id": (
        "memory.has_cluster_launch",
        lambda b, s, d, f, l: b.cluster_workgroup_max_flat_id(),
    ),
    "cluster_barrier": (
        "memory.has_cluster_launch",
        lambda b, s, d, f, l: b.cluster_barrier(),
    ),
    "cluster_load": (
        "memory.has_multicast_load",
        lambda b, s, d, f, l: b.cluster_load(d, b.const_i32(1)),
    ),
    "cluster_load_async_to_lds": (
        "memory.has_multicast_load",
        lambda b, s, d, f, l: b.cluster_load_async_to_lds(
            d, l, b.const_i32(1), width_bytes=4
        ),
    ),
    "s_prefetch_data": (
        "memory.has_scalar_data_prefetch",
        lambda b, s, d, f, l: b.s_prefetch_data(d, b.const_i32(1)),
    ),
    "s_buffer_prefetch_data": (
        "memory.has_scalar_data_prefetch",
        lambda b, s, d, f, l: b.s_buffer_prefetch_data(
            b.buffer_rsrc(d, b.const_i32(64)), b.const_i32(1)
        ),
    ),
    "global_prefetch": (
        "memory.has_global_prefetch",
        lambda b, s, d, f, l: b.global_prefetch(d),
    ),
    "flat_prefetch": (
        "memory.has_global_prefetch",
        lambda b, s, d, f, l: b.flat_prefetch(f),
    ),
    "s_setreg": (
        "target_family gfx12_cdna",
        lambda b, s, d, f, l: b.s_setreg(1537, b.const_i32(1)),
    ),
    "s_barrier_signal": (
        "target_family gfx12_cdna",
        lambda b, s, d, f, l: b.s_barrier_signal(1),
    ),
    "s_barrier_wait": (
        "target_family gfx12_cdna",
        lambda b, s, d, f, l: b.s_barrier_wait(2),
    ),
    "s_barrier_leave": (
        "target_family gfx12_cdna",
        lambda b, s, d, f, l: b.s_barrier_leave(4),
    ),
    "s_barrier_init": (
        "target_family gfx12_cdna",
        lambda b, s, d, f, l: b.s_barrier_init(l, b.const_i32(2)),
    ),
    "s_barrier_signal_var": (
        "target_family gfx12_cdna",
        lambda b, s, d, f, l: b.s_barrier_signal_var(l, b.const_i32(2)),
    ),
    "s_barrier_join": (
        "target_family gfx12_cdna",
        lambda b, s, d, f, l: b.s_barrier_join(l),
    ),
    "s_wakeup_barrier": (
        "target_family gfx12_cdna",
        lambda b, s, d, f, l: b.s_wakeup_barrier(l),
    ),
    "global_store_async_from_lds": (
        "target_family gfx12_cdna",
        lambda b, s, d, f, l: b.global_store_async_from_lds(d, l, width_bytes=16),
    ),
    "global_load_tr16_b128": (
        "target_family gfx12_cdna",
        _transposed_load,
    ),
}


@pytest.mark.parametrize("arch", known_arches())
@pytest.mark.parametrize("op", sorted(_CASES))
def test_operation_follows_target_fact(op, arch):
    fact, emit = _CASES[op]
    kernel = _kernel(emit)
    if _FACTS[fact](ArchTarget.from_gfx(arch)):
        lower_with_both_engines(kernel, arch, "llvm23")
    else:
        assert_both_engines_reject(
            kernel,
            arch,
            ValueError,
            f"{op} requires a target with {fact}, got {arch}",
            "llvm23",
        )


@pytest.mark.parametrize("fact", sorted({fact for fact, _ in _CASES.values()}))
def test_each_fact_is_recorded_by_some_target_and_missing_on_another(fact):
    """Guard against a vacuous sweep: every fact must both admit and refuse."""
    holds = [_FACTS[fact](ArchTarget.from_gfx(arch)) for arch in known_arches()]
    assert any(holds) and not all(holds)


# ---- scaled WMMA: admitted by the target's MMA catalog ----


def _scaled_atom(scale16: bool):
    """A real scaled-WMMA atom, taken from the catalog that lists it."""
    atom = ArchTarget.from_gfx("gfx1250").mma.op_for_shape(
        family="wmma_scaled",
        a_dtype="fp8",
        b_dtype="fp8",
        c_dtype="fp32",
        scales=("e8m0", "e8m0", 16 if scale16 else 32),
        m=16,
        n=16,
        k=128,
    )
    assert atom is not None
    return atom


def _scaled_kernel(atom, scale16: bool):
    b = IRBuilder("scaled_wmma")
    matrix = b.param("matrix", PtrType(I32, "global"), readonly=True)
    accum = b.param("accum", PtrType(F32, "global"))
    scale_ty = I64 if scale16 else I32
    scales = b.param("scales", PtrType(scale_ty, "global"), readonly=True)
    lane = b.thread_id_x()
    lo = b.global_load_vN(matrix, lane, I32, 8)
    hi = b.global_load_vN(matrix, b.add(lane, b.const_i32(8)), I32, 8)
    fragment = b.vec_concat(lo, hi)
    c = b.global_load_vN(accum, lane, F32, 8)
    scale = b.global_load(scales, lane, scale_ty)
    d = b.mma(atom, fragment, fragment, c, scale, scale)
    b.global_store(accum, lane, b.vec_extract(d, 0))
    b.ret()
    return b.kernel


@pytest.mark.parametrize("arch", known_arches())
@pytest.mark.parametrize("scale16", [False, True])
def test_scaled_wmma_follows_the_target_catalog(scale16, arch):
    atom = _scaled_atom(scale16)
    kernel = _scaled_kernel(atom, scale16)
    if ArchTarget.from_gfx(arch).mma.by_op_id(atom.op_id) is not None:
        lower_with_both_engines(kernel, arch, "llvm23")
    else:
        assert_both_engines_reject(
            kernel,
            arch,
            NotImplementedError,
            f"tile.{atom.op_id} is not available on {arch}: "
            "its MMA catalog does not list it",
            "llvm23",
        )


@pytest.mark.parametrize("arch", known_arches())
@pytest.mark.parametrize("scale16", [False, True])
def test_scaled_wmma_hip_follows_the_target_catalog(scale16, arch):
    atom = _scaled_atom(scale16)
    kernel = _scaled_kernel(atom, scale16)
    target = ArchTarget.from_gfx(arch)
    if target.mma.by_op_id(atom.op_id) is not None:
        lower_hip_with_both_engines(kernel, arch)
    else:
        assert_both_engines_reject_hip(
            kernel,
            arch,
            NotImplementedError,
            f"WMMA op {atom.op_id!r} is not available on {arch} "
            "(WMMA is an RDNA/gfx11 instruction; this is a "
            f"{target.family.upper()} target). The MFMA atoms are the "
            "matrix path on CDNA.",
        )
