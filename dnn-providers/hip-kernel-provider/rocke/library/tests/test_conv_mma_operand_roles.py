# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CPU emission-role checks, not qualification of asymmetric MMA pipelines.

Synthetic source widths distinguish input initializers from output partials.
These kernels are inspected before lowering: each physical instruction still
requires its fixed src2/dst ABI, and these cases do not claim otherwise.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace

import pytest

from rocke.core.ir import F32, IRBuilder, Op, Region, Value, VectorType
from rocke.core.ir_serialize import serialize

from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common import conv_implicit_gemm as fwd
from kernels.common import conv_implicit_gemm_dgrad as dgrad
from kernels.common import conv_implicit_gemm_wgrad as wgrad


_BUILDERS = {
    "fwd": (
        fwd,
        fwd.ImplicitGemmConvSpec,
        fwd.build_implicit_gemm_conv,
        "_resolve_conv_op",
    ),
    "wgrad": (
        wgrad,
        wgrad.WgradConvSpec,
        wgrad.build_implicit_gemm_conv_wgrad,
        "_resolve_wgrad_op",
    ),
    "dgrad": (
        dgrad,
        dgrad.DgradConvSpec,
        dgrad.build_implicit_gemm_conv_dgrad,
        "_resolve_dgrad_op",
    ),
}
_ARCHES = ("gfx90a", "gfx942", "gfx950", "gfx1151", "gfx1201", "gfx1250")


def _walk(region: Region) -> Iterator[Op]:
    for op in region.ops:
        yield op
        for child in op.regions:
            yield from _walk(child)


def _spec(
    direction: str, arch: str, **kwargs
) -> fwd.ImplicitGemmConvSpec | wgrad.WgradConvSpec | dgrad.DgradConvSpec:
    _, spec_type, _, _ = _BUILDERS[direction]
    wave_size = 32 if arch in ("gfx1151", "gfx1201", "gfx1250") else 64
    return spec_type(
        problem=ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1),
        data=ConvDataSpec(dtype_a="fp16", dtype_b="fp16", dtype_d="fp16"),
        tile_m=64,
        tile_n=64,
        tile_k=32,
        warp_m=2,
        warp_n=2,
        warp_tile_m=16,
        warp_tile_n=16,
        warp_tile_k=32 if arch == "gfx1250" else 16,
        wave_size=wave_size,
        vector_size_c=1,
        split_k=1,
        **kwargs,
    )


def _source_initializers(ops: list[Op]) -> list[Op]:
    definitions = {value.name: op for op in ops for value in op.results}
    initializers = {}
    for op in ops:
        if op.name != "scf.for":
            continue
        for value in op.operands[3:]:
            definition = definitions.get(value.name)
            if (
                definition is not None
                and definition.name == "arith.constant_vec"
                and value.type.elem == F32
            ):
                initializers[value.name] = definition
    return list(initializers.values())


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("direction", _BUILDERS)
@pytest.mark.parametrize("extra_source_slots", [1, 3])
def test_mma_initializers_use_source_width(
    monkeypatch: pytest.MonkeyPatch, arch: str, direction: str, extra_source_slots: int
) -> None:
    module, _, build, resolver_name = _BUILDERS[direction]
    spec = _spec(direction, arch)
    atom = getattr(module, resolver_name)(spec, arch)
    source_width = atom.srcs[2].frag_len + extra_source_slots
    custom = replace(
        atom, srcs=(*atom.srcs[:2], replace(atom.srcs[2], frag_len=source_width))
    )
    monkeypatch.setattr(module, resolver_name, lambda spec, arch: custom)
    ops = list(_walk(build(spec, arch=arch).body))
    initializers = _source_initializers(ops)
    assert initializers
    assert all(op.result.type == VectorType(F32, source_width) for op in initializers)
    mmas = [op for op in ops if op.name == "tile.mma"]
    assert mmas
    assert all(op.result.type == VectorType(F32, atom.dst.frag_len) for op in mmas)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize("mode", ["dp_sk", "persistent"])
def test_streamk_tree_neutral_partials_use_result_width(
    monkeypatch: pytest.MonkeyPatch, arch: str, mode: str
) -> None:
    spec = _spec("wgrad", arch, streamk=mode, streamk_reduction="tree", streamk_ctas=6)
    atom = wgrad._resolve_wgrad_op(spec, arch)
    source_width = atom.srcs[2].frag_len + 1
    custom = replace(
        atom, srcs=(*atom.srcs[:2], replace(atom.srcs[2], frag_len=source_width))
    )
    monkeypatch.setattr(wgrad, "_resolve_wgrad_op", lambda spec, arch: custom)
    zero_vec_f32 = IRBuilder.zero_vec_f32
    source_initializers = []

    def compatible_loop_seed(builder: IRBuilder, width: int) -> Value:
        # The output reduction needs a type-compatible loop result. Observe the
        # synthetic input request, then replace only that seed with a physical
        # ABI-sized placeholder; every output zero still uses the real builder.
        if not source_initializers:
            source_initializers.append(width)
            return zero_vec_f32(builder, atom.dst.frag_len)
        return zero_vec_f32(builder, width)

    monkeypatch.setattr(IRBuilder, "zero_vec_f32", compatible_loop_seed)
    ops = list(_walk(wgrad.build_implicit_gemm_conv_wgrad(spec, arch=arch).body))
    assert source_initializers == [source_width]
    definitions = {value.name: op for op in ops for value in op.results}
    neutral_partials = [
        definitions[value.name]
        for op in ops
        if op.name == "arith.select"
        for value in op.operands[1:]
        if value.name in definitions
        and definitions[value.name].name == "arith.constant_vec"
        and isinstance(value.type, VectorType)
        and value.type.elem == F32
    ]
    assert neutral_partials
    assert all(
        op.result.type == VectorType(F32, atom.dst.frag_len) for op in neutral_partials
    )
    # These selected partials are added to instruction results, not fed to MMA.
    selected = {
        op.result.name
        for op in ops
        if op.name == "arith.select"
        and isinstance(op.result.type, VectorType)
        and op.result.type.elem == F32
    }
    assert any(
        op.name == "vector.add" and any(v.name in selected for v in op.operands)
        for op in ops
    )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "lds_k_outer,async_dma", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("replicas", [1, 3])
def test_wmma_two_stage_store_uses_result_layout(
    monkeypatch: pytest.MonkeyPatch,
    dtype: str,
    lds_k_outer: bool,
    async_dma: bool,
    replicas: int,
) -> None:
    """A source map is not needed to store already-computed MMA results.

    Keep the physical dtypes, widths and destination map unchanged; remove only
    src2's coordinate map. The full builder must still emit the same kernel,
    including its guarded workspace atomics. This does not qualify an
    asymmetric hardware instruction or change its recurrence contract.
    """
    spec = replace(
        _spec("wgrad", "gfx1250"),
        data=ConvDataSpec(dtype_a=dtype, dtype_b=dtype, dtype_d=dtype),
        split_k=2,
        two_stage=True,
        lds_k_outer=lds_k_outer,
        async_dma=async_dma,
        ws_replicas=replicas,
    )
    expected = wgrad.build_implicit_gemm_conv_wgrad(spec, arch="gfx1250")
    atom = wgrad._resolve_wgrad_op(spec, "gfx1250")
    custom = replace(atom, srcs=(*atom.srcs[:2], replace(atom.srcs[2], layout=None)))
    monkeypatch.setattr(wgrad, "_resolve_wgrad_op", lambda spec, arch: custom)
    actual = wgrad.build_implicit_gemm_conv_wgrad(spec, arch="gfx1250")
    assert any(op.name == "memref.global_atomic_add" for op in _walk(actual.body))
    assert serialize(actual) == serialize(expected)
