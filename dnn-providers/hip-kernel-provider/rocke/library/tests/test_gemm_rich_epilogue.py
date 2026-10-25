# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-side checks of the gfx950 rich GEMM epilogues
(``kernels.gfx950.gemm_rich_epilogue``) and their ``spec._mono`` hook.

Pure codegen (no GPU): knob validation, the hook's kernel params, that an
absent hook leaves the universal GEMM body byte-identical, that the hook
replaces the body's own tile store, and that each knob changes the emission.
"""

from __future__ import annotations

import dataclasses

import pytest
from kernels.gfx950.gemm_rich_epilogue import (
    BODIES,
    EpiHooks,
    EpiKnobs,
    RichEpilogueSpec,
    body_spec,
    build_rich_epilogue,
    rich_epilogue_gemm_spec,
)
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.instances.common.gemm_universal import build_universal_gemm, is_valid_spec


def build_epi(knobs, dtype="f16", body="HBR"):
    """(KernelDef, universal GEMM spec with the hook, hook) of one rich epilogue."""
    spec = RichEpilogueSpec(knobs=knobs, dtype=dtype, body=body)
    gs = rich_epilogue_gemm_spec(spec)
    return build_rich_epilogue(spec, arch="gfx950"), gs, gs._mono


def _ir(kd):
    return lower_kernel_to_llvm(kd, arch="gfx950").replace(kd.name, "KNAME")


@pytest.mark.parametrize("body", sorted(BODIES))
def test_bodies_are_valid_and_unhooked_body_is_unchanged(body):
    sp = body_spec("bf16", body=body)
    ok, why = is_valid_spec(sp, arch="gfx950")
    assert ok, why
    # without the hook the side channel is absent: the same body as a plain spec
    a = build_universal_gemm(sp, "gfx950")
    b = build_universal_gemm(dataclasses.replace(sp), "gfx950")
    assert _ir(a) == _ir(b)
    assert [p.name for p in a.params] == ["A", "B", "C", "M", "N", "K"]


@pytest.mark.parametrize(
    "knobs,params",
    [
        (EpiKnobs(), ["alpha", "ldd"]),
        (EpiKnobs(beta="same"), ["alpha", "ldd", "Cin", "beta", "ldc"]),
        (EpiKnobs(bias="col", act="silu"), ["alpha", "ldd", "Bias"]),
        (EpiKnobs(gate=True, resid=True), ["alpha", "ldd", "G", "R"]),
        (EpiKnobs(out="fp8"), ["alpha", "ldd", "Q", "scale"]),
        (EpiKnobs(out="fp8b", qblock=32), ["alpha", "ldd", "Q", "S"]),
        (EpiKnobs(amax="row", out="none"), ["alpha", "ldd", "Amax"]),
        (
            EpiKnobs(rowsum=True, rowsq=True, colsum=True, zero_next=True),
            ["alpha", "ldd", "RowSum", "RowSq", "ColSum", "slot"],
        ),
    ],
)
def test_hook_params(knobs, params):
    kd, sp, hooks = build_epi(knobs, "f16")
    assert [p.name for p in kd.params] == ["A", "B", "C", "M", "N", "K"] + params
    assert sp._mono is hooks
    assert kd.name.startswith("gemm_rich_epi_")


@pytest.mark.parametrize(
    "knobs",
    [
        EpiKnobs(swiglu=True, act="relu"),
        EpiKnobs(swiglu=True, beta="same"),
        EpiKnobs(out="fp8b", qblock=16),
        EpiKnobs(out="fp8", swiglu=True),
    ],
)
def test_invalid_knobs_rejected(knobs):
    with pytest.raises(ValueError):
        EpiHooks(knobs)


def test_hook_replaces_the_tile_store_and_knobs_change_emission():
    base_kd = build_universal_gemm(body_spec("f16"), "gfx950")
    seen = {_ir(base_kd)}
    for kn in (
        EpiKnobs(),
        EpiKnobs(store_nt=False),
        EpiKnobs(bias="col"),
        EpiKnobs(bias="row"),
        EpiKnobs(act="gelu"),
        EpiKnobs(act="gelu", fast_math=False),
        EpiKnobs(beta="t"),
        EpiKnobs(beta="t", ct_vec=False),
        EpiKnobs(rowsum=True),
        EpiKnobs(rowsum=True, red_scatter=False),
        EpiKnobs(beta="same", aux_pipe=1),
        EpiKnobs(swiglu=True),
        EpiKnobs(out="fp8b", qblock=64),
    ):
        kd, _sp, _h = build_epi(kn, "f16")
        ir = _ir(kd)
        assert ir not in seen, kn.tag()
        seen.add(ir)


def test_builder_contract_and_arch_guard():
    spec = RichEpilogueSpec(knobs=EpiKnobs(bias="col"), dtype="bf16")
    kd = build_rich_epilogue(spec, arch="gfx950")
    assert kd.name.startswith(spec.kernel_name())
    assert RichEpilogueSpec().kernel_name() != spec.kernel_name()
    with pytest.raises(ValueError, match="arch"):
        build_rich_epilogue(spec, arch="gfx942")


def test_reductions_use_device_atomics():
    kd, _sp, _h = build_epi(EpiKnobs(amax="row", rowsum=True, out="none"), "bf16")
    ir = _ir(kd)
    assert "global_atomic_umax" in ir
    assert "atomicrmw fadd" in ir
