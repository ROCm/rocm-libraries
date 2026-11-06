# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU spec / ABI / IR coverage of ALiBi in the gfx942 and gfx950 dense kernels."""

from __future__ import annotations

import pytest
from kernels.common.attention_dense_spec import attention_dense_cache_key
from kernels.gfx942 import attention_dense as dense942
from kernels.gfx950 import attention_dense as dense950

_ARCHES = {
    "gfx942": (dense942, dense942.Gfx942AttentionDenseSpec),
    "gfx950": (dense950, dense950.Gfx950AttentionDenseSpec),
}


def _spec(arch, **over):
    kw = dict(
        batch=1,
        seqlen_q=512,
        seqlen_kv=512,
        num_query_heads=12,
        num_kv_heads=4,
        head_size=128,
        dtype="bf16",
    )
    kw.update(over)
    return _ARCHES[arch][1](**kw)


def _params(arch, spec):
    kernel = _ARCHES[arch][0].build_attention_dense(spec, arch=arch)
    return [p.name for p in kernel.params], kernel


def _lowered(arch, spec):
    from rocke.helpers.compile import _lower_llvm_via_backend

    kernel = _ARCHES[arch][0].build_attention_dense(spec, arch=arch)
    llvm = _lower_llvm_via_backend(kernel, arch=arch, backend="python", spec=None)
    return llvm.replace(kernel.name, "KERNEL")


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize(
    "over,match",
    [
        ({"use_alibi": True, "use_sinks": True}, "use_sinks"),
        (
            {
                "use_alibi": True,
                "causal_bottom_right": True,
                "seqlen_q": 256,
                "emit_lse": True,
            },
            "emit_lse",
        ),
    ],
)
def test_spec_refusals(arch, over, match):
    with pytest.raises(ValueError, match=match):
        _spec(arch, **over)


def test_gfx950_refuses_persistent():
    with pytest.raises(ValueError, match="persistent"):
        _spec("gfx950", use_alibi=True, persistent=True)


@pytest.mark.parametrize("arch", _ARCHES)
def test_name_token_and_cache_identity(arch):
    off = _spec(arch)
    on = _spec(arch, use_alibi=True)
    assert "_alibi_" in on.kernel_name() + "_"
    assert off.kernel_name() != on.kernel_name()
    assert attention_dense_cache_key(off, arch=arch) != attention_dense_cache_key(
        on, arch=arch
    )


@pytest.mark.parametrize("arch", _ARCHES)
def test_slopes_param_iff_alibi(arch):
    mod = _ARCHES[arch][0]
    off_params, _ = _params(arch, _spec(arch))
    on_spec = _spec(arch, use_alibi=True)
    on_params, _ = _params(arch, on_spec)
    assert "alibi_slopes_ptr" not in off_params
    assert [p for p in on_params if p not in off_params] == ["alibi_slopes_ptr"]
    sig = [a["name"] for a in mod.attention_dense_signature(on_spec)]
    assert sig == on_params


@pytest.mark.parametrize("arch", _ARCHES)
def test_slope_source_is_picked_at_run_time(arch):
    """One binary serves caller and built-in slopes: a null-pointer test selects
    between the loaded slope and the built-in one, without a branch."""
    on = _lowered(arch, _spec(arch, use_alibi=True))
    assert "ptrtoint" in on
    assert "buffer.load" in on


@pytest.mark.parametrize("arch", _ARCHES)
@pytest.mark.parametrize("causal", [True, False])
def test_alibi_changes_the_body(arch, causal):
    off = _lowered(arch, _spec(arch, causal=causal))
    on = _lowered(arch, _spec(arch, causal=causal, use_alibi=True))
    assert on != off
    assert "exp2" in on


@pytest.mark.parametrize("arch", _ARCHES)
def test_lse_shift_only_off_the_top_left_diagonal(arch):
    """Top-left causal anchors at the query row, so the LSE needs no correction;
    full attention anchors at the last key and does."""
    base = dict(use_alibi=True)
    tl_lse = _lowered(arch, _spec(arch, emit_lse=True, **base))
    tl = _lowered(arch, _spec(arch, **base))
    full_lse = _lowered(arch, _spec(arch, causal=False, emit_lse=True, **base))
    full = _lowered(arch, _spec(arch, causal=False, **base))
    # Each LSE variant adds its store; full attention also adds the row shift.
    assert tl_lse.count("fadd") - tl.count("fadd") < full_lse.count(
        "fadd"
    ) - full.count("fadd")
