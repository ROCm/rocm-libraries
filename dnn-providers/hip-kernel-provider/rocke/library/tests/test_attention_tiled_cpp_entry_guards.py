# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ tiled-attention entries refuse what the Python builders refuse, with
the same reason, instead of crashing.

* A zero or non-dividing head pair: the 2D ``_new`` entries used to fail at
  naming before initialising the builder, and the caller then freed that
  uninitialised builder (signal 11).
* fastKV register-P on gfx942: the body is gfx950-only, so the C++ builder must
  refuse it as Python does rather than emit gfx950 MFMA for gfx942.
"""

from __future__ import annotations

import dataclasses

import pytest
from kernels.gfx942.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx942Spec2D,
)
from kernels.gfx950.attention_tiled_2d import (
    UnifiedAttention2DTiledSpec as Gfx950Spec2D,
)
from kernels.gfx950.attention_tiled_2d_fastkv_regp import (
    build_unified_attention_2d_fastkv_register_p,
    make_fastkv_register_p_spec,
)

rocke_engine = pytest.importorskip("rocke_engine")

_PROBLEM = {
    "head_size": 128,
    "block_size": 16,
    "dtype": "bf16",
    "use_sinks": False,
    "sliding_window": 0,
    "has_softcap": False,
}


@pytest.mark.parametrize(
    "arch, spec_cls",
    [("gfx942", Gfx942Spec2D), ("gfx950", Gfx950Spec2D)],
)
@pytest.mark.parametrize(
    "heads, reason",
    [((32, 0), "must be > 0"), ((32, 6), "must be divisible by num_kv_heads")],
)
def test_2d_entry_refuses_bad_head_counts_like_python(arch, spec_cls, heads, reason):
    hq, hkv = heads
    spec = spec_cls(num_query_heads=hq, num_kv_heads=hkv, **_PROBLEM)
    with pytest.raises(ValueError, match=reason) as py_err:
        spec.kernel_name()
    lower = getattr(rocke_engine, f"{arch}_attention_tiled_2d_lower_llvm")
    with pytest.raises(Exception) as cpp_err:
        lower(dataclasses.asdict(spec), arch)
    assert str(py_err.value) in str(cpp_err.value)


def test_fastkv_register_p_entry_refuses_gfx942_like_python():
    spec = make_fastkv_register_p_spec(
        Gfx950Spec2D(
            num_query_heads=64,
            num_kv_heads=8,
            **{**_PROBLEM, "head_size": 64, "block_size": 32},
        )
    )
    with pytest.raises(ValueError, match="gfx950 kernel") as py_err:
        build_unified_attention_2d_fastkv_register_p(spec, arch="gfx942")
    with pytest.raises(Exception) as cpp_err:
        rocke_engine.gfx950_attention_tiled_2d_fastkv_regp_lower_llvm(
            dataclasses.asdict(spec), "gfx942"
        )
    assert str(py_err.value) in str(cpp_err.value)
