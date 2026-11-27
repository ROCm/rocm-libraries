# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The optional library binding must expose the same kernel the Python spec builds."""

from dataclasses import asdict

import pytest
from parity.attention_dense_gfx950_emit import (
    _CONFIGS,
    build_attention_dense,
    spec_for_index,
)
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.core.verify import verify
from rocke.tests.lanes import BUILD_VARIANT, import_or_skip_until

native = import_or_skip_until("rocke_library_engine", BUILD_VARIANT)


@pytest.mark.parametrize("index", range(len(_CONFIGS)))
def test_binding_matches_python(index, monkeypatch):
    arch = "gfx950"
    monkeypatch.setenv("ROCKE_BACKEND", "python")
    spec = spec_for_index(index)
    fields = asdict(spec)  # Pass raw optional fields, not their resolved values.
    kernel = build_attention_dense(spec, arch=arch)
    assert native.gfx950_attention_dense_serialize_ir(fields, arch) == serialize(kernel)
    assert native.gfx950_attention_dense_lower_llvm(
        fields, arch
    ) == lower_kernel_to_llvm(kernel, arch=arch)
    assert native.gfx950_attention_dense_verify(fields, arch) == [
        str(d) for d in verify(kernel)
    ]


@pytest.mark.parametrize(
    ("key", "value", "reason"),
    [
        ("batch", True, "batch must be an int"),
        ("pv_sched_fence", 1, "pv_sched_fence must be a bool"),
        ("lazy_rescale_threshold", 8, "lazy_rescale_threshold must be a float"),
        ("pv_loop_order", False, "pv_loop_order must be a str"),
        ("unknown_knob", 1, "unexpected field"),
    ],
    ids=("bool-as-int", "int-as-bool", "int-as-float", "bool-as-str", "unknown-field"),
)
def test_binding_refuses_invalid_fields(key, value, reason):
    fields = asdict(spec_for_index(0))
    fields[key] = value
    with pytest.raises(TypeError, match=reason):
        native.gfx950_attention_dense_serialize_ir(fields, "gfx950")
