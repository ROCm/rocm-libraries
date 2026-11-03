# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The universal GEMM test helper, checked on the host.

A typo in the engine name must fail with a clear error instead of silently
picking the C++ engine. The kernel arguments the launcher packs must be the
parameters the builder declares, in order, for every argument layout, and the
default C buffer must be in the kernel's storage dtype.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest
from rocke.core.ir import I32, PtrType
from rocke.instances.common.gemm_universal import (
    DataSpec,
    TileSpec,
    TraitSpec,
    UniversalGemmSpec,
    build_universal_gemm,
)
from rocke.tests.universal_gemm import (
    lower_universal_gemm,
    universal_gemm_arguments,
    universal_gemm_nan_output,
)


def _spec(dtype: str = "fp64", *, batched: bool = False, **trait) -> UniversalGemmSpec:
    return UniversalGemmSpec(
        name="argument_layout",
        tile=TileSpec(32, 32, 8, 1, 1, 1, 16, 16, 4),
        trait=TraitSpec(pipeline="mem", epilogue="default", **trait),
        data=DataSpec(dtype, dtype, dtype, "fp64" if dtype == "fp64" else "fp32"),
        batched=batched,
    )


def test_rejects_an_unknown_engine():
    with pytest.raises(ValueError) as error:
        lower_universal_gemm(_spec(), "c++", arch="gfx942")
    assert str(error.value) == "engine must be 'python' or 'cpp', got 'c++'"


def _declared_layout(spec: UniversalGemmSpec) -> tuple[str, list[str]]:
    """The struct layout and the names of the parameters the builder declares:
    an 8-byte pointer or a 4-byte i32 each, in order."""
    kernel = build_universal_gemm(spec, arch="gfx942")
    layout, names = "", []
    for param in kernel.params:
        assert isinstance(param.type, PtrType) or param.type == I32, param
        layout += "Q" if isinstance(param.type, PtrType) else "i"
        names.append(param.name)
    return layout, names


@pytest.mark.parametrize(
    ("batched", "trait", "pointers", "values"),
    [
        (False, {}, (1, 2, 3), {}),
        (
            True,
            {},
            (1, 2, 3),
            {"stride_a": 5 * 7, "stride_b": 6 * 7, "stride_c": 5 * 6},
        ),
        (
            True,
            {"active_tile_skip": True},
            (1, 2, 3, 4),
            {
                "stride_a": 5 * 7,
                "stride_b": 6 * 7,
                "stride_c": 5 * 6,
                "SortedTokenIds": 4,
                "slot_size": 5,
            },
        ),
    ],
    ids=["plain", "batched", "batched_active_tile_skip"],
)
def test_the_arguments_are_the_parameters_the_builder_declares(
    batched, trait, pointers, values
):
    """A batched kernel without active-tile skip still takes the three batch
    strides; packing only A, B, C, M, N and K for it left them unset."""
    spec = _spec(batched=batched, **trait)
    layout, names = _declared_layout(spec)
    packed = universal_gemm_arguments(spec, pointers, 5, 6, 7)
    assert len(packed) == struct.calcsize(layout)
    expected = {"A": 1, "B": 2, "C": 3, "M": 5, "N": 6, "K": 7, **values}
    assert dict(zip(names, struct.unpack(layout, packed))) == expected


def test_sorted_ids_are_passed_exactly_to_a_kernel_that_reads_them():
    with pytest.raises(ValueError) as error:
        universal_gemm_arguments(_spec(batched=True), (1, 2, 3, 4), 5, 6, 7)
    assert str(error.value).endswith(
        "does not read sorted token ids; pass sorted_ids only for a batched kernel "
        "with active_tile_skip"
    )
    with pytest.raises(ValueError) as error:
        universal_gemm_arguments(
            _spec(batched=True, active_tile_skip=True), (1, 2, 3), 5, 6, 7
        )
    assert str(error.value).endswith(
        "skips inactive tiles and reads sorted token ids; pass sorted_ids"
    )


@pytest.mark.parametrize(
    ("dtype", "host_dtype", "bits"),
    [
        ("fp64", np.float64, None),
        ("fp32", np.float32, None),
        ("fp16", np.float16, None),
        ("bf16", np.uint16, 0x7FC0),
    ],
)
def test_the_default_output_is_nan_in_the_storage_dtype(dtype, host_dtype, bits):
    """A two-byte GEMM writes two-byte elements, so C defaults to NaN in that
    width; bf16 is its uint16 encoding."""
    output = universal_gemm_nan_output(_spec(dtype), (2, 3))
    assert output.dtype == host_dtype and output.shape == (2, 3)
    if bits is None:
        assert np.isnan(output).all()
    else:
        assert (output == bits).all()


def test_a_split_k_output_is_refused():
    with pytest.raises(ValueError, match=r"^split-K GEMM \(split_k=2\) accumulates"):
        universal_gemm_nan_output(_spec("fp16", split_k=2), (2, 3))
