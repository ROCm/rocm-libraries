# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The HIP source of the packed global atomics, the same in both engines.

``__builtin_amdgcn_global_atomic_fadd_v2bf16`` returns short vectors, which do
not convert to ``bf16x2`` implicitly, so the result is bit-cast;
``__builtin_amdgcn_global_atomic_fadd_v2f16`` returns ``_Float16`` vectors, which
``half2`` accepts, so its result is assigned as it is.
"""

from __future__ import annotations

import re

import pytest
from rocke.core.ir import BF16, F16, I32, IRBuilder, PtrType
from rocke.tests.engines import ENGINES, lower_hip_with_engine
from rocke.tests.hipcc import assert_hipcc_accepts


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("elem", "result"),
    [
        (
            BF16,
            (
                r"bf16x2 \w+ = __builtin_bit_cast\(bf16x2, "
                r"__builtin_amdgcn_global_atomic_fadd_v2bf16\(p \+ idx, \w+\)\);"
            ),
        ),
        (
            F16,
            r"half2 \w+ = __builtin_amdgcn_global_atomic_fadd_v2f16\(p \+ idx, \w+\);",
        ),
    ],
    ids=["bf16", "f16"],
)
@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_hip_assigns_the_result_as_the_operand_type(
    arch, elem, result, engine, tmp_path
):
    b = IRBuilder(f"pk_{elem.name}_atomic")
    p = b.param("p", PtrType(elem, "global"))
    idx = b.param("idx", I32)
    val = b.global_load_vN(p, idx, elem, 2, align=4)
    getattr(b, f"global_atomic_add_pk_{elem.name}")(p, idx, val)
    b.ret()
    source = lower_hip_with_engine(b.kernel, arch, engine=engine)
    assert re.search(result, source)
    assert_hipcc_accepts(source, arch, tmp_path)
