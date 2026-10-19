# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LDS alignment claims of the backward main kernel with the slices dK / dV sink.

With ``dkv_mode="atomic", dkv_sink="slices"`` the d64 bf16 main kernel reads
its dK / dV partials from LDS as ``<8 x float>`` (32 bytes). Every smem
allocation lives in one 16-byte aligned LDS pool, so no access can truthfully
claim more than 16 bytes: the kernel must lower, and each such load must claim
the pool alignment rather than its 32-byte width. The planner forces this sink
for deterministic requests, so a refusal here would break real dispatches.
"""

from __future__ import annotations

import re

import pytest
from rocke.core.ir_serialize import serialize
from rocke.core.lower_llvm import LLVM_FLAVOR_LLVM20
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python as lower

from kernels.common.attention_bwd import (
    AttnBwdSpec,
    build_attn_bwd_main,
    resource_refusal,
)

_ARCHS = ("gfx942", "gfx950")

_POOL_ALIGN_RE = re.compile(
    r"^@smem_pool\.\S+ = internal unnamed_addr addrspace\(3\) global "
    r"\[\d+ x i8\] poison, align (\d+)$",
    re.MULTILINE,
)
_LDS_ALIGN_RE = re.compile(r"ptr addrspace\(3\) \S+, align (\d+)")
_WIDE_LOAD_RE = re.compile(
    r"load <8 x float>, ptr addrspace\(3\) \S+, align (\d+)$", re.MULTILINE
)


def _slices_kernel(arch):
    spec = AttnBwdSpec(head_size=64, dtype="bf16", dkv_mode="atomic", dkv_sink="slices")
    # The resource gate accepts this configuration; the test must not depend
    # on bypassing it.
    assert resource_refusal(spec, arch) is None
    return build_attn_bwd_main(spec, arch=arch)


@pytest.mark.parametrize("arch", _ARCHS)
def test_slices_sink_lds_claims_fit_the_pool(arch):
    ll = lower(_slices_kernel(arch), arch=arch, llvm_flavor=LLVM_FLAVOR_LLVM20)
    (pool_align,) = (int(a) for a in _POOL_ALIGN_RE.findall(ll))
    claims = [int(a) for a in _LDS_ALIGN_RE.findall(ll)]
    assert claims and max(claims) <= pool_align
    wide = _WIDE_LOAD_RE.findall(ll)
    assert wide and set(wide) == {str(pool_align)}


@pytest.mark.parametrize("arch", _ARCHS)
def test_slices_sink_both_engines_emit_the_same_ir(arch):
    from rocke.core.backend import BackendError, _lower_via_cpp_engine

    kernel = _slices_kernel(arch)
    try:
        cpp = _lower_via_cpp_engine(serialize(kernel), arch, LLVM_FLAVOR_LLVM20)
    except BackendError as e:
        pytest.skip(f"C++ engine not importable: {str(e)[:200]}")
    assert cpp == lower(kernel, arch=arch, llvm_flavor=LLVM_FLAVOR_LLVM20)
