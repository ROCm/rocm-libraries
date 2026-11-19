# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""GPU numeric verification of ``make_a_descriptor_addr64`` /
``make_d_descriptor_addr64`` (``kernels/common/_conv_implicit_gemm_common.py``):
the addr64 forward-conv descriptors that exclude the ``n``/``p_*_stride_n``
term entirely (folded into the i64 tile origin instead, per
``library/kernels/conv/docs/large_tensor_addr64_plan.md`` section 3).

Cross-checks ``.offset(b, m=d, k=...)`` against a plain-Python reference that
recomputes the same conv addressing formula (y/x/c channel decode, the
padded hi/wi embed, the final element offset) independently -- proving the
descriptor is correct, not just that it builds and lowers.

Requires a ROCm GPU; skips otherwise.
"""

from __future__ import annotations

import ctypes
import struct

import pytest

from kernels.common._conv_implicit_gemm_common import (
    make_a_descriptor_addr64,
    make_d_descriptor_addr64,
)
from rocke.core.ir import I32, IRBuilder, PtrType
from rocke.helpers.compile import compile_kernel
from rocke.helpers.magic_division import calculate_magic_numbers
from rocke.runtime.hip_module import Runtime
from rocke.tests.gpu import probe_device_arch

pytestmark = pytest.mark.skipif(
    probe_device_arch()[0] is None, reason=probe_device_arch()[1]
)

_ARCH = probe_device_arch()[0] or "gfx950"

# A fixed small conv shape, shared by every case.
_SHAPE = dict(
    Hi=16,
    Wi=16,
    Ho=14,
    Wo=14,
    sH=1,
    sW=1,
    pH=1,
    pW=1,
    dH=1,
    dW=1,
    cpg=8,
    Y=3,
    X=3,
)
_A_STRIDE_HI = _SHAPE["Wi"] * _SHAPE["cpg"]
_A_STRIDE_WI = _SHAPE["cpg"]
_D_STRIDE_HO = _SHAPE["Wo"] * _SHAPE["cpg"]
_D_STRIDE_WO = _SHAPE["cpg"]

_MAGIC_X_MULT, _MAGIC_X_SHIFT = calculate_magic_numbers(_SHAPE["X"])
_MAGIC_CPG_MULT, _MAGIC_CPG_SHIFT = calculate_magic_numbers(_SHAPE["cpg"])


def _param_dict(b: IRBuilder):
    names = [
        "p_pH",
        "p_pW",
        "p_sH",
        "p_sW",
        "p_dH",
        "p_dW",
        "p_Hi",
        "p_Wi",
        "p_Ho",
        "p_Wo",
        "p_cpg",
        "p_Y",
        "p_X",
        "p_A_stride_hi",
        "p_A_stride_wi",
        "p_D_stride_ho",
        "p_D_stride_wo",
        "p_magic_k_X_mult",
        "p_magic_k_X_shift",
        "p_magic_k_cpg_mult",
        "p_magic_k_cpg_shift",
    ]
    return {n: b.param(n, I32) for n in names}


def _build_probe_kernel():
    """Writes ``[a_offset, a_valid, d_offset, d_valid]`` (i32) to ``out``."""
    b = IRBuilder("addr64_descriptor_probe")
    out = b.param("out", PtrType(I32, "global"))
    params = _param_dict(b)
    n0 = b.param("n0", I32)
    ho0 = b.param("ho0", I32)
    wo0 = b.param("wo0", I32)
    d = b.param("d", I32)
    k = b.param("k", I32)

    a_desc = make_a_descriptor_addr64(b, params, origin_coords=[n0, ho0, wo0])
    a_off, a_valid = a_desc.offset(b, m=d, k=k)
    d_desc = make_d_descriptor_addr64(b, params, origin_coords=[n0, ho0, wo0])
    d_off, d_valid = d_desc.offset(b, m=d, k_out=k)

    def _valid_to_i32(v):
        return b.const_i32(1) if v is None else b.select(v, b.const_i32(1), b.const_i32(0))

    b.global_store(out, b.const_i32(0), a_off)
    b.global_store(out, b.const_i32(1), _valid_to_i32(a_valid))
    b.global_store(out, b.const_i32(2), d_off)
    b.global_store(out, b.const_i32(3), _valid_to_i32(d_valid))
    b.ret()
    return b.kernel


def _launch_probe(n0: int, ho0: int, wo0: int, d: int, k: int):
    kernel = _build_probe_kernel()
    artifact = compile_kernel(kernel, arch=_ARCH)
    rt = Runtime()
    module = rt.load_module(artifact.hsaco)
    fn = module.get_function(artifact.kernel_name)

    out_ptr = rt.alloc(16)
    rt.memset(out_ptr, 0, 16)

    s = _SHAPE
    scalar_values = [
        s["pH"],
        s["pW"],
        s["sH"],
        s["sW"],
        s["dH"],
        s["dW"],
        s["Hi"],
        s["Wi"],
        s["Ho"],
        s["Wo"],
        s["cpg"],
        s["Y"],
        s["X"],
        _A_STRIDE_HI,
        _A_STRIDE_WI,
        _D_STRIDE_HO,
        _D_STRIDE_WO,
        _MAGIC_X_MULT,
        _MAGIC_X_SHIFT,
        _MAGIC_CPG_MULT,
        _MAGIC_CPG_SHIFT,
        n0,
        ho0,
        wo0,
        d,
        k,
    ]
    args = struct.pack("<Q" + "i" * len(scalar_values), out_ptr, *scalar_values)
    rt.launch_blocking(fn, (1, 1, 1), (1, 1, 1), args)

    out = (ctypes.c_int32 * 4)()
    rt.memcpy_d2h(out, out_ptr, 16)
    rt.free(out_ptr)
    return tuple(out)


def _python_reference(n0, ho0, wo0, d, k):
    """Independent re-derivation of the same conv addressing formula."""
    s = _SHAPE
    m_abs = (n0 * s["Ho"] + ho0) * s["Wo"] + wo0 + d
    wo = m_abs % s["Wo"]
    ho = (m_abs // s["Wo"]) % s["Ho"]

    c = k % s["cpg"]
    tmp = k // s["cpg"]
    x = tmp % s["X"]
    y = tmp // s["X"]

    hi = ho * s["sH"] + y * s["dH"] - s["pH"]
    wi = wo * s["sW"] + x * s["dW"] - s["pW"]
    a_valid = (0 <= hi < s["Hi"]) and (0 <= wi < s["Wi"]) and (0 <= y < s["Y"]) and (
        0 <= x < s["X"]
    )
    a_off = hi * _A_STRIDE_HI + wi * _A_STRIDE_WI + c

    d_off = ho * _D_STRIDE_HO + wo * _D_STRIDE_WO + k
    return a_off, int(a_valid), d_off, 1


def test_addr64_descriptors_exclude_n_from_their_coord_chain():
    """Regression guard for the whole point of these descriptors: "n" (and
    its stride) must never reappear in the base coord chain, since that is
    exactly the i32-unsafe product (n * stride_n) the i64 tile origin
    absorbs instead. A CPU-only, no-GPU check."""
    b = IRBuilder("addr64_descriptor_shape_check")
    params = _param_dict(b)
    n0 = b.param("n0", I32)
    ho0 = b.param("ho0", I32)
    wo0 = b.param("wo0", I32)
    a_desc = make_a_descriptor_addr64(b, params, origin_coords=[n0, ho0, wo0])
    d_desc = make_d_descriptor_addr64(b, params, origin_coords=[n0, ho0, wo0])
    assert "n" not in a_desc.base_names
    assert "n" not in d_desc.base_names
    assert a_desc.base_names == ("hi", "wi", "c")
    assert d_desc.base_names == ("ho", "wo", "k_out")


@pytest.mark.parametrize(
    "n0,ho0,wo0,d,k",
    [
        (0, 0, 0, 0, 0),
        (0, 5, 5, 0, 10),
        (0, 0, 13, 1, 5),  # carries wo -> ho
        (100_000_000, 7, 7, 3, 12),  # the primary addr64 scenario: huge n0
        (0, 13, 13, 20, 0),  # padding edge: hi/wi walk outside bounds
    ],
)
def test_addr64_descriptor_matches_python_reference(n0, ho0, wo0, d, k):
    got = _launch_probe(n0, ho0, wo0, d, k)
    want = _python_reference(n0, ho0, wo0, d, k)
    assert got == want
