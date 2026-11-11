# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""``validate()`` and ``is_valid_*_spec`` apply one TF32 gate per direction.

Each implicit-GEMM direction routes both through one helper: the forward
kernel's ``_fwd_tf32_reason``, dgrad's ``_mma_dtype_conflict`` and wgrad's
``_wgrad_tf32_reason``. ``validate()`` has no arch and calls it arch-free;
``is_valid_*_spec`` calls it with the arch. So per arch (gfx942 native XF32,
gfx950 BF16x3, gfx1151 WMMA):

* ``is_valid_*_spec`` refuses a TF32 spec with exactly the gate's reason on
  that arch, and admits it when the gate does;
* ``validate()`` raises exactly the arch-free reason, and only for a spec that
  every arch refuses;
* the arch-independent rules (wave size, an atom that is TF32 on no arch, the
  LDS row stride, the dtype spelling) are caught by ``validate()`` too, not
  only by ``is_valid_*_spec``.

CPU only.
"""

from __future__ import annotations

import pytest
from kernels.common._conv_implicit_gemm_common import ConvDataSpec, ConvProblem
from kernels.common.conv_implicit_gemm import (
    ImplicitGemmConvSpec,
    _fwd_tf32_reason,
    is_valid_spec,
)
from kernels.common.conv_implicit_gemm_dgrad import (
    DgradConvSpec,
    _mma_dtype_conflict,
    is_valid_dgrad_spec,
)
from kernels.common.conv_implicit_gemm_wgrad import (
    WgradConvSpec,
    _wgrad_tf32_reason,
    is_valid_wgrad_spec,
)

_ARCHES = ("gfx942", "gfx950", "gfx1151")
_TF32 = ConvDataSpec("fp32", "fp32", "fp32", mma_dtype="tf32")
_PROBLEM = ConvProblem(N=2, Hi=14, Wi=14, C=64, K=64, Y=3, X=3, pH=1, pW=1)


def _geom(atom, **kw):
    base = {
        "problem": _PROBLEM,
        "data": _TF32,
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 32,
        "warp_m": 2,
        "warp_n": 2,
        "warp_tile_m": atom[0],
        "warp_tile_n": atom[1],
        "warp_tile_k": atom[2],
    }
    base.update(kw)
    return base


def _fwd(atom, **kw):
    return ImplicitGemmConvSpec(**_geom(atom, epilogue="cshuffle", **kw))


def _dgrad(atom, **kw):
    return DgradConvSpec(**_geom(atom, **kw))


def _wgrad(atom, **kw):
    return WgradConvSpec(**_geom(atom, **kw))


_DIRECTIONS = {
    "fwd": (_fwd, _fwd_tf32_reason, is_valid_spec),
    "dgrad": (_dgrad, _mma_dtype_conflict, is_valid_dgrad_spec),
    "wgrad": (_wgrad, _wgrad_tf32_reason, is_valid_wgrad_spec),
}

# (case, atom, spec overrides, arches that admit it). An empty tuple means
# no arch can build it, so validate() must refuse it as well.
_CASES = (
    ("gfx950_atom", (16, 16, 32), {}, ("gfx950",)),
    ("gfx950_atom_32", (32, 32, 16), {}, ("gfx950",)),
    ("gfx942_atom", (16, 16, 8), {}, ("gfx942",)),
    ("gfx942_atom_32", (32, 32, 4), {}, ("gfx942",)),
    ("wave32", (16, 16, 16), {"wave_size": 32}, ()),
    ("wave32_gfx942_atom", (16, 16, 8), {"wave_size": 32}, ()),
    ("no_tf32_atom", (16, 16, 16), {}, ()),
    ("fp32_atom", (16, 16, 4), {}, ()),
    # 8-wide gfx950 fragment: tile_k + pad must be a multiple of 4.
    ("misaligned_gfx950", (16, 16, 32), {"lds_k_pad": 2}, ()),
    # 2-wide gfx942 fragment: a multiple of 2.
    ("misaligned_gfx942", (16, 16, 8), {"lds_k_pad": 1}, ()),
    ("aligned_gfx942", (16, 16, 8), {"lds_k_pad": 2}, ("gfx942",)),
    ("tf32_storage", (16, 16, 32), {"data": ConvDataSpec("tf32", "tf32", "fp32")}, ()),
    (
        "fp16_storage",
        (16, 16, 32),
        {"data": ConvDataSpec("fp16", "fp16", "fp16", mma_dtype="tf32")},
        (),
    ),
)


def _matrix():
    for direction in _DIRECTIONS:
        for case in _CASES:
            yield pytest.param(direction, *case, id=f"{direction}-{case[0]}")


@pytest.mark.parametrize("direction,case,atom,kw,admitted", list(_matrix()))
def test_validate_and_is_valid_agree(direction, case, atom, kw, admitted):
    make, gate, valid = _DIRECTIONS[direction]
    spec = make(atom, **kw)

    for arch in _ARCHES:
        why = gate(spec, arch)
        ok, reason = valid(spec, arch)
        if arch in admitted:
            assert not why, (arch, why)
            assert ok, (arch, reason)
        else:
            # Refused on this arch, by the gate, with the gate's reason.
            assert why, f"{arch}: the gate admits a spec it must refuse"
            assert (ok, reason) == (False, why), arch

    arch_free = gate(spec)
    if admitted:
        assert not arch_free, arch_free
        spec.validate()
    else:
        assert arch_free, "validate() would admit a spec no arch can build"
        with pytest.raises(ValueError) as e:
            spec.validate()
        assert str(e.value) == arch_free
