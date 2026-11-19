# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit coverage for both fp8 pk8 saturation paths in PackData_FLOAT8.

``PackData_FLOAT8`` clamps each f32 before the pk8 cvt. On a target with the
``HasVMaximumMinimumF32`` cap it uses the fast ``v_pk_fma_f32 x*0+x`` +
``v_maximumminimum_f32`` sequence; otherwise it falls back to the
``v_cmp_class`` / ``v_med3_f32`` / ``v_cndmask`` sequence. Full-kernel
characterization only exercises whichever branch the host toolchain selects,
so this drives both by forcing the cap, keeping both paths covered.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gpu_test_helpers import init_rocisa  # noqa: E402

import rocisa  # noqa: E402
from tensilelite.Components.PackData import PackData_FLOAT8  # noqa: E402

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"


def _emit(monkeypatch, has_maxmin):
    """Emit the gwvw=8 pk8 pack with HasVMaximumMinimumF32 forced to @p has_maxmin."""
    init_rocisa(_ARCH, 32)
    real = rocisa.rocIsa.getInstance()

    class CapWrap:
        # Force the two caps the pk8 saturation branch reads; delegate the rest
        # (getKernel, output options, ...) to the real singleton so emission works.
        def getAsmCaps(self):
            caps = dict(real.getAsmCaps())
            caps["HasCvtScalePk8Fp8F32"] = True
            caps["HasVMaximumMinimumF32"] = has_maxmin
            return caps

        def __getattr__(self, name):
            return getattr(real, name)

    monkeypatch.setattr(rocisa.rocIsa, "getInstance", staticmethod(lambda: CapWrap()))

    struct = SimpleNamespace(
        vgprFp8NanInf=20, vgprFp8Temp=21, vgprFp8Min=22, vgprFp8Max=23
    )
    module = PackData_FLOAT8()(
        gwvw=8, destIdx=0, elementSumIdx=0, fp8CVTVgprStruct=struct,
        tmpS01=30, laneSGPRC=2,
    )
    return str(module)


def test_pk8_saturation_maxmin_path(monkeypatch):
    """With HasVMaximumMinimumF32: fma(x*0+x) + maximumminimum, no med3/cndmask."""
    asm = _emit(monkeypatch, True)
    assert "v_maximumminimum_f32" in asm
    assert "v_pk_fma_f32" in asm
    assert "v_med3_f32" not in asm
    assert "v_cndmask_b32" not in asm
    # 4 fma (pairs) + 8 clamp, one cvt, for a single 8-wide group.
    assert asm.count("v_maximumminimum_f32") == 8
    assert "v_cvt_scalef32_pk8" in asm


def test_pk8_saturation_fallback_path(monkeypatch):
    """Without HasVMaximumMinimumF32: cmp_class/med3/cndmask, no maximumminimum."""
    asm = _emit(monkeypatch, False)
    assert "v_med3_f32" in asm
    assert "v_cndmask_b32" in asm
    assert "v_maximumminimum_f32" not in asm
    assert "v_pk_fma_f32" not in asm
    assert asm.count("v_med3_f32") == 8
    assert "v_cvt_scalef32_pk8" in asm
