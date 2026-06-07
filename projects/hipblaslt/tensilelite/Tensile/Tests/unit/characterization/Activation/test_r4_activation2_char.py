################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R4 — remaining activation-function emit arms characterization test.

Covers Tensile/Activation.py target ranges 489-617, 639-718, 727-836, 852-936
by directly calling ActivationModule.getModule() for the activation functions
and data-type variants not yet exercised by the existing test_activation_char.py:

  Range 489-617 (getAbsModule Double/Int32 + getClippedReluModule + getExpModule):
    abs(D)             -> lines 489-490
    abs(I) saturate    -> lines 491-499
    clippedrelu(S)     already covered; clippedrelu(D) -> 531-537, (I) -> 538-543
    exp(S)             -> lines 567-571

  Range 639-718 (getGeluModule Single + getLeakyReluModule Double/Int32 +
                 getReluModule Int32-saturate + getSigmoidModule Single):
    gelu(S)            -> lines 622-634
    geluscaling(S)     -> lines 635-637
    leakyrelu(D)       -> lines 661-666
    leakyrelu(I)       -> lines 667-671
    relu(I) saturate   -> lines 684-688
    sigmoid(S)         -> lines 719-725

  Range 727-836 (getTanhModule Single + getDGeluModule isAlt/guard paths):
    tanh(S)            -> lines 772-785
    dgelu(S) isAlt=T   -> lines 816-834
    dgelu(S) guard=T   -> lines 825-831

  Range 852-936 (getDGeluModule end + getSiluModule + getSwishModule + getClampModule):
    silu(S)            -> lines 866-881
    swish(S)           -> lines 883-900
    clamp(D)           -> lines 904-916
    clamp(I)           -> lines 910-928

Strategy: initialize the rocisa singleton with gfx942 arch caps (required by
getExpModule / getSigmoidModule / getTanhModule for the TransOpWait guard), then
call getModule() with dummy vgpr indices (0, 1). All calls are CPU-only; no GPU.
"""

import importlib
import shutil

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Module-level imports and rocisa init
# ---------------------------------------------------------------------------
A = importlib.import_module("Tensile.Activation")
DataType = importlib.import_module("Tensile.Common.DataType").DataType


def _init_rocisa():
    """Initialize the rocisa singleton with gfx942 arch caps.

    getExpModule, getSigmoidModule, getTanhModule, getDGeluModule all call
    ``rocIsa.getInstance().getArchCaps()["TransOpWait"]``.  Without an
    initialized ISA, getArchCaps() returns an empty dict causing a KeyError.
    Init once per test module (the singleton is process-global) using the same
    (9,4,2) / wavefront-64 that codegen_harness uses for gfx942.
    """
    import rocisa

    isa = (9, 4, 2)
    wavefront = 64
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri = rocisa.rocIsa.getInstance()
    ri.init(isa, asmpath)
    ri.setKernel(isa, wavefront)
    return ri


# Module-level init: done once at import time so all tests share it.
_RI = _init_rocisa()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _mod(dtype_code, act_name, *, saturate=False, usePK=True, isAlt=False, guard=False):
    """Call ActivationModule.getModule with a fresh module per invocation."""
    m = A.ActivationModule()
    m.setSaturationForInt8(saturate)
    m.setUsePK(usePK)
    m.setAlt(isAlt)
    m.setGuard(guard)
    return m.getModule(DataType(dtype_code), act_name, 0, 1)


# ---------------------------------------------------------------------------
# Range 489-617: getAbsModule(Double/Int32) + getClippedReluModule + getExpModule
# ---------------------------------------------------------------------------

def test_abs_double_branch():
    """abs(Double): lines 489-490 — VAndB32 on vgprIn+1 (Double sign bit)."""
    mod = _mod("D", "abs")
    assert type(mod).__name__ == "Module"


def test_abs_int32_branch_no_saturate():
    """abs(Int32) saturate=False: lines 498-499 — VMaxI32 variant."""
    mod = _mod("I", "abs", saturate=False)
    assert type(mod).__name__ == "Module"


def test_abs_int32_branch_saturate():
    """abs(Int32) saturate=True: lines 494-497 — VMed3I32 variant (saturateI8 path)."""
    mod = _mod("I", "abs", saturate=True)
    assert type(mod).__name__ == "Module"


def test_clippedrelu_double():
    """clippedrelu(Double): lines 531-537."""
    mod = _mod("D", "clippedrelu")
    assert type(mod).__name__ == "Module"


def test_clippedrelu_int32():
    """clippedrelu(Int32): lines 538-543."""
    mod = _mod("I", "clippedrelu")
    assert type(mod).__name__ == "Module"


def test_exp_single():
    """exp(Single): lines 567-571 — VMulF32 + VExpF32 + optional SNop."""
    mod = _mod("S", "exp")
    assert type(mod).__name__ == "Module"


# ---------------------------------------------------------------------------
# Range 639-718: getGeluModule(Single) + getLeakyReluModule(Double/Int32)
#               + getReluModule(Int32-saturate) + getSigmoidModule(Single)
# ---------------------------------------------------------------------------

def test_gelu_single():
    """gelu(Single): lines 622-634 — standard gelu without scaling."""
    mod = _mod("S", "gelu")
    assert type(mod).__name__ == "Module"


def test_geluscaling_single():
    """geluscaling(Single): lines 635-637 — gelu + alpha scaling branch."""
    mod = _mod("S", "geluscaling")
    assert type(mod).__name__ == "Module"


def test_leakyrelu_double():
    """leakyrelu(Double): lines 661-666 — VMulF64 + VCmpGEF64 + pair VCndMask."""
    mod = _mod("D", "leakyrelu")
    assert type(mod).__name__ == "Module"


def test_leakyrelu_int32():
    """leakyrelu(Int32): lines 667-671 — VMulLOU32 + VCmpGEI32 + VCndMaskB32."""
    mod = _mod("I", "leakyrelu")
    assert type(mod).__name__ == "Module"


def test_relu_int32_saturate():
    """relu(Int32) saturate=True: lines 684-688 — VMed3I32 path."""
    mod = _mod("I", "relu", saturate=True)
    assert type(mod).__name__ == "Module"


def test_sigmoid_single():
    """sigmoid(Single): lines 719-725 — VMulF32 + getExpModule + VAddF32 + VRcpF32."""
    mod = _mod("S", "sigmoid")
    assert type(mod).__name__ == "Module"


# ---------------------------------------------------------------------------
# Range 727-836: getTanhModule(Single) + getDGeluModule isAlt/guard paths
# ---------------------------------------------------------------------------

def test_tanh_single():
    """tanh(Single): lines 772-785 — VMulF32 + getExpModule + VAddF32 + VRcpF32 + VFmaF32."""
    mod = _mod("S", "tanh")
    assert type(mod).__name__ == "Module"


def test_dgelu_single_isalt_true():
    """dgelu(Single) isAlt=True: lines 816-834 — alternate cosh-2 path."""
    mod = _mod("S", "dgelu", isAlt=True)
    assert type(mod).__name__ == "Module"


def test_dgelu_single_isalt_guard():
    """dgelu(Single) isAlt=True guard=True: lines 825-831 — inf-guard sub-path."""
    mod = _mod("S", "dgelu", isAlt=True, guard=True)
    assert type(mod).__name__ == "Module"


# ---------------------------------------------------------------------------
# Range 852-936: getSiluModule + getSwishModule + getClampModule(Double/Int32)
# ---------------------------------------------------------------------------

def test_silu_single():
    """silu(Single): lines 866-881 — getSigmoidModule + VMulF32."""
    mod = _mod("S", "silu")
    assert type(mod).__name__ == "Module"


def test_swish_single():
    """swish(Single): lines 883-900 — alpha-scaled silu variant."""
    mod = _mod("S", "swish")
    assert type(mod).__name__ == "Module"


def test_clamp_double():
    """clamp(Double): lines 904-916 — VMinF64 + VMaxF64 pair."""
    mod = _mod("D", "clamp")
    assert type(mod).__name__ == "Module"


def test_clamp_int32():
    """clamp(Int32): lines 910-928 — VMinI32 + VMaxI32 pair."""
    mod = _mod("I", "clamp")
    assert type(mod).__name__ == "Module"


# ---------------------------------------------------------------------------
# Parametric sweep: every covered activation x Single to exercise getModule dispatch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("act_name", [
    "gelu", "geluscaling", "sigmoid", "tanh", "exp", "silu", "swish",
])
def test_single_dtype_sweep(act_name):
    """All remaining activation functions on Single dtype return a Module."""
    mod = _mod("S", act_name)
    assert type(mod).__name__ == "Module"


@pytest.mark.parametrize("dtype,act", [
    ("D", "abs"), ("I", "abs"),
    ("D", "clippedrelu"), ("I", "clippedrelu"),
    ("D", "leakyrelu"), ("I", "leakyrelu"),
    ("D", "clamp"), ("I", "clamp"),
])
def test_double_and_int32_dispatch(dtype, act):
    """Double/Int32 branches of multi-type functions return a Module."""
    mod = _mod(dtype, act)
    assert type(mod).__name__ == "Module"


# ---------------------------------------------------------------------------
# P6 survivor-1 kill: pin the Relu clamp FLOOR operand (src1 == 0).
#
# getReluModule emits a max-style instruction per compute dtype to compute
# max(0, x) (int8-saturate additionally clamps to 127 via VMed3I32). The
# existing relu tests assert only that a Module emits and which VALU op is used
# — none pins the clamp bound, so mutating the floor (e.g. VMaxF32 src1 0 -> 1)
# was undetected. These assertions pin the floor operand to 0 for every dtype,
# killing that mutant and its per-dtype analogues (half/single/double/int).
# Renderings (code before the '//' comment):
#   half      v_pk_max_f16 v1, v0, 0
#   single    v_max_f32    v1, v0, 0
#   double    v_max_f64    v[1:2], v[0:1], 0
#   int32     v_max_i32    v1, v0, 0
#   int8-sat  v_med3_i32   v1, v0, 0, v0     (// x = min(127, max(0, x)))
# In every case the third token (src1) is the clamp floor and must be 0.
# ---------------------------------------------------------------------------

def _relu_clamp_floor_operand(dtype_code, *, saturate=False):
    """Return the src1 (clamp-floor) operand string of the Relu max instruction.

    Locates the emitted instruction whose comment computes ``max(0, x)`` and
    returns its third operand (dst, src0, src1) with whitespace stripped.
    """
    mod = _mod(dtype_code, "relu", saturate=saturate)
    for it in mod.items():
        line = str(it)
        code = line.split("//", 1)[0]
        if "max(0, x" in line:  # matches both 'x = max(0, x)' and the int8 min/max
            operands = [tok.strip() for tok in code.split(",")]
            assert len(operands) >= 3, f"unexpected relu instruction: {code!r}"
            return operands[2]
    raise AssertionError(f"no max(0,x) clamp instruction emitted for relu {dtype_code} (sat={saturate})")


@pytest.mark.parametrize("dtype_code,saturate", [
    ("H", False),   # VMaxPKF16
    ("S", False),   # VMaxF32   (the P6 survivor's exact mutant line)
    ("D", False),   # VMaxF64
    ("I", False),   # VMaxI32
    ("I", True),    # VMed3I32 (int8 saturate: floor 0, ceil 127)
])
def test_relu_clamp_floor_is_zero(dtype_code, saturate):
    """Relu clamp floor (src1 of the max op) must be exactly 0 — pins max(0, x)."""
    assert _relu_clamp_floor_operand(dtype_code, saturate=saturate) == "0"
