#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""On-device correctness for the three grouped block-scale quant bridges that
had none: ``grouped_gemm_aquant``, ``grouped_gemm_abquant``, ``grouped_gemm_bquant``.

Every defect the round-3 ``default_config`` sweep found lived on a *default
config family* that no authored test instantiated -- including a
``grouped_gemm_bquant`` default whose C was 98% exactly zero on gfx950 while the
whole registered suite stayed green.  So the unit of coverage here is the
shipped ``default_*_config`` factory: the test enumerates them by ``inspect``
rather than naming a hand-picked few, which means a factory added later is
covered the day it is added.

Per config it asserts, at a shape that is non-square in every quantized axis
(M != N and QK != QN):

  A0  C is all-finite and >= 99% non-zero          -- the all-zeros / NaN class
  A1  C matches an independent NumPy fp32 reference built from the same
      quantized operand values, global ``max|C-R| / (max|R| + 1e-6) <= 0.05``
  A2  ``warp_tile_k`` equals the canonical arch rule (a static assert, kept here
      so a wrong tile is reported next to the numeric result it produces)

Configs whose kernel does not build, or which the runner refuses, are reported
as failures with the reason -- never silently skipped.  A skip is only produced
for the whole module, and only when there is no usable GPU/hipcc/ml_dtypes or
the device is not on the shared native-fp8 allowlist.

The known-broken families are listed in ``_EXPECTED_BROKEN`` with the defect
each one carries.  They are asserted to *still* be broken, so the day one is
fixed this test fails and the entry has to be removed -- the exemption cannot
outlive the defect.

Run:
    python3 -m pytest test_grouped_quant_gpu_correctness.py -v
"""

import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))
for _p in (_HERE.parent.parent / "python", _HERE.parent.parent / "codegen"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from conftest import (  # noqa: E402
    gpu_available as _have_gpu,
    ml_dtypes_available as _have_ml_dtypes,
    native_fp8_skip_marker as _native_fp8_skip_marker,
)
from codegen_common import quant_warp_tile_k, variant_is_8bit_float  # noqa: E402

# Imported by name, not only through __import__, so the I1 invariant in
# test_quant_bridge_invariants.py can see that these three bridges now have
# on-device coverage.
import grouped_gemm_abquant_utils  # noqa: E402,F401
import grouped_gemm_aquant_utils  # noqa: E402,F401
import grouped_gemm_bquant_utils  # noqa: E402,F401

_SKIP_NO_GPU = pytest.mark.skipif(not _have_gpu(), reason="no ROCm GPU detected")
_SKIP_NO_MLD = pytest.mark.skipif(not _have_ml_dtypes(), reason="ml_dtypes not installed")
_SKIP_NO_FP8 = _native_fp8_skip_marker()

# Non-square in every quantized axis: M != N, and with quant_group_n=1
# QN = N = 512 while QK = K/128 = 8.
_M, _N, _K = 256, 512, 1024
_TOL = 0.05
_SEED = 1234


# ---------------------------------------------------------------------------
# codecs -- deliberately independent of the modules under test
# ---------------------------------------------------------------------------

def _ml_type(kind: str, arch: str):
    import ml_dtypes
    ocp = ("gfx950" in arch) or ("gfx12" in arch)
    if kind == "fp8":
        return ml_dtypes.float8_e4m3fn if ocp else ml_dtypes.float8_e4m3fnuz
    return ml_dtypes.float8_e5m2 if ocp else ml_dtypes.float8_e5m2fnuz


def _enc8(a, kind, arch):
    return np.ascontiguousarray(
        np.asarray(a, np.float32).astype(_ml_type(kind, arch))).view(np.uint8)


def _qdq8(a, kind, arch):
    return np.asarray(a, np.float32).astype(_ml_type(kind, arch)).astype(np.float32)


def _enc_e8m0(a):
    a = np.clip(np.asarray(a, np.float32), 0.0, np.float32(2.0 ** 127))
    out = np.zeros(a.shape, np.uint8)
    nz = a > 0.0
    out[nz] = np.clip(np.floor(np.log2(a[nz])).astype(np.int32) + 127, 0, 254).astype(np.uint8)
    return out


def _dec_e8m0(a):
    return np.exp2(np.asarray(a, np.uint8).astype(np.float32) - 127.0)


def _bf16_raw(x):
    return np.frombuffer(np.asarray(x, np.float32).tobytes(),
                         dtype=np.uint16)[1::2].reshape(np.shape(x))


def _bf16_f32(a):
    u = np.asarray(a).flatten().astype(np.uint16)
    w = np.zeros(len(u) * 2, np.uint16)
    w[1::2] = u
    return w.view(np.float32).reshape(np.shape(a))


_FP4_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)


def _pack_nibbles(codes, order):
    """2 codes per byte along the first axis: even -> HIGH nibble, odd -> LOW.

    The convention the in-repo bquant test establishes for pk_int4 B
    (test_bquant_gpu_correctness.py:375-393): value = code - 8, and
    pk_int4_t_to_fp32x2_t + the (k & 1) pick in load_b put the even-k code in
    the high nibble.  ``order`` is 'F' for column-major B, 'C' for row-major A.
    """
    flat = np.asarray(codes, np.uint8).flatten(order=order)
    even, odd = flat[0::2] & 0x0F, flat[1::2] & 0x0F
    return (odd | (even << 4)).astype(np.uint8)


def _global_max_rel(got, ref):
    g, r = np.asarray(got, np.float32), np.asarray(ref, np.float32)
    return float(np.max(np.abs(g - r)) / (np.max(np.abs(r)) + 1e-6))


# ---------------------------------------------------------------------------
# case enumeration
# ---------------------------------------------------------------------------

_OPS = ("grouped_gemm_aquant", "grouped_gemm_abquant", "grouped_gemm_bquant")


def _factories(op):
    mod = __import__(f"{op}_utils")
    return [
        (name, fn)
        for name, fn in sorted(vars(mod).items())
        if name.startswith("default_") and name.endswith("_config")
        and inspect.isfunction(fn) and "gfx_arch" in inspect.signature(fn).parameters
    ]


def _all_cases():
    cases = []
    for op in _OPS:
        for name, _fn in _factories(op):
            cases.append((op, name))
    return cases


_CASES = _all_cases()
_CASE_IDS = [f"{op}.{name}" for op, name in _CASES]


# Families that are known to be broken today, with the defect.  Asserted to
# still fail; fixing one makes this test fail until its entry is deleted.
_EXPECTED_BROKEN = {
    # PreshuffleB B-weight path is numerically wrong on device (global max_rel
    # 0.477 / 0.463 on gfx950, i.e. the error is the size of the output).  The
    # preshuffle-quant siblings of the same op are correct, so this is specific
    # to the B-weight shuffle.
    ("grouped_gemm_abquant", "default_fp8_preshuffleb_config"):
        "PreshuffleB B-weight path is numerically wrong on device (max_rel ~0.48)",
    ("grouped_gemm_abquant", "default_bf8_preshuffleb_config"):
        "PreshuffleB B-weight path is numerically wrong on device (max_rel ~0.46)",
    # The grouped aquant ctypes lib copies pk_int4 A straight to the device; the
    # non-grouped twin permutes it first (gemm_aquant_ctypes_lib.cpp:111-119).
    # Pre-applying that permute on the host takes these from 1.445 to 0.00031,
    # so the missing call is the whole remaining defect.
    ("grouped_gemm_aquant", "default_fp8i4_config"):
        "pk_int4 A is not permute_i4_inplace'd by grouped_gemm_aquant_ctypes_lib.cpp",
    ("grouped_gemm_aquant", "default_bf8i4_config"):
        "pk_int4 A is not permute_i4_inplace'd by grouped_gemm_aquant_ctypes_lib.cpp",
    # Same defect on the B side: gemm_bquant_ctypes_lib.cpp:148-150 permutes a
    # pk_int4 B, grouped_gemm_bquant_ctypes_lib.cpp:186 copies it straight
    # through.  Pre-applying the permute on the host takes these from 1.343 /
    # 1.372 to 0.00029 / 0.00030 (global metric), so the missing call is the
    # whole remaining defect.
    ("grouped_gemm_bquant", "default_fp8i4_config"):
        "pk_int4 B is not permute_i4_inplace'd by grouped_gemm_bquant_ctypes_lib.cpp",
    ("grouped_gemm_bquant", "default_bf8i4_config"):
        "pk_int4 B is not permute_i4_inplace'd by grouped_gemm_bquant_ctypes_lib.cpp",
}


def test_expected_broken_entries_all_name_a_live_config():
    """A stale entry is an exemption for a config that no longer exists.

    grouped_gemm_bquant's twelve preshuffle defaults used to be listed here;
    they were removed from the op instead, and this assertion is what makes that
    removal visible rather than leaving dead exemptions behind.
    """
    live = set(_CASES)
    stale = sorted(k for k in _EXPECTED_BROKEN if k not in live)
    assert not stale, f"_EXPECTED_BROKEN names configs that no longer ship: {stale}"


# ---------------------------------------------------------------------------
# operands + reference
# ---------------------------------------------------------------------------

def _variant_kinds(op, variant):
    """(a_kind, b_kind, q_kind, c_kind) for one op/variant."""
    if variant in ("fp8", "bf8"):
        return (variant, variant, "f32", "f16")
    if variant in ("fp8i4", "bf8i4"):
        base = variant[:3]
        # AQuant scales A, so A is the int4 operand there; BQuant scales B.
        if op == "grouped_gemm_aquant":
            return ("pk_int4", base, base, "f16")
        return (base, "pk_int4", base, "f16")
    if variant == "fp4":
        return ("pk_fp4", "pk_fp4", "f32", "f16")
    if variant == "mx_bf16bf16":
        return ("bf16", "bf16", "e8m0", "bf16")
    if variant == "mx_bf16bf8":
        return ("bf16", "bf8", "e8m0", "bf16")
    if variant == "mx_bf16fp4":
        return ("bf16", "pk_fp4", "e8m0", "bf16")
    raise AssertionError(f"unhandled variant {variant!r} for {op}")


def _make_a(kind, rng, arch):
    if kind in ("fp8", "bf8"):
        f = rng.uniform(-1.0, 1.0, (_M, _K)).astype(np.float32)
        return _enc8(f, kind, arch), _qdq8(f, kind, arch)
    if kind == "bf16":
        f = rng.uniform(-1.0, 1.0, (_M, _K)).astype(np.float32)
        raw = _bf16_raw(f)
        return raw, _bf16_f32(raw)
    codes = rng.integers(0, 16, size=(_M, _K), dtype=np.uint8)
    dec = (codes.astype(np.float32) - 8.0) if kind == "pk_int4" else _FP4_LUT[codes]
    return _pack_nibbles(codes, "C"), dec


def _make_b(kind, rng, arch):
    if kind in ("fp8", "bf8"):
        f = rng.uniform(-1.0, 1.0, (_K, _N)).astype(np.float32)
        return _enc8(f, kind, arch), _qdq8(f, kind, arch)
    if kind == "bf16":
        f = rng.uniform(-1.0, 1.0, (_K, _N)).astype(np.float32)
        raw = _bf16_raw(f)
        return raw, _bf16_f32(raw)
    codes = rng.integers(0, 16, size=(_K, _N), dtype=np.uint8)
    dec = (codes.astype(np.float32) - 8.0) if kind == "pk_int4" else _FP4_LUT[codes]
    return _pack_nibbles(codes, "F"), dec


def _scale_a(A_dec, AQ, group_k):
    out = np.array(A_dec, np.float32, copy=True)
    AQb = AQ if AQ.shape[0] == _M else np.broadcast_to(AQ, (_M, AQ.shape[1]))
    for qi in range(AQ.shape[1]):
        out[:, qi * group_k:min((qi + 1) * group_k, _K)] *= AQb[:, qi][:, None]
    return out


def _scale_b(B_dec, BQ, group_k, group_n):
    out = np.array(B_dec, np.float32, copy=True)
    for qi in range(BQ.shape[0]):
        for qj in range(BQ.shape[1]):
            out[qi * group_k:min((qi + 1) * group_k, _K),
                qj * group_n:min((qj + 1) * group_n, _N)] *= float(BQ[qi, qj])
    return out


def _run_case(op, factory_name, arch, out_dir):
    """Build and run one default config; return (C_f32, ref_f32, kernel_name)."""
    mod = __import__(f"{op}_utils")
    cfg = dict(_factories(op))[factory_name](gfx_arch=arch)
    variant = cfg.variant_key
    a_kind, b_kind, q_kind, c_kind = _variant_kinds(op, variant)
    rng = np.random.default_rng(_SEED)

    so_paths = mod.setup_multiple_aquant_dispatchers(
        configs=[cfg], output_dir=out_dir, gfx_arch=arch,
    ) if op == "grouped_gemm_aquant" else (
        mod.setup_multiple_abquant_dispatchers(
            configs=[cfg], output_dir=out_dir, gfx_arch=arch)
        if op == "grouped_gemm_abquant"
        else mod.setup_multiple_bquant_dispatchers(
            configs=[cfg], output_dir=out_dir, gfx_arch=arch)
    )
    assert so_paths and so_paths[0] is not None, (
        f"{op}.{factory_name}: kernel {cfg.name} failed to build for {arch}"
    )

    A_raw, A_dec = _make_a(a_kind, rng, arch)
    B_raw, B_dec = _make_b(b_kind, rng, arch)
    c_dtype = np.float16 if c_kind == "f16" else np.uint16

    if op == "grouped_gemm_aquant":
        gK = cfg.quant_group_k
        prob = mod.AQuantGemmProblem(M=_M, N=_N, K=_K, quant_group_k=gK,
                                     quant_group_n=cfg.quant_group_n,
                                     quant_group_m=cfg.quant_group_m)
        AQ = rng.uniform(0.5, 1.5, (prob.QM_A, prob.QK_A)).astype(np.float32)
        AQ_arg, AQ_ref = AQ, AQ
        if q_kind in ("fp8", "bf8"):
            AQ_arg, AQ_ref = _enc8(AQ, q_kind, arch), _qdq8(AQ, q_kind, arch)
        res = mod.AQuantGpuGemmRunner(so_paths[0]).run(
            A=A_raw, B=B_raw, AQ=AQ_arg, problem=prob, c_dtype=c_dtype)
        ref = _scale_a(A_dec, AQ_ref, gK) @ B_dec

    elif op == "grouped_gemm_abquant":
        agK, bgK, bgN = cfg.aquant_group_k, cfg.bquant_group_k, cfg.bquant_group_n
        prob = mod.ABQuantGemmProblem(M=_M, N=_N, K=_K, aquant_group_k=agK,
                                      bquant_group_k=bgK, bquant_group_n=bgN)
        AQ = rng.uniform(0.5, 1.5, (prob.QM_A, prob.QK_A)).astype(np.float32)
        BQ = rng.uniform(0.5, 1.5,
                         (math.ceil(_K / bgK), math.ceil(_N / bgN))).astype(np.float32)
        res = mod.ABQuantGpuGemmRunner(so_paths[0]).run(
            A=A_raw, B=B_raw, AQ=AQ, BQ=np.asfortranarray(BQ),
            problem=prob, c_dtype=c_dtype)
        ref = _scale_a(A_dec, AQ, agK) @ _scale_b(B_dec, BQ, bgK, bgN)

    else:  # grouped_gemm_bquant
        gK, gN = cfg.quant_group_k, cfg.quant_group_n
        prob = mod.BQuantGemmProblem(M=_M, N=_N, K=_K, quant_group_k=gK,
                                     quant_group_n=gN,
                                     quant_group_m=cfg.quant_group_m)
        BQf = rng.uniform(0.5, 1.5,
                          (math.ceil(_K / gK), math.ceil(_N / gN))).astype(np.float32)
        if q_kind == "e8m0":
            BQ_arg = _enc_e8m0(BQf)
            BQ_ref = _dec_e8m0(BQ_arg)
        elif q_kind in ("fp8", "bf8"):
            BQ_arg = BQf                       # the runner encodes to QDataType
            BQ_ref = _qdq8(BQf, q_kind, arch)
        else:
            BQ_arg, BQ_ref = BQf, BQf
        res = mod.BQuantGpuGemmRunner(so_paths[0]).run(
            A=A_raw, B=B_raw, BQ=BQ_arg, problem=prob, c_dtype=c_dtype)
        ref = A_dec @ _scale_b(B_dec, BQ_ref, gK, gN)

    C = np.asarray(res.C)
    C = C.astype(np.float32) if c_kind == "f16" else _bf16_f32(C)
    return C, np.asarray(ref, np.float32), res.kernel_name


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

@_SKIP_NO_GPU
@_SKIP_NO_MLD
@_SKIP_NO_FP8
@pytest.mark.parametrize("op,factory_name", _CASES, ids=_CASE_IDS)
def test_grouped_quant_default_config_on_device(op, factory_name, gpu_arch, tmp_path):
    """Every shipped default config of the three grouped quant bridges."""
    expected_broken = _EXPECTED_BROKEN.get((op, factory_name))

    try:
        C, ref, kernel_name = _run_case(op, factory_name, gpu_arch, tmp_path)
    except Exception as exc:  # noqa: BLE001
        if expected_broken:
            pytest.xfail(f"{op}.{factory_name}: {expected_broken} ({exc})")
        raise

    finite = bool(np.all(np.isfinite(C)))
    nonzero = float(np.count_nonzero(C)) / C.size
    max_rel = _global_max_rel(C, ref) if finite else float("inf")
    ok = finite and nonzero >= 0.99 and max_rel <= _TOL
    detail = (f"{kernel_name}: finite={finite} nonzero_frac={nonzero:.4f} "
              f"global_max_rel={max_rel:.5f}")

    if expected_broken:
        assert not ok, (
            f"{op}.{factory_name} now PASSES ({detail}); delete its "
            f"_EXPECTED_BROKEN entry so this config is enforced from now on"
        )
        pytest.xfail(f"{op}.{factory_name}: {expected_broken}; {detail}")

    assert finite, f"{op}.{factory_name}: C contains NaN/Inf -- {detail}"
    assert nonzero >= 0.99, (
        f"{op}.{factory_name}: C is {100 * (1 - nonzero):.1f}% exactly zero -- "
        f"the silent-all-zeros mode. {detail}"
    )
    assert max_rel <= _TOL, f"{op}.{factory_name}: wrong answer -- {detail}"


@pytest.mark.parametrize("op,factory_name", _CASES, ids=_CASE_IDS)
def test_grouped_quant_default_config_warp_tile_k(op, factory_name):
    """Static companion to the numeric case; runs with no GPU.

    ``warp_tile_k`` is the one config field whose wrong value is invisible: it
    compiles and returns zeros or garbage.  Asserting it next to the numeric
    result means a future wrong tile is reported by name, not by a mystery
    ``max_rel``.
    """
    mod = __import__(f"{op}_utils")
    for arch in ("gfx942", "gfx950"):
        cfg = dict(_factories(op))[factory_name](gfx_arch=arch)
        is_flat = cfg.pipeline == "preshuffleb" or getattr(cfg, "preshuffle_aq", False)
        if getattr(cfg, "variant_key", "").startswith("mx_"):
            continue  # MX tiles come from GemmConfigMixedPrecision / QuantPrefill
        expected = quant_warp_tile_k(
            arch,
            is_8bit_float=variant_is_8bit_float(cfg.variant_key),
            is_flat_mm=is_flat,
            m_warp_tile=cfg.warp_tile_m,
        )
        assert cfg.warp_tile_k == expected, (
            f"{op}.{factory_name}(gfx_arch={arch!r}).warp_tile_k == "
            f"{cfg.warp_tile_k}, arch rule says {expected}"
        )
    assert mod is not None
