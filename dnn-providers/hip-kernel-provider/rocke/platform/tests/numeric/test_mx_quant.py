# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""MXFP8 activation quantizer (``rocke.helpers.mx_quant``).

CPU: the numpy reference against an independent E4M3 codec (``ml_dtypes``) and
the rule's edge cases; the emitted IR of every knob value. gfx950: the device
emitters bitwise against the reference (codes and E8M0 bytes) over random
blocks, zero blocks, denormal inputs, the lower E8M0 clamp, the exact
``amax = 448 * 2^k`` boundary and RNE ties, for every ``cvt`` / ``reduce`` /
``nan`` / ``xor16`` knob and every lane split of a 32-value block. Machines
without a gfx950 skip the device lane (``ROCKE_REQUIRE_GFX950=1`` turns that
into a failure).

Engines: the helper emits mirrored ops only, so every build here except the
``cvt="intrinsic"`` one (a test-local ``cvt`` callable around the Python-only
``IRBuilder.cvt_scalef32_pk_fp8_f32``; the library family owns that knob) is
lowered by both engines: ``test_mx_quant_ir_cpp_python_byte_identity`` at
every golden flavor, and the device tests compile with ``backend="both"``
when ``rocke_engine`` imports. A cpp -> python fallback fails the test.
"""

import ctypes
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from rocke.core.ir import F32, I32, IRBuilder, PtrType, uses_python_only_ops
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers import mx_quant as mq

CVTS = ("asm", "intrinsic")


def _intrinsic_cvt(b, old, lo, hi_v, scale, *, hi):
    return b.cvt_scalef32_pk_fp8_f32(
        b.const_i32(0) if old is None else old, lo, hi_v, scale, hi=hi
    )


def _cvt_arg(cvt):
    return _intrinsic_cvt if cvt == "intrinsic" else cvt


def _engine_expected():
    return os.environ.get("ROCKE_BACKEND") in ("cpp", "both")


def _dual_backend():
    """Backend for a build both engines lower: ``"both"`` (differential, never
    falls back) when ``rocke_engine`` imports; else ``"python"`` explicitly,
    unless ``ROCKE_BACKEND=cpp|both`` expects the engine (then a failure)."""
    try:
        import rocke_engine  # noqa: F401
    except ImportError as e:
        if _engine_expected():
            pytest.fail(f"ROCKE_BACKEND expects the C++ engine: {e}")
        return "python"
    return "both"


@pytest.fixture(autouse=True)
def _no_cpp_fallback():
    from rocke.core.backend import cpp_fallbacks, reset_cpp_fallbacks

    reset_cpp_fallbacks()
    yield
    assert cpp_fallbacks() == [], f"cpp -> python fallback: {cpp_fallbacks()}"


# ----------------------------------------------------------------- inputs


def _bf16(x):
    """Round f32 to bf16 (RNE) and widen back."""
    u = np.asarray(x, np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.astype(np.uint32).view(np.float32)


def edge_blocks():
    """Named [n, 32] f32 blocks covering the rule's edges."""
    rng = np.random.default_rng(7)
    blocks = {}
    blocks["zero"] = np.zeros(32, np.float32)
    blocks["neg_zero"] = np.full(32, -0.0, np.float32)
    for k in (-20, -3, 0, 5, 40):
        b = _bf16(rng.uniform(-1, 1, 32) * 448 * 2.0**k)
        b[3] = 448 * 2.0**k  # amax exactly 448 * 2^k: ratio rounds to 2^k
        blocks[f"edge448_{k}"] = b
        c = b.copy()
        c[3] = np.nextafter(np.float32(448 * 2.0**k), np.float32(np.inf))
        blocks[f"above448_{k}"] = c
    # E8M0 lower clamp: ratio subnormal or zero.
    blocks["tiny"] = (rng.uniform(-1, 1, 32) * 1e-36).astype(np.float32)
    blocks["f32_denormal"] = (rng.uniform(-1, 1, 32) * 1e-39).astype(np.float32)
    mixed = _bf16(rng.standard_normal(32))
    mixed[::4] = np.float32(1e-40)
    blocks["mixed_denormal"] = mixed
    big = rng.uniform(-1, 1, 32).astype(np.float32) * np.float32(3.0e38)
    blocks["near_f32_max"] = big
    # RNE ties after scaling: amax 448 -> scale 1; 0.5-step ties in [16, 32)
    # (E4M3 step 2 there) and subnormal ties.
    ties = np.zeros(32, np.float32)
    ties[0] = 448
    ties[1:9] = [17, 19, 21, 23, -17, -19, -25, -27]
    ties[9:17] = np.array([1, 3, 5, 7, 9, 11, 13, 15], np.float32) * 2.0**-10
    ties[17:21] = [0.5 * 2.0**-9, -0.5 * 2.0**-9, 1.5 * 2.0**-9, 2.5 * 2.0**-9]
    blocks["rne_ties"] = ties
    return blocks


def make_input(nblocks, seed=0):
    rng = np.random.default_rng(seed)
    x = _bf16(rng.standard_normal((nblocks, 32)))
    x *= np.exp2(rng.integers(-30, 30, size=(nblocks, 1))).astype(np.float32)
    edges = np.stack(list(edge_blocks().values()))
    x[: len(edges)] = edges
    return x.astype(np.float32)


# ----------------------------------------------------------------- CPU


def test_reference_against_ml_dtypes():
    ml = pytest.importorskip("ml_dtypes")
    x = make_input(512, seed=3)
    codes, e = mq.ref_quantize_mxfp8(x)
    y = x.astype(np.float64) * np.exp2(127.0 - e.astype(np.float64))[:, None]
    want = y.astype(ml.float8_e4m3fn).view(np.uint8)
    assert np.array_equal(codes, want)


def test_reference_rule_edges():
    blocks = edge_blocks()
    _, e = mq.ref_quantize_mxfp8(np.stack([blocks["zero"], blocks["tiny"]]))
    assert list(e) == [1, 1]
    for k in (-20, -3, 0, 5, 40):
        codes, e = mq.ref_quantize_mxfp8(blocks[f"edge448_{k}"][None])
        assert e[0] == 127 + k and codes[0, 3] == 0x7E
        _, e2 = mq.ref_quantize_mxfp8(blocks[f"above448_{k}"][None])
        assert e2[0] == 128 + k
    codes, e = mq.ref_quantize_mxfp8(blocks["rne_ties"][None])
    assert e[0] == 127
    # 17 -> 16, 19 -> 20, 21 -> 20, 23 -> 24 (ties to even mantissa)
    vals = mq.ref_fp8e4m3_value(codes[0, 1:9])
    assert list(vals) == [16, 20, 20, 24, -16, -20, -24, -28]
    x = make_input(256, seed=11)
    codes, e = mq.ref_quantize_mxfp8(x)
    back = mq.ref_dequant_mxfp8(codes, e)
    amax = np.max(np.abs(x), axis=1)
    assert np.all(np.max(np.abs(back), axis=1) <= amax * (1 + 2.0**-3) + 1e-300)
    assert np.all(mq.ref_fp8e4m3_value(codes) <= 448)


def _quant_kernel(vpl, cvt, reduce, name, nan="propagate", xor16="swizzle"):
    lanes = 32 // vpl
    b = IRBuilder(name)
    px = b.param("X", PtrType(F32, "global"), noalias=True, readonly=True)
    pq = b.param("Q", PtrType(I32, "global"), noalias=True)
    pe = b.param("E", PtrType(I32, "global"), noalias=True)
    gl = b.add(b.mul(b.block_id_x(), b.const_i32(64)), b.thread_id_x())
    vals = [
        b.global_load(
            px, b.add(b.mul(gl, b.const_i32(vpl)), b.const_i32(i)), F32, align=4
        )
        for i in range(vpl)
    ]
    if vpl % 4 == 0:
        words, e = mq.quantize_mxfp8(
            b, vals, lanes, cvt=_cvt_arg(cvt), reduce=reduce, nan=nan, xor16=xor16
        )
        for i, w in enumerate(words):
            b.global_store(
                pq, b.add(b.mul(gl, b.const_i32(vpl // 4)), b.const_i32(i)), w, align=4
            )
    else:
        e, _ = mq.e8m0_fp8(
            b, mq.block_amax(b, vals, lanes, reduce=reduce, nan=nan, xor16=xor16)
        )
    b.global_store(pe, gl, e, align=4)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("cvt", CVTS)
@pytest.mark.parametrize("reduce", mq.REDUCE_MODES)
@pytest.mark.parametrize("nan", mq.NAN_MODES)
def test_emission(cvt, reduce, nan):
    k = _quant_kernel(8, cvt, reduce, f"mxq_{cvt}_{reduce}", nan)
    assert bool(uses_python_only_ops(k)) == (cvt == "intrinsic")
    ir = _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor="llvm22")
    # 8 values: 4 in-lane maximum3 + 2 cross-lane steps, or none (maxnum).
    assert ir.count('"v_maximum3_f32') == (6 if nan == "propagate" else 0)
    assert "v_mov_b32_dpp" not in ir  # 4 lanes: quad_perm steps only
    calls = ir.count("@llvm.amdgcn.cvt.scalef32.pk.fp8.f32(")
    asm = ir.count('"v_cvt_scalef32_pk_fp8_f32')
    if cvt == "intrinsic":
        assert (calls, asm) == (4 + 1, 0)  # 4 calls + 1 declare
    else:
        assert (calls, asm) == (0, 4) and "op_sel:[0,0,0,1]" in ir
    if reduce == "swizzle":
        assert "ds.swizzle" in ir
    with pytest.raises(ValueError):
        mq.pack_fp8x4(IRBuilder("x"), *([None] * 5), cvt="bogus")
    with pytest.raises(ValueError):
        mq.pack_fp8x4(IRBuilder("x"), *([None] * 5), cvt="intrinsic")


@pytest.mark.parametrize("vpl,pairs", [(4, 1), (2, 2), (1, 2)])
def test_emission_rowshift(vpl, pairs):
    """``dpp_rowshift``: xor 4 and 8 are inline-asm bank-masked row-shift
    pairs behind an ``s_nop 1``; xor 1 and 2 stay compiler DPP."""
    k = _quant_kernel(vpl, "asm", "dpp_rowshift", f"mxq_rs_{vpl}")
    ir = _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor="llvm22")
    assert ir.count("v_mov_b32_dpp") == 2 * pairs
    assert ir.count(r"s_nop 1\0A\09v_mov_b32_dpp") == pairs


@pytest.mark.parametrize("xor16", mq.XOR16_MODES)
def test_emission_xor16(xor16):
    k = _quant_kernel(1, "asm", "dpp", f"mxq_x16_{xor16}", xor16=xor16)
    ir = _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor="llvm22")
    assert ir.count("v_permlane16_swap_b32") == (1 if xor16 == "permlane" else 0)
    # 16-lane step on ds_swizzle with "swizzle"; the in-row steps stay DPP.
    assert ("ds.swizzle" in ir) == (xor16 == "swizzle")


def _parity_cases():
    """Every mirrored knob combination: name -> builder."""
    cases = {}
    for reduce in mq.REDUCE_MODES:
        for nan in mq.NAN_MODES:
            for vpl in (4, 8, 32):
                n = f"mxq_par_v{vpl}_{reduce}_{nan}"
                cases[n] = lambda vpl=vpl, reduce=reduce, nan=nan, n=n: _quant_kernel(
                    vpl, "asm", reduce, n, nan
                )
            for vpl in (1, 2):
                for xor16 in mq.XOR16_MODES:
                    n = f"mxq_par_v{vpl}_{reduce}_{nan}_{xor16}"
                    cases[n] = (
                        lambda vpl=vpl, reduce=reduce, nan=nan, xor16=xor16, n=n: (
                            _quant_kernel(vpl, "asm", reduce, n, nan, xor16)
                        )
                    )
    return cases


def test_mx_quant_ir_cpp_python_byte_identity():
    """The C++ engine lowers every mirrored quantizer build to the Python
    engine's bytes at every golden flavor. Engine absence is a skip on a CPU
    run and a failure under ``ROCKE_BACKEND=cpp|both``."""
    try:
        import rocke_engine as engine
    except ImportError as e:
        if _engine_expected():
            pytest.fail(f"ROCKE_BACKEND expects the C++ engine: {e}")
        pytest.skip(f"rocke_engine extension not built/importable: {e}")
    from rocke.core.ir_golden import GOLDEN_FLAVORS
    from rocke.core.ir_serialize import serialize

    cases = _parity_cases()
    mism, compared = [], 0
    for flavor in GOLDEN_FLAVORS:
        for name, build in cases.items():
            kernel = build()
            assert uses_python_only_ops(kernel) == [], name
            py = _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor=flavor)
            try:
                cpp = engine.lower_serialized_ir(
                    serialize(kernel), arch="gfx950", flavor=flavor
                )
            except Exception as e:  # noqa: BLE001 -- reported as a mismatch
                mism.append(f"{flavor}/{name}: engine error {str(e)[:200]}")
                continue
            compared += 1
            if cpp != py:
                mism.append(f"{flavor}/{name}")
    assert not mism, "mx_quant C++/Python IR mismatch:\n  " + "\n  ".join(mism)
    assert compared == len(cases) * len(GOLDEN_FLAVORS)


def test_argument_checks():
    b = IRBuilder("bad")
    p = b.param("X", PtrType(F32, "global"))
    v = b.global_load(p, b.const_i32(0), F32, align=4)
    with pytest.raises(ValueError):
        mq.quantize_mxfp8(b, [v] * 3, 8)
    with pytest.raises(ValueError):
        mq.block_amax(b, [v], 64)
    with pytest.raises(ValueError):
        mq.e8m0_fp8(b, b.const_i32(1))


# ----------------------------------------------------------------- gfx950


def _require_gpu():
    if not Path("/dev/kfd").exists():
        if os.environ.get("ROCKE_REQUIRE_GFX950") == "1":
            pytest.fail("requires a gfx950 GPU")
        pytest.skip("requires an AMD GPU")


def _require_gfx950():
    """Device check. Called after the first compile: loading HIP first can map
    a COMGR other than the one ``ROCKE_COMGR_LIB`` names, which the loader
    refuses to open beside it."""
    strict = os.environ.get("ROCKE_REQUIRE_GFX950") == "1"
    from rocke.runtime.hip_module import get_device_arch

    arch = get_device_arch(0)
    if arch != "gfx950":
        if strict:
            pytest.fail(f"requires gfx950, found {arch}")
        pytest.skip(f"requires gfx950, found {arch}")


def _isa(hsaco):
    tool = shutil.which("llvm-objdump")
    if tool is None:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "k.hsaco"
        path.write_bytes(hsaco)
        out = subprocess.run(
            [tool, "-d", str(path)], check=False, capture_output=True, text=True
        )
    return out.stdout


def run_quant(x, vpl, cvt, reduce, nan="propagate", xor16="swizzle"):
    """Launch the quantizer over ``x[nblocks, 32]``; return (codes, E per block)."""
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import Runtime

    nblocks = x.shape[0]
    per_wg = 64 * vpl // 32
    assert nblocks % per_wg == 0
    wgs = nblocks // per_wg
    kernel = _quant_kernel(
        vpl, cvt, reduce, f"mxq_v{vpl}_{cvt}_{reduce}_{nan}_{xor16}", nan, xor16
    )
    art = compile_kernel(
        kernel,
        arch="gfx950",
        backend="python" if cvt == "intrinsic" else _dual_backend(),
    )
    _require_gfx950()
    isa = _isa(art.hsaco)
    if isa is not None and vpl % 4 == 0:
        assert "v_cvt_scalef32_pk_fp8_f32" in isa
    rt = Runtime()
    module = rt.load_module(art.hsaco)
    nq = wgs * 64 * max(vpl // 4, 1)
    bufs = []
    try:
        xs = rt.alloc(x.nbytes)
        bufs.append(xs)
        rt.memcpy_h2d(
            xs, (ctypes.c_uint8 * x.nbytes).from_buffer_copy(x.tobytes()), x.nbytes
        )
        q = rt.alloc(nq * 4)
        e = rt.alloc(wgs * 64 * 4)
        bufs += [q, e]
        rt.memset(q, 0xFF, nq * 4)
        rt.memset(e, 0xFF, wgs * 64 * 4)
        rt.launch(
            module.get_function(art.kernel_name),
            (wgs, 1, 1),
            (64, 1, 1),
            struct.pack("<QQQ", xs, q, e),
        )
        rt.sync()
        hq = (ctypes.c_uint8 * (nq * 4))()
        he = (ctypes.c_uint8 * (wgs * 64 * 4))()
        rt.memcpy_d2h(hq, q, nq * 4)
        rt.memcpy_d2h(he, e, wgs * 64 * 4)
    finally:
        for p in bufs:
            rt.free(p)
        module.unload()
    codes = (
        np.frombuffer(bytes(hq), np.uint8).reshape(nblocks, 32)
        if vpl % 4 == 0
        else None
    )
    e_lane = np.frombuffer(bytes(he), np.int32).reshape(nblocks, 32 // vpl)
    # Every lane of a block must hold the block's E.
    assert np.all(e_lane == e_lane[:, :1]), "E differs across the lanes of a block"
    return codes, e_lane[:, 0]


@pytest.mark.parametrize("vpl", [4, 8, 16, 32])
@pytest.mark.parametrize("cvt", CVTS)
@pytest.mark.parametrize("reduce", mq.REDUCE_MODES)
@pytest.mark.parametrize("nan", mq.NAN_MODES)
def test_gfx950_bitwise(vpl, cvt, reduce, nan):
    _require_gpu()
    x = make_input(2048, seed=vpl)
    codes, e = run_quant(x, vpl, cvt, reduce, nan)
    want_codes, want_e = mq.ref_quantize_mxfp8(x)
    bad_e = np.nonzero(e != want_e)[0]
    assert bad_e.size == 0, (
        f"E mismatch at blocks {bad_e[:8]}: {e[bad_e[:8]]} vs {want_e[bad_e[:8]]}"
    )
    bad = np.argwhere(codes != want_codes)
    assert bad.size == 0, (
        f"{len(bad)} code mismatches, first {bad[:4].tolist()}: "
        f"{codes[tuple(bad[:4].T)]} vs {want_codes[tuple(bad[:4].T)]} "
        f"x={x[tuple(bad[:4].T)]}"
    )


@pytest.mark.parametrize("vpl", [1, 2])
@pytest.mark.parametrize("reduce", mq.REDUCE_MODES)
@pytest.mark.parametrize("nan", mq.NAN_MODES)
@pytest.mark.parametrize("xor16", mq.XOR16_MODES)
def test_gfx950_amax_e8m0_wide_blocks(vpl, reduce, nan, xor16):
    _require_gpu()
    if vpl == 2 and xor16 != "swizzle":
        pytest.skip("xor16 only applies to 32-lane blocks")
    x = make_input(1024, seed=40 + vpl)
    _, e = run_quant(x, vpl, "asm", reduce, nan, xor16)
    assert np.array_equal(e, mq.ref_e8m0_fp8(np.max(np.abs(x), axis=1)))


@pytest.mark.parametrize("nan", mq.NAN_MODES)
@pytest.mark.parametrize(
    "vpl,reduce,xor16",
    [
        (8, "dpp", "swizzle"),
        (8, "dpp_rowshift", "swizzle"),
        (1, "dpp", "permlane"),
        (1, "dpp_rowshift", "permlane"),
    ],
)
def test_gfx950_nan_and_inf_blocks(nan, vpl, reduce, xor16):
    """A NaN in a block gives E = 254 with ``nan="propagate"`` (the reference
    kernels' behaviour) and is dropped from the maximum with ``"ignore"``; an
    infinity gives E = 254 either way."""
    _require_gpu()
    x = make_input(64, seed=9)
    x[20:24] = 1.0
    x[20, 5] = np.nan
    x[21, 0] = np.inf
    x[22, 31] = -np.inf
    x[23, :] = np.nan
    _, e = run_quant(x, vpl, "asm", reduce, nan, xor16)
    assert e[21] == e[22] == 254
    assert e[20] == (
        254 if nan == "propagate" else int(mq.ref_e8m0_fp8(np.float32(1.0)))
    )
    finite = np.all(np.isfinite(x), axis=1)
    assert np.array_equal(e[finite], mq.ref_quantize_mxfp8(x[finite])[1])
