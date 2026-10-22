# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx950 scaled-MFMA compute paths (``kernels.gfx950.mx_mfma``).

CPU: the ``compute`` table, lane maps (bijective over the tile), operand
validation, the emitted call / asm per compute value and LLVM flavor.

gfx950 (skips elsewhere; ``ROCKE_REQUIRE_GFX950=1`` fails instead):

* every compute value x format pair x scale packing, bitwise against an exact
  fp64 reference on fixtures whose every partial sum is exact in fp32
  (unused scale-dword bytes hold random garbage);
* full-range random data, bitwise across compute values that issue the same
  instruction sequence (16x16x128 op vs asm; the three 32x32x64 routes), and
  within a stated tolerance of fp64;
* the 32x32 output against ``instances/gfx950/block_scaled_gemm`` bitwise;
* the ISA: the expected instruction per K step, ``cbsz``/``blgp`` formats,
  four-VGPR fp4 operands, and the asm route's ``op_sel`` encoding equal to the
  compiler's for every scale byte;
* mixed routes on one accumulator (compiler MFMA -> asm MFMA and back), the
  XDL-write hazard case of ``AsmHazard``.

Engines: ``test_mx_mfma_ir_cpp_python_byte_identity`` requires the C++ engine
to lower every dual-engine build (catalog and asm routes, probe and rate
kernels) to the Python engine's bytes at every golden flavor. Device tests
compile dual-engine builds with ``backend="both"`` when ``rocke_engine``
imports (``"python"`` otherwise; ``ROCKE_BACKEND=cpp|both`` makes engine
absence a failure), and any cpp -> python fallback fails the test.
"""

import ctypes
import os
import re
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest
from kernels.gfx950 import mx_mfma as K
from rocke.core.arch import mfma_scale_mx as mx
from rocke.core.ir import I32, IRBuilder, VectorType, uses_python_only_ops
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

FMT_PAIRS = [
    ("fp8", "fp4"),
    ("fp8", "fp8"),
    ("fp4", "fp4"),
    ("bf8", "fp4"),
    ("fp6", "fp4"),
    ("bf6", "fp8"),
    ("fp4", "fp8"),
]
_FP4 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])


# ----------------------------------------------------------------- engines


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


def _backend(compute, kernel):
    return K.backend_for(compute, kernel) or _dual_backend()


@pytest.fixture(autouse=True)
def _no_cpp_fallback():
    from rocke.core.backend import cpp_fallbacks, reset_cpp_fallbacks

    reset_cpp_fallbacks()
    yield
    assert cpp_fallbacks() == [], f"cpp -> python fallback: {cpp_fallbacks()}"


# ----------------------------------------------------------------- host model


def decode(codes, fmt):
    f = mx.format_code(fmt)
    c = np.asarray(codes).astype(np.int64)
    if f == 4:
        return _FP4[c]
    if f in (0, 1):
        sign = np.where(c & 0x80, -1.0, 1.0)
        eb, mb, bias = (4, 3, 7) if f == 0 else (5, 2, 15)
    else:
        sign = np.where(c & 0x20, -1.0, 1.0)
        eb, mb, bias = (2, 3, 1) if f == 2 else (3, 2, 3)
    e = (c >> mb) & ((1 << eb) - 1)
    m = c & ((1 << mb) - 1)
    val = np.where(
        e == 0, m / 2.0**mb * 2.0 ** (1 - bias), (1 + m / 2.0**mb) * 2.0 ** (e - bias)
    )
    return sign * val


def gen_codes(rng, shape, fmt, exact):
    f = mx.format_code(fmt)
    sign = rng.integers(0, 2, size=shape)
    if f == 4:
        return rng.integers(0, 16, size=shape)
    if f == 0:
        e = rng.integers(6, 9, size=shape) if exact else rng.integers(0, 16, size=shape)
        m = rng.integers(0, 8, size=shape)
        if not exact:
            m = np.where(e == 15, np.minimum(m, 6), m)  # 0x7F / 0xFF are NaN
        return (sign << 7) | (e << 3) | m
    if f == 1:
        e = (
            rng.integers(14, 17, size=shape)
            if exact
            else rng.integers(0, 31, size=shape)
        )
        return (sign << 7) | (e << 2) | rng.integers(0, 4, size=shape)
    if f == 2:
        return rng.integers(0, 64, size=shape)
    e = rng.integers(2, 5, size=shape) if exact else rng.integers(0, 8, size=shape)
    return (sign << 5) | (e << 2) | rng.integers(0, 4, size=shape)


def make_case(shape, fmt_a, fmt_b, k_iters, tiles, *, exact, seed):
    sh = mx.shape(shape)
    rng = np.random.default_rng(seed)
    kk = k_iters * sh.k
    a = gen_codes(rng, (tiles, sh.m, kk), fmt_a, exact)
    b = gen_codes(rng, (tiles, sh.n, kk), fmt_b, exact)
    if exact:
        sa = rng.integers(127, 129, size=(tiles, sh.m, kk // 32))
        sb = rng.integers(126, 128, size=(tiles, sh.n, kk // 32))
    else:
        sa = rng.integers(120, 135, size=(tiles, sh.m, kk // 32))
        sb = rng.integers(120, 135, size=(tiles, sh.n, kk // 32))
    return a, b, sa, sb


def reference(a, b, sa, sb, fmt_a, fmt_b):
    da = decode(a, fmt_a) * np.exp2(np.repeat(sa, 32, axis=-1) - 127.0)
    db = decode(b, fmt_b) * np.exp2(np.repeat(sb, 32, axis=-1) - 127.0)
    return np.einsum("tmk,tnk->tmn", da, db), np.einsum(
        "tmk,tnk->tmn", np.abs(da), np.abs(db)
    )


# ----------------------------------------------------------------- CPU


def test_compute_table():
    assert K.DEFAULT_COMPUTE == "mfma_scaled_32x32x64"
    assert {c.engines for c in K.COMPUTE_MODES.values()} == {"both", "python"}
    for name, c in K.COMPUTE_MODES.items():
        assert K.kernel_name_suffix(name) == (
            "_pyonly" if c.engines == "python" else ""
        )
        assert K.backend_for(name) == ("python" if c.engines == "python" else None)
        # The table agrees with the ops a build actually uses.
        k = K.build_mx_mfma_probe(K.MxMfmaProbeSpec(compute=name, opsel=c.opsel))
        assert bool(uses_python_only_ops(k)) == (c.engines == "python")
        assert k.name.endswith("_pyonly") == (c.engines == "python")
        assert K.backend_for(name, k) == K.backend_for(name)
    with pytest.raises(ValueError):
        K.compute_mode("nope")
    # A kernel that also uses another Python-only op goes to the Python engine.
    from kernels.gfx950 import mx_quant as Q
    from rocke.core.ir import F32, PtrType

    b = IRBuilder("mixed")
    px = b.param("X", PtrType(F32, "global"))
    vals = [b.global_load(px, b.const_i32(i), F32, align=4) for i in range(8)]
    Q.quantize_mxfp8(b, vals, 4, cvt="intrinsic")
    b.ret()
    assert K.backend_for("mfma_asm", b.kernel) == "python"
    with pytest.raises(ValueError, match="uses no Python-only op"):
        K.backend_for("mfma_16x16x128_op", IRBuilder("empty").kernel)


@pytest.mark.parametrize("shape", sorted(mx.SHAPES))
@pytest.mark.parametrize("fmt", ["fp8", "fp6", "fp4"])
def test_lane_maps_bijective(shape, fmt):
    sh = mx.shape(shape)
    A = {
        mx.a_coord(mx.HostArith, shape, fmt, l, s) for l in range(64) for s in range(32)
    }
    B = {
        mx.b_coord(mx.HostArith, shape, fmt, l, s) for l in range(64) for s in range(32)
    }
    assert A == {(r, k) for r in range(sh.m) for k in range(sh.k)}
    assert B == {(k, c) for k in range(sh.k) for c in range(sh.n)}
    C = [
        mx.c_coord(mx.HostArith, shape, l, i)
        for l in range(64)
        for i in range(sh.c_len)
    ]
    assert sorted(C) == [(r, c) for r in range(sh.m) for c in range(sh.n)]
    S = {mx.scale_a_coord(mx.HostArith, shape, l) for l in range(64)}
    assert S == {(r, kb) for r in range(sh.m) for kb in range(sh.k_blocks)}
    # Scale lanes of K block g are the lanes whose A data starts in block g
    # for the non-fp8 formats (contiguous 32-K runs).
    if mx.FORMAT_BITS[mx.format_code(fmt)] != 8:
        for l in range(64):
            _, k0 = mx.a_coord(mx.HostArith, shape, fmt, l, 0)
            assert k0 // 32 == mx.scale_a_coord(mx.HostArith, shape, l)[1]


def test_op_validation():
    b = IRBuilder("v")
    v8 = b.vec_bitcast(b.zero_vec_f32(8), VectorType(I32, 8))
    v4 = b.vec_bitcast(b.zero_vec_f32(4), VectorType(I32, 4))
    s = b.const_i32(127)
    with pytest.raises(ValueError, match="expects operands"):
        b.mfma_scale_mx(v8, v8, b.zero_vec_f32(4), s, s, fmt_b="fp4")
    with pytest.raises(ValueError, match="opsel_a"):
        b.mfma_scale_mx(v8, v4, b.zero_vec_f32(4), s, s, opsel_a=4)
    with pytest.raises(ValueError, match="shape"):
        b.mfma_scale_mx(v8, v4, b.zero_vec_f32(4), s, s, shape="16x16x64")
    with pytest.raises(ValueError):
        b.mfma_scale_mx(v8, v4, b.zero_vec_f32(4), s, s, shape="32x32x64")
    assert (
        b.mfma_scale_mx(v8, v4, b.zero_vec_f32(16), s, s, shape="32x32x64").type.name
        == "vec<f32x16>"
    )
    acc = b.mfma_scale_mx(v8, v4, b.zero_vec_f32(4), s, s, opsel_a=2, opsel_b=1)
    assert acc.type.name == "vec<f32x4>"


@pytest.mark.parametrize("compute", sorted(K.COMPUTE_MODES))
@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
def test_emission(compute, flavor):
    mode = K.compute_mode(compute)
    spec = K.MxMfmaProbeSpec(k_iters=5, compute=compute, opsel=mode.opsel)
    ir = _lower_kernel_to_llvm_python(
        K.build_mx_mfma_probe(spec, arch="gfx950"), arch="gfx950", llvm_flavor=flavor
    )
    if mode.route == "op":
        calls = re.findall(
            r"call <\d+ x float> @(llvm\.amdgcn\.mfma\.scale[^(]*)\((.*)\)", ir
        )
        assert len(calls) == 5
        want = (
            "16x16x128.f8f6f4.v8i32.v4i32"
            if mode.shape == "16x16x128"
            else "32x32x64.f8f6f4.v8i32.v4i32"
        )
        assert all(c[0].endswith(want) for c in calls)
        sels = [re.findall(r"i32 (\d+), i32 %", c[1]) for c in calls]
        assert [s[0] for s in sels] == ["0", "1", "2", "3", "0"]  # opsel_a per K step
        assert ", i32 0, i32 4, " in calls[0][1]  # cbsz fp8, blgp fp4
    elif mode.route == "catalog":
        assert ir.count("call <16 x float> @llvm.amdgcn.mfma.scale.f32.32x32x64") == 5
    else:
        assert ir.count(f"v_mfma_scale_f32_{mode.shape}_f8f6f4") == 5
        assert "blgp:4" in ir and "op_sel_hi:[1,1,0]" in ir


def test_asm_template():
    t = K.asm_template("16x16x128", 0, 4, [(0, 0), (3, 1)], K.AsmHazard())
    lines = t.split("\n\t")
    assert lines[0] == "s_nop 1" and lines[-1] == "s_nop 15"
    assert lines[1] == (
        "v_mfma_scale_f32_16x16x128_f8f6f4 $0, $2, $3, $0, $4, $5 "
        "op_sel:[0,0,0] op_sel_hi:[0,0,0] blgp:4"
    )
    assert (
        "op_sel:[1,1,0] op_sel_hi:[1,0,0]" in lines[2]
        and "$6, $7, $0, $8, $9" in lines[2]
    )
    with pytest.raises(ValueError):
        K.asm_template("16x16x128", 0, 4, [(0, 0)], K.AsmHazard(lead=(16,)))
    # XDL-written sources raise the lead to 24 wait states.
    t = K.asm_template("16x16x128", 0, 4, [(0, 0)], K.AsmHazard(), xdl_source=True)
    assert t.split("\n\t")[:2] == ["s_nop 15", "s_nop 7"]
    assert K.AsmHazard(after_xdl=True).lead_for() == (15, 7)
    assert K.AsmHazard(auto_xdl=False).lead_for(True) == (1,)
    assert K.AsmHazard(lead=(15, 15)).lead_for(True) == (15, 15)


@pytest.mark.parametrize(
    "pair",
    [
        ("mfma_16x16x128_op", "mfma_asm"),
        ("mfma_scaled_32x32x64", "mfma_asm_32x32x64"),
    ],
    ids=["16x16x128", "32x32x64"],
)
def test_mixed_route_lead_padding(pair):
    """An asm chain whose accumulator a compiler MFMA just wrote gets the XDL
    lead; one following an asm chain keeps the VALU lead."""
    compute, alt = pair
    opsel = K.compute_mode(compute).opsel
    spec = K.MxMfmaProbeSpec(
        k_iters=4, compute=compute, compute_alt=alt, chain=1, opsel=opsel
    )
    ir = _lower_kernel_to_llvm_python(
        K.build_mx_mfma_probe(spec), arch="gfx950", llvm_flavor="llvm22"
    )
    asm = re.findall(r'asm sideeffect "([^"]*)"', ir)
    asm = [a for a in asm if "v_mfma_scale" in a]
    assert len(asm) == 2
    assert all(a.startswith(r"s_nop 15\0A\09s_nop 7\0A\09v_mfma") for a in asm)
    spec = K.MxMfmaProbeSpec(
        k_iters=4, compute=alt, compute_alt=compute, chain=1, opsel=opsel
    )
    ir = _lower_kernel_to_llvm_python(
        K.build_mx_mfma_probe(spec), arch="gfx950", llvm_flavor="llvm22"
    )
    asm = [a for a in re.findall(r'asm sideeffect "([^"]*)"', ir) if "v_mfma" in a]
    # First chain reads a zero accumulator (VALU lead); the third follows an
    # op-route MFMA (XDL lead).
    assert asm[0].startswith(r"s_nop 1\0A\09v_mfma")
    assert asm[1].startswith(r"s_nop 15\0A\09s_nop 7\0A\09v_mfma")


def _parity_cases():
    """Every dual-engine build: name -> builder."""
    cases = {}
    for compute, mode in sorted(K.COMPUTE_MODES.items()):
        if mode.engines != "both":
            continue
        for fmts in FMT_PAIRS:
            for opsel in (False, True) if mode.opsel else (False,):
                spec = K.MxMfmaProbeSpec(
                    k_iters=5,
                    compute=compute,
                    fmt_a=fmts[0],
                    fmt_b=fmts[1],
                    opsel=opsel,
                )
                cases[spec.kernel_name()] = lambda spec=spec: K.build_mx_mfma_probe(
                    spec
                )
        extra = [K.MxMfmaProbeSpec(k_iters=6, compute=compute, valu_operands=True)]
        if mode.route == "asm":
            extra += [
                K.MxMfmaProbeSpec(k_iters=6, compute=compute, opsel=True, chain=1),
                K.MxMfmaProbeSpec(
                    k_iters=6,
                    compute=compute,
                    opsel=True,
                    hazard=K.AsmHazard(src="a"),
                    name="mx_mfma_probe_agpr",
                ),
            ]
        for spec in extra:
            cases[spec.kernel_name()] = lambda spec=spec: K.build_mx_mfma_probe(spec)
        for n_acc in (1, 4):
            rs = K.MxMfmaRateSpec(compute=compute, n_acc=n_acc, iters=8)
            cases[rs.kernel_name()] = lambda rs=rs: K.build_mx_mfma_rate(rs)
    for pair in (("mfma_scaled_32x32x64", "mfma_asm_32x32x64"),):
        spec = K.MxMfmaProbeSpec(
            k_iters=4, compute=pair[0], compute_alt=pair[1], chain=1
        )
        cases[spec.kernel_name()] = lambda spec=spec: K.build_mx_mfma_probe(spec)
    return cases


def test_mx_mfma_ir_cpp_python_byte_identity():
    """The C++ engine lowers every dual-engine build to the Python engine's
    bytes at every golden flavor; the Python-only builds are excluded (they
    use ``PYTHON_ONLY_OPCODES``). Engine absence is a skip on a CPU run and a
    failure under ``ROCKE_BACKEND=cpp|both``."""
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
    assert not mism, "mx_mfma C++/Python IR mismatch:\n  " + "\n  ".join(mism)
    assert compared == len(cases) * len(GOLDEN_FLAVORS)


# ----------------------------------------------------------------- gfx950


def _require_gfx950():
    strict = os.environ.get("ROCKE_REQUIRE_GFX950") == "1"
    if not Path("/dev/kfd").exists():
        if strict:
            pytest.fail("requires a gfx950 GPU")
        pytest.skip("requires an AMD GPU")
    from rocke.runtime.hip_module import get_device_arch

    arch = get_device_arch(0)
    if arch != "gfx950":
        if strict:
            pytest.fail(f"requires gfx950, found {arch}")
        pytest.skip(f"requires gfx950, found {arch}")


def disassemble(hsaco):
    tool = shutil.which("llvm-objdump")
    if tool is None:
        pytest.skip("llvm-objdump is required for the ISA checks")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "k.hsaco"
        path.write_bytes(hsaco)
        out = subprocess.run(
            [tool, "-d", str(path)], check=False, capture_output=True, text=True
        )
    assert out.returncode == 0, out.stderr
    return out.stdout


def compile_probe(spec):
    from rocke.helpers import compile_kernel

    kernel = K.build_mx_mfma_probe(spec, arch="gfx950")
    return compile_kernel(kernel, arch="gfx950", backend=_backend(spec.compute, kernel))


def run_probe(spec, a, b, sa, sb, art=None):
    """Launch the probe over ``tiles`` tiles; return C ``[tiles, M, N]`` (f32)."""
    from rocke.runtime.hip_module import Runtime

    mode = K.compute_mode(spec.compute)
    sh = mode.mx_shape
    tiles = a.shape[0]
    A = np.stack(
        [K.pack_a(a[t], sh.name, spec.fmt_a, spec.k_iters) for t in range(tiles)]
    )
    B = np.stack(
        [K.pack_b(b[t], sh.name, spec.fmt_b, spec.k_iters) for t in range(tiles)]
    )
    SA = np.stack(
        [
            K.pack_scales(sa[t], sh.name, spec.k_iters, opsel=spec.opsel)
            for t in range(tiles)
        ]
    )
    SB = np.stack(
        [
            K.pack_scales(sb[t], sh.name, spec.k_iters, opsel=spec.opsel)
            for t in range(tiles)
        ]
    )
    art = art or compile_probe(spec)
    rt = Runtime()
    module = rt.load_module(art.hsaco)
    nc = tiles * 64 * sh.c_len
    bufs = []
    try:
        ptrs = []
        for arr in (A, B, SA, SB):
            if arr.shape[-1] == 6:
                # A <6 x i32> element occupies 32 bytes in LLVM's layout.
                arr = np.concatenate(
                    [arr, np.zeros(arr.shape[:-1] + (2,), arr.dtype)], -1
                )
            arr = np.ascontiguousarray(arr.astype(np.uint32))
            p = rt.alloc(arr.nbytes)
            bufs.append(p)
            rt.memcpy_h2d(
                p,
                (ctypes.c_uint8 * arr.nbytes).from_buffer_copy(arr.tobytes()),
                arr.nbytes,
            )
            ptrs.append(p)
        pc = rt.alloc(nc * 4)
        bufs.append(pc)
        rt.memset(pc, 0xFF, nc * 4)
        rt.launch(
            module.get_function(art.kernel_name),
            (tiles, 1, 1),
            (64, 1, 1),
            struct.pack("<QQQQQ", *ptrs, pc)
            + (struct.pack("<i", 0) if spec.valu_operands else b""),
        )
        rt.sync()
        host = (ctypes.c_uint8 * (nc * 4))()
        rt.memcpy_d2h(host, pc, nc * 4)
    finally:
        for p in bufs:
            rt.free(p)
        module.unload()
    acc = np.frombuffer(bytes(host), np.float32).reshape(tiles, 64, sh.c_len)
    return np.stack([K.unpack_acc(acc[t], sh.name) for t in range(tiles)])


def _k_iters(compute):
    return 5 if K.compute_mode(compute).shape == "16x16x128" else 6


@pytest.mark.parametrize("compute", sorted(K.COMPUTE_MODES))
@pytest.mark.parametrize("fmts", FMT_PAIRS, ids=lambda p: f"{p[0]}x{p[1]}")
@pytest.mark.parametrize("opsel", [False, True])
def test_gfx950_exact_bitwise(compute, fmts, opsel):
    _require_gfx950()
    mode = K.compute_mode(compute)
    if opsel and not mode.opsel:
        pytest.skip(f"{compute} reads scale byte 0 only")
    spec = K.MxMfmaProbeSpec(
        k_iters=_k_iters(compute),
        compute=compute,
        fmt_a=fmts[0],
        fmt_b=fmts[1],
        opsel=opsel,
    )
    a, b, sa, sb = make_case(mode.shape, *fmts, spec.k_iters, 3, exact=True, seed=11)
    got = run_probe(spec, a, b, sa, sb)
    want, _ = reference(a, b, sa, sb, *fmts)
    assert np.array_equal(want.astype(np.float32).astype(np.float64), want), (
        "fixture not exact"
    )
    bad = np.argwhere(got != want.astype(np.float32))
    assert bad.size == 0, (
        f"{len(bad)} mismatches, first {bad[:4].tolist()}: {got[tuple(bad[:4].T)]} vs {want[tuple(bad[:4].T)]}"
    )


@pytest.mark.parametrize(
    "group",
    [
        ("mfma_16x16x128_op", "mfma_asm"),
        ("mfma_scaled_32x32x64", "mfma_32x32x64_op", "mfma_asm_32x32x64"),
    ],
    ids=["16x16x128", "32x32x64"],
)
@pytest.mark.parametrize(
    "fmts",
    [("fp8", "fp4"), ("fp8", "fp8"), ("fp4", "fp4")],
    ids=lambda p: f"{p[0]}x{p[1]}",
)
def test_gfx950_random_cross_route_bitwise(group, fmts):
    _require_gfx950()
    shape = K.compute_mode(group[0]).shape
    k_iters = 8
    a, b, sa, sb = make_case(shape, *fmts, k_iters, 4, exact=False, seed=5)
    want, mag = reference(a, b, sa, sb, *fmts)
    results = {}
    for compute in group:
        for chain in (1, 4) if K.compute_mode(compute).route == "asm" else (4,):
            spec = K.MxMfmaProbeSpec(
                k_iters=k_iters,
                compute=compute,
                fmt_a=fmts[0],
                fmt_b=fmts[1],
                chain=chain,
            )
            results[(compute, chain)] = run_probe(spec, a, b, sa, sb)
    first = next(iter(results.values()))
    for key, got in results.items():
        assert np.array_equal(got, first), f"{key} differs from {next(iter(results))}"
        err = np.abs(got.astype(np.float64) - want)
        # The f8f6f4 MFMA does not keep full fp32 precision inside one
        # instruction's dot product (on gfx950 the worst case observed over
        # these fixtures is about 2^-12 of the magnitude sum), so the bound
        # is 2^-11 of the magnitude sum, not K fp32 ulps.
        assert np.all(err <= mag * 2.0**-11 + 1e-30), (key, (err / mag).max())


def test_gfx950_vs_block_scaled_gemm():
    _require_gfx950()
    ml = pytest.importorskip("ml_dtypes")
    from rocke.examples.common.scaled_gemm_reference import pack_fp4_codes
    from rocke.examples.gfx950.gemm import block_scaled_gemm_verify as bsv
    from rocke.helpers import compile_kernel
    from rocke.instances.gfx950.block_scaled_gemm import (
        BlockScaledGemmSpec,
        build_block_scaled_gemm,
    )
    from rocke.runtime.hip_module import Runtime

    M = N = 32
    kk = 256
    rng = np.random.default_rng(3)
    a = gen_codes(rng, (M, kk), "fp8", True).astype(np.uint8)
    b = gen_codes(rng, (N, kk), "fp4", True).astype(np.uint8)
    sa = rng.integers(127, 129, size=(M, kk // 32)).astype(np.uint8)
    sb = rng.integers(126, 128, size=(N, kk // 32)).astype(np.uint8)
    spec = BlockScaledGemmSpec("mx_mfma_xcheck", M, N, kk, "fp8", "fp4", "bf16")
    art = compile_kernel(
        build_block_scaled_gemm(spec), arch="gfx950", backend=_dual_backend()
    )
    rt = Runtime()
    module = rt.load_module(art.hsaco)
    try:
        bsg = bsv._launch(
            rt,
            module.get_function(art.kernel_name),
            spec,
            (a, pack_fp4_codes(b), sa, np.ascontiguousarray(sb.T)),
        )
    finally:
        module.unload()
    for compute in sorted(K.COMPUTE_MODES):
        mode = K.compute_mode(compute)
        sh = mode.mx_shape
        k_iters = kk // sh.k
        spec_p = K.MxMfmaProbeSpec(k_iters=k_iters, compute=compute, opsel=mode.opsel)
        t = M // sh.m
        # Tiles (i, j) of the 32x32 output, row-major.
        at = np.stack(
            [a[i * sh.m : (i + 1) * sh.m] for i in range(t) for j in range(t)]
        )
        bt = np.stack(
            [b[j * sh.n : (j + 1) * sh.n] for i in range(t) for j in range(t)]
        )
        sat = np.stack(
            [sa[i * sh.m : (i + 1) * sh.m] for i in range(t) for j in range(t)]
        )
        sbt = np.stack(
            [sb[j * sh.n : (j + 1) * sh.n] for i in range(t) for j in range(t)]
        )
        tiles = run_probe(spec_p, at, bt, sat, sbt)
        c = np.zeros((M, N), np.float32)
        for idx, (i, j) in enumerate((i, j) for i in range(t) for j in range(t)):
            c[i * sh.m : (i + 1) * sh.m, j * sh.n : (j + 1) * sh.n] = tiles[idx]
        mine = c.astype(ml.bfloat16).astype(np.float32)
        assert np.array_equal(mine, bsg), f"{compute} differs from block_scaled_gemm"


@pytest.mark.parametrize("compute", sorted(K.COMPUTE_MODES))
def test_gfx950_isa(compute):
    _require_gfx950()
    mode = K.compute_mode(compute)
    spec = K.MxMfmaProbeSpec(k_iters=5, compute=compute, opsel=mode.opsel)
    isa = disassemble(compile_probe(spec).hsaco)
    lines = [
        l for l in isa.splitlines() if f"v_mfma_scale_f32_{mode.shape}_f8f6f4" in l
    ]
    assert len(lines) == 5, isa
    for i, line in enumerate(lines):
        assert "blgp:4" in line and "cbsz" not in line, line
        tuples = re.findall(r"(?:a|v)\[(\d+):(\d+)\]", line)
        widths = [int(h) - int(l) + 1 for l, h in tuples[:3]]
        # fp4 operands are four VGPRs on every route (the backend narrows the
        # catalog route's padded eight-word operand).
        assert widths == [mode.mx_shape.c_len, 8, 4], line


def test_gfx950_asm_opsel_matches_compiler():
    """The asm route's op_sel / op_sel_hi spelling equals the compiler's for
    every scale byte (the disassembler prints both the same way)."""
    _require_gfx950()

    def mods(compute):
        spec = K.MxMfmaProbeSpec(k_iters=4, compute=compute, opsel=True)
        isa = disassemble(compile_probe(spec).hsaco)
        out = []
        for line in isa.splitlines():
            if "v_mfma_scale_f32_16x16x128_f8f6f4" in line:
                out.append(
                    (re.search(r"op_sel:\[[^\]]*\]", line) or [""])[0]
                    + " "
                    + (re.search(r"op_sel_hi:\[[^\]]*\]", line) or [""])[0]
                )
        return out

    assert mods("mfma_16x16x128_op") == mods("mfma_asm")


@pytest.mark.parametrize("compute", ["mfma_asm", "mfma_asm_32x32x64"])
@pytest.mark.parametrize(
    "lead",
    [
        (1,),
        pytest.param(
            (),
            marks=pytest.mark.xfail(
                strict=False,
                reason="VALU write -> inline-asm MFMA read hazard: the compiler "
                "does not pad it, and gfx950 returned wrong results without "
                "the lead s_nop",
            ),
        ),
    ],
    ids=["lead_nop", "no_lead_nop"],
)
def test_gfx950_asm_valu_written_operands(compute, lead):
    """asm MFMAs reading registers a VALU wrote just before, bitwise against
    the exact reference. The default ``AsmHazard.lead`` padding is required:
    without it the result is wrong (the ``no_lead_nop`` case documents that)."""
    _require_gfx950()
    mode = K.compute_mode(compute)
    hazard = K.AsmHazard(lead=lead)
    spec = K.MxMfmaProbeSpec(
        k_iters=_k_iters(compute),
        compute=compute,
        opsel=True,
        chain=1,
        hazard=hazard,
        valu_operands=True,
    )
    a, b, sa, sb = make_case(
        mode.shape, "fp8", "fp4", spec.k_iters, 3, exact=True, seed=17
    )
    got = run_probe(spec, a, b, sa, sb)
    want, _ = reference(a, b, sa, sb, "fp8", "fp4")
    assert np.array_equal(got, want.astype(np.float32))


def test_gfx950_asm_agpr_sources():
    """``AsmHazard(src="a")`` (A/B operands in AGPRs) is bitwise equal to the
    VGPR-source asm route on full-range random data."""
    _require_gfx950()
    a, b, sa, sb = make_case("16x16x128", "fp8", "fp4", 8, 3, exact=False, seed=23)
    out = [
        run_probe(
            K.MxMfmaProbeSpec(k_iters=8, compute="mfma_asm", opsel=True, hazard=h),
            a,
            b,
            sa,
            sb,
        )
        for h in (K.AsmHazard(), K.AsmHazard(src="a"))
    ]
    assert np.array_equal(out[0], out[1])


@pytest.mark.parametrize(
    "pair",
    [
        ("mfma_16x16x128_op", "mfma_asm"),
        ("mfma_asm", "mfma_16x16x128_op"),
        ("mfma_scaled_32x32x64", "mfma_asm_32x32x64"),
        ("mfma_asm_32x32x64", "mfma_scaled_32x32x64"),
        ("mfma_32x32x64_op", "mfma_asm_32x32x64"),
    ],
    ids=lambda p: f"{p[0]}-then-{p[1]}",
)
@pytest.mark.parametrize(
    "hazard",
    [
        K.AsmHazard(),
        pytest.param(
            K.AsmHazard(auto_xdl=False),
            marks=pytest.mark.xfail(
                strict=False,
                reason="compiler MFMA write -> inline-asm MFMA read with only the "
                "VALU lead: the hazard recognizer cannot see into the asm; "
                "observed correct on gfx950 (XPASS) because the compiler's AGPR -> "
                "VGPR accumulator copies already supply the wait states; kept as "
                "a witness for layouts without those copies",
            ),
        ),
    ],
    ids=["xdl_lead", "valu_lead_only"],
)
def test_gfx950_mixed_routes_one_accumulator(pair, hazard):
    """Routes alternate per K step on one accumulator (compiler MFMA writes
    the accumulator, an asm MFMA reads it right after, and the reverse),
    bitwise equal to the single-route result on full-range random data."""
    _require_gfx950()
    compute, alt = pair
    mode = K.compute_mode(compute)
    opsel = mode.opsel and K.compute_mode(alt).opsel
    k_iters = 8
    a, b, sa, sb = make_case(mode.shape, "fp8", "fp4", k_iters, 3, exact=False, seed=29)
    spec = K.MxMfmaProbeSpec(
        k_iters=k_iters,
        compute=compute,
        compute_alt=alt,
        chain=1,
        opsel=opsel,
        hazard=hazard,
    )
    got = run_probe(spec, a, b, sa, sb)
    ref = run_probe(
        K.MxMfmaProbeSpec(k_iters=k_iters, compute=compute, opsel=opsel), a, b, sa, sb
    )
    assert np.array_equal(got, ref)
