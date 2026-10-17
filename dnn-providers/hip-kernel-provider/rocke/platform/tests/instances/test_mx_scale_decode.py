# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""E8M0 scale decoder: all 256 codes, both engines, LLVM and HIP lowering.

``decode_mx_scale_e8m0`` turns a raw E8M0 byte into an f32 multiplier. The
reference is ``2^(code - 127)`` for codes 0..254 and NaN for code 255, as the
OCP MX E8M0 format defines. The oracle in ``mx_scale_oracle.py`` uses
``math.ldexp`` and the f32 bit pattern ``code << 23``. It never calls the
implementation's own conversion.

The decoder kernel in ``parity/mx_scale_decode_emit.py`` has no thread-id use.
A host compiler builds the emitted IR, or the HIP-lowered source, and the test
calls the result once for each of the 256 codes. This is host evidence. It
checks the arithmetic that each lowerer emits. It does not replace a GPU run.

Lanes:

* Python engine: static lowering checks, host LLVM build, host HIP build.
* Native engine: needs the engine archive and the public headers, so it runs
  from a source tree only. Set ``ROCKE_CPP_STRICT=1`` to fail when the lane
  cannot run, instead of skipping it.
* Host builds skip when no host compiler exists.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pytest

from rocke.core.lower_llvm import LLVM_FLAVORS

HERE = Path(__file__).resolve().parent
PLATFORM = HERE.parents[1]
PARITY = HERE / "parity"

sys.path.insert(0, str(HERE))
from mx_scale_oracle import (  # noqa: E402  (shared helper, not a test module)
    KERNEL,
    NAN_CODE,
    NUM_CODES,
    check_all_codes,
    dequantized_gemm,
    emitter,
    expected_bits,
    mismatches,
)


# ---------------------------------------------------------------------------
# Kernel builders and lowering (Python engine)
# ---------------------------------------------------------------------------


@cache
def python_ll(flavor: str) -> str:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(
        emitter().build_mx_scale_decode(), llvm_flavor=flavor, arch="gfx950"
    )


@cache
def python_hip() -> str:
    from rocke.core.lower_hip import lower_kernel_to_hip

    return lower_kernel_to_hip(emitter().build_mx_scale_decode(), arch="gfx950")


def legacy_ll(flavor: str) -> str:
    """LLVM text of the decoder as it was before the fix.

    The old decoder sign-extended the byte and mapped codes 0 and 255 to 0.0.
    It is the negative control for the oracle.
    """
    from rocke.core.ir import F32, I8, I32, IRBuilder, PtrType
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    b = IRBuilder(KERNEL)
    b.kernel.attrs["max_workgroup_size"] = 1
    codes = b.param("codes", PtrType(I8, "global"), readonly=True, align=1)
    out = b.param("out", PtrType(F32, "global"), writeonly=True, align=4)
    idx = b.const_i32(0)
    e_i32 = b.sext(b.global_load_i8(codes, idx), I32)
    is_zero = b.cmp_eq(e_i32, b.const_i32(0))
    is_nan = b.cmp_eq(e_i32, b.const_i32(255))
    exponent = b.fsub(b.sitofp_f32(e_i32), b.const_f32(127.0))
    scale = b.select(b.lor(is_zero, is_nan), b.const_f32(0.0), b.exp2(exponent))
    b.global_store(out, idx, scale, align=4)
    b.ret()
    return _lower_kernel_to_llvm_python(b.kernel, llvm_flavor=flavor, arch="gfx950")


def kernel_body(text: str) -> str:
    """The text of the ``rocke_mx_scale_decode`` function in a lowered module."""
    return text[text.index(KERNEL) :]


# ---------------------------------------------------------------------------
# Host builds
# ---------------------------------------------------------------------------


def _llvm_tool_dirs() -> list[Path]:
    """Directories that may hold LLVM tools, most specific first.

    ``ROCKE_LLVM_BIN`` is authoritative when set. Next come the ROCm root of the
    COMGR that rocke resolved, then ``ROCM_PATH`` / ``ROCM_HOME``.
    """
    explicit = os.environ.get("ROCKE_LLVM_BIN")
    if explicit:
        return [Path(explicit)]
    dirs = []
    try:
        from rocke.runtime.comgr import resolved_lib_path

        lib = resolved_lib_path()
    except Exception:
        lib = None
    if lib:
        library = Path(lib).resolve()
        dirs += [p / "llvm" / "bin" for p in library.parents[:4]]
    for var in ("ROCM_PATH", "ROCM_HOME"):
        root = os.environ.get(var)
        if root:
            dirs.append(Path(root) / "llvm" / "bin")
    return dirs


def _run(command, **kwargs):
    return subprocess.run(
        [str(c) for c in command],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        **kwargs,
    )


@cache
def host_clang() -> tuple[str, int] | None:
    """A clang that compiles LLVM IR for the host CPU, with its major version."""
    names = ["clang.exe"] if os.name == "nt" else ["clang"]
    candidates = [d / n for d in _llvm_tool_dirs() for n in names]
    on_path = shutil.which("clang")
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        banner = _run([candidate, "--version"])
        match = re.search(r"clang version (\d+)", banner.stdout)
        if banner.returncode != 0 or not match:
            continue
        return str(candidate), int(match.group(1))
    return None


@cache
def host_cxx() -> str | None:
    for name in ("c++", "g++", "clang++"):
        found = shutil.which(name)
        if found:
            return found
    return None


def flavor_major(flavor: str) -> int:
    return int(flavor.removeprefix("llvm"))


def host_ir(ll: str) -> str:
    """Rewrite the AMDGPU module so that a host clang accepts it.

    The rewrite drops the target lines, the kernel calling convention, and the
    function attributes. The instructions stay as the lowerer emitted them.
    """
    out = re.sub(r"^target (datalayout|triple) = .*\n", "", ll, flags=re.M)
    out, kernels = re.subn(
        r"^define amdgpu_kernel void", "define void", out, flags=re.M
    )
    out, groups = re.subn(
        r"^attributes #0 = \{.*\}$", "attributes #0 = { nounwind }", out, flags=re.M
    )
    assert (kernels, groups) == (1, 1), "unexpected module layout"
    return out


def build_host_llvm(ll: str, flavor: str, directory: Path) -> Path:
    """Compile the lowered IR for the host. Skip when the host clang is too old."""
    if sys.platform != "linux":
        pytest.skip("host builds use Linux shared libraries")
    found = host_clang()
    if found is None:
        pytest.skip("no host clang found (set ROCKE_LLVM_BIN or ROCM_PATH)")
    clang, major = found
    source = directory / "decoder_host.ll"
    library = directory / "decoder_host.so"
    source.write_text(host_ir(ll))
    # -O0 keeps every instruction that the lowerer emitted.
    result = _run(
        [clang, "-shared", "-fPIC", "-O0", "-Wno-override-module", "-x", "ir",
         source, "-o", library, "-lm"]
    )  # fmt: skip
    if result.returncode != 0:
        if flavor_major(flavor) > major:
            pytest.skip(f"host clang {major} cannot read {flavor} IR")
        pytest.fail(f"host build of the {flavor} IR failed:\n{result.stderr}")
    return library


def build_host_hip(hip: str, directory: Path) -> Path:
    """Compile the HIP-lowered kernel as plain C++ for the host.

    The build keeps the kernel function and drops the HIP prologue. The prologue
    declares device builtins that a host compiler cannot see.
    """
    if sys.platform != "linux":
        pytest.skip("host builds use Linux shared libraries")
    cxx = host_cxx()
    if cxx is None:
        pytest.skip("no host C++ compiler found")
    marker = 'extern "C" __global__'
    assert hip.count(marker) == 1, "expected one kernel in the HIP source"
    source = directory / "decoder_host.cpp"
    library = directory / "decoder_hip_host.so"
    source.write_text(
        "#include <math.h>\n#include <stdint.h>\n"
        "#define __global__\n#define __launch_bounds__(...)\n"
        + hip[hip.index(marker) :]
        + "\n"
    )
    result = _run([cxx, "-O0", "-shared", "-fPIC", source, "-o", library, "-lm"])
    assert (
        result.returncode == 0
    ), f"host build of the HIP source failed:\n{result.stderr}"
    return library


def decode_all_codes(library: Path) -> np.ndarray:
    """Call the host kernel once per code. Return the f32 bit patterns."""
    function = getattr(ctypes.CDLL(str(library)), KERNEL)
    function.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    function.restype = None
    bits = np.empty(NUM_CODES, dtype=np.uint32)
    for code in range(NUM_CODES):
        byte = ctypes.c_uint8(code)
        result = ctypes.c_uint32(0xDEADBEEF)
        function(ctypes.byref(byte), ctypes.byref(result))
        bits[code] = result.value
    return bits


# ---------------------------------------------------------------------------
# Oracle self-checks. The GPU lane trusts these helpers, so they run here.
# ---------------------------------------------------------------------------


def test_expected_bits_spot_values():
    assert expected_bits(0) == 0x00400000  # 2^-127, subnormal
    assert expected_bits(1) == 0x00800000  # 2^-126, smallest normal
    assert expected_bits(127) == 0x3F800000  # 1.0
    assert expected_bits(128) == 0x40000000  # 2.0
    assert expected_bits(254) == 0x7F000000  # 2^127
    assert expected_bits(NAN_CODE) is None
    assert expected_bits(0, flush_subnormals=True) == 0
    assert expected_bits(1, flush_subnormals=True) == 0x00800000


def test_gemm_reference_ticket_example():
    ones_a = np.ones((16, 32))
    ones_b = np.ones((32, 16))
    codes_a = np.full((16, 1), 128, np.uint8)
    codes_b = np.full((1, 16), 127, np.uint8)
    result = dequantized_gemm(ones_a, ones_b, codes_a, codes_b)
    assert result.dtype == np.float32 and (result == 64.0).all()


def test_gemm_reference_applies_scales_on_the_right_axes():
    ones_a = np.ones((2, 64))
    ones_b = np.ones((64, 2))
    # A codes: row 0 = [127, 128], row 1 = [126, 126].
    codes_a = np.array([[127, 128], [126, 126]], np.uint8)
    # B codes: column 0 = [127, 127], column 1 = [129, 126].
    codes_b = np.array([[127, 129], [127, 126]], np.uint8)
    result = dequantized_gemm(ones_a, ones_b, codes_a, codes_b)
    group = 32.0
    want = [
        [group * (1 * 1 + 2 * 1), group * (1 * 4 + 2 * 0.5)],
        [group * (0.5 * 1 + 0.5 * 1), group * (0.5 * 4 + 0.5 * 0.5)],
    ]
    assert result.tolist() == want


def test_gemm_reference_nan_scale_rules():
    ones_a = np.ones((2, 64))
    ones_b = np.ones((64, 2))
    finite_a = np.full((2, 2), 127, np.uint8)
    finite_b = np.full((2, 2), 127, np.uint8)
    # NaN in one A row poisons that row only.
    codes_a = finite_a.copy()
    codes_a[1, 0] = NAN_CODE
    result = dequantized_gemm(ones_a, ones_b, codes_a, finite_b)
    assert np.isfinite(result[0]).all() and np.isnan(result[1]).all()
    # NaN in one B column poisons that column only.
    codes_b = finite_b.copy()
    codes_b[1, 1] = NAN_CODE
    result = dequantized_gemm(ones_a, ones_b, finite_a, codes_b)
    assert np.isfinite(result[:, 0]).all() and np.isnan(result[:, 1]).all()


# ---------------------------------------------------------------------------
# Device compile (COMGR needs no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_decoder_and_mx_gemm_compile_for_cdna(arch):
    from rocke.helpers import compile_kernel
    from rocke.instances.common.mx_gemm import MxGemmSpec, build_mx_gemm
    from rocke.runtime.comgr import ComgrError

    kernels = [emitter().build_mx_scale_decode()]
    for mantissa in ("fp8e4m3", "bf8e5m2"):
        spec = MxGemmSpec(M=16, N=16, K=64, mantissa_dtype=mantissa)
        kernels.append(build_mx_gemm(spec, arch=arch))
    for kernel in kernels:
        try:
            artifact = compile_kernel(kernel, arch=arch)
        except ComgrError as exc:
            pytest.skip(f"COMGR is not usable: {exc}")
        assert artifact.hsaco, kernel.name


# ---------------------------------------------------------------------------
# Native engine
# ---------------------------------------------------------------------------


def _native_unavailable(reason: str):
    """Skip the native lane. Fail instead when ROCKE_CPP_STRICT=1."""
    if os.environ.get("ROCKE_CPP_STRICT") == "1":
        pytest.fail(f"native lane cannot run under ROCKE_CPP_STRICT=1: {reason}")
    pytest.skip(reason)


@pytest.fixture(scope="module")
def native_emitter(tmp_path_factory):
    """Build ``mx_scale_decode_emit.c`` against the production engine archive."""
    include = PLATFORM / "cpp" / "include"
    source = PARITY / "mx_scale_decode_emit.c"
    if sys.platform != "linux":
        _native_unavailable("the native emitter build is Linux-only")
    if not include.is_dir() or not source.is_file():
        _native_unavailable("an installed tree has no engine headers or C emitters")
    override = os.environ.get("ROCKE_TEST_ENGINE_ARCHIVE")
    archive = Path(override) if override else HERE.parent / "librocke_core.a"
    if not archive.is_file():
        _native_unavailable(
            "no engine archive; set ROCKE_TEST_ENGINE_ARCHIVE or use tests/run_all.py"
        )
    cxx = host_cxx()
    if cxx is None:
        _native_unavailable("no host C++ compiler found")
    executable = tmp_path_factory.mktemp("native-decoder") / "mx_scale_decode_emit"
    result = _run(
        [cxx, "-std=c++20", "-I", include, "-x", "c++", source, "-x", "none",
         archive, "-lm", "-ldl", "-pthread", "-o", executable]
    )  # fmt: skip
    assert result.returncode == 0, f"native emitter build failed:\n{result.stderr}"
    return executable


def native_output(executable: Path, mode: str, flavor: str | None = None) -> str:
    env = dict(os.environ)
    env.pop("ROCKE_LLVM_FLAVOR", None)
    if flavor:
        env["ROCKE_LLVM_FLAVOR"] = flavor
    result = _run([executable, "0", mode], env=env)
    assert result.returncode == 0, f"native {mode} emission failed:\n{result.stderr}"
    return result.stdout


# ---------------------------------------------------------------------------
# Python engine: static lowering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_llvm_zero_extends_and_has_no_zero_code(flavor):
    body = kernel_body(python_ll(flavor))
    assert re.search(r"= zext i8 %\S+ to i32", body)
    assert "sext" not in body
    # One compare, for the NaN code. The legacy decoder also compared with 0.
    assert body.count("icmp") == 1
    assert re.search(r"icmp eq i32 %\S+, 255\b", body)
    # The select returns a quiet NaN built from its bits, not 0.0.
    assert re.search(r"= bitcast i32 2143289344 to float", body)
    assert re.search(r"select i1 %\S+, float %\S+, float %\S+", body)
    assert "float 0x0000000000000000" not in body
    # The only float literal is 127.0. No NaN or infinity literal enters the IR.
    assert re.findall(r"0x[0-9A-Fa-f]{16}", body) == ["0x405FC00000000000"]


def test_hip_zero_extends_through_the_unsigned_type():
    body = kernel_body(python_hip())
    assert re.search(r"int zx\d+ = \(int\)\(uint8_t\)gl\d+;", body)
    assert not re.search(r"int sx\d+ =", body)
    assert re.search(r"int c\d+ = 2143289344;", body)  # the NaN bits
    assert "NAN" not in body


def test_exported_json_has_no_non_finite_number():
    """The portable-IR JSON cannot carry NaN or infinity, so the IR must not."""
    import json

    from rocke.core.ir_export import export_kernel_ir

    def reject(token):
        raise AssertionError(f"non-finite JSON number: {token}")

    text = json.dumps(export_kernel_ir(emitter().build_mx_scale_decode()))
    json.loads(text, parse_constant=reject)


def test_decoder_accepts_only_raw_i8_input():
    from rocke.core.ir import F32, I32, IRBuilder
    from rocke.helpers.mx_scale import decode_mx_scale_e8m0

    b = IRBuilder("reject")
    for ty in (I32, F32):
        with pytest.raises(ValueError, match="expects i8 input"):
            decode_mx_scale_e8m0(b, b.param(f"p_{ty.name}", ty))


# ---------------------------------------------------------------------------
# Python engine: host execution of every code
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_python_llvm_all_codes_match_reference(flavor, tmp_path):
    library = build_host_llvm(python_ll(flavor), flavor, tmp_path)
    check_all_codes(decode_all_codes(library))


def test_python_hip_all_codes_match_reference(tmp_path):
    check_all_codes(decode_all_codes(build_host_hip(python_hip(), tmp_path)))


def test_oracle_detects_the_signed_extension_defect(tmp_path):
    """The legacy decoder differs from the oracle in exactly 129 codes."""
    flavor = LLVM_FLAVORS[0]
    bits = decode_all_codes(build_host_llvm(legacy_ll(flavor), flavor, tmp_path))
    wrong = [code for code, _, _ in mismatches(bits)]
    assert wrong == [0, *range(128, NUM_CODES)]
    values = bits.view(np.float32)
    assert values[0] == 0.0  # code 0 decoded to zero
    assert values[128] == 0.0  # 2^1 decoded to zero
    assert not np.isnan(values[NAN_CODE])  # code 255 did not decode to NaN


# ---------------------------------------------------------------------------
# Native engine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_native_llvm_is_byte_identical_to_python(native_emitter, flavor):
    assert native_output(native_emitter, "ll", flavor) == python_ll(flavor)


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_native_llvm_all_codes_match_reference(native_emitter, flavor, tmp_path):
    ll = native_output(native_emitter, "ll", flavor)
    check_all_codes(decode_all_codes(build_host_llvm(ll, flavor, tmp_path)))


def test_native_hip_zero_extends_through_the_unsigned_type(native_emitter):
    # This test reads the cast line only. The native HIP lowerer prints f32
    # constants with %g, so 127.0 appears as `127f`, which a C++ compiler
    # rejects. That literal formatting is separate from the decoder.
    body = kernel_body(native_output(native_emitter, "hip"))
    assert re.search(r"int zx\d+ = \(int\)\(uint8_t\)gl\d+;", body)
    assert not re.search(r"int sx\d+ =", body)
    assert re.search(r"int c\d+ = 2143289344;", body)  # the NaN bits
    assert "NAN" not in body
