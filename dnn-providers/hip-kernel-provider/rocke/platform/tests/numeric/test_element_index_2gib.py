# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tensors with 2^31 or more elements must not be indexed through an i32 GEP.

Universal GEMM and the row-norm kernels address global memory with a flat
element index built from IRBuilder ``mul``/``add`` (``mul nsw`` / ``add nsw``
i32) and fed to ``getelementptr <ty>, ptr addrspace(1) %P, i32 %idx``:

- GEMM C: ``c_off = c_m * N + c_n`` (direct and cshuffle epilogues).
- GEMM A: ``row * K + k`` (B is ``col * K + k``, same shape of bug).
- rmsnorm2d / layernorm2d: ``row * N + col`` through the packed tensor view.

The GEP index is sign-extended to 64 bits (``v_ashrrev_i32 31`` in the ISA),
so once ``idx >= 2^31`` the access lands ``2^32`` elements *below* the tensor:
loads read foreign memory, stores corrupt it (or the GPU faults if that page
is unmapped). The dispatchers (``basic_gemm_request_errors``, norm
``_request_errors``) only check that the extents are positive, so these shapes
are admitted.

Predicted boundaries (all exact, all checked below):

- GEMM C, M=32768 N=65792: first bad element row 32640, column 32768.
- GEMM A, M=32768 K=65792: C rows 32640..32767 wrong.
- rmsnorm/layernorm, N=8192: rows >= 262144 wrong.

Each control is the same kernel with the largest extent that keeps every index
below 2^31 (N=65536, K=65536, rows=262144), which must be exact.

The CPU tests are fix-agnostic: the dispatcher may refuse the shape, or the
kernel may use 64-bit addressing; a 32-bit GEP on the base pointer is what
fails. The GPU tests run each case in a child process, because a fault
poisons the HIP context for every later case. Before the launch the child
allocates a guard region right after the tensor (HIP hands out descending
addresses, so it usually sits where the sign-extended index lands) and fills
it with 0x11 bytes; the report counts guard elements the kernel changed.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from rocke.dispatch import (
    GemmRequest,
    NormRequest,
    dispatch_gemm_bf16,
    dispatch_gemm_fp16,
    dispatch_norm,
)

_I32_LIMIT = 1 << 31
_GUARD_BYTES = 5 << 30  # covers [base - 4 GiB, base - 4 GiB + span) for every case

_BF16_ONE = 0x3F80
_BF16_MINUS_ONE = 0xBF80
_BF16_GUARD = 0x1111

# "bad" has more than 2^31 elements; "ctl" has exactly 2^31 (max index 2^31 - 1).
_GEMM_C = {"bad": (32768, 65792, 256), "ctl": (32768, 65536, 256)}  # (M, N, K)
_GEMM_A = {"bad": (32768, 256, 65792), "ctl": (32768, 256, 65536)}
_NORM = {"bad": (262400, 8192), "ctl": (262144, 8192)}  # (rows, cols)


# ---------------------------------------------------------------------------
# CPU: admitted shapes must not use an i32 GEP on the base pointer
# ---------------------------------------------------------------------------


def _i32_geps(ll: str, ptr: str) -> list[str]:
    pat = re.compile(rf"getelementptr inbounds \w+, ptr addrspace\(1\) %{ptr}, i32 ")
    return [line.strip() for line in ll.splitlines() if pat.search(line)]


def _lower(kernel, arch: str) -> str:
    from rocke.core.lower_llvm import lower_kernel_to_llvm

    return lower_kernel_to_llvm(kernel, arch=arch)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
@pytest.mark.parametrize(
    "dispatch", [dispatch_gemm_bf16, dispatch_gemm_fp16], ids=["bf16", "fp16"]
)
@pytest.mark.parametrize(
    "operand,dims",
    [("C", _GEMM_C["bad"]), ("A", _GEMM_A["bad"])],
    ids=["C_MxN", "A_MxK"],
)
def test_gemm_admitted_shape_has_no_i32_index(arch, dispatch, operand, dims):
    m, n, k = dims
    dtype = "bf16" if dispatch is dispatch_gemm_bf16 else "fp16"
    try:
        r = dispatch(GemmRequest(M=m, N=n, K=k, arch=arch, dtype=dtype))
    except ValueError:
        return  # refused: acceptable
    geps = _i32_geps(_lower(r.build(), arch), operand)
    extent = m * (n if operand == "C" else k)
    assert not geps, (
        f"{r.spec.name} admitted M={m} N={n} K={k} ({operand} has {extent} "
        f"elements >= 2^31) but indexes {operand} with an i32 GEP; indices "
        f">= 2^31 sign-extend to 2^32 elements below {operand}:\n  " + geps[0]
    )


@pytest.mark.parametrize("kind", ["rmsnorm", "layernorm"])
def test_norm_admitted_shape_has_no_i32_index(kind):
    rows, cols = _NORM["bad"]
    arch = "gfx942"
    try:
        r = dispatch_norm(
            NormRequest(rows=rows, cols=cols, arch=arch, kind=kind, dtype="bf16")
        )
    except ValueError:
        return
    ll = _lower(r.build(), arch)
    geps = _i32_geps(ll, "X") + _i32_geps(ll, "Y")
    assert not geps, (
        f"{kind} admitted rows={rows} cols={cols} ({rows * cols} elements >= "
        f"2^31) but indexes X/Y with an i32 GEP (row * N in i32):\n  " + geps[0]
    )


# ---------------------------------------------------------------------------
# GPU: run the admitted shape and check the boundary rows exactly
# ---------------------------------------------------------------------------


def _run_worker(case: str, variant: str) -> dict:
    from rocke.runtime.hip_module import get_device_arch

    if not get_device_arch(0):
        pytest.skip("requires a ROCm GPU")
    env = dict(os.environ)
    py_root = str(Path(__file__).resolve().parents[2] / "python")
    env["PYTHONPATH"] = py_root + os.pathsep + env.get("PYTHONPATH", "")
    env["ROCKE_BACKEND"] = "python"  # runtime behaviour only; engines are identical
    proc = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), case, variant],
        check=False,
        text=True,
        capture_output=True,
        timeout=1200,
        env=env,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    lines = [ln for ln in (proc.stdout or "").splitlines() if ln.startswith("{")]
    if not lines:
        pytest.fail(
            f"worker {case}/{variant} died (rc={proc.returncode}):\n{out[-3000:]}"
        )
    report = json.loads(lines[-1])
    if report["status"] == "skip":
        pytest.skip(report["reason"])
    return report


def _assert_clean(case: str, variant: str, report: dict) -> None:
    if report["status"] == "fault":
        pytest.fail(
            f"{case}/{variant}: GPU fault ({report['error']}); the sign-extended "
            f"index left the allocation"
        )
    assert report["status"] == "ok", report
    msg = (
        f"{case}/{variant} {report['shape']}: {report['bad']} wrong elements, "
        f"first at {report['first_bad']} (predicted {report['predicted']}); "
        f"{report['guard_hits']} guard elements changed 4 GiB below the tensor "
        f"(guard covers the wild range: {report['guard_covers']})"
    )
    assert report["bad"] == 0 and report["guard_hits"] == 0, msg


@pytest.mark.parametrize("variant", ["ctl", "bad"])
@pytest.mark.parametrize("case", ["gemm_c", "gemm_a", "rmsnorm", "layernorm"])
def test_gpu_index_crossing_2_31(case, variant):
    _assert_clean(case, variant, _run_worker(case, variant))


# ---------------------------------------------------------------------------
# Worker (child process)
# ---------------------------------------------------------------------------


def _bf16_rne(x: float) -> int:
    u = int(np.array([x], np.float32).view(np.uint32)[0])
    return (u + 0x7FFF + ((u >> 16) & 1)) >> 16


class _Device:
    def __init__(self):
        from rocke.runtime.hip_module import Runtime

        self.rt = Runtime()  # the child exits after one case; no frees

    def alloc(self, nbytes: int, fill: int | None = None) -> int:
        p = self.rt.alloc(nbytes)
        if fill is not None:
            self.rt.memset(p, fill, nbytes)
        return p

    def upload_tiled(self, dst: int, row, nbytes: int) -> None:
        """Fill ``nbytes`` at ``dst`` by repeating the u16 array ``row``."""
        from rocke.runtime.host_buffers import as_u8_buffer

        reps = max(1, (64 << 20) // row.nbytes)
        chunk = np.tile(row, reps)
        for off in range(0, nbytes, chunk.nbytes):
            n = min(chunk.nbytes, nbytes - off)
            self.rt.memcpy_h2d(dst + off, as_u8_buffer(chunk[: n // 2]), n)

    def read_u16(self, src: int, count: int):
        from rocke.runtime.host_buffers import as_u8_buffer

        out = np.empty(count, np.uint16)
        self.rt.memcpy_d2h(as_u8_buffer(out), src, out.nbytes)
        return out


def _scan(dev: _Device, base: int, cols: int, spans, expect) -> tuple[int, list | None]:
    """Count elements != ``expect`` (scalar or row) over row ranges [r0, r1)."""
    bad, first = 0, None
    for r0, r1 in spans:
        got = dev.read_u16(base + r0 * cols * 2, (r1 - r0) * cols).reshape(-1, cols)
        wrong = np.argwhere(got != expect)
        if wrong.size and first is None:
            first = [r0 + int(wrong[0][0]), int(wrong[0][1])]
        bad += len(wrong)
    return bad, first


def _guard_after(dev: _Device, base: int, span_elems: int) -> tuple[int, bool]:
    """Allocate the guard; return (wild_base, guard_covers_wild_range)."""
    guard = dev.alloc(_GUARD_BYTES, fill=0x11)
    wild = base - (1 << 32)  # element 2^31 of a bf16 tensor, sign-extended
    covers = guard <= wild and wild + 2 * span_elems <= guard + _GUARD_BYTES
    return wild, covers


def _guard_hits(dev: _Device, wild: int, covers: bool, span_elems: int) -> int:
    if not covers or span_elems == 0:
        return 0
    w = dev.read_u16(wild, span_elems)
    return int((w != _BF16_GUARD).sum())


def _worker_gemm(operand: str, variant: str) -> dict:
    from rocke.helpers import compile_kernel
    from rocke.runtime.hip_module import get_device_arch

    arch = get_device_arch(0)
    m, n, k = (_GEMM_C if operand == "C" else _GEMM_A)[variant]
    try:
        r = dispatch_gemm_bf16(GemmRequest(M=m, N=n, K=k, arch=arch, dtype="bf16"))
    except ValueError as e:
        return {"status": "skip", "reason": f"dispatch refused on {arch}: {e}"}
    art = compile_kernel(r.build(), arch=arch)
    dev = _Device()
    fn = dev.rt.load_module(art.hsaco).get_function(art.kernel_name)
    one = np.array([_BF16_ONE], np.uint16)
    # A is the big operand in the A case, C in the C case; the guard follows it.
    a_bytes, b_bytes, c_bytes = m * k * 2, n * k * 2, m * n * 2
    if operand == "C":
        c = dev.alloc(c_bytes, fill=0)
        span = max(0, m * n - _I32_LIMIT)
        wild, covers = _guard_after(dev, c, span)
        a, b = dev.alloc(a_bytes), dev.alloc(b_bytes)
    else:
        a = dev.alloc(a_bytes)
        span = 0  # loads only; wrong rows of C are the signal
        wild, covers = _guard_after(dev, a, max(0, m * k - _I32_LIMIT))
        b, c = dev.alloc(b_bytes), dev.alloc(c_bytes, fill=0)
    dev.upload_tiled(a, one, a_bytes)
    dev.upload_tiled(b, one, b_bytes)
    args = struct.pack("<QQQiii", a, b, c, m, n, k)
    dev.rt.launch(fn, tuple(r.grid), tuple(r.block), args)
    dev.rt.sync()

    expect = _bf16_rne(float(k))  # all-ones: every C element is exactly K
    crosses = m * (n if operand == "C" else k) > _I32_LIMIT
    if operand == "C":
        split_row = _I32_LIMIT // n
        predicted = [split_row, _I32_LIMIT - split_row * n]
        spans = [(0, 1), (min(split_row, m) - 1, m)]
    else:
        predicted = [_I32_LIMIT // k, 0]
        spans = [(0, m)]
    bad, first = _scan(dev, c, n, spans, expect)
    return {
        "status": "ok",
        "shape": {"M": m, "N": n, "K": k, "kernel": r.spec.name, "arch": arch},
        "bad": bad,
        "first_bad": first,
        "predicted": predicted if crosses else None,
        "guard_hits": _guard_hits(dev, wild, covers, span),
        "guard_covers": covers,
    }


def _worker_norm(kind: str, variant: str) -> dict:
    from rocke.helpers import compile_kernel
    from rocke.instances.common.layernorm2d import layernorm2d_signature
    from rocke.instances.common.rmsnorm2d import rmsnorm2d_signature
    from rocke.runtime.hip_module import get_device_arch
    from rocke.runtime.packing import pack_args

    arch = get_device_arch(0)
    rows, cols = _NORM[variant]
    try:
        r = dispatch_norm(
            NormRequest(rows=rows, cols=cols, arch=arch, kind=kind, dtype="bf16")
        )
    except ValueError as e:
        return {"status": "skip", "reason": f"dispatch refused on {arch}: {e}"}
    art = compile_kernel(r.build(), arch=arch)
    dev = _Device()
    fn = dev.rt.load_module(art.hsaco).get_function(art.kernel_name)
    nbytes = rows * cols * 2
    span = max(0, rows * cols - _I32_LIMIT)
    # X = +1, -1, +1, ...: mean 0, mean square 1, so Y = X / sqrt(1 + eps) = X
    # in bf16 for both kinds. Unwritten Y (0) and rows normalized from the
    # constant guard (Y = beta = 0 / tiny) both differ from +-1.
    x_row = np.tile(np.array([_BF16_ONE, _BF16_MINUS_ONE], np.uint16), cols // 2)
    y = dev.alloc(nbytes, fill=0)
    wild_y, covers_y = _guard_after(dev, y, span)
    x = dev.alloc(nbytes)
    _guard_after(dev, x, span)  # keeps the wild X reads on mapped memory
    dev.upload_tiled(x, x_row, nbytes)
    gamma = dev.alloc(cols * 2)
    dev.upload_tiled(gamma, np.array([_BF16_ONE], np.uint16), cols * 2)
    vals = {"X": x, "Gamma": gamma, "Y": y, "M": rows, "N": cols, "eps": 1e-6}
    if kind == "layernorm":
        vals["Beta"] = dev.alloc(cols * 2, fill=0)
        sig = layernorm2d_signature(r.spec)
    else:
        sig = rmsnorm2d_signature(r.spec)
    dev.rt.launch(fn, tuple(r.grid), tuple(r.block), pack_args(sig, vals))
    dev.rt.sync()

    split_row = _I32_LIMIT // cols
    bad, first = _scan(dev, y, cols, [(0, 1), (min(split_row, rows) - 1, rows)], x_row)
    return {
        "status": "ok",
        "shape": {"rows": rows, "cols": cols, "kernel": art.kernel_name, "arch": arch},
        "bad": bad,
        "first_bad": first,
        "predicted": [split_row, 0] if span else None,
        "guard_hits": _guard_hits(dev, wild_y, covers_y, span),
        "guard_covers": covers_y,
    }


def _worker(case: str, variant: str) -> dict:
    from rocke.runtime.hip_module import HipError

    try:
        if case == "gemm_c":
            return _worker_gemm("C", variant)
        if case == "gemm_a":
            return _worker_gemm("A", variant)
        return _worker_norm(case, variant)
    except HipError as e:
        if "out of memory" in str(e).lower() or "hipError(2)" in str(e):
            return {"status": "skip", "reason": f"not enough device memory: {e}"}
        return {"status": "fault", "error": str(e)}


if __name__ == "__main__":
    print(json.dumps(_worker(sys.argv[1], sys.argv[2])), flush=True)
