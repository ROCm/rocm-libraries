# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Register Geko-tuned kernels with a running hipBLASLt, through ctypes.

Experimental. Needs a libhipblaslt.so built with HIPBLASLT_ENABLE_USER_KERNELS=ON,
which exports the entry points in hipblaslt/hipblaslt-user-kernel.h. No hipBLASLt
Python bindings are involved: every call goes straight into the shared library.

The user kernel tier is process-global, so a kernel registered here is selected by
every hipBLASLt handle in the process, PyTorch's included, provided this module calls
into the same libhipblaslt.so the process already has loaded. ``HipBLASLt()`` makes
sure of that: if a libhipblaslt.so is already mapped it is the one used, and asking
for a different file is an error rather than a silent second copy.

Registrations last for the process unless ``HIPBLASLT_USER_KERNEL_LIBRARY`` names
an absolute directory, a user kernel library. hipBLASLt reads the variable once, the
first time it initializes a device, so it must be set before then; ``UserKernels``
sets it when given ``library`` and ``register_workdir`` checks that registrations were
saved. Every later process that names the directory loads the saved kernels and
mappings when it initializes the device, with no API call at all.

From a shell::

    python -m geko.user_kernels info [--lib PATH]
    HIPBLASLT_USER_KERNEL_LIBRARY=/abs/dir python -m geko.user_kernels register WORKDIR
    HIPBLASLT_USER_KERNEL_LIBRARY=/abs/dir python -m geko.user_kernels status

To run a workload, e.g. PyTorch, with what ``register`` saved::

    HIPBLASLT_USER_KERNEL_LIBRARY=/abs/dir python train.py

A process that is already running with the variable set picks up kernels saved since
it started with ``UserKernels().refresh()``.

A payload is the ``(<stem>.dat[.zlib], <stem>.co)`` pair Geko builds under
``<workdir>/build/library/<arch>/``. Its matching table lists the ``[M, N, batch, K]``
each kernel was tuned for, which is what ``register_workdir`` maps.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import csv
import json
import os
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

HIPBLAS_STATUS = [
    "SUCCESS", "NOT_INITIALIZED", "ALLOC_FAILED", "INVALID_VALUE", "MAPPING_ERROR",
    "EXECUTION_FAILED", "INTERNAL_ERROR", "NOT_SUPPORTED", "ARCH_MISMATCH",
    "HANDLE_IS_NULLPTR", "INVALID_ENUM", "UNKNOWN",
]

# Tensile type name -> (hipDataType, Geko type name, numpy dtype or None, bytes).
TENSILE_TYPES = {
    "Float": (0, "f32_r", "float32", 4),
    "Half": (2, "f16_r", "float16", 2),
    "BFloat16": (14, "bf16_r", "bfloat16", 2),
    "Float8": (28, "f8_r", None, 1),
    "BFloat8": (29, "bf8_r", None, 1),
    "Float8_fnuz": (1000, "f8_fnuz_r", None, 1),
    "BFloat8_fnuz": (1001, "bf8_fnuz_r", None, 1),
}
HIP_R_32F = 0
HIPBLAS_COMPUTE_32F = 2
HIPBLAS_OP_N, HIPBLAS_OP_T = 111, 112
HIPBLASLT_MATMUL_DESC_TRANSA, HIPBLASLT_MATMUL_DESC_TRANSB = 0, 1
HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 0, 1
HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1
HIP_MEMCPY_H2D, HIP_MEMCPY_D2H = 1, 2

LIBRARY_ENV = "HIPBLASLT_USER_KERNEL_LIBRARY"
USER_KERNEL_SYMBOLS = (
    "hipblasLtUserKernelRefresh", "hipblasLtUserKernelRegister",
    "hipblasLtUserKernelSetExactMatch", "hipblasLtUserKernelGetCounts",
    "hipblasLtMatmulAlgoIsUserKernel",
)


class HipBLASLtError(RuntimeError):
    pass


class MatmulAlgo(ctypes.Structure):
    _fields_ = [("data", ctypes.c_uint8 * 16), ("max_workspace_bytes", ctypes.c_size_t)]

    @property
    def index(self) -> int:
        return int.from_bytes(bytes(self.data[:4]), "little", signed=True)

    @classmethod
    def from_index(cls, index: int, max_workspace: int) -> "MatmulAlgo":
        # The layout getAlgosFromIndex produces: the solution index, then zeros.
        algo = cls()
        algo.data[:4] = list(index.to_bytes(4, "little", signed=True))
        algo.max_workspace_bytes = max_workspace
        return algo


class HeuristicResult(ctypes.Structure):
    _fields_ = [("algo", MatmulAlgo), ("workspaceSize", ctypes.c_size_t),
                ("state", ctypes.c_int), ("wavesCount", ctypes.c_float),
                ("reserved", ctypes.c_int * 4)]


def mapped_libraries(name: str) -> list[str]:
    """Paths of every mapped file of this process whose basename starts with ``name``."""
    out = []
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split(None, 5)
            path = parts[5].strip() if len(parts) == 6 else ""
            if os.path.basename(path).startswith(name) and path not in out:
                out.append(path)
    return out


def _dlopen(path: str) -> ctypes.CDLL:
    try:
        return ctypes.CDLL(path)
    except OSError as e:
        if "libomp" not in str(e):
            raise
    # A dev install tree's RUNPATH does not reach ROCm's llvm/lib, where libomp lives.
    rocm = Path(os.environ.get("ROCM_PATH", "/opt/rocm"))
    ctypes.CDLL(str(rocm / "llvm/lib/libomp.so"), mode=ctypes.RTLD_GLOBAL)
    return ctypes.CDLL(path)


class HipBLASLt:
    """The libhipblaslt.so this process uses, or will use."""

    def __init__(self, path: str | os.PathLike | None = None):
        loaded = mapped_libraries("libhipblaslt.so")
        want = path or os.environ.get("GEKO_HIPBLASLT_LIB")
        if loaded:
            if want and os.path.realpath(want) not in {os.path.realpath(p) for p in loaded}:
                raise HipBLASLtError(
                    f"this process already loaded {loaded[0]}; registering into {want} would "
                    "load a second copy whose kernels nothing else in the process sees")
            want = loaded[0]
        self.lib = _dlopen(str(want or "libhipblaslt.so.1"))
        self.path = (mapped_libraries("libhipblaslt.so") or [str(want)])[0]
        self.hip = ctypes.CDLL(mapped_libraries("libamdhip64.so")[0])
        self._declare()

    def _declare(self):
        P, I, S, U64, I64 = (ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
                             ctypes.c_uint64, ctypes.c_int64)
        pP, pI, pS = ctypes.POINTER(P), ctypes.POINTER(I), ctypes.POINTER(S)
        sigs = {
            "hipblasLtGetVersion": [P, pI],
            "hipblasLtGetGitRevision": [P, ctypes.c_char_p],
            "hipblasLtCreate": [pP],
            "hipblasLtDestroy": [P],
            "hipblasLtMatmulDescCreate": [pP, I, I],
            "hipblasLtMatmulDescDestroy": [P],
            "hipblasLtMatmulDescSetAttribute": [P, I, P, S],
            "hipblasLtMatrixLayoutCreate": [pP, I, U64, U64, I64],
            "hipblasLtMatrixLayoutDestroy": [P],
            "hipblasLtMatrixLayoutSetAttribute": [P, I, P, S],
            "hipblasLtMatmulPreferenceCreate": [pP],
            "hipblasLtMatmulPreferenceDestroy": [P],
            "hipblasLtMatmulPreferenceSetAttribute": [P, I, P, S],
            "hipblasLtMatmulAlgoGetHeuristic": [P, P, P, P, P, P, P, I,
                                                ctypes.POINTER(HeuristicResult), pI],
            "hipblasLtMatmul": [P, P, P, P, P, P, P, P, P, P, P, P,
                                ctypes.POINTER(MatmulAlgo), P, S, P],
            "hipblasLtUserKernelRefresh": [P, pS],
            "hipblasLtUserKernelRegister": [P, ctypes.c_char_p, ctypes.c_char_p, pI, I, pI],
            "hipblasLtUserKernelSetExactMatch": [P, I, S, S, S, S],
            "hipblasLtUserKernelGetCounts": [P, pS, pS],
            "hipblasLtMatmulAlgoIsUserKernel": [ctypes.POINTER(MatmulAlgo), pI],
        }
        for name, argtypes in sigs.items():
            fn = getattr(self.lib, name, None)
            if fn is not None:
                fn.argtypes, fn.restype = argtypes, ctypes.c_int
        for name, argtypes in {"hipGetDevice": [pI], "hipSetDevice": [I],
                               "hipMalloc": [pP, S], "hipFree": [P],
                               "hipMemcpy": [P, P, S, I], "hipMemset": [P, I, S],
                               "hipDeviceSynchronize": [], "hipEventCreate": [pP],
                               "hipEventDestroy": [P], "hipEventRecord": [P, P],
                               "hipEventSynchronize": [P],
                               "hipEventElapsedTime": [ctypes.POINTER(ctypes.c_float), P, P],
                               }.items():
            fn = getattr(self.hip, name)
            fn.argtypes, fn.restype = argtypes, ctypes.c_int

    @property
    def has_user_kernels(self) -> bool:
        return all(hasattr(self.lib, s) for s in USER_KERNEL_SYMBOLS)

    @property
    def has_old_user_kernel_api(self) -> bool:
        """The earlier API, whose Refresh writes an int rather than a size_t."""
        return hasattr(self.lib, "hipblasLtUserKernelLibraryOpen")

    def call(self, name: str, *args) -> None:
        st = getattr(self.lib, name)(*args)
        if st != 0:
            label = HIPBLAS_STATUS[st] if 0 <= st < len(HIPBLAS_STATUS) else str(st)
            raise HipBLASLtError(f"{name} returned HIPBLAS_STATUS_{label}")

    def hip_call(self, name: str, *args) -> None:
        st = getattr(self.hip, name)(*args)
        if st != 0:
            raise HipBLASLtError(f"{name} returned hipError {st}")

    def info(self) -> dict:
        # Both calls only check the handle for null, so this needs no GPU.
        dummy = ctypes.c_void_p(1)
        version, rev = ctypes.c_int(0), ctypes.create_string_buffer(256)
        self.call("hipblasLtGetVersion", dummy, ctypes.byref(version))
        self.call("hipblasLtGetGitRevision", dummy, rev)
        v = version.value
        api = ("old (hipblasLtUserKernelLibraryOpen)" if self.has_old_user_kernel_api
               else "yes" if self.has_user_kernels else "no")
        return {"path": self.path, "realpath": os.path.realpath(self.path),
                "version": f"{v // 100000}.{v // 100 % 1000}.{v % 100}",
                "git_revision": rev.value.decode(), "user_kernel_api": api,
                LIBRARY_ENV: os.environ.get(LIBRARY_ENV)}


@dataclass
class Problem:
    """One GEMM: column-major, D = A * B, plain epilogue."""

    trans: str  # "NN", "NT", "TN" or "TT"
    m: int
    n: int
    batch: int
    k: int
    a_type: str  # Tensile type names, as in a payload's problemType
    b_type: str
    c_type: str

    @property
    def shapes(self) -> list[tuple[int, int, str]]:
        a = (self.m, self.k) if self.trans[0] == "N" else (self.k, self.m)
        b = (self.k, self.n) if self.trans[1] == "N" else (self.n, self.k)
        return [(*a, self.a_type), (*b, self.b_type),
                (self.m, self.n, self.c_type), (self.m, self.n, self.c_type)]

    def label(self) -> str:
        return (f"{self.trans} {TENSILE_TYPES[self.a_type][1]} "
                f"M={self.m} N={self.n} B={self.batch} K={self.k}")


def user_kernel_library() -> Path | None:
    """The user kernel library HIPBLASLT_USER_KERNEL_LIBRARY names, if any."""
    value = os.environ.get(LIBRARY_ENV, "")
    return Path(value) if value else None


def saved_in(root: Path, co: Path) -> bool:
    """Whether a journal in ``root`` records a payload with this code object.

    The store names objects by a hash only hipBLASLt computes, so this looks for a
    byte-identical copy of the code object and a payload record naming it.
    """
    data = Path(co).read_bytes()
    hashes = {o.stem for o in (root / "objects").glob("*.co")
              if o.stat().st_size == len(data) and o.read_bytes() == data}
    for journal in root.glob("*/journal"):
        for line in journal.read_text(errors="replace").splitlines():
            f = line.split()
            if f[:1] == ["R"] and len(f) >= 3 and f[2] in hashes:
                return True
    return False


def saved_mappings(root: Path) -> list[tuple[str, Problem, Path, int]]:
    """Every size a user kernel library maps: (arch, problem, payload .dat, shard index).

    Later mappings of a size replace earlier ones, in the journal as in hipBLASLt,
    so each size appears once, with its newest mapping.
    """
    out: dict[tuple, tuple] = {}
    payloads: dict[Path, Payload] = {}
    for journal in sorted(root.glob("*/journal")):
        arch = journal.parent.name
        dats = {}
        for line in journal.read_text(errors="replace").splitlines():
            f = line.split()
            if f[:1] == ["R"] and len(f) == 4:
                dats[f[1]] = root / "objects" / f"{f[1]}.{f[3]}"
            elif f[:1] == ["E"] and len(f) == 8 and f[1] in dats:
                dat = dats[f[1]]
                if dat not in payloads:
                    payloads[dat] = read_payload(dat)
                shard = int(f[3])
                pt = payloads[dat].problem_types.get(shard)
                if pt is None:
                    continue
                prob = problem_of(pt, *map(int, f[4:8]))
                out[(arch, _key(prob))] = (arch, prob, dat, shard)
    return list(out.values())


class UserKernels:
    """A hipBLASLt handle on one device, and the user kernel calls made through it.

    With ``library``, sets HIPBLASLT_USER_KERNEL_LIBRARY so registrations and
    mappings are saved there, the same as setting it before starting Python. That
    only works if hipBLASLt has not initialized a device in this process yet;
    ``register_workdir`` checks that payloads were saved.
    """

    def __init__(self, lib: HipBLASLt | str | None = None, device: int | None = None,
                 library: str | os.PathLike | None = None):
        if library is not None:
            root = Path(library).expanduser().absolute()
            current = os.environ.get(LIBRARY_ENV)
            if current and Path(current) != root:
                raise HipBLASLtError(f"{LIBRARY_ENV} is already {current!r}; hipBLASLt "
                                     f"reads it once, so it cannot be changed to {root}")
            os.environ[LIBRARY_ENV] = str(root)
        self.hb = lib if isinstance(lib, HipBLASLt) else HipBLASLt(lib)
        if not self.hb.has_user_kernels or self.hb.has_old_user_kernel_api:
            info = self.hb.info()
            why = ("has the earlier user kernel API, which this module does not support"
                   if self.hb.has_old_user_kernel_api else
                   "has no user kernel API; it was built without HIPBLASLT_ENABLE_USER_KERNELS")
            raise HipBLASLtError(f"{info['path']} (hipBLASLt {info['version']}, "
                                 f"git {info['git_revision']}) {why}")
        self.device = device
        self.handle = ctypes.c_void_p()
        with self.on_device():
            self.hb.call("hipblasLtCreate", ctypes.byref(self.handle))

    @contextlib.contextmanager
    def on_device(self) -> Iterator[None]:
        """Run on self.device, then restore the caller's current device."""
        if self.device is None:
            yield
            return
        prev = ctypes.c_int()
        self.hb.hip_call("hipGetDevice", ctypes.byref(prev))
        self.hb.hip_call("hipSetDevice", self.device)
        try:
            yield
        finally:
            self.hb.hip_call("hipSetDevice", prev.value)

    def close(self) -> None:
        if self.handle:
            with self.on_device():
                self.hb.call("hipblasLtDestroy", self.handle)
            self.handle = ctypes.c_void_p()

    def __enter__(self) -> "UserKernels":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def refresh(self) -> int:
        """Load what other processes saved since this one last read the library."""
        n = ctypes.c_size_t(0)
        with self.on_device():
            self.hb.call("hipblasLtUserKernelRefresh", self.handle, ctypes.byref(n))
        return n.value

    def register(self, dat: str | os.PathLike, co: str | os.PathLike) -> list[int]:
        """Indices for the payload's solutions, in ascending payload-index order.

        Registering a payload already registered or loaded returns its indices.
        """
        cap = 4096
        buf, n = (ctypes.c_int * cap)(), ctypes.c_int(0)
        with self.on_device():
            self.hb.call("hipblasLtUserKernelRegister", self.handle, str(dat).encode(),
                         str(co).encode(), buf, cap, ctypes.byref(n))
        if n.value > cap:
            raise HipBLASLtError(f"payload has {n.value} kernels, more than {cap}")
        return list(buf[: n.value])

    def set_exact_match(self, index: int, m: int, n: int, batch: int, k: int) -> None:
        with self.on_device():
            self.hb.call("hipblasLtUserKernelSetExactMatch", self.handle, index, m, n, batch, k)

    def counts(self) -> tuple[int, int]:
        """(registered kernels, distinct shapes and types with a mapped kernel)."""
        r, s = ctypes.c_size_t(0), ctypes.c_size_t(0)
        with self.on_device():
            self.hb.call("hipblasLtUserKernelGetCounts", self.handle, ctypes.byref(r),
                         ctypes.byref(s))
        return r.value, s.value

    def is_user_kernel(self, algo: MatmulAlgo) -> bool:
        out = ctypes.c_int(0)
        self.hb.call("hipblasLtMatmulAlgoIsUserKernel", ctypes.byref(algo), ctypes.byref(out))
        return bool(out.value)

    @contextlib.contextmanager
    def _descriptors(self, p: Problem, max_workspace: int):
        hb, made = self.hb, []
        try:
            desc = ctypes.c_void_p()
            hb.call("hipblasLtMatmulDescCreate", ctypes.byref(desc), HIPBLAS_COMPUTE_32F, HIP_R_32F)
            made.append(("hipblasLtMatmulDescDestroy", desc))
            for attr, ch in ((HIPBLASLT_MATMUL_DESC_TRANSA, p.trans[0]),
                             (HIPBLASLT_MATMUL_DESC_TRANSB, p.trans[1])):
                op = ctypes.c_int32(HIPBLAS_OP_N if ch == "N" else HIPBLAS_OP_T)
                hb.call("hipblasLtMatmulDescSetAttribute", desc, attr, ctypes.byref(op), 4)
            layouts = []
            for rows, cols, ty in p.shapes:
                lay = ctypes.c_void_p()
                hb.call("hipblasLtMatrixLayoutCreate", ctypes.byref(lay),
                        TENSILE_TYPES[ty][0], rows, cols, rows)
                made.append(("hipblasLtMatrixLayoutDestroy", lay))
                if p.batch > 1:
                    count, stride = ctypes.c_int32(p.batch), ctypes.c_int64(rows * cols)
                    hb.call("hipblasLtMatrixLayoutSetAttribute", lay,
                            HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, ctypes.byref(count), 4)
                    hb.call("hipblasLtMatrixLayoutSetAttribute", lay,
                            HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, ctypes.byref(stride), 8)
                layouts.append(lay)
            pref = ctypes.c_void_p()
            hb.call("hipblasLtMatmulPreferenceCreate", ctypes.byref(pref))
            made.append(("hipblasLtMatmulPreferenceDestroy", pref))
            ws = ctypes.c_uint64(max_workspace)
            hb.call("hipblasLtMatmulPreferenceSetAttribute", pref,
                    HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, ctypes.byref(ws), 8)
            yield desc, layouts, pref
        finally:
            for name, obj in reversed(made):
                hb.call(name, obj)

    def heuristics(self, p: Problem, count: int,
                   max_workspace: int = 128 << 20) -> list[HeuristicResult]:
        """hipBLASLt's first ``count`` choices for ``p``, best first.

        A mapped user kernel comes first, followed by what hipBLASLt would pick
        without it, so ``heuristics(p, 2)`` gives both today's and the shipped choice.
        """
        with self.on_device(), self._descriptors(p, max_workspace) as (desc, lay, pref):
            res, n = (HeuristicResult * count)(), ctypes.c_int(0)
            st = self.hb.lib.hipblasLtMatmulAlgoGetHeuristic(
                self.handle, desc, *lay, pref, count, res, ctypes.byref(n))
            return list(res[: n.value]) if st == 0 else []

    def heuristic(self, p: Problem, max_workspace: int = 128 << 20) -> HeuristicResult | None:
        """hipBLASLt's first choice for ``p``, or None if it has none."""
        best = self.heuristics(p, 1, max_workspace)
        return best[0] if best else None

    @contextlib.contextmanager
    def _buffers(self, p: Problem, sets: int, workspace: int):
        """``sets`` copies of A, B, C, D holding _test_data's inputs, and a workspace.

        Yields the numpy reference, D's numpy type, D's size, and [A, B, C, D] per set.
        """
        hb = self.hb
        nps, sizes, hosts, ref = _test_data(p)
        ptrs = []

        def alloc(nbytes):
            ptr = ctypes.c_void_p()
            hb.hip_call("hipMalloc", ctypes.byref(ptr), max(nbytes, 1))
            ptrs.append(ptr)
            return ptr

        with self.on_device():
            try:
                bufs = []
                for _ in range(sets):
                    abcd = [alloc(n) for n in sizes]
                    for ptr, raw in zip(abcd, hosts):
                        hb.hip_call("hipMemcpy", ptr, raw, len(raw), HIP_MEMCPY_H2D)
                    hb.hip_call("hipMemset", abcd[2], 0, sizes[2])
                    bufs.append(abcd)
                yield ref, nps[3], sizes[3], bufs, alloc(workspace)
            finally:
                for ptr in ptrs:
                    hb.hip_call("hipFree", ptr)

    def _matmul(self, desc, lay, abcd, algo, ws, workspace):
        alpha, beta = ctypes.c_float(1.0), ctypes.c_float(0.0)
        A, B, C, D = abcd
        self.hb.call("hipblasLtMatmul", self.handle, desc, ctypes.byref(alpha),
                     A, lay[0], B, lay[1], ctypes.byref(beta), C, lay[2], D, lay[3],
                     ctypes.byref(algo), ws, workspace, None)

    def run(self, p: Problem, algos: list[MatmulAlgo], workspace: int = 128 << 20):
        """D = A * B with each algo on the same small-integer inputs.

        Returns the numpy reference and each algo's D, all as flat column-major
        float32. Small integers keep every fp32 partial sum exact, so a correct
        kernel matches the reference whatever its reduction order.
        """
        hb, out = self.hb, []
        with self._descriptors(p, workspace) as (desc, lay, _), \
                self._buffers(p, 1, workspace) as (ref, d_type, d_size, (abcd,), ws):
            for algo in algos:
                hb.hip_call("hipMemset", abcd[3], 0, d_size)
                self._matmul(desc, lay, abcd, algo, ws, workspace)
                hb.hip_call("hipDeviceSynchronize")
                raw = ctypes.create_string_buffer(d_size)
                hb.hip_call("hipMemcpy", raw, abcd[3], d_size, HIP_MEMCPY_D2H)
                out.append(_from_bytes(raw.raw, d_type))
        return ref, out

    def time(self, p: Problem, algos: list[MatmulAlgo], rotating: int = 512 << 20,
             warmup: int = 50, iters: int = 1000, passes: int = 3,
             workspace: int = 128 << 20) -> list[float]:
        """Microseconds per call of each algo, the best of ``passes``.

        Each pass times ``iters`` back-to-back calls with GPU events. The calls cycle
        through copies of the operands totalling at least ``rotating`` bytes, so
        operands come from memory rather than cache, as in hipblaslt-bench and Geko.
        """
        hb = self.hb
        per_set = sum(r * c * p.batch * TENSILE_TYPES[t][3] for r, c, t in p.shapes)
        sets = max(1, -(-rotating // per_set))
        times = []
        with self._descriptors(p, workspace) as (desc, lay, _), \
                self._buffers(p, sets, workspace) as (_, _, _, bufs, ws):
            start, stop = ctypes.c_void_p(), ctypes.c_void_p()
            hb.hip_call("hipEventCreate", ctypes.byref(start))
            hb.hip_call("hipEventCreate", ctypes.byref(stop))
            try:
                for algo in algos:
                    for i in range(warmup):
                        self._matmul(desc, lay, bufs[i % sets], algo, ws, workspace)
                    best = float("inf")
                    for _ in range(passes):
                        hb.hip_call("hipEventRecord", start, None)
                        for i in range(iters):
                            self._matmul(desc, lay, bufs[i % sets], algo, ws, workspace)
                        hb.hip_call("hipEventRecord", stop, None)
                        hb.hip_call("hipEventSynchronize", stop)
                        ms = ctypes.c_float()
                        hb.hip_call("hipEventElapsedTime", ctypes.byref(ms), start, stop)
                        best = min(best, ms.value * 1000 / iters)
                    times.append(best)
            finally:
                hb.hip_call("hipEventDestroy", start)
                hb.hip_call("hipEventDestroy", stop)
        return times


def _round_to(x, ty: str):
    """x as float32 after rounding to ty, and its raw bytes in ty."""
    import numpy as np

    if ty == "bfloat16":
        u = x.astype(np.float32).view(np.uint32)
        b = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
        return (b.astype(np.uint32) << 16).view(np.float32), b.tobytes()
    y = x.astype(ty)
    return y.astype(np.float32), y.tobytes()


def _from_bytes(raw: bytes, ty: str):
    import numpy as np

    if ty == "bfloat16":
        return (np.frombuffer(raw, np.uint16).astype(np.uint32) << 16).view(np.float32)
    return np.frombuffer(raw, ty).astype(np.float32)


def _test_data(p: Problem):
    """Numpy types and byte sizes of A, B, C, D; raw small-integer A and B; reference D."""
    import numpy as np

    nps = []
    for _, _, ty in p.shapes:
        name = TENSILE_TYPES[ty][2]
        if name is None:
            raise HipBLASLtError(f"cannot fill or read {ty} buffers")
        nps.append(name)
    rng = np.random.default_rng(0)
    sizes = [r * c * p.batch * TENSILE_TYPES[t][3] for r, c, t in p.shapes]
    ins = [rng.integers(-2, 3, r * c * p.batch).astype(np.float32) for r, c, _ in p.shapes[:2]]
    hosts = [_round_to(x, t)[1] for x, t in zip(ins, nps[:2])]
    (ar, ac, _), (br, bc, _) = p.shapes[:2]
    a = ins[0].reshape(p.batch, ac, ar).transpose(0, 2, 1)
    b = ins[1].reshape(p.batch, bc, br).transpose(0, 2, 1)
    a = a if p.trans[0] == "N" else a.transpose(0, 2, 1)
    b = b if p.trans[1] == "N" else b.transpose(0, 2, 1)
    ref = _round_to(np.matmul(a, b).transpose(0, 2, 1).reshape(-1), nps[3])[0]
    return nps, sizes, hosts, ref


# ---------------------------------------------------------------------------
# Geko payloads
# ---------------------------------------------------------------------------

SIZE_PROPS = {"FreeSizeA": "m", "FreeSizeB": "n", "BatchSize": "batch", "BoundSize": "k"}


@dataclass
class Payload:
    dat: Path
    co: Path
    shard_indices: list[int]  # ascending, the order Register assigns indices in
    sizes: list[tuple[Problem, int]] = field(default_factory=list)  # (problem, shard index)
    names: dict[int, str] = field(default_factory=dict)  # shard index -> solution name
    problem_types: dict[int, dict] = field(default_factory=dict)  # shard index -> problemType


def read_payload(dat: Path) -> Payload:
    import msgpack

    raw = dat.read_bytes()
    if dat.suffix == ".zlib":
        raw = zlib.decompress(raw)
    lib = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    ptype = {s["index"]: s["problemType"] for s in lib["solutions"]}
    stem = str(dat)[: -len(".dat.zlib")] if dat.suffix == ".zlib" else str(dat)[: -len(".dat")]
    payload = Payload(dat=dat, co=Path(stem + ".co"), shard_indices=sorted(ptype),
                      names={s["index"]: s.get("name", "") for s in lib["solutions"]},
                      problem_types=ptype)

    def tables(x):
        if isinstance(x, dict):
            if "table" in x and "properties" in x:
                yield x
            for v in x.values():
                yield from tables(v)
        elif isinstance(x, list):
            for v in x:
                yield from tables(v)

    for t in tables(lib["library"]):
        names = [SIZE_PROPS.get(prop.get("type")) for prop in t["properties"]]
        if sorted(filter(None, names)) != sorted(SIZE_PROPS.values()):
            continue
        for row in t["table"]:
            pt = ptype.get(row["index"])
            if pt is None:
                continue
            d = dict(zip(names, row["key"]))
            prob = problem_of(pt, d["m"], d["n"], d["batch"], d["k"])
            payload.sizes.append((prob, row["index"]))
    return payload


def problem_of(problem_type: dict, m: int, n: int, batch: int, k: int) -> Problem:
    trans = ("T" if problem_type["transA"] else "N") + ("T" if problem_type["transB"] else "N")
    return Problem(trans, m, n, batch, k, problem_type["aType"], problem_type["bType"],
                   problem_type["dType"])


def geko_payloads(workdir: Path) -> list[Payload]:
    dats = sorted((workdir / "build/library").glob("*/TensileLibrary_*_Contraction_*.dat*"))
    return [read_payload(d) for d in dats if d.name.endswith((".dat", ".dat.zlib"))]


def geko_winners(workdir: Path) -> set[tuple] | None:
    """Sizes Geko kept (results/final_results.csv), or None if it wrote no verdict."""
    final = workdir / "results/final_results.csv"
    if not final.is_file():
        # Geko writes no final_results.csv when it keeps nothing.
        return set() if (workdir / "results/raw_results.csv").is_file() else None
    return {(r["transA"] + r["transB"], int(r["m"]), int(r["n"]), int(r["batch_count"]),
             int(r["k"]), r["a_type"]) for r in csv.DictReader(open(final))}


def _key(p: Problem) -> tuple:
    return (p.trans, p.m, p.n, p.batch, p.k, TENSILE_TYPES[p.a_type][1])


def register_workdir(uk: UserKernels, workdir: str | os.PathLike, all_sizes: bool = False,
                     check: bool = True, tolerance: float = 1e-2, replace: bool = False,
                     log=print) -> list[dict]:
    """Register every payload in a Geko workdir and map the sizes it won.

    With ``check``, each kernel is first run by index against a host reference and
    mapped only if it matches, since a mapping cannot be undone. After mapping, the
    heuristic must return that exact kernel. With a user kernel library, each
    payload must also have been saved there.

    The newest mapping of a shape wins and nothing here compares speed, so a shape
    that already selects another user kernel is left alone unless ``replace``.
    """
    workdir = Path(workdir)
    root = user_kernel_library()
    payloads = geko_payloads(workdir)
    if not payloads:
        raise HipBLASLtError(f"no payload under {workdir / 'build/library'}")
    winners = None if all_sizes else geko_winners(workdir)
    if winners is None and not all_sizes:
        raise HipBLASLtError(f"{workdir} has no results/final_results.csv to say which sizes "
                             "Geko kept; pass all_sizes=True to map every tuned size")

    report = []
    for pl in payloads:
        assigned = uk.register(pl.dat, pl.co)
        if len(assigned) != len(pl.shard_indices):
            raise HipBLASLtError(f"{pl.dat.name}: registered {len(assigned)} kernels, "
                                 f"payload lists {len(pl.shard_indices)}")
        if root is not None and not saved_in(root, pl.co):
            raise HipBLASLtError(
                f"{pl.dat.name} was registered for this process only, not saved to {root}: "
                f"{LIBRARY_ENV} was set after hipBLASLt initialized the device, hipBLASLt "
                "refused the directory (see its warning), or this libhipblaslt.so reads the "
                "earlier name HIPBLASLT_LOCAL_KERNEL_LIBRARIES")
        by_shard = dict(zip(pl.shard_indices, assigned))
        log(f"registered {len(assigned)} kernel(s) from {pl.dat.name}"
            + (f", saved to {root}" if root is not None else ""))

        for prob, shard in pl.sizes:
            idx = by_shard[shard]
            r = {"size": prob.label(), "payload": pl.dat.name, "index": idx, "mapped": False}
            report.append(r)
            if winners is not None and _key(prob) not in winners:
                r["result"] = "not kept by Geko"
                continue
            before = uk.heuristic(prob)
            r["before"] = before.algo.index if before else None
            if r["before"] == idx:
                r.update(mapped=True, result="already selected")
                log(f"  {prob.label():48} index {idx}: {r['result']}")
                continue
            if before is not None and uk.is_user_kernel(before.algo) and not replace:
                r["result"] = f"not mapped: already selects user kernel {r['before']}"
                log(f"  {prob.label():48} index {idx}: {r['result']}")
                continue
            if check:
                try:
                    ref, (got,) = uk.run(prob, [MatmulAlgo.from_index(idx, 128 << 20)])
                except HipBLASLtError as e:
                    r["result"] = f"not mapped: {e}"
                    log(f"  {prob.label():48} {r['result']}")
                    continue
                rel = float(abs(got - ref).max() / max(abs(ref).max(), 1e-30))
                r["rel_diff"] = rel
                if rel > tolerance:
                    r["result"] = f"not mapped: output differs from the reference (rel {rel:.3g})"
                    log(f"  {prob.label():48} {r['result']}")
                    continue
            uk.set_exact_match(idx, prob.m, prob.n, prob.batch, prob.k)
            r["mapped"] = True
            after = uk.heuristic(prob)
            r["after"] = after.algo.index if after else None
            ok = after is not None and after.algo.index == idx and uk.is_user_kernel(after.algo)
            r["result"] = "selected" if ok else "mapped, but the heuristic does not return it"
            log(f"  {prob.label():48} index {idx}: {r['result']}"
                + (f" (was {r['before']})" if r.get("before") is not None else "")
                + (f", rel diff {r['rel_diff']:.2g}" if "rel_diff" in r else ""))
    return report


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m geko.user_kernels",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--lib", help="libhipblaslt.so to use if the process has not loaded one "
                    "(default: $GEKO_HIPBLASLT_LIB, else the loader's search path)")
    ap.add_argument("--device", type=int, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info", help="which libhipblaslt.so, its version and git revision")
    reg = sub.add_parser("register", help="register a Geko workdir's payloads and map its sizes")
    reg.add_argument("workdir", type=Path)
    reg.add_argument("--all", action="store_true",
                     help="map every tuned size, not only those in results/final_results.csv")
    reg.add_argument("--no-check", action="store_true",
                     help="map without first comparing outputs with a host reference")
    reg.add_argument("--replace", action="store_true",
                     help="remap sizes that already select another user kernel")
    reg.add_argument("--json", type=Path, help="write the per-size report here")
    sub.add_parser("status", help=f"what a new process loads from the user kernel library "
                   f"${LIBRARY_ENV} names")
    args = ap.parse_args(argv)

    hb = HipBLASLt(args.lib)
    info = hb.info()
    print(f"hipBLASLt {info['version']} (git {info['git_revision']}) at {info['realpath']}; "
          f"user kernel API: {info['user_kernel_api']}")
    if args.cmd == "info":
        return 0

    with UserKernels(hb, args.device) as uk:
        root = user_kernel_library()
        none = f"none ({LIBRARY_ENV} is unset); registrations last for this process"
        print(f"user kernel library: {root or none}")
        if args.cmd == "register":
            report = register_workdir(uk, args.workdir, all_sizes=args.all,
                                      check=not args.no_check, replace=args.replace)
            if args.json:
                args.json.write_text(json.dumps(report, indent=2))
        else:
            # Initializes the device, which loads everything saved for its architecture.
            uk.refresh()
        registered, selectable = uk.counts()
        print(f"{registered} kernel(s) registered, {selectable} shape(s) mapped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
