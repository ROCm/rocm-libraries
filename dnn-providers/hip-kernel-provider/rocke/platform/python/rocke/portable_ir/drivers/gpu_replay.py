#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# gpu_replay.py -- the device end of the portable-IR claim. parity_matrix proves
# the replayed .ll is byte-identical to Python's; this proves the thing actually
# runs and computes the right answer on hardware.
#
# The run splits into two phases separated by a real file on disk, so the
# runtime half cannot accidentally lean on the Python IR stack:
#
#   author time : spec -> build_*() -> record_kernel() -> recipe -> CBOR on disk
#   run time    : CBOR bytes -> C recipe VM -> .ll -> comgr -> HSACO -> launch
#
# Only the second phase is "runtime", and the only thing crossing the boundary
# is the CBOR artifact. Nothing in phase 2 builds IR in Python -- the recipe VM
# inside librocke rebuilds the kernel and lowers it.
#
# Per case we gate two things:
#   ll      : replayed .ll is byte-identical to the Python lowerer's .ll
#   numeric : device output matches a numpy reference. The linear and
#             non-transcendental unary ops are gated BIT-EXACT (rtol=atol=0):
#             the kernel widens to f32, computes, and casts back, so there is
#             no legitimate source of drift. The transcendental ops get ~1 f16
#             ULP of slack because the GPU's v_exp/v_tanh are not correctly
#             rounded while the numpy reference is.
#
# Deliberately numpy-only: the device path here is rocke's torch-free one
# (DeviceMem over hipMalloc + the ctypes HIP module), which keeps this runnable
# on a bare ROCm box and matches the "no heavy runtime" story of portable IR.
#
#   python3 -m rocke.portable_ir.drivers.gpu_replay [--device 2] [--verbose]
#
# Needs a GPU, comgr, numpy, and a shared librocke (ROCKE_ONLINE_LIB).

import argparse
import ctypes
import os
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# (rtol, atol). (0, 0) means the device result must match numpy bit-for-bit.
EXACT = (0.0, 0.0)
ULP_F16 = (1e-3, 1e-3)


# ---------------------------------------------------------------------
# bf16 helpers -- numpy has no bfloat16, so bf16 buffers are carried as raw
# uint16 bit patterns and converted for the reference math.
# ---------------------------------------------------------------------
def _f32_to_bf16(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    # round-to-nearest-even on the discarded low 16 bits
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return rounded.astype(np.uint16)


def _bf16_to_f32(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


def _as_f32(arr: np.ndarray, dtype: str) -> np.ndarray:
    return _bf16_to_f32(arr) if dtype == "bf16" else arr.astype(np.float32)


def _from_f32(arr: np.ndarray, dtype: str) -> np.ndarray:
    return _f32_to_bf16(arr) if dtype == "bf16" else arr.astype(np.float16)


def _storage(dtype: str):
    return np.uint16 if dtype == "bf16" else np.float16


@dataclass
class Plan:
    """What phase 3 needs to run one kernel on device."""

    buffers: Dict[str, np.ndarray]
    scalars: Dict[str, int]
    out: str
    ref_f32: np.ndarray
    dtype: str
    grid: tuple
    block: tuple
    signature: list
    tol: Tuple[float, float]


@dataclass
class Case:
    label: str
    family: str
    build: Callable[[], Any]  # phase 1: the production KernelDef
    plan: Callable[[], Plan]  # phase 3: inputs + reference


@dataclass
class Result:
    label: str
    family: str
    status: str = "PASS"
    detail: str = ""
    cbor_bytes: int = 0
    ll_lines: int = 0
    max_abs: float = 0.0
    margin: float = 0.0
    exact: bool = False
    timings: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------
_N_ELEM = 1 << 20

_ELEMENTWISE = [
    ("add", EXACT),
    ("sub", EXACT),
    ("mul", EXACT),
    ("max", EXACT),
    ("min", EXACT),
    ("copy", EXACT),
    ("neg", EXACT),
    ("abs", EXACT),
    ("relu", EXACT),
    ("silu", ULP_F16),
    ("gelu_tanh", ULP_F16),
    ("exp2", ULP_F16),
]


def _elementwise_case(op: str, tol: Tuple[float, float], dtype: str = "f16") -> Case:
    from rocke.instances.common.elementwise import ElementwiseSpec, build_elementwise

    spec = ElementwiseSpec(op=op, dtype=dtype)

    def plan() -> Plan:
        from rocke.instances.common.elementwise import (
            elementwise_grid,
            elementwise_signature,
        )

        rng = np.random.default_rng(0xC0FFEE)
        a32 = rng.standard_normal(_N_ELEM, dtype=np.float32)
        b32 = rng.standard_normal(_N_ELEM, dtype=np.float32)
        A, B = _from_f32(a32, dtype), _from_f32(b32, dtype)
        # Reference operands are the ROUNDED values the kernel actually sees.
        x, y = _as_f32(A, dtype), _as_f32(B, dtype)
        refs = {
            "add": lambda: x + y,
            "sub": lambda: x - y,
            "mul": lambda: x * y,
            "max": lambda: np.maximum(x, y),
            "min": lambda: np.minimum(x, y),
            "copy": lambda: x,
            "neg": lambda: -x,
            "abs": lambda: np.abs(x),
            "relu": lambda: np.maximum(x, 0.0),
            "silu": lambda: x / (1.0 + np.exp(-x)),
            "gelu_tanh": lambda: 0.5
            * x
            * (
                1.0
                + np.tanh(np.float32(0.7978845608) * (x + np.float32(0.044715) * x**3))
            ),
            "exp2": lambda: np.exp2(x),
        }
        # Round the reference through the storage type: the kernel stores f16 /
        # bf16, so the comparison must too, or every case looks like drift.
        ref = _as_f32(_from_f32(refs[op]().astype(np.float32), dtype), dtype)

        buffers = {"A": A, "C": np.zeros_like(A)}
        if spec.is_binary():
            buffers["B"] = B
        return Plan(
            buffers=buffers,
            scalars={"N": _N_ELEM},
            out="C",
            ref_f32=ref,
            dtype=dtype,
            grid=elementwise_grid(_N_ELEM, spec),
            block=(spec.block_size, 1, 1),
            signature=elementwise_signature(spec),
            tol=tol,
        )

    suffix = "" if dtype == "f16" else f".{dtype}"
    return Case(
        f"elementwise.{op}{suffix}",
        "elementwise",
        lambda: build_elementwise(spec),
        plan,
    )


def _gemm_case(
    label: str,
    m: int,
    n: int,
    k: int,
    dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    warp_m: int,
    warp_n: int,
    *,
    arch: str,
    tol: Tuple[float, float],
) -> Case:
    from rocke.instances.common.gemm_universal import (
        DataSpec,
        TileSpec,
        TraitSpec,
        UniversalGemmSpec,
        build_universal_gemm,
    )

    # rocke spells the gemm dtypes fp16/bf16; the storage helpers here use the
    # elementwise spelling (f16/bf16).
    store = "bf16" if dtype == "bf16" else "f16"
    spec = UniversalGemmSpec(
        name=label,
        tile=TileSpec(
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            warp_m=warp_m,
            warp_n=warp_n,
            warp_k=1,
            warp_tile_m=16,
            warp_tile_n=16,
            warp_tile_k=16,
        ),
        trait=TraitSpec(pipeline="compv3", epilogue="default"),
        data=DataSpec(
            dtype_a=dtype, dtype_b=dtype, dtype_c=dtype, dtype_acc="fp32", layout="RCR"
        ),
        wave_size=64,
        block_size=warp_m * warp_n * 64,
    )

    def plan() -> Plan:
        from rocke.helpers.manifest import gemm_args_signature

        rng = np.random.default_rng(0xC0FFEE)
        # Small symmetric half-integers: exactly representable in f16/bf16 and
        # in the f32 accumulator, so the reference is exact and any mismatch is
        # a real bug rather than accumulation-order noise.
        a32 = rng.integers(-4, 5, size=(m, k)).astype(np.float32) * 0.5
        b32 = rng.integers(-4, 5, size=(n, k)).astype(np.float32) * 0.5
        A, B = _from_f32(a32, store), _from_f32(b32, store)
        ref = _as_f32(A, store) @ _as_f32(B, store).T
        ref = _as_f32(_from_f32(ref, store), store)

        return Plan(
            buffers={"A": A, "B": B, "C": np.zeros((m, n), dtype=_storage(store))},
            scalars={"M": m, "N": n, "K": k},
            out="C",
            ref_f32=ref,
            dtype=store,
            grid=((n + tile_n - 1) // tile_n, (m + tile_m - 1) // tile_m, 1),
            block=(spec.block_size, 1, 1),
            signature=gemm_args_signature(),
            tol=tol,
        )

    return Case(label, "gemm", lambda: build_universal_gemm(spec, arch=arch), plan)


def _cases(arch: str) -> List[Case]:
    cases = [_elementwise_case(op, tol) for op, tol in _ELEMENTWISE]
    cases += [
        _elementwise_case("add", EXACT, dtype="bf16"),
        _elementwise_case("mul", EXACT, dtype="bf16"),
        _gemm_case(
            "rocke_gpu_replay_gemm_fp16",
            512,
            512,
            256,
            "fp16",
            128,
            128,
            32,
            2,
            2,
            arch=arch,
            tol=EXACT,
        ),
        _gemm_case(
            "rocke_gpu_replay_gemm_bf16",
            512,
            512,
            256,
            "bf16",
            128,
            128,
            32,
            2,
            2,
            arch=arch,
            tol=EXACT,
        ),
    ]
    return cases


# ---------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------
def _author(case: Case, outdir: str, arch: str) -> Tuple[str, str, str]:
    """Phase 1 (Python): record the kernel, write the CBOR recipe artifact.

    Returns (cbor_path, kernel_name, python_ll). The .ll is the oracle for the
    byte-identity gate and is deliberately NOT handed to phase 2.
    """
    from rocke.core.lower_llvm import lower_kernel_to_llvm
    from rocke.portable_ir.src import recipe_bundle
    from rocke.portable_ir.src.recording_builder import record_kernel

    kernel, recipe = record_kernel(case.build)
    path = os.path.join(outdir, f"{case.label}.cbor")
    with open(path, "wb") as f:
        f.write(recipe_bundle.cbor_encode(recipe))
    return path, kernel.name, lower_kernel_to_llvm(kernel, arch=arch)


def _replay(cbor_path: str, arch: str) -> Tuple[str, bytes, Dict[str, float]]:
    """Phase 2 (no Python IR): CBOR file -> C recipe VM -> .ll -> HSACO."""
    from rocke.core.arch import ArchTarget
    from rocke.portable_ir.src import online
    from rocke.runtime.comgr import build_hsaco_from_llvm_ir

    with open(cbor_path, "rb") as f:
        cbor = f.read()
    ll, t = online.recipe_cbor_to_llvm(cbor, arch=arch)
    t0 = time.perf_counter()
    hsaco, _ = build_hsaco_from_llvm_ir(
        ll, isa=ArchTarget.from_gfx(arch).isa_triple, options=["-O3"]
    )
    t["comgr_ms"] = (time.perf_counter() - t0) * 1e3
    return ll, hsaco, t


def _launch(plan: Plan, hsaco: bytes, kernel_name: str) -> np.ndarray:
    """Phase 3: upload, launch the replayed HSACO, download the output."""
    from rocke.runtime.launcher import (
        DeviceMem,
        KernelLauncher,
        LaunchConfig,
        _runtime,
    )

    rt = _runtime()
    dev: Dict[str, DeviceMem] = {}
    host: Dict[str, np.ndarray] = {}
    for name, arr in plan.buffers.items():
        a = np.ascontiguousarray(arr)
        host[name] = a
        mem = DeviceMem(a.nbytes)
        rt.memcpy_h2d(mem.ptr(), (ctypes.c_ubyte * a.nbytes).from_buffer(a), a.nbytes)
        dev[name] = mem

    values: Dict[str, Any] = dict(dev)
    values.update(plan.scalars)
    launcher = KernelLauncher(
        hsaco=hsaco, kernel_name=kernel_name, signature=plan.signature
    )
    launcher(values, config=LaunchConfig(grid=plan.grid, block=plan.block, fence=True))

    out = host[plan.out]
    rt.memcpy_d2h(
        (ctypes.c_ubyte * out.nbytes).from_buffer(out), dev[plan.out].ptr(), out.nbytes
    )
    return out


def _run_case(case: Case, arch: str, outdir: str) -> Result:
    res = Result(case.label, case.family)

    try:
        cbor_path, kernel_name, py_ll = _author(case, outdir, arch)
    except Exception as e:  # noqa: BLE001
        res.status, res.detail = "BUILD_FAIL", f"{type(e).__name__}: {e}"
        return res
    res.cbor_bytes = os.path.getsize(cbor_path)

    try:
        ll, hsaco, timings = _replay(cbor_path, arch)
    except Exception as e:  # noqa: BLE001
        res.status, res.detail = "REPLAY_FAIL", f"{type(e).__name__}: {e}"
        return res
    res.ll_lines = ll.count("\n") + 1
    res.timings = timings
    res.timings["hsaco_kb"] = len(hsaco) / 1024.0

    if ll != py_ll:
        res.status, res.detail = "LL_DIFF", "replayed .ll != python .ll"
        return res

    try:
        plan = case.plan()
    except Exception as e:  # noqa: BLE001
        res.status, res.detail = "PLAN_FAIL", f"{type(e).__name__}: {e}"
        return res

    try:
        out = _launch(plan, hsaco, kernel_name)
    except Exception as e:  # noqa: BLE001
        res.status, res.detail = "LAUNCH_FAIL", f"{type(e).__name__}: {e}"
        return res

    got = _as_f32(out, plan.dtype)
    diff = np.abs(got - plan.ref_f32)
    rtol, atol = plan.tol
    res.max_abs = float(diff.max())
    res.margin = float((diff - (atol + rtol * np.abs(plan.ref_f32))).max())
    res.exact = res.max_abs == 0.0
    if not (res.margin <= 0.0):
        res.status = "NUMERIC"
    kind = "bit-exact" if (rtol, atol) == EXACT else f"rtol={rtol:.0e}"
    res.detail = (
        f"grid={plan.grid} block={plan.block} " f"max_abs={res.max_abs:.3e} [{kind}]"
    )
    return res


# ---------------------------------------------------------------------
def _resolve_flavor(requested: str) -> str:
    """The flavor to lower at. 'auto' asks the loaded comgr what it can accept.

    comgr rejects IR from the wrong LLVM generation, so on a device run the
    flavor is not a free choice -- it is dictated by the installed ROCm.
    """
    if requested != "auto":
        return requested
    try:
        from rocke.core.lower_llvm import _flavor_for_rocm
        from rocke.runtime.comgr import resolved_lib_rocm_version

        ver = resolved_lib_rocm_version()
        if ver is not None:
            return _flavor_for_rocm(*ver)
    except Exception:  # noqa: BLE001 - fall through to the documented default
        pass
    return os.environ.get("ROCKE_LLVM_FLAVOR", "llvm20")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device", default="2", help="GPU ordinal to bind via HIP_VISIBLE_DEVICES"
    )
    ap.add_argument("--arch", default="", help="target gfx (default: probe the device)")
    ap.add_argument(
        "--flavor",
        default="auto",
        help="LLVM flavor; 'auto' matches the installed comgr",
    )
    ap.add_argument("--only", default="", help="substring filter on case labels")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    # Must precede any HIP init: this pins the run to one GPU, which then
    # presents itself as ordinal 0.
    os.environ["HIP_VISIBLE_DEVICES"] = args.device
    # The oracle is the Python lowerer by definition, and the C++ pybind
    # extension is not needed on this path -- saying so avoids a fallback
    # warning per kernel.
    os.environ["ROCKE_BACKEND"] = "python"

    flavor = _resolve_flavor(args.flavor)
    # Pin both engines to it. Python takes a flavor argument; the C side reads
    # the environment. Unlike parity_matrix (device-free, any flavor will do so
    # long as both agree) this one must additionally match the comgr vintage,
    # or the compile is refused -- llvm22 IR on a pre-7.2 comgr SIGABRTs.
    os.environ["ROCKE_LLVM_FLAVOR"] = flavor

    from rocke.runtime.hip_module import get_device_arch, get_device_count

    if get_device_count() < 1:
        print(f"no GPU visible with HIP_VISIBLE_DEVICES={args.device}")
        return 2
    arch = args.arch or get_device_arch(0) or "gfx950"

    from rocke.portable_ir.src import online

    online.load()

    print(
        f"== portable-IR GPU replay (HIP_VISIBLE_DEVICES={args.device}, "
        f"arch={arch}, flavor={flavor}) =="
    )
    print("   author : build -> record -> CBOR recipe on disk")
    print("   runtime: CBOR -> C recipe VM -> .ll -> comgr -> HSACO -> launch")
    print("   gates  : replayed .ll byte-identical to Python  +  device numerics\n")

    cases = [c for c in _cases(arch) if args.only in c.label]
    results = []
    with tempfile.TemporaryDirectory(prefix="rocke_gpu_replay_") as outdir:
        for case in cases:
            res = _run_case(case, arch, outdir)
            results.append(res)
            if res.status != "PASS" or args.verbose:
                mark = "ok" if res.status == "PASS" else res.status
                print(f"  [{mark:^11}] {res.label:<26} {res.detail}")
                if args.verbose and res.timings:
                    t = res.timings
                    print(
                        f"{'':15}{'':26} vm={t.get('build_ms', 0):.2f}ms "
                        f"lower={t.get('lower_ms', 0):.2f}ms "
                        f"comgr={t.get('comgr_ms', 0):.0f}ms  "
                        f"{res.cbor_bytes / 1024:.1f}KiB cbor -> "
                        f"{res.ll_lines} ll -> "
                        f"{t.get('hsaco_kb', 0):.1f}KiB hsaco"
                    )

    npass = sum(1 for r in results if r.status == "PASS")
    nfail = len(results) - npass
    exact = sum(1 for r in results if r.status == "PASS" and r.exact)
    jit_ms = sum(
        r.timings.get("build_ms", 0) + r.timings.get("lower_ms", 0) for r in results
    )

    print("\n" + "=" * 72)
    print(
        f"cases: {len(results)}   passed: {npass}   failed: {nfail}   "
        f"bit-exact vs numpy: {exact}"
    )
    print(
        f"replay cost (recipe VM rebuild + lower, all cases): {jit_ms:.1f}ms "
        f"-- comgr excluded, it dominates and is common to every path"
    )
    print(
        "\n"
        + (
            "PASS: every replayed kernel is byte-identical to Python and "
            "numerically correct on device."
            if not nfail
            else f"FAIL: {nfail} case(s) regressed."
        )
    )
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
