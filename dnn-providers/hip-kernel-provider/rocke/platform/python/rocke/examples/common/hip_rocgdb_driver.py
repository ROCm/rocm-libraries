# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate a standalone, ``rocgdb``-steppable HIP executable for a lowered kernel.

The HIP lowering path (``rocke.core.lower_hip.lower_kernel_to_hip``) emits a
``.hip`` kernel, but there is no way to *step through it line-by-line*. This
module closes that gap: for a registered debug **case** it emits

  1. ``<stem>.hip``  -- the lowered kernel (with the standard prologue), and
  2. ``main.cpp``    -- a host ``main()`` that ``#include``s the ``.hip``,
     allocates buffers, fills deterministic inputs, launches the kernel,
     copies results back, and **self-checks** against a host reference,

then builds them into one executable with ``hipcc -g -O0`` so ``rocgdb`` can set
a breakpoint on a line of the ``.hip`` and single-step with real data resident.

Why a generated ``main()`` and not module-load debugging: linking the kernel and
a host driver into a single ``-g -O0`` executable is the canonical rocgdb
workflow -- source lines resolve reliably and locals are inspectable, which the
production ``LLVM -> COMGR -> .hsaco`` path (optimized, no driver) does not give.

This is a debug / inspection path only. ``-O0`` deliberately does NOT match the
production ``-O3`` ``.hsaco`` -- use it to understand semantics, then verify
performance separately.

Usage::

    PYTHONPATH=python python3 -m rocke.examples.common.hip_rocgdb_driver \\
        --case elementwise.add --arch gfx942 --out-dir ./dbg --build --run
    PYTHONPATH=python python3 -m rocke.examples.common.hip_rocgdb_driver --list

Adding a case: append a :class:`DebugCase` to :data:`_CASES`. Each case supplies
its :class:`KernelDef` builder, the launch grid/block, the ordered arg list
(matching the emitted signature), and two C++ snippets -- ``fill_cpp`` (populate
the ``h_<input>`` host vectors) and ``ref_cpp`` (populate ``h_ref`` for the
checked output). GEMM-family kernels launch through the manifest runner rather
than a flat packed signature, so they are not seeded here; wire one in by
providing a C++ reference GEMM in ``ref_cpp``.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, Union

from rocke.core.ir import KernelDef
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.instances.common.elementwise import (
    ElementwiseSpec,
    build_elementwise,
    elementwise_grid,
)

# Host C++ scalar type per rocke dtype, matching the prologue typedefs in
# ``lower_hip.HIP_PROLOGUE`` (``using fp16 = _Float16;`` etc.).
_HOST_CTYPE = {
    "f16": "fp16",
    "bf16": "bf16",
    "f32": "float",
    "i32": "int",
    "i8": "int8_t",
}


@dataclass(frozen=True)
class PtrArg:
    """A pointer parameter -- a device buffer of ``count`` elements.

    ``count`` is a C++ expression evaluated in ``main()`` where ``N`` (the
    problem size) is in scope, e.g. ``"N"`` or ``"M * K"``.
    """

    name: str
    dtype: str
    count: str
    is_output: bool = False


@dataclass(frozen=True)
class ScalarArg:
    """A by-value scalar parameter passed straight to the launch."""

    name: str
    dtype: str
    value: int


Arg = Union[PtrArg, ScalarArg]


@dataclass(frozen=True)
class DebugCase:
    """Everything the generated ``main()`` needs for one kernel."""

    name: str
    build: Callable[[str], KernelDef]
    grid: Callable[[], Tuple[int, int, int]]
    block: Tuple[int, int, int]
    args: List[Arg]
    n: int
    out_name: str
    fill_cpp: str
    ref_cpp: str
    tol: float = 0.0
    notes: str = ""


def _stem(name: str) -> str:
    """Filesystem/identifier-safe stem for a case name (``elementwise.add``)."""
    return name.replace(".", "_")


# ---------------------------------------------------------------------------
# Case registry
# ---------------------------------------------------------------------------

_EW_ADD = ElementwiseSpec(op="add")
_EW_RELU = ElementwiseSpec(op="relu")
_EW_N = 4096


def _make_cases() -> List[DebugCase]:
    return [
        # Binary elementwise: signature (A, B, C, N).
        DebugCase(
            name="elementwise.add",
            build=lambda arch: build_elementwise(_EW_ADD),
            grid=lambda: elementwise_grid(_EW_N, _EW_ADD),
            block=(_EW_ADD.block_size, 1, 1),
            args=[
                PtrArg("A", "f16", "N"),
                PtrArg("B", "f16", "N"),
                PtrArg("C", "f16", "N", is_output=True),
                ScalarArg("N", "i32", _EW_N),
            ],
            n=_EW_N,
            out_name="C",
            fill_cpp=(
                "  for (int i = 0; i < N; ++i) {\n"
                "    h_A[i] = (fp16)(float)((i % 17) - 8);\n"
                "    h_B[i] = (fp16)(float)((i % 13) - 6);\n"
                "  }"
            ),
            ref_cpp=(
                "  for (int i = 0; i < N; ++i)\n"
                "    h_ref[i] = (fp16)((float)h_A[i] + (float)h_B[i]);"
            ),
            tol=0.0,
            notes="integer-valued f16 add is exact -> tol 0",
        ),
        # Unary elementwise: signature (A, C, N) -- proves the generator
        # generalizes across a different arg count.
        DebugCase(
            name="elementwise.relu",
            build=lambda arch: build_elementwise(_EW_RELU),
            grid=lambda: elementwise_grid(_EW_N, _EW_RELU),
            block=(_EW_RELU.block_size, 1, 1),
            args=[
                PtrArg("A", "f16", "N"),
                PtrArg("C", "f16", "N", is_output=True),
                ScalarArg("N", "i32", _EW_N),
            ],
            n=_EW_N,
            out_name="C",
            fill_cpp=(
                "  for (int i = 0; i < N; ++i)\n"
                "    h_A[i] = (fp16)(float)((i % 23) - 11);"
            ),
            ref_cpp=(
                "  for (int i = 0; i < N; ++i) {\n"
                "    float x = (float)h_A[i];\n"
                "    h_ref[i] = (fp16)(x > 0.0f ? x : 0.0f);\n"
                "  }"
            ),
            tol=0.0,
            notes="integer-valued f16 relu is exact -> tol 0",
        ),
    ]


_CASES = _make_cases()


def _get_case(name: str) -> DebugCase:
    for c in _CASES:
        if c.name == name:
            return c
    names = ", ".join(c.name for c in _CASES)
    raise SystemExit(f"unknown case {name!r}; available: {names}")


# ---------------------------------------------------------------------------
# main.cpp generation
# ---------------------------------------------------------------------------


def _ctype(dtype: str) -> str:
    try:
        return _HOST_CTYPE[dtype]
    except KeyError as exc:  # pragma: no cover - guard for new dtypes
        raise SystemExit(f"no host C++ type mapped for dtype {dtype!r}") from exc


def _render_main(case: DebugCase, kernel: KernelDef, hip_include: str) -> str:
    """Render the host ``main.cpp`` that drives ``kernel`` and self-checks."""
    ptrs = [a for a in case.args if isinstance(a, PtrArg)]
    gx, gy, gz = case.grid()
    bx, by, bz = case.block
    out = next(a for a in ptrs if a.name == case.out_name)
    out_ct = _ctype(out.dtype)

    lines: List[str] = []
    lines.append(f'#include "{hip_include}"')
    lines.append("#include <cstdio>")
    lines.append("#include <cmath>")
    lines.append("#include <vector>")
    lines.append("")
    lines.append(
        "#define HIP_CHECK(x) do { hipError_t _e = (x); if (_e != hipSuccess) { \\\n"
        '  printf("HIP error: %s at %s:%d\\n", hipGetErrorString(_e), __FILE__, __LINE__); \\\n'
        "  return 1; } } while (0)"
    )
    lines.append("")
    lines.append("int main() {")
    lines.append(f"  const int N = {case.n};")
    # Host buffers.
    for a in ptrs:
        lines.append(f"  std::vector<{_ctype(a.dtype)}> h_{a.name}({a.count});")
    lines.append(f"  std::vector<{out_ct}> h_ref({out.count});")
    lines.append("")
    lines.append("  // --- fill inputs (deterministic) ---")
    lines.append(case.fill_cpp)
    lines.append("")
    lines.append("  // --- device buffers ---")
    for a in ptrs:
        ct = _ctype(a.dtype)
        lines.append(f"  {ct} *d_{a.name} = nullptr;")
        lines.append(
            f"  HIP_CHECK(hipMalloc(&d_{a.name}, sizeof({ct}) * ({a.count})));"
        )
    for a in ptrs:
        ct = _ctype(a.dtype)
        if a.is_output:
            lines.append(
                f"  HIP_CHECK(hipMemset(d_{a.name}, 0, sizeof({ct}) * ({a.count})));"
            )
        else:
            lines.append(
                f"  HIP_CHECK(hipMemcpy(d_{a.name}, h_{a.name}.data(), "
                f"sizeof({ct}) * ({a.count}), hipMemcpyHostToDevice));"
            )
    lines.append("")
    lines.append("  // --- launch (breakpoint the kernel below) ---")
    lines.append(f"  dim3 grid({gx}, {gy}, {gz});")
    lines.append(f"  dim3 block({bx}, {by}, {bz});")
    call_args = []
    for a in case.args:
        if isinstance(a, PtrArg):
            call_args.append(f"d_{a.name}")
        else:
            call_args.append(str(a.value))
    lines.append(f"  {kernel.name}<<<grid, block>>>({', '.join(call_args)});")
    lines.append("  HIP_CHECK(hipGetLastError());")
    lines.append("  HIP_CHECK(hipDeviceSynchronize());")
    lines.append("")
    lines.append(
        f"  HIP_CHECK(hipMemcpy(h_{out.name}.data(), d_{out.name}, "
        f"sizeof({out_ct}) * ({out.count}), hipMemcpyDeviceToHost));"
    )
    lines.append("")
    lines.append("  // --- host reference + compare ---")
    lines.append(case.ref_cpp)
    lines.append(f"  const double tol = {case.tol!r};")
    lines.append("  double max_abs = 0.0;")
    lines.append("  int bad = 0;")
    lines.append(f"  for (int i = 0; i < ({out.count}); ++i) {{")
    lines.append(
        f"    double d = fabs((double)(float)h_{out.name}[i] - (double)(float)h_ref[i]);"
    )
    lines.append("    if (d > max_abs) max_abs = d;")
    lines.append("    if (d > tol) ++bad;")
    lines.append("  }")
    for a in ptrs:
        lines.append(f"  HIP_CHECK(hipFree(d_{a.name}));")
    lines.append(
        f'  printf("[{case.name}] N=%d max_abs=%g bad=%d/%d tol=%g -> %s\\n",\n'
        f'         N, max_abs, bad, N, tol, bad == 0 ? "PASS" : "FAIL");'
    )
    lines.append("  return bad == 0 ? 0 : 1;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Emit / build
# ---------------------------------------------------------------------------


@dataclass
class DriverPaths:
    hip: Path
    main: Path
    exe: Path
    stem: str
    kernel_name: str
    ptr_line: int = 0  # 1-based line of the kernel signature inside the .hip


def emit_driver(case: DebugCase, *, arch: str, out_dir: Path) -> DriverPaths:
    """Write ``<stem>.hip`` and ``main.cpp`` for ``case`` into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(case.name)
    kernel = case.build(arch)
    hip_src = lower_kernel_to_hip(kernel, arch=arch)

    hip_path = out_dir / f"{stem}.hip"
    hip_path.write_text(hip_src, encoding="utf-8")

    # Locate the kernel signature line so we can print a ready-to-paste
    # rocgdb ``break <file>:<line>`` recipe.
    ptr_line = 0
    for i, ln in enumerate(hip_src.splitlines(), start=1):
        if ln.startswith(f"void {kernel.name}"):
            ptr_line = i
            break

    main_src = _render_main(case, kernel, hip_include=hip_path.name)
    main_path = out_dir / "main.cpp"
    main_path.write_text(main_src, encoding="utf-8")

    exe_path = out_dir / f"{stem}_dbg"
    return DriverPaths(
        hip=hip_path,
        main=main_path,
        exe=exe_path,
        stem=stem,
        kernel_name=kernel.name,
        ptr_line=ptr_line,
    )


def build_driver(paths: DriverPaths, *, arch: str, timeout_s: int = 240) -> None:
    """Compile ``main.cpp`` (+ included ``.hip``) with debug info via hipcc."""
    if shutil.which("hipcc") is None:
        raise SystemExit("hipcc not found in PATH (required to build the driver)")
    cmd = [
        "hipcc",
        "-g",
        "-O0",
        f"--offload-arch={arch}",
        str(paths.main),
        "-o",
        str(paths.exe),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        head = "\n".join(proc.stdout.splitlines()[:60])
        raise SystemExit(f"hipcc failed (rc={proc.returncode}):\n{head}")


def _print_recipe(paths: DriverPaths) -> None:
    exe = paths.exe
    line_hint = (
        f"{paths.hip.name}:{paths.ptr_line}" if paths.ptr_line else paths.hip.name
    )
    print("")
    print("rocgdb recipe:")
    print(f"  rocgdb --args {exe}")
    print(f"  (rocgdb) break {paths.kernel_name}        # break by kernel name")
    print(f"  (rocgdb) break {line_hint}   # or by source line")
    print("  (rocgdb) run")
    print("  (rocgdb) step        # single-step; 'info locals' to inspect")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default="elementwise.add", help="debug case name")
    p.add_argument("--arch", default="gfx942")
    p.add_argument("--out-dir", type=Path, default=Path("./rocgdb_dbg"))
    p.add_argument("--build", action="store_true", help="compile via hipcc -g -O0")
    p.add_argument("--run", action="store_true", help="run the built executable")
    p.add_argument("--list", action="store_true", help="list available cases")
    p.add_argument("--build-timeout-s", type=int, default=240)
    args = p.parse_args()

    if args.list:
        for c in _CASES:
            note = f"  ({c.notes})" if c.notes else ""
            print(f"{c.name:20} N={c.n}{note}")
        return 0

    case = _get_case(args.case)
    paths = emit_driver(case, arch=args.arch, out_dir=args.out_dir)
    print(f"[{args.arch}] emitted {paths.hip} and {paths.main}")

    if args.build or args.run:
        build_driver(paths, arch=args.arch, timeout_s=args.build_timeout_s)
        print(f"[{args.arch}] built {paths.exe}")

    if args.run:
        proc = subprocess.run([str(paths.exe)], text=True)
        _print_recipe(paths)
        return proc.returncode

    _print_recipe(paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
