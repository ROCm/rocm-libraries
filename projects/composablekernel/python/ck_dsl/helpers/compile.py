# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""One-shot `IR -> LLVM IR -> HSACO` compile pipeline.

`compile_kernel(kernel)` is the most-common end of the pipeline: it
takes a `KernelDef` produced by an instance builder and returns a
`KernelArtifact` containing:

  - `kernel`        : the original `KernelDef`
  - `ir_text`       : MLIR-style textual IR dump (for inspection)
  - `llvm_text`     : the AMDGPU LLVM IR text the comgr toolchain
                      consumes
  - `hsaco`         : the assembled HSA code object as bytes
  - `timings`       : per-stage `time.perf_counter()` measurements in
                      milliseconds (`ir_build`, `ir_lower_llvm`,
                      `comgr_bc`, `comgr_relocatable`, `comgr_executable`,
                      `total`)

Use `compile_kernel(...)` from a kernel-author script when:

  - You want to *run* the kernel: feed `artifact.hsaco` into
    `ck_dsl._hip_module.Runtime.load_module()`.

  - You want to *inspect* the lowered IR: write `artifact.llvm_text`
    next to a `.ll` file and run `llc -mtriple=amdgcn-amd-amdhsa
    -mcpu=gfx950 ...` or feed it through `clang -x ir -target ...`.

  - You want to *measure* codegen time without the per-call overhead
    of re-importing the comgr ctypes wrapper: the helper memoises the
    comgr load.

Typical use:

    from ck_dsl.helpers import compile_kernel
    artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")
    print(f"codegen total {artifact.timings['total']:.2f} ms")
    Path("out.hsaco").write_bytes(artifact.hsaco)
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..core.ir import KernelDef
from ..core.ir_print import print_ir
from ..core.lower_hip import lower_kernel_to_hip
from ..core.lower_llvm import lower_kernel_to_llvm
from ..core.passes import PassStats, optimize_kernel
from ..runtime.comgr import build_hsaco_from_llvm_ir


@dataclass
class KernelArtifact:
    """The compiled output of one `compile_kernel(...)` call."""

    kernel: KernelDef
    ir_text: str
    llvm_text: str
    hsaco: bytes
    timings: Dict[str, float] = field(default_factory=dict)
    pass_stats: PassStats = field(default_factory=PassStats)
    isa: str = "amdgcn-amd-amdhsa--gfx950"

    @property
    def kernel_name(self) -> str:
        return self.kernel.name

    @property
    def hsaco_bytes(self) -> int:
        return len(self.hsaco)


def compile_kernel(
    kernel: KernelDef,
    *,
    arch: Optional[str] = None,
    isa: str = "amdgcn-amd-amdhsa--gfx950",
    capture_ir_text: bool = True,
    optimize_ir: bool = False,
    backend: Optional[str] = None,
    spec: Optional[object] = None,
) -> KernelArtifact:
    """Lower `kernel` to a `KernelArtifact` ready for HIP module load.

    `arch` is the preferred way to select the target (e.g. ``"gfx942"``,
    ``"gfx950"``): when given, the comgr ISA triple is derived from
    :class:`ck_dsl.core.arch.ArchTarget`, so callers don't hand-spell the
    triple. `arch` takes precedence over `isa`.

    `isa` is the raw comgr target triple and stays accepted for backward
    compatibility; `gfx950` is the historical default every example uses.

    `capture_ir_text` controls whether the MLIR-style textual dump is
    populated. Disable for tight sweep loops where the dump is
    discarded.

    `backend` selects which engine produces the lowered AMDGPU ``.ll``:
    ``"python"`` (native, the default and byte-identical to historical
    behaviour), ``"cpp"`` (route through the C++ engine binding), or
    ``"both"`` (lower with both and assert byte-equality, returning the
    Python result). When unset, the ``CK_DSL_BACKEND`` environment
    variable is consulted, else ``"python"``. Routing to the cpp/both
    engines requires the instance-level ``spec`` (the C++ engine lowers
    from the spec, not the post-IR ``KernelDef``); pass it via ``spec``.
    Only the universal-GEMM family is wired to the cpp engine today; any
    other ``spec`` type raises a clear error rather than silently
    diverging.
    """
    if arch is not None:
        from ..core.arch import ArchTarget

        isa = ArchTarget.from_gfx(arch).isa_triple
        _lower_arch = arch
    else:
        # Derive the lowering arch from the isa triple so the ISA backend
        # (datalayout/triple/waitcnt) matches the comgr target even when a
        # caller passes isa= directly.
        from ..core.arch import arch_from_isa, known_arches

        _gfx = arch_from_isa(isa)
        _lower_arch = _gfx if _gfx in known_arches() else None

    timings: Dict[str, float] = {}

    t0 = time.perf_counter()
    pass_stats = optimize_kernel(kernel) if optimize_ir else PassStats()
    t_pass = time.perf_counter()
    ir_text = print_ir(kernel) if capture_ir_text else ""
    t1 = time.perf_counter()
    llvm_text = _lower_llvm_via_backend(
        kernel, arch=_lower_arch, backend=backend, spec=spec
    )
    t2 = time.perf_counter()
    hsaco, comgr_t = build_hsaco_from_llvm_ir(
        llvm_text, isa=isa, options=_comgr_options_for_kernel(kernel)
    )
    t3 = time.perf_counter()

    timings["ir_opt"] = (t_pass - t0) * 1000.0
    timings["ir_build"] = (t1 - t_pass) * 1000.0
    timings["ir_lower_llvm"] = (t2 - t1) * 1000.0
    timings["comgr_bc"] = comgr_t.bc * 1000.0
    timings["comgr_relocatable"] = comgr_t.relocatable * 1000.0
    timings["comgr_executable"] = comgr_t.executable * 1000.0
    timings["total"] = (t3 - t0) * 1000.0

    return KernelArtifact(
        kernel=kernel,
        ir_text=ir_text,
        llvm_text=llvm_text,
        hsaco=hsaco,
        timings=timings,
        pass_stats=pass_stats,
        isa=isa,
    )


def _lower_llvm_via_backend(
    kernel: KernelDef,
    *,
    arch: Optional[str],
    backend: Optional[str],
    spec: Optional[object],
) -> str:
    """Produce the AMDGPU ``.ll`` text through the selected backend.

    The default (``backend`` unset and ``CK_DSL_BACKEND`` unset, resolving
    to ``"python"``) calls the native lowerer directly with no behaviour
    change. ``"cpp"``/``"both"`` route through the C++ engine binding and
    require the instance-level ``spec``; ``"both"`` additionally asserts
    the two engines agree before returning the Python result.
    """
    from ..core.backend import BACKEND_PYTHON, lower_universal_gemm, resolve_backend

    chosen = resolve_backend(backend)
    if chosen == BACKEND_PYTHON:
        # Native path -- byte-identical to the historical default.
        return lower_kernel_to_llvm(kernel, arch=arch)

    if spec is None:
        raise ValueError(
            f"compile_kernel(backend={chosen!r}) needs the instance-level spec "
            "to route through the C++ engine (it lowers from the spec, not the "
            "post-IR KernelDef); pass spec=<UniversalGemmSpec>."
        )

    _arch = arch or "gfx950"
    result = lower_universal_gemm(spec, arch=_arch, backend=chosen)
    return result.llvm_text


def _comgr_options_for_kernel(kernel: KernelDef) -> List[str]:
    """Return AMDGPU codegen options implied by kernel attrs."""

    options = ["-O3"]
    agpr_alloc = kernel.attrs.get("agpr_alloc")
    if kernel.attrs.get("mfma_vgpr_form") or _is_zero_agpr_alloc(agpr_alloc):
        options.extend(["-mllvm", "-amdgpu-mfma-vgpr-form"])
    return options


def _is_zero_agpr_alloc(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        parts = value.strip().split(",")
    elif isinstance(value, (tuple, list)):
        parts = list(value)
    else:
        return False
    if len(parts) != 2:
        return False
    try:
        return int(parts[0]) == 0 and int(parts[1]) == 0
    except (TypeError, ValueError):
        return False


def compile_kernel_via_hipcc(
    kernel: KernelDef,
    *,
    arch: str = "gfx950",
    extra_flags: Optional[List[str]] = None,
    timeout_s: int = 240,
) -> KernelArtifact:
    """Lower ``kernel`` to HIP C++, compile through ``hipcc --genco``, and
    return a :class:`KernelArtifact` whose ``hsaco`` is the hipcc output.

    Use this **only** when the LLVM-direct pipeline (``compile_kernel``)
    is leaving performance on the table for a specific workload. The HIP
    path goes through the full clang frontend + AMDGPU backend, which
    sometimes generates better-scheduled code for long-running attention
    kernels (we measured a ~5% win on prefill_b4_q1000 with the kernel
    in ``instances/attention_tiled_2d.py``). The trade-offs:

      - Compile is ~5× slower (~450ms vs ~90ms via libamd_comgr) due
        to the heavier clang frontend.
      - Requires ``hipcc`` in ``$PATH`` (build-time only, since the HSACO
        bytes are cacheable and identical to the LLVM-direct path's
        runtime ABI).
      - The HIP debug backend has narrower op coverage than the LLVM
        backend; verify the kernel actually lowers via ``lower_kernel_to_hip``
        before relying on this path.

    Returns the same ``KernelArtifact`` shape as :func:`compile_kernel`;
    ``ir_text`` is the textual MLIR-style IR, ``llvm_text`` is empty
    (this path doesn't go through the LLVM-direct lowering), ``hsaco``
    is the hipcc output, and ``timings`` records the lower + hipcc steps.
    """
    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    ir_text = print_ir(kernel)
    t1 = time.perf_counter()
    hip_src = lower_kernel_to_hip(kernel, arch=arch)
    t2 = time.perf_counter()
    flags = ["-O3"]
    if extra_flags:
        flags.extend(extra_flags)
    with tempfile.TemporaryDirectory() as td:
        stem = kernel.name.replace(".", "_")[:80] or "kernel"
        src_path = Path(td) / f"{stem}.hip"
        hsaco_path = Path(td) / f"{stem}.hsaco"
        src_path.write_text(hip_src, encoding="utf-8")
        proc = subprocess.run(
            [
                "hipcc",
                f"--offload-arch={arch}",
                "--genco",
                *flags,
                str(src_path),
                "-o",
                str(hsaco_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"hipcc failed for kernel '{kernel.name}' (arch={arch}):\n"
                f"--- stdout ---\n{proc.stdout[-2000:]}\n"
                f"--- stderr ---\n{proc.stderr[-2000:]}"
            )
        hsaco = hsaco_path.read_bytes()
    t3 = time.perf_counter()
    timings["ir_build"] = (t1 - t0) * 1000.0
    timings["ir_lower_hip"] = (t2 - t1) * 1000.0
    timings["hipcc"] = (t3 - t2) * 1000.0
    timings["total"] = (t3 - t0) * 1000.0
    isa = f"amdgcn-amd-amdhsa--{arch}"
    return KernelArtifact(
        kernel=kernel,
        ir_text=ir_text,
        llvm_text="",  # HIP path doesn't produce LLVM IR text directly
        hsaco=hsaco,
        timings=timings,
        pass_stats=PassStats(),
        isa=isa,
    )
