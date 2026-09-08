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
    `rocke.runtime.Runtime.load_module()`.

  - You want to *inspect* the lowered IR: write `artifact.llvm_text`
    next to a `.ll` file and run `llc -mtriple=amdgcn-amd-amdhsa
    -mcpu=gfx950 ...` or feed it through `clang -x ir -target ...`.

  - You want to *measure* codegen time without the per-call overhead
    of re-importing the comgr ctypes wrapper: the helper memoises the
    comgr load.

Typical use:

    from rocke.helpers import compile_kernel
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

from ..core.arch import (
    ArchTarget,
    arch_from_isa,
    base_arch_from_target_id,
    compiler_target_from_target_id,
    known_arches,
    target_id_from_isa,
)
from ..core.codegen_policy import codegen_policy_for_kernel
from ..core.ir import KernelDef
from ..core.ir_print import print_ir
from ..core.lower_hip import lower_kernel_to_hip
from ..core.lower_llvm import _lower_kernel_to_llvm_python
from ..core.passes import PassStats, optimize_kernel
from ..runtime.comgr import build_hsaco_from_llvm_ir


@dataclass
class KernelArtifact:
    """Output from ``compile_kernel`` or ``compile_kernel_via_hipcc``.

    ``isa`` records the COMGR ISA name for the compiler target used to build
    ``hsaco``, including any target features. It is not a device query result.
    """

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

    Pass a target ID through `arch`, such as ``"gfx942"`` or ``"gfx1250-strict"``.
    Alternatively, pass a COMGR ISA name through `isa`, such as
    ``"amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"``. `arch` takes precedence;
    if neither is supplied, the target is ``gfx950``. No GPU query is made.

    The target helpers remove profile suffixes such as ``-strict`` from the
    compiler target and retain features such as ``:sramecc+:xnack-``. The base
    architecture selects rocKE lowering. The resulting COMGR ISA name is
    passed to `build_hsaco_from_llvm_ir` and recorded in `KernelArtifact.isa`.
    COMGR checks compiler support when compilation runs.

    `capture_ir_text` controls whether the MLIR-style textual dump is
    populated. Disable for tight sweep loops where the dump is
    discarded.

    `backend` selects which engine produces the lowered AMDGPU ``.ll``:
    ``"python"`` (the native lowerer, byte-identical to the historical
    path), ``"cpp"`` (serialize the kernel and lower it through the C++
    engine), or ``"both"`` (lower with both and assert byte-equality,
    returning the Python result). When unset, the ``ROCKE_BACKEND``
    environment variable is consulted, else the package default (currently
    the **C++** engine). The cpp/both paths are **family-agnostic**: they
    lower from the built kernel's serialized ``ck.dsl.ir/v1`` IR, so no
    per-family wiring is needed (the legacy ``spec`` argument is accepted
    for backward compatibility but is no longer consulted).
    """
    if arch is not None:
        _lower_arch = base_arch_from_target_id(arch)
        compiler_target = compiler_target_from_target_id(arch)
        base_isa = ArchTarget.from_gfx(_lower_arch).isa_triple
        isa = f"{base_isa[: -len(_lower_arch)]}{compiler_target}"
    else:
        # Derive both names from the caller's ISA name. Use the base architecture
        # for lowering and preserve compiler features in the COMGR ISA name.
        target_id = target_id_from_isa(isa)
        compiler_target = compiler_target_from_target_id(target_id)
        isa = f"{isa[: -len(target_id)]}{compiler_target}"
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

    The default (``backend`` unset and ``ROCKE_BACKEND`` unset) resolves to
    the package default backend. Lowering is family-agnostic: it goes through
    the serialized ``ck.dsl.ir/v1`` artifact of the built ``kernel``, so the
    instance-level ``spec`` is NOT required for any family.

      - ``"python"`` calls the native lowerer (byte-identical historical path);
      - ``"cpp"`` serializes ``kernel`` and lowers it through the C++ engine,
        falling back to the native lowerer (recorded) on engine unavailability
        or rejection;
      - ``"both"`` lowers with both and asserts byte-equality.

    ``backend`` overrides the env/default precedence when given. ``spec`` is
    accepted for backward compatibility but is no longer consulted (the
    serialized-IR hand-off makes it unnecessary).
    """
    from ..core.backend import lower_kernel_via_backend, resolve_backend

    chosen = resolve_backend(backend)
    if backend is not None and chosen != resolve_backend():
        # An explicit backend= argument was passed that differs from the
        # resolved default; honour it by lowering against that backend
        # directly rather than the chokepoint's env-resolved default.
        import os

        prev = os.environ.get("ROCKE_BACKEND")
        os.environ["ROCKE_BACKEND"] = chosen
        try:
            return lower_kernel_via_backend(
                kernel,
                arch=arch,
                python_lower=_lower_kernel_to_llvm_python,
            )
        finally:
            if prev is None:
                os.environ.pop("ROCKE_BACKEND", None)
            else:
                os.environ["ROCKE_BACKEND"] = prev

    return lower_kernel_via_backend(
        kernel,
        arch=arch,
        python_lower=_lower_kernel_to_llvm_python,
    )


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
    """Lower ``kernel`` to HIP C++ and compile it with ``hipcc --genco``.

    ``arch`` accepts the same target IDs as :func:`compile_kernel`. The base
    architecture is passed to ``lower_kernel_to_hip``; the compiler target
    is passed to hipcc as ``--offload-arch``. Neither is discovered from a GPU.

    Requires hipcc on ``PATH`` and a kernel supported by the HIP lowerer.
    An explicit scheduler policy is rejected because this path does not emit
    the LLVM function attribute used by the COMGR path.

    Returns a :class:`KernelArtifact` with hipcc's output in ``hsaco``, an
    empty ``llvm_text``, and timings for HIP lowering and compilation.
    """
    policy = codegen_policy_for_kernel(kernel)
    if policy.scheduler_strategy is not None:
        raise ValueError(
            "compile_kernel_via_hipcc does not support scheduler_strategy; "
            "use compile_kernel so the policy is emitted as an LLVM function attribute"
        )

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    ir_text = print_ir(kernel)
    t1 = time.perf_counter()
    lower_arch = base_arch_from_target_id(arch)
    compiler_target = compiler_target_from_target_id(arch)
    hip_src = lower_kernel_to_hip(kernel, arch=lower_arch)
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
                f"--offload-arch={compiler_target}",
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
    isa = f"amdgcn-amd-amdhsa--{compiler_target}"
    return KernelArtifact(
        kernel=kernel,
        ir_text=ir_text,
        llvm_text="",  # HIP path doesn't produce LLVM IR text directly
        hsaco=hsaco,
        timings=timings,
        pass_stats=PassStats(),
        isa=isa,
    )


def emit_device_llvm_ir_via_hipcc(
    kernel: KernelDef,
    *,
    arch: str = "gfx950",
    extra_flags: Optional[List[str]] = None,
    timeout_s: int = 120,
) -> str:
    """Lower ``kernel`` to HIP C++ and ask hipcc to emit device LLVM IR.

    Target selection matches :func:`compile_kernel_via_hipcc`: the base
    architecture selects HIP lowering and the compiler target becomes
    hipcc's ``--offload-arch`` argument.

    Uses ``-S -emit-llvm --cuda-device-only``. Tests can compare the returned
    ``target datalayout`` with rocKE's LLVM lowering for the same target.

    Args:
        kernel: The kernel to lower.
        arch: Target ID, such as ``gfx950`` or ``gfx1250-strict``.
        extra_flags: hipcc flags appended after ``-O3``.
        timeout_s: Subprocess timeout in seconds.

    Returns:
        The generated ``.ll`` files joined with filename comments.

    Raises:
        RuntimeError: If hipcc fails or produces no ``.ll`` files.
        FileNotFoundError: If hipcc cannot be located.
        subprocess.TimeoutExpired: If hipcc exceeds ``timeout_s``.
    """
    lower_arch = base_arch_from_target_id(arch)
    compiler_target = compiler_target_from_target_id(arch)
    hip_src = lower_kernel_to_hip(kernel, arch=lower_arch)
    flags = ["-O3"]
    if extra_flags:
        flags.extend(extra_flags)
    with tempfile.TemporaryDirectory() as td:
        stem = kernel.name.replace(".", "_")[:80] or "kernel"
        src_path = Path(td) / f"{stem}.hip"
        ll_path = Path(td) / f"{stem}.ll"
        src_path.write_text(hip_src, encoding="utf-8")
        proc = subprocess.run(
            [
                "hipcc",
                f"--offload-arch={compiler_target}",
                "-S",
                "-emit-llvm",
                "--cuda-device-only",
                *flags,
                str(src_path),
                "-o",
                str(ll_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"hipcc -emit-llvm failed for kernel '{kernel.name}' (arch={arch}):\n"
                f"--- stdout ---\n{proc.stdout[-2000:]}\n"
                f"--- stderr ---\n{proc.stderr[-2000:]}"
            )
        # hipcc may emit multiple .ll files (per-arch or per-module splits);
        # collect all of them.
        ll_files = sorted(Path(td).glob("*.ll"))
        if not ll_files:
            raise RuntimeError(
                f"hipcc -emit-llvm produced no .ll files in {td} (expected at least one)"
            )
        parts = []
        for f in ll_files:
            parts.append(f"; ===== {f.name} =====\n")
            parts.append(f.read_text(encoding="utf-8"))
        return "".join(parts)
