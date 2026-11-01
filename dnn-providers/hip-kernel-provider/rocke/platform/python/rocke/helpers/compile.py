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

Existing callers choose targets before calling this module:

  - ``examples/common/bake_off_direct_conv_4c.py`` forwards ``--isa`` as
    ``isa=`` when present; otherwise it forwards ``--arch`` (default ``gfx950``).
  - ``benchmark/gemm/fp16_rcr_sweep.py`` starts with ``--arch`` or
    ``GemmSweepConfig.arch``, carries it through dispatch records, then passes
    it as ``arch=`` in ``compile_variant``.
  - ``instances/common/moe_sorting.py`` resolves ``MoeSortingLauncher.arch``
    before compilation: an explicit value wins; otherwise ``get_device_arch()``
    queries HIP, with ``gfx950`` as the fallback if discovery fails.

With ``arch=``, ``compile_kernel`` builds the COMGR ISA name from
``ArchTarget.isa_triple`` and the derived compiler target. With neither argument,
it uses its own ``isa="amdgcn-amd-amdhsa--gfx950"`` default without querying HIP.

Example with a fixed target:

    from rocke.helpers import compile_kernel
    artifact = compile_kernel(kernel, arch="gfx950")
    print(f"codegen total {artifact.timings['total']:.2f} ms")
    Path("out.hsaco").write_bytes(artifact.hsaco)

A builder whose launch needs more than one kernel (a split-KV decode and the
kernel merging its splits, say) returns a :class:`KernelModule` instead: the
primary kernel plus role-keyed companions. :func:`compile_module` lowers each
kernel exactly as ``compile_kernel`` would and links them into ONE code object
that exports every symbol:

    module = KernelModule(primary=decode, companions={"merge": merge})
    artifact = compile_module(module, arch="gfx11-generic")
    artifact.kernel_name        # decode.name
    artifact.companion_names    # {"merge": merge.name}
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

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
from ..runtime.comgr import build_hsaco_from_llvm_ir, build_hsaco_from_llvm_modules

# Kernel attribute (bool) a builder sets on a kernel whose code generation
# depends on what the compiling process compiled before (it emits
# ``iglp_opt(1)``; see ``build_hsaco_from_llvm_ir``): ``compile_kernel`` then
# compiles it in a fresh process. Compile policy only: no lowerer reads it, so
# the LLVM IR is the same with or without it.
COMGR_FRESH_PROCESS_ATTR = "comgr_fresh_process"

# A builder's request for the code-object backend of one kernel (a knob such as
# ``qwen38_mono.framework.FrameworkKnobs.code_object_backend``). It is a Python
# attribute of the ``KernelDef`` object, not a kernel attribute: the IR text,
# its serialization and the LLVM lowering never see it, so an unmarked kernel
# and a marked one lower to the same bytes. ``compile_kernel`` uses it when the
# call names no ``code_object_backend``.
_CODE_OBJECT_BACKEND_REQUEST = "_rocke_code_object_backend"

# The direct code-object backend (``rocke.runtime.code_object``, the Rust
# compiler of rocKE IR; ``code_object_backend="direct"``):
# - ``DIRECT_LEVERS_ATTR``: a kernel's own direct-compiler levers (a list or a
#   ``,`` / ``+`` joined string of ``ROCKE_DIRECT_LEVERS`` names), joined to the
#   environment's for its compile only; not IR (stripped before serialization).
# - ``DIRECT_SCOPES_ATTR`` + the op attribute ``SCOPE_OP_ATTR``: per-op lever
#   scopes by name (``"<scope>:<lever>+<lever>;..."`` on the kernel, the scope
#   name on its ops; ``qwen38_mono.persist.knobs.RustScope``). The direct
#   backend turns them into the compiler's ``dlevers`` op scopes; the LLVM
#   lowering ignores both, so COMGR objects do not change.
# - ``DIRECT_DEFAULT_LEVERS``: levers the direct backend turns on for every
#   kernel (D0.6; rust-track INTEGRATION.md 5 items 1-2): ``acqfence`` (an
#   acquire-only fence lowers as LLVM's: wait + ``buffer_inv``, no L2
#   write-back) and ``bufoffany`` (LLVM's buffer offset split).
#   ``ROCKE_DIRECT_DEFAULT_LEVERS`` replaces the set (``""``: none, the plain
#   direct object).
DIRECT_LEVERS_ATTR = "direct_levers"
DIRECT_SCOPES_ATTR = "direct_scopes"
SCOPE_OP_ATTR = "rocke.scope"
DIRECT_DEFAULT_LEVERS = ("acqfence", "bufoffany")
_DIRECT_DEFAULTS_ENV = "ROCKE_DIRECT_DEFAULT_LEVERS"


def request_code_object_backend(kernel: KernelDef, backend: str) -> KernelDef:
    """Mark ``kernel`` to compile with ``backend`` (``"comgr"`` | ``"direct"``)
    when ``compile_kernel`` is not told otherwise; returns ``kernel``."""
    if backend not in ("comgr", "direct"):
        raise ValueError(f"code-object backend must be 'comgr' or 'direct', not {backend!r}")
    setattr(kernel, _CODE_OBJECT_BACKEND_REQUEST, backend)
    return kernel


def requested_code_object_backend(kernel: KernelDef) -> Optional[str]:
    """The backend :func:`request_code_object_backend` marked ``kernel`` with,
    or ``None``."""
    return getattr(kernel, _CODE_OBJECT_BACKEND_REQUEST, None)


@dataclass
class KernelArtifact:
    """Output from ``compile_kernel`` or ``compile_kernel_via_hipcc``.

    ``isa`` records the COMGR ISA name for the compiler target used to build
    ``hsaco``, including any target features. It is not a device query result.
    ``code_object_backend`` is the compiler that built ``hsaco`` (``"comgr"``
    or ``"direct"``); when the direct backend was asked for but does not
    compile the kernel for the target, COMGR built it and ``direct_fallback``
    holds the direct compiler's reason.
    """

    kernel: KernelDef
    ir_text: str
    llvm_text: str
    hsaco: bytes
    timings: Dict[str, float] = field(default_factory=dict)
    pass_stats: PassStats = field(default_factory=PassStats)
    isa: str = "amdgcn-amd-amdhsa--gfx950"
    code_object_backend: str = "comgr"
    direct_fallback: str = ""

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
    code_object_backend: Optional[str] = None,
) -> KernelArtifact:
    """Lower `kernel` to a `KernelArtifact` ready for HIP module load.

    Pass a target ID through `arch`, such as ``"gfx942"`` or ``"gfx1250-strict"``.
    Alternatively, pass a COMGR ISA name through `isa`, such as
    ``"amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-"``. `arch` takes precedence;
    if neither is supplied, the target is ``gfx950``. This function does not
    query a GPU; callers may resolve `arch` from HIP before calling it. See the
    module docstring for examples of CLI, configuration, and launcher inputs.

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

    A kernel carrying ``COMGR_FRESH_PROCESS_ATTR`` is compiled in a fresh
    process (``build_hsaco_from_llvm_ir(..., fresh_process=True)``); every
    other kernel compiles in this process.

    `code_object_backend` (``"comgr"`` | ``"direct"``): when unset, the
    kernel's :func:`request_code_object_backend` mark, else
    ``ROCKE_CODEOBJ_BACKEND``, else COMGR. ``"direct"`` compiles the
    serialized rocKE IR with the Rust direct compiler (no LLVM; see
    ``DIRECT_LEVERS_ATTR`` for its levers and scopes); when it refuses the
    kernel (``DirectUnsupported``), COMGR compiles it and
    ``artifact.direct_fallback`` holds the refusal (``ROCKE_DIRECT_STRICT=1``
    raises instead). The COMGR path is the same code with or without this.
    """
    _lower_arch, isa = _resolve_compile_target(arch, isa)
    if _selected_code_object_backend(code_object_backend, kernel) == "direct":
        from ..runtime.code_object import DirectUnsupported

        try:
            return _compile_kernel_direct(
                kernel, isa=isa, capture_ir_text=capture_ir_text, optimize_ir=optimize_ir
            )
        except DirectUnsupported as refused:
            # ``ROCKE_DIRECT_STRICT=1`` (validation runs): a refusal is an error, not a COMGR fallback.
            if os.environ.get("ROCKE_DIRECT_STRICT", "0") == "1":
                raise
            # The passes ran on `kernel` before the direct compiler refused it.
            artifact = compile_kernel(
                kernel,
                arch=arch,
                isa=isa,
                capture_ir_text=capture_ir_text,
                optimize_ir=False,
                backend=backend,
                spec=spec,
                code_object_backend="comgr",
            )
            artifact.pass_stats = getattr(refused, "pass_stats", artifact.pass_stats)
            artifact.direct_fallback = str(refused)
            return artifact

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
        llvm_text,
        isa=isa,
        options=_comgr_options_for_kernel(kernel),
        fresh_process=bool(kernel.attrs.get(COMGR_FRESH_PROCESS_ATTR)),
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


def _selected_code_object_backend(code_object_backend: Optional[str], kernel: KernelDef) -> str:
    """``code_object_backend``, else the kernel's mark, else the environment
    (``ROCKE_CODEOBJ_BACKEND``), else ``"comgr"``; validated."""
    if code_object_backend is None:
        code_object_backend = requested_code_object_backend(kernel)
    if code_object_backend is None and not os.environ.get("ROCKE_CODEOBJ_BACKEND"):
        return "comgr"
    from ..runtime.code_object import resolve_code_object_backend

    return resolve_code_object_backend(code_object_backend)


def _lever_names(value) -> List[str]:
    """A lever list (a list / tuple or a ``,`` / ``+`` joined string) as names."""
    if not value:
        return []
    if isinstance(value, str):
        value = value.replace("+", ",").split(",")
    return [str(x).strip() for x in value if str(x).strip()]


def direct_default_levers() -> List[str]:
    """The levers the direct backend turns on for every kernel:
    ``ROCKE_DIRECT_DEFAULT_LEVERS`` if set (``""``: none), else
    ``DIRECT_DEFAULT_LEVERS``."""
    env = os.environ.get(_DIRECT_DEFAULTS_ENV)
    return _lever_names(env) if env is not None else list(DIRECT_DEFAULT_LEVERS)


def parse_direct_scopes(text: str) -> Dict[str, List[str]]:
    """``attrs["direct_scopes"]`` (``"<scope>:<lever>+<lever>;..."``) as
    scope name -> lever names (``-name`` turns a kernel lever off)."""
    out: Dict[str, List[str]] = {}
    for part in filter(None, (x.strip() for x in str(text or "").split(";"))):
        name, sep, levers = part.partition(":")
        if not sep or not name.strip():
            raise ValueError(f"direct_scopes: {part!r} is not <scope>:<levers>")
        out[name.strip()] = _lever_names(levers)
    return out


def direct_source(kernel: KernelDef) -> KernelDef:
    """The kernel the direct compiler reads: a copy of ``kernel`` (which keeps
    its attributes for a COMGR fallback) whose ops carry their named scope's
    levers as the compiler's ``dlevers`` (an op's own ``dlevers``, from
    ``IRBuilder.lever_scope``, wins), without any ``rocke.*`` op attribute
    (markers such as ``rocke.kv`` / ``rocke.scope``: IR the compiler does not
    read) and without the kernel's ``direct_scopes`` / ``direct_levers``.
    ``kernel`` itself when there is nothing to change."""
    from ..core.ir import LEVER_SCOPE_ATTR, Region

    scopes = parse_direct_scopes(kernel.attrs.get(DIRECT_SCOPES_ATTR, ""))
    text = {k: ",".join(v) for k, v in scopes.items() if v}

    def op_of(op):
        attrs = op.attrs
        if any(k.startswith("rocke.") for k in attrs):
            sc = attrs.get(SCOPE_OP_ATTR)
            attrs = {k: v for k, v in attrs.items() if not k.startswith("rocke.")}
            if sc is not None and LEVER_SCOPE_ATTR not in attrs and text.get(str(sc)):
                attrs[LEVER_SCOPE_ATTR] = text[str(sc)]
        regions = [region_of(r) for r in op.regions] if op.regions else op.regions
        if attrs is op.attrs and all(a is b for a, b in zip(regions, op.regions)):
            return op
        return dataclasses.replace(op, attrs=attrs, regions=regions)

    def region_of(region):
        ops = [op_of(o) for o in region.ops]
        if all(a is b for a, b in zip(ops, region.ops)):
            return region
        return Region(label=region.label, ops=ops)

    body = region_of(kernel.body)
    attrs = {k: v for k, v in kernel.attrs.items() if k not in (DIRECT_SCOPES_ATTR, DIRECT_LEVERS_ATTR)}
    if body is kernel.body and len(attrs) == len(kernel.attrs):
        return kernel
    return dataclasses.replace(kernel, body=body, attrs=attrs, params=list(kernel.params))


def _compile_kernel_direct(
    kernel: KernelDef,
    *,
    isa: str,
    capture_ir_text: bool,
    optimize_ir: bool,
) -> KernelArtifact:
    """``compile_kernel`` for ``code_object_backend="direct"``: no LLVM text."""
    from ..core.ir_serialize import serialize
    from ..runtime.code_object import DirectUnsupported, build_direct_rocke_ir_hsaco
    from ..runtime.direct_lift import lift, lift_enabled

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    pass_stats = optimize_kernel(kernel) if optimize_ir else PassStats()
    t_pass = time.perf_counter()
    ir_text = print_ir(kernel) if capture_ir_text else ""
    t1 = time.perf_counter()
    # The backend's defaults and the kernel's own levers (``attrs["direct_levers"]``)
    # join the environment's for this compile; named op scopes become ``dlevers``
    # before the lifting (a lifted asm statement keeps its scope).
    levers = direct_default_levers() + _lever_names(kernel.attrs.get(DIRECT_LEVERS_ATTR))
    source = direct_source(kernel)
    # Inline-asm idioms the direct compiler lowers as IR ops, on a copy
    # (`kernel` itself keeps its asm for a COMGR fallback).
    if lift_enabled():
        source = lift(source)
    serialized = serialize(source)
    t2 = time.perf_counter()
    saved = os.environ.get("ROCKE_DIRECT_LEVERS")
    if levers:
        merged = _lever_names(saved)
        os.environ["ROCKE_DIRECT_LEVERS"] = ",".join(dict.fromkeys(merged + levers))
    try:
        hsaco, codeobj_t, _info, _resources = build_direct_rocke_ir_hsaco(serialized, isa=isa)
    except DirectUnsupported as refused:
        refused.pass_stats = pass_stats
        raise
    finally:
        if levers:
            if saved is None:
                os.environ.pop("ROCKE_DIRECT_LEVERS", None)
            else:
                os.environ["ROCKE_DIRECT_LEVERS"] = saved
    t3 = time.perf_counter()

    timings["ir_opt"] = (t_pass - t0) * 1000.0
    timings["ir_build"] = (t1 - t_pass) * 1000.0
    timings["ir_serialize"] = (t2 - t1) * 1000.0
    timings["ir_lower_llvm"] = 0.0
    timings["comgr_bc"] = 0.0
    timings["comgr_relocatable"] = 0.0
    timings["comgr_executable"] = 0.0
    timings["codeobj_executable"] = codeobj_t.executable * 1000.0
    timings["total"] = (t3 - t0) * 1000.0
    return KernelArtifact(
        kernel=kernel,
        ir_text=ir_text,
        llvm_text="",
        hsaco=hsaco,
        timings=timings,
        pass_stats=pass_stats,
        isa=isa,
        code_object_backend="direct",
    )


# A companion role names metadata fields (`<role>_symbol`, `<role>_signature`)
# in packaged descriptors, so it is held to a lowercase identifier.
_COMPANION_ROLE = re.compile(r"[a-z][a-z0-9_]*\Z")


@dataclass(frozen=True)
class KernelModule:
    """Several kernels that ship, load and launch as ONE code object.

    ``primary`` is the kernel a catalog descriptor names as its symbol.
    ``companions`` maps a role -- a lowercase identifier such as ``"merge"`` --
    to each further kernel the same object exports. A host that launched the
    primary from a loaded module finds a companion in that same module by its
    name; packaged descriptors carry it as ``<role>_symbol`` metadata.

    Every kernel name must be distinct (they share one symbol table), and there
    must be at least one companion: a lone kernel is a plain ``KernelDef``.
    """

    primary: KernelDef
    companions: Mapping[str, KernelDef]

    def __post_init__(self) -> None:
        if not isinstance(self.primary, KernelDef):
            raise TypeError(
                f"KernelModule.primary must be a KernelDef, not "
                f"{type(self.primary).__name__}"
            )
        companions = dict(self.companions)
        if not companions:
            raise ValueError(
                "KernelModule needs at least one companion; return the KernelDef "
                "itself for a single kernel"
            )
        for role, kernel in companions.items():
            if not isinstance(role, str) or not _COMPANION_ROLE.match(role):
                raise ValueError(
                    f"companion role {role!r} must be a lowercase identifier "
                    f"([a-z][a-z0-9_]*): it names descriptor metadata fields"
                )
            if not isinstance(kernel, KernelDef):
                raise TypeError(
                    f"companion {role!r} must be a KernelDef, not "
                    f"{type(kernel).__name__}"
                )
        names = [self.primary.name] + [k.name for k in companions.values()]
        repeated = sorted({n for n in names if names.count(n) > 1})
        if repeated:
            raise ValueError(
                f"KernelModule kernels share the name(s) {repeated}; one code "
                f"object exports each symbol once"
            )
        object.__setattr__(self, "companions", companions)

    @property
    def kernels(self) -> Tuple[KernelDef, ...]:
        """Every kernel, primary first, then companions in declaration order."""
        return (self.primary, *self.companions.values())


@dataclass
class ModuleArtifact:
    """Output from :func:`compile_module`: one code object, several kernels.

    ``ir_texts`` and ``llvm_texts`` follow ``module.kernels`` order. ``isa``
    is the COMGR ISA name, as in :class:`KernelArtifact`.
    """

    module: KernelModule
    ir_texts: Tuple[str, ...]
    llvm_texts: Tuple[str, ...]
    hsaco: bytes
    timings: Dict[str, float] = field(default_factory=dict)
    pass_stats: Tuple[PassStats, ...] = ()
    isa: str = "amdgcn-amd-amdhsa--gfx950"

    @property
    def kernel_name(self) -> str:
        """The primary kernel's symbol."""
        return self.module.primary.name

    @property
    def companion_names(self) -> Dict[str, str]:
        """``{role: symbol}`` for every companion, in declaration order."""
        return {role: k.name for role, k in self.module.companions.items()}

    @property
    def hsaco_bytes(self) -> int:
        return len(self.hsaco)


def compile_module(
    module: KernelModule,
    *,
    arch: Optional[str] = None,
    isa: str = "amdgcn-amd-amdhsa--gfx950",
    capture_ir_text: bool = True,
    optimize_ir: bool = False,
    backend: Optional[str] = None,
) -> ModuleArtifact:
    """Lower every kernel of `module` and link them into one code object.

    Target resolution, IR passes and the lowering backend are exactly
    :func:`compile_kernel`'s, applied per kernel, so each kernel's ``.ll`` is
    the text ``compile_kernel`` would produce for it alone (and the cpp/both
    backends hold the same byte-identity contract per kernel). The texts are
    then linked at bitcode level and code-generated once, see
    :func:`rocke.runtime.comgr.build_hsaco_from_llvm_modules`.

    COMGR codegen options are module-wide, so every kernel must imply the same
    options (see ``_comgr_options_for_kernel``); a disagreement raises
    ``ValueError`` rather than compiling one kernel under another's flags.
    A kernel carrying ``COMGR_FRESH_PROCESS_ATTR`` raises ``ValueError`` too:
    a linked module compiles in this process, which that kernel must not.
    Target resolution and per-kernel lowering can also raise ``ValueError``,
    as in :func:`compile_kernel`, before any link.
    """
    lower_arch, isa = _resolve_compile_target(arch, isa)
    kernels = module.kernels

    fresh = [k.name for k in kernels if k.attrs.get(COMGR_FRESH_PROCESS_ATTR)]
    if fresh:
        raise ValueError(
            f"KernelModule kernels {fresh} need a fresh-process compile "
            f"({COMGR_FRESH_PROCESS_ATTR}); compile them with compile_kernel"
        )

    options = [_comgr_options_for_kernel(k) for k in kernels]
    if any(o != options[0] for o in options[1:]):
        detail = ", ".join(f"{k.name}: {o}" for k, o in zip(kernels, options))
        raise ValueError(
            f"KernelModule kernels need one set of COMGR codegen options to share "
            f"a code object; they imply {detail}"
        )

    timings: Dict[str, float] = {}
    t0 = time.perf_counter()
    pass_stats = tuple(
        optimize_kernel(k) if optimize_ir else PassStats() for k in kernels
    )
    t_pass = time.perf_counter()
    ir_texts = tuple(print_ir(k) if capture_ir_text else "" for k in kernels)
    t1 = time.perf_counter()
    llvm_texts = tuple(
        _lower_llvm_via_backend(k, arch=lower_arch, backend=backend, spec=None)
        for k in kernels
    )
    t2 = time.perf_counter()
    hsaco, comgr_t = build_hsaco_from_llvm_modules(
        llvm_texts, isa=isa, options=options[0]
    )
    t3 = time.perf_counter()

    timings["ir_opt"] = (t_pass - t0) * 1000.0
    timings["ir_build"] = (t1 - t_pass) * 1000.0
    timings["ir_lower_llvm"] = (t2 - t1) * 1000.0
    timings["comgr_bc"] = comgr_t.bc * 1000.0
    timings["comgr_link"] = comgr_t.link * 1000.0
    timings["comgr_relocatable"] = comgr_t.relocatable * 1000.0
    timings["comgr_executable"] = comgr_t.executable * 1000.0
    timings["total"] = (t3 - t0) * 1000.0

    return ModuleArtifact(
        module=module,
        ir_texts=ir_texts,
        llvm_texts=llvm_texts,
        hsaco=hsaco,
        timings=timings,
        pass_stats=pass_stats,
        isa=isa,
    )


def _resolve_compile_target(arch: Optional[str], isa: str):
    """``(lowering arch, COMGR ISA name)`` for :func:`compile_kernel` inputs."""
    if arch is not None:
        lower_arch = base_arch_from_target_id(arch)
        compiler_target = compiler_target_from_target_id(arch)
        base_isa = ArchTarget.from_gfx(lower_arch).isa_triple
        return lower_arch, f"{base_isa[: -len(lower_arch)]}{compiler_target}"
    # Derive both names from the caller's ISA name. Use the base architecture
    # for lowering and preserve compiler features in the COMGR ISA name.
    target_id = target_id_from_isa(isa)
    compiler_target = compiler_target_from_target_id(target_id)
    isa = f"{isa[: -len(target_id)]}{compiler_target}"
    gfx = arch_from_isa(isa)
    return (gfx if gfx in known_arches() else None), isa


@dataclass(frozen=True)
class ComgrInput:
    """Everything :func:`build_hsaco_from_llvm_ir` needs to compile a kernel.

    The HSACO is a pure function of these three fields and the COMGR library
    that compiles them, which is what lets a caller key a binary cache on them
    and skip the (expensive) COMGR step when they have not changed.
    ``fresh_process`` (the kernel's ``COMGR_FRESH_PROCESS_ATTR``) is how to
    compile it, so that this holds for a kernel whose code generation would
    otherwise depend on the process's earlier compiles.
    """

    kernel_name: str
    llvm_text: str
    isa: str
    options: tuple
    fresh_process: bool = False


def lower_kernel_for_comgr(
    kernel: KernelDef,
    *,
    arch: Optional[str] = None,
    isa: str = "amdgcn-amd-amdhsa--gfx950",
    backend: Optional[str] = None,
) -> ComgrInput:
    """The lowering half of :func:`compile_kernel`, without running COMGR.

    Resolves the target the same way and lowers through the same backend, so
    ``build_hsaco_from_llvm_ir(c.llvm_text, isa=c.isa, options=list(c.options),
    fresh_process=c.fresh_process)`` produces the binary
    ``compile_kernel(kernel, arch=arch)`` would.
    """
    lower_arch, isa = _resolve_compile_target(arch, isa)
    llvm_text = _lower_llvm_via_backend(
        kernel, arch=lower_arch, backend=backend, spec=None
    )
    return ComgrInput(
        kernel_name=kernel.name,
        llvm_text=llvm_text,
        isa=isa,
        options=tuple(_comgr_options_for_kernel(kernel)),
        fresh_process=bool(kernel.attrs.get(COMGR_FRESH_PROCESS_ATTR)),
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
    if kernel.attrs.get("machine_licm") == "off":
        # Qwen3.8 true-mono platform rule R4 (P6): persistent round modules
        # compile without MachineLICM (free on hardware; it hoists id math out
        # of the layer loop and holds it live across every layer)
        options.extend(["-mllvm", "-disable-machine-licm"])
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
