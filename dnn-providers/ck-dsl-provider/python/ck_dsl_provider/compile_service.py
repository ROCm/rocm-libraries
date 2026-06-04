# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Thin Python compile-service entry point for the CK DSL provider.

M1 step I-3 ships :func:`noop_smoke`, which proves the C++ -> Python ->
``ck_dsl`` import path works inside the embedded interpreter.

M1 step I-4 adds :func:`compile_smoke`, which exercises the full DSL
compile pipeline (``build_elementwise`` -> ``compile_kernel`` -> HSACO
bytes) and returns the kernel artifact plus the minimum launch metadata
the C++ side needs to load and launch it via ``hipModuleLoadData`` +
``hipModuleLaunchKernel``. The smoke kernel is the simplest cleanly-
compiling instance in ``ck_dsl.instances``: an FP16 copy elementwise
with ``block_size=64`` and ``vec=2`` (no MFMA, no LDS, no scaled
converts). Because the smoke kernel uses no arch-specific MFMA atom
the DSL compiles it for any supported arch (gfx942/gfx950/gfx1151);
the caller supplies the target gfx token via the ``arch`` parameter.
The resulting kernel signature is ``(A: ptr, C: ptr, N: i32)`` so
launching over a one-element buffer copies a single FP16 value.

M1 step I-7 adds :func:`compile`, the production entry the C++
``CompileServiceBridge::compile`` calls on a JitCache miss. It
dispatches on ``op_kind`` and, for ``"conv_implicit_gemm"``,
instantiates the matching ``ck_dsl`` dataclasses from the payload dict
that ``ConvImplicitGemmPayload::convImplicitGemmSpecToPayload``
emitted, runs ``build_implicit_gemm_conv`` + ``compile_kernel``, and
returns the artifact plus the launch metadata derived from the spec
(grid from M / K + tile sizes, block from spec.block_size).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    from ck_dsl.instances.common.conv_implicit_gemm import ImplicitGemmConvSpec


def noop_smoke() -> dict:
    """Return a constant. Used by I-3 to prove import path + GIL handling."""
    import ck_dsl  # forces the cross-package import to succeed

    return {
        "service": "ck_dsl_provider.compile_service",
        "ck_dsl_module_path": ck_dsl.__file__,
        "smoke": "ok",
    }


# Pointer ABI on AMDGPU host-side: 8 bytes, 8-byte aligned. Matches the
# natural-alignment scheme used by `launcher.cpp` for the DSL kernels.
_PTR_SIZE = 8
_PTR_ALIGN = 8
_I32_SIZE = 4
_I32_ALIGN = 4


def _smoke_arg_schema() -> List[Dict[str, Any]]:
    """Schema for the elementwise-copy smoke kernel signature.

    Kernel ABI is ``(A: ptr<f16>, C: ptr<f16>, N: i32)`` as built by
    ``ck_dsl.instances.common.elementwise.build_elementwise`` for a unary
    op (see ``ElementwiseSpec`` docstring + ``build_elementwise`` body in
    ``ck_dsl/instances/common/elementwise.py``). The C++ ``LaunchAbi`` packs
    args back-to-back honouring each slot's ``align`` -- exactly the
    layout the AMDGPU calling convention expects when args are handed
    to ``hipModuleLaunchKernel`` via ``HIP_LAUNCH_PARAM_BUFFER_*``.
    """
    return [
        {"name": "A", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "C", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "N", "kind": "I32", "size": _I32_SIZE, "align": _I32_ALIGN},
    ]


def compile_smoke(arch: str) -> dict:
    """Compile the trivial elementwise-copy kernel for ``arch``.

    Used by I-4 to prove the C++ ``KernelArtifact`` / ``HipModule`` /
    ``LaunchAbi`` round-trip. The returned dict is the on-wire shape the
    C++ ``CompileServiceBridge::compileSmoke`` translates into a
    ``KernelArtifact``; keep the field names stable -- the C++ side
    looks them up by string.

    Args:
        arch: Target gfx token (e.g. ``"gfx950"``). Threaded to
            ``compile_kernel`` exactly as ``_compile_conv_implicit_gemm``
            does, so the resulting HSACO targets the requested arch.

    Returned fields:

    ``hsaco`` (bytes)
        The HSA code object produced by ``compile_kernel``. Loadable
        via ``hipModuleLoadData``.

    ``kernel_name`` (str)
        The mangled kernel symbol to pass to ``hipModuleGetFunction``.

    ``kind`` (str)
        Free-form tag for logs / debugging. Not consumed by the C++
        launch path.

    ``grid`` (tuple[int, int, int])
        ``(gx, gy, gz)`` for ``hipModuleLaunchKernel``. Sized for a
        one-element launch -- one block of ``block_size`` threads.

    ``block`` (tuple[int, int, int])
        ``(bx, by, bz)``. Always ``(block_size, 1, 1)`` for this kernel.

    ``lds_bytes`` (int)
        Dynamic shared-memory bytes (``sharedMemBytes`` arg). Zero for
        this kernel; the field exists so future kernels that need
        dynamic LDS can populate it without changing the wire shape
        (the launcher.cpp per-kind packing gap).

    ``arg_schema`` (list[dict])
        Per-arg metadata used by the C++ ``LaunchAbi::pack`` to lay out
        the argument buffer. Each entry is
        ``{"name": str, "kind": str, "size": int, "align": int}``.

    ``isa`` (str)
        The comgr ISA triple the artifact was built for, recording which
        arch this artifact targets (e.g. ``"amdgcn-amd-amdhsa--gfx950"``
        when ``arch="gfx950"``).
    """
    from ck_dsl.helpers.compile import compile_kernel
    from ck_dsl.instances.common.elementwise import ElementwiseSpec, build_elementwise

    # The simplest cleanly-compiling instance: FP16 copy, single warp.
    # Keeping block_size small (one wave) and vec small (one 32-bit
    # load) avoids stressing anything we don't need to exercise for a
    # smoke-level "did the launch succeed" check.
    spec = ElementwiseSpec(
        op="copy",
        dtype="f16",
        block_size=64,
        vec=2,
        name="ck_dsl_provider_smoke_copy",
    )
    kernel_def = build_elementwise(spec)
    artifact = compile_kernel(kernel_def, arch=arch)

    return {
        "hsaco": artifact.hsaco,
        "kernel_name": artifact.kernel_name,
        "kind": "elementwise_copy_smoke",
        # One-element launch: one block, one wave. The kernel's scalar
        # fallback path handles the N < block*vec tail; for N=1 all
        # threads except the lane-0 path are masked out by its in-bounds
        # check (see emit_scalar_path in instances/common/elementwise.py).
        "grid": (1, 1, 1),
        "block": (spec.block_size, 1, 1),
        "lds_bytes": 0,
        "arg_schema": _smoke_arg_schema(),
        "isa": artifact.isa,
    }


def _conv_implicit_gemm_arg_schema() -> List[Dict[str, Any]]:
    """Schema for the implicit-GEMM conv kernel signature.

    From ``ck_dsl.instances.conv_implicit_gemm.build_implicit_gemm_conv``
    the kernel takes six positional parameters:

      A: ptr<f16> global    (input  NHWC)
      B: ptr<f16> global    (weight KRSC)
      D: ptr<f16> global    (output NHWK)
      A_bytes: i32          (buffer-rsrc bounds for OOB clamping)
      B_bytes: i32
      D_bytes: i32

    Natural alignment is sufficient; the AMDGPU host-side calling
    convention packs these into 36 bytes (3 * 8 ptrs + 3 * 4 i32).
    """
    return [
        {"name": "A", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "B", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "D", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "A_bytes", "kind": "I32", "size": _I32_SIZE, "align": _I32_ALIGN},
        {"name": "B_bytes", "kind": "I32", "size": _I32_SIZE, "align": _I32_ALIGN},
        {"name": "D_bytes", "kind": "I32", "size": _I32_SIZE, "align": _I32_ALIGN},
    ]


# Whitelisted keys that the C++ ConvImplicitGemmPayload may set. Any
# other key surfaced through the wire dict is rejected before we let
# `**kwargs` reach the dataclass: a future C++ regression that leaks
# an unintended field (a debugging tag, an unsanitised FlatBuffer
# attribute, etc.) fails closed rather than silently configuring the
# JIT compile.
_CONV_PROBLEM_KEYS = frozenset(
    {"N", "Hi", "Wi", "C", "K", "R", "S", "sH", "sW", "pH", "pW", "dH", "dW"}
)
_CONV_IMPLICIT_GEMM_SPEC_TOP_KEYS = frozenset(
    {
        "problem",
        "name",
        "tile_m",
        "tile_n",
        "tile_k",
        "warp_m",
        "warp_n",
        "warp_tile_m",
        "warp_tile_n",
        "warp_tile_k",
        "wave_size",
        "pipeline",
        "epilogue",
        "async_dma",
        "unroll_k",
        "lds_k_pad",
        "chiplet_swizzle",
        "chiplet_wgm",
        "chiplet_num_xcds",
        "chiplet_chunk_size",
        "waves_per_eu",
    }
)


def _reject_unexpected(payload_keys, expected, context: str) -> None:
    extras = set(payload_keys) - expected
    if extras:
        raise ValueError(
            f"ck_dsl_provider.compile_service: {context} payload has unexpected "
            f"keys {sorted(extras)!r}; allowed keys are {sorted(expected)!r}. "
            "If a new field is needed, add it to the C++ payload contract and "
            "to the whitelist together."
        )


def _conv_spec_from_payload(payload: dict) -> "ImplicitGemmConvSpec":
    """Deserialize the wire payload into an ``ImplicitGemmConvSpec``.

    Shared by the compile and applicability paths so both honour one
    deserialization contract -- the whitelist and the dataclass
    construction can't drift between them.

    Caller contract: ``payload`` is the dict
    ``ConvImplicitGemmPayload::convImplicitGemmSpecToPayload`` emits --
    a nested ``"problem"`` dict plus the top-level ``ImplicitGemmConvSpec``
    kwargs, field-for-field with the dataclass. The payload keys are
    whitelisted against ``_CONV_IMPLICIT_GEMM_SPEC_TOP_KEYS`` /
    ``_CONV_PROBLEM_KEYS`` so an unexpected field fails closed rather than
    being silently forwarded into the dataclass via ``**kwargs``.

    The target arch is NOT part of the payload: it is an orthogonal
    compile target threaded as a separate argument (mirroring the DSL,
    whose ``ImplicitGemmConvSpec`` likewise has no arch field).
    """
    from ck_dsl.instances.common.conv_implicit_gemm import (
        ConvProblem,
        ImplicitGemmConvSpec,
    )

    _reject_unexpected(
        payload.keys(), _CONV_IMPLICIT_GEMM_SPEC_TOP_KEYS, "conv_implicit_gemm spec"
    )

    problem_payload = dict(payload["problem"])
    _reject_unexpected(
        problem_payload.keys(), _CONV_PROBLEM_KEYS, "conv_implicit_gemm problem"
    )
    problem = ConvProblem(**problem_payload)

    spec_kwargs = {k: v for k, v in payload.items() if k != "problem"}
    return ImplicitGemmConvSpec(problem=problem, **spec_kwargs)


def _compile_conv_implicit_gemm(payload: dict, arch: str) -> dict:
    """Build + compile an implicit-GEMM conv kernel from the payload.

    ``arch`` is threaded to BOTH ``build_implicit_gemm_conv`` (which
    resolves per-arch MMA/WMMA atoms and validates the spec) and
    ``compile_kernel`` (which selects the ISA triple + lowering
    backend). Passing it to only one would mis-build the kernel on any
    non-default arch.
    """
    from ck_dsl.helpers.compile import compile_kernel
    from ck_dsl.instances.common.conv_implicit_gemm import build_implicit_gemm_conv

    spec = _conv_spec_from_payload(payload)

    kernel_def = build_implicit_gemm_conv(spec, arch=arch)
    artifact = compile_kernel(kernel_def, arch=arch)

    # Grid derivation:
    #   M = N*Ho*Wo, num_pid_m = ceil(M / tile_m)
    #   num_pid_n = ceil(K / tile_n)
    #   grid = (num_pid_n, num_pid_m, 1)  -- matches the kernel's
    #     ``grid_order="NM"`` convention where block.x indexes N tiles
    #     and block.y indexes M tiles (set in
    #     ``build_implicit_gemm_conv``).
    #   block = (block_size, 1, 1)
    problem = spec.problem
    M = problem.M
    num_pid_m = (M + spec.tile_m - 1) // spec.tile_m
    num_pid_n = (problem.N_gemm + spec.tile_n - 1) // spec.tile_n

    return {
        "hsaco": artifact.hsaco,
        "kernel_name": artifact.kernel_name,
        "kind": "conv_implicit_gemm",
        "grid": (num_pid_n, num_pid_m, 1),
        "block": (spec.block_size, 1, 1),
        # The implicit-GEMM kernel allocates its LDS via smem_alloc with
        # statically-known shapes (A_smem / B_smem / optional C_smem for
        # cshuffle), so the dynamic-LDS arg to hipModuleLaunchKernel is
        # zero. Static LDS lives inside the HSACO's kernarg descriptor.
        "lds_bytes": 0,
        "arg_schema": _conv_implicit_gemm_arg_schema(),
        "isa": artifact.isa,
    }


def compile(op_kind: str, payload: dict, arch: str) -> dict:
    """Compile one DSL kernel from a typed payload dict for ``arch``.

    Called by the C++ ``CompileServiceBridge::compile`` on a
    ``JitCache`` miss. Dispatches on ``op_kind``; the only kind in M1
    is ``"conv_implicit_gemm"``. Unknown kinds raise ``ValueError``,
    which the bridge surfaces as a ``HipdnnPluginException``.

    ``arch`` is the target gfx token, passed separately from ``payload``
    (an orthogonal compile target, not a spec field -- mirroring the DSL
    entry points).

    The returned dict shape matches :func:`compile_smoke` so the C++
    side can use the same translation path for both smoke and
    production compiles.
    """
    if op_kind == "conv_implicit_gemm":
        return _compile_conv_implicit_gemm(payload, arch)
    raise ValueError(
        f"ck_dsl_provider.compile_service: unsupported op_kind {op_kind!r}"
    )


def _is_applicable_conv_implicit_gemm(payload: dict, arch: str) -> Tuple[bool, str]:
    """Arch-aware applicability for an implicit-GEMM conv spec.

    Consults the DSL's ``is_valid_spec`` for ``arch`` -- the exact
    predicate ``build_implicit_gemm_conv`` enforces internally -- so the
    C++ ``isApplicable`` gate matches the compile path. No kernel is
    built and comgr is never invoked; this is a pure data-driven check
    against the target's :class:`ck_dsl.core.arch.ArchTarget` (atom
    catalog, wave size, LDS caps). ``is_valid_spec`` also returns
    ``(False, ...)`` for an unknown arch, so this single call covers both
    "arch unsupported" and "knobs invalid for this arch".
    """
    from ck_dsl.instances.common.conv_implicit_gemm import is_valid_spec

    spec = _conv_spec_from_payload(payload)
    ok, reason = is_valid_spec(spec, arch)
    return bool(ok), str(reason)


def is_applicable(op_kind: str, payload: dict, arch: str) -> Tuple[bool, str]:
    """Return ``(ok, reason)`` for running ``op_kind`` on ``arch``.

    Called by the C++ ``CompileServiceBridge::isApplicable`` from the
    plan builder's ``isApplicable``. Dispatches on ``op_kind`` (mirroring
    :func:`compile`); ``arch`` is the target gfx token, passed separately
    from ``payload`` exactly as for :func:`compile`. Each op exposes its
    own DSL validator. Unknown kinds raise ``ValueError``, which the
    bridge surfaces as a ``HipdnnPluginException``.

    Unlike :func:`compile` this never compiles: the check itself is a
    cheap, data-driven ``is_valid_spec`` against the cached
    :class:`ck_dsl.core.arch.ArchTarget` (the catalog is ``lru_cache``d,
    so no per-call file I/O). Note, though, that *reaching* it from C++
    still costs a GIL acquire and a pybind round-trip per call, and the
    plan builder invokes ``isApplicable`` several times per finalize --
    so it is cheap, not free.
    """
    if op_kind == "conv_implicit_gemm":
        return _is_applicable_conv_implicit_gemm(payload, arch)
    raise ValueError(
        f"ck_dsl_provider.compile_service: unsupported op_kind {op_kind!r}"
    )
