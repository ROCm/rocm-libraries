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
converts -- but still gfx950 ISA since the DSL is gfx950-only). The
resulting kernel signature is ``(A: ptr, C: ptr, N: i32)`` so launching
over a one-element buffer copies a single FP16 value.

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

from typing import Any, Dict, List


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
_F32_SIZE = 4
_F32_ALIGN = 4


def _smoke_arg_schema() -> List[Dict[str, Any]]:
    """Schema for the elementwise-copy smoke kernel signature.

    Kernel ABI is ``(A: ptr<f16>, C: ptr<f16>, N: i32)`` as built by
    ``ck_dsl.instances.elementwise.build_elementwise`` for a unary op
    (see ``ElementwiseSpec`` docstring + ``build_elementwise`` body in
    ``ck_dsl/instances/elementwise.py``). The C++ ``LaunchAbi`` packs
    args back-to-back honouring each slot's ``align`` -- exactly the
    layout the AMDGPU calling convention expects when args are handed
    to ``hipModuleLaunchKernel`` via ``HIP_LAUNCH_PARAM_BUFFER_*``.
    """
    return [
        {"name": "A", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "C", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "N", "kind": "I32", "size": _I32_SIZE, "align": _I32_ALIGN},
    ]


def compile_smoke() -> dict:
    """Compile a trivial gfx950 HSACO and return the launch metadata.

    Used by I-4 to prove the C++ ``KernelArtifact`` / ``HipModule`` /
    ``LaunchAbi`` round-trip. The returned dict is the on-wire shape the
    C++ ``CompileServiceBridge::compileSmoke`` translates into a
    ``KernelArtifact``; keep the field names stable -- the C++ side
    looks them up by string.

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
        (the launcher.cpp gap noted in PREP_FINDINGS P-1).

    ``arg_schema`` (list[dict])
        Per-arg metadata used by the C++ ``LaunchAbi::pack`` to lay out
        the argument buffer. Each entry is
        ``{"name": str, "kind": str, "size": int, "align": int}``.

    ``isa`` (str)
        The comgr ISA triple the artifact was built for. Returned for
        logging only; gfx950 is the only target the DSL supports today.
    """
    from ck_dsl.helpers.compile import compile_kernel
    from ck_dsl.instances.elementwise import ElementwiseSpec, build_elementwise

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
    artifact = compile_kernel(kernel_def)

    return {
        "hsaco": artifact.hsaco,
        "kernel_name": artifact.kernel_name,
        "kind": "elementwise_copy_smoke",
        # One-element launch: one block, one wave. The kernel's scalar
        # fallback path handles the N < block*vec tail; for N=1 all
        # threads except the lane-0 path are masked out by its in-bounds
        # check (see emit_scalar_path in instances/elementwise.py).
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


def _sdpa_fwd_arg_schema() -> List[Dict[str, Any]]:
    """Schema for the tiled FMHA forward kernel signature.

    From ``ck_dsl.instances.common.fmha_mfma._declare_params`` the kernel
    takes fifteen positional parameters in this order:

      Q: ptr<f16> global            (query  -- one BLOCK_M Q tile / CTA)
      K: ptr<f16> global            (key)
      V: ptr<f16> global            (value)
      O: ptr<f16> global            (output)
      scale_log2: f32               (softmax log2 scale; launch-time value)
      seqlen_q: i32
      seqlen_k: i32
      stride_q_token: i32           (Q / K / V / O strides, token then head
      stride_q_head:  i32            for each tensor, per ``add_strides``)
      stride_k_token: i32
      stride_k_head:  i32
      stride_v_token: i32
      stride_v_head:  i32
      stride_o_token: i32
      stride_o_head:  i32

    The host packs these back-to-back at natural alignment, matching the
    ``<QQQQfiiiiiiiiii`` struct format the C++ launch path packs: four
    8-byte pointers, one 4-byte f32, then ten 4-byte i32 scalars. The
    pointer / scale / stride values are all supplied at launch time -- the
    schema describes the slot layout, not baked-in data.
    """
    return [
        {"name": "Q", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "K", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "V", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "O", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "scale_log2", "kind": "F32", "size": _F32_SIZE, "align": _F32_ALIGN},
        {"name": "seqlen_q", "kind": "I32", "size": _I32_SIZE, "align": _I32_ALIGN},
        {"name": "seqlen_k", "kind": "I32", "size": _I32_SIZE, "align": _I32_ALIGN},
        {
            "name": "stride_q_token",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
        {
            "name": "stride_q_head",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
        {
            "name": "stride_k_token",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
        {
            "name": "stride_k_head",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
        {
            "name": "stride_v_token",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
        {
            "name": "stride_v_head",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
        {
            "name": "stride_o_token",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
        {
            "name": "stride_o_head",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
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

# Whitelisted keys that the C++ ``sdpaSpecToPayload`` may set. Mirrors the
# conv whitelist rationale: an unexpected field (a leaked debugging tag, an
# unsanitised FlatBuffer attribute) fails closed before any value reaches
# the dataclass construction rather than silently configuring the compile.
# ``batch`` is grid-only (not a ``FmhaMfmaSpec`` field); strides and scale
# are launch-time args, so they are absent here by design.
_SDPA_FWD_SHAPE_KEYS = frozenset({"head_size", "num_query_heads", "num_kv_heads"})
_SDPA_FWD_TOP_KEYS = frozenset(
    {"batch", "shape", "dtype", "mask_mode", "seqlen_q", "seqlen_k"}
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


def _conv_spec_from_payload(payload: dict):
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


def _sdpa_fwd_spec_from_payload(payload: dict):
    """Deserialize the wire payload into a ``(FmhaMfmaSpec, batch)`` pair.

    Shared by the compile and applicability paths so both honour one
    deserialization contract -- the whitelist and the dataclass
    construction can't drift between them.

    Caller contract: ``payload`` is the dict ``sdpaSpecToPayload`` emits --
    a nested ``"shape"`` dict (head_size / num_query_heads / num_kv_heads)
    plus the top-level ``dtype`` / ``mask_mode`` / ``seqlen_q`` /
    ``seqlen_k`` codegen inputs and the ``batch`` grid extent. Keys are
    whitelisted against ``_SDPA_FWD_TOP_KEYS`` / ``_SDPA_FWD_SHAPE_KEYS`` so
    an unexpected field fails closed rather than being silently forwarded
    into a dataclass via ``**kwargs``.

    ``batch`` is returned alongside the spec because it sizes the launch
    grid (the z axis) but is not a :class:`FmhaMfmaSpec` field -- the spec
    carries only the per-CTA codegen shape. ``scale_log2`` and the Q/K/V/O
    strides are likewise absent from the payload: they are launch-time args
    the host supplies via the arg buffer, not values baked into the kernel,
    so the common spec keeps its default ``scale_log2`` and the strides
    never enter codegen at all.

    The target arch is NOT part of the payload: it is an orthogonal compile
    target threaded as a separate argument (mirroring the DSL, whose
    ``FmhaMfmaSpec`` likewise has no arch field).
    """
    from ck_dsl.instances import FmhaCommonSpec, FmhaShape
    from ck_dsl.instances.common.fmha_mfma import FmhaMfmaSpec

    _reject_unexpected(payload.keys(), _SDPA_FWD_TOP_KEYS, "sdpa_fmha_fwd")

    shape_payload = dict(payload["shape"])
    _reject_unexpected(
        shape_payload.keys(), _SDPA_FWD_SHAPE_KEYS, "sdpa_fmha_fwd.shape"
    )
    shape = FmhaShape(**shape_payload)

    common = FmhaCommonSpec(
        shape=shape,
        dtype=payload["dtype"],
        mask_mode=payload["mask_mode"],
    )
    spec = FmhaMfmaSpec(
        common=common,
        seqlen_q=payload["seqlen_q"],
        seqlen_k=payload["seqlen_k"],
    )
    return spec, int(payload["batch"])


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


def _compile_sdpa_fwd(payload: dict, arch: str) -> dict:
    """Build + compile a tiled FMHA forward kernel from the payload.

    ``arch`` is threaded to BOTH ``build_fmha_fwd_mfma`` (which resolves the
    per-arch f16 ``16x16x16`` MMA/WMMA atom, sizes the block to the target
    wave, and validates the spec) and ``compile_kernel`` (which selects the
    ISA triple + lowering backend). Passing it to only one would mis-build
    the kernel on any non-default arch.

    ``scale_log2`` and the Q/K/V/O strides are launch-time arguments the
    host packs into the arg buffer per :func:`_sdpa_fwd_arg_schema`; they
    are not codegen inputs, so they are absent from the payload and never
    baked into the kernel. The block dim is one wave (``wave_size`` from the
    target -- 64 on CDNA, 32 on RDNA), matching the one-wave-per-CTA grid the
    DSL builder emits.
    """
    from ck_dsl.core.arch import ArchTarget
    from ck_dsl.helpers.compile import compile_kernel
    from ck_dsl.instances.common.fmha_mfma import (
        build_fmha_fwd_mfma,
        fmha_fwd_mfma_grid,
    )

    spec, batch = _sdpa_fwd_spec_from_payload(payload)

    kernel_def = build_fmha_fwd_mfma(spec, arch=arch)
    artifact = compile_kernel(kernel_def, arch=arch)

    # Grid: (seqlen_q // BLOCK_M, num_query_heads, batch) -- one wave64/32
    # CTA per (q_tile, head, batch) triple. Block is one wave; the inner
    # body assumes a single wave per CTA.
    grid = fmha_fwd_mfma_grid(spec, batch=batch)
    wave_size = ArchTarget.from_gfx(arch).wave_size
    block = (wave_size, 1, 1)

    return {
        "hsaco": artifact.hsaco,
        "kernel_name": artifact.kernel_name,
        "kind": "sdpa_fmha_fwd",
        "grid": tuple(grid),
        "block": block,
        # The FMHA body stages a single BLOCK_M x BLOCK_K f16 P buffer via
        # smem_alloc with statically-known shapes, so the dynamic-LDS arg to
        # hipModuleLaunchKernel is zero -- static LDS lives in the HSACO's
        # kernarg descriptor.
        "lds_bytes": 0,
        "arg_schema": _sdpa_fwd_arg_schema(),
        "isa": artifact.isa,
    }


def compile(op_kind: str, payload: dict, arch: str) -> dict:
    """Compile one DSL kernel from a typed payload dict for ``arch``.

    Called by the C++ ``CompileServiceBridge::compile`` on a
    ``JitCache`` miss. Dispatches on ``op_kind`` (``"conv_implicit_gemm"``
    or ``"sdpa_fmha_fwd"``). Unknown kinds raise ``ValueError``,
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
    elif op_kind == "sdpa_fmha_fwd":
        return _compile_sdpa_fwd(payload, arch)
    raise ValueError(
        f"ck_dsl_provider.compile_service: unsupported op_kind {op_kind!r}"
    )


def _is_applicable_conv_implicit_gemm(payload: dict, arch: str):
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


def _is_applicable_sdpa_fwd(payload: dict, arch: str):
    """Arch-aware applicability for a tiled FMHA forward spec.

    Consults the DSL's ``is_valid_spec`` for ``arch`` -- the exact predicate
    ``build_fmha_fwd_mfma`` enforces internally -- so the C++ ``isApplicable``
    gate matches the compile path. No kernel is built and comgr is never
    invoked; this is a pure data-driven check against the target's
    :class:`ck_dsl.core.arch.ArchTarget` (f16 16x16x16 atom presence, wave
    size, LDS capacity). ``is_valid_spec`` also returns ``(False, ...)`` for
    an unknown arch, so this single call covers both "arch unsupported" and
    "shape / mask invalid for this arch". ``batch`` is grid-only and does not
    affect applicability, so the unpacked value is discarded.
    """
    from ck_dsl.instances.common.fmha_mfma import is_valid_spec

    spec, _batch = _sdpa_fwd_spec_from_payload(payload)
    ok, reason = is_valid_spec(spec, arch)
    return bool(ok), str(reason)


def is_applicable(op_kind: str, payload: dict, arch: str):
    """Return ``(ok, reason)`` for running ``op_kind`` on ``arch``.

    Called by the C++ ``CompileServiceBridge::isApplicable`` from the
    plan builder's ``isApplicable``. Dispatches on ``op_kind`` (mirroring
    :func:`compile`); ``arch`` is the target gfx token, passed separately
    from ``payload`` exactly as for :func:`compile`. Each op exposes its
    own DSL validator. Unknown kinds raise ``ValueError``, which the
    bridge surfaces as a ``HipdnnPluginException``.

    Unlike :func:`compile` this never compiles -- it is a fast predicate
    safe to call on the plan-finding hot path.
    """
    if op_kind == "conv_implicit_gemm":
        return _is_applicable_conv_implicit_gemm(payload, arch)
    elif op_kind == "sdpa_fmha_fwd":
        return _is_applicable_sdpa_fwd(payload, arch)
    raise ValueError(
        f"ck_dsl_provider.compile_service: unsupported op_kind {op_kind!r}"
    )
