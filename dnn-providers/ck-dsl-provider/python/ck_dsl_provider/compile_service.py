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


def _sdpa_fwd_arg_schema(generate_stats: bool = False) -> List[Dict[str, Any]]:
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

    When ``generate_stats`` is set the kernel appends ONE additional f32
    pointer at ABI position 16 (the 16th slot, 0-indexed 15):

      LSE_out: ptr<f32> global      (opt-in natural-log LSE, head-major
                                     [B, Hq, Sq] contiguous)

    matching ``build_fmha_fwd_mfma``'s opt-in ``LSE_out`` parameter. The
    stats-off schema is byte-identical to the historical 15-slot ABI.

    The host packs these back-to-back at natural alignment, matching the
    ``<QQQQfiiiiiiiiii`` struct format the C++ launch path packs (plus a
    trailing 8-byte pointer when stats are on): four 8-byte pointers, one
    4-byte f32, then ten 4-byte i32 scalars, and the optional 8-byte
    ``LSE_out`` pointer. The pointer / scale / stride values are all
    supplied at launch time -- the schema describes the slot layout, not
    baked-in data.
    """
    schema: List[Dict[str, Any]] = [
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
    if generate_stats:
        # Opt-in forward stats (LSE) output: one f32 pointer at ABI
        # position 16 (slot index 15), after the 15 base args.
        schema.append(
            {
                "name": "LSE_out",
                "kind": "Pointer",
                "size": _PTR_SIZE,
                "align": _PTR_ALIGN,
            }
        )
    return schema


def _sdpa_fwd_unified_arg_schema() -> List[Dict[str, Any]]:
    """Schema for the unified paged/varlen tiled-2D attention kernel.

    From ``build_unified_attention_2d_tiled`` (the parameter declarations
    in ``instances/gfx950/attention_tiled_2d.py``) the kernel takes
    eighteen positional parameters in this fixed order:

      output_ptr:          ptr<dtype>  global  (attention output)
      query_ptr:           ptr<dtype>  global
      key_cache_ptr:       ptr<kv>     global  (paged K cache)
      value_cache_ptr:     ptr<kv>     global  (paged V cache)
      sink_ptr:            ptr<dtype>  global  (attention sinks; 0 if unused)
      block_tables_ptr:    ptr<i32>    global  ([num_seqs, blocks_per_seq])
      seq_lens_ptr:        ptr<i32>    global  (seqused_k per sequence)
      alibi_slopes_ptr:    ptr<f32>    global  (0 if unused)
      qq_bias_ptr:         ptr<f32>    global  (0 if unused)
      query_start_len_ptr: ptr<i32>    global  (cu_seqlens_q, len num_seqs+1)
      scale:               f32                 (softmax scale; launch-time)
      k_scale:             f32                 (fp8 KV dequant; 1.0 default)
      v_scale:             f32
      out_scale:           f32
      softcap:             f32                 (0.0 == no softcap)
      num_seqs:            i32
      block_table_stride:  i32                 (ceil(max_seqlen_k/block_size))
      qq_bias_stride_0:    i32

    The host packs these back-to-back at natural alignment: ten 8-byte
    pointers, five 4-byte f32, then three 4-byte i32. The pointer / scale
    / stride values are all supplied at launch time -- the schema
    describes the 18-slot layout, not baked-in data. This ABI is fixed
    regardless of the chosen perf knobs (the knobs steer codegen, not the
    parameter list).
    """
    return [
        {
            "name": "output_ptr",
            "kind": "Pointer",
            "size": _PTR_SIZE,
            "align": _PTR_ALIGN,
        },
        {
            "name": "query_ptr",
            "kind": "Pointer",
            "size": _PTR_SIZE,
            "align": _PTR_ALIGN,
        },
        {
            "name": "key_cache_ptr",
            "kind": "Pointer",
            "size": _PTR_SIZE,
            "align": _PTR_ALIGN,
        },
        {
            "name": "value_cache_ptr",
            "kind": "Pointer",
            "size": _PTR_SIZE,
            "align": _PTR_ALIGN,
        },
        {"name": "sink_ptr", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {
            "name": "block_tables_ptr",
            "kind": "Pointer",
            "size": _PTR_SIZE,
            "align": _PTR_ALIGN,
        },
        {
            "name": "seq_lens_ptr",
            "kind": "Pointer",
            "size": _PTR_SIZE,
            "align": _PTR_ALIGN,
        },
        {
            "name": "alibi_slopes_ptr",
            "kind": "Pointer",
            "size": _PTR_SIZE,
            "align": _PTR_ALIGN,
        },
        {
            "name": "qq_bias_ptr",
            "kind": "Pointer",
            "size": _PTR_SIZE,
            "align": _PTR_ALIGN,
        },
        {
            "name": "query_start_len_ptr",
            "kind": "Pointer",
            "size": _PTR_SIZE,
            "align": _PTR_ALIGN,
        },
        {"name": "scale", "kind": "F32", "size": _F32_SIZE, "align": _F32_ALIGN},
        {"name": "k_scale", "kind": "F32", "size": _F32_SIZE, "align": _F32_ALIGN},
        {"name": "v_scale", "kind": "F32", "size": _F32_SIZE, "align": _F32_ALIGN},
        {"name": "out_scale", "kind": "F32", "size": _F32_SIZE, "align": _F32_ALIGN},
        {"name": "softcap", "kind": "F32", "size": _F32_SIZE, "align": _F32_ALIGN},
        {"name": "num_seqs", "kind": "I32", "size": _I32_SIZE, "align": _I32_ALIGN},
        {
            "name": "block_table_stride",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
        {
            "name": "qq_bias_stride_0",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
    ]


def _sdpa_bwd_arg_schema() -> List[Dict[str, Any]]:
    """Schema for the FMHA backward kernel signature.

    From ``ck_dsl.instances.common.fmha_bwd._declare_params`` the kernel takes
    twenty-four positional parameters in this order:

      Q: ptr<f16> global            (query)
      K: ptr<f16> global            (key)
      V: ptr<f16> global            (value)
      dO: ptr<f16> global           (output gradient)
      M_saved: ptr<f32> global      (saved softmax max, read-only)
      L_saved: ptr<f32> global      (saved softmax denominator, read-only)
      dQ: ptr<f32> global           (query gradient -- atomic accumulator)
      dK: ptr<f32> global           (key gradient   -- atomic accumulator)
      dV: ptr<f32> global           (value gradient -- atomic accumulator)
      scale_log2: f32               (softmax log2 scale; launch-time value)
      scale_inv:  f32               (1 / softmax scale; launch-time value)
      seqlen_q: i32
      seqlen_k: i32
      stride_q_token:  i32          (Q / K / V strides, token then head for
      stride_q_head:   i32           each tensor, per ``add_strides``)
      stride_k_token:  i32
      stride_k_head:   i32
      stride_v_token:  i32
      stride_v_head:   i32
      stride_do_token: i32          (dO token + head stride, per add_strides)
      stride_do_head:  i32
      stride_dq_token: i32          (gradient tensors carry token stride only;
      stride_dk_token: i32           their head stride is implicit == the
      stride_dv_token: i32           matching Q / K / V head stride)

    The host packs these back-to-back at natural alignment, matching the
    ``<9Q2f13i`` struct format the C++ launch path packs: nine 8-byte
    pointers, two 4-byte f32, then thirteen 4-byte i32 scalars. The pointer /
    scale / stride values are all supplied at launch time -- the schema
    describes the slot layout, not baked-in data.
    """
    return [
        {"name": "Q", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "K", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "V", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "dO", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "M_saved", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "L_saved", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "dQ", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "dK", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "dV", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "scale_log2", "kind": "F32", "size": _F32_SIZE, "align": _F32_ALIGN},
        {"name": "scale_inv", "kind": "F32", "size": _F32_SIZE, "align": _F32_ALIGN},
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
            "name": "stride_do_token",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
        {
            "name": "stride_do_head",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
        {
            "name": "stride_dq_token",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
        {
            "name": "stride_dk_token",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
        {
            "name": "stride_dv_token",
            "kind": "I32",
            "size": _I32_SIZE,
            "align": _I32_ALIGN,
        },
    ]


def _sdpa_lse_prep_arg_schema() -> List[Dict[str, Any]]:
    """Schema for the FMHA-bwd stats-prep kernel signature.

    From ``ck_dsl.instances.common.sdpa_lse_prep._declare_params`` the kernel
    takes six positional parameters in this order:

      stats: ptr<f32> global        (head-major source [B, Hq, Sq], read-only)
      M_out: ptr<f32> global        (per-batch q-major dest, write)
      L_out: ptr<f32> global        (per-batch q-major dest, write)
      B:  i32                       (batch extent)
      Hq: i32                       (query-head extent)
      Sq: i32                       (query-seqlen extent)

    The host packs these back-to-back at natural alignment, matching the
    ``<3Q3i`` struct format the C++ launch path packs: three 8-byte pointers
    then three 4-byte i32 scalars. The pointer values are supplied at launch
    time -- the schema describes the slot layout, not baked-in data.
    """
    return [
        {"name": "stats", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "M_out", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "L_out", "kind": "Pointer", "size": _PTR_SIZE, "align": _PTR_ALIGN},
        {"name": "B", "kind": "I32", "size": _I32_SIZE, "align": _I32_ALIGN},
        {"name": "Hq", "kind": "I32", "size": _I32_SIZE, "align": _I32_ALIGN},
        {"name": "Sq", "kind": "I32", "size": _I32_SIZE, "align": _I32_ALIGN},
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
# ``generate_stats`` is the opt-in forward-training stats (LSE) flag: when set
# the kernel appends one f32 ``LSE_out`` pointer at ABI position 16 (after the
# 15 base args). It is a ``FmhaCommonSpec`` field (codegen-relevant), so it
# rides on the payload alongside the shape / dtype / mask codegen inputs.
_SDPA_FWD_TOP_KEYS = frozenset(
    {"batch", "shape", "dtype", "mask_mode", "seqlen_q", "seqlen_k", "generate_stats"}
)

# Whitelisted nested ``knobs`` keys the C++ ``sdpaSpecToPayload`` may set for
# the unified paged/varlen tiled-2D path. These map field-for-field onto the
# ``SdpaPerfKnobs`` POD the scorer-driven selection produced and, downstream,
# onto the ``UnifiedAttention2DTiledSpec`` keyword arguments. Same fail-closed
# rationale as every other whitelist: an unexpected knob name fails before any
# value reaches the dataclass via ``**kwargs``.
_SDPA_FWD_UNIFIED_KNOB_KEYS = frozenset(
    {
        "num_warps",
        "block_m_per_warp",
        "tile_size",
        "waves_per_eu",
        "use_mfma_32x32",
        "use_transposed_qk_32x32",
        "use_register_pv",
        "use_early_v_schedule",
        "use_fast_paged_kv_desc",
    }
)
# Whitelisted top-level keys the C++ ``sdpaSpecToPayload`` may set for the
# unified path. Adds the four paged/varlen problem lanes (is_paged,
# block_size, is_varlen, sliding_window, use_sinks) plus the nested ``knobs``
# dict on top of the shape / dtype / mask / seqlen codegen inputs. ``batch``
# is grid-only (sizes ``num_seqs`` for the dense-degenerate problem). The
# strides and scale are launch-time args, absent by design. ``generate_stats``
# is NOT part of the unified payload: the unified kernel has no opt-in LSE
# slot in its 18-arg ABI (forward-training stats are a dense-path concern).
_SDPA_FWD_UNIFIED_TOP_KEYS = frozenset(
    {
        "batch",
        "shape",
        "dtype",
        "mask_mode",
        "seqlen_q",
        "seqlen_k",
        "is_paged",
        "block_size",
        "is_varlen",
        "sliding_window",
        "use_sinks",
        "knobs",
    }
)

# Whitelisted top-level keys the C++ ``sdpaBwdSpecToPayload`` may set. Shares
# the ``_SDPA_FWD_SHAPE_KEYS`` nested-shape whitelist with the forward path
# (identical ``FmhaShape`` field set). ``batch`` is grid-only and not a
# ``FmhaBwdSpec`` field; the scales (``scale_log2`` / ``scale_inv``) and the
# Q/K/V/dO/dQ/dK/dV strides are launch-time arg-buffer slots, so they are
# absent here by design -- same fail-closed rationale as the conv / fwd
# whitelists.
_SDPA_BWD_TOP_KEYS = frozenset(
    {"batch", "shape", "dtype", "mask_mode", "seqlen_q", "seqlen_k"}
)

# Whitelisted keys the C++ stats-prep payload may set. The stats-prep kernel
# takes a flat ``(batch, num_query_heads, seqlen_q)`` shape -- no nested shape
# dict -- so the whitelist is a single flat frozenset.
_SDPA_LSE_PREP_KEYS = frozenset({"batch", "num_query_heads", "seqlen_q"})


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
    """Deserialize the wire payload into a ``(FmhaMfmaSpec, batch, generate_stats)`` tuple.

    Shared by the compile and applicability paths so both honour one
    deserialization contract -- the whitelist and the dataclass
    construction can't drift between them.

    Caller contract: ``payload`` is the dict ``sdpaSpecToPayload`` emits --
    a nested ``"shape"`` dict (head_size / num_query_heads / num_kv_heads)
    plus the top-level ``dtype`` / ``mask_mode`` / ``seqlen_q`` /
    ``seqlen_k`` codegen inputs, the ``batch`` grid extent, and the
    ``generate_stats`` opt-in stats flag. Keys are whitelisted against
    ``_SDPA_FWD_TOP_KEYS`` / ``_SDPA_FWD_SHAPE_KEYS`` so an unexpected field
    fails closed rather than being silently forwarded into a dataclass via
    ``**kwargs``.

    ``batch`` is returned alongside the spec because it sizes the launch
    grid (the z axis) but is not a :class:`FmhaMfmaSpec` field -- the spec
    carries only the per-CTA codegen shape. ``generate_stats`` is returned
    too because it sizes the arg schema (the 16th ``LSE_out`` slot is
    appended only when set); it IS a :class:`FmhaCommonSpec` field, so it
    also rides on the spec to drive the kernel's opt-in LSE store.
    ``scale_log2`` and the Q/K/V/O strides are likewise absent from the
    payload: they are launch-time args the host supplies via the arg
    buffer, not values baked into the kernel, so the common spec keeps its
    default ``scale_log2`` and the strides never enter codegen at all.

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

    generate_stats = bool(payload.get("generate_stats", False))
    common = FmhaCommonSpec(
        shape=shape,
        dtype=payload["dtype"],
        mask_mode=payload["mask_mode"],
        generate_stats=generate_stats,
    )
    spec = FmhaMfmaSpec(
        common=common,
        seqlen_q=payload["seqlen_q"],
        seqlen_k=payload["seqlen_k"],
    )
    return spec, int(payload["batch"]), generate_stats


def _sdpa_bwd_spec_from_payload(payload: dict):
    """Deserialize the wire payload into an ``FmhaBwdSpec``.

    Shared by the compile and applicability paths so both honour one
    deserialization contract -- the whitelist and the dataclass
    construction can't drift between them.

    Caller contract: ``payload`` is the dict the C++ ``sdpaBwdSpecToPayload``
    emits -- a nested ``"shape"`` dict (head_size / num_query_heads /
    num_kv_heads) plus the top-level ``dtype`` / ``mask_mode`` / ``seqlen_q`` /
    ``seqlen_k`` codegen inputs and the ``batch`` extent. Keys are whitelisted
    against ``_SDPA_BWD_TOP_KEYS`` / ``_SDPA_FWD_SHAPE_KEYS`` (the shape field
    set is identical to the forward path) so an unexpected field fails closed
    rather than being silently forwarded into a dataclass via ``**kwargs``.

    Unlike the forward path, ``batch`` does not size the backward grid
    (:func:`fmha_bwd_grid` is ``(seqlen_q, num_query_heads, 1)`` -- batch is
    folded into the launch via offsetting, not a grid axis), so it is consumed
    by the whitelist but not returned. The scales (``scale_log2`` /
    ``scale_inv``) and the Q/K/V/dO/dQ/dK/dV strides are launch-time args the
    host supplies via the arg buffer, not values baked into the kernel, so they
    are absent from the payload and never enter codegen.

    The target arch is NOT part of the payload: it is an orthogonal compile
    target threaded as a separate argument (mirroring the DSL, whose
    ``FmhaBwdSpec`` likewise has no arch field).
    """
    from ck_dsl.instances import FmhaCommonSpec, FmhaShape
    from ck_dsl.instances.common.fmha_bwd import FmhaBwdSpec

    _reject_unexpected(payload.keys(), _SDPA_BWD_TOP_KEYS, "sdpa_fmha_bwd")

    shape_payload = dict(payload["shape"])
    _reject_unexpected(
        shape_payload.keys(), _SDPA_FWD_SHAPE_KEYS, "sdpa_fmha_bwd.shape"
    )
    shape = FmhaShape(**shape_payload)

    common = FmhaCommonSpec(
        shape=shape,
        dtype=payload["dtype"],
        mask_mode=payload["mask_mode"],
    )
    return FmhaBwdSpec(
        common=common,
        seqlen_q=payload["seqlen_q"],
        seqlen_k=payload["seqlen_k"],
    )


def _sdpa_lse_prep_spec_from_payload(payload: dict):
    """Deserialize the wire payload into an ``SdpaLsePrepSpec``.

    Shared by the compile and applicability paths so both honour one
    deserialization contract -- the whitelist and the dataclass
    construction can't drift between them.

    Caller contract: ``payload`` is the flat ``(batch, num_query_heads,
    seqlen_q)`` dict the C++ stats-prep payload emits -- no nested shape dict.
    Keys are whitelisted against ``_SDPA_LSE_PREP_KEYS`` so an unexpected field
    fails closed rather than being silently forwarded. The C++ field names map
    onto the DSL's terse ``B`` / ``Hq`` / ``Sq`` spec fields here.

    The target arch is NOT part of the payload: it is an orthogonal compile
    target threaded as a separate argument (mirroring the DSL, whose
    ``SdpaLsePrepSpec`` likewise has no arch field).
    """
    from ck_dsl.instances.common.sdpa_lse_prep import SdpaLsePrepSpec

    _reject_unexpected(payload.keys(), _SDPA_LSE_PREP_KEYS, "sdpa_lse_prep")

    return SdpaLsePrepSpec(
        B=payload["batch"],
        Hq=payload["num_query_heads"],
        Sq=payload["seqlen_q"],
    )


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

    spec, batch, generate_stats = _sdpa_fwd_spec_from_payload(payload)

    kernel_def = build_fmha_fwd_mfma(spec, arch=arch)
    artifact = compile_kernel(kernel_def, arch=arch)

    # Grid: (seqlen_q // BLOCK_M, num_query_heads, batch) -- one wave64/32
    # CTA per (q_tile, head, batch) triple. Block is one wave; the inner
    # body assumes a single wave per CTA. Stats-on / stats-off share the
    # same grid + block (the LSE store rides the existing CTA mapping).
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
        # The schema gains the 16th ``LSE_out`` pointer slot only when the
        # opt-in stats output is requested; the stats-off schema is the
        # byte-identical 15-slot ABI.
        "arg_schema": _sdpa_fwd_arg_schema(generate_stats),
        "isa": artifact.isa,
    }


def _normalize_unified_dtype(dtype: str) -> str:
    """Map the C++ spec dtype spelling onto the kernel's spelling.

    ``SdpaSpec::dtype`` uses ``"f16"`` whereas the unified tiled-2D kernel
    (``UnifiedAttention2DTiledSpec`` / ``UnifiedAttentionProblem``) uses
    ``"fp16"``. ``"bf16"`` is identical in both. Any other spelling is left
    as-is so the dataclass's own validation surfaces the unsupported type.
    """
    return "fp16" if dtype == "f16" else dtype


def _sdpa_fwd_unified_problem_and_knobs(payload: dict):
    """Deserialize the unified wire payload into ``(problem, knobs_dict, block_size)``.

    Shared by the compile and applicability paths so both honour one
    deserialization contract -- the whitelist and the dataclass
    construction can't drift between them.

    Caller contract: ``payload`` is the dict ``sdpaSpecToPayload`` emits for
    the unified path -- a nested ``"shape"`` dict (head_size /
    num_query_heads / num_kv_heads) and a nested ``"knobs"`` dict (the nine
    ``SdpaPerfKnobs`` fields), plus the top-level ``dtype`` / ``mask_mode`` /
    ``seqlen_q`` / ``seqlen_k`` codegen inputs, the ``batch`` extent, and the
    paged/varlen lanes (``is_paged`` / ``block_size`` / ``is_varlen`` /
    ``sliding_window`` / ``use_sinks``). Keys are whitelisted against
    ``_SDPA_FWD_UNIFIED_TOP_KEYS`` / ``_SDPA_FWD_SHAPE_KEYS`` /
    ``_SDPA_FWD_UNIFIED_KNOB_KEYS`` so an unexpected field fails closed.

    The dense-degenerate problem maps ``num_seqs = batch``,
    ``total_q = batch * seqlen_q``, ``max_seqlen_q = seqlen_q``, and
    ``max_seqlen_k = seqlen_k``. The unified kernel is always paged; the
    payload's ``block_size`` (finalised on the C++ side for the dense path)
    is the cache-block size the marshalling lays out.

    The target arch is NOT part of the payload: it is an orthogonal compile
    target threaded as a separate argument (mirroring the DSL).
    """
    from ck_dsl.instances.common.attention_unified import UnifiedAttentionProblem

    _reject_unexpected(
        payload.keys(), _SDPA_FWD_UNIFIED_TOP_KEYS, "sdpa_fmha_fwd_unified"
    )

    shape_payload = dict(payload["shape"])
    _reject_unexpected(
        shape_payload.keys(), _SDPA_FWD_SHAPE_KEYS, "sdpa_fmha_fwd_unified.shape"
    )

    knobs_payload = dict(payload["knobs"])
    _reject_unexpected(
        knobs_payload.keys(),
        _SDPA_FWD_UNIFIED_KNOB_KEYS,
        "sdpa_fmha_fwd_unified.knobs",
    )

    batch = int(payload["batch"])
    seqlen_q = int(payload["seqlen_q"])
    seqlen_k = int(payload["seqlen_k"])
    block_size = int(payload["block_size"])
    dtype = _normalize_unified_dtype(payload["dtype"])

    problem = UnifiedAttentionProblem(
        total_q=batch * seqlen_q,
        num_seqs=batch,
        num_query_heads=int(shape_payload["num_query_heads"]),
        num_kv_heads=int(shape_payload["num_kv_heads"]),
        head_size=int(shape_payload["head_size"]),
        block_size=block_size,
        max_seqlen_q=seqlen_q,
        max_seqlen_k=seqlen_k,
        dtype=dtype,
        sliding_window=int(payload.get("sliding_window", 0)),
        use_sinks=bool(payload.get("use_sinks", False)),
    )
    return problem, knobs_payload, block_size


def _unified_tiled_spec_from_problem(problem, knobs: dict):
    """Build a ``UnifiedAttention2DTiledSpec`` from the problem + chosen knobs.

    Mirrors the field mapping the runtime dispatcher's
    ``_tiled_spec_from_problem`` performs, but takes the perf knobs from the
    provider's scorer-driven selection (passed on the wire) rather than
    re-deriving them from a device-detected selector. The nine knob fields
    map 1:1 onto the kernel spec; the remaining spec fields come from the
    problem shape. ``tile_size`` of 0 ("unset") is forwarded as ``None`` so
    the kernel defaults ``T = block_size``; ``waves_per_eu`` of 0 likewise
    maps to ``None`` (let the LLVM heuristic decide).

    ``num_seqs`` is carried onto the spec so the binary-search trip count
    specialises to the problem's batch (matching the dispatcher).
    """
    from ck_dsl.instances import UnifiedAttention2DTiledSpec

    tile_size = int(knobs.get("tile_size", 0))
    waves_per_eu = int(knobs.get("waves_per_eu", 0))
    return UnifiedAttention2DTiledSpec(
        head_size=problem.head_size,
        block_size=problem.block_size,
        num_query_heads=problem.num_query_heads,
        num_kv_heads=problem.num_kv_heads,
        dtype=problem.dtype,
        use_sinks=problem.use_sinks,
        sliding_window=problem.sliding_window,
        has_softcap=problem.softcap > 0.0,
        num_seqs=problem.num_seqs,
        num_warps=int(knobs.get("num_warps", 1)),
        block_m_per_warp=int(knobs.get("block_m_per_warp", 16)),
        tile_size=tile_size if tile_size > 0 else None,
        waves_per_eu=waves_per_eu if waves_per_eu > 0 else None,
        use_mfma_32x32=bool(knobs.get("use_mfma_32x32", False)),
        use_transposed_qk_32x32=bool(knobs.get("use_transposed_qk_32x32", False)),
        use_register_pv=bool(knobs.get("use_register_pv", False)),
        use_early_v_schedule=bool(knobs.get("use_early_v_schedule", False)),
        use_fast_paged_kv_desc=bool(knobs.get("use_fast_paged_kv_desc", False)),
    )


def _unified_grid(problem, num_warps: int, block_m_per_warp: int):
    """Recompute the unified tiled-2D launch grid for the chosen BLOCK_M.

    Mirrors the dispatcher's hot-path grid math
    (``attention_unified._run_2d_tiled``): the grid is
    ``(num_kv_heads, total_num_q_blocks, 1)`` where

      block_m = num_warps * block_m_per_warp
      block_q = block_m // num_queries_per_kv  (if NQK <= block_m, else 1)
      total_num_q_blocks = total_q // block_q + num_seqs

    The provider MUST recompute the grid with the SAME num_warps /
    block_m_per_warp the kernel was built with: a mismatch launches the
    wrong number of CTAs and the kernel's q_block_local_idx math touches
    the wrong query positions.
    """
    block_m = num_warps * block_m_per_warp
    nqk = problem.num_queries_per_kv
    block_q = block_m // nqk if nqk <= block_m else 1
    total_num_q_blocks = problem.total_q // block_q + problem.num_seqs
    return (int(problem.num_kv_heads), int(total_num_q_blocks), 1)


def _compile_sdpa_fwd_unified(payload: dict, arch: str) -> dict:
    """Build + compile the unified paged/varlen tiled-2D attention kernel.

    ``arch`` is threaded to BOTH ``build_unified_attention_2d_tiled`` (which
    rejects non-gfx950 targets before any IR is emitted and resolves the
    per-arch MFMA atoms) and ``compile_kernel`` (which selects the ISA
    triple). The arch is the explicit compile target -- this path
    deliberately does NOT call ``_resolve_attention_arch`` (which
    device-detects the running GPU); the provider always knows its target
    arch and a Phase-2 host compile must not depend on a present device.

    The scale / k_scale / v_scale / out_scale / softcap floats and the
    block_table_stride / num_seqs / qq_bias_stride_0 i32s are launch-time
    arguments the host packs into the 18-slot arg buffer per
    :func:`_sdpa_fwd_unified_arg_schema`; they are not codegen inputs, so
    they are absent from the payload and never baked into the kernel.
    """
    from ck_dsl.helpers.compile import compile_kernel
    from ck_dsl.instances import build_unified_attention_2d_tiled

    problem, knobs, _block_size = _sdpa_fwd_unified_problem_and_knobs(payload)
    spec = _unified_tiled_spec_from_problem(problem, knobs)

    kernel = build_unified_attention_2d_tiled(spec, arch=arch)
    artifact = compile_kernel(kernel, arch=arch)

    num_warps = spec.num_warps
    block = (64 * num_warps, 1, 1)
    grid = _unified_grid(problem, num_warps, spec.block_m_per_warp)

    return {
        "hsaco": artifact.hsaco,
        "kernel_name": artifact.kernel_name,
        "kind": "sdpa_fmha_fwd_unified",
        "grid": grid,
        "block": block,
        # The tiled-2D body stages Q / K / V / P / Acc slabs via smem_alloc
        # with statically-known shapes, so the dynamic-LDS arg to
        # hipModuleLaunchKernel is zero -- static LDS lives in the HSACO's
        # kernarg descriptor.
        "lds_bytes": 0,
        "arg_schema": _sdpa_fwd_unified_arg_schema(),
        "isa": artifact.isa,
    }


def _compile_sdpa_bwd(payload: dict, arch: str) -> dict:
    """Build + compile an FMHA backward kernel from the payload.

    ``arch`` is threaded to BOTH ``build_fmha_bwd`` (which validates the spec
    against the target's wave size / thread cap and sizes the warp body to the
    target wave) and ``compile_kernel`` (which selects the ISA triple +
    lowering backend). Passing it to only one would mis-build the kernel on any
    non-default arch.

    The scales (``scale_log2`` / ``scale_inv``) and the Q/K/V/dO/dQ/dK/dV
    strides are launch-time arguments the host packs into the arg buffer per
    :func:`_sdpa_bwd_arg_schema`; they are not codegen inputs, so they are
    absent from the payload and never baked into the kernel. The block dim is
    one wave (``wave_size`` from the target -- 64 on CDNA, 32 on RDNA),
    matching the one-warp-per-CTA grid the DSL builder emits.
    """
    from ck_dsl.core.arch import ArchTarget
    from ck_dsl.helpers.compile import compile_kernel
    from ck_dsl.instances.common.fmha_bwd import build_fmha_bwd, fmha_bwd_grid

    spec = _sdpa_bwd_spec_from_payload(payload)

    kernel_def = build_fmha_bwd(spec, arch=arch)
    artifact = compile_kernel(kernel_def, arch=arch)

    # Grid: (seqlen_q, num_query_heads, 1) -- one wave64/32 warp CTA per
    # (q_token, head) pair; batch is folded into the launch via offsetting
    # rather than a grid axis. Block is one wave; the warp body assumes a
    # single wave per CTA.
    grid = fmha_bwd_grid(spec)
    wave_size = ArchTarget.from_gfx(arch).wave_size
    block = (wave_size, 1, 1)

    return {
        "hsaco": artifact.hsaco,
        "kernel_name": artifact.kernel_name,
        "kind": "sdpa_fmha_bwd",
        "grid": tuple(grid),
        "block": block,
        # The backward warp body keeps its accumulators in registers and
        # writes dQ / dK / dV via global atomics, so it allocates no dynamic
        # LDS -- the dynamic-LDS arg to hipModuleLaunchKernel is zero.
        "lds_bytes": 0,
        "arg_schema": _sdpa_bwd_arg_schema(),
        "isa": artifact.isa,
    }


def _compile_sdpa_lse_prep(payload: dict, arch: str) -> dict:
    """Build + compile an FMHA-bwd stats-prep kernel from the payload.

    ``arch`` is threaded to BOTH ``build_sdpa_lse_prep`` (which validates the
    spec resolves to a known target) and ``compile_kernel`` (which selects the
    ISA triple + lowering backend). Passing it to only one would mis-build the
    kernel on any non-default arch -- though the emitted IR here is
    arch-independent, only the validation consults the target.

    The stats / M_out / L_out pointers are launch-time arguments the host packs
    into the arg buffer per :func:`_sdpa_lse_prep_arg_schema`; they are not
    codegen inputs. The block dim is a fixed 64 threads (one thread per
    q-position within a tile), matching the ``(ceil(Sq/64), Hq, B)`` grid the
    DSL builder emits.
    """
    from ck_dsl.helpers.compile import compile_kernel
    from ck_dsl.instances.common.sdpa_lse_prep import (
        build_sdpa_lse_prep,
        sdpa_lse_prep_grid,
    )

    spec = _sdpa_lse_prep_spec_from_payload(payload)

    kernel_def = build_sdpa_lse_prep(spec, arch=arch)
    artifact = compile_kernel(kernel_def, arch=arch)

    # Grid: (ceil(Sq/64), Hq, B) -- one CTA per (Sq-tile, head, batch),
    # one thread per q-position. Block is a fixed 64 threads.
    grid = sdpa_lse_prep_grid(spec)
    block = (64, 1, 1)

    return {
        "hsaco": artifact.hsaco,
        "kernel_name": artifact.kernel_name,
        "kind": "sdpa_lse_prep",
        "grid": tuple(grid),
        "block": block,
        # The stats-prep body is a plain transpose + rescale with all state in
        # registers, so it allocates no dynamic LDS -- the dynamic-LDS arg to
        # hipModuleLaunchKernel is zero.
        "lds_bytes": 0,
        "arg_schema": _sdpa_lse_prep_arg_schema(),
        "isa": artifact.isa,
    }


def compile(op_kind: str, payload: dict, arch: str) -> dict:
    """Compile one DSL kernel from a typed payload dict for ``arch``.

    Called by the C++ ``CompileServiceBridge::compile`` on a
    ``JitCache`` miss. Dispatches on ``op_kind`` (``"conv_implicit_gemm"``,
    ``"sdpa_fmha_fwd"``, ``"sdpa_fmha_fwd_unified"``, ``"sdpa_fmha_bwd"``,
    or ``"sdpa_lse_prep"``). Unknown kinds raise ``ValueError``, which the
    bridge surfaces as a ``HipdnnPluginException``.

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
    elif op_kind == "sdpa_fmha_fwd_unified":
        return _compile_sdpa_fwd_unified(payload, arch)
    elif op_kind == "sdpa_fmha_bwd":
        return _compile_sdpa_bwd(payload, arch)
    elif op_kind == "sdpa_lse_prep":
        return _compile_sdpa_lse_prep(payload, arch)
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

    spec, _batch, _generate_stats = _sdpa_fwd_spec_from_payload(payload)
    ok, reason = is_valid_spec(spec, arch)
    return bool(ok), str(reason)


def _is_applicable_sdpa_fwd_unified(payload: dict, arch: str):
    """Arch-aware applicability for the unified paged/varlen tiled-2D spec.

    Consults the DSL's ``supports_tiled_2d`` for ``arch`` -- the exact gate
    ``build_unified_attention_2d_tiled`` enforces (via
    ``require_tiled_attention_arch`` + the shape checks) -- so the C++
    ``isApplicable`` gate matches the compile path. No kernel is built and
    comgr is never invoked; this is a pure data-driven check against the
    target's :class:`ck_dsl.core.arch.ArchTarget` (wide-K MFMA atom
    presence) plus the head_size / block_size / GQA shape constraints.
    ``supports_tiled_2d`` returns ``(False, ...)`` for an unsupported arch,
    so this single call covers both "arch unsupported" and "shape invalid
    for this arch". The boolean knob combo is validated at spec
    ``__post_init__`` time on the compile path; the provider's enumerator
    only emits valid combos, so it is not re-checked here.
    """
    from ck_dsl.instances.gfx950.attention_tiled_2d import supports_tiled_2d

    problem, knobs, _block_size = _sdpa_fwd_unified_problem_and_knobs(payload)
    ok, reason = supports_tiled_2d(
        head_size=problem.head_size,
        block_size=problem.block_size,
        dtype=problem.dtype,
        num_queries_per_kv=problem.num_queries_per_kv,
        use_alibi=False,
        use_qq_bias=False,
        use_fp8=False,
        q_dtype=problem.dtype,
        num_warps=int(knobs.get("num_warps", 1)),
        tile_size=(
            int(knobs["tile_size"]) if int(knobs.get("tile_size", 0)) > 0 else None
        ),
        arch=arch,
    )
    return bool(ok), str(reason)


def _is_applicable_sdpa_bwd(payload: dict, arch: str):
    """Arch-aware applicability for an FMHA backward spec.

    Consults the DSL's ``is_valid_spec`` for ``arch`` -- the exact predicate
    ``build_fmha_bwd`` enforces internally -- so the C++ ``isApplicable`` gate
    matches the compile path. No kernel is built and comgr is never invoked;
    this is a pure data-driven check against the target's
    :class:`ck_dsl.core.arch.ArchTarget` (wave size, per-WG thread cap, the
    ``head_size % wave_size == 0`` warp-body requirement). ``is_valid_spec``
    also returns ``(False, ...)`` for an unknown arch, so this single call
    covers both "arch unsupported" and "shape / mask invalid for this arch".
    """
    from ck_dsl.instances.common.fmha_bwd import is_valid_spec

    spec = _sdpa_bwd_spec_from_payload(payload)
    ok, reason = is_valid_spec(spec, arch)
    return bool(ok), str(reason)


def _is_applicable_sdpa_lse_prep(payload: dict, arch: str):
    """Arch-aware applicability for an FMHA-bwd stats-prep spec.

    Consults the DSL's ``is_valid_spec`` for ``arch`` -- the exact predicate
    ``build_sdpa_lse_prep`` enforces internally -- so the C++ ``isApplicable``
    gate matches the compile path. No kernel is built and comgr is never
    invoked; the stats-prep body issues no MMA atoms and no atomics, so the
    only architecture fact consulted is that ``arch`` resolves to a known
    :class:`ck_dsl.core.arch.ArchTarget` (fail closed on an unknown target),
    plus the positivity check on the three extents. The imported predicate is
    aliased to avoid shadowing the other ``is_valid_spec`` imports in this
    module.
    """
    from ck_dsl.instances.common.sdpa_lse_prep import is_valid_spec as _prep_valid

    spec = _sdpa_lse_prep_spec_from_payload(payload)
    ok, reason = _prep_valid(spec, arch)
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
    elif op_kind == "sdpa_fmha_fwd_unified":
        return _is_applicable_sdpa_fwd_unified(payload, arch)
    elif op_kind == "sdpa_fmha_bwd":
        return _is_applicable_sdpa_bwd(payload, arch)
    elif op_kind == "sdpa_lse_prep":
        return _is_applicable_sdpa_lse_prep(payload, arch)
    raise ValueError(
        f"ck_dsl_provider.compile_service: unsupported op_kind {op_kind!r}"
    )
