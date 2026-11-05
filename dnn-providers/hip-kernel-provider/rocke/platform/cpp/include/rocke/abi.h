/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/abi.h -- the two compatibility contracts between a rocke engine and the
 * things that talk to it.
 *
 * These are NOT the same as rocke/rocke_build_id.h. Those stamps are provenance:
 * a content hash and a date, which change on every commit and every build. They
 * answer "were these artifacts built together". Comparing them for
 * compatibility would force a lockstep upgrade of the entire stack on any source
 * change, which is why they cannot serve as the contract and why these exist.
 *
 * There are two numbers here because two different things can be mismatched,
 * for different reasons, with different blast radii. Conflating them would mean
 * that adding a recipe instruction invalidates every hipDNN binary, and that
 * changing a struct invalidates every bundle on disk. Neither is true.
 *
 *   ROCKE_ABI_VERSION   BINARY. Does this header match this .so? Governs struct
 *                       layouts, enum values and function signatures -- the
 *                       things a C++ caller and the ctypes bindings in
 *                       src/online.py hard-code at their own build time. A
 *                       mismatch here is memory-unsafe, not merely wrong, so
 *                       callers should check it once at load and refuse.
 *
 *   ROCKE_RECIPE_ABI    WIRE. Can this engine correctly read this CBOR artifact?
 *                       Governs the recipe/bundle/guard content vocabulary. A
 *                       bundle is a persisted artifact that outlives the engine
 *                       that wrote it and may be read by an engine older or
 *                       newer than the writer, so this is a data-format question
 *                       and is answered per artifact, not per process.
 *
 * -------------------------------------------------------------------- wire
 * The wire check is deliberately NOT "does the artifact's version equal mine".
 * Each artifact declares the OLDEST reader that can read it correctly:
 *
 *     "abi": {"min_reader": 1, "writer": 1, "engine": "...", "build_id": "..."}
 *
 * and a reader refuses exactly when `min_reader > ROCKE_RECIPE_ABI`. The
 * generator derives min_reader from what the artifact actually USES (see
 * python/rocke/portable_ir/src/abi.py), so a recipe built by a newer generator
 * that happens to use only old constructs still declares 1 and every engine
 * reads it. An equality check, or a plain monotonic format version, would
 * instead reject every newer bundle regardless of content -- turning a
 * generator upgrade into a fleet-wide flag day for artifacts that did not
 * change in any way that matters.
 *
 * `writer`, `engine` and `build_id` in that block are provenance for debugging
 * a bad artifact. They are never compared. Only `min_reader` decides anything.
 *
 * A MISSING abi block means level 1: every recipe recorded before this existed
 * stays readable, which is the same additive rule guards follow.
 *
 * ------------------------------------------------------------- when to bump
 * Bump ROCKE_ABI_VERSION when a C caller compiled against the old header would
 * be WRONG about memory: a struct gains/loses/reorders a field, an enumerator is
 * inserted rather than appended, a function's signature changes. Appending a new
 * function, or a new enumerator at the end, does not require a bump.
 *
 * Bump ROCKE_RECIPE_ABI when the engine learns a construct that an older engine
 * would MISREAD rather than reject -- and register the construct in abi.py so
 * that artifacts using it declare the higher min_reader. Note the asymmetry
 * worth understanding before reaching for a bump: the VM already fails loudly on
 * an unknown instruction op, an unknown opcode and an unknown intexpr node, so
 * adding one of those is self-policing and an old engine refuses it on its own.
 * The bump matters for changes an old engine would silently accept and get
 * wrong -- a changed default, a reinterpreted field, a relaxed invariant.
 *
 * LIMIT worth stating: this gate covers what the VM dispatches on. Attribute
 * VALUES are passed through to the IR builder uninterpreted, so their meaning is
 * the lowerer's contract, not the VM's, and a lowerer that silently ignores an
 * attribute it does not know is not something this version can catch.
 */
#ifndef ROCKE_ABI_H
#define ROCKE_ABI_H

#ifdef __cplusplus
extern "C" {
#endif

/* Binary ABI of the shared library: struct layout, enum values, signatures.
 * 2: rocke_memory_caps_t gained mid-struct fields; rocke_opcode gained
 *    mid-enum gfx1250 cluster/prefetch opcodes.
 * 3: rocke_mma_op_t gained emulation_recipe and emulation_physical_op_id at its
 *    end, which changes sizeof and with it the stride of every ops[] array a
 *    catalog hands out.
 * 4: rocke_memory_caps_t gained has_global_float_atomic_add at its end, which
 *    changes its size and with it the offset of rocke_arch_target_t.limits.
 * 5: rocke_tensor_view_t gained buffer (rocke_tv_buffer_resource_t) at its end,
 *    which changes its size and that of every struct embedding a view.
 * 6: rocke_attention_tiled_2d_spec_t (gfx942) and
 *    rocke_unified_attention_reduce_tiled_spec_t (gfx942, gfx950) gained
 *    emit_lse at their end, which changes their size.
 * 7: rocke_mma_op_t gained dst_dtype at its end (sizeof again), and the
 *    rocke_mma_catalog_enumerate / _has_shape / _select_largest_k /
 *    _op_for_shape signatures gained a trailing dst_dtype parameter.
 * 8: rocke_direct_fwd_wmma_spec_t gained strided_view, fold_n and poly_view
 *    at its end, which changes its size; rocke_direct_conv_problem_total_c /
 *    _total_k return long long (they returned int), which changes their
 *    signature.
 * 9: rocke_block_scale_gemm_spec_t lost per_input_row at its end, which
 *    changes its size.
 * 10: rocke_param_opts_t gained nocapture, nonnull and dereferenceable (each
 *     with a *_set flag) and the integer facts (rocke_int_facts_t) at its
 *     end, which changes its size.
 * 11: rocke_implicit_gemm_conv_wgrad_spec_t gained register_prefetch in its
 *     old tail padding: size unchanged, but an old caller leaves that byte
 *     unset.
 * 12: indexed matrix sources, independent destination, and source-owned scales
 *     replace the flat MMA fields and role enum. Indexed queries supplement the
 *     result-aware A/B/C queries; native consumers must rebuild. ABI11 libraries
 *     do not share this combined layout and must be refused.
 * 13: rocke_gemm_trait_spec_t gained the gfx1250 TDM and K-loop fields (tdm,
 *     tdm_depth, ab_load_elem_bytes and its _set flag, split_barrier,
 *     tdm_alternate_issuers, tdm_early_refill), which changes its size and the
 *     layout of every GEMM spec that embeds it.
 * 14: rocke_implicit_gemm_conv_spec_t, rocke_dgrad_conv_spec_t and
 *     rocke_implicit_gemm_conv_wgrad_spec_t gained mma_dtype (TF32 compute)
 *     mid-struct, and rocke_coalesced_tile_loader_t and
 *     rocke_async_tile_loader_t gained elem_dtype, which changes their layout
 *     and size.
 * 15: rocke_direct_conv_wgrad_spec_t gained image_per_block through
 *     uniform_wave at its end, which changes its size.
 * 16: rocke_implicit_gemm_conv_wgrad_spec_t gained prefetch_global_regs in
 *     its old tail padding: size unchanged, but an old caller leaves that byte
 *     unset.
 * 17: rocke_attention_tiled_2d_spec_t gained lse_head_major and
 *     rocke_unified_attention_3d_tiled_spec_t gained emit_lse, each a bool in
 *     its old tail padding: sizes unchanged, but an old caller leaves that byte
 *     unset.
 * 18: rocke_rdna_attn_params_t gained block_table_v mid-struct and k_ptr,
 *     v_ptr and live_rows at its end, which changes its size and field
 *     offsets; rocke_wmma_fmha_fwd_v2_spec_t gained dense_q and seqused_q in
 *     the padding after page_block_size (its size and offsets are unchanged). */
#define ROCKE_ABI_VERSION 18

/* Wire ABI: the CBOR recipe/bundle/guard vocabulary this engine understands.
 * 2: list-tagged ("l") kernel attrs such as cluster_dims; level-1 engines drop them.
 * 3: the scf_if_else instruction (converging runtime if/else, scf.if_else);
 *    older engines reject it as an unknown instruction op.
 * 4: the workgroup_size kernel attr (the exact launch block size); an older
 *    lowerer ignores it and sizes the kernel for max_workgroup_size instead.
 *    And the integer facts on a param instruction (ge, gt, le, lt, eq,
 *    multiple_of); an older reader drops a param attr it does not know, so it
 *    would replay the kernel and plan a launch that checks none of them. */
#define ROCKE_RECIPE_ABI 4

/* The values the LOADED library was compiled with. A caller compares these
 * against the ROCKE_* macros it was itself compiled with; they differ exactly
 * when the header and the binary come from different builds.
 *
 * rocke_abi_version() is the one that matters for a ctypes or dlopen caller,
 * because every argument it passes is described by the old header. Check it
 * before the first real call. */
int rocke_abi_version(void);
int rocke_recipe_abi_level(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_ABI_H */
