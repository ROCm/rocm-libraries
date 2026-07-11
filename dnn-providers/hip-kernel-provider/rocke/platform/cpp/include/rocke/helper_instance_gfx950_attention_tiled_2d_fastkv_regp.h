/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_instance_gfx950_attention_tiled_2d_fastkv_regp.h -- C99 port of the
 * symbols named by the porting task from
 * rocke/instances/gfx950/attention_tiled_2d_fastkv_regp.py (the experimental
 * gfx950 "fast paged-KV descriptor + register-P" 2D-attention wrapper that
 * forces the production tiled R4 builder down the register-P residency path).
 *
 * Ported symbols (task list):
 *
 *   Python (attention_tiled_2d_fastkv_regp.py)        C99 (this header)
 *   -----------------------------------------------
 * ------------------------------------------------ class _FastKvRegisterPProxy
 * rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_t (spec proxy: use_register_pv property + +
 * rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_make kernel_name() suffix) +
 * rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_use_register_pv
 *                                                       +
 * rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_kernel_name make_fastkv_register_p_spec(...)
 * rocke_gfx950_make_fastkv_register_p_spec supports_fastkv_register_p_2d(...)
 * rocke_gfx950_supports_fastkv_register_p_2d build_unified_attention_2d_fastkv_register_p(...)
 * rocke_build_unified_attention_2d_fastkv_register_p
 *
 * RELATIONSHIP TO THE PRODUCTION BUILDER.
 *   The Python module does not re-implement any kernel math. It imports the
 *   gfx950 sibling's UnifiedAttention2DTiledSpec, supports_tiled_2d and
 *   build_unified_attention_2d_tiled, and:
 *     - make_fastkv_register_p_spec: dataclasses.replace() over the spec, forcing
 *       the fastKV + transposed-R4 + register-P residency knobs.
 *     - supports_fastkv_register_p_2d: delegates to supports_tiled_2d, then adds
 *       the experiment's hard shape restriction (bf16 d64_b32_h64kv8).
 *     - _FastKvRegisterPProxy: a __getattr__ pass-through to the wrapped spec that
 *       overrides use_register_pv -> True and kernel_name() -> "<base>_fastkv_regp".
 *     - build_unified_attention_2d_fastkv_register_p: 3 ValueError guards, then
 *       build_unified_attention_2d_tiled(proxy, arch).
 *
 *   In the C port the spec dataclass + supports gate + tiled builder were already
 *   ported (and are arch-parameterised: arch="gfx950" threads straight through):
 *     - rocke_attention_tiled_2d_spec_t / @property accessors / validator / config
 *         (rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h)
 *     - rocke_gfx942_attention_tiled_2d_supports                (supports_tiled_2d)
 *     - rocke_build_unified_attention_2d_tiled_scalar           (build tiled)
 *         (rocke/instance_gfx942_attention_tiled_2d.h)
 *   This header REUSES those by include and does NOT redeclare them.
 *
 * THE PROXY.
 *   Python's _FastKvRegisterPProxy is a thin attribute wrapper. The C analogue is
 *   a value struct holding the wrapped rocke_attention_tiled_2d_spec_t plus the two
 *   overrides made explicit:
 *     - use_register_pv -> always true (the @property override);
 *     - kernel_name()   -> "<wrapped.kernel_name()>_fastkv_regp".
 *   Because the wrapped spec's kernel_name() lives as a private static in the
 *   tiled glue translation unit (not a public C symbol), the proxy kernel-name
 *   helper takes the already-computed base name and appends the experiment suffix,
 *   byte-faithfully reproducing the Python f-string. Pass-through of every other
 *   attribute is just direct access to proxy.spec (the wrapped struct), exactly
 *   as Python's __getattr__ forwards to self._spec.
 *
 * BUILD ENTRY (stub-to-link kernel body).
 *   rocke_build_unified_attention_2d_fastkv_register_p reproduces the 3 Python
 *   ValueError guards (sticky-error on b) and then calls the tiled builder with
 *   the proxy applied (use_register_pv forced true). The tiled builder's
 *   ~4000-line kernel body is the stub-to-link surface ALREADY established by the
 *   rocke_build_unified_attention_2d_tiled_scalar port; nothing long is duplicated
 *   here. ``arch`` NULL == the Python default "gfx950".
 *
 * Error model mirrors the rest of the C port: pure helpers return a sentinel
 * (false / NULL); the builder variants route the first failure through the
 * sticky-error IRBuilder; a dead builder is a NULL/false no-op. Spec/proxy
 * structs are plain by-value PODs the caller owns; strings are caller-owned and
 * not copied.
 */
#ifndef ROCKE_HELPER_INSTANCE_GFX950_ATTENTION_TILED_2D_FASTKV_REGP_H
#define ROCKE_HELPER_INSTANCE_GFX950_ATTENTION_TILED_2D_FASTKV_REGP_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h" /* rocke_ir_builder_t, rocke_kernel_def_t, rocke_status_t */
/* The already-ported spec struct + @property accessors + validator + config and
 * the supports gate + tiled builder entry points. REUSED, not redeclared. */
#include "rocke/helper_helper_rocke.instances.gfx942.attention_tiled_2d.h"
#include "rocke/instance_gfx942_attention_tiled_2d.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================ *
 * _FastKvRegisterPProxy  (Python lines 29-46)
 * ============================================================ *
 *
 * Spec proxy that skips the unused P_lds allocation on transposed R4. The
 * wrapped spec is held by value; every non-overridden attribute is read directly
 * from .spec (the Python __getattr__ pass-through). The two overrides are exposed
 * as helpers below. */
typedef struct rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy
{
    /* self._spec -- the wrapped UnifiedAttention2DTiledSpec. Read any
     * non-overridden attribute straight from here (== Python __getattr__). */
    rocke_attention_tiled_2d_spec_t spec;
} rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_t;

/* __init__(self, spec): wrap the supplied spec by value. */
rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_t
    rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_make(
        const rocke_attention_tiled_2d_spec_t* spec);

/* @property use_register_pv -> True (the residency override; always true,
 * independent of the wrapped spec's own use_register_pv). */
bool rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_use_register_pv(
    const rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_t* p);

/* kernel_name(self) -> f"{self._spec.kernel_name()}_fastkv_regp".
 *
 * The wrapped spec's kernel_name() is a private static in the tiled-2D glue
 * translation unit, so this helper takes the already-computed base name
 * (``base``, the wrapped spec.kernel_name()) and writes "<base>_fastkv_regp"
 * NUL-terminated into ``out`` (capacity ``out_cap``). On success returns ROCKE_OK
 * and, if out_len != NULL, sets *out_len to the byte length (excluding NUL); a
 * too-small buffer (or NULL base/out) returns ROCKE_ERR_VALUE and writes nothing
 * usable. */
rocke_status_t rocke_gfx950_attention_tiled_2d_fastkv_regp_spec_proxy_kernel_name(const char* base,
                                                                                  char* out,
                                                                                  size_t out_cap,
                                                                                  size_t* out_len);

/* ============================================================ *
 * make_fastkv_register_p_spec(...)  (Python lines 49-94)
 * ============================================================ *
 *
 * Return a targeted fastKV + register-P R4 spec: a copy of ``spec``
 * (dataclasses.replace) with the experiment knobs forced. Faithful port:
 *
 *   num_warps                       = 4
 *   waves_per_eu                    = 2          (has_waves_per_eu -> true)
 *   tile_size                       = 2 * spec.block_size  (has_tile_size -> true)
 *   block_m_per_warp                = 32
 *   use_mfma_32x32                  = True
 *   use_transposed_qk_32x32         = True
 *   use_transposed_scalar_state     = scalar_state
 *   use_transposed_mask_once        = mask_once
 *   use_transposed_half_local_pv    = half_local_pv
 *   use_mfma32_skip_legacy_qreg     = skip_legacy_qreg
 *   use_transposed_mask_limit       = use_mask_limit (see below)
 *   use_fast_paged_kv_desc          = True
 *   use_agpr_alloc_zero             = False
 *   use_register_pv                 = False
 *
 * use_mask_limit resolution (Python keyword default mask_limit=None ->
 * has_mask_limit == false here):
 *   if !has_mask_limit:
 *       use_mask_limit = scalar_state && mask_once
 *                        && spec.sliding_window == 0
 *                        && !spec.has_softcap
 *                        && !spec.use_alibi
 *                        && !spec.use_qq_bias
 *   else:
 *       use_mask_limit = mask_limit
 *
 * Pure: emits no IR. Returns the new spec by value. The defaulted keyword args
 * (scalar_state, mask_once, half_local_pv, skip_legacy_qreg) all default False in
 * Python; the caller passes them explicitly. */
rocke_attention_tiled_2d_spec_t
    rocke_gfx950_make_fastkv_register_p_spec(const rocke_attention_tiled_2d_spec_t* spec,
                                             bool scalar_state,
                                             bool mask_once,
                                             bool has_mask_limit,
                                             bool mask_limit,
                                             bool half_local_pv,
                                             bool skip_legacy_qreg);

/* ============================================================ *
 * supports_fastkv_register_p_2d(...)  (Python lines 97-141)
 * ============================================================ *
 *
 * Admission gate for the experimental kernel. Faithful port: first delegate to
 * supports_tiled_2d (here rocke_gfx942_attention_tiled_2d_supports) with the
 * experiment's fixed args (num_warps=4, kv_storage_dtype=None,
 * tile_size = tile_size if has_tile_size else 2*block_size, arch threaded
 * through). On a tiled reject, return that (ok=false, reason copied). On accept,
 * apply the experiment's hard shape restriction:
 *
 *   dtype == "bf16" && head_size == 64 && block_size == 32
 *     && num_query_heads == 64 && num_kv_heads == 8
 *
 * else return (false, "fastKV register-P experiment is restricted to bf16
 * d64_b32_h64kv8"). On full accept return (true, "supported").
 *
 * Optional[int] tile_size: has_tile_size == false encodes Python None. q_dtype
 * NULL == Python None. ``arch`` NULL == "gfx950". ``reason`` may be NULL (then
 * only the bool is produced). Pure: emits no IR. */
typedef struct rocke_gfx950_supports_fastkv_register_p_2d_args
{
    int head_size;
    int block_size;
    const char* dtype; /* "bf16" expected; other rejected by shape gate */
    int num_queries_per_kv;
    bool use_alibi;
    bool use_qq_bias;
    bool use_fp8;
    const char* q_dtype; /* NULL == Python None                          */
    int num_query_heads;
    int num_kv_heads;
    bool has_tile_size; /* Optional[int] None                           */
    int tile_size;
    const char* arch; /* NULL == "gfx950"                             */
} rocke_gfx950_supports_fastkv_register_p_2d_args_t;

bool rocke_gfx950_supports_fastkv_register_p_2d(
    const rocke_gfx950_supports_fastkv_register_p_2d_args_t* args, char* reason, size_t reason_cap);

/* ============================================================ *
 * build_unified_attention_2d_fastkv_register_p(...)  (Python lines 144-162)
 * ============================================================ *
 *
 * Build the experimental fastKV + register-P 2D attention kernel into ``b``.
 * Faithful port of the wrapper: three ValueError guards, then the tiled builder
 * with the register-P proxy applied (use_register_pv forced true).
 *
 *   if not spec.use_fast_paged_kv_desc:
 *       ValueError "fastKV register-P experiment requires use_fast_paged_kv_desc"
 *   if not (spec.use_mfma_32x32 and spec.use_transposed_qk_32x32):
 *       ValueError "fastKV register-P experiment requires transposed R4"
 *   if spec.kv_storage_dtype is not None:
 *       ValueError "fastKV register-P experiment does not support FP8 KV cache"
 *   return build_unified_attention_2d_tiled(_FastKvRegisterPProxy(spec), arch)
 *
 * On a failing guard it records the Python ValueError text on ``b`` (ROCKE_ERR_VALUE)
 * and returns NULL. Otherwise it forms the proxy (use_register_pv -> true) and
 * calls rocke_build_unified_attention_2d_tiled_scalar with arch threaded through;
 * that builder's kernel body is the established stub-to-link surface. ``arch``
 * NULL == the Python default "gfx950". A dead/NULL builder is a NULL no-op. */
rocke_kernel_def_t* rocke_build_unified_attention_2d_fastkv_register_p(
    rocke_ir_builder_t* b, const rocke_attention_tiled_2d_spec_t* spec, const char* arch);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_INSTANCE_GFX950_ATTENTION_TILED_2D_FASTKV_REGP_H */
