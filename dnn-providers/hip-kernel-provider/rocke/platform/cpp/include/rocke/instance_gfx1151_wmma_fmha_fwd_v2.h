/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1151_wmma_fmha_fwd_v2.h -- C99 port of the RDNA (gfx11 /
 * gfx12) multi-wave WMMA attention forward (prefill) kernel
 * kernels/gfx1151/wmma_fmha_fwd_v2.py.
 *
 *   Python (wmma_fmha_fwd_v2.py)          C99 (this header)
 *   ----------------------------------    --------------------------------------
 *   @dataclass WmmaFmhaFwdV2Spec          rocke_wmma_fmha_fwd_v2_spec_t
 *     .kernel_name()                      rocke_wmma_fmha_fwd_v2_kernel_name(...)
 *     .threads                            rocke_wmma_fmha_fwd_v2_threads(...)
 *   is_valid_spec(spec, arch=)            rocke_wmma_fmha_fwd_v2_is_valid_spec(...)
 *   wmma_fmha_fwd_v2_lds_bytes(spec, arch=) rocke_wmma_fmha_fwd_v2_lds_bytes(...)
 *   build_wmma_fmha_fwd_v2(spec, arch=)   rocke_build_wmma_fmha_fwd_v2(...)
 *   wmma_fmha_fwd_v2_grid(spec, ...)      rocke_wmma_fmha_fwd_v2_grid(...)
 *   wmma_fmha_fwd_v2_signature(spec)      rocke_wmma_fmha_fwd_v2_signature(...)
 *   (build -> lower .ll convenience)      rocke_wmma_fmha_fwd_v2_lower_to_llvm(...)
 *
 * The spec carries only compile-time facts (dtype, head-width buckets, mask
 * class, bias kind, feature flags -- softcap, sinks, ALiBi, QQ-bias, packed
 * layout / page size, FP8 K/V -- and the tuning knobs). Batch, lengths, head
 * counts, head widths, strides, alignment, window bounds, the LSE gate, the
 * bias strides and every feature operand are kernel arguments. The QK ->
 * softmax -> PV body is rocke_rdna_attention_fwd_body
 * (rocke/helper_rocke.helpers.rdna_attention.h).
 *
 * Error model: the build routine routes errors through the sticky-error
 * IRBuilder; the validity gate returns a bool + reason string; the
 * convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_GFX1151_WMMA_FMHA_FWD_V2_H
#define ROCKE_INSTANCE_GFX1151_WMMA_FMHA_FWD_V2_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Flat mirror of @dataclass(frozen=True) WmmaFmhaFwdV2Spec. String fields
 * are referenced as-is; keep them alive with the spec. Start from
 * rocke_wmma_fmha_fwd_v2_spec_default() so new fields take their defaults. */
typedef struct rocke_wmma_fmha_fwd_v2_spec
{
    const char* dtype; /* "fp16" | "bf16"; default "bf16" */
    int head_dim; /* Q/K head-width bucket; default 128 */
    int head_dim_v; /* V head-width bucket; 0 => head_dim */
    const char* mask; /* "none" (default) | "causal" */
    const char* bias; /* NULL (default) | "f32" | "q" */
    bool use_softcap; /* default false */
    bool use_sinks; /* default false */
    bool use_alibi; /* default false */
    bool use_qq_bias; /* default false */
    const char* layout; /* "dense" (default) | "ragged" | "paged" */
    int page_block_size; /* paged only: tokens per page (power of two); default 0 */
    const char* kv_dtype; /* "" (default) | "fp8e4m3" */
    int block_m; /* query rows per CTA; 0 => 128, or 64 with the wave pair */
    int block_n; /* keys per tile; default 32 */
    int waves_per_eu; /* occupancy hint; default 2; 0 => none */
    const char* scheduler_strategy; /* NULL (default) => LLVM default scheduler */
    bool lazy_rescale; /* default true */
    int k_pad; /* LDS K row padding (elements); default 8 */
    int v_pad; /* LDS V^T row padding (elements); default 8 */
    int wave_pair; /* -1 (default): split above 128; 0: never; 1: always */
    const char* cta_order; /* "hqb" (default) | "hbq" */
    const char* name; /* NULL => "rocke_wmma_fmha_fwd_v2" */
} rocke_wmma_fmha_fwd_v2_spec_t;

/* Default-constructed spec (Python dataclass defaults). */
rocke_wmma_fmha_fwd_v2_spec_t rocke_wmma_fmha_fwd_v2_spec_default(void);

/* WmmaFmhaFwdV2Spec.threads: threads per CTA. */
int rocke_wmma_fmha_fwd_v2_threads(const rocke_wmma_fmha_fwd_v2_spec_t* spec);

/* WmmaFmhaFwdV2Spec.kernel_name(), NUL-terminated into out (capacity
 * out_cap). ROCKE_ERR_VALUE on an invalid enumeration or a short buffer. */
rocke_status_t rocke_wmma_fmha_fwd_v2_kernel_name(const rocke_wmma_fmha_fwd_v2_spec_t* spec,
                                                  char* out,
                                                  size_t out_cap);

/* is_valid_spec(spec, arch=arch) -> (ok, reason), including the spec's own
 * __post_init__ checks. On reject `reason` receives the message. */
bool rocke_wmma_fmha_fwd_v2_is_valid_spec(const rocke_wmma_fmha_fwd_v2_spec_t* spec,
                                          const char* arch,
                                          char* reason,
                                          size_t reason_cap);

/* wmma_fmha_fwd_v2_lds_bytes(spec, arch=arch); -1 for an unknown arch. */
int rocke_wmma_fmha_fwd_v2_lds_bytes(const rocke_wmma_fmha_fwd_v2_spec_t* spec, const char* arch);

/* build_wmma_fmha_fwd_v2(spec, arch=arch). `b` must already be initialised
 * with the spec's kernel_name(). Returns b->kernel, or NULL on an invalid
 * spec or IR-emission error (builder error set). */
rocke_kernel_def_t* rocke_build_wmma_fmha_fwd_v2(rocke_ir_builder_t* b,
                                                 const rocke_wmma_fmha_fwd_v2_spec_t* spec,
                                                 const char* arch);

/* wmma_fmha_fwd_v2_grid: grid {batch * ceil(Sq / block_m) * Hq, 1, 1} and
 * block {threads, 1, 1}. ROCKE_ERR_VALUE if the grid is empty or exceeds I32. */
rocke_status_t rocke_wmma_fmha_fwd_v2_grid(const rocke_wmma_fmha_fwd_v2_spec_t* spec,
                                           int batch,
                                           int seqlen_q,
                                           int num_query_heads,
                                           int grid[3],
                                           int block[3]);

/* wmma_fmha_fwd_v2_signature(spec): the kernel ABI. Names and type strings
 * are owned by `arena`; failure leaves the outputs untouched. */
rocke_status_t rocke_wmma_fmha_fwd_v2_signature(const rocke_wmma_fmha_fwd_v2_spec_t* spec,
                                                rocke_arena_t* arena,
                                                const rocke_sig_entry_t** out_items,
                                                size_t* out_count);

/* Convenience: build and lower to LLVM .ll text. On ROCKE_OK *out_ll is a
 * malloc'd string the caller frees; on failure it is NULL and `err`
 * (capacity err_cap) receives a diagnostic. */
rocke_status_t rocke_wmma_fmha_fwd_v2_lower_to_llvm(const rocke_wmma_fmha_fwd_v2_spec_t* spec,
                                                    const char* arch,
                                                    rocke_llvm_flavor_t flavor,
                                                    char** out_ll,
                                                    char* err,
                                                    size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1151_WMMA_FMHA_FWD_V2_H */
