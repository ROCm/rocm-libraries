/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1151_wmma_fmha_fwd.h -- C99 port of the gfx1151 (RDNA3.5 /
 * Strix Halo) WMMA FMHA forward kernel instance builder
 * rocke/instances/gfx1151/wmma_fmha_fwd.py.
 *
 * QK^T and PV use dtype-matched gfx11 FP16/BF16 WMMA instructions with
 * a wave32 thread mapping. This adapter owns the gfx1151 kernel ABI, the
 * query/head/batch-and-value-tile grid decode, and the per-batch
 * pointer arithmetic, and hands the wave32 QK -> online-softmax -> PV loop to
 * the already-ported common inner body
 * rocke.helpers.mfma_attention.mfma_attention_fwd_inner_body, which dispatches
 * to the WMMA wave32 path on an RDNA target (and the MFMA wave64 path on CDNA).
 *
 *   Python (wmma_fmha_fwd.py)             C99 (this header)
 *   ----------------------------------    --------------------------------------
 *   @dataclass WmmaFmhaFwdSpec            rocke_wmma_fmha_fwd_spec_t
 *     .kernel_name()                      rocke_wmma_fmha_fwd_kernel_name(...)
 *   is_valid_spec(spec, arch)             rocke_wmma_fmha_fwd_is_valid_spec(...)
 *   build_wmma_fmha_fwd(spec, arch)       rocke_build_wmma_fmha_fwd(...)
 *   wmma_fmha_fwd_grid(spec, sq, hq, batch) rocke_wmma_fmha_fwd_grid(...)
 *   (signature helper, parity)            rocke_wmma_fmha_fwd_signature(...)
 *   (build -> lower .ll convenience)      rocke_wmma_fmha_fwd_lower_to_llvm(...)
 *
 * SPEC AS A FLAT C STRUCT. The spec carries only compile-time algorithm and
 * dtype facts. Batch, sequence lengths, query/KV head counts, strides,
 * bottom-right alignment, window bounds, and optional LSE output are runtime
 * kernel arguments so one code object serves every matching shape. Python and
 * C pass the same spec to the shared inner body and emit byte-identical IR.
 *
 * Error model mirrors the rest of the C port: the build routine routes errors
 * through the sticky-error IRBuilder; the validity gate returns a bool + reason
 * string; the convenience lower returns a rocke_status_t.
 */
#ifndef ROCKE_INSTANCE_GFX1151_WMMA_FMHA_FWD_H
#define ROCKE_INSTANCE_GFX1151_WMMA_FMHA_FWD_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h" /* rocke_sig_entry_t */
#include "rocke/helper_rocke.instances.common._fmha_common.h" /* mask mode enum */
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

/* WMMA tile constants (Python module-level _BLOCK_*). */
#define ROCKE_WMMA_FMHA_FWD_BLOCK_M 16 /* Q rows per wave per CTA            */
#define ROCKE_WMMA_FMHA_FWD_BLOCK_K 16 /* K positions per K-tile (WMMA N)    */

/* ------------------------------------------------------------ WmmaFmhaFwdSpec
 *
 * Flat mirror of @dataclass(frozen=True) WmmaFmhaFwdSpec.
 *   head_size        : multiple of 16 (WMMA K/N tile); includes 64/96/128/256.
 *   mask_mode        : NONE, CAUSAL, or SLIDING_WINDOW (arbitrary diagonal band).
 *   v_lds_stage      : optional V staging through LDS; default false.
 *   name             : NULL => "rocke_wmma_fmha_fwd".
 *   query_tail/kv_tail : independent partial-tile specializations; default false.
 *   use_softcap/use_sinks/use_alibi/use_qq_bias : optional runtime score inputs.
 *   layout           : dense, ragged (packed Q/K/V), or paged (packed Q).
 *   page_block_size  : positive power of two for paged; zero otherwise.
 *   kv_dtype         : empty for Q-matched storage, or OCP fp8e4m3 bytes.
 *
 * String fields are referenced as-is; keep them alive with the spec. */
/* ABI: this struct is passed by value layout, so changing fields changes it.
 * The kv_dtype, transposed_qk, block_n, num_waves, scheduler_strategy,
 * value_tile_size, causal_tile_skip, v_head_size and
 * use_attn_bias/bias_dtype fields (and the batch * value_tiles grid z axis)
 * form the "rocke-attention-gfx1151/v5" ABI. Code built against an older
 * header must be recompiled; zero-initialise the struct with
 * rocke_wmma_fmha_fwd_spec_default so new fields take their defaults. */
typedef struct rocke_wmma_fmha_fwd_spec
{
    int head_size;
    rocke_fmha_mask_mode_t mask_mode; /* NONE / CAUSAL / SLIDING_WINDOW */
    bool v_lds_stage; /* default false                         */
    const char* name; /* NULL => "rocke_wmma_fmha_fwd"        */
    const char* dtype; /* "fp16" default; "f16" alias or "bf16" */
    bool query_tail; /* default false */
    bool kv_tail; /* default false */
    bool use_softcap; /* runtime softcap scalar when enabled */
    bool use_sinks;
    bool use_alibi;
    bool use_qq_bias;
    const char* layout; /* "dense" default; "ragged" or "paged" use packed Q */
    int page_block_size; /* positive power of two for paged; zero otherwise */
    const char* kv_dtype; /* "" -> Q dtype; "fp8e4m3" -> OCP E4M3FN byte storage */
    bool transposed_qk; /* FP16/BF16 D64/D128 aligned dense attention */
    int block_n; /* transposed-QK key tile: 32 or 64; default 32 */
    int num_waves; /* transposed-QK waves per CTA: 1 or 2; default 1 */
    const char*
        scheduler_strategy; /* NULL: backend default; otherwise a validated codegen policy */
    int value_tile_size; /* 0 => full head; proper multiple-of-16 head divisor otherwise */
    bool
        causal_tile_skip; /* standard path: bound the K loop at the causal diagonal; default false */
    int v_head_size; /* 0 => V/O width equals head_size; else a distinct multiple of 16 */
    bool use_attn_bias; /* dense additive bias [B|1, H|1, Sq|1, Sk]; unit-stride keys */
    const char* bias_dtype; /* "f32" (default) or "q" (the Q dtype) */
} rocke_wmma_fmha_fwd_spec_t;

/* Default-constructed spec (Python dataclass defaults). The caller must set
 * head_size; every other problem dimension is supplied at launch. */
rocke_wmma_fmha_fwd_spec_t rocke_wmma_fmha_fwd_spec_default(void);

/* WmmaFmhaFwdSpec.kernel_name(): compile-time algorithm, head dimension,
 * dtype, mask class, storage/layout, scheduler, and tile configuration.
 * Runtime dimensions and window bounds are intentionally absent. Writes
 * NUL-terminated into out (capacity out_cap). */
rocke_status_t rocke_wmma_fmha_fwd_kernel_name(const rocke_wmma_fmha_fwd_spec_t* spec,
                                               char* out,
                                               size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). The dtype-matched WMMA atom must
 * exist on `arch`, the target must be wave32, and the compile-time combination
 * must be internally legal. `arch` NULL => "gfx1151". On reject `reason`
 * receives the structured message and the function returns false. */
bool rocke_wmma_fmha_fwd_is_valid_spec(const rocke_wmma_fmha_fwd_spec_t* spec,
                                       const char* arch,
                                       char* reason,
                                       size_t reason_cap);

/* build_wmma_fmha_fwd(spec, arch). Validates and emits WMMA FMHA forward IR.
 * Use wmma_fmha_fwd_grid for query/value-tile geometry. `arch` NULL defaults
 * to "gfx1151". An invalid spec or any IR-emission error returns NULL.
 *
 * `b` is the destination IR builder to emit into; if NULL this instance owns a
 * transient builder (see the .c note / sibling instance_fmha_mfma entry). For
 * parity with the documented CALL PATTERN prefer the lower-to-llvm convenience
 * when the kernel must outlive the call. */
rocke_kernel_def_t* rocke_build_wmma_fmha_fwd(rocke_ir_builder_t* b,
                                              const rocke_wmma_fmha_fwd_spec_t* spec,
                                              const char* arch);

/* wmma_fmha_fwd_grid(spec, seqlen_q, num_query_heads, batch) returns query
 * groups, runtime query heads, and batch * value_tiles. */
rocke_status_t rocke_wmma_fmha_fwd_grid(const rocke_wmma_fmha_fwd_spec_t* spec,
                                        int seqlen_q,
                                        int num_query_heads,
                                        int batch,
                                        int out[3]);

/* Fixed runtime-shape kernel ABI: Q/K/V/O/LSE, lengths, head counts,
 * bottom-right/window controls, the LSE write gate, and all dense
 * batch/token/head strides, followed by enabled score arguments and
 * packed-layout metadata. Names and type strings are owned by `arena`;
 * failure leaves output pointers untouched. */
rocke_status_t rocke_wmma_fmha_fwd_signature(const rocke_wmma_fmha_fwd_spec_t* spec,
                                             rocke_arena_t* arena,
                                             const rocke_sig_entry_t** out_items,
                                             size_t* out_count);

/* Convenience: build the kernel and lower it to LLVM .ll text. `arch` NULL =>
 * "gfx1151". On ROCKE_OK *out_ll receives a malloc'd NUL-terminated string the
 * caller frees with free(); on failure it is left NULL and (if err!=NULL,
 * capacity err_cap) a diagnostic is written. Owns and frees its IRBuilder. */
rocke_status_t rocke_wmma_fmha_fwd_lower_to_llvm(const rocke_wmma_fmha_fwd_spec_t* spec,
                                                 const char* arch,
                                                 rocke_llvm_flavor_t flavor,
                                                 char** out_ll,
                                                 char* err,
                                                 size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX1151_WMMA_FMHA_FWD_H */
