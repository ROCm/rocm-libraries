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
 *   wmma_fmha_fwd_grid(spec, sq, batch)   rocke_wmma_fmha_fwd_grid(...)
 *   (signature helper, parity)            rocke_wmma_fmha_fwd_signature(...)
 *   (build -> lower .ll convenience)      rocke_wmma_fmha_fwd_lower_to_llvm(...)
 *
 * SPEC AS A FLAT C STRUCT. The Python spec carries only the compile-time tile
 * facts (head_size / heads / mask / v_lds_stage); seqlen_q / seqlen_k are
 * runtime kernel args (the grid is sized from seqlen_q at launch). The dtype is
 * fp16/f16 or bf16. The build routine passes it to the shared inner body so
 * IR emission is byte-identical to the Python path.
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
 *   head_size        : multiple of 16 (WMMA K/N tile); 16|32|64|128|256.
 *   num_query_heads  : Q heads.
 *   num_kv_heads     : 0 => equal to num_query_heads (MHA); else GQA.
 *   mask_mode        : shared FMHA mask enum; WMMA supports NONE / CAUSAL only.
 *   sliding_window   : default 0 (passed through to the inner body).
 *   v_lds_stage      : optional V staging through LDS; default false.
 *   name             : NULL => "rocke_wmma_fmha_fwd".
 *   causal_bottom_right : shift the causal diagonal by Sk-Sq; requires CAUSAL.
 *   query_tail/kv_tail : independent partial-tile specializations; default false.
 *   use_softcap/use_sinks/use_alibi/use_qq_bias : optional runtime score inputs.
 *   layout           : dense, ragged (packed Q/K/V), or paged (packed Q).
 *   page_block_size  : positive power of two for paged; zero otherwise.
 *   kv_dtype         : empty for Q-matched storage, or OCP fp8e4m3 bytes.
 *
 * String fields are referenced as-is; keep them alive with the spec. */
/* ABI: this struct is passed by value layout, so appending fields changes it.
 * The kv_dtype, transposed_qk, block_n, num_waves, scheduler_strategy and
 * value_tile_size, causal_tile_skip, v_head_size, window_right and store_lse fields (and the batch * value_tiles grid z axis) are the
 * "rocke-attention-gfx1151/v3" ABI. Code built against an older header must be
 * recompiled; zero-initialise the struct with rocke_wmma_fmha_fwd_spec_default
 * so new fields take their defaults. */
typedef struct rocke_wmma_fmha_fwd_spec
{
    int head_size;
    int num_query_heads;
    int num_kv_heads; /* 0 => MHA (== num_query_heads)         */
    rocke_fmha_mask_mode_t mask_mode; /* ROCKE_FMHA_MASK_NONE default            */
    bool v_lds_stage; /* default false                         */
    int sliding_window; /* default 0                             */
    const char* name; /* NULL => "rocke_wmma_fmha_fwd"        */
    const char* dtype; /* "fp16" default; "f16" alias or "bf16" */
    bool causal_bottom_right; /* default false; causal diagonal is seqlen_k - seqlen_q */
    bool query_tail; /* default false */
    bool kv_tail; /* default false */
    bool use_softcap; /* runtime softcap scalar when enabled */
    bool use_sinks;
    bool use_alibi;
    bool use_qq_bias;
    const char* layout; /* "dense" default; "ragged" or "paged" use packed Q */
    int page_block_size; /* positive power of two for paged; zero otherwise */
    const char* kv_dtype; /* "" -> Q dtype; "fp8e4m3" -> OCP E4M3FN byte storage */
    bool transposed_qk; /* FP16 D64/D128, aligned dense none/either causal alignment */
    int block_n; /* transposed-QK key tile: 32 or 64; default 32 */
    int num_waves; /* transposed-QK waves per CTA: 1 or 2; default 1 */
    const char*
        scheduler_strategy; /* NULL: backend default; otherwise a validated codegen policy */
    int value_tile_size; /* 0 => full head; proper multiple-of-16 head divisor otherwise */
    bool causal_tile_skip; /* standard path: bound the K loop at the causal diagonal; default false */
    int v_head_size; /* 0 => V/O width equals head_size; else a distinct multiple of 16 */
    int window_right; /* -1 => off; >=0 keeps k <= q + ctx + window_right (mask NONE, standard path) */
    bool store_lse; /* also write the FP32 natural-log softmax LSE per query row */
} rocke_wmma_fmha_fwd_spec_t;

/* Default-constructed spec (Python dataclass defaults). The caller must still
 * set the required head-shape fields. */
rocke_wmma_fmha_fwd_spec_t rocke_wmma_fmha_fwd_spec_default(void);

/* WmmaFmhaFwdSpec.kernel_name(): kernel_name_join(name, "wmma16x16x16",
 * "H{hd}", "HQ{hq}", "HK{kv_heads}", dtype, mask tag,
 * "vlds" if v_lds_stage else "vgather"). The mask tag is "causal_br" for
 * bottom-right alignment. Writes NUL-terminated into out (capacity out_cap).
 * Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small). */
rocke_status_t rocke_wmma_fmha_fwd_kernel_name(const rocke_wmma_fmha_fwd_spec_t* spec,
                                               char* out,
                                               size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). The dtype-matched WMMA atom must
 * exist on `arch` (family "wmma"), the target must be wave32, and GQA requires
 * num_query_heads to be divisible by num_kv_heads (zero denotes MHA). `arch`
 * NULL => "gfx1151". On reject `reason` (if non-NULL, capacity reason_cap)
 * receives the structured message and the function returns false; on accept it
 * returns true and writes "ok". */
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

/* wmma_fmha_fwd_grid(spec, seqlen_q, batch) returns query groups, query heads,
 * and batch * value_tiles. Group width includes transposed-QK waves when used.
 * Dense partial groups require query_tail; output-partition overflow is rejected
 * before modifying out. Packed modes always bound tails; pass maximum query length. */
rocke_status_t rocke_wmma_fmha_fwd_grid(const rocke_wmma_fmha_fwd_spec_t* spec,
                                        int seqlen_q,
                                        int batch,
                                        int out[3]);

/* The specialized kernel ABI signature: Q/K/V/O, scale, lengths and strides,
 * followed by enabled score arguments and layout metadata. Names and type strings
 * are owned by `arena` and remain valid after the internal probe builder is freed.
 * Failure leaves the output pointers untouched. */
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
