/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1151_wmma_fmha_fwd.h -- C99 port of the gfx1151 (RDNA3.5 /
 * Strix Halo) WMMA FMHA forward kernel instance builder
 * rocke/instances/gfx1151/wmma_fmha_fwd.py.
 *
 * QK^T and PV use dtype-matched gfx11 FP16/BF16 WMMA instructions with
 * a wave32 thread mapping. This adapter owns the gfx1151 kernel ABI, the
 * (seqlen_q // 16, num_query_heads, batch) grid decode, and the per-batch
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
    const char* scheduler_strategy; /* NULL: backend default; otherwise a validated codegen policy */
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

/* build_wmma_fmha_fwd(spec, arch). Validates, then builds the gfx1151 WMMA FMHA
 * forward IR (one wave per CTA) and returns the kernel def. `arch` NULL =>
 * "gfx1151". Grid: (seqlen_q // 16, num_query_heads, batch). On an invalid spec
 * or any IR-emission error returns NULL.
 *
 * `b` is the destination IR builder to emit into; if NULL this instance owns a
 * transient builder (see the .c note / sibling instance_fmha_mfma entry). For
 * parity with the documented CALL PATTERN prefer the lower-to-llvm convenience
 * when the kernel must outlive the call. */
rocke_kernel_def_t* rocke_build_wmma_fmha_fwd(rocke_ir_builder_t* b,
                                              const rocke_wmma_fmha_fwd_spec_t* spec,
                                              const char* arch);

/* wmma_fmha_fwd_grid(spec, seqlen_q, batch) ->
 * (ceil(seqlen_q / BLOCK_M), num_query_heads, batch). Writes three axes to out.
 * Dense mode without query_tail rejects a non-multiple of BLOCK_M without
 * modifying out. Packed modes always bound tails; pass the maximum query length. */
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
