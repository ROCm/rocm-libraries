/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx1151_wmma_fmha_fwd.h -- C99 port of the gfx1151 (RDNA3.5 /
 * Strix Halo) WMMA FMHA forward kernel instance builder
 * rocke/instances/gfx1151/wmma_fmha_fwd.py.
 *
 * RDNA has no MFMA, so the QK^T and PV matmuls are built around the gfx11
 * wmma_f32_16x16x16_f16 instruction with a wave32 thread mapping. This module
 * is a thin ADAPTER: it owns the gfx1151 kernel ABI, the
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
 * runtime kernel args (the grid is sized from seqlen_q at launch). dtype is fp16
 * only. The build routine reconstitutes the equivalent inner-body params so the
 * helper-driven IR emission is byte-identical to the Python path.
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

/* WMMA atom id + tile constants (Python module-level _WMMA_OP_ID / _BLOCK_*). */
#define ROCKE_WMMA_FMHA_FWD_OP_ID "wmma_f32_16x16x16_f16"
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
 *   v_lds_stage      : opt lever; default false (the measured winner). Staging
 *                      the K-tile's V rows through LDS for the PV B-operand cuts
 *                      global loads ~3.3x but is a 1.5-1.8x regression on
 *                      gfx1151; kept togglable for the A/B study.
 *   name             : NULL => "rocke_wmma_fmha_fwd".
 *
 * dtype is fp16 only (the Python __post_init__ rejects anything else). */
typedef struct rocke_wmma_fmha_fwd_spec
{
    int head_size;
    int num_query_heads;
    int num_kv_heads; /* 0 => MHA (== num_query_heads)         */
    rocke_fmha_mask_mode_t mask_mode; /* ROCKE_FMHA_MASK_NONE default            */
    bool v_lds_stage; /* default false                         */
    int sliding_window; /* default 0                             */
    const char* name; /* NULL => "rocke_wmma_fmha_fwd"        */
} rocke_wmma_fmha_fwd_spec_t;

/* Default-constructed spec (Python dataclass defaults). The caller must still
 * set the required head-shape fields. */
rocke_wmma_fmha_fwd_spec_t rocke_wmma_fmha_fwd_spec_default(void);

/* WmmaFmhaFwdSpec.kernel_name(): kernel_name_join(name, "wmma16x16x16",
 * "H{hd}", "HQ{hq}", "HK{kv_heads}", "fp16", mask_mode,
 * "vlds" if v_lds_stage else "vgather"). Writes NUL-terminated into out
 * (capacity out_cap). Returns ROCKE_OK or ROCKE_ERR_VALUE (buffer too small). */
rocke_status_t rocke_wmma_fmha_fwd_kernel_name(const rocke_wmma_fmha_fwd_spec_t* spec,
                                               char* out,
                                               size_t out_cap);

/* is_valid_spec(spec, arch) -> (ok, reason). The WMMA 16x16x16 f16 atom must
 * exist on `arch` (family "wmma") and the target must be wave32 (WMMA is an
 * RDNA/gfx11 instruction; gfx1151-only). `arch` NULL => "gfx1151". On reject
 * `reason` (if non-NULL, capacity reason_cap) receives the structured message
 * and the function returns false; on accept it returns true and writes "ok". */
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
 * (seqlen_q / BLOCK_M, num_query_heads, batch). Writes the three axes to
 * out[0..2]; `out` must hold 3 ints. seqlen_q must be a multiple of BLOCK_M
 * (Python raises otherwise); on a non-multiple this leaves out untouched and
 * returns ROCKE_ERR_VALUE, else ROCKE_OK. */
rocke_status_t rocke_wmma_fmha_fwd_grid(const rocke_wmma_fmha_fwd_spec_t* spec,
                                        int seqlen_q,
                                        int batch,
                                        int out[3]);

/* The kernel ABI signature (Q/K/V/O ptrs, scale_log2/seqlen_q/seqlen_k
 * scalars, q/k/v/o stride pairs). On ROCKE_OK *out_items / *out_count hold the
 * arena-owned array; `arena` backs the storage. On failure the out-params are
 * untouched and the status is returned. */
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
