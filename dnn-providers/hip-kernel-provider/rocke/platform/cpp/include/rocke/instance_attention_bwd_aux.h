/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_attention_bwd_aux.h
 *
 * C port of library/kernels/common/attention_bwd_aux.py: the shared prep /
 * convert / reduce / page_zero / page_convert kernels of the attention
 * backward (all families).
 *
 *   Python (attention_bwd_aux.py)        C (this header)
 *   ----------------------------------   --------------------------------------
 *   AttnBwdAuxSpec (+ __post_init__)     rocke_attn_bwd_aux_spec_t
 *                                        (+ rocke_attn_bwd_aux_spec_validate)
 *   AttnBwdAuxSpec.kernel_name(stage)    rocke_attn_bwd_aux_kernel_name
 *   attn_bwd_aux_rows_per_cta            rocke_attn_bwd_aux_rows_per_cta
 *   attn_bwd_aux_block_threads           rocke_attn_bwd_aux_block_threads
 *   build_attn_bwd_prep                  rocke_build_attn_bwd_prep_new
 *   build_attn_bwd_convert               rocke_build_attn_bwd_convert_new
 *   build_attn_bwd_reduce                rocke_build_attn_bwd_reduce_new
 *   build_attn_bwd_page_zero             rocke_build_attn_bwd_page_zero_new
 *   build_attn_bwd_page_convert          rocke_build_attn_bwd_page_convert_new
 *
 * The kernels are byte-identical (lowered .ll) to the Python builders; the
 * parity pair is library/tests/parity/attention_bwd_aux_emit.{py,c}.
 *
 * Host-only parts stay in Python: the launch grids (attn_bwd_prep_grid,
 * attn_bwd_convert_grid, attn_bwd_page_grid), the ABI names and the public
 * attn_bwd_aux_params list (the builders here declare the same parameters).
 */
#ifndef ROCKE_INSTANCE_ATTENTION_BWD_AUX_H
#define ROCKE_INSTANCE_ATTENTION_BWD_AUX_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Mirror of Python AttnBwdAuxSpec (frozen dataclass). Strings carry the
 * Python spellings; has_rows_per_cta false is Python's rows_per_cta=None
 * (then rows_per_cta is ignored), so an explicit 0 is rejected like Python. */
typedef struct rocke_attn_bwd_aux_spec
{
    int head_size; /* 32, 64 or 128 (no default)        */
    const char* dtype; /* "fp16" | "bf16"                   */
    const char* seq_mode; /* "batched" | "thd"                 */
    int stage_vec; /* 8 | 1                             */
    const char* ws_layout; /* "head_major" | "token_major"      */
    const char* kv_target; /* "contiguous" | "paged"            */
    bool has_rows_per_cta; /* false => Python None              */
    int rows_per_cta; /* read only when has_rows_per_cta   */
    int convert_vec; /* 8 | 4 | 16                        */
    bool worklist; /* prep (THD, contiguous) only       */
    int narrow_vec; /* 1 | 2 | 4 (not 1: stage_vec 1)    */
    const char* name; /* "rocke_attn_bwd"                  */
} rocke_attn_bwd_aux_spec_t;

/* Python's defaults; the caller sets head_size. */
rocke_attn_bwd_aux_spec_t rocke_attn_bwd_aux_spec_default(void);

/* AttnBwdAuxSpec.__post_init__: true when valid, else false + reason. */
bool rocke_attn_bwd_aux_spec_validate(const rocke_attn_bwd_aux_spec_t* spec,
                                      char* reason,
                                      size_t reason_cap);

/* AttnBwdAuxSpec.kernel_name(stage); stage in AUX_STAGES ("prep", "convert",
 * "page_zero", "page_convert", "reduce"). */
rocke_status_t rocke_attn_bwd_aux_kernel_name(const rocke_attn_bwd_aux_spec_t* spec,
                                              const char* stage,
                                              char* out,
                                              size_t out_cap);

/* attn_bwd_aux_rows_per_cta / attn_bwd_aux_block_threads: the value, or -1
 * with a reason (unknown arch, rows that do not fill whole waves or exceed
 * 1024 threads). */
int rocke_attn_bwd_aux_rows_per_cta(const rocke_attn_bwd_aux_spec_t* spec,
                                    const char* arch,
                                    char* reason,
                                    size_t reason_cap);
int rocke_attn_bwd_aux_block_threads(const rocke_attn_bwd_aux_spec_t* spec,
                                     const char* arch,
                                     char* reason,
                                     size_t reason_cap);

/* Init `b` (kernel name from the spec) and build one kernel. Return the
 * kernel, or NULL with the builder's error set (the Python ValueError). The
 * caller frees `b` with rocke_ir_builder_free either way. */
rocke_kernel_def_t* rocke_build_attn_bwd_prep_new(rocke_ir_builder_t* b,
                                                  const rocke_attn_bwd_aux_spec_t* spec,
                                                  const char* arch);
rocke_kernel_def_t* rocke_build_attn_bwd_convert_new(rocke_ir_builder_t* b,
                                                     const rocke_attn_bwd_aux_spec_t* spec,
                                                     const char* arch);
rocke_kernel_def_t* rocke_build_attn_bwd_reduce_new(rocke_ir_builder_t* b,
                                                    const rocke_attn_bwd_aux_spec_t* spec,
                                                    const char* arch);
rocke_kernel_def_t* rocke_build_attn_bwd_page_zero_new(rocke_ir_builder_t* b,
                                                       const rocke_attn_bwd_aux_spec_t* spec,
                                                       const char* arch);
rocke_kernel_def_t* rocke_build_attn_bwd_page_convert_new(rocke_ir_builder_t* b,
                                                          const rocke_attn_bwd_aux_spec_t* spec,
                                                          const char* arch);

/* Dispatch on `stage` (an AUX_STAGES name) to the builders above. */
rocke_kernel_def_t* rocke_build_attn_bwd_aux_new(rocke_ir_builder_t* b,
                                                 const char* stage,
                                                 const rocke_attn_bwd_aux_spec_t* spec,
                                                 const char* arch);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_INSTANCE_ATTENTION_BWD_AUX_H */
