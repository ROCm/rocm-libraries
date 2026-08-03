/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_gfx942_fp8_mqa_logits.h -- C99 port of
 * rocke/instances/gfx942/fp8_mqa_logits.py.
 *
 * Python (fp8_mqa_logits.py)          C99 (this header)
 * ----------------------------------  -----------------------------------------
 * class Fp8MqaLogitsSpec              rocke_fp8_mqa_logits_spec_t
 * Fp8MqaLogitsSpec.kernel_name()      rocke_fp8_mqa_logits_kernel_name(...)
 * is_valid_spec(spec, arch)           rocke_fp8_mqa_logits_is_valid_spec(...)
 * build_fp8_mqa_logits(spec, arch)    rocke_build_fp8_mqa_logits(...)
 * fp8_mqa_logits_num_splits(...)      rocke_fp8_mqa_logits_num_splits(...)
 * fp8_mqa_logits_grid(...)            rocke_fp8_mqa_logits_grid(...)
 * fp8_mqa_logits_signature(spec)      rocke_fp8_mqa_logits_signature(...)
 */
#ifndef ROCKE_INSTANCE_GFX942_FP8_MQA_LOGITS_H
#define ROCKE_INSTANCE_GFX942_FP8_MQA_LOGITS_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rocke_fp8_mqa_logits_spec
{
    int num_heads;
    int head_dim;
    int block_kv;
    int rows_per_block;
    int waves_per_block;
    bool has_waves_per_eu;
    int waves_per_eu;
    const char* name;
} rocke_fp8_mqa_logits_spec_t;

rocke_fp8_mqa_logits_spec_t rocke_fp8_mqa_logits_spec_default(void);

int rocke_fp8_mqa_logits_block_size(const rocke_fp8_mqa_logits_spec_t* spec);

rocke_status_t rocke_fp8_mqa_logits_kernel_name(const rocke_fp8_mqa_logits_spec_t* spec,
                                                char* out,
                                                size_t out_cap);

bool rocke_fp8_mqa_logits_is_valid_spec(const rocke_fp8_mqa_logits_spec_t* spec,
                                        const char* arch,
                                        char* reason,
                                        size_t reason_cap);

rocke_kernel_def_t* rocke_build_fp8_mqa_logits(rocke_ir_builder_t* b,
                                               const rocke_fp8_mqa_logits_spec_t* spec,
                                               const char* arch);

rocke_kernel_def_t* rocke_build_fp8_mqa_logits_new(rocke_ir_builder_t* b,
                                                   const rocke_fp8_mqa_logits_spec_t* spec,
                                                   const char* arch);

rocke_status_t rocke_fp8_mqa_logits_lower_to_llvm(const rocke_fp8_mqa_logits_spec_t* spec,
                                                  const char* arch,
                                                  rocke_llvm_flavor_t flavor,
                                                  char** out_ll,
                                                  char* err,
                                                  size_t err_cap);

int rocke_fp8_mqa_logits_num_splits(
    int seq_len_padded, int seq_len_kv, int rows_per_block, int block_kv, int num_cus);

rocke_status_t rocke_fp8_mqa_logits_grid(int seq_len_padded,
                                         int num_splits,
                                         const rocke_fp8_mqa_logits_spec_t* spec,
                                         int out[3]);

rocke_status_t rocke_fp8_mqa_logits_signature(rocke_arena_t* arena,
                                              const rocke_fp8_mqa_logits_spec_t* spec,
                                              const rocke_sig_entry_t** out_items,
                                              size_t* out_count);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_GFX942_FP8_MQA_LOGITS_H */
