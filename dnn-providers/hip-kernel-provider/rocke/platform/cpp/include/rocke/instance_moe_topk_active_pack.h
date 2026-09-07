/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_moe_topk_active_pack.h -- C99 port of the fused top-k and
 * active-expert packing builder in
 * rocke/instances/common/moe_topk_active_pack.py.
 *
 * Python (moe_topk_active_pack.py)       C99 (this header)
 * ------------------------------------   -------------------------------------
 * MoeTopkActivePackSpec                  rocke_moe_topk_active_pack_spec_t
 * is_valid_spec                          rocke_moe_topk_active_pack_is_valid_spec
 * build_moe_topk_active_pack             rocke_build_moe_topk_active_pack
 * moe_topk_active_pack_grid              rocke_moe_topk_active_pack_grid
 * moe_topk_active_pack_signature         rocke_moe_topk_active_pack_signature
 */
#ifndef ROCKE_INSTANCE_MOE_TOPK_ACTIVE_PACK_H
#define ROCKE_INSTANCE_MOE_TOPK_ACTIVE_PACK_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rocke_moe_topk_active_pack_spec
{
    int tokens;
    int experts;
    int topk;
    int tile_m;
    int block_size;
    int num_expert_groups;
    int topk_groups;
    bool renormalize;
    const char* name;
} rocke_moe_topk_active_pack_spec_t;

/* Dataclass defaults with the required problem dimensions zeroed. */
rocke_moe_topk_active_pack_spec_t rocke_moe_topk_active_pack_spec_default(void);

int rocke_moe_topk_active_pack_total_pairs(const rocke_moe_topk_active_pack_spec_t* spec);
int rocke_moe_topk_active_pack_max_blocks(const rocke_moe_topk_active_pack_spec_t* spec);
int rocke_moe_topk_active_pack_max_padded_pairs(const rocke_moe_topk_active_pack_spec_t* spec);
int rocke_moe_topk_active_pack_max_blocks_per_expert(const rocke_moe_topk_active_pack_spec_t* spec);

rocke_status_t rocke_moe_topk_active_pack_kernel_name(const rocke_moe_topk_active_pack_spec_t* spec,
                                                      char* out,
                                                      size_t out_cap);

bool rocke_moe_topk_active_pack_is_valid_spec(const rocke_moe_topk_active_pack_spec_t* spec,
                                              const char* arch,
                                              char* reason,
                                              size_t reason_cap);

/* Build into an initialized builder whose kernel name matches the spec. */
rocke_kernel_def_t* rocke_build_moe_topk_active_pack(rocke_ir_builder_t* b,
                                                     const rocke_moe_topk_active_pack_spec_t* spec,
                                                     const char* arch);

/* Initialize the builder with spec.kernel_name(), then build. */
rocke_kernel_def_t* rocke_build_moe_topk_active_pack_new(
    rocke_ir_builder_t* b, const rocke_moe_topk_active_pack_spec_t* spec, const char* arch);

rocke_status_t rocke_moe_topk_active_pack_grid(const rocke_moe_topk_active_pack_spec_t* spec,
                                               int out[3]);

rocke_status_t rocke_moe_topk_active_pack_signature(rocke_arena_t* arena,
                                                    const rocke_moe_topk_active_pack_spec_t* spec,
                                                    const rocke_sig_entry_t** out_items,
                                                    size_t* out_count);

rocke_status_t
    rocke_moe_topk_active_pack_lower_to_llvm(const rocke_moe_topk_active_pack_spec_t* spec,
                                             const char* arch,
                                             rocke_llvm_flavor_t flavor,
                                             char** out_ll,
                                             char* err,
                                             size_t err_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_MOE_TOPK_ACTIVE_PACK_H */
