/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * C engine mirror of rocke/instances/common/moe_rank_reduce.py.
 */
#ifndef ROCKE_INSTANCE_MOE_RANK_REDUCE_H
#define ROCKE_INSTANCE_MOE_RANK_REDUCE_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rocke_moe_rank_reduce_rmsnorm_spec
{
    int width;
    int world_size;
    const char* dtype;
    int block_size;
    int vec;
    int wave_size;
    bool fp32_internal;
    const char* name;
} rocke_moe_rank_reduce_rmsnorm_spec_t;

typedef struct rocke_moe_rank_reduce_scatter_spec
{
    int width;
    int world_size;
    const char* dtype;
    int block_size;
    int vec;
    const char* name;
} rocke_moe_rank_reduce_scatter_spec_t;

rocke_moe_rank_reduce_rmsnorm_spec_t rocke_moe_rank_reduce_rmsnorm_spec_default(void);
rocke_moe_rank_reduce_scatter_spec_t rocke_moe_rank_reduce_scatter_spec_default(void);

rocke_status_t rocke_moe_rank_reduce_rmsnorm_kernel_name(
    const rocke_moe_rank_reduce_rmsnorm_spec_t* spec, char* out, size_t out_cap);
rocke_status_t rocke_moe_rank_reduce_scatter_kernel_name(
    const rocke_moe_rank_reduce_scatter_spec_t* spec, char* out, size_t out_cap);

bool rocke_is_valid_moe_rank_reduce_rmsnorm_spec(const rocke_moe_rank_reduce_rmsnorm_spec_t* spec,
                                                 const char* arch,
                                                 char* reason,
                                                 size_t reason_cap);
bool rocke_is_valid_moe_rank_reduce_scatter_spec(const rocke_moe_rank_reduce_scatter_spec_t* spec,
                                                 const char* arch,
                                                 char* reason,
                                                 size_t reason_cap);

rocke_kernel_def_t* rocke_build_moe_rank_reduce_rmsnorm(
    rocke_ir_builder_t* b, const rocke_moe_rank_reduce_rmsnorm_spec_t* spec, const char* arch);
rocke_kernel_def_t* rocke_build_moe_rank_reduce_scatter(
    rocke_ir_builder_t* b, const rocke_moe_rank_reduce_scatter_spec_t* spec, const char* arch);
rocke_kernel_def_t* rocke_build_moe_rank_reduce_rmsnorm_new(
    rocke_ir_builder_t* b, const rocke_moe_rank_reduce_rmsnorm_spec_t* spec, const char* arch);
rocke_kernel_def_t* rocke_build_moe_rank_reduce_scatter_new(
    rocke_ir_builder_t* b, const rocke_moe_rank_reduce_scatter_spec_t* spec, const char* arch);

rocke_status_t rocke_moe_rank_reduce_rmsnorm_grid(int rows,
                                                  const rocke_moe_rank_reduce_rmsnorm_spec_t* spec,
                                                  int out[3]);
rocke_status_t rocke_moe_rank_reduce_scatter_grid(int rows,
                                                  const rocke_moe_rank_reduce_scatter_spec_t* spec,
                                                  int out[3]);

rocke_status_t
    rocke_moe_rank_reduce_rmsnorm_signature(rocke_arena_t* arena,
                                            const rocke_moe_rank_reduce_rmsnorm_spec_t* spec,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count);
rocke_status_t
    rocke_moe_rank_reduce_scatter_signature(rocke_arena_t* arena,
                                            const rocke_moe_rank_reduce_scatter_spec_t* spec,
                                            const rocke_sig_entry_t** out_items,
                                            size_t* out_count);

rocke_status_t
    rocke_moe_rank_reduce_rmsnorm_lower_to_llvm(const rocke_moe_rank_reduce_rmsnorm_spec_t* spec,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap);
rocke_status_t
    rocke_moe_rank_reduce_scatter_lower_to_llvm(const rocke_moe_rank_reduce_scatter_spec_t* spec,
                                                const char* arch,
                                                rocke_llvm_flavor_t flavor,
                                                char** out_ll,
                                                char* err,
                                                size_t err_cap);

#ifdef __cplusplus
}
#endif

#endif
