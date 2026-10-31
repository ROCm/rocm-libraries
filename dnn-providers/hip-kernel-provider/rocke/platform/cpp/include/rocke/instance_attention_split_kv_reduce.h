// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Weighted segment sum of the split-KV reduce (its pass 3), shared by the
 * gfx942 and gfx950 tiled-3D reduce kernels. C++ twin of
 * kernels/common/_split_kv_reduce.py::reduce_weighted_segments; the
 * builder-call order is identical.
 */
#ifndef ROCKE_INSTANCE_ATTENTION_SPLIT_KV_REDUCE_H
#define ROCKE_INSTANCE_ATTENTION_SPLIT_KV_REDUCE_H

#include <stdbool.h>

#include "rocke/helper_rocke.helpers.transforms.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rocke_split_kv_reduce_args
{
    int head_size;
    int num_segments;
    int threads;
    const rocke_type_t* dtype;
    rocke_value_t* tid;
    rocke_value_t* q_token;
    rocke_value_t* q_head;
    rocke_value_t* seg_out;
    rocke_value_t* out;
    rocke_value_t* factor_lds;
    const rocke_tensor_descriptor_t* seg_acc_desc; /* (token, head, seg, dim) */
    const rocke_tensor_descriptor_t* out_desc; /* (token, head, dim)      */
    rocke_value_t* inv_l;
    rocke_value_t* zero_f;
} rocke_split_kv_reduce_args_t;

/* Occupancy hint for the reduce kernel (Python reduce_waves_per_eu): the
 * spec's value when it has one, else 1 for num_segments >= 32, else 0 (no
 * hint, backend default). */
int rocke_split_kv_reduce_waves_per_eu(bool has_requested, int requested, int num_segments);

/* Emit the weighted segment sum, normalize, and store one output row. Sets a
 * builder error (ROCKE_ERR_VALUE) when head_size / 4 lanes do not fit in
 * `threads`. */
void rocke_split_kv_reduce_weighted_segments(rocke_ir_builder_t* b,
                                             const rocke_split_kv_reduce_args_t* a);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_ATTENTION_SPLIT_KV_REDUCE_H */
