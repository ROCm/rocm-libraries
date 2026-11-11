/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT */
#ifndef ROCKE_HELPER_GLOBAL_READ_H
#define ROCKE_HELPER_GLOBAL_READ_H

#include "rocke/helper_rocke.helpers.tensor_view.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Allocation-free simultaneous input bound; descriptor rank uses TV_MAX_RANK.
 * All accesses must be complete in-bounds scalars/vectors, not masked loads.
 * Views/extents live through emit; plans and their bound cursors live only
 * during the callback. Cursors reference plans, never copy their descriptors. */
#define ROCKE_GLOBAL_READ_MAX_INPUTS 8

typedef struct rocke_global_read_spec
{
    const rocke_tensor_view_t* view;
    const rocke_stride_t* extents; /* actual extents, int64 or signed i32/i64 SSA */
    int rank;
} rocke_global_read_spec_t;

typedef struct rocke_global_read_plan
{
    const rocke_tensor_view_t* view;
    const char* addressing; /* "buffer" or "global64" */
    rocke_tv_buffer_resource_t buffer;
} rocke_global_read_plan_t;

typedef struct rocke_global_read_cursor
{
    const rocke_global_read_plan_t* plan;
    rocke_value_t* base; /* cached byte offset for buffer, pointer for global64 */
} rocke_global_read_cursor_t;

typedef void (*rocke_global_read_body_fn)(rocke_ir_builder_t* b,
                                          const rocke_global_read_plan_t* plans,
                                          int count,
                                          void* user);

/* addressing is "auto" or "global64" (NULL means "auto"). No force-buffer
 * mode. Unknown/nonuniform metadata and unsupported buffer dtypes fall back
 * to one global64 body. Otherwise one uniform if/else surrounds both complete
 * bodies, preserving convergent barriers/publication. Errors are sticky. */
rocke_status_t rocke_emit_global_read_paths(rocke_ir_builder_t* b,
                                            const rocke_global_read_spec_t* specs,
                                            int count,
                                            rocke_global_read_body_fn body_fn,
                                            void* user,
                                            const char* addressing);
rocke_status_t rocke_global_read_bind(rocke_ir_builder_t* b,
                                      rocke_global_read_cursor_t* out_cursor,
                                      const rocke_global_read_plan_t* plan,
                                      rocke_value_t* const* indices,
                                      int rank);
rocke_value_t* rocke_global_read_load_vec(rocke_ir_builder_t* b,
                                          const rocke_global_read_cursor_t* cursor,
                                          rocke_value_t* const* deltas,
                                          int rank,
                                          int n);
rocke_value_t* rocke_global_read_load_scalar(rocke_ir_builder_t* b,
                                             const rocke_global_read_cursor_t* cursor,
                                             rocke_value_t* const* deltas,
                                             int rank);
#ifdef __cplusplus
}
#endif
#endif
