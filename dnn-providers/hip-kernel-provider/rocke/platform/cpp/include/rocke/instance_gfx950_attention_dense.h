/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * Host-only mirror of library/kernels/gfx950/attention_dense.py. No IR builders
 * live here. The shared spec, validation, decode policy and name join belong to
 * instance_attention_dense_spec.h; this layer supplies the gfx950 overrides.
 *
 * Values use the Python annotations' types: int32 integers, bools, double for
 * lazy_rescale_threshold. Optional values retain their presence, not just their
 * resolved meaning, because explicit defaults affect kernel names. Arbitrarily
 * typed Python inputs (e.g. an int in a float field) are not C representations.
 * Strings are caller-owned and use rocke_py_str_repr's existing Unicode limits.
 *
 * validate mirrors __post_init__; supports additionally runs shared preflight,
 * tile/LDS and KV-segment admission. Reasons follow the shared header's contract.
 * Property/name/launch accessors require a supported spec. runtime_args has the
 * additional fast-division dividend check, just as Python does; it is not folded
 * into supports. All output arrays are caller-owned; signature storage is owned
 * by the supplied arena. NULL/output errors return ROCKE_ERR_VALUE (or false).
 */
#ifndef ROCKE_INSTANCE_GFX950_ATTENTION_DENSE_H
#define ROCKE_INSTANCE_GFX950_ATTENTION_DENSE_H

#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/instance_attention_dense_spec.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_GFX950_DENSE_LDS_CAPACITY_BYTES 163840
#define ROCKE_GFX950_DENSE_MAX_TILES 7
#define ROCKE_GFX950_DENSE_MAX_RUNTIME_FIELDS 11

typedef struct rocke_gfx950_attention_dense_spec
{
    rocke_attention_dense_spec_t base;
    int lds_v_row_pad; /* 32 */
    bool wide_lds_dma; /* false */
    bool emit_lse; /* false */
    int right_bound; /* 0 */
    int kv_segments; /* 1 */
    int lds_num_buffers; /* 2 */
    int lds_k_row_pad; /* 8 */
    double lazy_rescale_threshold; /* 8.0 */
    bool use_exp2_fast; /* true */
    bool has_exp_per_pv_step;
    int exp_per_pv_step;
    bool partial_vmcnt_prefetch; /* true */
    int pv_priority; /* 1 */
    bool has_pv_sched_fence;
    bool pv_sched_fence;
    int pv_sched_fence_mask; /* 0 */
    bool has_pv_sched_group_template;
    bool pv_sched_group_template;
    int pv_sched_group_ds_read; /* 2 */
    bool has_iglp_mode;
    int iglp_mode;
    const char* pv_loop_order; /* NULL = None */
    bool causal_diag_split; /* true */
    int o_store_width; /* 4 */
} rocke_gfx950_attention_dense_spec_t;

rocke_gfx950_attention_dense_spec_t rocke_gfx950_attention_dense_spec_default(void);
rocke_attention_dense_traits_t rocke_gfx950_attention_dense_traits(void);
bool rocke_gfx950_attention_dense_spec_validate(const rocke_gfx950_attention_dense_spec_t* spec,
                                                char* reason,
                                                size_t reason_cap);
bool rocke_gfx950_supports_attention_dense(const rocke_gfx950_attention_dense_spec_t* spec,
                                           const char* arch,
                                           char* reason,
                                           size_t reason_cap);

int rocke_gfx950_attention_dense_resolved_exp_per_pv_step(
    const rocke_gfx950_attention_dense_spec_t* spec);
bool rocke_gfx950_attention_dense_resolved_pv_sched_fence(
    const rocke_gfx950_attention_dense_spec_t* spec);
bool rocke_gfx950_attention_dense_resolved_pv_sched_group_template(
    const rocke_gfx950_attention_dense_spec_t* spec);
int rocke_gfx950_attention_dense_resolved_iglp_mode(
    const rocke_gfx950_attention_dense_spec_t* spec);
const char* rocke_gfx950_attention_dense_resolved_pv_loop_order(
    const rocke_gfx950_attention_dense_spec_t* spec);
bool rocke_gfx950_attention_dense_runtime_shape(const rocke_gfx950_attention_dense_spec_t* spec);

/* Names refer to static strings. runtime_param_fields are the five fields
 * excluded from binary identity; shape_kernarg_fields is the complete scalar
 * ABI after scale, including fastdiv pairs (or the three baked-grid scalars). */
typedef struct rocke_gfx950_dense_fields
{
    const char* names[ROCKE_GFX950_DENSE_MAX_RUNTIME_FIELDS];
    size_t count;
} rocke_gfx950_dense_fields_t;
rocke_status_t rocke_gfx950_attention_dense_runtime_param_fields(
    const rocke_gfx950_attention_dense_spec_t* spec, rocke_gfx950_dense_fields_t* out);
rocke_status_t rocke_gfx950_attention_dense_shape_kernarg_fields(
    const rocke_gfx950_attention_dense_spec_t* spec, rocke_gfx950_dense_fields_t* out);
rocke_status_t
    rocke_gfx950_attention_dense_name_parts(const rocke_gfx950_attention_dense_spec_t* spec,
                                            rocke_attention_dense_name_parts_t* out);
rocke_status_t rocke_gfx950_attention_dense_kernel_name(
    const rocke_gfx950_attention_dense_spec_t* spec, char* out, size_t out_cap);

/* The pure formula accepts the full pad/wide layout parameters. Unknown head
 * sizes produce zero; arithmetic outside int64 returns ROCKE_ERR_VALUE, never
 * wraps. Supported tiles always fit int64, including their LDS refusal value. */
rocke_status_t rocke_gfx950_dense_static_lds_bytes(int head_size,
                                                   int block_n,
                                                   int k_row_pad,
                                                   int v_row_pad,
                                                   int k_group_pad,
                                                   bool wide_lds_dma,
                                                   int64_t* out);
/* Ordered (block_m, block_n) pairs at default padding. out may be NULL to count. */
size_t rocke_gfx950_dense_tile_legality(int head_size, int out[ROCKE_GFX950_DENSE_MAX_TILES][2]);
rocke_status_t rocke_gfx950_attention_dense_grid(const rocke_gfx950_attention_dense_spec_t* spec,
                                                 int out[3]);
rocke_status_t rocke_gfx950_attention_dense_block(const rocke_gfx950_attention_dense_spec_t* spec,
                                                  int out[3]);
rocke_status_t
    rocke_gfx950_attention_dense_signature(rocke_arena_t* arena,
                                           const rocke_gfx950_attention_dense_spec_t* spec,
                                           const rocke_sig_entry_t** out_items,
                                           size_t* out_count);

typedef struct rocke_gfx950_dense_runtime_arg
{
    const char* name;
    int32_t value;
} rocke_gfx950_dense_runtime_arg_t;
typedef struct rocke_gfx950_dense_runtime_args
{
    rocke_gfx950_dense_runtime_arg_t items[ROCKE_GFX950_DENSE_MAX_RUNTIME_FIELDS];
    size_t count;
} rocke_gfx950_dense_runtime_args_t;
bool rocke_gfx950_attention_dense_runtime_args(const rocke_gfx950_attention_dense_spec_t* spec,
                                               rocke_gfx950_dense_runtime_args_t* out,
                                               char* reason,
                                               size_t reason_cap);

#ifdef __cplusplus
} /* extern "C" */
#endif
#endif /* ROCKE_INSTANCE_GFX950_ATTENTION_DENSE_H */
