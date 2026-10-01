/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/tdm.h -- PUBLIC API for the C99 port of rocke.core.tdm (gfx1250 TDM
 * descriptor construction).
 *
 *   Python (rocke.core.tdm)          C99 (this header)
 *   ------------------------------    --------------------------------------
 *   TDM_PAD_INTERVAL_MAX           ->  ROCKE_TDM_PAD_INTERVAL_MAX
 *   TDM_PAD_AMOUNT_MAX             ->  ROCKE_TDM_PAD_AMOUNT_MAX
 *   TDM_MAX_DIM1_STRIDE            ->  ROCKE_TDM_MAX_DIM1_STRIDE
 *   tdm_data_size_code()           ->  rocke_tdm_data_size_code()
 *   encode_tdm_padding()           ->  rocke_tdm_encode_padding()
 *   tdm_padding_for_tile()         ->  rocke_tdm_padding_for_tile()
 *   pack_tdm_group0()              ->  rocke_tdm_pack_group0()
 *   pack_tdm_group1_2d()           ->  rocke_tdm_pack_group1_2d()
 *   build_tdm_descriptor_2d()      ->  rocke_tdm_build_descriptor_2d()
 *
 * Error model: where Python raises ValueError the host-side functions return
 * ROCKE_ERR_VALUE and copy the Python message verbatim into `reason` (NULL /
 * zero-capacity buffers are allowed). rocke_tdm_build_descriptor_2d records the
 * same message on the builder's sticky error instead.
 */
#ifndef ROCKE_TDM_H
#define ROCKE_TDM_H

#include <stddef.h>
#include <stdint.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ROCKE_TDM_PAD_INTERVAL_MAX 7
#define ROCKE_TDM_PAD_AMOUNT_MAX 127
#define ROCKE_TDM_MAX_DIM1_STRIDE 0xFFFF

rocke_status_t
    rocke_tdm_data_size_code(int64_t elem_bytes, int* out_code, char* reason, size_t reason_cap);

rocke_status_t rocke_tdm_encode_padding(int64_t interval_bytes,
                                        int64_t pad_bytes,
                                        int* out_pad_interval,
                                        int* out_pad_amount,
                                        char* reason,
                                        size_t reason_cap);

rocke_status_t rocke_tdm_padding_for_tile(int64_t elem_bytes,
                                          int64_t row_elems,
                                          int64_t pad_elems,
                                          int* out_pad_enable,
                                          int* out_pad_interval,
                                          int* out_pad_amount,
                                          char* reason,
                                          size_t reason_cap);

rocke_status_t rocke_tdm_pack_group0(int64_t lds_addr,
                                     int64_t global_addr,
                                     int64_t gather_index_size,
                                     int64_t gather_mode,
                                     uint32_t out_words[4],
                                     char* reason,
                                     size_t reason_cap);

/* Keyword arguments of pack_tdm_group1_2d. The trailing fields default to 0 in
 * Python; zero-initialise the struct to get the same defaults. */
typedef struct rocke_tdm_group1_2d_args
{
    int64_t elem_bytes;
    int64_t tensor_dim0;
    int64_t tensor_dim1;
    int64_t tile_dim0;
    int64_t tile_dim1;
    int64_t dim0_stride;
    int64_t dim1_stride;
    int64_t pad_enable;
    int64_t pad_interval;
    int64_t pad_amount;
    int64_t workgroup_mask;
    int64_t atomic_barrier_enable;
    int64_t atomic_barrier_address;
} rocke_tdm_group1_2d_args_t;

rocke_status_t rocke_tdm_pack_group1_2d(const rocke_tdm_group1_2d_args_t* args,
                                        uint32_t out_words[8],
                                        char* reason,
                                        size_t reason_cap);

/* Keyword arguments of build_tdm_descriptor_2d. `global_addr` and `lds_addr` are
 * i64 IR values, `tensor_dim0` / `tensor_dim1` i32 IR values. Python accepts the
 * row pitch as either an int or an IR value: set `dim0_stride_value` for the
 * latter, otherwise leave it NULL and `dim0_stride` is folded as a constant. */
typedef struct rocke_tdm_descriptor_2d_args
{
    rocke_value_t* global_addr;
    rocke_value_t* lds_addr;
    int64_t elem_bytes;
    rocke_value_t* tensor_dim0;
    rocke_value_t* tensor_dim1;
    int64_t tile_dim0;
    int64_t tile_dim1;
    rocke_value_t* dim0_stride_value;
    int64_t dim0_stride;
    int64_t dim1_stride;
    int64_t pad_enable;
    int64_t pad_interval;
    int64_t pad_amount;
} rocke_tdm_descriptor_2d_args_t;

/* Emit the five descriptor groups (d0..d4) for a rank-2 non-gather TDM load,
 * ready for rocke_b_tensor_load_to_lds. The returned values are builder-owned.
 * Returns the builder status; on failure the out slots are NULL. */
rocke_status_t rocke_tdm_build_descriptor_2d(rocke_ir_builder_t* b,
                                             const rocke_tdm_descriptor_2d_args_t* args,
                                             rocke_value_t* out_groups[5]);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_TDM_H */
