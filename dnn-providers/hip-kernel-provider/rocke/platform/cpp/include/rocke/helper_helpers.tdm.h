/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_helpers.tdm.h -- C99 port of rocke.helpers.tdm.
 *
 * gfx1250 tensor-DMA (TDM) descriptor construction: packs the five `D#`
 * groups consumed by `tensor_load_to_lds` / `tensor_store_from_lds`. Only the
 * rank-1/2, non-gather, non-iterate case is modelled -- groups 2/3 stay zero,
 * because their mode union (dims 2-4 vs gather indices vs iterate config) is
 * not represented here.
 *
 * Byte-faithful translation: the builder-call sequence mirrors the Python in
 * the exact same order, so emitted IR and SSA value numbering stay identical.
 * Changing the order of a `_pack_word` field list, or hoisting a `const_i32`,
 * renumbers values and breaks byte-identity even when the descriptor bits are
 * unchanged.
 *
 * Field-encoding note (the one asymmetry worth knowing): the two strides are
 * NOT encoded alike. `tensor_strides[0]` splits at bit 32 (lo32 + hi16);
 * `tensor_strides[1]` splits at bit 16 (lo16 + hi32). Splitting the second at
 * 32 silently drops bits [16:32] -- harmless only while that stride stays
 * below 2**16, and a wild DMA address above it.
 */
#ifndef ROCKE_HELPER_HELPERS_TDM_H
#define ROCKE_HELPER_HELPERS_TDM_H

#include "rocke/ir.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Groups 2/3 carry dims 2-4 / gather / iterate; only rank<=2 leaves them 0. */
#define ROCKE_TDM_MAX_RANK 2

/* Python `ScalarOrValue = Union[int, Value]`. A compile-time int folds into
 * the packed constant; an SSA value is masked/shifted/or-ed in. */
typedef struct rocke_tdm_operand
{
    bool is_value;
    int64_t k; /* used when is_value == false */
    rocke_value_t* v; /* used when is_value == true  */
} rocke_tdm_operand_t;

/* LDS-side padding applied during a load (not honoured for stores). Both
 * descriptor fields are biased by one; the bias is applied on encode. */
typedef struct rocke_tdm_padding
{
    int interval_dwords; /* power of two in [2, 256] */
    int amount_dwords; /* [1, 128]                 */
} rocke_tdm_padding_t;

typedef struct rocke_tdm_desc_args
{
    rocke_value_t* global_addr; /* i64 byte address of the tensor origin   */
    rocke_value_t* lds_addr; /* i32, or i64 (truncated here)            */
    int rank; /* 1 or 2                                  */
    rocke_tdm_operand_t tensor_dims[ROCKE_TDM_MAX_RANK]; /* fastest-varying first */
    rocke_tdm_operand_t tensor_strides[ROCKE_TDM_MAX_RANK]; /* element pitches   */
    int tile_dims[ROCKE_TDM_MAX_RANK]; /* compile-time, 16 bits each  */
    int elem_bytes; /* 1, 2, 4, or 8                           */
    const rocke_tdm_padding_t* padding; /* NULL => pad_enable 0           */
    int workgroup_mask;
    bool early_timeout;
    bool is_restore;
    bool scalarize; /* lift each word into an SGPR             */
} rocke_tdm_desc_args_t;

/* log2(elem_bytes) as the 2-bit group1.data_size field; -1 if unsupported. */
int rocke_h_tdm_data_size_code(int elem_bytes);

/* Build the five descriptor groups. Writes out[0..4] and returns true on
 * success; on a rejected argument it sets the builder error and returns
 * false, leaving out[] untouched. */
bool rocke_h_tdm_descriptor_groups(rocke_ir_builder_t* b,
                                   const rocke_tdm_desc_args_t* args,
                                   rocke_value_t* out[5]);

/* Row-major 2D convenience wrapper: takes natural (rows, cols) order and
 * performs the fastest-first transposition the descriptor wants, so the
 * dimension order cannot be got backwards. row_pitch is the element distance
 * between consecutive rows. Fields other than the ones named here are taken
 * from `base` (elem_bytes, padding, scalarize, ...). */
bool rocke_h_tdm_row_major_2d(rocke_ir_builder_t* b,
                              const rocke_tdm_desc_args_t* base,
                              rocke_tdm_operand_t rows,
                              rocke_tdm_operand_t cols,
                              rocke_tdm_operand_t row_pitch,
                              int tile_rows,
                              int tile_cols,
                              rocke_value_t* out[5]);

/* Build a descriptor and issue tensor_load_to_lds in one step. When `wait`,
 * follows with s_wait_tensorcnt(0). */
bool rocke_h_tdm_load_to_lds(rocke_ir_builder_t* b,
                             const rocke_tdm_desc_args_t* args,
                             int cachepolicy,
                             bool wait);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_HELPER_HELPERS_TDM_H */
