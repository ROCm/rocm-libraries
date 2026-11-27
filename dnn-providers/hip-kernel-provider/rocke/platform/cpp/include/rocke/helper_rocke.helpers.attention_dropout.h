/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.attention_dropout.h -- C port of
 * rocke.helpers.attention_dropout: the SDPA dropout mask of SdpaSemanticsRegister SEM-015,
 * drawn in IR, bit-exact against the host oracle
 * rocke.numeric.philox_dropout. Each helper emits the Python builder calls in
 * the same order, one builder call per statement, so both engines emit the
 * same IR.
 *
 * Symbol map:
 *   dropout_row_counter -> rocke_b_dropout_row_counter (out-params lo, hi)
 *   dropout_words       -> rocke_b_dropout_words (out[4])
 *   dropout_keep        -> rocke_b_dropout_keep
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_DROPOUT_H
#define ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_DROPOUT_H

#include <stdbool.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Counter words of key 0 of a row: offset + row * ceil(max_sk / 4) mod 2**64,
 * row an i64 value (the product in i64). */
void rocke_b_dropout_row_counter(rocke_ir_builder_t* b,
                                 rocke_value_t* offset_lo,
                                 rocke_value_t* offset_hi,
                                 rocke_value_t* row,
                                 rocke_value_t* max_sk,
                                 rocke_value_t** lo,
                                 rocke_value_t** hi);

/* The four SEM-015 words of counter (ctr_lo, ctr_hi): one PhiloxCall. */
void rocke_b_dropout_words(rocke_ir_builder_t* b,
                           rocke_value_t* seed_lo,
                           rocke_value_t* seed_hi,
                           rocke_value_t* ctr_lo,
                           rocke_value_t* ctr_hi,
                           bool xor3,
                           bool mad64,
                           rocke_value_t* out[4]);

/* i1 keep flag of one SEM-015 word: int32(word) > threshold. */
rocke_value_t*
    rocke_b_dropout_keep(rocke_ir_builder_t* b, rocke_value_t* word, rocke_value_t* threshold);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_ATTENTION_DROPOUT_H */
