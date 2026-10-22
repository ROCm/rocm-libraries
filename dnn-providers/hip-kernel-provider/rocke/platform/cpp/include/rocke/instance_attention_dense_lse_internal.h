/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_attention_dense_lse_internal.h -- C port of the dense-attention
 * IR emitters in library/kernels/common/_lse_store.py, shared by the gfx942 and
 * gfx950 dense attention builders.
 *
 *   Python (_lse_store.py)                      C (this header)
 *   -----------------------------------------   ------------------------------
 *   lse_store(b, dst, off, m, l, row_has_mass)  rocke_attention_dense_lse_store
 *   dense_row_has_key(b, q, kv_len, ...)        rocke_attention_dense_row_has_key
 *   zero_keyless_row(b, packed, has_key, ...)   rocke_attention_dense_zero_keyless_row
 *
 * Not ported: validate_dense_lse (host-side torch tensor checks, no C caller)
 * and the make_*_lse_epilogue hooks (not used by the dense builders).
 *
 * Each emitter reproduces its Python builder-call sequence op for op. A Python
 * ValueError is recorded as a sticky ROCKE_ERR_VALUE on the builder and the call
 * returns NULL. A dead builder is a no-op. Not a public caller-owned ABI.
 */
#ifndef ROCKE_INSTANCE_ATTENTION_DENSE_LSE_INTERNAL_H
#define ROCKE_INSTANCE_ATTENTION_DENSE_LSE_INTERNAL_H

#include <stdbool.h>

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* lse_store(b, dst, element_offset, m_log2, l, *, row_has_mass): one FP32
 * natural-log LSE store, -inf where row_has_mass is false. The caller guards
 * row existence and writer ownership. */
void rocke_attention_dense_lse_store(rocke_ir_builder_t* b,
                                     rocke_value_t* dst,
                                     rocke_value_t* element_offset,
                                     rocke_value_t* m_log2,
                                     rocke_value_t* l,
                                     rocke_value_t* row_has_mass);

/* dense_row_has_key(b, query_row, kv_length, *, causal, diagonal, window,
 * right_bound=0): i1, true iff the row's visible key range [lower, upper] is
 * non-empty. diagonal / window / right_bound are compile-time ints; 0 skips the
 * corresponding op exactly as the Python truthiness tests do. */
rocke_value_t* rocke_attention_dense_row_has_key(rocke_ir_builder_t* b,
                                                 rocke_value_t* query_row,
                                                 rocke_value_t* kv_length,
                                                 bool causal,
                                                 int diagonal,
                                                 int window,
                                                 int right_bound);

/* zero_keyless_row(b, packed, row_has_key, dtype, width=4): zero the packed
 * <width x dtype> O value for rows without keys with an integer AND on its bits.
 * width must be 1, 2 or 4 (else ValueError "unsupported O store width {width}"). */
rocke_value_t* rocke_attention_dense_zero_keyless_row(rocke_ir_builder_t* b,
                                                      rocke_value_t* packed,
                                                      rocke_value_t* row_has_key,
                                                      const rocke_type_t* dtype,
                                                      int width);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ROCKE_INSTANCE_ATTENTION_DENSE_LSE_INTERNAL_H */
