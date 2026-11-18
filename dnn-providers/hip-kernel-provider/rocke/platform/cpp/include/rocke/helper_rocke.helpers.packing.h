// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Encoded-pattern packing; mirrors helpers/packing.py.
 * pack_fragment_bits -> rocke_h_pack_fragment_bits
 * The callback and output array are borrowed for the duration of the call.
 */
#ifndef ROCKE_HELPER_PACKING_H
#define ROCKE_HELPER_PACKING_H

#include "rocke/ir.h"
#include "rocke/storage.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef rocke_value_t* (*rocke_load_bits_fn)(rocke_ir_builder_t* b, int index, void* ctx);

/* Callback returns a canonical unsigned pattern in an integer carrier.
 * Encoded fields are at most 32 bits; i64 carriers support eight-byte scales. */
rocke_status_t rocke_h_pack_fragment_bits(rocke_ir_builder_t* b,
                                          rocke_load_bits_fn load_bits,
                                          void* ctx,
                                          const rocke_fragment_packing_t* fragment,
                                          rocke_value_t** words,
                                          size_t word_count);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_PACKING_H */
