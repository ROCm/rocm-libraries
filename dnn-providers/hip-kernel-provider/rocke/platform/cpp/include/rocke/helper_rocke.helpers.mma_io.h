// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Mirrors helpers/mma_io.py; descriptors expand to ordinary IR before serialization.
 * storage_ir_type -> rocke_storage_ir_type
 * load_matrix_fragment -> rocke_h_load_matrix_fragment
 * pack_fragment_bits -> rocke_h_pack_fragment_bits
 */
#ifndef ROCKE_HELPER_MMA_IO_H
#define ROCKE_HELPER_MMA_IO_H

#include "rocke/ir.h"
#include "rocke/storage.h"

#ifdef __cplusplus
extern "C" {
#endif

const rocke_type_t* rocke_storage_ir_type(const char* dtype);
rocke_value_t* rocke_h_load_matrix_fragment(rocke_ir_builder_t* b,
                                            rocke_value_t* ptr,
                                            rocke_value_t* row_base,
                                            rocke_value_t* lane_group,
                                            uint64_t k0,
                                            const rocke_tensor_storage_t* storage,
                                            const rocke_matrix_fragment_layout_t* layout,
                                            const rocke_type_t* carrier_type);

/* Callback returns a canonical unsigned pattern in an integer carrier.
 * ctx and output pointer array are borrowed for the duration of this call. */
typedef rocke_value_t* (*rocke_load_bits_fn)(rocke_ir_builder_t* b, int index, void* ctx);
rocke_status_t rocke_h_pack_fragment_bits(rocke_ir_builder_t* b,
                                          rocke_load_bits_fn load_bits,
                                          void* ctx,
                                          const rocke_fragment_packing_t* fragment,
                                          rocke_value_t** words,
                                          size_t word_count);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_HELPER_MMA_IO_H */
