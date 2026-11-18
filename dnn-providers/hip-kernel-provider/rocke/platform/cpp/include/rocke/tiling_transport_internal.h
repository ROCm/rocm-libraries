// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Internal storage-run load policy; mirrors helpers/tiling/transport.py.
 * Coordinates, bounds and alignment guarantees belong to the descriptor caller.
 * Output array is borrowed; produced SSA values are builder-owned.
 */
#ifndef ROCKE_TILING_TRANSPORT_INTERNAL_H
#define ROCKE_TILING_TRANSPORT_INTERNAL_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

rocke_status_t rocke_h_load_storage_chunks(rocke_ir_builder_t* b,
                                           rocke_value_t* ptr,
                                           rocke_value_t* offset,
                                           const rocke_type_t* element,
                                           int count,
                                           int alignment_bytes,
                                           bool byte_addressed,
                                           bool word_vectors,
                                           rocke_value_t** pieces,
                                           size_t capacity,
                                           int* piece_count);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_TILING_TRANSPORT_INTERNAL_H */
