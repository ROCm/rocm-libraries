// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Compatibility raw-chunk adapter; mirrors helpers/tiling/compat.py.
 * New authoring uses rocke_h_load_fragment / rocke_h_store_fragment.
 * This existing ABI remains available for raw slot/carrier contracts. Actual
 * loads are emitted solely by the shared internal tiling storage policy.
 */
#ifndef ROCKE_TILING_COMPAT_H
#define ROCKE_TILING_COMPAT_H

#include "rocke/ir.h"
#include "rocke/storage.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Caller owns row indexing, allocation bounds, and lane_group validity.
 * row_base is in pointer storage units; alignment is guaranteed at that row. */
rocke_value_t* rocke_h_load_matrix_fragment(rocke_ir_builder_t* b,
                                            rocke_value_t* ptr,
                                            rocke_value_t* row_base,
                                            rocke_value_t* lane_group,
                                            uint64_t k0,
                                            const char* dtype,
                                            const rocke_matrix_fragment_layout_t* layout,
                                            const rocke_type_t* carrier_type,
                                            int alignment_bytes);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_TILING_COMPAT_H */
