// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* IR storage types, independent of matrix transport.
 * Python core/ir_storage.py        C99
 * storage_ir_type(dtype)       -> rocke_storage_ir_type(dtype)
 * Unknown dtypes return NULL. Returned scalar types are borrowed.
 */
#ifndef ROCKE_IR_STORAGE_H
#define ROCKE_IR_STORAGE_H

#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

const rocke_type_t* rocke_storage_ir_type(const char* dtype);
const rocke_type_t* rocke_storage_element_type(const rocke_type_t* dtype);
/* Builder-owned vector type; packed logical counts must fill whole words. */
const rocke_type_t*
    rocke_storage_vector_type(rocke_ir_builder_t* b, const rocke_type_t* dtype, int count);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_IR_STORAGE_H */
