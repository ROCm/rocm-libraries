// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Shared bit storage; mirrors core/storage.py.
 * BitPacking      -> rocke_bit_packing_t
 * FragmentPacking -> rocke_fragment_packing_t
 * TensorStorage   -> rocke_tensor_storage_t
 * All descriptors are caller-owned. Functions return false for invalid inputs,
 * overflow or insufficient buffers; no exceptions cross this C API.
 */
#ifndef ROCKE_STORAGE_H
#define ROCKE_STORAGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "rocke/dtypes.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rocke_bit_packing
{
    int element_bits;
    int slot_bits; /* 0 selects dense packing at construction. */
} rocke_bit_packing_t;

typedef struct rocke_fragment_packing
{
    rocke_bit_packing_t packing;
    uint64_t count;
    int carrier_bits;
    uint64_t carrier_count;
} rocke_fragment_packing_t;

typedef struct rocke_matrix_fragment_layout
{
    rocke_fragment_packing_t fragment;
    int chunk_elements;
    int lane_groups;
    int lanes_per_group;
} rocke_matrix_fragment_layout_t;

bool rocke_matrix_fragment_layout_init(rocke_matrix_fragment_layout_t* out,
                                       const rocke_fragment_packing_t* fragment,
                                       int chunk_elements,
                                       int lane_groups,
                                       int lanes_per_group);
bool rocke_matrix_fragment_coord(
    const rocke_matrix_fragment_layout_t* p, int lane, uint64_t slot, uint64_t* row, uint64_t* k);

typedef struct rocke_tensor_storage
{
    const rocke_dtype_info_t* dtype; /* Borrowed static descriptor. */
    uint64_t rows;
    uint64_t cols;
    uint64_t row_stride_bytes;
    uint64_t base_bit_offset;
    uint64_t alignment_bytes;
    rocke_bit_packing_t packing;
} rocke_tensor_storage_t;

bool rocke_bit_packing_init(rocke_bit_packing_t* out, int element_bits, int slot_bits);
bool rocke_bit_packing_group(const rocke_bit_packing_t* p,
                             int carrier_bits,
                             uint64_t* values,
                             uint64_t* carriers);
bool rocke_bit_packing_offset(const rocke_bit_packing_t* p, uint64_t index, uint64_t* bits);
bool rocke_bit_packing_bytes(const rocke_bit_packing_t* p,
                             uint64_t count,
                             uint64_t bit_offset,
                             uint64_t* bytes);
/* Pack writes a fresh zero-padded stream; caller must exclusively own dst.
 * Unpack reads only the allocated bytes, including a partial final byte. */
bool rocke_bit_pack(const rocke_bit_packing_t* p,
                    const uint64_t* patterns,
                    size_t count,
                    uint64_t bit_offset,
                    uint8_t* dst,
                    size_t dst_size);
bool rocke_bit_unpack(const rocke_bit_packing_t* p,
                      const uint8_t* src,
                      size_t src_size,
                      size_t count,
                      uint64_t bit_offset,
                      uint64_t* patterns);
bool rocke_fragment_packing_init(rocke_fragment_packing_t* out,
                                 const rocke_bit_packing_t* p,
                                 uint64_t count,
                                 int carrier_bits,
                                 uint64_t carrier_count);
bool rocke_fragment_pack(const rocke_fragment_packing_t* p,
                         const uint64_t* patterns,
                         size_t count,
                         uint64_t* carriers,
                         size_t carrier_count);
/* row_stride_bytes=UINT64_MAX selects a minimal packed row stride. */
bool rocke_tensor_storage_init(rocke_tensor_storage_t* out,
                               const char* dtype,
                               uint64_t rows,
                               uint64_t cols,
                               uint64_t row_stride_bytes,
                               int slot_bits,
                               uint64_t base_bit_offset,
                               uint64_t alignment_bytes);
bool rocke_tensor_storage_bytes(const rocke_tensor_storage_t* p, uint64_t* bytes);
bool rocke_tensor_storage_address(
    const rocke_tensor_storage_t* p, uint64_t row, uint64_t col, uint64_t* byte, int* shift);

#ifdef __cplusplus
}
#endif
#endif /* ROCKE_STORAGE_H */
