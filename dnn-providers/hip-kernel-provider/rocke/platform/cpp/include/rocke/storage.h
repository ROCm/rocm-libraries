// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Shared bit storage; mirrors core/storage.py.
 * BitPacking      -> rocke_bit_packing_t
 * FragmentPacking -> rocke_fragment_packing_t
 * register_packing -> rocke_register_packing
 * packed_word_range -> rocke_packed_word_range
 * All descriptors are caller-owned. Functions return false for invalid inputs,
 * overflow or insufficient buffers; no exceptions cross this C API. The two
 * register functions also write the reason into a caller-owned `err` buffer
 * (truncated to `err_cap`, may be NULL), worded as core/storage.py words the
 * ValueError it raises.
 */
#ifndef ROCKE_STORAGE_H
#define ROCKE_STORAGE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

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

/* Interleaved contiguous K chunks, not arbitrary matrix distributions,
 * transposed axes, or swizzled tensor addresses. */
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

/* How `count` elements of `dtype` (any accepted spelling) sit densely in 32-bit
 * registers: low-bit-first, no padding between elements, whole registers only.
 * On success *packed is false for a dtype of 8 or more bits (Python's None: it
 * has an LLVM element type, and LLVM packs its vectors into registers itself),
 * and true with the dense packing in *out for a sub-byte dtype. Fails for an
 * unknown dtype, a count below 1, a predicate, or a payload that leaves a
 * register partly filled. Python's TypeError for a non-integer count has no C
 * twin. */
bool rocke_register_packing(const char* dtype,
                            int64_t count,
                            bool* packed,
                            rocke_fragment_packing_t* out,
                            char* err,
                            size_t err_cap);

/* The registers [*first_word, *end_word) that hold elements
 * [start, start + length) of `packing`. Fails for a start below 0, a length
 * below 1, a range that runs past the fragment, or one that does not start and
 * end on a register boundary (the message names the element step that does).
 * Python's TypeError for a non-integer start or length has no C twin. */
bool rocke_packed_word_range(const rocke_fragment_packing_t* packing,
                             int64_t start,
                             int64_t length,
                             uint64_t* first_word,
                             uint64_t* end_word,
                             char* err,
                             size_t err_cap);
#ifdef __cplusplus
}
#endif
#endif /* ROCKE_STORAGE_H */
