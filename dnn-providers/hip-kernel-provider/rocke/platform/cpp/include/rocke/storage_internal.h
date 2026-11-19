// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_STORAGE_INTERNAL_H
#define ROCKE_STORAGE_INTERNAL_H

#include <stdint.h>

/* Native counterpart to core.storage.elements_to_storage. Use the encoded
 * width of a validated register-data dtype. Byte-or-wider elements are held
 * one to one; sub-byte elements fill 32-bit words, including a partial last
 * word. Counts are nonnegative; signed displacements are not counts. */
int64_t rocke_i_elements_to_storage(int encoded_bits, int64_t elements);

#endif /* ROCKE_STORAGE_INTERNAL_H */
