// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#ifndef ROCKE_SHA256_H
#define ROCKE_SHA256_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* SHA-256 of contiguous input, in standard big-endian digest byte order.
 * No allocation. data may be NULL only when size is zero; out must have room
 * for 32 bytes and must not overlap the input. */
void rocke_sha256(const void* data, size_t size, uint8_t out[32]);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_SHA256_H */
