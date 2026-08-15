// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCKE_DEVICE_PRINT_INTERNAL_H
#define ROCKE_DEVICE_PRINT_INTERNAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Validate NUL-terminated ASCII print text and optionally return its byte count. */
int rocke_i_valid_print_text(const unsigned char* text, size_t* bytes);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_DEVICE_PRINT_INTERNAL_H */
