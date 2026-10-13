// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef ROCKE_DEVICE_PRINT_INTERNAL_H
#define ROCKE_DEVICE_PRINT_INTERNAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct rocke_device_print_limits
{
    size_t max_literal_bytes;
    int max_value_count;
} rocke_device_print_limits_t;

/* Validate NUL-terminated ASCII print text and optionally return its byte count. */
int rocke_i_valid_print_text(const unsigned char* text, size_t* bytes);

/* Resolve strict environment overrides for the internal record safety limits. */
int rocke_i_device_print_limits(rocke_device_print_limits_t* limits,
                                char* error,
                                size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_DEVICE_PRINT_INTERNAL_H */
