/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_conv_fwd_small_wmma_internal.h -- the piece of
 * conv_fwd_small_wmma.cpp the wave64 MFMA forms of the same kernels
 * (conv_fwd_small_mfma.cpp) reuse:
 *
 *   Python (conv_fwd_small_wmma.py)      C
 *   _common_shape_reason(p, square)      rocke_fwd_small_common_shape_ok
 */
#ifndef ROCKE_INSTANCE_CONV_FWD_SMALL_WMMA_INTERNAL_H
#define ROCKE_INSTANCE_CONV_FWD_SMALL_WMMA_INTERNAL_H

#include <stdbool.h>
#include <stddef.h>

#include "rocke/instance_conv_direct_grouped.h" /* rocke_direct_conv_problem_t */

/* The problem shapes small C (square = true) and small K (square = false:
 * rectangular filters with their own column padding) serve: ungrouped,
 * undilated, PAD within the filter, stride >= 1, i32 tensor byte offsets. On
 * failure writes the Python reason and returns false. */
bool rocke_fwd_small_common_shape_ok(const rocke_direct_conv_problem_t* p,
                                     bool square,
                                     char* reason,
                                     size_t reason_cap);

#endif /* ROCKE_INSTANCE_CONV_FWD_SMALL_WMMA_INTERNAL_H */
