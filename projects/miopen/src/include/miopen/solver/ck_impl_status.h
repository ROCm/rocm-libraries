// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// \file ck_impl_status.h
/// Status codes and macros for the CK implementation C-API.
///
/// This header is C-compatible and can be included from both C and C++ code.
/// It follows the same pattern as hipDNN's PluginApiDataTypes.h.

// NOLINTBEGIN
#ifdef __cplusplus
extern "C" {
#endif

/// Status codes returned by CK implementation API functions.
typedef enum
{
    CK_IMPL_STATUS_SUCCESS        = 0,
    CK_IMPL_STATUS_BAD_PARAM      = 1,
    CK_IMPL_STATUS_INVALID_VALUE  = 2,
    CK_IMPL_STATUS_INTERNAL_ERROR = 3,
    CK_IMPL_STATUS_ALLOC_FAILED   = 4,
} ck_impl_status_t;

/// Maximum length for error strings stored in the per-thread error buffer.
/// The length includes the null-terminating character.
#define CK_IMPL_ERROR_STRING_MAX_LENGTH 2048

#ifdef __cplusplus
}
#endif
// NOLINTEND

/// Nodiscard attribute for status-returning functions.
#if defined(__cplusplus) && __cplusplus >= 201703L
#define CK_IMPL_NODISCARD [[nodiscard]]
#elif defined(__GNUC__) || defined(__clang__)
#define CK_IMPL_NODISCARD __attribute__((warn_unused_result))
#else
#define CK_IMPL_NODISCARD
#endif
