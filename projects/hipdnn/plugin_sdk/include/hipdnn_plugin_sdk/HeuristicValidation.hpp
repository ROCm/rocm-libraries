// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/logging/CallbackTypes.h>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

/**
 * @file HeuristicValidation.hpp
 * @brief Argument validation macros shared by heuristic plugin C ABI entry points.
 *
 * Heuristic plugins repeat the same null-pointer / serialized-buffer / array
 * checks at every C ABI entry point. The macros below collapse the three
 * recurring patterns into one-liners that log through the plugin's existing
 * logging macro and return @c HIPDNN_PLUGIN_STATUS_BAD_PARAM on failure.
 *
 * Each macro takes the per-plugin log macro as an argument so error messages
 * keep the plugin's own prefix and routing — there is no global logging
 * dependency in this header.
 */

// NOLINTBEGIN(bugprone-macro-parentheses)

/// Reject a null pointer at a C ABI entry point.
///
/// @param ptr  Pointer to validate. Must be a plain identifier or expression.
/// @param log  Per-plugin logging macro (e.g. CONFIG_LOG).
/// @param msg  String literal logged at HIPDNN_SEV_ERROR when @p ptr is null.
#define HIPDNN_PLUGIN_REQUIRE_NOT_NULL(ptr, log, msg) \
    do                                                \
    {                                                 \
        if((ptr) == nullptr)                          \
        {                                             \
            log(HIPDNN_SEV_ERROR, msg);               \
            return HIPDNN_PLUGIN_STATUS_BAD_PARAM;    \
        }                                             \
    } while(0)

/// Reject a malformed @c hipdnnPluginConstData_t* argument.
///
/// @param data         Pointer to the const-data struct.
/// @param requireSize  When true, also rejects buffers with @c size == 0.
/// @param log          Per-plugin logging macro.
/// @param msg          String literal logged at HIPDNN_SEV_ERROR on failure.
#define HIPDNN_PLUGIN_REQUIRE_CONST_DATA(data, requireSize, log, msg)                           \
    do                                                                                          \
    {                                                                                           \
        if((data) == nullptr || (data)->ptr == nullptr || ((requireSize) && (data)->size == 0)) \
        {                                                                                       \
            log(HIPDNN_SEV_ERROR, msg);                                                         \
            return HIPDNN_PLUGIN_STATUS_BAD_PARAM;                                              \
        }                                                                                       \
    } while(0)

/// Reject a (pointer, count) pair that claims a non-empty array but supplies
/// a null pointer.
///
/// @param ptr    Array pointer.
/// @param count  Element count claimed by the caller.
/// @param log    Per-plugin logging macro.
/// @param msg    String literal logged at HIPDNN_SEV_ERROR on failure.
#define HIPDNN_PLUGIN_REQUIRE_ARRAY(ptr, count, log, msg) \
    do                                                    \
    {                                                     \
        if((ptr) == nullptr && (count) > 0)               \
        {                                                 \
            log(HIPDNN_SEV_ERROR, msg);                   \
            return HIPDNN_PLUGIN_STATUS_BAD_PARAM;        \
        }                                                 \
    } while(0)

// NOLINTEND(bugprone-macro-parentheses)
