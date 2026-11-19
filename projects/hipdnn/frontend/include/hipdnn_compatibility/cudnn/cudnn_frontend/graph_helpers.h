// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend, used under the MIT license:
//   include/cudnn_frontend/graph_helpers.h, include/cudnn_frontend_Logging.h
//     Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
// License text and pinned upstream version: THIRD_PARTY_LICENSES.md
// (installed to share/doc/hipdnn_frontend).

/**
 * @file graph_helpers.h
 * @brief cuDNN FE error types and error/log macros for the shim.
 */

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Logging.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{

/// @brief Alias of hipdnn_frontend::error_code_t.
using hipdnn_frontend::error_code_t;
/// @brief Alias of hipdnn_frontend::error_object.
using hipdnn_frontend::error_object;
/// @brief Alias of hipdnn_frontend::error_t.
using hipdnn_frontend::error_t;

} // namespace hipdnn_frontend::compatibility::cudnn_frontend

// NV_CUDNN_FRONTEND_DISABLE_LOGGING compiles the log macros out, as in cuDNN FE. This is
// separate from the runtime env bridge in logging_bridge.h, which applies at cudnnCreate().
#ifdef NV_CUDNN_FRONTEND_DISABLE_LOGGING

#ifndef CUDNN_FE_LOG
#define CUDNN_FE_LOG(X) \
    do                  \
    {                   \
    } while(0)
#endif
#ifndef CUDNN_FE_LOG_LABEL
#define CUDNN_FE_LOG_LABEL(X) \
    do                        \
    {                         \
    } while(0)
#endif
#ifndef CUDNN_FE_LOG_LABEL_ENDL
#define CUDNN_FE_LOG_LABEL_ENDL(X) \
    do                             \
    {                              \
    } while(0)
#endif
#ifndef CUDNN_FE_LOG_BANNER
#define CUDNN_FE_LOG_BANNER(X) \
    do                         \
    {                          \
    } while(0)
#endif

#else

/// @brief Logs a `<<` stream expression via HIPDNN_FE_LOG_INFO; no-op under
/// NV_CUDNN_FRONTEND_DISABLE_LOGGING.
#ifndef CUDNN_FE_LOG
#define CUDNN_FE_LOG(X) HIPDNN_FE_LOG_INFO(X)
#endif

// X stays un-parenthesized (hence the NOLINTs) so it chains off the leftmost `<<` operand.
/// @brief Logs a `[cudnn_frontend] `-prefixed stream expression via HIPDNN_FE_LOG_INFO;
/// no-op under NV_CUDNN_FRONTEND_DISABLE_LOGGING.
#ifndef CUDNN_FE_LOG_LABEL
#define CUDNN_FE_LOG_LABEL(X) \
    HIPDNN_FE_LOG_INFO("[cudnn_frontend] " << X) // NOLINT(bugprone-macro-parentheses)
#endif

/// @brief Same as CUDNN_FE_LOG_LABEL; no-op under NV_CUDNN_FRONTEND_DISABLE_LOGGING.
#ifndef CUDNN_FE_LOG_LABEL_ENDL
#define CUDNN_FE_LOG_LABEL_ENDL(X) \
    HIPDNN_FE_LOG_INFO("[cudnn_frontend] " << X) // NOLINT(bugprone-macro-parentheses)
#endif

/// @brief Logs a `[cudnn_frontend] === X ===` banner via HIPDNN_FE_LOG_INFO; no-op under
/// NV_CUDNN_FRONTEND_DISABLE_LOGGING.
#ifndef CUDNN_FE_LOG_BANNER
#define CUDNN_FE_LOG_BANNER(X) \
    HIPDNN_FE_LOG_INFO("[cudnn_frontend] === " << X << " ===") // NOLINT(bugprone-macro-parentheses)
#endif

#endif // NV_CUDNN_FRONTEND_DISABLE_LOGGING

/// @brief Evaluates an `error_t` expression; if bad, logs and returns it.
#ifndef CHECK_CUDNN_FRONTEND_ERROR
#define CHECK_CUDNN_FRONTEND_ERROR(x)                                                          \
    do                                                                                         \
    {                                                                                          \
        if(auto retval = (x); retval.is_bad())                                                 \
        {                                                                                      \
            CUDNN_FE_LOG_LABEL_ENDL("ERROR: " << #x << " at " << __FILE__ << ":" << __LINE__); \
            return retval;                                                                     \
        }                                                                                      \
    } while(0)
#endif

/// @brief If `cond`, logs and returns `{retval, message}`.
#ifndef RETURN_CUDNN_FRONTEND_ERROR_IF
#define RETURN_CUDNN_FRONTEND_ERROR_IF(cond, retval, message)                           \
    do                                                                                  \
    {                                                                                   \
        if(cond)                                                                        \
        {                                                                               \
            if((retval) == error_code_t::OK)                                            \
            {                                                                           \
                CUDNN_FE_LOG_LABEL("INFO: ");                                           \
            }                                                                           \
            else                                                                        \
            {                                                                           \
                CUDNN_FE_LOG_LABEL("ERROR: ");                                          \
            }                                                                           \
            CUDNN_FE_LOG((message) << ". because (" << #cond ") at " << __FILE__ << ":" \
                                   << __LINE__ << "\n");                                \
            return {retval, message};                                                   \
        }                                                                               \
    } while(0)
#endif
