// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// Portions derived from NVIDIA cuDNN frontend (include/cudnn_frontend/graph_helpers.h
// and include/cudnn_frontend_Logging.h), used under the MIT license. The error
// type names, error-code enum, and error/log macro names below intentionally
// mirror cuDNN frontend so hipified v9 consumers compile unchanged.

/**
 * @file graph_helpers.h
 * @brief Error-type aliases and error/log macros for the hipDNN cuDNN shim.
 *
 * cuDNN frontend's `error_object` / `error_t` / `error_code_t` (defined upstream
 * in `cudnn_frontend/graph_helpers.h`) are structurally identical to
 * hipDNN's `Error` / `ErrorCode`, and hipDNN's error-code enum is a
 * name-superset of cuDNN's (RFC 0012 §4.6). This header therefore **aliases**
 * the hipDNN types into `<shim_ns>` with `using` declarations rather than
 * wrapping them — `cudnn_frontend::error_t` *is* `hipdnn_frontend::error_t`, so
 * an error returned by a wrapped hipDNN call flows out unchanged, with no cast
 * between enum families (RFC 0012 §4.3).
 *
 * It also re-exports the cuDNN frontend error/log macros
 * (`CHECK_CUDNN_FRONTEND_ERROR`, `RETURN_CUDNN_FRONTEND_ERROR_IF`,
 * `CUDNN_FE_LOG*`) — PyTorch's `AT_CUDNN_FRONTEND_CHECK` expands to them. The
 * log macros delegate to hipDNN's frontend logging (off by default, gated by
 * `HIPDNN_LOG_LEVEL`); the full cuDNN-FE log env-var bridge (RFC 0012 §4.6)
 * lands in a later phase.
 *
 * @note Internal-to-shim; pulled in by the umbrella `cudnn_frontend.h`.
 */

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Logging.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend
{

// Error types aliased 1:1 (RFC 0012 §4.6). hipDNN's `error_code_t` is a
// name-superset of cuDNN's 16 codes, and `error_object` / `error_t` share
// cuDNN FE's public surface (code, err_msg, get_code/get_message/is_good/
// is_bad/operator==/!=), so the alias is exact.
using hipdnn_frontend::error_code_t;
using hipdnn_frontend::error_object;
using hipdnn_frontend::error_t;

} // namespace hipdnn_frontend::compatibility::cudnn_frontend

// ===========================================================================
// Error/log macros (RFC 0012 §4.6) — re-exported with the cuDNN FE names.
// ===========================================================================
// These are global (un-namespaced), exactly as upstream ships them, because
// consumer code (e.g. PyTorch's AT_CUDNN_FRONTEND_CHECK) uses the bare names.
// `error_t` / `error_code_t` referenced inside expand against whatever
// `cudnn_frontend` alias the consumer has in scope, matching upstream behavior.

// The log macros forward their `<<`-streamed argument straight into hipDNN's
// frontend INFO logger. HIPDNN_FE_LOG_INFO gates on HIPDNN_LOG_LEVEL (off by
// default) and accepts a stream expression, so chained `<<` arguments work.
#ifndef CUDNN_FE_LOG
#define CUDNN_FE_LOG(X) HIPDNN_FE_LOG_INFO(X)
#endif

// The X argument is a stream expression: it must stay un-parenthesized so it
// chains off the leftmost LogStream operand (parenthesizing it would force a
// `const char* << ...` evaluation and break compilation). Same NOLINT pattern
// the SDK uses for its own streaming log macros.
#ifndef CUDNN_FE_LOG_LABEL
#define CUDNN_FE_LOG_LABEL(X) \
    HIPDNN_FE_LOG_INFO("[cudnn_frontend] " << X) // NOLINT(bugprone-macro-parentheses)
#endif

#ifndef CUDNN_FE_LOG_LABEL_ENDL
#define CUDNN_FE_LOG_LABEL_ENDL(X) \
    HIPDNN_FE_LOG_INFO("[cudnn_frontend] " << X) // NOLINT(bugprone-macro-parentheses)
#endif

#ifndef CUDNN_FE_LOG_BANNER
#define CUDNN_FE_LOG_BANNER(X) \
    HIPDNN_FE_LOG_INFO("[cudnn_frontend] === " << X << " ===") // NOLINT(bugprone-macro-parentheses)
#endif

/// @brief Evaluate an expression returning `error_t`; on `is_bad()`, log and
/// propagate it. Mirrors cuDNN FE's `CHECK_CUDNN_FRONTEND_ERROR`.
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

/// @brief If `cond`, log and `return {retval, message}`. Mirrors cuDNN FE's
/// `RETURN_CUDNN_FRONTEND_ERROR_IF`.
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
