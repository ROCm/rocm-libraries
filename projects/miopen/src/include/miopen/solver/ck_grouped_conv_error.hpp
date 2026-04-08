// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// \file ck_grouped_conv_error.hpp
/// Error infrastructure for the CK grouped convolution implementation library.
///
/// This header is used by the ck_impl files (CK library side) and provides:
/// - CKGrpConvException: typed exception carrying a status code
/// - CKGRPCONV_THROW_IF_* macros: throw helpers for input validation
/// - CKGrpConvLastError: thread-local last-error string storage
/// - ckgrpconv_try_catch(): exception-safe C-API boundary wrapper
/// - toString(ckgrpconv_status_t): diagnostic string helper
///
/// Follows the same patterns as hipDNN's PluginException.hpp,
/// PluginLastErrorManager.hpp, PluginHelpers.hpp, and PluginDataTypeHelpers.hpp.

#include <miopen/solver/ck_grouped_conv_status.h>

#include <cstring>
#include <exception>
#include <new>
#include <string>

// ---------------------------------------------------------------------------
// Exception class (follows HipdnnPluginException)
// ---------------------------------------------------------------------------

class CKGrpConvException : public std::exception
{
public:
    explicit CKGrpConvException(ckgrpconv_status_t status, std::string message)
        : _status(status), _message(std::move(message))
    {
    }

    const char* what() const noexcept override { return _message.c_str(); }

    ckgrpconv_status_t getStatus() const noexcept { return _status; }

private:
    ckgrpconv_status_t _status;
    std::string _message;
};

// ---------------------------------------------------------------------------
// Throw macros (follows PLUGIN_THROW_IF_* from PluginException.hpp)
// ---------------------------------------------------------------------------

// NOLINTBEGIN(bugprone-macro-parentheses) message is a string expression
#define CKGRPCONV_THROW_IF_NULL(x, failureStatus, message)    \
    do                                                        \
    {                                                         \
        if((x) == nullptr)                                    \
        {                                                     \
            throw CKGrpConvException(failureStatus, message); \
        }                                                     \
    } while(0)

#define CKGRPCONV_THROW_IF_FALSE(x, failureStatus, message)   \
    do                                                        \
    {                                                         \
        if(!(x))                                              \
        {                                                     \
            throw CKGrpConvException(failureStatus, message); \
        }                                                     \
    } while(0)

#define CKGRPCONV_THROW_IF_TRUE(x, failureStatus, message)    \
    do                                                        \
    {                                                         \
        if(x)                                                 \
        {                                                     \
            throw CKGrpConvException(failureStatus, message); \
        }                                                     \
    } while(0)

#define CKGRPCONV_THROW_IF_NE(x, y, failureStatus, message)   \
    do                                                        \
    {                                                         \
        if((x) != (y))                                        \
        {                                                     \
            throw CKGrpConvException(failureStatus, message); \
        }                                                     \
    } while(0)

#define CKGRPCONV_THROW_IF_EQ(x, y, failureStatus, message)   \
    do                                                        \
    {                                                         \
        if((x) == (y))                                        \
        {                                                     \
            throw CKGrpConvException(failureStatus, message); \
        }                                                     \
    } while(0)
// NOLINTEND(bugprone-macro-parentheses)

// ---------------------------------------------------------------------------
// Thread-local last-error manager (follows PluginLastErrorManager)
// ---------------------------------------------------------------------------

class CKGrpConvLastError
{
public:
    static ckgrpconv_status_t setLastError(ckgrpconv_status_t status, const char* message)
    {
        if(status == CKGRPCONV_STATUS_SUCCESS)
        {
            return status;
        }

        auto* buf = buffer();
        if(message != nullptr)
        {
            std::strncpy(buf, message, CKGRPCONV_ERROR_STRING_MAX_LENGTH - 1);
            buf[CKGRPCONV_ERROR_STRING_MAX_LENGTH - 1] = '\0';
        }
        else
        {
            buf[0] = '\0';
        }

        return status;
    }

    static ckgrpconv_status_t setLastError(ckgrpconv_status_t status, const std::string& message)
    {
        return setLastError(status, message.c_str());
    }

    static const char* getLastError() { return buffer(); }

private:
    // Function-local thread_local static avoids ODR issues across translation
    // units. C++17 guarantees a single instance for inline functions.
    static char* buffer()
    {
        // NOLINTNEXTLINE
        static thread_local char s_lastError[CKGRPCONV_ERROR_STRING_MAX_LENGTH] = {'\0'};
        return s_lastError;
    }
};

// ---------------------------------------------------------------------------
// tryCatch wrapper (follows hipdnn_plugin_sdk::tryCatch from PluginHelpers.hpp)
// ---------------------------------------------------------------------------

/// Wraps a callable in a try/catch block, converting exceptions to status
/// codes and storing the error message in the thread-local last-error buffer.
///
/// Four catch clauses matching hipDNN's pattern:
/// 1. CKGrpConvException — preserves the specific status code
/// 2. std::bad_alloc — maps to ALLOC_FAILED
/// 3. std::exception — maps to INTERNAL_ERROR with what()
/// 4. catch-all — maps to INTERNAL_ERROR with generic message
template <class F>
ckgrpconv_status_t ckgrpconv_try_catch(F f)
{
    try
    {
        f();
    }
    catch(const CKGrpConvException& ex)
    {
        return CKGrpConvLastError::setLastError(ex.getStatus(), ex.what());
    }
    catch(const std::bad_alloc& ex)
    {
        return CKGrpConvLastError::setLastError(CKGRPCONV_STATUS_ALLOC_FAILED, ex.what());
    }
    catch(const std::exception& ex)
    {
        return CKGrpConvLastError::setLastError(CKGRPCONV_STATUS_INTERNAL_ERROR, ex.what());
    }
    catch(...)
    {
        return CKGrpConvLastError::setLastError(CKGRPCONV_STATUS_INTERNAL_ERROR,
                                                "Unknown exception occurred");
    }
    return CKGRPCONV_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// toString helper (follows PluginDataTypeHelpers.hpp)
// ---------------------------------------------------------------------------

inline const char* toString(ckgrpconv_status_t status)
{
    switch(status)
    {
    case CKGRPCONV_STATUS_SUCCESS: return "CKGRPCONV_STATUS_SUCCESS";
    case CKGRPCONV_STATUS_BAD_PARAM: return "CKGRPCONV_STATUS_BAD_PARAM";
    case CKGRPCONV_STATUS_INVALID_VALUE: return "CKGRPCONV_STATUS_INVALID_VALUE";
    case CKGRPCONV_STATUS_INTERNAL_ERROR: return "CKGRPCONV_STATUS_INTERNAL_ERROR";
    case CKGRPCONV_STATUS_ALLOC_FAILED: return "CKGRPCONV_STATUS_ALLOC_FAILED";
    default: return "CKGRPCONV_STATUS_UNKNOWN";
    }
}
