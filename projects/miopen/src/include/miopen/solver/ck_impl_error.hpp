// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/// \file ck_impl_error.hpp
/// Error infrastructure for the CK implementation library.
///
/// This header is used by the ck_impl files (CK library side) and provides:
/// - CkImplException: typed exception carrying a status code
/// - CK_IMPL_THROW_IF_* macros: throw helpers for input validation
/// - CkImplLastError: thread-local last-error string storage
/// - ck_impl_try_catch(): exception-safe C-API boundary wrapper
/// - toString(ck_impl_status_t): diagnostic string helper
///
/// Follows the same patterns as hipDNN's PluginException.hpp,
/// PluginLastErrorManager.hpp, PluginHelpers.hpp, and PluginDataTypeHelpers.hpp.

#include <miopen/solver/ck_impl_status.h>

#include <cstring>
#include <exception>
#include <new>
#include <string>

// ---------------------------------------------------------------------------
// Exception class (follows HipdnnPluginException)
// ---------------------------------------------------------------------------

class CkImplException : public std::exception
{
public:
    explicit CkImplException(ck_impl_status_t status, std::string message)
        : _status(status), _message(std::move(message))
    {
    }

    const char* what() const noexcept override { return _message.c_str(); }

    ck_impl_status_t getStatus() const noexcept { return _status; }

private:
    ck_impl_status_t _status;
    std::string _message;
};

// ---------------------------------------------------------------------------
// Throw macros (follows PLUGIN_THROW_IF_* from PluginException.hpp)
// ---------------------------------------------------------------------------

// NOLINTBEGIN(bugprone-macro-parentheses) message is a string expression
#define CK_IMPL_THROW_IF_NULL(x, failureStatus, message)   \
    do                                                     \
    {                                                      \
        if((x) == nullptr)                                 \
        {                                                  \
            throw CkImplException(failureStatus, message); \
        }                                                  \
    } while(0)

#define CK_IMPL_THROW_IF_FALSE(x, failureStatus, message)  \
    do                                                     \
    {                                                      \
        if(!(x))                                           \
        {                                                  \
            throw CkImplException(failureStatus, message); \
        }                                                  \
    } while(0)

#define CK_IMPL_THROW_IF_TRUE(x, failureStatus, message)   \
    do                                                     \
    {                                                      \
        if(x)                                              \
        {                                                  \
            throw CkImplException(failureStatus, message); \
        }                                                  \
    } while(0)

#define CK_IMPL_THROW_IF_NE(x, y, failureStatus, message)  \
    do                                                     \
    {                                                      \
        if((x) != (y))                                     \
        {                                                  \
            throw CkImplException(failureStatus, message); \
        }                                                  \
    } while(0)

#define CK_IMPL_THROW_IF_EQ(x, y, failureStatus, message)  \
    do                                                     \
    {                                                      \
        if((x) == (y))                                     \
        {                                                  \
            throw CkImplException(failureStatus, message); \
        }                                                  \
    } while(0)
// NOLINTEND(bugprone-macro-parentheses)

// ---------------------------------------------------------------------------
// Thread-local last-error manager (follows PluginLastErrorManager)
// ---------------------------------------------------------------------------

class CkImplLastError
{
public:
    static ck_impl_status_t setLastError(ck_impl_status_t status, const char* message)
    {
        if(status == CK_IMPL_STATUS_SUCCESS)
        {
            return status;
        }

        auto* buf = buffer();
        if(message != nullptr)
        {
            std::strncpy(buf, message, CK_IMPL_ERROR_STRING_MAX_LENGTH - 1);
            buf[CK_IMPL_ERROR_STRING_MAX_LENGTH - 1] = '\0';
        }
        else
        {
            buf[0] = '\0';
        }

        return status;
    }

    static ck_impl_status_t setLastError(ck_impl_status_t status, const std::string& message)
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
        static thread_local char s_lastError[CK_IMPL_ERROR_STRING_MAX_LENGTH] = {'\0'};
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
/// 1. CkImplException — preserves the specific status code
/// 2. std::bad_alloc — maps to ALLOC_FAILED
/// 3. std::exception — maps to INTERNAL_ERROR with what()
/// 4. catch-all — maps to INTERNAL_ERROR with generic message
template <class F>
ck_impl_status_t ck_impl_try_catch(F f)
{
    try
    {
        f();
    }
    catch(const CkImplException& ex)
    {
        return CkImplLastError::setLastError(ex.getStatus(), ex.what());
    }
    catch(const std::bad_alloc& ex)
    {
        return CkImplLastError::setLastError(CK_IMPL_STATUS_ALLOC_FAILED, ex.what());
    }
    catch(const std::exception& ex)
    {
        return CkImplLastError::setLastError(CK_IMPL_STATUS_INTERNAL_ERROR, ex.what());
    }
    catch(...)
    {
        return CkImplLastError::setLastError(CK_IMPL_STATUS_INTERNAL_ERROR,
                                             "Unknown exception occurred");
    }
    return CK_IMPL_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// toString helper (follows PluginDataTypeHelpers.hpp)
// ---------------------------------------------------------------------------

inline const char* toString(ck_impl_status_t status)
{
    switch(status)
    {
    case CK_IMPL_STATUS_SUCCESS: return "CK_IMPL_STATUS_SUCCESS";
    case CK_IMPL_STATUS_BAD_PARAM: return "CK_IMPL_STATUS_BAD_PARAM";
    case CK_IMPL_STATUS_INVALID_VALUE: return "CK_IMPL_STATUS_INVALID_VALUE";
    case CK_IMPL_STATUS_INTERNAL_ERROR: return "CK_IMPL_STATUS_INTERNAL_ERROR";
    case CK_IMPL_STATUS_ALLOC_FAILED: return "CK_IMPL_STATUS_ALLOC_FAILED";
    }

    return "CK_IMPL_STATUS_UNKNOWN";
}
