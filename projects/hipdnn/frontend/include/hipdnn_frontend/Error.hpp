// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "Types.hpp"
#include <string>

#define HIPDNN_CHECK_ERROR(x) \
    do                        \
    {                         \
        auto err = x;         \
        if(err.is_bad())      \
        {                     \
            return err;       \
        }                     \
    } while(0)

namespace hipdnn_frontend
{
enum class ErrorCode
{
    OK,
    INVALID_VALUE,
    HIPDNN_BACKEND_ERROR,
    ATTRIBUTE_NOT_SET
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline std::string to_string(ErrorCode code)
{
    static std::vector<std::string> s_errorCodes{
        "OK", "INVALID_VALUE", "HIPDNN_BACKEND_ERROR", "ATTRIBUTE_NOT_SET"};

    return s_errorCodes[static_cast<size_t>(code)];
}

inline std::ostream& operator<<(std::ostream& os, const ErrorCode& error)
{
    return os << to_string(error);
}

typedef ErrorCode error_code_t; // NOLINT(readability-identifier-naming)

struct Error
{
    ErrorCode code;
    std::string err_msg; // NOLINT(readability-identifier-naming)

    Error()
        : code(ErrorCode::OK)
    {
    }

    Error(ErrorCode errorCode, std::string message)
        : code(errorCode)
        , err_msg(std::move(message))
    {
    }

    std::string get_message() const // NOLINT(readability-identifier-naming)
    {
        return err_msg;
    }

    ErrorCode get_code() const // NOLINT(readability-identifier-naming)
    {
        return code;
    }

    bool is_good() const // NOLINT(readability-identifier-naming)
    {
        return code == ErrorCode::OK;
    }
    bool is_bad() const // NOLINT(readability-identifier-naming)
    {
        return code != ErrorCode::OK;
    }

    bool operator==(ErrorCode otherCode) const
    {
        return code == otherCode;
    }
    bool operator!=(ErrorCode otherCode) const
    {
        return code != otherCode;
    }
    bool operator==(const Error& other) const
    {
        return code == other.code;
    }
    bool operator!=(const Error& other) const
    {
        return code != other.code;
    }
};

inline std::ostream& operator<<(std::ostream& os, const Error& error)
{
    return os << "{" << error.code << ", " << error.get_message() << "}";
}

typedef Error error_object; // NOLINT(readability-identifier-naming)
typedef Error error_t; // NOLINT(readability-identifier-naming)

#define HIPDNN_RETURN_IF_NE(x, y, error_status, message) \
    if(x != y)                                           \
    {                                                    \
        return {error_status, message};                  \
    }

#define HIPDNN_RETURN_IF_EQ(x, y, error_status, message) \
    if(x == y)                                           \
    {                                                    \
        return {error_status, message};                  \
    }

#define HIPDNN_RETURN_IF_TRUE(x, error_status, message) \
    if(x)                                               \
    {                                                   \
        return {error_status, message};                 \
    }

#define HIPDNN_RETURN_IF_FALSE(x, error_status, message) \
    if(!(x))                                             \
    {                                                    \
        return {error_status, message};                  \
    }

#define HIPDNN_RETURN_IF_NULL(x, error_status, message) \
    if(x == nullptr)                                    \
    {                                                   \
        return {error_status, message};                 \
    }

#define HIPDNN_RETURN_IF_LT(x, y, error_status, message) \
    if(x < y)                                            \
    {                                                    \
        return {error_status, message};                  \
    }

#define HIPDNN_RETURN_IF_GE(x, y, error_status, message) \
    if(x >= y)                                           \
    {                                                    \
        return {error_status, message};                  \
    }

#define HIPDNN_RETURN_IF_LE(x, y, error_status, message) \
    if(x <= y)                                           \
    {                                                    \
        return {error_status, message};                  \
    }
}
