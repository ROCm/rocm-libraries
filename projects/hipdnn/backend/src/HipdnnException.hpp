// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "HipdnnStatus.h"
#include <exception>
#include <string>

namespace hipdnn_backend
{

class HipdnnException : public std::exception
{
public:
    explicit HipdnnException(hipdnnStatus_t status, std::string message)
        : _status(status)
        , _message(std::move(message))
    {
    }

    const char* what() const noexcept override
    {
        return _message.c_str();
    }

    std::string getMessage() const noexcept
    {
        return _message;
    }

    hipdnnStatus_t getStatus() const noexcept
    {
        return _status;
    }

private:
    hipdnnStatus_t _status;
    std::string _message;
};

#define THROW_IF_NE(x, y, failureStatus, message)                      \
    if(x != y)                                                         \
    {                                                                  \
        throw hipdnn_backend::HipdnnException(failureStatus, message); \
    }

#define THROW_IF_EQ(x, y, failureStatus, message)                      \
    if(x == y)                                                         \
    {                                                                  \
        throw hipdnn_backend::HipdnnException(failureStatus, message); \
    }

#define THROW_IF_TRUE(x, failureStatus, message)                       \
    if(x)                                                              \
    {                                                                  \
        throw hipdnn_backend::HipdnnException(failureStatus, message); \
    }

#define THROW_IF_FALSE(x, failureStatus, message)                      \
    if(!(x))                                                           \
    {                                                                  \
        throw hipdnn_backend::HipdnnException(failureStatus, message); \
    }

#define THROW_IF_NULL(x, failureStatus, message)                       \
    if(x == nullptr)                                                   \
    {                                                                  \
        throw hipdnn_backend::HipdnnException(failureStatus, message); \
    }

#define THROW_IF_LT(x, y, failureStatus, message)                      \
    if(x < y)                                                          \
    {                                                                  \
        throw hipdnn_backend::HipdnnException(failureStatus, message); \
    }

} // namespace hipdnn_backend
