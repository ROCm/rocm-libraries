// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <exception>
#include <string>

#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

namespace hipdnn_plugin_sdk
{

class HipdnnPluginException : public std::exception
{
public:
    explicit HipdnnPluginException(hipdnnPluginStatus_t status, std::string message)
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

    hipdnnPluginStatus_t getStatus() const noexcept
    {
        return _status;
    }

private:
    hipdnnPluginStatus_t _status;
    std::string _message;
};

} // namespace hipdnn_plugin_sdk

#define PLUGIN_THROW_IF_NE(x, y, failureStatus, message)                            \
    do                                                                              \
    {                                                                               \
        if((x) != (y))                                                              \
        {                                                                           \
            throw hipdnn_plugin_sdk::HipdnnPluginException(failureStatus, message); \
        }                                                                           \
    } while(0)

#define PLUGIN_THROW_IF_EQ(x, y, failureStatus, message)                            \
    do                                                                              \
    {                                                                               \
        if((x) == (y))                                                              \
        {                                                                           \
            throw hipdnn_plugin_sdk::HipdnnPluginException(failureStatus, message); \
        }                                                                           \
    } while(0)

#define PLUGIN_THROW_IF_TRUE(x, failureStatus, message)                             \
    do                                                                              \
    {                                                                               \
        if(x)                                                                       \
        {                                                                           \
            throw hipdnn_plugin_sdk::HipdnnPluginException(failureStatus, message); \
        }                                                                           \
    } while(0)

#define PLUGIN_THROW_IF_FALSE(x, failureStatus, message)                            \
    do                                                                              \
    {                                                                               \
        if(!(x))                                                                    \
        {                                                                           \
            throw hipdnn_plugin_sdk::HipdnnPluginException(failureStatus, message); \
        }                                                                           \
    } while(0)

#define PLUGIN_THROW_IF_NULL(x, failureStatus, message)                             \
    do                                                                              \
    {                                                                               \
        if((x) == nullptr)                                                          \
        {                                                                           \
            throw hipdnn_plugin_sdk::HipdnnPluginException(failureStatus, message); \
        }                                                                           \
    } while(0)

#define PLUGIN_THROW_IF_LT(x, y, failureStatus, message)                            \
    do                                                                              \
    {                                                                               \
        if((x) < (y))                                                               \
        {                                                                           \
            throw hipdnn_plugin_sdk::HipdnnPluginException(failureStatus, message); \
        }                                                                           \
    } while(0)
