// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file logging_bridge.h
 * @brief Best-effort cuDNN logging env to hipDNN logging env bridge.
 */

#pragma once

#include <cstdlib>
#include <string>

#include <hipdnn_data_sdk/logging/LogLevel.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_frontend/Logging.hpp>

namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
{

inline std::string getCudnnEnv(const char* name)
{
    return hipdnn_data_sdk::utilities::getEnv(name);
}

inline bool cudnnLoggingDisabled()
{
#ifdef NV_CUDNN_FRONTEND_DISABLE_LOGGING
    return true;
#else
    const auto disabled = getCudnnEnv("NV_CUDNN_FRONTEND_DISABLE_LOGGING");
    return !disabled.empty() && disabled != "0";
#endif
}

inline int cudnnLogLevel()
{
    if(cudnnLoggingDisabled())
    {
        return 0;
    }

    const auto envValue = getCudnnEnv("CUDNN_FRONTEND_LOG_INFO");
    return !envValue.empty() ? std::atoi(envValue.c_str()) : 0;
}

inline bool hasCudnnLoggingEnv()
{
    return !getCudnnEnv("CUDNN_FRONTEND_LOG_INFO").empty()
           || !getCudnnEnv("CUDNN_FRONTEND_LOG_FILE").empty()
           || !getCudnnEnv("NV_CUDNN_FRONTEND_DISABLE_LOGGING").empty();
}

inline void configureHipdnnLoggingFromCudnnEnv()
{
    if(!hasCudnnLoggingEnv())
    {
        return;
    }

    const int level = cudnnLogLevel();
    const auto target = getCudnnEnv("CUDNN_FRONTEND_LOG_FILE");
    const bool enabled = level > 0 && !target.empty();

    if(enabled)
    {
        hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "info");
        if(target == "stdout" || target == "stderr")
        {
            hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_FILE");
        }
        else
        {
            hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_FILE", target.c_str());
        }
        static_cast<void>(hipdnn_frontend::setGlobalLogLevel(HIPDNN_SEV_INFO));
        return;
    }

    hipdnn_data_sdk::utilities::setEnv("HIPDNN_LOG_LEVEL", "off");
    hipdnn_data_sdk::utilities::unsetEnv("HIPDNN_LOG_FILE");
    hipdnn_data_sdk::logging::setLogLevel(HIPDNN_SEV_OFF);
    static_cast<void>(hipdnn_frontend::setGlobalLogLevel(HIPDNN_SEV_OFF));
}

} // namespace hipdnn_frontend::compatibility::cudnn_frontend::detail
