// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Logging.hpp>

// Helper macro to check for HIP API errors
#define HIP_CHECK(cmd, message...)                                                    \
    do                                                                                \
    {                                                                                 \
        hipError_t e = cmd;                                                           \
        if(e != hipSuccess)                                                           \
        {                                                                             \
            std::ostringstream msg;                                                   \
            msg << "HIP failure at line " << __LINE__ << ": " << hipGetErrorString(e) \
                << concatenate("", ##message) << std::endl;                           \
            Log::error(msg.str());                                                    \
            Throw<FatalError>(msg.str());                                             \
        }                                                                             \
    } while(0)
