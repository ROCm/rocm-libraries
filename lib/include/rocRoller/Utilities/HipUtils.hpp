#pragma once

#include <rocRoller/Utilities/Error.hpp>
#include <rocRoller/Utilities/Logging.hpp>

// Helper macro to check for HIP API errors
#define HIP_CHECK(cmd)                                                                \
    do                                                                                \
    {                                                                                 \
        hipError_t e = cmd;                                                           \
        if(e != hipSuccess)                                                           \
        {                                                                             \
            std::ostringstream msg;                                                   \
            msg << "HIP failure at line " << __LINE__ << ": " << hipGetErrorString(e) \
                << std::endl;                                                         \
            Log::error(msg.str());                                                    \
            AssertFatal(false, msg.str());                                            \
        }                                                                             \
    } while(0)
