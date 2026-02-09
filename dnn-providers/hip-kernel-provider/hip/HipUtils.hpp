#pragma once
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>
#include <stdexcept>
#include <string>

namespace hip_plugin
{

// For HIP runtime API calls
#define HIP_CHECK(call)                                                          \
    do                                                                           \
    {                                                                            \
        hipError_t status = (call);                                              \
        if(status != hipSuccess)                                                 \
        {                                                                        \
            throw std::runtime_error(std::string(#call)                          \
                                     + " failed: " + hipGetErrorString(status)); \
        }                                                                        \
    } while(0)

// For hipRTC API calls
#define HIPRTC_CHECK(call)                                                          \
    do                                                                              \
    {                                                                               \
        hiprtcResult status = (call);                                               \
        if(status != HIPRTC_SUCCESS)                                                \
        {                                                                           \
            throw std::runtime_error(std::string(#call)                             \
                                     + " failed: " + hiprtcGetErrorString(status)); \
        }                                                                           \
    } while(0)

} // namespace hip_plugin
