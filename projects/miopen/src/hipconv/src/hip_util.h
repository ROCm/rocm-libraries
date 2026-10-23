#pragma once

#include <hip/hip_runtime.h>
#include <sstream>
#include <stdexcept>

// Thrown when a HIP call fails.
//
// A distinct type carrying the hipError_t so callers can distinguish a HIP
// failure from an ordinary host-side exception such as std::bad_alloc, and
// branch on the specific code.
struct HipError : std::runtime_error
{
    hipError_t code;
    HipError(hipError_t code_, std::string message)
        : std::runtime_error(std::move(message))
        , code(code_)
    {
    }
};

inline void hip_check(hipError_t err, const char* file, int line)
{
    if(err != hipSuccess)
    {
        std::ostringstream s;
        s << "HIP error at " << file << ":" << line << ": " << hipGetErrorString(err);
        throw HipError(err, s.str());
    }
}

#define HIP_CHECK(call) hip_check(call, __FILE__, __LINE__)

// Compute-unit count of the active device, queried once. Callers size grids with it, so a
// failed query falls back to an MI300-class 256 rather than throwing: a wrong count costs
// occupancy, an exception costs the launch. The cache pins the device of the first call.
inline int cu_count()
{
    static const int cu = [] {
        int dev = 0;
        if(hipGetDevice(&dev) != hipSuccess)
            return 256;
        hipDeviceProp_t props{};
        if(hipGetDeviceProperties(&props, dev) != hipSuccess || props.multiProcessorCount <= 0)
            return 256;
        return props.multiProcessorCount;
    }();
    return cu;
}
