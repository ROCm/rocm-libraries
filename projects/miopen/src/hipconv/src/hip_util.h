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
