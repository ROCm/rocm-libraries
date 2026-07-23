#pragma once

#include "hipconv/conv2d_params.hpp"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <stdexcept>
#include <string>

namespace hipconv::pointwise
{

// Thrown when a hipBLASLt API call returns a non-success status. Derives from
// std::runtime_error so it is still caught by generic handlers, while carrying
// the raw status code for callers that want to react in a targeted way (e.g.
// treat HIPBLAS_STATUS_NOT_SUPPORTED as a graceful fallback vs. a hard error).
class HipblasltError : public std::runtime_error
{
public:
    HipblasltError(hipblasStatus_t status, const char* where)
        : std::runtime_error(std::string("hipBLASLt error in ") + where + ": " +
                             std::to_string(static_cast<int>(status)))
        , status_(status)
    {
    }

    hipblasStatus_t status() const noexcept { return status_; }

private:
    hipblasStatus_t status_;
};

void launch_pointwise_gemm(const Conv2dParams& par,
                           const void* in,
                           const void* wei,
                           void* out,
                           hipStream_t stream);

} // namespace hipconv::pointwise
