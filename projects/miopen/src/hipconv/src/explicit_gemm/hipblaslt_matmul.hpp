#pragma once

#include "hip_util.h"
#include "hipconv/conv_params.hpp"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <string>

namespace hipconv::explicit_gemm
{

// Thrown when a hipBLASLt API call returns a non-success status.
//
// A HipError, so launch() returns it as hipErrorUnknown; hipErrorNotSupported is reserved for
// a GEMM hipBLASLt has no algorithm for. status() carries hipBLASLt's own code.
class HipblasltError : public HipError
{
public:
    HipblasltError(hipblasStatus_t status, const char* where)
        : HipError(hipErrorUnknown,
                   std::string("hipBLASLt error in ") + where + ": " +
                       std::to_string(static_cast<int>(status)))
        , status_(status)
    {
    }

    hipblasStatus_t status() const noexcept { return status_; }

private:
    hipblasStatus_t status_;
};

// Whether hipBLASLt has an algorithm on the current device for the GEMM launch_gemm runs
// for `par` (#244).
bool has_algorithm(const ConvParams& par);

void launch_gemm(const ConvParams& par,
                 const void* in,
                 const void* wei,
                 void* out,
                 hipStream_t stream);

} // namespace hipconv::explicit_gemm
