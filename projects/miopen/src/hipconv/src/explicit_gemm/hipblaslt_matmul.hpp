#pragma once

#include "hip_util.h"
#include "hipconv/conv_params.hpp"

#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include <cstddef>
#include <memory>
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

// The hipBLASLt workspace a wgrad GEMM launch takes, and the most its algorithm may use.
//
// A constant, so the size a caller allocates depends on no device. It covers every gfx950 top
// pick #282 found, whose Stream-K workspaces hold fp32 partial tiles.
inline constexpr std::size_t GEMM_WORKSPACE_BYTES = std::size_t{128} << 20;

// The hipBLASLt workspace a GEMM in `direction` takes; none for fprop and dgrad.
//
// docs/explicit-gemm-workspace.md: hipBLASLt before rocm-libraries #11691 can return wrong
// results from Stream-K's fix-up on gfx950.
constexpr std::size_t gemm_workspace_bytes(Direction direction)
{
    return direction == Direction::Wgrad ? GEMM_WORKSPACE_BYTES : 0;
}

// The GEMM a layer runs, with the algorithm hipBLASLt chose for it on each device it launched on.
class GemmPlan;

// The plan for `par`'s GEMM, before any launch; queries no device.
std::shared_ptr<GemmPlan> make_gemm_plan(const ConvParams& par);

// Launch `plan`'s GEMM. The first launch on the stream's device resolves its algorithm there.
//
// `workspace` holds gemm_workspace_bytes of the layer's direction, and may be null if that is 0.
//
// Deterministic on one device; the explicit-GEMM kernels keep ConvKernel::is_deterministic's true.
//
// hipBLASLt's gfx950 and gfx1250 kernels add the partial sums over K in a fixed order, without
// atomics. hipBLASLt does not guarantee this; spec_driven_test relaunches each case to check.
void launch_gemm(GemmPlan& plan,
                 const void* in,
                 const void* wei,
                 void* out,
                 void* workspace,
                 hipStream_t stream);

} // namespace hipconv::explicit_gemm
