#pragma once

#include <hip/hip_runtime.h>

#include <memory>

namespace hipconv::explicit_gemm
{
class GemmPlan;
}

struct LaunchParams
{
    dim3 grid;
    dim3 block_size;
    size_t dynamic_shared_bytes = 0;
    // The GEMM an explicit-GEMM launch runs, null for every other kernel. A ConvLaunch keeps it,
    // so its launches resolve the hipBLASLt algorithm once per device; launch() makes a new one
    // per call.
    std::shared_ptr<hipconv::explicit_gemm::GemmPlan> gemm_plan;
    // A kernel's plan for the launch, of a type the kernel defines; null if it has none.
    std::shared_ptr<const void> kernel_state;
};
