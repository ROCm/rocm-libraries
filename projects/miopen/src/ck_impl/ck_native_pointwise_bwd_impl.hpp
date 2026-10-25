// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

namespace miopen {
struct ExecutionContext;
namespace conv {
struct ProblemDescription;
}
namespace solver {
struct ConvSolution;
namespace conv {

std::vector<std::string> NativePointwiseBwdKernels(const miopen::conv::ProblemDescription& problem);
bool IsNativePointwiseBwdKernel(const std::string& kernel_id);
bool IsNativePointwiseBwdSupported(const miopen::conv::ProblemDescription& problem,
                                   const std::string& kernel_id);
ConvSolution NativePointwiseBwdSolution(const ExecutionContext& ctx,
                                        const miopen::conv::ProblemDescription& problem,
                                        const std::string& kernel_id);

} // namespace conv
} // namespace solver
} // namespace miopen
