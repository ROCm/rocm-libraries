// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "gemm_invocation.hpp"

namespace roc::host_numerics::detail {
GemmSupportInfo queryBlockedGemmSupport(const GemmInvocation& request);
bool isBlockedGemmPreferredForAutomaticExecution(const GemmInvocation& request);
GemmExecutionInfo runBlockedGemm(const GemmInvocation& request);
GemmExecutionInfo runBlockedGemmToSelectedOutput(const GemmInvocation& request,
                                                 Tensor& selectedOutput);
}  // namespace roc::host_numerics::detail
