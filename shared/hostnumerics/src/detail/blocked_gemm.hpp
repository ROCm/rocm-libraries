// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "gemm_invocation.hpp"

namespace roc::hostnumerics::detail {
GemmSupportInfo queryBlockedGemmSupport(const GemmInvocation& request);
GemmExecutionInfo runBlockedGemm(const GemmInvocation& request);
GemmExecutionInfo runBlockedGemmToSelectedOutput(const GemmInvocation& request,
                                                 Tensor& selectedOutput);
}  // namespace roc::hostnumerics::detail
