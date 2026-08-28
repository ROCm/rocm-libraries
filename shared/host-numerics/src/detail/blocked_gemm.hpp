// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <roc/host_numerics/gemm.hpp>

namespace roc::host_numerics::detail {
GemmSupportInfo queryBlockedGemmSupport(const GemmRequest& request);
bool isBlockedGemmPreferredForAutomaticExecution(const GemmRequest& request);
GemmRunInfo runBlockedGemm(const GemmRequest& request);
}  // namespace roc::host_numerics::detail
