// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>
#include <type_traits>
#include <variant>

#include "GpuBatchnormFwdInfSignatureKey.hpp"
#include "GpuBatchnormFwdInfVarianceSignatureKey.hpp"
#include "GpuBatchnormFwdTrainSignatureKey.hpp"
#include "GpuConvolutionFwdSignatureKey.hpp"
#include "GpuLayernormBwdSignatureKey.hpp"
#include "GpuLayernormFwdSignatureKey.hpp"
#include "GpuMatmulSignatureKey.hpp"
#include "GpuPointwiseSignatureKey.hpp"
#include "GpuRMSNormSignatureKey.hpp"
#include "GpuReductionSignatureKey.hpp"
#include "GpuResampleSignatureKey.hpp"
#include "GpuSdpaFwdSignatureKey.hpp"
#include "GpuSdpaRaggedFwdSignatureKey.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// Variant of all GPU plan signature key types.
// Add new signature key types here as GPU plans are implemented.
using GpuPlanRegistrySignatureKey = std::variant<GpuBatchnormFwdTrainSignatureKey,
                                                 GpuConvolutionFwdSignatureKey,
                                                 GpuLayernormFwdSignatureKey,
                                                 GpuLayernormBwdSignatureKey,
                                                 GpuMatmulSignatureKey,
                                                 GpuSdpaFwdSignatureKey,
                                                 GpuSdpaRaggedFwdSignatureKey,
                                                 GpuPointwiseSignatureKey,
                                                 GpuReductionSignatureKey,
                                                 GpuResampleFwdSignatureKey,
                                                 GpuResampleBwdSignatureKey,
                                                 GpuRMSNormFwdSignatureKey,
                                                 GpuRMSNormBwdSignatureKey,
                                                 GpuBatchnormFwdInfSignatureKey,
                                                 GpuBatchnormFwdInfVarianceSignatureKey>;

struct GpuPlanRegistrySignatureKeyHash
{
    std::size_t operator()(const GpuPlanRegistrySignatureKey& k) const
    {
        return std::visit([](const auto& x) { return x.hashSelf(); }, k);
    }
};

struct GpuPlanRegistrySignatureKeyEqual
{
    // Compare the active alternatives' indices first, then visit only `a` and fetch the matching
    // alternative from `b`. Visiting both variants would instantiate one visitor call per pair of
    // alternatives (N*N) instead of N.
    bool operator()(const GpuPlanRegistrySignatureKey& a,
                    const GpuPlanRegistrySignatureKey& b) const
    {
        if(a.index() != b.index())
        {
            return false;
        }

        return std::visit(
            [&b](const auto& x) { return x == std::get<std::decay_t<decltype(x)>>(b); }, a);
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuPlanRegistrySignatureKey& key)
{
    std::visit([&os](const auto& arg) { os << arg; }, key);
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
