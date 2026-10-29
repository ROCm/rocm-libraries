// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <functional>
#include <ostream>
#include <type_traits>
#include <variant>

#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormBwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferenceSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormFwdInferenceWithVarianceSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BatchnormTrainSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleDequantizeSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/BlockScaleQuantizeSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionBwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionFwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ConvolutionWrwSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/LayernormBpropSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/LayernormFpropSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MatmulSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MoeGroupedMatmulBwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/MoeGroupedMatmulSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PointwiseSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RMSNormBwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/RMSNormFwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ReductionSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ResampleBwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/ResampleFwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaBwdSignatureKey.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/SdpaFwdSignatureKey.hpp>

namespace hipdnn_test_sdk::detail
{

/*
 * For each new op we add to our Plan registry we need to update this variant key to support it.
 * This way, we can have a single registry for all operations which simplifies the graph executor.
 * Each key must have a
 *  - hashSelf() method
 *  - equality operator
 *  - hash operator
 *  - Constructor to build the key from a data_object::Node && tensorMap
 *  - A static method getPlanBuilders() which returns a map of keys to plan builders for that key
 *
*/
using PlanRegistrySignatureKey = std::variant<BatchnormFwdInferenceSignatureKey,
                                              BatchnormFwdInferenceWithVarianceSignatureKey,
                                              BatchnormBwdSignatureKey,
                                              BatchnormTrainSignatureKey,
                                              BlockScaleQuantizeSignatureKey,
                                              BlockScaleDequantizeSignatureKey,
                                              ConvolutionFwdSignatureKey,
                                              ConvolutionBwdSignatureKey,
                                              ConvolutionWrwSignatureKey,
                                              LayernormFpropSignatureKey,
                                              LayernormBpropSignatureKey,
                                              MatmulSignatureKey,
                                              MoeGroupedMatmulSignatureKey,
                                              MoeGroupedMatmulBwdSignatureKey,
                                              PointwiseSignatureKey,
                                              RMSNormFwdSignatureKey,
                                              RMSNormBwdSignatureKey,
                                              ResampleFwdSignatureKey,
                                              ResampleBwdSignatureKey,
                                              SdpaBwdSignatureKey,
                                              SdpaFwdSignatureKey,
                                              ReductionSignatureKey>;

struct PlanRegistrySignatureKeyHash
{
    std::size_t operator()(const PlanRegistrySignatureKey& k) const
    {
        return std::visit([](const auto& x) { return x.hashSelf(); }, k);
    }
};

struct PlanRegistrySignatureKeyEqual
{
    // Compare the active alternatives' indices first, then visit only `a` and fetch the matching
    // alternative from `b`. Visiting both variants would instantiate one visitor call per pair of
    // alternatives (N*N) instead of N.
    bool operator()(const PlanRegistrySignatureKey& a, const PlanRegistrySignatureKey& b) const
    {
        if(a.index() != b.index())
        {
            return false;
        }

        return std::visit(
            [&b](const auto& x) { return x == std::get<std::decay_t<decltype(x)>>(b); }, a);
    }
};

inline std::ostream& operator<<(std::ostream& os, const PlanRegistrySignatureKey& key)
{
    std::visit([&os](const auto& arg) { os << arg; }, key);
    return os;
}

} // namespace hipdnn_test_sdk::detail
