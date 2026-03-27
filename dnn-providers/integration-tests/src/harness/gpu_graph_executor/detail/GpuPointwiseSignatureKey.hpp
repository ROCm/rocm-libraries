// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <ostream>
#include <unordered_map>

#include "GpuPointwiseAddOnePlan.hpp"
#include "IGpuGraphNodePlanBuilder.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// Simplified GPU signature key for pointwise operations.
// Unlike the CPU executor which differentiates by data type and operation mode,
// this GPU version matches all pointwise nodes with a single dummy plan (add-one).
// As real GPU plans are added, this key should be extended to differentiate
// by operation mode and data types, following the CPU PointwiseSignatureKey pattern.
struct GpuPointwiseSignatureKey
{
    static constexpr auto nodeType
        = hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes;

    GpuPointwiseSignatureKey() = default;

    GpuPointwiseSignatureKey([[maybe_unused]] const hipdnn_data_sdk::data_objects::Node& node,
                             [[maybe_unused]] const std::unordered_map<
                                 int64_t,
                                 const hipdnn_data_sdk::data_objects::TensorAttributes*>& tensorMap)
    {
        // Currently matches all pointwise nodes regardless of type/mode.
        // When real GPU plans are added, extract operation, data types, etc.
    }

    constexpr std::size_t hashSelf() const
    {
        return static_cast<std::size_t>(static_cast<int>(nodeType));
    }

    std::size_t operator()(const GpuPointwiseSignatureKey& k) const noexcept
    {
        return k.hashSelf();
    }

    bool operator==(const GpuPointwiseSignatureKey& /*other*/) const noexcept
    {
        return true; // All pointwise nodes match the same key for now
    }

    static std::unordered_map<GpuPointwiseSignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuPointwiseSignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuPointwiseSignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuPointwiseSignatureKey>
            map;
        map[GpuPointwiseSignatureKey()] = std::make_unique<GpuAddOnePlanBuilder>();
        return map;
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuPointwiseSignatureKey& /*key*/)
{
    os << "GpuPointwise()";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
