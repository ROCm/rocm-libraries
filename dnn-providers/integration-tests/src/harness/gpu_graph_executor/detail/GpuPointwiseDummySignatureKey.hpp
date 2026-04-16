// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <ostream>
#include <unordered_map>

#include "GpuPointwiseDummyAddOnePlan.hpp"
#include "IGpuGraphNodePlanBuilder.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// Simplified GPU signature key for pointwise operations.
// Unlike the CPU executor which differentiates by data type and operation mode,
// this GPU version matches all pointwise nodes with a single dummy plan (add-one).
// As real GPU plans are added, this key should be extended to differentiate
// by operation mode and data types, following the CPU PointwiseSignatureKey pattern.
struct GpuPointwiseDummySignatureKey
{
    static constexpr auto NODE_TYPE
        = hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::PointwiseAttributes;

    GpuPointwiseDummySignatureKey() = default;

    GpuPointwiseDummySignatureKey(
        [[maybe_unused]] const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        [[maybe_unused]] const std::unordered_map<
            int64_t,
            const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>& tensorMap)
    {
        // Currently matches all pointwise nodes regardless of type/mode.
        // When real GPU plans are added, extract operation, data types, etc.
    }

    static constexpr std::size_t hashSelf()
    {
        return static_cast<std::size_t>(static_cast<int>(NODE_TYPE));
    }

    std::size_t operator()([[maybe_unused]] const GpuPointwiseDummySignatureKey& k) const noexcept
    {
        return GpuPointwiseDummySignatureKey::hashSelf();
    }

    bool operator==(const GpuPointwiseDummySignatureKey& /*other*/) const noexcept
    {
        return true; // All pointwise nodes match the same key for now
    }

    static std::unordered_map<GpuPointwiseDummySignatureKey,
                              std::unique_ptr<IGpuGraphNodePlanBuilder>,
                              GpuPointwiseDummySignatureKey>
        getPlanBuilders()
    {
        std::unordered_map<GpuPointwiseDummySignatureKey,
                           std::unique_ptr<IGpuGraphNodePlanBuilder>,
                           GpuPointwiseDummySignatureKey>
            map;
        map[GpuPointwiseDummySignatureKey()] = std::make_unique<GpuDummyAddOnePlanBuilder>();
        return map;
    }
};

inline std::ostream& operator<<(std::ostream& os, const GpuPointwiseDummySignatureKey& /*key*/)
{
    os << "GpuPointwiseDummy()";
    return os;
}

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
