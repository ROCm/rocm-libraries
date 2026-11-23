// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

/**
 * @file PackGraph.hpp
 * @brief Graph-reading helpers any native pack needs, whatever operation it serves.
 *
 * Operation-specific readers build on these (SdpaGraphMatch.hpp for SDPA forward), so a
 * conv or pointwise pack can read the graph without pulling in another op's schema.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::pack_graph
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
using hipdnn_plugin_sdk::ingestor::MatchContext;

/// The tensor the graph holds under @p uid, or nullptr if it has none.
inline const data_objects::TensorAttributes* findTensor(const MatchContext& context, int64_t uid)
{
    const auto& tensors = context.graph.getTensorMap();
    auto it = tensors.find(uid);
    return it == tensors.end() ? nullptr : it->second;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::pack_graph
