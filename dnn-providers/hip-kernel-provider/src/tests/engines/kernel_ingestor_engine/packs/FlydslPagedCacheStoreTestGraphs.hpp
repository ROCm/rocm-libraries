// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/paged_cache_store_attributes_generated.h>

#include "engines/kernel_ingestor_engine/packs/FlydslRmsNormTestGraphs.hpp"

namespace hip_kernel_provider::kernel_ingestor_engine::testing
{

/// hipkernel:flydsl_paged_cache_store: one pack, the node type its own union arm. The
/// three operand tokens are x, slot_mapping and the container; the scale's is optional.
inline constexpr PackSymbols FLYDSL_PAGED_CACHE_STORE{
    "hipkernel:flydsl_paged_cache_store",
    "hipkernel.flydsl_paged_cache_store.graph_match",
    "",
    "hipkernel.flydsl_paged_cache_store.kernel_match",
    "hipkernel.flydsl_paged_cache_store.score",
    "hipkernel.flydsl_paged_cache_store.dispatch",
    "flydsl_paged_cache_store.x.uid",
    "flydsl_paged_cache_store.slot_mapping.uid",
    "flydsl_paged_cache_store.container.uid"};

constexpr const char* FLYDSL_PCS_SCALE_TOKEN = "flydsl_paged_cache_store.scale.uid";

/// The KMD's fields beside `dtype` (x's element type) and `block_threads`.
constexpr const char* FLYDSL_CACHE_DTYPE_FIELD = "cache_dtype";

/// Tensor uids buildFlydslPagedCacheStoreGraph() uses.
constexpr int64_t FLYDSL_PCS_X_UID = 1;
constexpr int64_t FLYDSL_PCS_SLOT_UID = 2;
constexpr int64_t FLYDSL_PCS_CONTAINER_UID = 3;
constexpr int64_t FLYDSL_PCS_SCALE_UID = 4;

/// How the graph carries its scale.
enum class PcsScale
{
    NONE,
    DEVICE,
    BAKED,
    RUNTIME
};

/**
 * @brief One paged-cache-store graph, parameterized on everything the matcher gates.
 *
 * Defaults to token-packed bf16 x [T, H, D] = [5, 2, 64] with INT32 slots into a packed
 * bf16 container [4, 2, 8, 64].
 */
struct PagedCacheStoreGraphSpec
{
    hipdnn_flatbuffers_sdk::data_objects::DataType xDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16;
    /// The container's dtype; unset means x's.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> containerDataType;
    hipdnn_flatbuffers_sdk::data_objects::DataType slotDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::INT32;
    std::vector<int64_t> xDims{5, 2, 64};
    /// Unset means packed for the dims.
    std::optional<std::vector<int64_t>> xStrides;
    /// Unset means [T] for rank-3 x, [B, S] for rank-4.
    std::optional<std::vector<int64_t>> slotDims;
    std::optional<std::vector<int64_t>> slotStrides;
    std::vector<int64_t> containerDims{4, 2, 8, 64};
    std::optional<std::vector<int64_t>> containerStrides;
    PcsScale scale = PcsScale::NONE;
    float bakedScale = 1.0F;
    hipdnn_flatbuffers_sdk::data_objects::DataType scaleDataType
        = hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT;
    std::vector<int64_t> scaleDims{1};
    std::optional<hipdnn_flatbuffers_sdk::data_objects::DataType> computeDataType;
    bool virtualContainer = false;
    bool secondNode = false;
    bool overrideShapeEnabled = false;
};

inline std::vector<int64_t> pcsSlotDims(const PagedCacheStoreGraphSpec& spec)
{
    if(spec.slotDims.has_value())
    {
        return *spec.slotDims;
    }
    return spec.xDims.size() == 3 ? std::vector<int64_t>{spec.xDims[0]}
                                  : std::vector<int64_t>{spec.xDims[0], spec.xDims[2]};
}

inline flatbuffers::FlatBufferBuilder
    buildFlydslPagedCacheStoreGraph(const PagedCacheStoreGraphSpec& spec = {})
{
    namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;
    flatbuffers::FlatBufferBuilder builder;

    const auto xStrides = spec.xStrides.value_or(flydslPackedStrides(spec.xDims));
    const auto slotDims = pcsSlotDims(spec);
    const auto slotStrides = spec.slotStrides.value_or(flydslPackedStrides(slotDims));
    const auto containerStrides
        = spec.containerStrides.value_or(flydslPackedStrides(spec.containerDims));

    std::vector<flatbuffers::Offset<data_objects::TensorAttributes>> tensors;
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, FLYDSL_PCS_X_UID, "x", spec.xDataType, &xStrides, &spec.xDims));
    tensors.push_back(data_objects::CreateTensorAttributesDirect(
        builder, FLYDSL_PCS_SLOT_UID, "slot_mapping", spec.slotDataType, &slotStrides, &slotDims));
    tensors.push_back(
        data_objects::CreateTensorAttributesDirect(builder,
                                                   FLYDSL_PCS_CONTAINER_UID,
                                                   "container",
                                                   spec.containerDataType.value_or(spec.xDataType),
                                                   &containerStrides,
                                                   &spec.containerDims,
                                                   spec.virtualContainer));
    if(spec.scale != PcsScale::NONE)
    {
        const auto scaleStrides = flydslPackedStrides(spec.scaleDims);
        const bool baked = spec.scale == PcsScale::BAKED;
        const data_objects::Float32Value value(spec.bakedScale);
        tensors.push_back(data_objects::CreateTensorAttributesDirect(
            builder,
            FLYDSL_PCS_SCALE_UID,
            "scale",
            spec.scaleDataType,
            &scaleStrides,
            &spec.scaleDims,
            false,
            baked ? data_objects::TensorValue::Float32Value : data_objects::TensorValue::NONE,
            baked ? builder.CreateStruct(value).Union() : flatbuffers::Offset<void>(0),
            spec.scale == PcsScale::RUNTIME));
    }

    std::vector<flatbuffers::Offset<data_objects::Node>> nodes;
    for(int node = 0; node < (spec.secondNode ? 2 : 1); ++node)
    {
        auto attributes = data_objects::CreatePagedCacheStoreAttributes(
            builder,
            FLYDSL_PCS_X_UID,
            FLYDSL_PCS_SLOT_UID,
            FLYDSL_PCS_CONTAINER_UID,
            spec.scale == PcsScale::NONE ? flatbuffers::nullopt
                                         : flatbuffers::Optional<int64_t>(FLYDSL_PCS_SCALE_UID));
        nodes.push_back(data_objects::CreateNodeDirect(
            builder,
            "paged_cache_store",
            spec.computeDataType.value_or(data_objects::DataType::FLOAT),
            data_objects::NodeAttributes::PagedCacheStoreAttributes,
            attributes.Union()));
    }

    auto name = builder.CreateString("flydsl_paged_cache_store_test");
    auto tensorsVector = builder.CreateVector(tensors);
    auto nodesVector = builder.CreateVector(nodes);
    data_objects::GraphBuilder graphBuilder(builder);
    graphBuilder.add_name(name);
    graphBuilder.add_tensors(tensorsVector);
    graphBuilder.add_nodes(nodesVector);
    graphBuilder.add_is_override_shape_enabled(spec.overrideShapeEnabled);
    builder.Finish(graphBuilder.Finish());
    return builder;
}

} // namespace hip_kernel_provider::kernel_ingestor_engine::testing

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
