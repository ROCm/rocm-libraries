// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cassert>
#include <optional>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>

namespace hipdnn_sdk_test_utils
{

/// Owns every tensor a forward MoE grouped matmul graph needs, sized for one
/// mode/shape configuration. Routing tensors (`tokenIndex`/`tokenKs`) are only
/// allocated for the modes that use them. `setDefaultRouting()` (called by the
/// constructor) fills a deterministic, always-valid routing.
template <typename InputType>
struct MoeGroupedMatmulTensorBundle
{
    MoeGroupedMatmulTensorBundle(int64_t experts,
                                 int64_t hiddenK,
                                 int64_t weightN,
                                 int64_t tokenRowsIn,
                                 int64_t routedRowsIn, // == tokenRows for NONE
                                 hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode modeIn,
                                 int32_t topKIn,
                                 unsigned int seed
                                 = hipdnn_test_sdk::utilities::getGlobalTestSeed())
        : tokenTensor({1, tokenRowsIn, hiddenK},
                      hipdnn_data_sdk::utilities::generateStrides({1, tokenRowsIn, hiddenK}))
        , weightTensor({experts, hiddenK, weightN},
                       hipdnn_data_sdk::utilities::generateStrides({experts, hiddenK, weightN}))
        , firstTokenOffsetTensor({experts, 1, 1},
                                 hipdnn_data_sdk::utilities::generateStrides({experts, 1, 1}))
        , tokenIndexTensor(
              modeIn != Mode::NONE
                  ? std::make_optional(hipdnn_data_sdk::utilities::Tensor<int32_t>(
                        {1, routedRowsIn, 1},
                        hipdnn_data_sdk::utilities::generateStrides({1, routedRowsIn, 1})))
                  : std::nullopt)
        , tokenKsTensor(
              modeIn == Mode::SCATTER
                  ? std::make_optional(hipdnn_data_sdk::utilities::Tensor<int32_t>(
                        {1, routedRowsIn, 1},
                        hipdnn_data_sdk::utilities::generateStrides({1, routedRowsIn, 1})))
                  : std::nullopt)
        , outputTensor({1, (modeIn == Mode::GATHER) ? routedRowsIn : tokenRowsIn, weightN},
                       hipdnn_data_sdk::utilities::generateStrides(
                           {1, (modeIn == Mode::GATHER) ? routedRowsIn : tokenRowsIn, weightN}))
        , mode(modeIn)
        , topK(topKIn)
        , tokenRows(tokenRowsIn)
        , routedRows(routedRowsIn)
    {
        tokenTensor.fillWithRandomValues(
            static_cast<InputType>(0.0F), static_cast<InputType>(1.0F), seed);
        weightTensor.fillWithRandomValues(
            static_cast<InputType>(0.0F), static_cast<InputType>(1.0F), seed);
        outputTensor.fillWithValue(static_cast<InputType>(0.0F));

        setDefaultRouting();
    }

    /// Deterministic, always-valid routing: FirstTokenOffset splits rowsTotal
    /// evenly across the G groups; GATHER cycles through the token rows;
    /// SCATTER installs the identity permutation (every destination hit once).
    void setDefaultRouting()
    {
        const int64_t rowsTotal = (mode == Mode::GATHER) ? routedRows : tokenRows;
        const int64_t groupCount = firstTokenOffsetTensor.dims()[0];
        for(int64_t g = 0; g < groupCount; ++g)
        {
            firstTokenOffsetTensor.setHostValue(static_cast<int32_t>((g * rowsTotal) / groupCount),
                                                {g, 0, 0});
        }

        if(mode == Mode::GATHER)
        {
            for(int64_t r = 0; r < routedRows; ++r)
            {
                tokenIndexTensor->setHostValue(static_cast<int32_t>(r % tokenRows), {0, r, 0});
            }
        }
        else if(mode == Mode::SCATTER)
        {
            assert(routedRows == tokenRows);
            assert(topK > 0 && tokenRows % topK == 0);
            for(int64_t r = 0; r < routedRows; ++r)
            {
                tokenIndexTensor->setHostValue(static_cast<int32_t>(r / topK), {0, r, 0});
                tokenKsTensor->setHostValue(static_cast<int32_t>(r % topK), {0, r, 0});
            }
        }
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const hipdnn_frontend::graph::TensorAttributes& tokenAttr,
                          const hipdnn_frontend::graph::TensorAttributes& weightAttr,
                          const hipdnn_frontend::graph::TensorAttributes& firstTokenOffsetAttr,
                          const hipdnn_frontend::graph::TensorAttributes* tokenIndexAttr,
                          const hipdnn_frontend::graph::TensorAttributes* tokenKsAttr,
                          const hipdnn_frontend::graph::TensorAttributes& outputAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[tokenAttr.get_uid()] = tokenTensor.memory().hostData();
        variantPack[weightAttr.get_uid()] = weightTensor.memory().hostData();
        variantPack[firstTokenOffsetAttr.get_uid()] = firstTokenOffsetTensor.memory().hostData();
        if(tokenIndexAttr != nullptr)
        {
            variantPack[tokenIndexAttr->get_uid()] = tokenIndexTensor->memory().hostData();
        }
        if(tokenKsAttr != nullptr)
        {
            variantPack[tokenKsAttr->get_uid()] = tokenKsTensor->memory().hostData();
        }
        variantPack[outputAttr.get_uid()] = outputTensor.memory().hostData();
        return variantPack;
    }

    hipdnn_data_sdk::utilities::Tensor<InputType> tokenTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> weightTensor;
    hipdnn_data_sdk::utilities::Tensor<int32_t> firstTokenOffsetTensor;
    std::optional<hipdnn_data_sdk::utilities::Tensor<int32_t>> tokenIndexTensor;
    std::optional<hipdnn_data_sdk::utilities::Tensor<int32_t>> tokenKsTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> outputTensor;

    hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode mode;
    int32_t topK;
    int64_t tokenRows;
    int64_t routedRows;

private:
    using Mode = hipdnn_flatbuffers_sdk::data_objects::MoeGroupedMatmulMode;
};

} // namespace hipdnn_sdk_test_utils
