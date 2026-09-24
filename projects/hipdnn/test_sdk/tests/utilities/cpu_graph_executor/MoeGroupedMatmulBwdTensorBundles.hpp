// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceMoeGroupedMatmulBwd.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>

namespace hipdnn_sdk_test_utils
{

/// Owns every tensor a backward MoE grouped matmul graph needs. NONE-mode
/// routing only -- there is no tokenIndex/tokenKs, so `firstTokenOffsetTensor`
/// always has exactly `experts` rows and group `e` reduces exactly expert `e`.
/// setDefaultRouting() (called by the constructor) fills a deterministic,
/// always-valid FirstTokenOffset that splits tokenRows evenly across experts.
///
/// `dweightTensor` carries its own `DweightType`, independent of the `InputType`
/// shared by doutput and token, so the mixed-dtype plan-builder registrations
/// (half/bfloat16 inputs with a float dweight, and the reverse) are reachable.
///
/// `dweightStrides` selects the DWeight layout; leaving it empty picks the packed
/// column-major layout the node infers for an unset DWeight. Any strides are legal —
/// buildMoeGroupedMatmulBwdGraph() copies whatever this bundle allocated onto the
/// graph tensor, so the two always describe the same memory.
template <typename InputType, typename DweightType = InputType>
struct MoeGroupedMatmulBwdTensorBundle
{
    MoeGroupedMatmulBwdTensorBundle(int64_t experts,
                                    int64_t hiddenK,
                                    int64_t outputN,
                                    int64_t tokenRowsIn,
                                    unsigned int seed
                                    = hipdnn_test_sdk::utilities::getGlobalTestSeed(),
                                    const std::vector<int64_t>& dweightStrides = {})
        : doutputTensor({1, tokenRowsIn, outputN},
                        hipdnn_data_sdk::utilities::generateStrides({1, tokenRowsIn, outputN}))
        , tokenTensor({1, tokenRowsIn, hiddenK},
                      hipdnn_data_sdk::utilities::generateStrides({1, tokenRowsIn, hiddenK}))
        , firstTokenOffsetTensor({experts, 1, 1},
                                 hipdnn_data_sdk::utilities::generateStrides({experts, 1, 1}))
        , dweightTensor({experts, hiddenK, outputN},
                        resolveDweightStrides({experts, hiddenK, outputN}, dweightStrides))
    {
        doutputTensor.fillWithRandomValues(
            static_cast<InputType>(-1.0F), static_cast<InputType>(1.0F), seed);
        tokenTensor.fillWithRandomValues(
            static_cast<InputType>(-1.0F), static_cast<InputType>(1.0F), seed);
        dweightTensor.fillWithValue(static_cast<DweightType>(0.0F));

        setDefaultRouting();
    }

    /// Deterministic, always-valid routing: FirstTokenOffset splits tokenRows
    /// evenly across the `experts` groups.
    void setDefaultRouting()
    {
        const int64_t experts = firstTokenOffsetTensor.dims()[0];
        const int64_t tokenRows = tokenTensor.dims()[1];
        for(int64_t e = 0; e < experts; ++e)
        {
            firstTokenOffsetTensor.setHostValue(
                hipdnn_test_sdk::utilities::defaultMoeGroupedMatmulBwdRoutingOffset(
                    e, tokenRows, experts),
                {e, 0, 0});
        }
    }

    /// Falls back to the packed column-major layout the node infers when a caller
    /// leaves DWeight strides unset.
    static std::vector<int64_t> resolveDweightStrides(const std::vector<int64_t>& dweightDims,
                                                      const std::vector<int64_t>& requestedStrides)
    {
        if(requestedStrides.empty())
        {
            return hipdnn_test_sdk::utilities::moeGroupedMatmulBwdDweightStrides(dweightDims);
        }
        return requestedStrides;
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const hipdnn_frontend::graph::TensorAttributes& doutputAttr,
                          const hipdnn_frontend::graph::TensorAttributes& tokenAttr,
                          const hipdnn_frontend::graph::TensorAttributes& firstTokenOffsetAttr,
                          const hipdnn_frontend::graph::TensorAttributes& dweightAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[doutputAttr.get_uid()] = doutputTensor.memory().hostData();
        variantPack[tokenAttr.get_uid()] = tokenTensor.memory().hostData();
        variantPack[firstTokenOffsetAttr.get_uid()] = firstTokenOffsetTensor.memory().hostData();
        variantPack[dweightAttr.get_uid()] = dweightTensor.memory().hostData();
        return variantPack;
    }

    hipdnn_data_sdk::utilities::Tensor<InputType> doutputTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> tokenTensor;
    hipdnn_data_sdk::utilities::Tensor<int32_t> firstTokenOffsetTensor;
    hipdnn_data_sdk::utilities::Tensor<DweightType> dweightTensor;
};

} // namespace hipdnn_sdk_test_utils
