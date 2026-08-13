// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <optional>

#include "SdpaTensorBundles.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>

namespace hipdnn_sdk_test_utils
{

template <typename InputType>
std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>, std::unordered_map<int64_t, void*>>
    buildSdpaFwdGraph(
        SdpaFwdTensorBundle<InputType>& tensorBundle,
        hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
        bool causalMask = false,
        bool causalMaskBottomRight = false,
        std::optional<int64_t> leftBound = std::nullopt,
        std::optional<int64_t> rightBound = std::nullopt,
        hipdnn_frontend::DiagonalAlignment diagonalAlignment
        = hipdnn_frontend::DiagonalAlignment::TOP_LEFT,
        bool alibiMask = false,
        // NOLINT false positive: const float* does not convert to the void* variant pack.
        float* runtimeScaleHostPtr = nullptr) // NOLINT(readability-non-const-parameter)
{
    const auto frontendDataType = hipdnn_test_sdk::utilities::sdkToFrontendDataType(dataType);

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("SdpaFwdTest");
    graph->set_io_data_type(frontendDataType)
        .set_compute_data_type(frontendDataType)
        .set_intermediate_data_type(frontendDataType);

    int64_t uid = 1;
    auto qAttr
        = hipdnn_frontend::graph::makeTensorAttributes("Q", frontendDataType, tensorBundle.qTensor);
    qAttr.set_uid(uid++);
    auto qTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(qAttr));

    auto kAttr
        = hipdnn_frontend::graph::makeTensorAttributes("K", frontendDataType, tensorBundle.kTensor);
    kAttr.set_uid(uid++);
    auto kTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(kAttr));

    auto vAttr
        = hipdnn_frontend::graph::makeTensorAttributes("V", frontendDataType, tensorBundle.vTensor);
    vAttr.set_uid(uid++);
    auto vTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(vAttr));

    hipdnn_frontend::graph::SdpaAttributes sdpaAttrs;
    sdpaAttrs.set_name("SdpaFwd");
    sdpaAttrs.set_causal_mask(causalMask);
    sdpaAttrs.set_causal_mask_bottom_right(causalMaskBottomRight);
    sdpaAttrs.set_alibi_mask(alibiMask);
    sdpaAttrs.set_diagonal_alignment(diagonalAlignment);
    if(leftBound.has_value())
    {
        sdpaAttrs.set_diagonal_band_left_bound(leftBound.value());
    }
    if(rightBound.has_value())
    {
        sdpaAttrs.set_diagonal_band_right_bound(rightBound.value());
    }

    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> scaleTensorAttr;
    if(runtimeScaleHostPtr != nullptr)
    {
        // Pure runtime pass-by-value scale: FLOAT scalar, no baked value; the
        // host value is delivered through the variant pack at execute.
        scaleTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
        scaleTensorAttr->set_uid(uid++)
            .set_name("ScaleTensor")
            .set_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_dim({1})
            .set_stride({1})
            .set_as_runtime_parameter();
        sdpaAttrs.set_attn_scale(scaleTensorAttr);
    }

    auto [oTensorAttr, statsAttr] = graph->sdpa(qTensorAttr, kTensorAttr, vTensorAttr, sdpaAttrs);

    if(!oTensorAttr->has_uid())
    {
        oTensorAttr->set_uid(uid++);
    }

    const auto oDims = tensorBundle.oTensor.dims();
    const auto oStrides = hipdnn_data_sdk::utilities::generateStrides(oDims);
    oTensorAttr->set_data_type(frontendDataType)
        .set_dim(oDims)
        .set_stride(oStrides)
        .set_is_virtual(false);

    auto variantPack
        = tensorBundle.createVariantPack(*qTensorAttr, *kTensorAttr, *vTensorAttr, *oTensorAttr);
    if(scaleTensorAttr)
    {
        variantPack[scaleTensorAttr->get_uid()] = runtimeScaleHostPtr;
    }

    return std::make_tuple(graph, variantPack);
}

template <typename InputType>
std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>, std::unordered_map<int64_t, void*>>
    buildSdpaBwdGraph(
        SdpaBwdTensorBundle<InputType>& tensorBundle,
        hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
        // NOLINT false positive: const float* does not convert to the void* variant pack.
        float* runtimeScaleHostPtr = nullptr) // NOLINT(readability-non-const-parameter)
{
    const auto frontendDataType = hipdnn_test_sdk::utilities::sdkToFrontendDataType(dataType);

    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();
    graph->set_name("SdpaBwdTest");
    graph->set_io_data_type(frontendDataType)
        .set_compute_data_type(frontendDataType)
        .set_intermediate_data_type(frontendDataType);

    int64_t uid = 1;
    auto qAttr
        = hipdnn_frontend::graph::makeTensorAttributes("Q", frontendDataType, tensorBundle.qTensor);
    qAttr.set_uid(uid++);
    auto qTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(qAttr));

    auto kAttr
        = hipdnn_frontend::graph::makeTensorAttributes("K", frontendDataType, tensorBundle.kTensor);
    kAttr.set_uid(uid++);
    auto kTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(kAttr));

    auto vAttr
        = hipdnn_frontend::graph::makeTensorAttributes("V", frontendDataType, tensorBundle.vTensor);
    vAttr.set_uid(uid++);
    auto vTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(vAttr));

    auto oAttr
        = hipdnn_frontend::graph::makeTensorAttributes("O", frontendDataType, tensorBundle.oTensor);
    oAttr.set_uid(uid++);
    auto oTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(oAttr));

    auto doAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "dO", frontendDataType, tensorBundle.doTensor);
    doAttr.set_uid(uid++);
    auto doTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(doAttr));

    auto statsAttr = hipdnn_frontend::graph::makeTensorAttributes(
        "Stats", hipdnn_frontend::DataType::FLOAT, tensorBundle.statsTensor);
    statsAttr.set_uid(uid++);
    auto statsTensorAttr
        = std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(statsAttr));

    hipdnn_frontend::graph::SdpaBackwardAttributes sdpaBwdAttrs;
    sdpaBwdAttrs.set_name("SdpaBwd");

    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> scaleTensorAttr;
    if(runtimeScaleHostPtr != nullptr)
    {
        // Pure runtime pass-by-value scale: FLOAT scalar, no baked value; the
        // host value is delivered through the variant pack at execute.
        scaleTensorAttr = std::make_shared<hipdnn_frontend::graph::TensorAttributes>();
        scaleTensorAttr->set_uid(uid++)
            .set_name("ScaleTensor")
            .set_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_dim({1})
            .set_stride({1})
            .set_as_runtime_parameter();
        sdpaBwdAttrs.set_attn_scale(scaleTensorAttr);
    }

    auto [dqTensorAttr, dkTensorAttr, dvTensorAttr] = graph->sdpa_backward(qTensorAttr,
                                                                           kTensorAttr,
                                                                           vTensorAttr,
                                                                           oTensorAttr,
                                                                           doTensorAttr,
                                                                           statsTensorAttr,
                                                                           sdpaBwdAttrs);

    if(!dqTensorAttr->has_uid())
    {
        dqTensorAttr->set_uid(uid++);
    }
    dqTensorAttr->set_data_type(frontendDataType)
        .set_dim(tensorBundle.dqTensor.dims())
        .set_stride(hipdnn_data_sdk::utilities::generateStrides(tensorBundle.dqTensor.dims()))
        .set_is_virtual(false);

    if(!dkTensorAttr->has_uid())
    {
        dkTensorAttr->set_uid(uid++);
    }
    dkTensorAttr->set_data_type(frontendDataType)
        .set_dim(tensorBundle.dkTensor.dims())
        .set_stride(hipdnn_data_sdk::utilities::generateStrides(tensorBundle.dkTensor.dims()))
        .set_is_virtual(false);

    if(!dvTensorAttr->has_uid())
    {
        dvTensorAttr->set_uid(uid++);
    }
    dvTensorAttr->set_data_type(frontendDataType)
        .set_dim(tensorBundle.dvTensor.dims())
        .set_stride(hipdnn_data_sdk::utilities::generateStrides(tensorBundle.dvTensor.dims()))
        .set_is_virtual(false);

    auto variantPack = tensorBundle.createVariantPack(*qTensorAttr,
                                                      *kTensorAttr,
                                                      *vTensorAttr,
                                                      *oTensorAttr,
                                                      *doTensorAttr,
                                                      *statsTensorAttr,
                                                      *dqTensorAttr,
                                                      *dkTensorAttr,
                                                      *dvTensorAttr);
    if(scaleTensorAttr)
    {
        variantPack[scaleTensorAttr->get_uid()] = runtimeScaleHostPtr;
    }

    return std::make_tuple(graph, variantPack);
}

} // namespace hipdnn_sdk_test_utils
