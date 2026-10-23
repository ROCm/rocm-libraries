// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "SdpaTensorBundles.hpp"
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/SdkFrontendTypeConversions.hpp>

namespace hipdnn_sdk_test_utils
{

namespace detail
{
// Declares the bundle's mask as a graph tensor. The graph dtype comes from MaskT, the
// element type of the buffer that is bound, so the two cannot disagree.
template <class MaskT>
std::shared_ptr<hipdnn_frontend::graph::TensorAttributes>
    makeSdpaMaskAttributes(const hipdnn_data_sdk::utilities::Tensor<MaskT>& maskTensor, int64_t uid)
{
    const auto maskDataType = hipdnn_test_sdk::utilities::sdkToFrontendDataType(
        hipdnn_test_sdk::utilities::nativeTypeToDataType<MaskT>());
    auto maskAttr = hipdnn_frontend::graph::makeTensorAttributes("Mask", maskDataType, maskTensor);
    maskAttr.set_uid(uid);
    return std::make_shared<hipdnn_frontend::graph::TensorAttributes>(std::move(maskAttr));
}
} // namespace detail

// When `generateStats` is true, the SDPA node is configured to emit its softmax
// log-sum-exp (LSE) statistics tensor, shape [B, H, Sq, 1], FLOAT typed and non-virtual,
// so it becomes a bindable graph output exercising the LSE/stats path. The stats tensor's
// uid is discoverable from the serialized node's stats_tensor_uid(); callers that bind it
// must allocate a matching FLOAT buffer and add it to the variant pack themselves.
template <typename InputType, class MaskT>
std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>, std::unordered_map<int64_t, void*>>
    buildSdpaFwdGraph(SdpaFwdTensorBundle<InputType, MaskT>& tensorBundle,
                      hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                      bool causalMask = false,
                      bool causalMaskBottomRight = false,
                      std::optional<int64_t> leftBound = std::nullopt,
                      std::optional<int64_t> rightBound = std::nullopt,
                      hipdnn_frontend::DiagonalAlignment diagonalAlignment
                      = hipdnn_frontend::DiagonalAlignment::TOP_LEFT,
                      bool alibiMask = false,
                      bool generateStats = false,
                      // Non-const: the variant pack stores void*.
                      // NOLINTNEXTLINE(readability-non-const-parameter)
                      float* runtimeScaleHostPtr
                      = nullptr) // NOLINT(readability-non-const-parameter)
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
    sdpaAttrs.causal_mask = causalMask;
    sdpaAttrs.causal_mask_bottom_right = causalMaskBottomRight;
    sdpaAttrs.set_alibi_mask(alibiMask);
    sdpaAttrs.set_diagonal_alignment(diagonalAlignment);
    sdpaAttrs.set_generate_stats(generateStats);
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

    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> maskTensorAttr;
    if(tensorBundle.maskTensor.has_value())
    {
        maskTensorAttr = detail::makeSdpaMaskAttributes(tensorBundle.maskTensor.value(), uid++);
        sdpaAttrs.set_bias(maskTensorAttr);
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

    // Finalize the stats/LSE output as a bindable FLOAT tensor [B, H, Sq, 1].
    if(generateStats && statsAttr)
    {
        if(!statsAttr->has_uid())
        {
            statsAttr->set_uid(uid++);
        }
        const auto qDims = tensorBundle.qTensor.dims();
        const std::vector<int64_t> statsDims = {qDims[0], qDims[1], qDims[2], 1};
        const auto statsStrides = hipdnn_data_sdk::utilities::generateStrides(statsDims);
        statsAttr->set_data_type(hipdnn_frontend::DataType::FLOAT)
            .set_dim(statsDims)
            .set_stride(statsStrides)
            .set_is_virtual(false);
    }

    auto variantPack = tensorBundle.createVariantPack(
        *qTensorAttr, *kTensorAttr, *vTensorAttr, *oTensorAttr, maskTensorAttr.get());
    if(scaleTensorAttr)
    {
        variantPack[scaleTensorAttr->get_uid()] = runtimeScaleHostPtr;
    }

    return std::make_tuple(graph, variantPack);
}

template <typename InputType, class MaskT>
std::tuple<std::shared_ptr<hipdnn_frontend::graph::Graph>, std::unordered_map<int64_t, void*>>
    buildSdpaBwdGraph(SdpaBwdTensorBundle<InputType, MaskT>& tensorBundle,
                      hipdnn_flatbuffers_sdk::data_objects::DataType dataType,
                      // Non-const: the variant pack stores void*.
                      // NOLINTNEXTLINE(readability-non-const-parameter)
                      float* runtimeScaleHostPtr = nullptr,
                      bool causalMask = false,
                      bool causalMaskBottomRight = false,
                      std::optional<int64_t> leftBound = std::nullopt,
                      std::optional<int64_t> rightBound = std::nullopt,
                      hipdnn_frontend::DiagonalAlignment diagonalAlignment
                      = hipdnn_frontend::DiagonalAlignment::TOP_LEFT)
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
    sdpaBwdAttrs.causal_mask = causalMask;
    sdpaBwdAttrs.causal_mask_bottom_right = causalMaskBottomRight;
    sdpaBwdAttrs.set_diagonal_alignment(diagonalAlignment);
    if(leftBound.has_value())
    {
        sdpaBwdAttrs.set_diagonal_band_left_bound(leftBound.value());
    }
    if(rightBound.has_value())
    {
        sdpaBwdAttrs.set_diagonal_band_right_bound(rightBound.value());
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
        sdpaBwdAttrs.set_attn_scale(scaleTensorAttr);
    }

    std::shared_ptr<hipdnn_frontend::graph::TensorAttributes> maskTensorAttr;
    if(tensorBundle.maskTensor.has_value())
    {
        maskTensorAttr = detail::makeSdpaMaskAttributes(tensorBundle.maskTensor.value(), uid++);
        sdpaBwdAttrs.set_bias(maskTensorAttr);
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
                                                      *dvTensorAttr,
                                                      maskTensorAttr.get());
    if(scaleTensorAttr)
    {
        variantPack[scaleTensorAttr->get_uid()] = runtimeScaleHostPtr;
    }

    return std::make_tuple(graph, variantPack);
}

// Zero tolerance for comparing reference outputs that must be bitwise equal.
inline constexpr float EXACT_TOLERANCE = 0.0f;

// Mask-dtype test shapes: Q [1, 2, 4, 8], K and V [1, 2, 6, 8] (Sq != Skv).
inline const std::vector<int64_t> MASK_QDIMS = {1, 2, 4, 8};
inline const std::vector<int64_t> MASK_KDIMS = {1, 2, 6, 8};
inline const std::vector<int64_t> MASK_VDIMS = {1, 2, 6, 8};

// Fills a [B, H, Sq, Skv] additive mask with a non-uniform pattern. Softmax is shift-invariant,
// so a uniform mask cannot reveal a misread mask dtype. Every value is exactly representable in
// float, half and bfloat16, so masks of different dtypes carry identical numbers.
template <class MaskT>
void fillNonUniformMask(hipdnn_data_sdk::utilities::Tensor<MaskT>& mask)
{
    constexpr std::array<float, 8> VALUES = {0.25f, -1.0f, 2.0f, 0.5f, -0.5f, 1.0f, -2.0f, 0.75f};
    const auto& dims = mask.dims();
    for(int64_t b = 0; b < dims[0]; ++b)
    {
        for(int64_t h = 0; h < dims[1]; ++h)
        {
            for(int64_t s = 0; s < dims[2]; ++s)
            {
                for(int64_t t = 0; t < dims[3]; ++t)
                {
                    mask.setHostValue(
                        static_cast<MaskT>(VALUES.at(static_cast<size_t>((s * 5 + t * 3 + h) % 8))),
                        b,
                        h,
                        s,
                        t);
                }
            }
        }
    }
}

// Gives a forward or backward SDPA bundle a [B, H, Sq, Skv] mask filled by fillNonUniformMask.
template <class SdpaTensorBundle>
void emplaceNonUniformMask(SdpaTensorBundle& bundle)
{
    const auto& qDims = bundle.qTensor.dims();
    const auto& kDims = bundle.kTensor.dims();
    bundle.maskTensor.emplace(std::vector<int64_t>{qDims[0], qDims[1], qDims[2], kDims[2]});
    fillNonUniformMask(*bundle.maskTensor);
}

} // namespace hipdnn_sdk_test_utils
