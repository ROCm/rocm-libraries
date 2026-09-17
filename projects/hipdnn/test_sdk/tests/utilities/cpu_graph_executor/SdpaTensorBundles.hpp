// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_test_sdk/utilities/Seeds.hpp>

namespace hipdnn_sdk_test_utils
{

/// Tensor bundle for a forward SDPA operation.
/// Dims: Q [B, H, Sq, D], K [B, Hkv, Skv, D], V [B, Hkv, Skv, Dv], O [B, H, Sq, Dv]
template <typename InputType>
struct SdpaFwdTensorBundle
{
    SdpaFwdTensorBundle(const std::vector<int64_t>& qDims,
                        const std::vector<int64_t>& kDims,
                        const std::vector<int64_t>& vDims,
                        unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed())
        : qTensor(qDims)
        , kTensor(kDims)
        , vTensor(vDims)
        , oTensor({qDims[0], qDims[1], qDims[2], vDims[3]})
    {
        qTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
        kTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
        vTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const hipdnn_frontend::graph::TensorAttributes& qTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& kTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& vTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& oTensorAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[qTensorAttr.get_uid()] = qTensor.memory().hostData();
        variantPack[kTensorAttr.get_uid()] = kTensor.memory().hostData();
        variantPack[vTensorAttr.get_uid()] = vTensor.memory().hostData();
        variantPack[oTensorAttr.get_uid()] = oTensor.memory().hostData();
        return variantPack;
    }

    hipdnn_data_sdk::utilities::Tensor<InputType> qTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> kTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> vTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> oTensor;
};

/// Tensor bundle for a backward SDPA operation.
/// Dims: Q [B, H, Sq, D], K [B, Hkv, Skv, D], V [B, Hkv, Skv, Dv],
///       O/dO [B, H, Sq, Dv], stats [B, H, Sq, 1],
///       dQ [B, H, Sq, D], dK [B, Hkv, Skv, D], dV [B, Hkv, Skv, Dv]
template <typename InputType>
struct SdpaBwdTensorBundle
{
    SdpaBwdTensorBundle(const std::vector<int64_t>& qDims,
                        const std::vector<int64_t>& kDims,
                        const std::vector<int64_t>& vDims,
                        unsigned int seed = hipdnn_test_sdk::utilities::getGlobalTestSeed())
        : qTensor(qDims)
        , kTensor(kDims)
        , vTensor(vDims)
        , oTensor({qDims[0], qDims[1], qDims[2], vDims[3]})
        , doTensor({qDims[0], qDims[1], qDims[2], vDims[3]})
        , statsTensor({qDims[0], qDims[1], qDims[2], 1})
        , dqTensor(qDims)
        , dkTensor(kDims)
        , dvTensor(vDims)
    {
        qTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
        kTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
        vTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
        oTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
        doTensor.fillWithRandomValues(
            static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
        statsTensor.fillWithRandomValues(0.0f, 1.0f, seed);
    }

    std::unordered_map<int64_t, void*>
        createVariantPack(const hipdnn_frontend::graph::TensorAttributes& qTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& kTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& vTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& oTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& doTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& statsTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& dqTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& dkTensorAttr,
                          const hipdnn_frontend::graph::TensorAttributes& dvTensorAttr)
    {
        std::unordered_map<int64_t, void*> variantPack;
        variantPack[qTensorAttr.get_uid()] = qTensor.memory().hostData();
        variantPack[kTensorAttr.get_uid()] = kTensor.memory().hostData();
        variantPack[vTensorAttr.get_uid()] = vTensor.memory().hostData();
        variantPack[oTensorAttr.get_uid()] = oTensor.memory().hostData();
        variantPack[doTensorAttr.get_uid()] = doTensor.memory().hostData();
        variantPack[statsTensorAttr.get_uid()] = statsTensor.memory().hostData();
        variantPack[dqTensorAttr.get_uid()] = dqTensor.memory().hostData();
        variantPack[dkTensorAttr.get_uid()] = dkTensor.memory().hostData();
        variantPack[dvTensorAttr.get_uid()] = dvTensor.memory().hostData();
        return variantPack;
    }

    hipdnn_data_sdk::utilities::Tensor<InputType> qTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> kTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> vTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> oTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> doTensor;
    hipdnn_data_sdk::utilities::Tensor<float> statsTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> dqTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> dkTensor;
    hipdnn_data_sdk::utilities::Tensor<InputType> dvTensor;
};

} // namespace hipdnn_sdk_test_utils
