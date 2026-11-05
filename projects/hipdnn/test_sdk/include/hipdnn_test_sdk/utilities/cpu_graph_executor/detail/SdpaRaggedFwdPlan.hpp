// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn_data_sdk/utilities/RaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowRaggedTensor.hpp>
#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceSdpaRagged.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaFwdSupport.hpp>

namespace hipdnn_test_sdk::detail
{

// Runs CpuFpReferenceSdpaRagged on the variant-pack host buffers. Every ragged operand is a
// ShallowRaggedTensor view, so the reference addresses the packed data through its offset table.
template <typename QDataType, typename KDataType, typename VDataType, typename ODataType>
class SdpaRaggedFwdPlan : public IGraphNodePlanExecutor
{
public:
    explicit SdpaRaggedFwdPlan(RaggedSdpaFwdParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        std::vector<int64_t> ids = {_params.oTensor.uid};
        if(_params.lseTensor.has_value())
        {
            ids.push_back(_params.lseTensor->uid);
        }
        return ids;
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto qTensor
            = bindRagged<QDataType>(_params.qTensor, _params.raggedOffsetQTensor, variantPack);
        auto kTensor
            = bindRagged<KDataType>(_params.kTensor, _params.raggedOffsetKTensor, variantPack);
        auto vTensor
            = bindRagged<VDataType>(_params.vTensor, _params.raggedOffsetVTensor, variantPack);
        auto oTensor
            = bindRagged<ODataType>(_params.oTensor, _params.raggedOffsetOTensor, variantPack);

        // A packed LSE is ragged on its own offset table. A dense LSE is a plain view.
        std::optional<hipdnn_data_sdk::utilities::ShallowRaggedTensor<float>> raggedLseTensor;
        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> denseLseTensor;
        hipdnn_data_sdk::utilities::TensorBase<float>* lseTensor = nullptr;
        if(_params.lseTensor.has_value())
        {
            const auto& lse = *_params.lseTensor;
            if(_params.raggedOffsetLseTensor.has_value())
            {
                lseTensor = &raggedLseTensor.emplace(
                    variantPack.at(lse.uid),
                    lse.dims,
                    lse.strides,
                    hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS,
                    bindOffsets(*_params.raggedOffsetLseTensor, variantPack),
                    std::nullopt,
                    lse.ragged_offset_multiplier);
            }
            else
            {
                denseLseTensor = createShallowTensor<float>(lse, variantPack.at(lse.uid));
                lseTensor = denseLseTensor.get();
            }
        }

        // CpuFpReferenceSdpaRagged checks the descale shapes.
        const auto descaleQ
            = bindOptionalHostFloatOperand(_params.descaleQTensor, variantPack, "SDPA descale");
        const auto descaleK
            = bindOptionalHostFloatOperand(_params.descaleKTensor, variantPack, "SDPA descale");
        const auto descaleV
            = bindOptionalHostFloatOperand(_params.descaleVTensor, variantPack, "SDPA descale");

        std::optional<float> attnScale;
        if(_params.scaleTensor.has_value())
        {
            attnScale = readHostFloatOperand(*_params.scaleTensor, variantPack, "SDPA scale");
        }

        utilities::CpuFpReferenceSdpaRagged::
            forward<QDataType, KDataType, VDataType, ODataType, float>(qTensor,
                                                                       kTensor,
                                                                       vTensor,
                                                                       oTensor,
                                                                       attnScale,
                                                                       _params.leftBound,
                                                                       _params.rightBound,
                                                                       _params.topLeftAlignment,
                                                                       lseTensor,
                                                                       descaleQ.get(),
                                                                       descaleK.get(),
                                                                       descaleV.get());
    }

private:
    // The int32 ragged_offset aux viewed in place. Views are non-owning, so Q/O and K/V may share
    // one table.
    static std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>
        bindOffsets(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& raggedOffset,
                    const std::unordered_map<int64_t, void*>& variantPack)
    {
        return std::make_shared<hipdnn_data_sdk::utilities::ShallowTensor<int32_t>>(
            variantPack.at(raggedOffset.uid), raggedOffset.dims, raggedOffset.strides);
    }

    // Views a packed primary through its ragged_offset aux, scaled by the primary's
    // ragged_offset_multiplier.
    template <typename T>
    static hipdnn_data_sdk::utilities::ShallowRaggedTensor<T>
        bindRagged(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& tensor,
                   const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& raggedOffset,
                   const std::unordered_map<int64_t, void*>& variantPack)
    {
        return hipdnn_data_sdk::utilities::ShallowRaggedTensor<T>(
            variantPack.at(tensor.uid),
            tensor.dims,
            tensor.strides,
            hipdnn_data_sdk::utilities::SDPA_SEQ_AXIS,
            bindOffsets(raggedOffset, variantPack),
            std::nullopt,
            tensor.ragged_offset_multiplier);
    }

    RaggedSdpaFwdParams _params;
};

// Applicability is isSupportedRaggedSdpaFwdNode, which GpuSdpaRaggedFwdPlanBuilder shares, plus
// this builder's Q/K/V/O data types. Unlike the dense SdpaFwdPlanBuilder, an additive attn_mask is
// rejected and Q/K/V descales are optional. CpuReferenceGraphExecutor::buildSignatureKey picks
// dense vs ragged.
template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum>
class SdpaRaggedFwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using QDataType = utilities::DataTypeToNative<QDataTypeEnum>;
    using KDataType = utilities::DataTypeToNative<KDataTypeEnum>;
    using VDataType = utilities::DataTypeToNative<VDataTypeEnum>;
    using ODataType = utilities::DataTypeToNative<ODataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_SdpaAttributes();
        if(nodeAttributes == nullptr || !isSupportedRaggedSdpaFwdNode(*nodeAttributes, tensorMap))
        {
            return false;
        }

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->q_tensor_uid(), QDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->k_tensor_uid(), KDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->v_tensor_uid(), VDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->o_tensor_uid(), ODataTypeEnum);
        return true;
    }

    // The reference checks that O (and a packed LSE) has Q's per-batch lengths and V has K's.
    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_SdpaAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type SdpaAttributes");
        }

        auto [leftBound, rightBound, isTopLeft]
            = extractDiagonalBandParams(*nodeAttributes, "SdpaRaggedFwdPlan");

        return std::make_unique<SdpaRaggedFwdPlan<QDataType, KDataType, VDataType, ODataType>>(
            RaggedSdpaFwdParams(
                *nodeAttributes, graph.getTensorMap(), leftBound, rightBound, isTopLeft));
    }
};

} // namespace hipdnn_test_sdk::detail
