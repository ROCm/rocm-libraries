// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpaRagged.hpp>
#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/RaggedSdpaFwdSupport.hpp>

#include "GpuScalarOperand.hpp"
#include "GpuSdpaFwdPlan.hpp" // reuse REFERENCE_SDPA_PROBABILITY_MODE
#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// Runs the ragged SDPA GPU reference on the variant-pack device buffers.
template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename ComputeDataType = float>
class GpuSdpaRaggedFwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuSdpaRaggedFwdPlan(hipdnn_test_sdk::detail::RaggedSdpaFwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        hipdnn_gpu_ref::ShallowGpuTensor<QDataType> qTensor(
            variantPack.at(_params.qTensor.uid), _params.qTensor.dims, _params.qTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<KDataType> kTensor(
            variantPack.at(_params.kTensor.uid), _params.kTensor.dims, _params.kTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<VDataType> vTensor(
            variantPack.at(_params.vTensor.uid), _params.vTensor.dims, _params.vTensor.strides);
        hipdnn_gpu_ref::ShallowGpuTensor<ODataType> oTensor(
            variantPack.at(_params.oTensor.uid), _params.oTensor.dims, _params.oTensor.strides);

        const auto bindOffsets
            = [&variantPack](const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& t) {
                  return hipdnn_gpu_ref::ShallowGpuTensor<int32_t>(
                      variantPack.at(t.uid), t.dims, t.strides);
              };
        auto raggedOffsetQTensor = bindOffsets(_params.raggedOffsetQTensor);
        auto raggedOffsetKTensor = bindOffsets(_params.raggedOffsetKTensor);
        auto raggedOffsetVTensor = bindOffsets(_params.raggedOffsetVTensor);
        auto raggedOffsetOTensor = bindOffsets(_params.raggedOffsetOTensor);

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> lseTensor;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<int32_t>> raggedOffsetLseTensor;
        if(_params.lseTensor.has_value())
        {
            lseTensor.emplace(variantPack.at(_params.lseTensor->uid),
                              _params.lseTensor->dims,
                              _params.lseTensor->strides);
        }
        if(_params.raggedOffsetLseTensor.has_value())
        {
            raggedOffsetLseTensor.emplace(variantPack.at(_params.raggedOffsetLseTensor->uid),
                                          _params.raggedOffsetLseTensor->dims,
                                          _params.raggedOffsetLseTensor->strides);
        }

        // A pass-by-value descale (baked or runtime) is staged in a one-element tensor.
        // A device descale is viewed in place. fpropRagged checks the shape.
        const auto bindDescale
            = [&variantPack](
                  const std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT>& d,
                  std::optional<hipdnn_data_sdk::utilities::Tensor<float>>& staged,
                  std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>>& view)
            -> hipdnn_data_sdk::utilities::TensorBase<float>* {
            if(!d.has_value())
            {
                return nullptr;
            }
            if(hipdnn_flatbuffers_sdk::utilities::isPassByValueTensor(*d))
            {
                staged.emplace(std::vector<int64_t>{1});
                staged->fillWithValue(resolveScalarOperand(*d, variantPack, "SDPA descale"));
                return &staged.value();
            }
            view.emplace(variantPack.at(d->uid), d->dims, d->strides);
            return &view.value();
        };
        std::optional<hipdnn_data_sdk::utilities::Tensor<float>> stagedDescaleQ;
        std::optional<hipdnn_data_sdk::utilities::Tensor<float>> stagedDescaleK;
        std::optional<hipdnn_data_sdk::utilities::Tensor<float>> stagedDescaleV;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> descaleQView;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> descaleKView;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> descaleVView;
        auto* descaleQ = bindDescale(_params.descaleQTensor, stagedDescaleQ, descaleQView);
        auto* descaleK = bindDescale(_params.descaleKTensor, stagedDescaleK, descaleKView);
        auto* descaleV = bindDescale(_params.descaleVTensor, stagedDescaleV, descaleVView);

        std::optional<float> attnScale;
        if(_params.scaleTensor.has_value())
        {
            attnScale = resolveScalarOperand(*_params.scaleTensor, variantPack, "SDPA scale");
        }

        // Each tensor's ragged_offset_multiplier scales its stored offsets to elements.
        hipdnn_gpu_ref::RaggedOffsetMultipliers multipliers;
        multipliers.q = _params.qTensor.ragged_offset_multiplier;
        multipliers.k = _params.kTensor.ragged_offset_multiplier;
        multipliers.v = _params.vTensor.ragged_offset_multiplier;
        multipliers.o = _params.oTensor.ragged_offset_multiplier;
        if(_params.lseTensor.has_value())
        {
            multipliers.lse = _params.lseTensor->ragged_offset_multiplier;
        }

        hipdnn_gpu_ref::GpuFpReferenceSdpaRagged::
            fpropRagged<QDataType, KDataType, VDataType, ODataType, ComputeDataType>(
                qTensor,
                kTensor,
                vTensor,
                oTensor,
                raggedOffsetQTensor,
                raggedOffsetKTensor,
                raggedOffsetVTensor,
                raggedOffsetOTensor,
                attnScale,
                _params.leftBound,
                _params.rightBound,
                _params.topLeftAlignment,
                lseTensor.has_value() ? &lseTensor.value() : nullptr,
                raggedOffsetLseTensor.has_value() ? &raggedOffsetLseTensor.value() : nullptr,
                REFERENCE_SDPA_PROBABILITY_MODE,
                descaleQ,
                descaleK,
                descaleV,
                multipliers);
    }

private:
    hipdnn_test_sdk::detail::RaggedSdpaFwdParams _params;
};

// Applicability is hipdnn_test_sdk::detail::isSupportedRaggedSdpaFwdNode, which the CPU
// SdpaRaggedFwdPlanBuilder shares, plus this builder's Q/K/V/O data types.
// GpuReferenceGraphExecutor::buildSignatureKey picks dense vs ragged.
template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum>
class GpuSdpaRaggedFwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using QDataType = hipdnn_test_sdk::utilities::DataTypeToNative<QDataTypeEnum>;
    using KDataType = hipdnn_test_sdk::utilities::DataTypeToNative<KDataTypeEnum>;
    using VDataType = hipdnn_test_sdk::utilities::DataTypeToNative<VDataTypeEnum>;
    using ODataType = hipdnn_test_sdk::utilities::DataTypeToNative<ODataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        const auto* nodeAttributes = node.attributes_as_SdpaAttributes();
        if(nodeAttributes == nullptr
           || !hipdnn_test_sdk::detail::isSupportedRaggedSdpaFwdNode(*nodeAttributes, tensorMap))
        {
            return false;
        }

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->q_tensor_uid(), QDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->k_tensor_uid(), KDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->v_tensor_uid(), VDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->o_tensor_uid(), ODataTypeEnum);
        return true;
    }

    // fpropRagged checks that O (and a packed LSE) has Q's per-batch lengths and V has K's.
    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_SdpaAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type SdpaAttributes");
        }

        auto [leftBound, rightBound, isTopLeft]
            = extractDiagonalBandParams(*nodeAttributes, "GpuSdpaRaggedFwdPlan");

        return std::make_unique<
            GpuSdpaRaggedFwdPlan<QDataType, KDataType, VDataType, ODataType, float>>(
            hipdnn_test_sdk::detail::RaggedSdpaFwdParams(
                *nodeAttributes, graph.getTensorMap(), leftBound, rightBound, isTopLeft));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
