// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>
#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "GpuScalarOperand.hpp"
#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

struct GpuSdpaBwdParams
{
    using Attrs = hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT;

    static std::optional<Attrs>
        unpackOptional(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* attributes)
    {
        return attributes != nullptr
                   ? std::make_optional(
                         hipdnn_test_sdk::detail::unpackTensorAttributes(*attributes))
                   : std::nullopt;
    }

    Attrs qTensor;
    Attrs kTensor;
    Attrs vTensor;
    Attrs oTensor;
    Attrs doTensor;
    Attrs statsTensor;
    Attrs dqTensor;
    Attrs dkTensor;
    Attrs dvTensor;
    // Folded scale operand (scale tensor, else baked attn_scale_value); absent means 1.0.
    std::optional<Attrs> scaleTensor;
    std::optional<Attrs> attnMaskTensor;
    std::optional<Attrs> dBiasTensor;
    std::optional<Attrs> seqLenQTensor;
    std::optional<Attrs> seqLenKvTensor;
    // Explicit-mask dropout (cuDNN custom dropout): mask, scale, and optional scale_inv.
    std::optional<Attrs> dropoutMaskTensor;
    std::optional<Attrs> dropoutScaleTensor;
    std::optional<Attrs> dropoutScaleInvTensor;
    bool alibi = false;
    int64_t leftBound = -1;
    int64_t rightBound = -1;
    bool topLeftAlignment = true;
};

template <typename IoDataType, typename MaskDataType = float>
class GpuSdpaBwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuSdpaBwdPlan(GpuSdpaBwdParams&& params)
        : _params(std::move(params))
    {
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        using hipdnn_gpu_ref::ShallowGpuTensor;
        const auto shallow = [&variantPack](const GpuSdpaBwdParams::Attrs& attrs) {
            return ShallowGpuTensor<IoDataType>(
                variantPack.at(attrs.uid), attrs.dims, attrs.strides);
        };
        auto q = shallow(_params.qTensor);
        auto k = shallow(_params.kTensor);
        auto v = shallow(_params.vTensor);
        auto o = shallow(_params.oTensor);
        auto dO = shallow(_params.doTensor);
        auto dQ = shallow(_params.dqTensor);
        auto dK = shallow(_params.dkTensor);
        auto dV = shallow(_params.dvTensor);
        ShallowGpuTensor<float> stats(variantPack.at(_params.statsTensor.uid),
                                      _params.statsTensor.dims,
                                      _params.statsTensor.strides);

        std::optional<ShallowGpuTensor<MaskDataType>> mask;
        std::optional<ShallowGpuTensor<MaskDataType>> dBias;
        for(auto [attrs, tensor] :
            {std::pair{&_params.attnMaskTensor, &mask}, std::pair{&_params.dBiasTensor, &dBias}})
        {
            if(attrs->has_value())
            {
                tensor->emplace(variantPack.at((*attrs)->uid), (*attrs)->dims, (*attrs)->strides);
            }
        }
        std::optional<ShallowGpuTensor<int32_t>> seqLenQ;
        std::optional<ShallowGpuTensor<int32_t>> seqLenKv;
        if(_params.seqLenQTensor.has_value() && _params.seqLenKvTensor.has_value())
        {
            seqLenQ.emplace(variantPack.at(_params.seqLenQTensor->uid),
                            _params.seqLenQTensor->dims,
                            _params.seqLenQTensor->strides);
            seqLenKv.emplace(variantPack.at(_params.seqLenKvTensor->uid),
                             _params.seqLenKvTensor->dims,
                             _params.seqLenKvTensor->strides);
        }

        std::optional<ShallowGpuTensor<float>> dropoutMask;
        std::optional<float> dropoutScale;
        std::optional<float> dropoutScaleInv;
        if(_params.dropoutMaskTensor.has_value() && _params.dropoutScaleTensor.has_value())
        {
            dropoutMask.emplace(variantPack.at(_params.dropoutMaskTensor->uid),
                                _params.dropoutMaskTensor->dims,
                                _params.dropoutMaskTensor->strides);
            dropoutScale = resolveScalarOperand(
                *_params.dropoutScaleTensor, variantPack, "SDPA dropout scale");
        }
        if(_params.dropoutScaleInvTensor.has_value())
        {
            dropoutScaleInv = resolveScalarOperand(
                *_params.dropoutScaleInvTensor, variantPack, "SDPA dropout scale inverse");
        }

        std::optional<float> attnScale;
        if(_params.scaleTensor.has_value())
        {
            attnScale = resolveScalarOperand(*_params.scaleTensor, variantPack, "SDPA scale");
        }

        hipdnn_gpu_ref::GpuFpReferenceSdpa::bprop<IoDataType,
                                                  IoDataType,
                                                  IoDataType,
                                                  IoDataType,
                                                  IoDataType,
                                                  IoDataType,
                                                  IoDataType,
                                                  IoDataType,
                                                  float,
                                                  MaskDataType>(
            q,
            k,
            v,
            o,
            dO,
            dQ,
            dK,
            dV,
            hipdnn_gpu_ref::GpuSdpaBwdOptions<MaskDataType>{}
                .setScale(attnScale)
                .setLse(&stats)
                .setMask(mask.has_value() ? &mask.value() : nullptr)
                .setDBias(dBias.has_value() ? &dBias.value() : nullptr)
                .setBand(_params.leftBound, _params.rightBound, _params.topLeftAlignment)
                .setSeqLens(seqLenQ.has_value() ? &seqLenQ.value() : nullptr,
                            seqLenKv.has_value() ? &seqLenKv.value() : nullptr)
                .setDropout(dropoutMask.has_value() ? &dropoutMask.value() : nullptr,
                            dropoutScale,
                            dropoutScaleInv)
                .setAlibi(_params.alibi));
    }

private:
    GpuSdpaBwdParams _params;
};

// GPU SDPA backward plan builder: one I/O dtype for Q/K/V/O/dO/dQ/dK/dV (as the CPU reference
// registers) and one mask dtype (UNSET = no mask). Applicability mirrors the CPU SdpaBwdPlan, plus
// the device checks on stats and scale.
template <hipdnn_flatbuffers_sdk::data_objects::DataType IoDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MaskDataTypeEnum
          = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>
class GpuSdpaBwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using IoDataType = hipdnn_test_sdk::utilities::DataTypeToNative<IoDataTypeEnum>;
    using MaskDataType = hipdnn_test_sdk::utilities::MaskDataTypeToNative<MaskDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;
        const auto* attrs = node.attributes_as_SdpaBackwardAttributes();
        if(attrs == nullptr)
        {
            return false;
        }
        for(const auto uid : {attrs->q_tensor_uid(),
                              attrs->k_tensor_uid(),
                              attrs->v_tensor_uid(),
                              attrs->o_tensor_uid(),
                              attrs->do_tensor_uid(),
                              attrs->dq_tensor_uid(),
                              attrs->dk_tensor_uid(),
                              attrs->dv_tensor_uid()})
        {
            CHECK_TENSOR_EXISTS(tensorMap, uid);
            CHECK_TENSOR_TYPE(tensorMap, uid, IoDataTypeEnum);
        }

        // Stats: FLOAT, rank-3 [B, H, Sq] or rank-4 [B, H, Sq, 1].
        CHECK_TENSOR_EXISTS(tensorMap, attrs->stats_tensor_uid());
        {
            const auto* stats = tensorMap.at(attrs->stats_tensor_uid());
            const auto* dims = stats->dims();
            if(stats->data_type() != DataType::FLOAT || dims == nullptr
               || (dims->size() != 3 && (dims->size() != 4 || dims->Get(3) != 1)))
            {
                return false;
            }
        }

        if constexpr(MaskDataTypeEnum == DataType::UNSET)
        {
            if(attrs->attn_mask_tensor_uid().has_value())
            {
                return false;
            }
        }
        else
        {
            if(!attrs->attn_mask_tensor_uid().has_value())
            {
                return false;
            }
            CHECK_TENSOR_EXISTS(tensorMap, attrs->attn_mask_tensor_uid().value());
            CHECK_TENSOR_TYPE(tensorMap, attrs->attn_mask_tensor_uid().value(), MaskDataTypeEnum);
        }

        if(!isSupportedSdpaAlibi(*attrs) || !isSupportedSdpaPadding(*attrs, tensorMap))
        {
            return false;
        }

        // dBias has the bias tensor's shape and dtype (cuDNN).
        if(attrs->dbias_tensor_uid().has_value())
        {
            if(!attrs->attn_mask_tensor_uid().has_value())
            {
                return false;
            }
            CHECK_TENSOR_EXISTS(tensorMap, attrs->dbias_tensor_uid().value());
            const auto* dBias = tensorMap.at(attrs->dbias_tensor_uid().value());
            const auto* mask = tensorMap.at(attrs->attn_mask_tensor_uid().value());
            if(dBias->data_type() != mask->data_type() || dBias->dims() == nullptr
               || mask->dims() == nullptr
               || !std::equal(dBias->dims()->begin(),
                              dBias->dims()->end(),
                              mask->dims()->begin(),
                              mask->dims()->end()))
            {
                return false;
            }
        }

        // Dropout: only cuDNN's explicit-mask form; dropout_scale_inv one FLOAT element.
        if(!isSupportedSdpaDropout(*attrs, tensorMap))
        {
            return false;
        }
        if(attrs->dropout_scale_inv_tensor_uid().has_value())
        {
            CHECK_TENSOR_EXISTS(tensorMap, attrs->dropout_scale_inv_tensor_uid().value());
            const auto inv = hipdnn_test_sdk::detail::unpackTensorAttributes(
                *tensorMap.at(attrs->dropout_scale_inv_tensor_uid().value()));
            if(hipdnn_test_sdk::detail::elementCount(inv) != 1 || inv.data_type != DataType::FLOAT
               || !attrs->dropout_mask_tensor_uid().has_value())
            {
                return false;
            }
        }

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        // The scale may be baked, pass-by-value or device-resident: one element, FLOAT on device.
        return isSupportedSdpaScaleOperand(*attrs, tensorMap);
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* attrs = node.attributes_as_SdpaBackwardAttributes();
        if(attrs == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type SdpaBackwardAttributes");
        }
        const auto& tensorMap = graph.getTensorMap();
        const auto unpack = [&tensorMap](int64_t uid) {
            return hipdnn_test_sdk::detail::unpackTensorAttributes(*tensorMap.at(uid));
        };
        const auto optional = [&tensorMap](const ::flatbuffers::Optional<int64_t>& uid) {
            return GpuSdpaBwdParams::unpackOptional(uid.has_value() ? tensorMap.at(uid.value())
                                                                    : nullptr);
        };

        GpuSdpaBwdParams params;
        params.qTensor = unpack(attrs->q_tensor_uid());
        params.kTensor = unpack(attrs->k_tensor_uid());
        params.vTensor = unpack(attrs->v_tensor_uid());
        params.oTensor = unpack(attrs->o_tensor_uid());
        params.doTensor = unpack(attrs->do_tensor_uid());
        params.statsTensor = unpack(attrs->stats_tensor_uid());
        params.dqTensor = unpack(attrs->dq_tensor_uid());
        params.dkTensor = unpack(attrs->dk_tensor_uid());
        params.dvTensor = unpack(attrs->dv_tensor_uid());
        params.attnMaskTensor = optional(attrs->attn_mask_tensor_uid());
        params.dBiasTensor = optional(attrs->dbias_tensor_uid());
        params.seqLenQTensor = optional(attrs->seq_len_q_tensor_uid());
        params.seqLenKvTensor = optional(attrs->seq_len_kv_tensor_uid());
        params.dropoutMaskTensor = optional(attrs->dropout_mask_tensor_uid());
        params.dropoutScaleTensor = optional(attrs->dropout_scale_tensor_uid());
        params.dropoutScaleInvTensor = optional(attrs->dropout_scale_inv_tensor_uid());

        std::optional<float> attnScaleValue;
        if(attrs->attn_scale_value().has_value())
        {
            attnScaleValue = attrs->attn_scale_value();
        }
        params.scaleTensor = hipdnn_test_sdk::detail::foldSdpaScale(
            attrs->scale_tensor_uid().has_value() ? tensorMap.at(attrs->scale_tensor_uid().value())
                                                  : nullptr,
            attnScaleValue);

        const auto band = extractDiagonalBandParams(*attrs, "GpuSdpaBwdPlan");
        params.leftBound = band.leftBound;
        params.rightBound = band.rightBound;
        params.topLeftAlignment = band.topLeftAlignment;

        params.alibi = attrs->alibi_mask();
        return std::make_unique<GpuSdpaBwdPlan<IoDataType, MaskDataType>>(std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
