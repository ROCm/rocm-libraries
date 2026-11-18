// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include <hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp>
#include <hipdnn-gpu-ref/ShallowGpuTensor.hpp>
#include <hipdnn_data_sdk/types/Bfloat16.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_flatbuffers_sdk/utilities/FlatbufferUtils.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include "GpuScalarOperand.hpp"
#include "IGpuGraphNodePlanBuilder.hpp"
#include "IGpuGraphNodePlanExecutor.hpp"

namespace hipdnn_integration_tests::gpu_graph_executor::detail
{

// A graph stats/LSE tensor is conventionally rank-4 [B, H, Sq, 1], but the SDPA
// reference requires the LSE output to be rank-3 [B, H, Sq]. Drop a single trailing
// size-1 dim; a rank-3 input passes through unchanged. Any other shape is unsupported.
inline std::vector<int64_t> squeezeTrailingUnitDim(const std::vector<int64_t>& dims)
{
    if(dims.size() == 3)
    {
        return dims;
    }
    if(dims.size() == 4 && dims.back() == 1)
    {
        return {dims.begin(), dims.end() - 1};
    }
    throw std::invalid_argument(
        "GpuSdpaFwdPlan: stats/LSE tensor must be rank-3 [B, H, Sq] or rank-4 [B, H, Sq, 1]");
}

// Truncate a stride vector to match a squeezed dims vector by dropping trailing entries.
inline std::vector<int64_t> squeezeToRank(const std::vector<int64_t>& strides, size_t rank)
{
    return {strides.begin(), strides.begin() + static_cast<std::ptrdiff_t>(rank)};
}

// The graph plans use fp32 softmax probabilities, like the CPU reference and cuDNN's reference,
// so the GPU reference is an independent oracle for every dtype (SdpaSemanticsRegister.md SEM-012).
inline constexpr hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode REFERENCE_SDPA_PROBABILITY_MODE
    = hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode::FLOAT;

// Opt-in emulation of a provider kernel's P storage, for callers that want to model it
// explicitly (GpuFpReferenceSdpa::fprop's probabilityMode). Not used by the graph plans.
// The P->bf16 cast is a property of the P@V matmul inputs, not the output: the AITER BF16
// forward kernel materializes the softmax probabilities in bf16 before P@V whenever Q/K/V are
// bf16, regardless of the (fp32 or bf16) output dtype.
template <typename QDataType, typename KDataType, typename VDataType, typename ODataType>
constexpr hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode providerSdpaProbabilityMode()
{
    using hipdnn_data_sdk::types::bfloat16;
    if constexpr(std::is_same_v<QDataType, bfloat16> && std::is_same_v<KDataType, bfloat16>
                 && std::is_same_v<VDataType, bfloat16>)
    {
        return hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode::BFLOAT16_RTNE;
    }
    else
    {
        return hipdnn_gpu_ref::SdpaSoftmaxProbabilityMode::FLOAT;
    }
}

struct GpuSdpaFwdParams
{
    GpuSdpaFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& qAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& kAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& vAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& oAttributes,
        std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scaleTensor,
        int64_t leftBound,
        int64_t rightBound,
        bool topLeftAlignment,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* attnMaskAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* lseAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* seqLenQAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* seqLenKvAttributes = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* pageTableKAttributes
        = nullptr,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* pageTableVAttributes
        = nullptr,
        int64_t pagedMaxSeqKv = 0,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* sinkAttributes = nullptr)
        : qTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(qAttributes))
        , kTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(kAttributes))
        , vTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(vAttributes))
        , oTensor(hipdnn_test_sdk::detail::unpackTensorAttributes(oAttributes))
        , scaleTensor(std::move(scaleTensor))
        , leftBound(leftBound)
        , rightBound(rightBound)
        , topLeftAlignment(topLeftAlignment)
        , attnMaskTensor(attnMaskAttributes != nullptr
                             ? std::make_optional(hipdnn_test_sdk::detail::unpackTensorAttributes(
                                   *attnMaskAttributes))
                             : std::nullopt)
        , lseTensor(lseAttributes != nullptr
                        ? std::make_optional(
                              hipdnn_test_sdk::detail::unpackTensorAttributes(*lseAttributes))
                        : std::nullopt)
        , seqLenQTensor(seqLenQAttributes != nullptr
                            ? std::make_optional(hipdnn_test_sdk::detail::unpackTensorAttributes(
                                  *seqLenQAttributes))
                            : std::nullopt)
        , seqLenKvTensor(seqLenKvAttributes != nullptr
                             ? std::make_optional(hipdnn_test_sdk::detail::unpackTensorAttributes(
                                   *seqLenKvAttributes))
                             : std::nullopt)
        , pageTableKTensor(pageTableKAttributes != nullptr
                               ? std::make_optional(hipdnn_test_sdk::detail::unpackTensorAttributes(
                                     *pageTableKAttributes))
                               : std::nullopt)
        , pageTableVTensor(pageTableVAttributes != nullptr
                               ? std::make_optional(hipdnn_test_sdk::detail::unpackTensorAttributes(
                                     *pageTableVAttributes))
                               : std::nullopt)
        , pagedMaxSeqKv(pagedMaxSeqKv)
        , sinkTensor(sinkAttributes != nullptr
                         ? std::make_optional(
                               hipdnn_test_sdk::detail::unpackTensorAttributes(*sinkAttributes))
                         : std::nullopt)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT qTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT kTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT vTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT oTensor;
    // Folded scale operand (scale tensor, else baked attn_scale_value), resolved at execute time.
    // Absent means 1.0 (no scaling), as in cuDNN.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scaleTensor;
    int64_t leftBound;
    int64_t rightBound;
    bool topLeftAlignment;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> attnMaskTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> lseTensor;
    // Padding lengths (INT32, one per batch); applicability guarantees both or neither.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> seqLenQTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> seqLenKvTensor;
    // Paged KV cache: page tables for K and/or V and the logical key extent (0 = not paged).
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> pageTableKTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> pageTableVTensor;
    int64_t pagedMaxSeqKv;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> sinkTensor;
    // Explicit-mask dropout (cuDNN custom dropout): mask and single-element scale.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> dropoutMaskTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> dropoutScaleTensor;
    // Philox dropout checked through RNG_DUMP: dropoutMaskTensor is the rng_dump tensor and this is
    // 1 / (1 - p).
    std::optional<float> philoxDropoutScale;
    // cuDNN Max / Sum_exp row statistics outputs, FLOAT [B, H, Sq, 1].
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> maxTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> sumExpTensor;
    bool alibi = false;
    // fp8 Q/K/V descales, FLOAT scalar or [B, H_kv, 1, 1]; all three or none.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleQTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleKTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleVTensor;
    // cuDNN sdpa_fp8 softmax / output scaling: one-element FLOAT scales and amax outputs.
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scaleSTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> descaleSTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scaleOTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> amaxSTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> amaxOTensor;
    // cuDNN block-sparse mask, UINT8 (SEM-025).
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> blockMaskTensor;
};

template <typename QDataType,
          typename KDataType,
          typename VDataType,
          typename ODataType,
          typename ComputeDataType = float,
          typename MaskDataType = float>
class GpuSdpaFwdPlan : public IGpuGraphNodePlanExecutor
{
public:
    explicit GpuSdpaFwdPlan(GpuSdpaFwdParams&& params)
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

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<MaskDataType>> attnMaskTensor;
        if(_params.attnMaskTensor.has_value())
        {
            attnMaskTensor.emplace(variantPack.at(_params.attnMaskTensor->uid),
                                   _params.attnMaskTensor->dims,
                                   _params.attnMaskTensor->strides);
        }

        // The graph stats tensor is rank-4 [B, H, Sq, 1], but fprop requires LSE to be
        // rank-3 [B, H, Sq]. Drop the trailing size-1 dim (and its stride) so the shapes
        // line up without relaxing the reference's strict rank check.
        const auto bindRowStats =
            [&variantPack](const auto& stats,
                           std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>>& bound) {
                if(stats.has_value())
                {
                    const auto squeezedDims = squeezeTrailingUnitDim(stats->dims);
                    const auto squeezedStrides = squeezeToRank(stats->strides, squeezedDims.size());
                    bound.emplace(variantPack.at(stats->uid), squeezedDims, squeezedStrides);
                }
            };
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> lseTensor;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> maxTensor;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> sumExpTensor;
        bindRowStats(_params.lseTensor, lseTensor);
        bindRowStats(_params.maxTensor, maxTensor);
        bindRowStats(_params.sumExpTensor, sumExpTensor);

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<int32_t>> seqLenQTensor;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<int32_t>> seqLenKvTensor;
        if(_params.seqLenQTensor.has_value() && _params.seqLenKvTensor.has_value())
        {
            seqLenQTensor.emplace(variantPack.at(_params.seqLenQTensor->uid),
                                  _params.seqLenQTensor->dims,
                                  _params.seqLenQTensor->strides);
            seqLenKvTensor.emplace(variantPack.at(_params.seqLenKvTensor->uid),
                                   _params.seqLenKvTensor->dims,
                                   _params.seqLenKvTensor->strides);
        }

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<int32_t>> pageTableK;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<int32_t>> pageTableV;
        for(auto [attr, table] : {std::pair{&_params.pageTableKTensor, &pageTableK},
                                  std::pair{&_params.pageTableVTensor, &pageTableV}})
        {
            if(attr->has_value())
            {
                table->emplace(variantPack.at((*attr)->uid), (*attr)->dims, (*attr)->strides);
            }
        }

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> sinkTensor;
        if(_params.sinkTensor.has_value())
        {
            sinkTensor.emplace(variantPack.at(_params.sinkTensor->uid),
                               _params.sinkTensor->dims,
                               _params.sinkTensor->strides);
        }

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> dropoutMask;
        std::optional<float> dropoutScale;
        if(_params.dropoutMaskTensor.has_value() && _params.philoxDropoutScale.has_value())
        {
            dropoutMask.emplace(variantPack.at(_params.dropoutMaskTensor->uid),
                                _params.dropoutMaskTensor->dims,
                                _params.dropoutMaskTensor->strides);
            dropoutScale = _params.philoxDropoutScale;
        }
        else if(_params.dropoutMaskTensor.has_value() && _params.dropoutScaleTensor.has_value())
        {
            dropoutMask.emplace(variantPack.at(_params.dropoutMaskTensor->uid),
                                _params.dropoutMaskTensor->dims,
                                _params.dropoutMaskTensor->strides);
            dropoutScale = resolveScalarOperand(
                *_params.dropoutScaleTensor, variantPack, "SDPA dropout scale");
        }

        // A pass-by-value descale (baked or runtime) is staged in a one-element tensor: its
        // variant-pack slot, if any, holds a host pointer. A device descale is viewed in place.
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

        const auto scalar = [&variantPack](const auto& attrs, const char* name) {
            return attrs.has_value()
                       ? std::optional<float>(resolveScalarOperand(*attrs, variantPack, name))
                       : std::nullopt;
        };
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> amaxS;
        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<float>> amaxO;
        for(auto [attr, bound] :
            {std::pair{&_params.amaxSTensor, &amaxS}, std::pair{&_params.amaxOTensor, &amaxO}})
        {
            if(attr->has_value())
            {
                bound->emplace(variantPack.at((*attr)->uid), (*attr)->dims, (*attr)->strides);
            }
        }

        std::optional<hipdnn_gpu_ref::ShallowGpuTensor<uint8_t>> blockMask;
        if(_params.blockMaskTensor.has_value())
        {
            blockMask.emplace(variantPack.at(_params.blockMaskTensor->uid),
                              _params.blockMaskTensor->dims,
                              _params.blockMaskTensor->strides);
        }

        std::optional<float> attnScale;
        if(_params.scaleTensor.has_value())
        {
            attnScale = resolveScalarOperand(*_params.scaleTensor, variantPack, "SDPA scale");
        }

        hipdnn_gpu_ref::GpuFpReferenceSdpa::
            fprop<QDataType, KDataType, VDataType, ODataType, ComputeDataType, MaskDataType>(
                qTensor,
                kTensor,
                vTensor,
                oTensor,
                hipdnn_gpu_ref::GpuSdpaFwdOptions<MaskDataType>{}
                    .setScale(attnScale)
                    .setMask(attnMaskTensor.has_value() ? &attnMaskTensor.value() : nullptr)
                    .setBand(_params.leftBound, _params.rightBound, _params.topLeftAlignment)
                    .setSeqLens(seqLenQTensor.has_value() ? &seqLenQTensor.value() : nullptr,
                                seqLenKvTensor.has_value() ? &seqLenKvTensor.value() : nullptr)
                    .setPaged(pageTableK.has_value() ? &pageTableK.value() : nullptr,
                              pageTableV.has_value() ? &pageTableV.value() : nullptr,
                              _params.pagedMaxSeqKv)
                    .setSink(sinkTensor.has_value() ? &sinkTensor.value() : nullptr)
                    .setDropout(dropoutMask.has_value() ? &dropoutMask.value() : nullptr,
                                dropoutScale)
                    .setAlibi(_params.alibi)
                    .setBlockMask(blockMask.has_value() ? &blockMask.value() : nullptr)
                    .setLse(lseTensor.has_value() ? &lseTensor.value() : nullptr)
                    .setRowStats(maxTensor.has_value() ? &maxTensor.value() : nullptr,
                                 sumExpTensor.has_value() ? &sumExpTensor.value() : nullptr)
                    .setProbabilityMode(REFERENCE_SDPA_PROBABILITY_MODE)
                    .setDescales(descaleQ, descaleK, descaleV)
                    .setSoftmaxScales(scalar(_params.scaleSTensor, "SDPA scale_s"),
                                      scalar(_params.descaleSTensor, "SDPA descale_s"))
                    .setOutputScale(scalar(_params.scaleOTensor, "SDPA scale_o"))
                    .setAmax(amaxS.has_value() ? &amaxS.value() : nullptr,
                             amaxO.has_value() ? &amaxO.value() : nullptr));
    }

private:
    GpuSdpaFwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType QDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType KDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType VDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ODataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType MaskDataTypeEnum
          = hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>
class GpuSdpaFwdPlanBuilder : public IGpuGraphNodePlanBuilder
{
public:
    using QDataType = hipdnn_test_sdk::utilities::DataTypeToNative<QDataTypeEnum>;
    using KDataType = hipdnn_test_sdk::utilities::DataTypeToNative<KDataTypeEnum>;
    using VDataType = hipdnn_test_sdk::utilities::DataTypeToNative<VDataTypeEnum>;
    using ODataType = hipdnn_test_sdk::utilities::DataTypeToNative<ODataTypeEnum>;
    // UNSET (no mask) maps to a placeholder float: the plan then always passes a null mask.
    using MaskDataType = hipdnn_test_sdk::utilities::MaskDataTypeToNative<MaskDataTypeEnum>;
    // fp8 inputs need the Q/K/V descales and allow the sdpa_fp8 scaling (isSupportedSdpaFwdNode).
    static constexpr bool IS_FP8_INPUT
        = QDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3
          || QDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E4M3_FNUZ
          || QDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2
          || QDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::FP8_E5M2_FNUZ;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        // The dense gate is shared with the CPU reference (isSupportedSdpaFwdNode). The GPU kernel
        // reads a paged container in place, so a paged side needs a scalar descale.
        const auto* nodeAttributes = node.attributes_as_SdpaAttributes();
        if(nodeAttributes == nullptr
           || !isSupportedSdpaFwdNode(*nodeAttributes,
                                      tensorMap,
                                      MaskDataTypeEnum,
                                      IS_FP8_INPUT,
                                      /*requireScalarPagedDescale=*/true))
        {
            return false;
        }

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->q_tensor_uid(), QDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->k_tensor_uid(), KDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->v_tensor_uid(), VDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->o_tensor_uid(), ODataTypeEnum);

        return true;
    }

    std::unique_ptr<IGpuGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_SdpaAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type SdpaAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();

        const auto* scalePtr = nodeAttributes->scale_tensor_uid().has_value()
                                   ? tensorMap.at(nodeAttributes->scale_tensor_uid().value())
                                   : nullptr;
        std::optional<float> attnScaleValue;
        if(nodeAttributes->attn_scale_value().has_value())
        {
            attnScaleValue = nodeAttributes->attn_scale_value();
        }
        auto scaleTensor = hipdnn_test_sdk::detail::foldSdpaScale(scalePtr, attnScaleValue);

        const auto* attnMaskPtr = nodeAttributes->attn_mask_tensor_uid().has_value()
                                      ? tensorMap.at(nodeAttributes->attn_mask_tensor_uid().value())
                                      : nullptr;

        const auto* lsePtr = nodeAttributes->stats_tensor_uid().has_value()
                                 ? tensorMap.at(nodeAttributes->stats_tensor_uid().value())
                                 : nullptr;

        const auto band = extractDiagonalBandParams(*nodeAttributes, "GpuSdpaFwdPlan");

        GpuSdpaFwdParams params(
            *tensorMap.at(nodeAttributes->q_tensor_uid()),
            *tensorMap.at(nodeAttributes->k_tensor_uid()),
            *tensorMap.at(nodeAttributes->v_tensor_uid()),
            *tensorMap.at(nodeAttributes->o_tensor_uid()),
            std::move(scaleTensor),
            band.leftBound,
            band.rightBound,
            band.topLeftAlignment,
            attnMaskPtr,
            lsePtr,
            nodeAttributes->seq_len_q_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->seq_len_q_tensor_uid().value())
                : nullptr,
            nodeAttributes->seq_len_kv_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->seq_len_kv_tensor_uid().value())
                : nullptr,
            nodeAttributes->page_table_k_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->page_table_k_tensor_uid().value())
                : nullptr,
            nodeAttributes->page_table_v_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->page_table_v_tensor_uid().value())
                : nullptr,
            sdpaPagedMaxSeqKv(*nodeAttributes, tensorMap),
            nodeAttributes->sink_token_tensor_uid().has_value()
                ? tensorMap.at(nodeAttributes->sink_token_tensor_uid().value())
                : nullptr);
        params.alibi = nodeAttributes->alibi_mask();
        for(auto [uid, bound] :
            {std::pair{nodeAttributes->descale_q_tensor_uid(), &params.descaleQTensor},
             std::pair{nodeAttributes->descale_k_tensor_uid(), &params.descaleKTensor},
             std::pair{nodeAttributes->descale_v_tensor_uid(), &params.descaleVTensor},
             std::pair{nodeAttributes->scale_s_tensor_uid(), &params.scaleSTensor},
             std::pair{nodeAttributes->descale_s_tensor_uid(), &params.descaleSTensor},
             std::pair{nodeAttributes->scale_o_tensor_uid(), &params.scaleOTensor},
             std::pair{nodeAttributes->amax_s_tensor_uid(), &params.amaxSTensor},
             std::pair{nodeAttributes->amax_o_tensor_uid(), &params.amaxOTensor},
             std::pair{nodeAttributes->block_mask_tensor_uid(), &params.blockMaskTensor}})
        {
            if(uid.has_value())
            {
                *bound
                    = hipdnn_test_sdk::detail::unpackTensorAttributes(*tensorMap.at(uid.value()));
            }
        }
        if(nodeAttributes->max_tensor_uid().has_value())
        {
            params.maxTensor = hipdnn_test_sdk::detail::unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->max_tensor_uid().value()));
        }
        if(nodeAttributes->sum_exp_tensor_uid().has_value())
        {
            params.sumExpTensor = hipdnn_test_sdk::detail::unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->sum_exp_tensor_uid().value()));
        }
        if(nodeAttributes->dropout_mask_tensor_uid().has_value())
        {
            params.dropoutMaskTensor = hipdnn_test_sdk::detail::unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->dropout_mask_tensor_uid().value()));
            params.dropoutScaleTensor = hipdnn_test_sdk::detail::unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->dropout_scale_tensor_uid().value()));
        }
        else if(isSdpaPhiloxDropoutWithDump(*nodeAttributes, tensorMap))
        {
            // The engine's dumped keep mask is the reference's explicit mask.
            params.dropoutMaskTensor = hipdnn_test_sdk::detail::unpackTensorAttributes(
                *tensorMap.at(nodeAttributes->rng_dump_tensor_uid().value()));
            params.philoxDropoutScale
                = 1.0F / (1.0F - nodeAttributes->dropout_probability().value());
        }
        return std::make_unique<
            GpuSdpaFwdPlan<QDataType, KDataType, VDataType, ODataType, float, MaskDataType>>(
            std::move(params));
    }
};

} // namespace hipdnn_integration_tests::gpu_graph_executor::detail
