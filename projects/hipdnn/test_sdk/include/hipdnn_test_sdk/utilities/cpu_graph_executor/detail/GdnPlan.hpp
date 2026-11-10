// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceGdn.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace hipdnn_test_sdk::detail
{

struct GdnParams
{
    using TensorAttributes = hipdnn_flatbuffers_sdk::data_objects::TensorAttributes;
    using TensorAttributesT = hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT;

    GdnParams() = default;
    GdnParams(const hipdnn_flatbuffers_sdk::data_objects::GdnAttributes& attributes,
              const std::unordered_map<int64_t, const TensorAttributes*>& tensorMap)
        : qTensor(unpackTensorAttributes(*tensorMap.at(attributes.q_tensor_uid())))
        , kTensor(unpackTensorAttributes(*tensorMap.at(attributes.k_tensor_uid())))
        , vTensor(unpackTensorAttributes(*tensorMap.at(attributes.v_tensor_uid())))
        , gTensor(unpackTensorAttributes(*tensorMap.at(attributes.g_tensor_uid())))
        , betaTensor(unpackTensorAttributes(*tensorMap.at(attributes.beta_tensor_uid())))
        , cuSeqlensTensor(unpackTensorAttributes(*tensorMap.at(attributes.cu_seqlens_tensor_uid())))
        , oTensor(unpackTensorAttributes(*tensorMap.at(attributes.o_tensor_uid())))
        , initialStateTensor(unpackOptional(attributes.initial_state_tensor_uid(), tensorMap))
        , aLogTensor(unpackOptional(attributes.a_log_tensor_uid(), tensorMap))
        , dtBiasTensor(unpackOptional(attributes.dt_bias_tensor_uid(), tensorMap))
        , stateIndicesTensor(unpackOptional(attributes.state_indices_tensor_uid(), tensorMap))
        , finalStateTensor(unpackOptional(attributes.final_state_tensor_uid(), tensorMap))
        , stateCheckpointsTensor(
              unpackOptional(attributes.state_checkpoints_tensor_uid(), tensorMap))
    {
        if(attributes.scale().has_value())
        {
            config.scale = *attributes.scale();
        }
        config.useQkL2norm = attributes.use_qk_l2norm();
        config.checkpointEveryNTokens = attributes.checkpoint_every_n_tokens();
        config.useBetaSigmoid = attributes.use_beta_sigmoid();
        config.allowNegEigval = attributes.allow_neg_eigval();
        config.safeGate = attributes.safe_gate();
        config.gateDomain = attributes.gate_domain();
    }

    TensorAttributesT qTensor;
    TensorAttributesT kTensor;
    TensorAttributesT vTensor;
    TensorAttributesT gTensor;
    TensorAttributesT betaTensor;
    TensorAttributesT cuSeqlensTensor;
    TensorAttributesT oTensor;
    std::optional<TensorAttributesT> initialStateTensor;
    std::optional<TensorAttributesT> aLogTensor;
    std::optional<TensorAttributesT> dtBiasTensor;
    std::optional<TensorAttributesT> stateIndicesTensor;
    std::optional<TensorAttributesT> finalStateTensor;
    std::optional<TensorAttributesT> stateCheckpointsTensor;
    utilities::GdnConfig config;

private:
    static std::optional<TensorAttributesT>
        unpackOptional(const ::flatbuffers::Optional<int64_t>& uid,
                       const std::unordered_map<int64_t, const TensorAttributes*>& tensorMap)
    {
        if(!uid.has_value())
        {
            return std::nullopt;
        }
        return unpackTensorAttributes(*tensorMap.at(*uid));
    }
};

/// Gates, states and checkpoints each carry their own FLOAT, HALF or BFLOAT16 type. The plan
/// stages them through packed FLOAT copies (exact for every supported type), so the reference
/// runs with AuxType = float and each output is rounded once, to its own type, on store.
class GdnOperandStaging
{
public:
    using FloatTensor = hipdnn_data_sdk::utilities::Tensor<float>;
    using TensorAttributesT = hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT;
    using VariantPack = std::unordered_map<int64_t, void*>;

    static bool isFloatType(hipdnn_flatbuffers_sdk::data_objects::DataType dataType)
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;
        return dataType == DataType::FLOAT || dataType == DataType::HALF
               || dataType == DataType::BFLOAT16;
    }

    static std::unique_ptr<FloatTensor> load(const TensorAttributesT& operand,
                                             const VariantPack& variantPack)
    {
        auto staged = std::make_unique<FloatTensor>(operand.dims);
        dispatch(operand.data_type, [&](auto tag) {
            using T = typename decltype(tag)::type;
            auto source = bindShallowTensor<T>(operand, variantPack);
            forEachIndex(operand.dims, [&](const std::vector<int64_t>& index) {
                staged->setHostValue(static_cast<float>(source->getHostValue(index)), index);
            });
        });
        return staged;
    }

    static std::unique_ptr<FloatTensor>
        loadOptional(const std::optional<TensorAttributesT>& operand,
                     const VariantPack& variantPack)
    {
        return operand.has_value() ? load(*operand, variantPack) : nullptr;
    }

    static void store(const FloatTensor& staged,
                      const TensorAttributesT& operand,
                      const VariantPack& variantPack)
    {
        dispatch(operand.data_type, [&](auto tag) {
            using T = typename decltype(tag)::type;
            auto target = bindShallowTensor<T>(operand, variantPack);
            forEachIndex(operand.dims, [&](const std::vector<int64_t>& index) {
                target->setHostValue(static_cast<T>(staged.getHostValue(index)), index);
            });
            target->memory().markHostModified();
        });
    }

private:
    template <class T>
    struct TypeTag
    {
        using type = T;
    };

    template <class Fn>
    static void dispatch(hipdnn_flatbuffers_sdk::data_objects::DataType dataType, Fn&& fn)
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;
        switch(dataType)
        {
        case DataType::FLOAT:
            fn(TypeTag<float>{});
            return;
        case DataType::HALF:
            fn(TypeTag<utilities::half>{});
            return;
        case DataType::BFLOAT16:
            fn(TypeTag<utilities::bfloat16>{});
            return;
        default:
            throw std::runtime_error("GDN gate, state and checkpoint operands must be FLOAT, HALF "
                                     "or BFLOAT16");
        }
    }

    // Visits every index of `dims` in row-major order.
    template <class Fn>
    static void forEachIndex(const std::vector<int64_t>& dims, Fn&& fn)
    {
        for(const auto dim : dims)
        {
            if(dim <= 0)
            {
                return;
            }
        }
        std::vector<int64_t> index(dims.size(), 0);
        while(true)
        {
            fn(index);
            auto d = dims.size();
            while(d > 0)
            {
                --d;
                if(++index[d] < dims[d])
                {
                    break;
                }
                index[d] = 0;
                if(d == 0)
                {
                    return;
                }
            }
            if(dims.empty())
            {
                return;
            }
        }
    }
};

template <typename IoDataType, typename SeqlensDataType, typename ComputeDataType>
class GdnPlan : public IGraphNodePlanExecutor
{
public:
    explicit GdnPlan(GdnParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        std::vector<int64_t> ids{_params.oTensor.uid};
        if(_params.finalStateTensor.has_value())
        {
            ids.push_back(_params.finalStateTensor->uid);
        }
        if(_params.stateCheckpointsTensor.has_value())
        {
            ids.push_back(_params.stateCheckpointsTensor->uid);
        }
        return ids;
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto q = bindShallowTensor<IoDataType>(_params.qTensor, variantPack);
        auto k = bindShallowTensor<IoDataType>(_params.kTensor, variantPack);
        auto v = bindShallowTensor<IoDataType>(_params.vTensor, variantPack);
        auto o = bindShallowTensor<IoDataType>(_params.oTensor, variantPack);
        auto cuSeqlens = bindShallowTensor<SeqlensDataType>(_params.cuSeqlensTensor, variantPack);
        auto stateIndices
            = bindOptionalShallowTensor<int32_t>(_params.stateIndicesTensor, variantPack);

        // Every input, and the current contents of the outputs (pool rows and checkpoint rows
        // the node does not write stay as they are), is staged before the reference runs, so
        // final_state may share initial_state's buffer.
        auto g = GdnOperandStaging::load(_params.gTensor, variantPack);
        auto beta = GdnOperandStaging::load(_params.betaTensor, variantPack);
        auto initialState
            = GdnOperandStaging::loadOptional(_params.initialStateTensor, variantPack);
        auto aLog = GdnOperandStaging::loadOptional(_params.aLogTensor, variantPack);
        auto dtBias = GdnOperandStaging::loadOptional(_params.dtBiasTensor, variantPack);
        auto finalState = GdnOperandStaging::loadOptional(_params.finalStateTensor, variantPack);
        auto stateCheckpoints
            = GdnOperandStaging::loadOptional(_params.stateCheckpointsTensor, variantPack);

        utilities::GdnTensors<IoDataType, float, SeqlensDataType, int32_t> tensors;
        tensors.q = q.get();
        tensors.k = k.get();
        tensors.v = v.get();
        tensors.g = g.get();
        tensors.beta = beta.get();
        tensors.cuSeqlens = cuSeqlens.get();
        tensors.o = o.get();
        tensors.initialState = initialState.get();
        tensors.aLog = aLog.get();
        tensors.dtBias = dtBias.get();
        tensors.stateIndices = stateIndices.get();
        tensors.finalState = finalState.get();
        tensors.stateCheckpoints = stateCheckpoints.get();

        utilities::CpuFpReferenceGdn::forward<ComputeDataType>(tensors, _params.config);

        if(finalState != nullptr)
        {
            GdnOperandStaging::store(*finalState, *_params.finalStateTensor, variantPack);
        }
        if(stateCheckpoints != nullptr)
        {
            GdnOperandStaging::store(
                *stateCheckpoints, *_params.stateCheckpointsTensor, variantPack);
        }
    }

private:
    GdnParams _params;
};

/// Builds a GdnPlan for one io type (q, k, v, o) and cu_seqlens type. Gates, states and
/// checkpoints may each be FLOAT, HALF or BFLOAT16; state_indices is INT32 (as cuDNN).
template <hipdnn_flatbuffers_sdk::data_objects::DataType IoDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType SeqlensDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class GdnPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using IoDataType = utilities::DataTypeToNative<IoDataTypeEnum>;
    using SeqlensDataType
        = std::conditional_t<SeqlensDataTypeEnum
                                 == hipdnn_flatbuffers_sdk::data_objects::DataType::INT64,
                             int64_t,
                             int32_t>;
    using ComputeDataType = utilities::DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        using hipdnn_flatbuffers_sdk::data_objects::DataType;

        if(node.compute_data_type() != ComputeDataTypeEnum)
        {
            return false;
        }

        const auto* nodeAttributes = node.attributes_as_GdnAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        for(const auto uid : {nodeAttributes->q_tensor_uid(),
                              nodeAttributes->k_tensor_uid(),
                              nodeAttributes->v_tensor_uid(),
                              nodeAttributes->o_tensor_uid()})
        {
            CHECK_TENSOR_EXISTS(tensorMap, uid);
            CHECK_TENSOR_TYPE(tensorMap, uid, IoDataTypeEnum);
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->cu_seqlens_tensor_uid());
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->cu_seqlens_tensor_uid(), SeqlensDataTypeEnum);

        for(const auto uid : {nodeAttributes->g_tensor_uid(), nodeAttributes->beta_tensor_uid()})
        {
            CHECK_TENSOR_EXISTS(tensorMap, uid);
            if(!GdnOperandStaging::isFloatType(tensorMap.at(uid)->data_type()))
            {
                return false;
            }
        }

        for(const auto& uid : {nodeAttributes->initial_state_tensor_uid(),
                               nodeAttributes->a_log_tensor_uid(),
                               nodeAttributes->dt_bias_tensor_uid(),
                               nodeAttributes->final_state_tensor_uid(),
                               nodeAttributes->state_checkpoints_tensor_uid()})
        {
            if(uid.has_value())
            {
                CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, uid);
                if(!GdnOperandStaging::isFloatType(tensorMap.at(*uid)->data_type()))
                {
                    return false;
                }
            }
        }

        const auto stateIndicesUid = nodeAttributes->state_indices_tensor_uid();
        if(stateIndicesUid.has_value())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, stateIndicesUid);
            CHECK_OPTIONAL_TENSOR_TYPE(tensorMap, stateIndicesUid, DataType::INT32);
        }

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_GdnAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type GdnAttributes");
        }

        return std::make_unique<GdnPlan<IoDataType, SeqlensDataType, ComputeDataType>>(
            GdnParams(*nodeAttributes, graph.getTensorMap()));
    }
};

} // namespace hipdnn_test_sdk::detail
