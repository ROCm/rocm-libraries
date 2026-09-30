// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <variant>
#include <vector>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceResampleFwd.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

struct ResampleFwdParams
{
    ResampleFwdParams(
        const hipdnn_flatbuffers_sdk::data_objects::ResampleFwdAttributes& resampleAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& yAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* indexAttributes = nullptr)
        : xTensor(unpackTensorAttributes(xAttributes))
        , yTensor(unpackTensorAttributes(yAttributes))
        , indexTensor(indexAttributes != nullptr
                          ? std::make_optional(unpackTensorAttributes(*indexAttributes))
                          : std::nullopt)
        , prePadding(resampleAttributes.pre_padding()->begin(),
                     resampleAttributes.pre_padding()->end())
        , stride(resampleAttributes.stride()->begin(), resampleAttributes.stride()->end())
        , window(resampleAttributes.window()->begin(), resampleAttributes.window()->end())
        , mode(resampleAttributes.resample_mode())
        , paddingMode(resampleAttributes.padding_mode())
        , generateIndex(indexAttributes != nullptr)
    {
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT yTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> indexTensor;
    std::vector<int64_t> prePadding;
    std::vector<int64_t> stride;
    std::vector<int64_t> window;
    hipdnn_flatbuffers_sdk::data_objects::ResampleMode mode;
    hipdnn_flatbuffers_sdk::data_objects::PaddingMode paddingMode;
    bool generateIndex;
};

template <typename XDataType,
          typename YDataType,
          typename ComputeDataType,
          typename IndexDataType = int32_t>
class ResampleFwdPlan : public IGraphNodePlanExecutor
{
public:
    explicit ResampleFwdPlan(ResampleFwdParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        std::vector<int64_t> ids = {_params.yTensor.uid};
        if(_params.indexTensor.has_value())
        {
            ids.push_back(_params.indexTensor->uid);
        }
        return ids;
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto shallowXTensor
            = createShallowTensor<XDataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));
        auto shallowYTensor
            = createShallowTensor<YDataType>(_params.yTensor, variantPack.at(_params.yTensor.uid));

        std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<IndexDataType>> shallowIndexTensor;
        if(_params.generateIndex)
        {
            shallowIndexTensor = createShallowTensor<IndexDataType>(
                *_params.indexTensor, variantPack.at(_params.indexTensor->uid));
        }

        utilities::CpuFpReferenceResampleFwd::
            forward<XDataType, YDataType, ComputeDataType, IndexDataType>(*shallowXTensor,
                                                                          *shallowYTensor,
                                                                          _params.prePadding,
                                                                          _params.stride,
                                                                          _params.window,
                                                                          _params.mode,
                                                                          _params.paddingMode,
                                                                          shallowIndexTensor.get());
    }

private:
    ResampleFwdParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType IndexDataTypeEnum>
struct ResampleIndexNative
{
    using type = utilities::DataTypeToNative<IndexDataTypeEnum>;
};

template <>
struct ResampleIndexNative<hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET>
{
    using type = int32_t;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType YDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType IndexDataTypeEnum>
class ResampleFwdPlanBuilder : public IGraphNodePlanBuilder
{
public:
    using XDataType = utilities::DataTypeToNative<XDataTypeEnum>;
    using YDataType = utilities::DataTypeToNative<YDataTypeEnum>;
    using ComputeDataType = utilities::DataTypeToNative<ComputeDataTypeEnum>;
    using IndexDataType = typename ResampleIndexNative<IndexDataTypeEnum>::type;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        if(node.compute_data_type() != ComputeDataTypeEnum)
        {
            return false;
        }

        const auto* nodeAttributes = node.attributes_as_ResampleFwdAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        const auto mode = nodeAttributes->resample_mode();
        if(mode != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::MAXPOOL
           && mode != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_EXCLUDE_PADDING
           && mode != hipdnn_flatbuffers_sdk::data_objects::ResampleMode::AVGPOOL_INCLUDE_PADDING)
        {
            return false;
        }

        const auto paddingMode = nodeAttributes->padding_mode();
        if(paddingMode != hipdnn_flatbuffers_sdk::data_objects::PaddingMode::NEG_INF_PAD
           && paddingMode != hipdnn_flatbuffers_sdk::data_objects::PaddingMode::ZERO_PAD)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->y_tensor_uid());
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->y_tensor_uid(), YDataTypeEnum);

        if constexpr(IndexDataTypeEnum == hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET)
        {
            if(nodeAttributes->index_tensor_uid().has_value())
            {
                return false;
            }
        }
        else
        {
            if(!nodeAttributes->index_tensor_uid().has_value())
            {
                return false;
            }
            CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->index_tensor_uid().value());
            CHECK_TENSOR_TYPE(
                tensorMap, nodeAttributes->index_tensor_uid().value(), IndexDataTypeEnum);
        }

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_ResampleFwdAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type ResampleFwdAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        const auto* indexPtr = nodeAttributes->index_tensor_uid().has_value()
                                   ? tensorMap.at(nodeAttributes->index_tensor_uid().value())
                                   : nullptr;

        return std::make_unique<
            ResampleFwdPlan<XDataType, YDataType, ComputeDataType, IndexDataType>>(
            ResampleFwdParams(*nodeAttributes,
                              *tensorMap.at(nodeAttributes->x_tensor_uid()),
                              *tensorMap.at(nodeAttributes->y_tensor_uid()),
                              indexPtr));
    }
};

} // namespace hipdnn_test_sdk::detail
