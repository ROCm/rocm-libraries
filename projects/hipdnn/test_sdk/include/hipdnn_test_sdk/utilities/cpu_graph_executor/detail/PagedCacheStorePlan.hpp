// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_test_sdk/utilities/CpuFpReferencePagedCacheStore.hpp>
#include <hipdnn_test_sdk/utilities/FlatbufferDatatypeMapping.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanBuilder.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/IGraphNodePlanExecutor.hpp>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

#include <optional>
#include <stdexcept>

namespace hipdnn_test_sdk::detail
{

struct PagedCacheStoreParams
{
    PagedCacheStoreParams() = default;
    PagedCacheStoreParams(
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& xAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& slotMappingAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes& containerAttributes,
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scaleAttributes)
        : xTensor(unpackTensorAttributes(xAttributes))
        , slotMappingTensor(unpackTensorAttributes(slotMappingAttributes))
        , containerTensor(unpackTensorAttributes(containerAttributes))
    {
        if(scaleAttributes != nullptr)
        {
            scaleTensor = unpackTensorAttributes(*scaleAttributes);
        }
    }

    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT xTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT slotMappingTensor;
    hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT containerTensor;
    std::optional<hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT> scaleTensor;
};

template <typename XDataType,
          typename SlotDataType,
          typename ContainerDataType,
          typename ComputeDataType>
class PagedCacheStorePlan : public IGraphNodePlanExecutor
{
public:
    explicit PagedCacheStorePlan(PagedCacheStoreParams&& params)
        : _params(std::move(params))
    {
    }

    std::vector<int64_t> getOutputTensorIds() const override
    {
        return {_params.containerTensor.uid};
    }

    void execute(const std::unordered_map<int64_t, void*>& variantPack) override
    {
        auto x
            = createShallowTensor<XDataType>(_params.xTensor, variantPack.at(_params.xTensor.uid));
        auto slotMapping = createShallowTensor<SlotDataType>(
            _params.slotMappingTensor, variantPack.at(_params.slotMappingTensor.uid));
        auto container = createShallowTensor<ContainerDataType>(
            _params.containerTensor, variantPack.at(_params.containerTensor.uid));

        std::unique_ptr<hipdnn_data_sdk::utilities::ShallowTensor<float>> scale;
        if(_params.scaleTensor.has_value())
        {
            scale = createShallowTensor<float>(*_params.scaleTensor,
                                               variantPack.at(_params.scaleTensor->uid));
        }

        utilities::CpuFpReferencePagedCacheStore::
            store<XDataType, SlotDataType, ContainerDataType, ComputeDataType>(
                *x, *slotMapping, *container, scale.get());
    }

private:
    PagedCacheStoreParams _params;
};

template <hipdnn_flatbuffers_sdk::data_objects::DataType XDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType SlotDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ContainerDataTypeEnum,
          hipdnn_flatbuffers_sdk::data_objects::DataType ComputeDataTypeEnum>
class PagedCacheStorePlanBuilder : public IGraphNodePlanBuilder
{
public:
    using XDataType = utilities::DataTypeToNative<XDataTypeEnum>;
    using SlotDataType = utilities::DataTypeToNative<SlotDataTypeEnum>;
    using ContainerDataType = utilities::DataTypeToNative<ContainerDataTypeEnum>;
    using ComputeDataType = utilities::DataTypeToNative<ComputeDataTypeEnum>;

    bool isApplicable(
        const hipdnn_flatbuffers_sdk::data_objects::Node& node,
        const std::unordered_map<int64_t,
                                 const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes*>&
            tensorMap) const override
    {
        // No compute type check: the op has no compute-type attribute (lowered nodes carry
        // UNSET) and its arithmetic, x * scale, is FP32 by definition.
        static_assert(ComputeDataTypeEnum == DataType::FLOAT,
                      "PagedCacheStore computes in FP32 only");

        const auto* nodeAttributes = node.attributes_as_PagedCacheStoreAttributes();
        if(nodeAttributes == nullptr)
        {
            return false;
        }

        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->x_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->slot_mapping_tensor_uid());
        CHECK_TENSOR_EXISTS(tensorMap, nodeAttributes->container_tensor_uid());

        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->x_tensor_uid(), XDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->slot_mapping_tensor_uid(), SlotDataTypeEnum);
        CHECK_TENSOR_TYPE(tensorMap, nodeAttributes->container_tensor_uid(), ContainerDataTypeEnum);

        const auto scaleUid = nodeAttributes->scale_tensor_uid();
        if(scaleUid.has_value())
        {
            CHECK_OPTIONAL_TENSOR_EXISTS(tensorMap, scaleUid);
            CHECK_OPTIONAL_TENSOR_TYPE(tensorMap, scaleUid, DataType::FLOAT);
        }

        CHECK_NO_RAGGED_TENSORS(tensorMap);

        return true;
    }

    std::unique_ptr<IGraphNodePlanExecutor>
        buildNodePlan(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& graph,
                      const hipdnn_flatbuffers_sdk::data_objects::Node& node) const override
    {
        const auto* nodeAttributes = node.attributes_as_PagedCacheStoreAttributes();
        if(nodeAttributes == nullptr)
        {
            throw std::runtime_error("Node attributes are not of type PagedCacheStoreAttributes");
        }

        const auto& tensorMap = graph.getTensorMap();
        const auto scaleUid = nodeAttributes->scale_tensor_uid();
        const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes* scaleAttributes
            = scaleUid.has_value() ? tensorMap.at(*scaleUid) : nullptr;

        return std::make_unique<
            PagedCacheStorePlan<XDataType, SlotDataType, ContainerDataType, ComputeDataType>>(
            PagedCacheStoreParams(*tensorMap.at(nodeAttributes->x_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->slot_mapping_tensor_uid()),
                                  *tensorMap.at(nodeAttributes->container_tensor_uid()),
                                  scaleAttributes));
    }

private:
    using DataType = hipdnn_flatbuffers_sdk::data_objects::DataType;
};

} // namespace hipdnn_test_sdk::detail
