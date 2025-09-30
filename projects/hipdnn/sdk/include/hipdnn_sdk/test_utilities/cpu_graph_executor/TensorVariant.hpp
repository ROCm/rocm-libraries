// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>
#include <hipdnn_sdk/utilities/UtilsBfp16.hpp>
#include <hipdnn_sdk/utilities/UtilsFp16.hpp>

#pragma once

namespace hipdnn_sdk
{
namespace test_utilities
{

using TensorVariant = std::variant<std::unique_ptr<hipdnn_sdk::utilities::TensorBase<float>>,
                                   std::unique_ptr<hipdnn_sdk::utilities::TensorBase<half>>>;

class TensorVariantUtils
{
public:
    static TensorVariant createHostOnlyShallowTensorVariant(
        const hipdnn_sdk::data_objects::TensorAttributes& tensorAttributes, void* ptr)
    {
        return createHostOnlyShallowTensorVariantInternal(
            tensorAttributes.data_type(),
            ptr,
            flatbufferVectorToStd(tensorAttributes.dims()),
            flatbufferVectorToStd(tensorAttributes.strides()));
    }

    static TensorVariant& unwrapToTensorVariant(std::any& a)
    {
        if(auto p = std::any_cast<TensorVariant>(&a))
        {
            return *p;
        }
        if(auto pr = std::any_cast<std::reference_wrapper<TensorVariant>>(&a))
        {
            return pr->get();
        }
        throw std::bad_any_cast();
    }

private:
    template <typename T>
    static std::unique_ptr<hipdnn_sdk::utilities::TensorBase<T>> createHostOnlyShallowTensor(
        void* ptr, const std::vector<int64_t>& dims, const std::vector<int64_t>& strides)
    {
        return std::make_unique<hipdnn_sdk::utilities::ShallowTensor<T>>(ptr, dims, strides);
    }

    static TensorVariant
        createHostOnlyShallowTensorVariantInternal(hipdnn_sdk::data_objects::DataType dataType,
                                                   void* ptr,
                                                   const std::vector<int64_t>& dims,
                                                   const std::vector<int64_t>& strides)
    {
        switch(dataType)
        {
        case hipdnn_sdk::data_objects::DataType::FLOAT:
            return createHostOnlyShallowTensor<float>(ptr, dims, strides);
        case hipdnn_sdk::data_objects::DataType::HALF:
            return createHostOnlyShallowTensor<half>(ptr, dims, strides);
        case hipdnn_sdk::data_objects::DataType::UNSET:
        case hipdnn_sdk::data_objects::DataType::BFLOAT16:
        case hipdnn_sdk::data_objects::DataType::DOUBLE:
        case hipdnn_sdk::data_objects::DataType::UINT8:
        case hipdnn_sdk::data_objects::DataType::INT32:
        default:
            break;
        }

        throw std::runtime_error("Unsupported data type for shallow tensor creation");
    }

    static std::vector<int64_t> flatbufferVectorToStd(const ::flatbuffers::Vector<int64_t>* fbVec)
    {
        std::vector<int64_t> result;
        if(fbVec == nullptr)
        {
            return result;
        }
        result.reserve(fbVec->size());
        for(auto v : *fbVec)
        {
            result.push_back(v);
        }
        return result;
    }
};

}
}
