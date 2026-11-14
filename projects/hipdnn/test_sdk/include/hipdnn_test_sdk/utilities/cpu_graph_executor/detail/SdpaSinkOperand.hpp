// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>
#include <vector>

#include <hipdnn_data_sdk/utilities/ShallowTensor.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>
#include <hipdnn_test_sdk/utilities/detail/FlatbufferTensorAttributesUtils.hpp>

namespace hipdnn_test_sdk::detail
{

/// The attention sink of an SDPA node as the FLOAT [1, H, 1, 1] tensor the CPU reference reads.
/// A sink is FLOAT or Q's data type (NativeType); the Q-dtype one widens to float exactly, so
/// the reference sees the same logit the engine does.
template <typename NativeType>
class SdpaSinkOperand
{
public:
    SdpaSinkOperand(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& attrs,
                    void* memory)
    {
        if(attrs.data_type == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
        {
            _view = createShallowTensor<float>(attrs, memory);
            return;
        }
        const auto source = createShallowTensor<NativeType>(attrs, memory);
        auto widened = std::make_unique<hipdnn_data_sdk::utilities::Tensor<float>>(attrs.dims);
        for(int64_t h = 0; h < attrs.dims.at(1); ++h)
        {
            const std::vector<int64_t> index{0, h, 0, 0};
            widened->setHostValue(static_cast<float>(source->getHostValue(index)), index);
        }
        _view = std::move(widened);
    }

    hipdnn_data_sdk::utilities::TensorBase<float>* tensor() const
    {
        return _view.get();
    }

private:
    std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> _view;
};

/// The sink gradient of an SDPA backward node: the reference writes the FLOAT [1, H, 1, 1] tensor
/// tensor() and commit() stores it in the output's dtype, rounding once (a FLOAT dSink is written
/// in place and commit() does nothing).
template <typename NativeType>
class SdpaDSinkOperand
{
public:
    SdpaDSinkOperand(const hipdnn_flatbuffers_sdk::data_objects::TensorAttributesT& attrs,
                     void* memory)
    {
        if(attrs.data_type == hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT)
        {
            _view = createShallowTensor<float>(attrs, memory);
            return;
        }
        _output = createShallowTensor<NativeType>(attrs, memory);
        _view = std::make_unique<hipdnn_data_sdk::utilities::Tensor<float>>(attrs.dims);
    }

    hipdnn_data_sdk::utilities::TensorBase<float>* tensor() const
    {
        return _view.get();
    }

    void commit()
    {
        if(_output == nullptr)
        {
            return;
        }
        for(int64_t h = 0; h < _view->dims().at(1); ++h)
        {
            const std::vector<int64_t> index{0, h, 0, 0};
            _output->setHostValue(safeConvert<NativeType>(_view->getHostValue(index)), index);
        }
        _output->memory().markHostModified();
    }

private:
    std::unique_ptr<hipdnn_data_sdk::utilities::TensorBase<float>> _view;
    std::unique_ptr<hipdnn_data_sdk::utilities::ShallowTensor<NativeType>> _output;
};

} // namespace hipdnn_test_sdk::detail
