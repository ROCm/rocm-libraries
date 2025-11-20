// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
{
class ConvDgradAttributes : public Attributes<ConvDgradAttributes>
{
public:
    enum class InputNames
    {
        DY = 0, // Gradient of output tensor
        W = 1 // Weights/filter tensor
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        DX = 0 // Gradient of input tensor
    };
    typedef OutputNames output_names; // NOLINT(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;

    std::vector<int64_t> pre_padding; // NOLINT(readability-identifier-naming)
    std::vector<int64_t> post_padding; // NOLINT(readability-identifier-naming)
    std::vector<int64_t> stride;
    std::vector<int64_t> dilation;
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvolutionMode math_mode = ConvolutionMode::CROSS_CORRELATION;

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dy() const
    {
        return getInput(InputNames::DY);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_w() const
    {
        return getInput(InputNames::W);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dx() const
    {
        return getOutput(OutputNames::DX);
    }

    // Setters for tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_dy(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DY, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_dy(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DY, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_w(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::W, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_w(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::W, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_dx(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DX, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_dx(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DX, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_padding(std::vector<int64_t> padding)
    {
        set_pre_padding(padding);
        set_post_padding(std::move(padding));
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_pre_padding(const std::vector<int64_t>& padding)
    {
        pre_padding = padding;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_pre_padding(std::vector<int64_t>&& padding)
    {
        pre_padding = std::move(padding);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_post_padding(const std::vector<int64_t>& padding)
    {
        post_padding = padding;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_post_padding(std::vector<int64_t>&& padding)
    {
        post_padding = std::move(padding);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_stride(const std::vector<int64_t>& strideVals)
    {
        stride = strideVals;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_stride(std::vector<int64_t>&& strideVals)
    {
        stride = std::move(strideVals);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_dilation(const std::vector<int64_t>& dilationVals)
    {
        dilation = dilationVals;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_dilation(std::vector<int64_t>&& dilationVals)
    {
        dilation = std::move(dilationVals);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvDgradAttributes& set_convolution_mode(ConvolutionMode mode)
    {
        math_mode = mode;
        return *this;
    }

    // Getters for convolution parameters
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_pre_padding() const
    {
        return pre_padding;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_post_padding() const
    {
        return post_padding;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_stride() const
    {
        return stride;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::vector<int64_t>& get_dilation() const
    {
        return dilation;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvolutionMode get_convolution_mode() const
    {
        return math_mode;
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::ConvolutionBwdAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        return hipdnn_sdk::data_objects::CreateConvolutionBwdAttributesDirect(builder,
                                                                              get_dy()->get_uid(),
                                                                              get_w()->get_uid(),
                                                                              get_dx()->get_uid(),
                                                                              &pre_padding,
                                                                              &post_padding,
                                                                              &stride,
                                                                              &dilation,
                                                                              toSdkType(math_mode));
    }

private:
    std::shared_ptr<TensorAttributes> getInput(InputNames name) const
    {
        auto it = inputs.find(name);
        if(it != inputs.end())
        {
            return it->second;
        }
        return nullptr;
    }

    std::shared_ptr<TensorAttributes> getOutput(OutputNames name) const
    {
        auto it = outputs.find(name);
        if(it != outputs.end())
        {
            return it->second;
        }
        return nullptr;
    }

    ConvDgradAttributes& setInput(InputNames name, const std::shared_ptr<TensorAttributes>& value)
    {
        inputs[name] = value;
        return *this;
    }

    ConvDgradAttributes& setInput(InputNames name, std::shared_ptr<TensorAttributes>&& value)
    {
        inputs[name] = std::move(value);
        return *this;
    }

    ConvDgradAttributes& setOutput(OutputNames name, const std::shared_ptr<TensorAttributes>& value)
    {
        outputs[name] = value;
        return *this;
    }

    ConvDgradAttributes& setOutput(OutputNames name, std::shared_ptr<TensorAttributes>&& value)
    {
        outputs[name] = std::move(value);
        return *this;
    }
};
typedef ConvDgradAttributes Conv_dgrad_attributes;
}
}
