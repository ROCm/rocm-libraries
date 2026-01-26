// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_data_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::graph
{
class ConvWgradAttributes : public Attributes<ConvWgradAttributes>
{
public:
    enum class InputNames
    {
        DY = 0, // Gradient of output tensor
        X = 1 // Input tensor
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        DW = 0 // Gradient of weights tensor
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
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dy() const
    {
        return getInput(InputNames::DY);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_dw() const
    {
        return getOutput(OutputNames::DW);
    }

    // Setters for tensor
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_dy(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::DY, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_dy(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::DY, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_dw(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::DW, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_dw(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::DW, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_padding(std::vector<int64_t> padding)
    {
        set_pre_padding(padding);
        set_post_padding(std::move(padding));
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_pre_padding(const std::vector<int64_t>& padding)
    {
        pre_padding = padding;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_pre_padding(std::vector<int64_t>&& padding)
    {
        pre_padding = std::move(padding);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_post_padding(const std::vector<int64_t>& padding)
    {
        post_padding = padding;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_post_padding(std::vector<int64_t>&& padding)
    {
        post_padding = std::move(padding);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_stride(const std::vector<int64_t>& strideVals)
    {
        stride = strideVals;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_stride(std::vector<int64_t>&& strideVals)
    {
        stride = std::move(strideVals);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_dilation(const std::vector<int64_t>& dilationVals)
    {
        dilation = dilationVals;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_dilation(std::vector<int64_t>&& dilationVals)
    {
        dilation = std::move(dilationVals);
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    ConvWgradAttributes& set_convolution_mode(ConvolutionMode mode)
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

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        return hipdnn_data_sdk::data_objects::CreateConvolutionWrwAttributesDirect(
            builder,
            get_x()->get_uid(),
            get_dy()->get_uid(),
            get_dw()->get_uid(),
            &pre_padding,
            &post_padding,
            &stride,
            &dilation,
            toSdkType(math_mode));
    }

    static ConvWgradAttributes fromFlatBuffer(
        const hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes* fb,
        const std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap)
    {
        ConvWgradAttributes attr;

        attr.set_x(tensorMap.at(fb->x_tensor_uid()));
        attr.set_dy(tensorMap.at(fb->dy_tensor_uid()));
        attr.set_dw(tensorMap.at(fb->dw_tensor_uid()));

        if(fb->pre_padding() != nullptr)
        {
            std::vector<int64_t> prePadding(fb->pre_padding()->begin(), fb->pre_padding()->end());
            attr.set_pre_padding(std::move(prePadding));
        }
        if(fb->post_padding() != nullptr)
        {
            std::vector<int64_t> postPadding(fb->post_padding()->begin(),
                                             fb->post_padding()->end());
            attr.set_post_padding(std::move(postPadding));
        }
        if(fb->stride() != nullptr)
        {
            std::vector<int64_t> strideVec(fb->stride()->begin(), fb->stride()->end());
            attr.set_stride(std::move(strideVec));
        }
        if(fb->dilation() != nullptr)
        {
            std::vector<int64_t> dilationVec(fb->dilation()->begin(), fb->dilation()->end());
            attr.set_dilation(std::move(dilationVec));
        }
        attr.set_convolution_mode(fromSdkType(fb->conv_mode()));

        return attr;
    }
};
typedef ConvWgradAttributes Conv_wgrad_attributes;
} // namespace hipdnn_frontend::graph
