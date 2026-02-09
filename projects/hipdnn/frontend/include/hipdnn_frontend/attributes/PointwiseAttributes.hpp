// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend::graph
{
class PointwiseAttributes : public Attributes<PointwiseAttributes>
{
public:
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseMode get_mode() const
    {
        return mode;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_relu_lower_clip() const
    {
        return relu_lower_clip;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_relu_upper_clip() const
    {
        return relu_upper_clip;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_relu_lower_clip_slope() const
    {
        return relu_lower_clip_slope;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_swish_beta() const
    {
        return swish_beta;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_elu_alpha() const
    {
        return elu_alpha;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_softplus_beta() const
    {
        return softplus_beta;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<int64_t> get_axis() const
    {
        return axis;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_input_0() const
    {
        return getInput(InputNames::IN_0);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_input_1() const
    {
        return getInput(InputNames::IN_1);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_input_2() const
    {
        return getInput(InputNames::IN_2);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_output_0() const
    {
        return getOutput(OutputNames::OUT_0);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_mode(PointwiseMode value)
    {
        mode = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_relu_lower_clip(float value)
    {
        relu_lower_clip = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_relu_upper_clip(float value)
    {
        relu_upper_clip = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_relu_lower_clip_slope(float value)
    {
        relu_lower_clip_slope = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_swish_beta(float value)
    {
        swish_beta = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_elu_alpha(float value)
    {
        elu_alpha = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_softplus_beta(float value)
    {
        softplus_beta = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_axis(int64_t value)
    {
        axis = value;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_0(const std::shared_ptr<TensorAttributes>& input0)
    {
        return setInput(InputNames::IN_0, input0);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_0(std::shared_ptr<TensorAttributes>&& input0)
    {
        return setInput(InputNames::IN_0, std::move(input0));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_1(const std::shared_ptr<TensorAttributes>& input1)
    {
        return setInput(InputNames::IN_1, input1);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_1(std::shared_ptr<TensorAttributes>&& input1)
    {
        return setInput(InputNames::IN_1, std::move(input1));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_2(const std::shared_ptr<TensorAttributes>& input2)
    {
        return setInput(InputNames::IN_2, input2);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_2(std::shared_ptr<TensorAttributes>&& input2)
    {
        return setInput(InputNames::IN_2, std::move(input2));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_output_0(const std::shared_ptr<TensorAttributes>& output0)
    {
        return setOutput(OutputNames::OUT_0, output0);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_output_0(std::shared_ptr<TensorAttributes>&& output0)
    {
        return setOutput(OutputNames::OUT_0, std::move(output0));
    }

    enum class InputNames
    {
        IN_0 = 0,
        IN_1 = 1,
        IN_2 = 2,
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        OUT_0 = 0,
    };
    typedef OutputNames output_names; // NOLINT(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;

    // NOLINTBEGIN(readability-identifier-naming)
    PointwiseMode mode = PointwiseMode::NOT_SET;
    std::optional<float> relu_lower_clip = std::nullopt;
    std::optional<float> relu_upper_clip = std::nullopt;
    std::optional<float> relu_lower_clip_slope = std::nullopt;
    std::optional<int64_t> axis = std::nullopt;
    std::optional<float> swish_beta = std::nullopt;
    std::optional<float> elu_alpha = std::nullopt;
    std::optional<float> softplus_beta = std::nullopt;
    // NOLINTEND(readability-identifier-naming)

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::PointwiseAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto in0 = get_input_0();
        auto in1 = get_input_1();
        auto in2 = get_input_2();
        auto ot0 = get_output_0();

        return hipdnn_data_sdk::data_objects::CreatePointwiseAttributes(
            builder,
            toSdkType(mode),
            relu_lower_clip,
            relu_upper_clip,
            relu_lower_clip_slope,
            axis,
            in0->get_uid(),
            in1 ? flatbuffers::Optional<int64_t>(in1->get_uid()) : flatbuffers::nullopt,
            in2 ? flatbuffers::Optional<int64_t>(in2->get_uid()) : flatbuffers::nullopt,
            ot0->get_uid(),
            swish_beta,
            elu_alpha,
            softplus_beta);
    }

    static PointwiseAttributes fromFlatBuffer(
        const hipdnn_data_sdk::data_objects::PointwiseAttributes* fb,
        const std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap)
    {
        PointwiseAttributes attr;

        attr.set_mode(fromSdkType(fb->operation()));

        if(fb->relu_lower_clip().has_value())
        {
            attr.set_relu_lower_clip(fb->relu_lower_clip().value());
        }
        if(fb->relu_upper_clip().has_value())
        {
            attr.set_relu_upper_clip(fb->relu_upper_clip().value());
        }
        if(fb->relu_lower_clip_slope().has_value())
        {
            attr.set_relu_lower_clip_slope(fb->relu_lower_clip_slope().value());
        }
        if(fb->swish_beta().has_value())
        {
            attr.set_swish_beta(fb->swish_beta().value());
        }
        if(fb->elu_alpha().has_value())
        {
            attr.set_elu_alpha(fb->elu_alpha().value());
        }
        if(fb->softplus_beta().has_value())
        {
            attr.set_softplus_beta(fb->softplus_beta().value());
        }
        if(fb->axis_tensor_uid().has_value())
        {
            attr.set_axis(fb->axis_tensor_uid().value());
        }

        attr.set_input_0(tensorMap.at(fb->in_0_tensor_uid()));
        if(fb->in_1_tensor_uid().has_value())
        {
            attr.set_input_1(tensorMap.at(fb->in_1_tensor_uid().value()));
        }
        if(fb->in_2_tensor_uid().has_value())
        {
            attr.set_input_2(tensorMap.at(fb->in_2_tensor_uid().value()));
        }

        attr.set_output_0(tensorMap.at(fb->out_0_tensor_uid()));

        return attr;
    }
};
typedef PointwiseAttributes Pointwise_attributes;
} // namespace hipdnn_frontend::graph
