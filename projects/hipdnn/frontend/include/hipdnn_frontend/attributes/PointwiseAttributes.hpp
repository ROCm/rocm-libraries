// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_frontend/Types.hpp>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <memory>
#include <optional>
#include <unordered_map>

namespace hipdnn_frontend
{
namespace graph
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
        inputs[InputNames::IN_0] = input0;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_0(std::shared_ptr<TensorAttributes>&& input0)
    {
        inputs[InputNames::IN_0] = std::move(input0);
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_1(const std::shared_ptr<TensorAttributes>& input1)
    {
        inputs[InputNames::IN_1] = input1;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_1(std::shared_ptr<TensorAttributes>&& input1)
    {
        inputs[InputNames::IN_1] = std::move(input1);
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_2(const std::shared_ptr<TensorAttributes>& input2)
    {
        inputs[InputNames::IN_2] = input2;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_input_2(std::shared_ptr<TensorAttributes>&& input2)
    {
        inputs[InputNames::IN_2] = std::move(input2);
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_output_0(const std::shared_ptr<TensorAttributes>& output0)
    {
        outputs[OutputNames::OUT_0] = output0;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_output_0(std::shared_ptr<TensorAttributes>&& output0)
    {
        outputs[OutputNames::OUT_0] = std::move(output0);
        return *this;
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

    flatbuffers::Offset<hipdnn_sdk::data_objects::PointwiseAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto in0 = get_input_0();
        auto in1 = get_input_1();
        auto in2 = get_input_2();
        auto ot0 = get_output_0();

        return hipdnn_sdk::data_objects::CreatePointwiseAttributes(
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
};
typedef PointwiseAttributes Pointwise_attributes;
}
}
