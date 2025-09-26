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
        return _mode;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_relu_lower_clip() const
    {
        return _reluLowerClip;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_relu_upper_clip() const
    {
        return _reluUpperClip;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<float> get_relu_lower_slope() const
    {
        return _reluLowerSlope;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::optional<int64_t> get_axis() const
    {
        return _axis;
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
    PointwiseAttributes& set_mode(PointwiseMode mode)
    {
        _mode = mode;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_relu_lower_clip(float reluLowerClip)
    {
        _reluLowerClip = reluLowerClip;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_relu_upper_clip(float reluUpperClip)
    {
        _reluUpperClip = reluUpperClip;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_relu_lower_clip_slope(float reluLowerSlope)
    {
        _reluLowerSlope = reluLowerSlope;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    PointwiseAttributes& set_axis(std::optional<int64_t> axis)
    {
        _axis = axis;
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

    flatbuffers::Offset<hipdnn_sdk::data_objects::PointwiseAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto in0 = get_input_0();
        auto in1 = get_input_1();
        auto in2 = get_input_2();
        auto ot0 = get_output_0();

        return hipdnn_sdk::data_objects::CreatePointwiseAttributes(
            builder,
            toSdkType(_mode),
            _reluLowerClip,
            _reluUpperClip,
            _reluLowerSlope,
            _axis,
            in0->get_uid(),
            in1 ? flatbuffers::Optional<int64_t>(in1->get_uid()) : flatbuffers::nullopt,
            in2 ? flatbuffers::Optional<int64_t>(in2->get_uid()) : flatbuffers::nullopt,
            ot0->get_uid());
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
    PointwiseMode _mode = PointwiseMode::NOT_SET;
    std::optional<float> _reluLowerClip = std::nullopt;
    std::optional<float> _reluUpperClip = std::nullopt;
    std::optional<float> _reluLowerSlope = std::nullopt;
    std::optional<int64_t> _axis = std::nullopt;
};
typedef PointwiseAttributes Pointwise_attributes;
}
}
