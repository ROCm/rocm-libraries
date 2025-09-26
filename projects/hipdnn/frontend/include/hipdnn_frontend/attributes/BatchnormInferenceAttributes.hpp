// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_sdk/data_objects/batchnorm_inference_attributes_generated.h>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend
{
namespace graph
{
class BatchnormInferenceAttributes : public Attributes<BatchnormInferenceAttributes>
{
public:
    enum class InputNames
    {
        X = 0,
        MEAN = 1,
        INV_VARIANCE = 2,
        SCALE = 3,
        BIAS = 4
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        Y = 0
    };
    typedef OutputNames output_names; // NOLINT(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_x() const
    {
        return getInput(InputNames::X);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_mean() const
    {
        return getInput(InputNames::MEAN);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_inv_variance() const
    {
        return getInput(InputNames::INV_VARIANCE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getInput(InputNames::SCALE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_bias() const
    {
        return getInput(InputNames::BIAS);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(OutputNames::Y);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_mean(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::MEAN, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_mean(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::MEAN, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_inv_variance(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::INV_VARIANCE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_inv_variance(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::INV_VARIANCE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_bias(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::BIAS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_bias(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::BIAS, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    BatchnormInferenceAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }

    flatbuffers::Offset<hipdnn_sdk::data_objects::BatchnormInferenceAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto mean = get_mean();
        auto invVariance = get_inv_variance();

        return hipdnn_sdk::data_objects::CreateBatchnormInferenceAttributes(builder,
                                                                            get_x()->get_uid(),
                                                                            mean->get_uid(),
                                                                            invVariance->get_uid(),
                                                                            get_scale()->get_uid(),
                                                                            get_bias()->get_uid(),
                                                                            get_y()->get_uid());
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

    BatchnormInferenceAttributes& setInput(InputNames name,
                                           const std::shared_ptr<TensorAttributes>& value)
    {
        inputs[name] = value;
        return *this;
    }
    BatchnormInferenceAttributes& setInput(InputNames name,
                                           std::shared_ptr<TensorAttributes>&& value)
    {
        inputs[name] = std::move(value);
        return *this;
    }

    BatchnormInferenceAttributes& setOutput(OutputNames name,
                                            const std::shared_ptr<TensorAttributes>& value)
    {
        outputs[name] = value;
        return *this;
    }
    BatchnormInferenceAttributes& setOutput(OutputNames name,
                                            std::shared_ptr<TensorAttributes>&& value)
    {
        outputs[name] = std::move(value);
        return *this;
    }
};
typedef BatchnormInferenceAttributes Batchnorm_inference_attributes;
}
}
