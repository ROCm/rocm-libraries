// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_data_sdk/data_objects/matmul_attributes_generated.h>
#include <memory>
#include <unordered_map>
#include <vector>

namespace hipdnn_frontend::graph
{
class MatmulAttributes : public Attributes<MatmulAttributes>
{
public:
    enum class InputNames
    {
        A = 0, // Input A tensor
        B = 1 // Input b tensor
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        C = 0 // Output tensor
    };
    typedef OutputNames output_names; // NOLINT(readability-identifier-naming)

    std::unordered_map<InputNames, std::shared_ptr<TensorAttributes>> inputs;
    std::unordered_map<OutputNames, std::shared_ptr<TensorAttributes>> outputs;

    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_a() const
    {
        return getInput(InputNames::A);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_b() const
    {
        return getInput(InputNames::B);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_c() const
    {
        return getOutput(OutputNames::C);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    MatmulAttributes& set_a(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::A, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    MatmulAttributes& set_a(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::A, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    MatmulAttributes& set_b(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::B, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    MatmulAttributes& set_b(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::B, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    MatmulAttributes& set_c(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::C, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    MatmulAttributes& set_c(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::C, std::move(value));
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::MatmulAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        return hipdnn_data_sdk::data_objects::CreateMatmulAttributes(
            builder, get_a()->get_uid(), get_b()->get_uid(), get_c()->get_uid());
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

    MatmulAttributes& setInput(InputNames name, const std::shared_ptr<TensorAttributes>& value)
    {
        inputs[name] = value;
        return *this;
    }
    MatmulAttributes& setInput(InputNames name, std::shared_ptr<TensorAttributes>&& value)
    {
        inputs[name] = std::move(value);
        return *this;
    }

    MatmulAttributes& setOutput(OutputNames name, const std::shared_ptr<TensorAttributes>& value)
    {
        outputs[name] = value;
        return *this;
    }
    MatmulAttributes& setOutput(OutputNames name, std::shared_ptr<TensorAttributes>&& value)
    {
        outputs[name] = std::move(value);
        return *this;
    }
};

typedef MatmulAttributes Matmul_attributes;
} // namespace hipdnn_frontend::graph
