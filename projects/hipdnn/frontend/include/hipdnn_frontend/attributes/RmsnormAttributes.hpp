// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "Attributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_data_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <memory>
#include <unordered_map>

namespace hipdnn_frontend::graph
{
class RmsnormAttributes : public Attributes<RmsnormAttributes>
{
public:
    enum class InputNames
    {
        X = 0,
        SCALE = 1,
        EPSILON = 2
    };
    typedef InputNames input_names; // NOLINT(readability-identifier-naming)

    enum class OutputNames
    {
        Y = 0,
        INV_RMS = 1
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
    std::shared_ptr<TensorAttributes> get_scale() const
    {
        return getInput(InputNames::SCALE);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_epsilon() const
    {
        return getInput(InputNames::EPSILON);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_y() const
    {
        return getOutput(OutputNames::Y);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    std::shared_ptr<TensorAttributes> get_inv_rms() const
    {
        return getOutput(OutputNames::INV_RMS);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    RmsnormAttributes& set_x(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::X, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RmsnormAttributes& set_x(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::X, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RmsnormAttributes& set_scale(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::SCALE, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RmsnormAttributes& set_scale(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::SCALE, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RmsnormAttributes& set_epsilon(const std::shared_ptr<TensorAttributes>& value)
    {
        return setInput(InputNames::EPSILON, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RmsnormAttributes& set_epsilon(std::shared_ptr<TensorAttributes>&& value)
    {
        return setInput(InputNames::EPSILON, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RmsnormAttributes& set_y(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::Y, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RmsnormAttributes& set_y(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::Y, std::move(value));
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RmsnormAttributes& set_inv_rms(const std::shared_ptr<TensorAttributes>& value)
    {
        return setOutput(OutputNames::INV_RMS, value);
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    RmsnormAttributes& set_inv_rms(std::shared_ptr<TensorAttributes>&& value)
    {
        return setOutput(OutputNames::INV_RMS, std::move(value));
    }

    flatbuffers::Offset<hipdnn_data_sdk::data_objects::RmsnormAttributes>
        pack_attributes(flatbuffers::FlatBufferBuilder& builder) const // NOLINT
    {
        auto invRms = get_inv_rms();

        return hipdnn_data_sdk::data_objects::CreateRmsnormAttributes(
            builder,
            get_x()->get_uid(),
            get_scale()->get_uid(),
            get_epsilon()->get_uid(),
            get_y()->get_uid(),
            invRms ? flatbuffers::Optional<int64_t>(invRms->get_uid()) : flatbuffers::nullopt);
    }

    static RmsnormAttributes fromFlatBuffer(
        const hipdnn_data_sdk::data_objects::RmsnormAttributes* fb,
        const std::unordered_map<int64_t, std::shared_ptr<TensorAttributes>>& tensorMap)
    {
        RmsnormAttributes attr;

        attr.set_x(tensorMap.at(fb->x_tensor_uid()));
        attr.set_scale(tensorMap.at(fb->scale_tensor_uid()));
        attr.set_epsilon(tensorMap.at(fb->epsilon_tensor_uid()));
        attr.set_y(tensorMap.at(fb->y_tensor_uid()));

        if(fb->inv_rms_tensor_uid().has_value())
        {
            attr.set_inv_rms(tensorMap.at(fb->inv_rms_tensor_uid().value()));
        }

        return attr;
    }
};
} // namespace hipdnn_frontend::graph
