// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "GraphAttributes.hpp"
#include "TensorAttributes.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <string>
#include <utility>

namespace hipdnn_frontend::graph
{
// Any class extending Attributes must have an inputs & outputs map.
// The map needs to have TensorAttributes as the value.
// Attributes uses these maps to set the tensor data types.
template <typename DerivedT>
class Attributes
{
private:
    DerivedT& self()
    {
        return static_cast<DerivedT&>(*this);
    }
    const DerivedT& self() const
    {
        return static_cast<const DerivedT&>(*this);
    }

public:
    std::string name;
    DataType compute_data_type = DataType::NOT_SET; // NOLINT(readability-identifier-naming)

    DerivedT& set_name(const std::string& nameValue) // NOLINT(readability-identifier-naming)
    {
        name = nameValue;
        return self();
    }

    const std::string& get_name() const // NOLINT(readability-identifier-naming)
    {
        return name;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    DerivedT& set_compute_data_type(DataType value)
    {
        compute_data_type = value;
        return self();
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    DataType get_compute_data_type() const
    {
        return compute_data_type;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    Error fill_from_context(const GraphAttributes& graphAttributes)
    {
        for(auto& [_, tensor] : self().inputs)
        {
            if(tensor)
            {
                tensor->fill_from_context(graphAttributes);
            }
        }

        for(auto& [_, tensor] : self().outputs)
        {
            if(tensor)
            {
                tensor->fill_from_context(graphAttributes);
            }
        }

        if(get_compute_data_type() == DataType::NOT_SET)
        {
            set_compute_data_type(graphAttributes.get_compute_data_type());
        }

        return {};
    }

protected:
    template <typename InputNameT>
    std::shared_ptr<TensorAttributes> getInput(InputNameT inputName) const
    {
        auto it = self().inputs.find(inputName);
        if(it != self().inputs.end())
        {
            return it->second;
        }
        return nullptr;
    }

    template <typename OutputNameT>
    std::shared_ptr<TensorAttributes> getOutput(OutputNameT outputName) const
    {
        auto it = self().outputs.find(outputName);
        if(it != self().outputs.end())
        {
            return it->second;
        }
        return nullptr;
    }

    template <typename InputNameT>
    DerivedT& setInput(InputNameT inputName, const std::shared_ptr<TensorAttributes>& value)
    {
        self().inputs[inputName] = value;
        return self();
    }

    template <typename InputNameT>
    DerivedT& setInput(InputNameT inputName, std::shared_ptr<TensorAttributes>&& value)
    {
        self().inputs[inputName] = std::move(value);
        return self();
    }

    template <typename OutputNameT>
    DerivedT& setOutput(OutputNameT outputName, const std::shared_ptr<TensorAttributes>& value)
    {
        self().outputs[outputName] = value;
        return self();
    }

    template <typename OutputNameT>
    DerivedT& setOutput(OutputNameT outputName, std::shared_ptr<TensorAttributes>&& value)
    {
        self().outputs[outputName] = std::move(value);
        return self();
    }
};
} // namespace hipdnn_frontend::graph
