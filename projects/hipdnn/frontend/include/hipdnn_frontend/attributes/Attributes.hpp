// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include "GraphAttributes.hpp"
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <string>
#include <vector>

namespace hipdnn_frontend
{
namespace graph
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

        return {};
    }
};
}
}
