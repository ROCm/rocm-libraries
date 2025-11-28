// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace hipdnn_frontend
{
namespace graph
{
class GraphAttributes
{
public:
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::string& get_name() const
    {
        return _name;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    DataType get_compute_data_type() const
    {
        return _computeType;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    DataType get_intermediate_data_type() const
    {
        return _intermediateType;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    DataType get_io_data_type() const
    {
        return _ioType;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    GraphAttributes& set_compute_data_type(DataType computeType)
    {
        _computeType = computeType;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    GraphAttributes& set_intermediate_data_type(DataType intermediateType)
    {
        _intermediateType = intermediateType;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    GraphAttributes& set_io_data_type(DataType ioType)
    {
        _ioType = ioType;
        return *this;
    }
    // NOLINTNEXTLINE(readability-identifier-naming)
    GraphAttributes& set_name(const std::string& name)
    {
        _name = name;
        return *this;
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    GraphAttributes& fill_missing_properties(const GraphAttributes& other)
    {
        if(_computeType == DataType::NOT_SET)
        {
            _computeType = other._computeType;
        }

        if(_intermediateType == DataType::NOT_SET)
        {
            _intermediateType = other._intermediateType;
        }

        if(_ioType == DataType::NOT_SET)
        {
            _ioType = other._ioType;
        }

        if(_name.empty())
        {
            _name = other._name;
        }

        return *this;
    }

private:
    std::string _name;
    DataType _computeType = DataType::NOT_SET;
    DataType _intermediateType = DataType::NOT_SET;
    DataType _ioType = DataType::NOT_SET;
};
typedef GraphAttributes Context;
}
}
