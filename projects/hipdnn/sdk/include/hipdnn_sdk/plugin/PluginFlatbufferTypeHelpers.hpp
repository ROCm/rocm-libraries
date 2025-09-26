// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <ostream>

#include <hipdnn_sdk/data_objects/graph_generated.h>
#include <hipdnn_sdk/logging/Logger.hpp>

namespace hipdnn_sdk::data_objects
{
inline const char* toString(hipdnn_sdk::data_objects::NodeAttributes attributes)
{
    return hipdnn_sdk::data_objects::EnumNameNodeAttributes(attributes);
}

inline const char* toString(hipdnn_sdk::data_objects::DataType dataType)
{
    return hipdnn_sdk::data_objects::EnumNameDataType(dataType);
}
}

inline std::ostream& operator<<(std::ostream& os,
                                hipdnn_sdk::data_objects::NodeAttributes attributes)
{
    return os << hipdnn_sdk::data_objects::EnumNameNodeAttributes(attributes);
}

inline std::ostream& operator<<(std::ostream& os, hipdnn_sdk::data_objects::DataType dataType)
{
    return os << hipdnn_sdk::data_objects::EnumNameDataType(dataType);
}

template <>
struct fmt::formatter<hipdnn_sdk::data_objects::NodeAttributes> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(hipdnn_sdk::data_objects::NodeAttributes attributes, FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(
            hipdnn_sdk::data_objects::EnumNameNodeAttributes(attributes), ctx);
    }
};

template <>
struct fmt::formatter<hipdnn_sdk::data_objects::DataType> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(hipdnn_sdk::data_objects::DataType dataType, FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(
            hipdnn_sdk::data_objects::EnumNameDataType(dataType), ctx);
    }
};
