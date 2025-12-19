// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <ostream>

#include <hipdnn_data_sdk/data_objects/graph_generated.h>
#include <hipdnn_data_sdk/logging/Logger.hpp>

namespace hipdnn_data_sdk::data_objects
{

inline const char* toString(NodeAttributes attributes)
{
    return EnumNameNodeAttributes(attributes);
}

inline const char* toString(DataType dataType)
{
    return EnumNameDataType(dataType);
}

inline const char* toString(PointwiseMode pointwiseMode)
{
    return EnumNamePointwiseMode(pointwiseMode);
}

} // namespace hipdnn_data_sdk::data_objects

inline std::ostream& operator<<(std::ostream& os,
                                hipdnn_data_sdk::data_objects::NodeAttributes attributes)
{
    return os << hipdnn_data_sdk::data_objects::EnumNameNodeAttributes(attributes);
}

inline std::ostream& operator<<(std::ostream& os, hipdnn_data_sdk::data_objects::DataType dataType)
{
    return os << hipdnn_data_sdk::data_objects::EnumNameDataType(dataType);
}

inline std::ostream& operator<<(std::ostream& os,
                                hipdnn_data_sdk::data_objects::PointwiseMode pointwiseMode)
{
    return os << hipdnn_data_sdk::data_objects::EnumNamePointwiseMode(pointwiseMode);
}

template <>
struct fmt::formatter<hipdnn_data_sdk::data_objects::NodeAttributes> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(hipdnn_data_sdk::data_objects::NodeAttributes attributes, FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(
            hipdnn_data_sdk::data_objects::EnumNameNodeAttributes(attributes), ctx);
    }
};

template <>
struct fmt::formatter<hipdnn_data_sdk::data_objects::DataType> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(hipdnn_data_sdk::data_objects::DataType dataType, FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(
            hipdnn_data_sdk::data_objects::EnumNameDataType(dataType), ctx);
    }
};

template <>
struct fmt::formatter<hipdnn_data_sdk::data_objects::PointwiseMode> : fmt::formatter<const char*>
{
    template <typename FormatContext>
    auto format(hipdnn_data_sdk::data_objects::PointwiseMode pointwiseMode,
                FormatContext& ctx) const
    {
        return fmt::formatter<const char*>::format(
            hipdnn_data_sdk::data_objects::EnumNamePointwiseMode(pointwiseMode), ctx);
    }
};
