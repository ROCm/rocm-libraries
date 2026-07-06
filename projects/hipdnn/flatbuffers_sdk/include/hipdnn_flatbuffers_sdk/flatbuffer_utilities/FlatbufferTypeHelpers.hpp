// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <ostream>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>

namespace hipdnn_flatbuffers_sdk::data_objects
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

// operator<< definitions in the same namespace as the types for ADL
inline std::ostream& operator<<(std::ostream& os, NodeAttributes attributes)
{
    return os << EnumNameNodeAttributes(attributes);
}

inline std::ostream& operator<<(std::ostream& os, DataType dataType)
{
    return os << EnumNameDataType(dataType);
}

inline std::ostream& operator<<(std::ostream& os, PointwiseMode pointwiseMode)
{
    return os << EnumNamePointwiseMode(pointwiseMode);
}

} // namespace hipdnn_flatbuffers_sdk::data_objects
