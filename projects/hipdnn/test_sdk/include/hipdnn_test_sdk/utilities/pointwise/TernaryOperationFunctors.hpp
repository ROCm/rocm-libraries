// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace hipdnn_test_sdk::utilities::pointwise
{

template <typename OutputType>
struct BinarySelect
{
    template <typename Input0Type, typename Input1Type, typename MaskType>
    OutputType
        operator()(const Input0Type& input0, const Input1Type& input1, const MaskType& mask) const
    {
        return static_cast<bool>(mask) ? static_cast<OutputType>(input0)
                                       : static_cast<OutputType>(input1);
    }
};

} // namespace hipdnn_test_sdk::utilities::pointwise
