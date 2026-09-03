// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck/ck.hpp"

namespace ck {
namespace tensor_operation {
namespace device {

template <typename DataType, typename Lengths>
bool tensor_exceeds_2gb(const Lengths& lengths)
{
    constexpr long_index_t TwoGB = (long_index_t{1} << 31);
    long_index_t total           = 1;
    for(const auto& l : lengths)
        total *= l;
    long_index_t total_bytes = total * sizeof(DataType);
    // tensor number of elements is stored in int32_t so max value is TwoGB - 1,
    // while tensor number of bytes is stored in uint32_t so max value is TwoGB.
    // This double check is actually needed only for DataType with size 1
    return total >= TwoGB || total_bytes > TwoGB;
}

} // namespace device
} // namespace tensor_operation
} // namespace ck
