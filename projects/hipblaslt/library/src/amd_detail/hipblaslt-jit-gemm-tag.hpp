// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "rocblaslt-types.h"
#include <cstdint>
#include <cstring>

namespace hipblaslt_ext::experimental::detail
{
    constexpr uint32_t jitAlgoTag = 0x4a495431; // JIT1; data[0..3] remains local index 0.

    inline bool isJitAlgo(const rocblaslt_matmul_algo& algo)
    {
        uint32_t tag;
        std::memcpy(&tag, algo.data + sizeof(int32_t), sizeof(tag));
        return tag == jitAlgoTag;
    }
}
