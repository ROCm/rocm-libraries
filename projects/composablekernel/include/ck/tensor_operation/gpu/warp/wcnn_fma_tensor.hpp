// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/utility/math.hpp"
#include "ck/utility/amd_fma_tensor.hpp"

namespace ck {

template <typename InDataType,
          typename AccDataType,
          index_t HPerWcnn,
          index_t WPerWcnn,
          index_t ChanOff = 0,
          bool WaveGroup  = false>
struct WcnnFmaFromTensor
{
    static constexpr index_t WaveSize = 32;
    template <typename ScalarType>
    static constexpr index_t SizeOfBits()
    {
        return (std::is_same<int4_t, ScalarType>::value || std::is_same<uint4_t, ScalarType>::value)
                   ? 4
                   : sizeof(ScalarType) * 8;
    }

    static constexpr intrin_wcnn_fma_from_tensor<HPerWcnn, WPerWcnn, ChanOff> fma_instr;
    static constexpr index_t GetNumResidual()
    {
        constexpr index_t in_bits = SizeOfBits<InDataType>();
        static_assert(in_bits <= 16);
        if constexpr(in_bits == 16)
        {
            if constexpr((HPerWcnn == 4) && (WPerWcnn == 4))
            {
                return 4;
            }
            else
            {
                return 2;
            }
        }
        else if constexpr((in_bits == 8) && (WPerWcnn == 4)) // 4x4 or 8x4
        {
            if constexpr(HPerWcnn == 4)
            {
                return 2;
            }
            else
            {
                return 1;
            }
        }
        else
        {
            return 1;
        }
    }
    static constexpr index_t GetAccumChannelOrder()
    {
        constexpr index_t in_bits = SizeOfBits<InDataType>();
        static_assert(in_bits <= 16);

        if constexpr(in_bits == 16)
        {
            return 1;
        }
        else
        {
            return 0;
        }
    }

    static constexpr index_t GetNumAccum()
    {
        constexpr index_t in_bits = SizeOfBits<InDataType>();
        if constexpr(in_bits == 4)
        {
            static_assert(GetNumResidual() == 1);
            if constexpr((HPerWcnn == 4) && (WPerWcnn == 2))
            {
                return 2;
            }
            else if constexpr((HPerWcnn == 8) && (WPerWcnn == 4))
            {
                return 2;
            }
            else
            {
                return 1;
            }
        }
        else
        {
            return 1;
        }
    }
};

} // namespace ck
