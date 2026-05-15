// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/stream_utils.hpp"

namespace ck_tile {

struct PoolBwdCastHostArgs
{
    CK_TILE_HOST PoolBwdCastHostArgs(const void* p_src_, void* p_dst_, long_index_t length_)
        : p_src(p_src_), p_dst(p_dst_), length(length_)
    {
    }

    const void* p_src;
    void* p_dst;
    long_index_t length;
};

struct PoolBwdCastKernelArgs
{
    const void* p_src;
    void* p_dst;
    long_index_t length;
};

template <typename SrcDataType_,
          typename DstDataType_,
          index_t BlockSize_  = 256,
          index_t VectorSize_ = 4>
struct PoolBwdCastKernel
{
    using SrcDataType = remove_cvref_t<SrcDataType_>;
    using DstDataType = remove_cvref_t<DstDataType_>;

    static constexpr index_t kBlockSize = BlockSize_;
    // See note on kVectorSize in PoolBwdKernel: this is the number of
    // consecutive elements processed by each thread per grid-stride pass, not
    // a hardware-vector width.
    static constexpr index_t kVectorSize = VectorSize_;

    static_assert(kVectorSize == 1 || kVectorSize == 2 || kVectorSize == 4,
                  "kVectorSize must be 1, 2, or 4");

    CK_TILE_HOST static constexpr index_t BlockSize() { return kBlockSize; }

    CK_TILE_HOST static constexpr auto MakeKernelArgs(const PoolBwdCastHostArgs& host_args)
    {
        return PoolBwdCastKernelArgs{host_args.p_src, host_args.p_dst, host_args.length};
    }

    CK_TILE_HOST static index_t CalculateGridSize(const stream_config& s, long_index_t length)
    {
        const index_t num_cu = get_available_compute_units(s);
        const index_t cu_cap = num_cu > 0 ? num_cu : 1;

        const long_index_t work_per_block =
            static_cast<long_index_t>(kBlockSize) * static_cast<long_index_t>(kVectorSize);
        const long_index_t work_blocks =
            work_per_block > 0 ? (length + work_per_block - 1) / work_per_block : 1;

        const long_index_t capped =
            work_blocks < static_cast<long_index_t>(cu_cap) ? work_blocks : cu_cap;

        return capped > 0 ? static_cast<index_t>(capped) : 1;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const PoolBwdCastHostArgs& host_args)
    {
        if(host_args.length < 0)
        {
            return false;
        }
        if(host_args.length > 0 && (host_args.p_src == nullptr || host_args.p_dst == nullptr))
        {
            return false;
        }
        // The scalar tail loop in operator() handles unaligned lengths, so no
        // alignment constraint is enforced here.
        return true;
    }

    CK_TILE_DEVICE void operator()(PoolBwdCastKernelArgs kargs) const
    {
        const auto* p_src = static_cast<const SrcDataType*>(kargs.p_src);
        auto* p_dst       = static_cast<DstDataType*>(kargs.p_dst);

        const long_index_t total = kargs.length;
        const long_index_t step  = static_cast<long_index_t>(gridDim.x) * blockDim.x * kVectorSize;
        long_index_t i           = (static_cast<long_index_t>(blockIdx.x) * blockDim.x +
                          static_cast<long_index_t>(threadIdx.x)) *
                         kVectorSize;

        while(i + kVectorSize <= total)
        {
#pragma unroll
            for(index_t v = 0; v < kVectorSize; ++v)
            {
                const long_index_t flat = i + static_cast<long_index_t>(v);
                p_dst[flat]             = type_convert<DstDataType>(p_src[flat]);
            }
            i += step;
        }

        // Scalar tail loop for unaligned remainders.
        for(long_index_t j = i; j < total; ++j)
        {
            p_dst[j] = type_convert<DstDataType>(p_src[j]);
        }
    }
};

} // namespace ck_tile
