// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <hip/hip_runtime.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/stream_utils.hpp"
#include "ck_tile/ops/pooling_bwd/pipeline/pool_bwd_default_policy.hpp"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wlifetime-safety-intra-tu-suggestions"

namespace ck_tile {

struct PoolBwdHostArgs
{
    CK_TILE_HOST PoolBwdHostArgs(const void* p_dout_,
                                 const void* p_indices_,
                                 void* p_din_,
                                 void* p_workspace_,
                                 long_index_t dout_length_,
                                 long_index_t din_length_)
        : p_dout(p_dout_),
          p_indices(p_indices_),
          p_din(p_din_),
          p_workspace(p_workspace_),
          dout_length(dout_length_),
          din_length(din_length_)
    {
    }

    const void* p_dout;
    const void* p_indices;
    void* p_din;
    void* p_workspace;
    long_index_t dout_length;
    long_index_t din_length;
};

struct PoolBwdKernelArgs
{
    const void* p_dout;
    const void* p_indices;
    void* p_din;
    void* p_workspace;
    long_index_t dout_length;
    long_index_t din_length;
};

template <typename Problem_, typename Policy_ = PoolBwdDefaultPolicy>
struct PoolBwdKernel
{
    using Problem = remove_cvref_t<Problem_>;
    using Policy  = remove_cvref_t<Policy_>;

    using DOutDataType        = remove_cvref_t<typename Problem::DOutDataType>;
    using IndexDataType       = remove_cvref_t<typename Problem::IndexDataType>;
    using DInDataType         = remove_cvref_t<typename Problem::DInDataType>;
    using DInAtomicAddPreCast = remove_cvref_t<typename Problem::DInAtomicAddPreCast>;

    static constexpr index_t kBlockSize      = Policy::template GetBlockSize<Problem>();
    static constexpr index_t kVectorSize     = Policy::template GetVectorSize<Problem>();
    static constexpr bool kHasOverlap        = Problem::kHasOverlap;
    static constexpr bool kNeedFp32Workspace = Problem::kNeedFp32Workspace;
    static constexpr bool kAtomicTargetIsDIn = !kNeedFp32Workspace;

    CK_TILE_HOST static constexpr index_t BlockSize() { return kBlockSize; }

    CK_TILE_HOST static constexpr auto MakeKernelArgs(const PoolBwdHostArgs& host_args)
    {
        return PoolBwdKernelArgs{host_args.p_dout,
                                 host_args.p_indices,
                                 host_args.p_din,
                                 host_args.p_workspace,
                                 host_args.dout_length,
                                 host_args.din_length};
    }

    CK_TILE_HOST static index_t CalculateGridSize(const stream_config& s)
    {
        const index_t num_cu = get_available_compute_units(s);
        return num_cu > 0 ? num_cu : 1;
    }

    CK_TILE_HOST static bool IsSupportedArgument(const PoolBwdKernelArgs& kargs)
    {
        if(kargs.dout_length % kVectorSize != 0)
        {
            return false;
        }
        if(kargs.din_length % kVectorSize != 0)
        {
            return false;
        }
        if(kHasOverlap && !Problem::kDInIsFp32OrFp64 && kargs.p_workspace == nullptr)
        {
            return false;
        }
        return true;
    }

    CK_TILE_HOST static std::size_t GetWorkSpaceSize(const PoolBwdHostArgs& host_args)
    {
        if constexpr(Problem::kNeedFp32Workspace)
        {
            return static_cast<std::size_t>(host_args.din_length) * sizeof(float);
        }
        else
        {
            return 0;
        }
    }

    CK_TILE_DEVICE void operator()(PoolBwdKernelArgs kargs) const
    {
        const auto* p_dout    = static_cast<const DOutDataType*>(kargs.p_dout);
        const auto* p_indices = static_cast<const IndexDataType*>(kargs.p_indices);

        DInAtomicAddPreCast* p_atomic_dst = nullptr;
        DInDataType* p_set_dst            = nullptr;

        if constexpr(kHasOverlap)
        {
            if constexpr(kAtomicTargetIsDIn)
            {
                p_atomic_dst = static_cast<DInAtomicAddPreCast*>(kargs.p_din);
            }
            else
            {
                p_atomic_dst = static_cast<DInAtomicAddPreCast*>(kargs.p_workspace);
            }
        }
        else
        {
            p_set_dst = static_cast<DInDataType*>(kargs.p_din);
        }

        const long_index_t total = kargs.dout_length;
        const long_index_t step  = static_cast<long_index_t>(gridDim.x) * blockDim.x * kVectorSize;
        long_index_t i           = (static_cast<long_index_t>(blockIdx.x) * blockDim.x +
                          static_cast<long_index_t>(threadIdx.x)) *
                         kVectorSize;

        const long_index_t din_len = kargs.din_length;

        while(i + kVectorSize <= total)
        {
#pragma unroll
            for(index_t v = 0; v < kVectorSize; ++v)
            {
                const long_index_t flat = i + static_cast<long_index_t>(v);
                const IndexDataType idx = p_indices[flat];

                if(idx >= 0 && static_cast<long_index_t>(idx) < din_len)
                {
                    const DOutDataType d = p_dout[flat];

                    if constexpr(kHasOverlap)
                    {
                        const DInAtomicAddPreCast a = type_convert<DInAtomicAddPreCast>(d);
                        atomicAdd(p_atomic_dst + idx, a);
                    }
                    else
                    {
                        p_set_dst[idx] = type_convert<DInDataType>(d);
                    }
                }
            }
            i += step;
        }
    }
};

} // namespace ck_tile

#pragma clang diagnostic pop
