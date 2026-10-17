// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

namespace ck {

template <typename ThreadGroup, bool Enable>
struct NamedBarrier
{
    static constexpr bool IsEnabled    = Enable;
    static constexpr index_t num_waves = ThreadGroup::GetNumWaves();
    using barrier_t                    = __amdgpu_named_workgroup_barrier_t;

    barrier_t* p_data_ = nullptr;

    __device__ constexpr NamedBarrier() : p_data_() {}

    __device__ constexpr NamedBarrier(barrier_t* __restrict__ p_data) : p_data_(p_data) {}

    __device__ void Init() const
    {
        if constexpr(IsEnabled)
        {
            if(threadIdx.x == 0)
            {
                __builtin_amdgcn_s_barrier_init(p_data_, num_waves);
            }
            block_sync_lds();
        }
    }

    __device__ void Join() const
    {
        if constexpr(IsEnabled)
        {
            __builtin_amdgcn_s_barrier_join(p_data_);
        }
    }

    __device__ void Wait() const
    {
        if constexpr(IsEnabled)
        {
            __builtin_amdgcn_s_barrier_signal_var(p_data_, 0);
            __builtin_amdgcn_s_barrier_wait(num_waves);
        }
    }
};

} // namespace ck
