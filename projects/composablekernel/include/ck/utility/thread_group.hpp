// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "get_id.hpp"

namespace ck {

template <index_t ThreadPerBlock>
struct ThisThreadBlock
{
    static constexpr index_t kNumThread_ = ThreadPerBlock;

    static constexpr index_t GetNumOfThread() { return kNumThread_; }

    static constexpr bool IsBelong() { return true; }

    static constexpr bool InWaveGroup() { return false; }

    __device__ static index_t GetThreadId() { return get_thread_local_1d_id(); }

    static constexpr index_t GetNumWavePerGroup() { return 0; }

    static constexpr index_t GetNumWaveGroups() { return 0; }
};

template <index_t ThreadPerBlock, index_t WaveSize, index_t NumWaveGroup>
struct ThisThreadBlockWaveGroup
{
    static_assert(ThreadPerBlock % (WaveSize * NumWaveGroup) == 0, "");

    static constexpr index_t kNumWavePerGroup_ = ThreadPerBlock / (WaveSize * NumWaveGroup);
    static constexpr index_t kNumThread_       = ThreadPerBlock / kNumWavePerGroup_;

    static constexpr index_t GetNumOfThread() { return kNumThread_; }

    static constexpr bool IsBelong() { return true; }

    static constexpr bool InWaveGroup() { return true; }

    __device__ static index_t GetThreadId() { return GetWaveGroupId() * WaveSize + GetLaneId(); }

    __device__ static index_t GetLaneId() { return get_lane_id(); }

    __device__ static index_t GetWaveGroupId() { return get_wavegroup_id(); }

    __device__ static index_t GetWaveIdInWaveGroup() { return get_wave_id_in_wavegroup(); }

    static constexpr index_t GetNumWavePerGroup() { return kNumWavePerGroup_; }

    static constexpr index_t GetNumWaveGroups() { return NumWaveGroup; }
};

} // namespace ck
