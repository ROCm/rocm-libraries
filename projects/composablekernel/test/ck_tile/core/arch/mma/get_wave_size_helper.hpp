// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstdio>

#include "ck_tile/core/arch/arch.hpp"
#include <hip/hip_runtime.h>
#include "ck_tile/host/hip_check_error.hpp"

namespace ck_tile::core::arch::testing {

__global__ void getWaveSizeForSelectedOp(uint32_t* waveSize)
{
    using CompilerTarget = decltype(ck_tile::core::arch::get_compiler_target());

    if(waveSize)
    {
        *waveSize = static_cast<uint32_t>(CompilerTarget::WAVE_SIZE_ID);
    }
}

static CK_TILE_HOST uint32_t getDeviceWaveSize()
{
    uint32_t* d_wave_size;
    HIP_CHECK_ERROR(hipMalloc(&d_wave_size, sizeof(uint32_t)));
    getWaveSizeForSelectedOp<<<1, 32>>>(d_wave_size);
    HIP_CHECK_ERROR(hipDeviceSynchronize());
    uint32_t wave_size;
    HIP_CHECK_ERROR(hipMemcpy(&wave_size, d_wave_size, sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK_ERROR(hipFree(d_wave_size));
    return wave_size;
}

} // namespace ck_tile::core::arch::testing
