// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "ck_tile/core/config.hpp"

namespace ck_tile {

CK_TILE_DEVICE void set_gfx125_wave_sched_mode_dep_mode_2()
{
#if defined(__gfx125__)
    // aiter parity: writes SQ_WAVE_SCHED_MODE_DEP_MODE[1:0] = 2 on gfx1250.
    // hwreg id 26 is HW_REG_WAVE_SCHED_MODE; numeric form avoids LLVM symbol drift.
    // Static evidence has not found a distinct bit 2 DISABLE_XDL_ARB_STALL control.
    asm volatile("s_setreg_imm32_b32 hwreg(26, 0, 2), 2" ::: "memory");
#endif
}

} // namespace ck_tile
