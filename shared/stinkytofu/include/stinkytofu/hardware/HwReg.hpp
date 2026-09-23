/* ************************************************************************
 * Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#pragma once

#include <cstdint>
#include <string_view>

#include "stinkytofu/Export.hpp"
#include "stinkytofu/hardware/GfxIsa.hpp"

namespace stinkytofu {
namespace HwReg {

// HWREG ids addressable by s_setreg / s_getreg. `_GFX12` marks gfx12-only.
enum Id : uint16_t {
    ID_MODE = 1,
    ID_STATUS = 2,
    ID_GPR_ALLOC = 5,
    ID_LDS_ALLOC = 6,
    ID_HW_ID1 = 23,
    ID_HW_ID2 = 24,
    ID_SHADER_CYCLES_LO = 29,
    ID_IB_STS2 = 28,

    ID_STATE_PRIV_GFX12 = 4,
    ID_PERF_SNAPSHOT_DATA_GFX12 = 10,
    ID_PERF_SNAPSHOT_PC_LO_GFX12 = 11,
    ID_PERF_SNAPSHOT_PC_HI_GFX12 = 12,
    ID_PERF_SNAPSHOT_DATA1_GFX12 = 15,
    ID_PERF_SNAPSHOT_DATA2_GFX12 = 16,
    ID_EXCP_FLAG_PRIV_GFX12 = 17,
    ID_EXCP_FLAG_USER_GFX12 = 18,
    ID_TRAP_CTRL_GFX12 = 19,
    ID_SCRATCH_BASE_LO_GFX12 = 20,
    ID_SCRATCH_BASE_HI_GFX12 = 21,
    ID_SCHED_MODE_GFX12 = 26,
    ID_SHADER_CYCLES_HI_GFX12 = 30,
    ID_DVGPR_ALLOC_LO_GFX12 = 31,
    ID_DVGPR_ALLOC_HI_GFX12 = 32,
    ID_XNACK_STATE_PRIV_GFX12 = 33,
    ID_XNACK_MASK_GFX12 = 34,
};

// (offset, size) bundle for an HWREG sub-field. {0,0} = not present on arch.
struct SubField {
    uint16_t offset;  // starting bit position within the HWREG (LSB = 0)
    uint16_t size;    // width of the sub-field in bits
};

STINKYTOFU_EXPORT SubField schedModeDepMode(GfxArchID arch);
STINKYTOFU_EXPORT SubField schedModeDisableXdlArbStall(GfxArchID arch);

STINKYTOFU_EXPORT bool nameToId(GfxArchID arch, std::string_view name, Id& out);
STINKYTOFU_EXPORT std::string_view idToName(GfxArchID arch, uint16_t id);

STINKYTOFU_EXPORT uint16_t schedModeId(GfxArchID arch);

}  // namespace HwReg
}  // namespace stinkytofu
