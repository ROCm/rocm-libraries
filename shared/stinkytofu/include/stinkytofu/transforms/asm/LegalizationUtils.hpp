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

#ifndef STINKYTOFU_LEGALIZATION_UTILS_HPP
#define STINKYTOFU_LEGALIZATION_UTILS_HPP

#include <cstdint>
#include <map>
#include <string>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
// Forward declarations
struct StinkyInstruction;
class AsmIRBuilder;
enum class GfxArchID : uint32_t;

// Result of instruction legalization
struct Legalized {
    // First newly created instruction (nullptr if no legalization)
    StinkyInstruction* first = nullptr;

    // Last newly created instruction (nullptr if no legalization)
    StinkyInstruction* last = nullptr;
};

// Legalize v_nop instruction
// Expands v_nop N into N individual v_nop instructions.
// But if v_nop count is 0, the instruction is removed and <nullptr, nullptr> is returned.
//
// Example:
//      v_nop count=0  →  <removed>
//
//      v_nop count=3  →  v_nop
//                        v_nop
//                        v_nop
STINKYTOFU_EXPORT Legalized legalizeVNop(StinkyInstruction* inst, AsmIRBuilder& irBuilder,
                                         GfxArchID archId);

// Legalize v_cmpx instruction
// On architectures without CMPX SGPR write support, expands into v_cmp + s_and_saveexec
//
// Example:
//      v_cmpx_lt_f32 exec_lo, v0, v1  →  v_cmp_lt_f32 vcc_lo, v0, v1
//                                        s_mov_b32 exec_lo, vcc_lo
STINKYTOFU_EXPORT Legalized legalizeVCmpX(StinkyInstruction* inst, AsmIRBuilder& irBuilder,
                                          GfxArchID archId,
                                          const std::map<std::string, int>& archCaps);

// Legalize s_waitcnt instruction
// On gfx1250, expands into separate wait instructions
//
// Example:
//      s_waitcnt vmcnt(2) lgkmcnt(0)  →  s_wait_loadcnt_dscnt 0x0200
STINKYTOFU_EXPORT Legalized legalizeWaitCnt(StinkyInstruction* inst, AsmIRBuilder& irBuilder,
                                            GfxArchID archId);

// Legalize s_barrier instruction
// On gfx1250, expands into s_barrier_signal + s_barrier_wait
//
// Example:
//      s_barrier  →  s_barrier_signal -1
//                    s_barrier_wait -1
STINKYTOFU_EXPORT Legalized legalizeBarrier(StinkyInstruction* inst, AsmIRBuilder& irBuilder,
                                            GfxArchID archId);

// Legalize ds_load_b192 instruction
// Expands into two ds_load instructions (b128 + b64).
// Caller must insert s_set_vgpr_msb between the two when they use VGPRs in different MSB
// ranges (e.g. b128 in 256~511, b64 in 512~767).
//
// Example:
//      ds_load_b192 v[0:5], v0 offset:0  →  ds_load_b128 v[0:3], v0 offset:0
//                                            ds_load_b64 v[4:5], v0 offset:16
STINKYTOFU_EXPORT Legalized legalizeDSLoadB192(StinkyInstruction* inst, AsmIRBuilder& irBuilder,
                                               GfxArchID archId, bool hasVgprMsb);

// Legalize ds_store_b192 instruction
// Expands into two ds_store instructions (b128 + b64).
// Caller must insert s_set_vgpr_msb between the two when they use VGPRs in different MSB ranges.
//
// Example:
//      ds_store_b192 v[0:5], v0 offset:0  →  ds_store_b128 v[0:3], v0 offset:0
//                                            ds_store_b64 v[4:5], v0 offset:16
STINKYTOFU_EXPORT Legalized legalizeDSStoreB192(StinkyInstruction* inst, AsmIRBuilder& irBuilder,
                                                GfxArchID archId, bool hasVgprMsb);

// Legalize ds_store_b256 instruction
// Expands into two ds_store_b128 instructions.
// Caller must insert s_set_vgpr_msb between the two when they use VGPRs in different MSB ranges.
//
// Example:
//      ds_store_b256 v[0:7], v0 offset:0  →  ds_store_b128 v[0:3], v0 offset:0
//                                            ds_store_b128 v[4:7], v0 offset:16
STINKYTOFU_EXPORT Legalized legalizeDSStoreB256(StinkyInstruction* inst, AsmIRBuilder& irBuilder,
                                                GfxArchID archId, bool hasVgprMsb);

}  // namespace stinkytofu

#endif  // STINKYTOFU_LEGALIZATION_UTILS_HPP
