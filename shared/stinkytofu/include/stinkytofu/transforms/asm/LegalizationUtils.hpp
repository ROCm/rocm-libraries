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
                                               GfxArchID archId);

// Legalize ds_store_b192 instruction
// Expands into two ds_store instructions (b128 + b64).
// Caller must insert s_set_vgpr_msb between the two when they use VGPRs in different MSB ranges.
//
// Example:
//      ds_store_b192 v[0:5], v0 offset:0  →  ds_store_b128 v[0:3], v0 offset:0
//                                            ds_store_b64 v[4:5], v0 offset:16
STINKYTOFU_EXPORT Legalized legalizeDSStoreB192(StinkyInstruction* inst, AsmIRBuilder& irBuilder,
                                                GfxArchID archId);

// Legalize ds_store_b256 instruction
// Expands into two ds_store_b128 instructions.
// Caller must insert s_set_vgpr_msb between the two when they use VGPRs in different MSB ranges.
//
// Example:
//      ds_store_b256 v[0:7], v0 offset:0  →  ds_store_b128 v[0:3], v0 offset:0
//                                            ds_store_b128 v[4:7], v0 offset:16
STINKYTOFU_EXPORT Legalized legalizeDSStoreB256(StinkyInstruction* inst, AsmIRBuilder& irBuilder,
                                                GfxArchID archId);

// Legalize implicit special registers (SCC, VCC, EXEC) on an instruction.
//
// HW flags (Flags.def: IF_ImplicitRead/WriteSCC, IF_ImplicitReadVCC,
// IF_ImplicitRead/WriteEXEC) declare implicit reads/writes that are not
// encoded as explicit operands. This function inspects those flags and adds
// the corresponding singleton register (sized by `wavefrontSize` for
// VCC/EXEC) to the instruction's src/dest list — but only if the register
// is not already present. The check matches by RegType and idx, which is
// sufficient for SCC/VCC/EXEC since they are singletons.
STINKYTOFU_EXPORT void legalizeImplicitSpecialRegisters(StinkyInstruction* inst,
                                                        uint32_t wavefrontSize);

// Add the hidden source that each read-write destination of `inst` implies.
//
// A read-write (RW) destination keeps part of its old value: `s_cmov_b32` keeps
// all of it when the condition is false, and `v_cvt_pk_fp8_f32` keeps the half
// that op_sel does not write. That old value is an input, so the register must
// also be listed as a source, or use-def tracking, the SSA lift and the register
// allocator will not see the read.
//
// The register is added last, where the emitter does not print it, and only if
// it is not already a source, so calling this twice is safe. Call it when the
// instruction is built; the verifier and other early passes rely on it.
//
// Example (high half of an FP8 pack, which keeps the low half of v10):
//      v10 = v_cvt_pk_fp8_f32(v1, v2)  →  v10 = v_cvt_pk_fp8_f32(v1, v2, v10)
//      Both print as: v_cvt_pk_fp8_f32 v10, v1, v2 op_sel:[0,0,1]
STINKYTOFU_EXPORT void legalizeReadWriteSources(StinkyInstruction* inst);

}  // namespace stinkytofu

#endif  // STINKYTOFU_LEGALIZATION_UTILS_HPP
