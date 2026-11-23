/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc.
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

#include <cstddef>
#include <cstdint>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
struct StinkyInstruction;

/// One register, the unit a destination field is measured against: a field
/// declaring fewer bits than this writes part of a register and keeps the rest.
constexpr uint32_t kRegisterBits = 32;

/// Bits of destination slot \p slot that \p inst leaves as they were, as a mask
/// over one register, or 0 when the write covers the whole register.
///
/// Two facts on the instruction decide it: the instruction table gives the width
/// of the write, and the op_sel modifier gives its position. `v_cvt_pk_fp8_f32`
/// declares a 16-bit `D0`, so `op_sel:[0,0,1]` writes `[31:16]` and keeps
/// `[15:0]`.
///
/// A narrow write that states no position counts as a whole-register write.
/// Naming a half is how a producer says it is building one register out of
/// several writes; without that the kept half is a value nobody reads, and
/// tying it would cost a register in every kernel that converts to f16.
/// `AllocationConstraints::unreadPartialWrites()` names the shape that would
/// break the assumption: a narrow write whose result nothing reads.
STINKYTOFU_EXPORT uint32_t keptDestinationBits(const StinkyInstruction& inst, size_t slot);

/// True when \p inst reads the old value of destination slot \p slot: either the
/// instruction table marks the field read-write, a property of the opcode
/// (`s_cmov_b32`, `v_swap_b32`, `v_cvt_sr_fp8_f32`, the buffer atomics), or the
/// write keeps part of the register, a property of the instance.
///
/// \p slot counts destination fields, so it indexes `getDestRegs()`.
STINKYTOFU_EXPORT bool readsDestination(const StinkyInstruction& inst, size_t slot);

}  // namespace stinkytofu
