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

#include <memory>

namespace stinkytofu {
class Pass;

/// Creates a pass that inserts s_set_vgpr_msb instructions before VOP instructions
/// whose VGPR operands require MSB configuration.
///
/// For architectures with HasVgprMSB (e.g. gfx1250), VGPRs above index 255 require
/// the hardware VGPR_OFF register to be configured via s_set_vgpr_msb. This pass
/// scans instructions, computes the required MSB value per operand slot, and inserts
/// s_set_vgpr_msb when the required value differs from the current state.
///
/// After a label (branch target), the pass conservatively resets MSB state and
/// inserts an s_nop before s_set_vgpr_msb to satisfy hardware constraints.
std::unique_ptr<Pass> createInsertVgprMsbPass();

}  // namespace stinkytofu
