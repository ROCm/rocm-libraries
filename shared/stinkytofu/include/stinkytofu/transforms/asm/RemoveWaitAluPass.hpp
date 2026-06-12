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

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Strip s_setreg* that write the DEP_MODE sub-field of SCHED_MODE
/// (hwreg(26, offset=0, size=2)); other SCHED_MODE sub-fields are left intact.
/// Also clear all six GPR hazard fields (va_vdst / vm_vsrc / va_sdst / va_ssrc
/// / va_vcc / sa_sdst) on every s_wait_alu; s_wait_alu is deleted only when
/// every field including hold_cnt is no-wait. Pre-pass for InsertWaitAluPass so
/// existing mode1 artifacts cannot mislead the new mode2 placement.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRemoveWaitAluPass();

}  // namespace stinkytofu
