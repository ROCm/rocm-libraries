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

/**
 * @brief Strip wait-counter instructions from a function.
 *
 * Walks every basic block approved by @c PassContext::shouldProcessBasicBlock
 * and removes:
 *   1. Every instruction carrying the @c IF_WaitCnt flag, which covers the
 *      standard wait-counter opcodes (@c s_wait_dscnt, @c s_wait_loadcnt,
 *      @c s_wait_storecnt, @c s_wait_asynccnt, @c s_wait_kmcnt,
 *      @c s_wait_xcnt, @c s_wait_loadcnt_dscnt, @c s_wait_storecnt_dscnt,
 *      and @c s_waitcnt). These are the instructions for which
 *      @c isWaitCnt() returns true.
 *   2. Every instruction carrying the @c IF_WaitTensorCnt flag
 *      (@c s_wait_tensorcnt) iff @p removeTensorWaitCnt is true.
 *      @c IF_WaitTensorCnt is a separate flag from @c IF_WaitCnt, so an
 *      explicit second branch is needed; @c isWaitCnt() does not match it.
 *
 * @param removeTensorWaitCnt When true (default) also strip
 *                            @c s_wait_tensorcnt; when false leave tensor
 *                            waits in place so a subsequent insertion pass
 *                            can reuse them.
 */
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyRemoveWaitCntPass(
    bool removeTensorWaitCnt = true);

}  // namespace stinkytofu
