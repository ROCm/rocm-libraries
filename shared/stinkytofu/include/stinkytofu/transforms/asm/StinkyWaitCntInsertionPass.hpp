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

struct WaitCntInsertionOptions {
    /// Disabled by default: CK_Tensor dataflow freezes after the first solver
    /// sweep, preventing tensor state from propagating through back-edges.
    /// Enable to restore conservative tensor fixed-point iteration.
    bool enableLoopCarriedTokenDeps = false;
};

/**
 * @brief Creates a minimal wait-count insertion pass.
 *
 * Inserts architecture-specific wait instructions so asynchronous memory operations
 * complete before their results are used. Each tracked hardware counter gets its own
 * wait opcode: DS ops -> s_wait_dscnt, vector global/buffer ops -> s_wait_loadcnt,
 * SMRD scalar loads (s_load_*) -> s_wait_kmcnt, and tensor loads -> s_wait_tensorcnt.
 * Tensor waits are reinserted at barriers via a token-matching heuristic.
 */
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyWaitCntInsertionPass(
    WaitCntInsertionOptions options = {});

}  // namespace stinkytofu
