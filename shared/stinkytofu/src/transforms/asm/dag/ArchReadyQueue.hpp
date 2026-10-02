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
class PassContext;

namespace dag {
class ReadyQueue;

/// Behaviour whose right answer depends on what the caller is doing rather than
/// on the architecture, so it cannot live in the arch tables.
struct ArchReadyQueueOptions {
    /// Let a full co-issue window release the next matrix op even when the
    /// region's latency-hiding quota has not been met.
    ///
    /// That quota is a region-level target: it holds matrix ops back until enough
    /// non-matrix work has issued across the whole region. For a caller building a
    /// schedule from nothing that is the right question to ask. For one repairing
    /// a schedule that already satisfies it, it is not: the quota keeps filling a
    /// window that is already full, which hides nothing and only pushes the next
    /// matrix op past the window it was supposed to fill.
    bool fullWindowOverridesHideBudget = false;
};

/// The ready queue implementing \p passCtx's architecture, which is where the
/// co-issue and hazard rules for that architecture live.
///
/// This is the only way to obtain one. The arch queues are file-local to the
/// translation unit that defines them, so a new architecture adds a case here
/// and every caller picks it up without naming the type.
std::unique_ptr<ReadyQueue> createArchReadyQueue(const PassContext& passCtx,
                                                 ArchReadyQueueOptions options = {});

}  // namespace dag
}  // namespace stinkytofu
