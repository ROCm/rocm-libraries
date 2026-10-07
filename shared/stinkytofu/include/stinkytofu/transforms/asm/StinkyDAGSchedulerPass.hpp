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

#include <functional>
#include <memory>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;
struct StinkyInstruction;

/// Creates a DAG scheduler pass that reorders instructions within basic blocks
/// using a dependency DAG and architecture-specific ready queues (e.g. CDNA3/CDNA5).
STINKYTOFU_EXPORT std::unique_ptr<Pass> createStinkyDAGSchedulerPass();

/// One pick of CDNA5ReadyQueue, as its clock saw it.
struct SchedulerPickEvent {
    const StinkyInstruction* inst = nullptr;
    /// Clock at which the instruction's own issue began (a VALU: its co-issue slot).
    int issueClock = 0;
    /// Clock once the instruction's own issue cycles have elapsed.
    int clockAfter = 0;
    /// The clock moved before this pick for a reason other than the instruction's own
    /// timing rules: a policy skip to the end of a window, or a global-read queue credit.
    bool afterSkip = false;
    /// The first pick of a scheduling region; the queue restarts its clock there.
    bool regionStart = false;
};

/// Install a passive observer of CDNA5ReadyQueue picks, for timing-model tests; an
/// empty function removes it. It never changes a scheduling decision.
STINKYTOFU_EXPORT void setSchedulerPickObserver(
    std::function<void(const SchedulerPickEvent&)> observer);
/// The installed observer, or null.
STINKYTOFU_EXPORT const std::function<void(const SchedulerPickEvent&)>* schedulerPickObserver();

}  // namespace stinkytofu
