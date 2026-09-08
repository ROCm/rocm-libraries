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

namespace cluster_barrier {

/// Single toggle for Rule 3 cross-loop hoisting (0 = off, 1 = on).
///
/// To enable cross-loop in the pass **and** compile the cross-loop unit tests, set this macro
/// to 1 and rebuild (rocisa / stinkytofu and `unit_tests`). The tests gate on the same
/// `STINKY_KRULE3_CROSS_LOOP` via `IF_RULE3_CROSS_LOOP` in tests/unit/TestHelpers.hpp — there
/// is no separate UT define; pass and tests must stay in sync.
///
/// `kRule3CrossLoop` mirrors this macro for runtime `if` checks. `#if STINKY_KRULE3_CROSS_LOOP`
/// in tests reads the same value. The preprocessor cannot use `constexpr`, so change this macro
/// (not `kRule3CrossLoop`).
#ifndef STINKY_KRULE3_CROSS_LOOP
#define STINKY_KRULE3_CROSS_LOOP 0
#endif

/// Gate for Rule 3 hoisting and scheduler live-out SCC lead.
#if STINKY_KRULE3_CROSS_LOOP
inline constexpr bool kRule3CrossLoop = true;
#else
inline constexpr bool kRule3CrossLoop = false;
#endif

}  // namespace cluster_barrier

class Pass;

/// \p streamKMulticast and \p pgrValue only enable the Rule 3 producer-side
/// tensor drain for StreamK cluster multicast at PrefetchGlobalRead >= 2; the
/// barrier placement rules themselves derive everything they need from the IR.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createInsertClusterBarrierPass(
    bool streamKMulticast = false, int pgrValue = 1);

}  // namespace stinkytofu
