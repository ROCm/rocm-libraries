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
#include <string>

#include "stinkytofu/Export.hpp"

namespace stinkytofu {
class Pass;

/// Creates a pass that absorbs s_set_vgpr_msb side-channel state into operand
/// register identity, then deletes the s_set_vgpr_msb instructions.
///
/// Inverse of InsertVgprMsbPass. After this pass runs, every VGPR operand
/// carries its full physical index in reg.reg.idx with reg.reg.offset = 0,
/// and no s_set_vgpr_msb instructions remain in the IR.
///
/// Intended for the st-opt path when the input assembly was already lowered
/// (i.e. contains s_set_vgpr_msb + encoded VGPR indices). Run before any
/// def-use / dependency-analysis pass; re-lower with InsertVgprMsbPass before
/// emit to preserve round-trip.
///
/// State is reset to MSB=0 at every label boundary (matching the convention
/// in InsertVgprMsbPass). An unlabeled block (a CFG split after a branch) is
/// reached only by fall-through and inherits its layout predecessor's state.
///
/// Records the MSB state at the function's fall-through exit in
/// kFuncExitVgprMsbMetaKey, so InsertVgprMsbPass can hand code after a region
/// the state the input left there.
///
/// In a region (kFuncIsRegionMetaKey), code outside it may enter any label in
/// the mode the input left on that edge, so a label's state is the input's:
/// the previous-mode byte of a setter right after the label, else the mode
/// falling into it, else the mode at a branch to it, else 0. A non-zero state
/// is recorded under labelEntryVgprMsbMetaKey(label); InsertVgprMsbPass then
/// gives every edge into that label the same mode.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createRaiseVgprMsbPass();

inline constexpr const char* kFuncExitVgprMsbMetaKey = "stinkytofu.exitVgprMsb";
/// Set by the driver on a --from-label / --to-label region; absent for a whole function.
inline constexpr const char* kFuncIsRegionMetaKey = "stinkytofu.isRegion";

inline std::string labelEntryVgprMsbMetaKey(const std::string& label) {
    return "stinkytofu.labelEntryVgprMsb." + label;
}

}  // namespace stinkytofu
