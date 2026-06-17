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

/// LongBranchLoweringPass
///
/// Recognizes the rocisa long-branch idiom in raw .s input and stamps a
/// LabelData modifier on the terminating s_setpc_b64 so that downstream
/// passes (CFGBuilder, dominator analysis, etc.) can treat it as a
/// statically-known direct branch instead of an opaque indirect jump.
///
/// The recognized idioms (from rocisa SLongBranchPositive / SLongBranchNegative
/// / SLongBranch in rocisa/include/instruction/extension.hpp):
///
///   Positive form (and the positive arm of the dispatching SLongBranch):
///     s_getpc_b64 s[D:D+1]
///     s_add_i32   sX, LBL, 4
///     s_add_u32   sD,   sD,   sX
///     s_addc_u32  sD+1, sD+1, 0
///     s_setpc_b64 s[D:D+1]
///
///   Negative form (and the negative arm of SLongBranch):
///     s_getpc_b64 s[D:D+1]
///     s_add_i32   sX, LBL, 4
///     s_abs_i32   sX, sX                  // optional in negative arm
///     s_sub_u32   sD,   sD,   sX
///     s_subb_u32  sD+1, sD+1, 0
///     s_setpc_b64 s[D:D+1]
///
/// Pattern matching is local: the pass walks each basic block linearly and,
/// for every s_setpc_b64 missing branch metadata, looks back within the same
/// block for the producing s_add_i32 ?, LBL, 4 anchor. The label "LBL" comes
/// from that anchor's first source operand (a LiteralString).
///
/// Run order: this pass must run BEFORE CFGBuilderPass on the still-flat,
/// single-block function representation produced by RawAsmParser; CFGBuilder
/// reads the LabelData modifier when computing successor edges.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createLongBranchLoweringPass();

}  // namespace stinkytofu
