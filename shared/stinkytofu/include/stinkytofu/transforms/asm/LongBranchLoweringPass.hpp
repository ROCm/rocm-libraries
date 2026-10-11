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
class BasicBlock;
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
///   Positive form:
///     s_getpc_b64 s[D:D+1]
///     s_add_i32   sX, LBL, 4
///     s_add_u32   sD,   sD,   sX
///     s_addc_u32  sD+1, sD+1, 0
///     s_setpc_b64 s[D:D+1]
///
///   Negative form:
///     s_getpc_b64 s[D:D+1]
///     s_add_i32   sX, LBL, 4
///     s_abs_i32   sX, sX                  // optional
///     s_sub_u32   sD,   sD,   sX
///     s_subb_u32  sD+1, sD+1, 0
///     s_setpc_b64 s[D:D+1]
///
///   Dispatching SLongBranch (both s_setpc_b64 get LBL):
///     s_getpc_b64 s[D:D+1]
///     s_add_i32   sX, LBL, 4
///     s_cmp_ge_i32 sX, 0
///     s_cbranch_scc1 POS
///     <negative arm: s_abs_i32 .. s_setpc_b64>
///   POS:
///     <positive arm: s_add_u32 .. s_setpc_b64>
///
/// Pattern matching is local: for every s_setpc_b64 missing branch metadata,
/// the pass walks back within the same (flat) block through the def chain to
/// the s_add_i32 ?, LBL, +/-4 anchor and the s_getpc_b64. A link that is
/// clobbered in between, a mixed add/sub pair, or any other label or branch
/// on the path leaves the s_setpc_b64 unannotated.
///
/// CFGBuilderPass calls annotateLongBranches() on the flat block before it
/// splits, so every CFG build sees these edges; the pass remains for callers
/// that want the annotation without a CFG. It has to see the still-flat,
/// single-block function produced by RawAsmParser: the dispatching form spans
/// a label.
STINKYTOFU_EXPORT std::unique_ptr<Pass> createLongBranchLoweringPass();

/// Stamps LabelData on every unannotated s_setpc_b64 in \p bb that ends one of
/// the idioms above.
STINKYTOFU_EXPORT void annotateLongBranches(BasicBlock& bb);

}  // namespace stinkytofu
