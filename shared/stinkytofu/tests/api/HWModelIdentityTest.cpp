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

// HWModel address identity across the shared-library boundary.
//
// The models in HWModel.cpp are defined out of line, and hwModelForArch() is
// exported, specifically so that every caller observes ONE object per arch:
// PassContext caches a `const HWModel*`, so two copies would mean a cached
// pointer that does not compare equal to a freshly looked-up model.
//
// The equivalent assertions in unit/asm/HWModelTest.cpp cannot detect a
// violation. unit_tests links stinkytofu_static, so HWModel.cpp, PassManager.cpp
// and the caller all land in one binary and the addresses agree no matter how the
// models are defined. This binary links the SHARED library, where the project
// builds with CMAKE_CXX_VISIBILITY_PRESET=hidden and STINKYTOFU_EXPORT is empty
// for consumers — so here the comparison is real.
//
// What this would catch: making hwModelForArch() a header-inline function over
// header-inline model objects. The test binary would then resolve its own copy
// while PassContext::getHWModel(), compiled into libstinkytofu.so, keeps the
// library's — and these EXPECT_EQs would fail.
//
// Deliberately NOT asserted here: that hazards.rules equals the test binary's
// view of kCdna5HazardRules. That is an `inline constexpr` array in a header, and
// under hidden visibility a header-inline variable is exactly the thing that may
// legitimately differ across the boundary. Pinning it would test the toolchain,
// not this design.

#include <gtest/gtest.h>

#include <array>

#include "stinkytofu/core/PassManager.hpp"
#include "stinkytofu/hardware/HWModel.hpp"

using namespace stinkytofu;

namespace {
constexpr std::array<int, 3> kGfx1250 = {12, 5, 0};
}  // namespace

// Repeated lookups through the exported accessor must yield one object.
TEST(HWModelIdentityAcrossSharedLib, LookupIsStable) {
    EXPECT_EQ(&hwModelForArch(kGfx1250), &hwModelForArch(kGfx1250));
}

// An unlisted arch falls back to the gfx1250 object itself, not a second copy.
TEST(HWModelIdentityAcrossSharedLib, FallbackSharesTheGfx1250Object) {
    EXPECT_EQ(&hwModelForArch({9, 4, 2}), &hwModelForArch(kGfx1250));
}

// The pointer PassContext caches inside libstinkytofu.so must be the same object
// this binary sees. This is the assertion the static-linked unit test cannot make.
TEST(HWModelIdentityAcrossSharedLib, CachedPointerMatchesFreshLookup) {
    GemmTileConfig cfg;
    cfg.arch = kGfx1250;
    PassContext ctx;
    ctx.setGemmTileConfig(cfg);

    EXPECT_EQ(&ctx.getHWModel(), &hwModelForArch(kGfx1250));
}

// A context that was never configured falls back to the same shared default.
TEST(HWModelIdentityAcrossSharedLib, BareContextMatchesFreshLookup) {
    PassContext ctx;
    EXPECT_EQ(&ctx.getHWModel(), &hwModelForArch(kGfx1250));
}
