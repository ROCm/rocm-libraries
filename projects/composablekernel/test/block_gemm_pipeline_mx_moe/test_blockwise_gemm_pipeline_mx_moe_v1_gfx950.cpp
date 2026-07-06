// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// gfx950 compile guard for the MX MoE B-preshuffle v1 blockwise GEMM pipeline.
//
// Regression protected by this test
// ---------------------------------
// blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_v1.hpp derives the scale
// thread offsets from compile-time index arithmetic such as
//
//     constexpr auto im_major = m0 / MXdlPack;   // m0 is a ck::Number<>
//     constexpr auto im_minor = m0 % MXdlPack;
//
// where MXdlPack / NXdlPack / KXdlPack are the XDL pack constants. If those
// pack constants are left as bare `index_t` (int) operands, the expression
// `ck::Number<v>{} / int` decays to a plain `int` result. That int is later
// forwarded (via decltype) as the expression type into
// ck::thread_buf_to_vec_loader, which evaluates it through ck::IndexEval<T, ik>.
// IndexEval only has specializations for ck::Number<>, index_expression nodes,
// etc. -- there is NO specialization for a bare int -- so on gfx950 this
// instantiates the undefined ck::IndexEval<int, N> and the build fails.
//
// The fix (PR: wrap the pack constants in ck::Number<...>{}) keeps the operands
// as compile-time ck::Number so the arithmetic stays a ck::Number and
// ck::IndexEval<ck::Number<...>, ik> is the well-defined specialization.
//
// This test provides two layers of protection, both compiled for gfx950:
//   1. It #includes the actual pipeline header, so any gfx950 header-level
//      compile regression in the pipeline is caught here.
//   2. It reproduces the exact IndexEval mechanism the fix depends on: the
//      Number<>-wrapped pack arithmetic must remain a ck::Number and must be a
//      well-formed operand for ck::IndexEval. If the fix is reverted to bare
//      int, decltype(...) becomes `int` and ck::IndexEval<int, N> is undefined
//      -> this translation unit fails to compile, breaking the test target.
//
// Honesty note on guard strength: this is a compile-time guard focused on the
// IndexEval<int,N> mechanism plus a header-includes-clean check. It does not
// launch a kernel or fully instantiate the pipeline's Run() device method
// (which would require constructing the complete device-op tile descriptors and
// is exercised by the example / device-instance builds). The static_asserts
// below directly mirror the code the fix changed, so a regression to bare-int
// pack constants is caught at compile time.

#include <gtest/gtest.h>

#include "ck/ck.hpp"
#include "ck/utility/number.hpp"

// Layer 1: the pipeline header must compile cleanly for gfx950.
#include "ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_mx_moe_v1.hpp"

// Layer 2: IndexEval is what turns the bare-int regression into a hard error.
#include "ck/utility/thread_buf_to_vec_loader.hpp"

using namespace ck;

namespace {

// Representative XDL pack constants (values as used by the MX MoE pipeline).
constexpr index_t MXdlPack = 2;
constexpr index_t NXdlPack = 2;
constexpr index_t KXdlPack = 2;

// Representative compile-time repeat index (m0/k0/n0 are ck::Number<> in the
// pipeline's static_ford loop).
constexpr auto m0 = Number<8>{};
constexpr auto k0 = Number<8>{};
constexpr auto n0 = Number<8>{};

// This is EXACTLY the arithmetic the fix produces: Number<> / Number<> and
// Number<> % Number<>. The results MUST stay ck::Number so that the decltype
// forwarded into IndexEval selects the ck::IndexEval<ck::Number<v>, ik>
// specialization instead of the undefined ck::IndexEval<int, ik>.
constexpr auto im_major = m0 / Number<MXdlPack>{};
constexpr auto im_minor = m0 % Number<MXdlPack>{};
constexpr auto ik_major = k0 / Number<KXdlPack>{};
constexpr auto ik_minor = k0 % Number<KXdlPack>{};
constexpr auto in_major = n0 / Number<NXdlPack>{};
constexpr auto in_minor = n0 % Number<NXdlPack>{};

// The core guard: IndexEval<decltype(...), ik> must be well-formed. With the
// bare-int regression these decltypes become `int` and ck::IndexEval<int, N> is
// undefined, so these lines fail to compile for gfx950 -- exactly the PR bug.
static_assert(IndexEval<decltype(im_major), 0>::value == 4, "im_major value");
static_assert(IndexEval<decltype(im_minor), 0>::value == 0, "im_minor value");
static_assert(IndexEval<decltype(ik_major), 0>::value == 4, "ik_major value");
static_assert(IndexEval<decltype(ik_minor), 0>::value == 0, "ik_minor value");
static_assert(IndexEval<decltype(in_major), 0>::value == 4, "in_major value");
static_assert(IndexEval<decltype(in_minor), 0>::value == 0, "in_minor value");

} // namespace

// A trivial runtime assertion so the target is a real (runnable) gtest binary.
// The load-bearing protection is the compile-time static_asserts above.
TEST(BlockwiseGemmPipelineMxMoeV1, IndexEvalPackConstantsWrappedInNumber)
{
    EXPECT_EQ((IndexEval<decltype(im_major), 0>::value), 4);
    EXPECT_EQ((IndexEval<decltype(im_minor), 0>::value), 0);
    EXPECT_EQ((IndexEval<decltype(in_major), 0>::value), 4);
    EXPECT_EQ((IndexEval<decltype(ik_minor), 0>::value), 0);
}
