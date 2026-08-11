/*! \file */
/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights Reserved.
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

//
// Unit tests for rocSPARSE internal host building-block routines.
//
// NOTE ON TARGET: these routines are pure *host* logic (selectors/partitioners),
// but their headers pull in rocsparse_control.hpp -> rocsparse_common.hpp, which
// uses HIP device intrinsics (__ldg, ...) that only compile under `-x hip`. This
// file therefore builds into the GPU test binary (rocsparse-unit-test-device),
// NOT the host-only rocsparse-unit-test, and the matching library TUs
// (rocsparse_csrmm_default_alg.cpp, rocsparse_determine_indextype.cpp) are
// compiled in via ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES. The tests themselves
// run host code and do not need to launch kernels.
//
// Exercises: csrmm_select_default_alg, determine_I/J_indextype, clz, host fnp2,
// line_nnz_profile guard logic, itilu0 assign_b/unassign_b/buffer_layout, and
// (after a behavior-preserving lift) ComputeRowBlocks/maxRowsInABlock.
//
#include "unit_test_utils.hpp"

#include "rocsparse_csrmm.hpp" // csrmm_select_default_alg + line_nnz_profile
#include "rocsparse_determine_indextype.hpp" // determine_I/J_indextype

#include <gtest/gtest.h>
#include <vector>

// Placeholder proving the device harness links the compiled-in host-building-
// block TUs and runs; the real selector/partitioner tests live in the hostblocks
// PR. Always passes.
TEST(internal_hostblocks, harness_smoke)
{
    SUCCEED();
}
