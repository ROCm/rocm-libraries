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
// Unit tests for the rocSPARSE internal csrmm_select_default_alg selector and
// its line_nnz_profile guard logic
// (library/src/level3/rocsparse_csrmm_default_alg.cpp). Split out of
// unit_test_internal_hostblocks.cpp by topic.
//
// NOTE ON TARGET: host-pure selector logic, but its include chain
// (rocsparse_csrmm.hpp -> rocsparse_common.hpp) requires the HIP compile mode,
// so this file builds into the GPU test binary (rocsparse-unit-test-device) and
// the defining TU rocsparse_csrmm_default_alg.cpp is compiled in via
// ROCSPARSE_UNIT_TEST_DEVICE_LIB_SOURCES. No kernels are launched.
//
// The selector only ever upgrades the *default algorithm* (rocsparse_csrmm_alg_default)
// to the nnz-split kernel, and only when the profile is present and the
// longest-line imbalance test (profile.max * cu_count >= 3 * profile.nnz)
// fires. An explicit non-default algorithm is always preserved. Every guard
// that keeps the historical row-split default is checked below.
//
#include "unit_test_utils.hpp"

#include "rocsparse_csrmm.hpp" // csrmm_select_default_alg + line_nnz_profile

#include <gtest/gtest.h>

namespace
{
    // Compute-unit count large enough that the imbalance test fires for a
    // maximally imbalanced profile; used wherever a test needs a "big GPU".
    constexpr int32_t large_cu_count = 64;

    // The imbalance crossover constant baked into the selector: the default is
    // upgraded once profile.max * cu_count >= imbalance_C * profile.nnz.
    constexpr int64_t imbalance_C = 3;

    rocsparse_csrmm_alg select_alg(rocsparse_operation                trans_a,
                                   bool                               is_batched,
                                   int32_t                            cu_count,
                                   const rocsparse::line_nnz_profile& profile,
                                   rocsparse_csrmm_alg                start_alg)
    {
        rocsparse_csrmm_alg alg = start_alg;
        EXPECT_EQ(rocsparse::csrmm_select_default_alg(trans_a, is_batched, cu_count, profile, alg),
                  rocsparse_status_success);
        return alg;
    }

    // A profile that trips the imbalance test on a large GPU (max == nnz, so
    // max * cu_count = nnz * cu_count >= 3 * nnz for any cu_count >= 3).
    rocsparse::line_nnz_profile tripping_profile()
    {
        rocsparse::line_nnz_profile p{};
        p.known = true;
        p.nnz   = 100;
        p.max   = 100;
        return p;
    }
}

// An explicit (non-default) algorithm choice must be preserved even when the
// profile would otherwise trip the imbalance upgrade.
TEST(internal_hostblocks_csrmm_alg, explicit_alg_unchanged)
{
    EXPECT_EQ(select_alg(rocsparse_operation_none,
                         false,
                         large_cu_count,
                         tripping_profile(),
                         rocsparse_csrmm_alg_nnz_split),
              rocsparse_csrmm_alg_nnz_split);
}

// The selector auto-tunes ONLY the default algorithm. Every other explicit
// algorithm (row_split / nnz_split / merge_path) is returned unchanged, even
// under a profile that would trip the upgrade were the input the default.
TEST(internal_hostblocks_csrmm_alg, other_algorithms_preserved)
{
    const rocsparse::line_nnz_profile tripping = tripping_profile();
    for(rocsparse_csrmm_alg alg : {rocsparse_csrmm_alg_row_split,
                                   rocsparse_csrmm_alg_nnz_split,
                                   rocsparse_csrmm_alg_merge_path})
    {
        EXPECT_EQ(select_alg(rocsparse_operation_none, false, large_cu_count, tripping, alg), alg)
            << "explicit alg " << static_cast<int>(alg) << " must be preserved";
    }
}

TEST(internal_hostblocks_csrmm_alg, guards_keep_default)
{
    const rocsparse::line_nnz_profile tripping = tripping_profile();

    // Transposed multiply: profile is not applicable -> stays default.
    EXPECT_EQ(select_alg(rocsparse_operation_transpose,
                         false,
                         large_cu_count,
                         tripping,
                         rocsparse_csrmm_alg_default),
              rocsparse_csrmm_alg_default);

    // Batched multiply -> stays default.
    EXPECT_EQ(
        select_alg(
            rocsparse_operation_none, true, large_cu_count, tripping, rocsparse_csrmm_alg_default),
        rocsparse_csrmm_alg_default);

    // Unknown profile -> stays default.
    {
        rocsparse::line_nnz_profile unknown{};
        unknown.known = false;
        unknown.nnz   = 100;
        unknown.max   = 100;
        EXPECT_EQ(select_alg(rocsparse_operation_none,
                             false,
                             large_cu_count,
                             unknown,
                             rocsparse_csrmm_alg_default),
                  rocsparse_csrmm_alg_default);
    }

    // Non-positive nnz -> stays default.
    {
        rocsparse::line_nnz_profile zero_nnz{};
        zero_nnz.known = true;
        zero_nnz.nnz   = 0;
        zero_nnz.max   = 100;
        EXPECT_EQ(select_alg(rocsparse_operation_none,
                             false,
                             large_cu_count,
                             zero_nnz,
                             rocsparse_csrmm_alg_default),
                  rocsparse_csrmm_alg_default);
    }

    // Non-positive compute-unit count -> stays default.
    EXPECT_EQ(select_alg(rocsparse_operation_none, false, 0, tripping, rocsparse_csrmm_alg_default),
              rocsparse_csrmm_alg_default);
}

TEST(internal_hostblocks_csrmm_alg, imbalance_threshold)
{
    // Balanced enough to keep row-split: max * cu = 1 * 1 = 1 < 3 * 100.
    {
        rocsparse::line_nnz_profile p{};
        p.known = true;
        p.nnz   = 100;
        p.max   = 1;
        EXPECT_EQ(select_alg(rocsparse_operation_none, false, 1, p, rocsparse_csrmm_alg_default),
                  rocsparse_csrmm_alg_default);
    }

    // Just below the crossover: max * cu = 100 * 2 = 200 < 3 * 100 = 300.
    {
        rocsparse::line_nnz_profile p{};
        p.known = true;
        p.nnz   = 100;
        p.max   = 100;
        EXPECT_EQ(select_alg(rocsparse_operation_none, false, 2, p, rocsparse_csrmm_alg_default),
                  rocsparse_csrmm_alg_default);
    }

    // Exactly at the crossover (>=): 100 * 3 = 300 >= 300 -> upgrade to nnz-split.
    {
        rocsparse::line_nnz_profile p{};
        p.known                    = true;
        p.nnz                      = 100;
        p.max                      = 100;
        const int32_t crossover_cu = static_cast<int32_t>(imbalance_C); // 3
        EXPECT_EQ(
            select_alg(
                rocsparse_operation_none, false, crossover_cu, p, rocsparse_csrmm_alg_default),
            rocsparse_csrmm_alg_nnz_split);
    }

    // Clearly imbalanced -> nnz-split.
    {
        rocsparse::line_nnz_profile p{};
        p.known = true;
        p.nnz   = 100;
        p.max   = 90;
        EXPECT_EQ(
            select_alg(
                rocsparse_operation_none, false, large_cu_count, p, rocsparse_csrmm_alg_default),
            rocsparse_csrmm_alg_nnz_split);
    }
}
