/* **************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

// Regression tests for rocsolver_get_alg_mode / rocsolver_set_alg_mode
// argument validation (GitHub issue #15665).

#include <gtest/gtest.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

class AlgModeArgCheck : public ::testing::Test
{
protected:
    rocblas_handle handle = nullptr;

    void SetUp() override
    {
        ASSERT_EQ(rocblas_create_handle(&handle), rocblas_status_success);
    }

    void TearDown() override
    {
        if(handle)
            rocblas_destroy_handle(handle);
    }
};

// rocsolver_get_alg_mode must return rocblas_status_invalid_handle for a null handle.
TEST_F(AlgModeArgCheck, get_null_handle)
{
    rocsolver_alg_mode mode;
    EXPECT_EQ(rocsolver_get_alg_mode(nullptr, rocsolver_function_sytrd_hetrd, &mode),
              rocblas_status_invalid_handle);
}

// rocsolver_get_alg_mode must return rocblas_status_invalid_pointer for a null
// output pointer rather than crashing (GitHub issue #15665).
TEST_F(AlgModeArgCheck, get_null_mode_pointer)
{
    // Verify that a valid call succeeds first (control case).
    rocsolver_alg_mode mode;
    EXPECT_EQ(rocsolver_get_alg_mode(handle, rocsolver_function_sytrd_hetrd, &mode),
              rocblas_status_success);

    // Null output pointer must not crash and must return invalid_pointer.
    EXPECT_EQ(rocsolver_get_alg_mode(handle, rocsolver_function_sytrd_hetrd, nullptr),
              rocblas_status_invalid_pointer);
}

// Exercise every supported selector to confirm none crash with a null pointer.
TEST_F(AlgModeArgCheck, get_null_mode_pointer_all_selectors)
{
    const rocsolver_function selectors[] = {
        rocsolver_function_gesvd,
        rocsolver_function_bdsqr,
        rocsolver_function_sterf,
        rocsolver_function_steqr,
        rocsolver_function_syev_heev,
        rocsolver_function_sytrd_hetrd,
    };
    for(auto sel : selectors)
    {
        EXPECT_EQ(rocsolver_get_alg_mode(handle, sel, nullptr),
                  rocblas_status_invalid_pointer)
            << "selector=" << int(sel);
    }
}

// rocsolver_set_alg_mode must return rocblas_status_invalid_handle for a null handle.
TEST_F(AlgModeArgCheck, set_null_handle)
{
    EXPECT_EQ(
        rocsolver_set_alg_mode(nullptr, rocsolver_function_sytrd_hetrd, rocsolver_alg_mode_gpu),
        rocblas_status_invalid_handle);
}
