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
// Unit tests for the internal BLAS dispatch layer:
//   library/src/common/rocsparse_blas.cpp
//   library/src/common/rocsparse_blas_rocblas.cpp
// Both are compiled into rocsparse-unit-test-device (their include chain needs
// HIP mode). Without ROCSPARSE_WITH_ROCBLAS the rocBLAS backend is a set of
// stubs returning rocsparse_status_not_implemented; with it, the backend is
// only driven through a null rocBLAS handle, which rocBLAS rejects.
//
#include "rocsparse_blas.hpp"
#include "rocsparse_enum_utils.hpp"

#include <gtest/gtest.h>

namespace
{
    rocsparse_status gemm_ex_zero_size(rocsparse::blas_handle h)
    {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        return rocsparse::blas_gemm_ex(h,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       0,
                                       0,
                                       0,
                                       &alpha,
                                       nullptr,
                                       rocsparse_datatype_f32_r,
                                       1,
                                       nullptr,
                                       rocsparse_datatype_f32_r,
                                       1,
                                       &beta,
                                       nullptr,
                                       rocsparse_datatype_f32_r,
                                       1,
                                       nullptr,
                                       rocsparse_datatype_f32_r,
                                       1,
                                       rocsparse_datatype_f32_r,
                                       rocsparse::blas_gemm_alg_standard,
                                       0,
                                       0);
    }
}

TEST(internal_blas, impl_enum_is_invalid)
{
    EXPECT_FALSE(rocsparse::enum_utils::is_invalid(rocsparse::blas_impl_none));
    EXPECT_FALSE(rocsparse::enum_utils::is_invalid(rocsparse::blas_impl_default));
    EXPECT_FALSE(rocsparse::enum_utils::is_invalid(rocsparse::blas_impl_rocblas));
    EXPECT_TRUE(rocsparse::enum_utils::is_invalid(static_cast<rocsparse::blas_impl>(99)));
}

TEST(internal_blas, create_handle_bad_args)
{
    rocsparse::blas_handle h = nullptr;
    EXPECT_EQ(rocsparse::blas_create_handle(nullptr, rocsparse::blas_impl_none),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse::blas_create_handle(&h, static_cast<rocsparse::blas_impl>(99)),
              rocsparse_status_invalid_value);
    EXPECT_EQ(h, nullptr);
}

TEST(internal_blas, impl_none_lifecycle)
{
    rocsparse::blas_handle h = nullptr;
    ASSERT_EQ(rocsparse::blas_create_handle(&h, rocsparse::blas_impl_none),
              rocsparse_status_success);
    ASSERT_NE(h, nullptr);
    EXPECT_EQ(h->blas_impl, rocsparse::blas_impl_none);

    EXPECT_EQ(rocsparse::blas_set_stream(h, nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse::blas_set_pointer_mode(h, rocsparse_pointer_mode_host),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse::blas_set_pointer_mode(h, rocsparse_pointer_mode_device),
              rocsparse_status_success);
    EXPECT_EQ(rocsparse::blas_set_pointer_mode(h, static_cast<rocsparse_pointer_mode>(99)),
              rocsparse_status_invalid_value);
    EXPECT_EQ(gemm_ex_zero_size(h), rocsparse_status_not_implemented);

    EXPECT_EQ(rocsparse::blas_destroy_handle(h), rocsparse_status_success);
}

TEST(internal_blas, null_handle)
{
    EXPECT_EQ(rocsparse::blas_destroy_handle(nullptr), rocsparse_status_success);
    EXPECT_EQ(rocsparse::blas_set_stream(nullptr, nullptr), rocsparse_status_invalid_pointer);
    EXPECT_EQ(rocsparse::blas_set_pointer_mode(nullptr, rocsparse_pointer_mode_host),
              rocsparse_status_invalid_pointer);
    EXPECT_EQ(gemm_ex_zero_size(nullptr), rocsparse_status_invalid_pointer);
}

// A handle whose rocBLAS backend handle is null: every backend call fails,
// either in the stub (no rocBLAS) or in rocBLAS itself (invalid handle).
TEST(internal_blas, rocblas_backend_without_backend_handle)
{
    for(const auto impl : {rocsparse::blas_impl_default, rocsparse::blas_impl_rocblas})
    {
        rocsparse::_blas_handle h;
        h.blas_impl = impl;

        EXPECT_NE(rocsparse::blas_set_stream(&h, nullptr), rocsparse_status_success);
        EXPECT_NE(rocsparse::blas_set_pointer_mode(&h, rocsparse_pointer_mode_host),
                  rocsparse_status_success);
        EXPECT_NE(gemm_ex_zero_size(&h), rocsparse_status_success);
        // The failed backend teardown returns before `delete`, so a stack
        // handle is safe here.
        EXPECT_NE(rocsparse::blas_destroy_handle(&h), rocsparse_status_success);
    }
}

TEST(internal_blas, create_handle_rocblas_backend)
{
    for(const auto impl : {rocsparse::blas_impl_default, rocsparse::blas_impl_rocblas})
    {
        rocsparse::blas_handle h = nullptr;
#ifdef ROCSPARSE_WITH_ROCBLAS
        ASSERT_EQ(rocsparse::blas_create_handle(&h, impl), rocsparse_status_success);
        EXPECT_EQ(rocsparse::blas_set_stream(h, nullptr), rocsparse_status_success);
        EXPECT_EQ(rocsparse::blas_set_pointer_mode(h, rocsparse_pointer_mode_host),
                  rocsparse_status_success);
        EXPECT_EQ(rocsparse::blas_destroy_handle(h), rocsparse_status_success);
#else
        EXPECT_EQ(rocsparse::blas_create_handle(&h, impl), rocsparse_status_not_implemented);
        // The wrapper is allocated before the backend fails; it holds no
        // backend handle, so release it directly.
        ASSERT_NE(h, nullptr);
        EXPECT_EQ(h->blas_impl, impl);
        delete h;
#endif
    }
}
