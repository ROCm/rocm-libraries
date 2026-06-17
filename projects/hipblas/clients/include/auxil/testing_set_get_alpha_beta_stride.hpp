/* ************************************************************************
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * ************************************************************************ */

#include "testing_common.hpp"

/* ============================================================================================ */

inline void testname_set_get_alpha_beta_stride(const Arguments& arg, std::string& name)
{
    ArgumentModel<>{}.test_name(arg, name);
}

void testing_set_get_alpha_beta_stride(const Arguments& arg)
{
    bool FORTRAN = arg.api == hipblas_client_api::FORTRAN;
    auto hipblasSetBatchAlphaStrideFn
        = FORTRAN ? hipblasSetBatchAlphaStrideFortran : hipblasSetBatchAlphaStride;
    auto hipblasGetBatchAlphaStrideFn
        = FORTRAN ? hipblasGetBatchAlphaStrideFortran : hipblasGetBatchAlphaStride;
    auto hipblasSetBatchBetaStrideFn
        = FORTRAN ? hipblasSetBatchBetaStrideFortran : hipblasSetBatchBetaStride;
    auto hipblasGetBatchBetaStrideFn
        = FORTRAN ? hipblasGetBatchBetaStrideFortran : hipblasGetBatchBetaStride;

    hipblasLocalHandle handle(arg);

#ifndef __HIP_PLATFORM_NVCC__
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_DEVICE));

    hipblasStride batch_alpha_stride = -1;
    hipblasStride batch_beta_stride  = -1;
    CHECK_HIPBLAS_ERROR(hipblasGetBatchAlphaStrideFn(handle, &batch_alpha_stride));
    CHECK_HIPBLAS_ERROR(hipblasGetBatchBetaStrideFn(handle, &batch_beta_stride));
    EXPECT_EQ(0, batch_alpha_stride);
    EXPECT_EQ(0, batch_beta_stride);

    CHECK_HIPBLAS_ERROR(hipblasSetBatchAlphaStrideFn(handle, 7));
    CHECK_HIPBLAS_ERROR(hipblasGetBatchAlphaStrideFn(handle, &batch_alpha_stride));
    EXPECT_EQ(7, batch_alpha_stride);
    CHECK_HIPBLAS_ERROR(hipblasSetBatchBetaStrideFn(handle, 7));
    CHECK_HIPBLAS_ERROR(hipblasGetBatchBetaStrideFn(handle, &batch_beta_stride));
    EXPECT_EQ(7, batch_beta_stride);

    CHECK_HIPBLAS_ERROR(hipblasSetBatchAlphaStrideFn(handle, 0));
    CHECK_HIPBLAS_ERROR(hipblasSetBatchBetaStrideFn(handle, 0));
    CHECK_HIPBLAS_ERROR(hipblasGetBatchAlphaStrideFn(handle, &batch_alpha_stride));
    CHECK_HIPBLAS_ERROR(hipblasGetBatchBetaStrideFn(handle, &batch_beta_stride));
    EXPECT_EQ(0, batch_alpha_stride);
    EXPECT_EQ(0, batch_beta_stride);

    // stored regardless of mode, but only utilized in device mode kernels
    CHECK_HIPBLAS_ERROR(hipblasSetPointerMode(handle, HIPBLAS_POINTER_MODE_HOST));

    CHECK_HIPBLAS_ERROR(hipblasSetBatchAlphaStrideFn(handle, 7));
    CHECK_HIPBLAS_ERROR(hipblasSetBatchBetaStrideFn(handle, 11));

    CHECK_HIPBLAS_ERROR(hipblasGetBatchAlphaStrideFn(handle, &batch_alpha_stride));
    EXPECT_EQ(7, batch_alpha_stride);
    CHECK_HIPBLAS_ERROR(hipblasGetBatchBetaStrideFn(handle, &batch_beta_stride));
    EXPECT_EQ(11, batch_beta_stride);
#else
    hipblasStride batch_alpha_stride = 0;
    hipblasStride batch_beta_stride  = 0;
    EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, hipblasSetBatchAlphaStrideFn(handle, 7));
    EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED,
              hipblasGetBatchAlphaStrideFn(handle, &batch_alpha_stride));
    EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED, hipblasSetBatchBetaStrideFn(handle, 11));
    EXPECT_EQ(HIPBLAS_STATUS_NOT_SUPPORTED,
              hipblasGetBatchBetaStrideFn(handle, &batch_beta_stride));
#endif
}
