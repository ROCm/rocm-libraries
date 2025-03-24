/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************/

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/internal/hiptensor_utility.hpp>
#include <hiptensor_unary_element_wise_operation.hpp>
#include <iostream>

using ck::tensor_operation::element_wise::HiptensorBinaryOp;
using ck::tensor_operation::element_wise::HiptensorUnaryOp;

__global__ void unary_op_kernel(const float* x, float* y, int size, hiptensorOperator_t op_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        HiptensorUnaryOp op(op_type);
        op(y[idx], x[idx]);
    }
}

__global__ void binary_op_kernel(
    const float* x1, const float* x2, float* y, int size, hiptensorOperator_t op_type)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        HiptensorBinaryOp op(op_type);
        op(y[idx], x1[idx], x2[idx]);
    }
}

float unaryOpOnDeviceTest(float x, hiptensorOperator_t op_type)
{
    const int array_size  = 1;
    const int array_bytes = array_size * sizeof(float);

    // Host arrays
    float array_x[array_size] = {0.0F};
    float array_y[array_size] = {0.0F};

    array_x[0] = x;

    // Device arrays
    float *d_x, *d_y;
    CHECK_HIP_ERROR(hipMalloc(&d_x, array_bytes));
    CHECK_HIP_ERROR(hipMalloc(&d_y, array_bytes));

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(d_x, array_x, array_bytes, hipMemcpyHostToDevice));

    // Launch kernel
    int block_size = 64;
    int grid_size  = (array_size + block_size - 1) / block_size;
    hipLaunchKernelGGL(
        unary_op_kernel, dim3(grid_size), dim3(block_size), 0, 0, d_x, d_y, array_size, op_type);

    // Copy result back to host
    CHECK_HIP_ERROR(hipMemcpy(array_y, d_y, array_bytes, hipMemcpyDeviceToHost));

    // Free device memory
    CHECK_HIP_ERROR(hipFree(d_x));
    CHECK_HIP_ERROR(hipFree(d_y));

    return array_y[0];
}

float unaryOpOnHostTest(float x, hiptensorOperator_t op_type)
{
    HiptensorUnaryOp op(op_type);
    float            y;
    op(y, x);
    return y;
}

float binaryOpOnDeviceTest(float x1, float x2, hiptensorOperator_t op_type)
{
    const int array_size  = 1;
    const int array_bytes = array_size * sizeof(float);

    // Host arrays
    float array_x1[array_size] = {0.0F};
    float array_x2[array_size] = {0.0F};
    float array_y[array_size]  = {0.0F};

    array_x1[0] = x1;
    array_x2[0] = x2;

    // Device arrays
    float *d_x1, *d_x2, *d_y;
    CHECK_HIP_ERROR(hipMalloc(&d_x1, array_bytes));
    CHECK_HIP_ERROR(hipMalloc(&d_x2, array_bytes));
    CHECK_HIP_ERROR(hipMalloc(&d_y, array_bytes));

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(d_x1, array_x1, array_bytes, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(d_x2, array_x2, array_bytes, hipMemcpyHostToDevice));

    // Launch kernel
    int block_size = 64;
    int grid_size  = (array_size + block_size - 1) / block_size;
    hipLaunchKernelGGL(binary_op_kernel,
                       dim3(grid_size),
                       dim3(block_size),
                       0,
                       0,
                       d_x1,
                       d_x2,
                       d_y,
                       array_size,
                       op_type);

    // Copy result back to host
    CHECK_HIP_ERROR(hipMemcpy(array_y, d_y, array_bytes, hipMemcpyDeviceToHost));

    // Free device memory
    CHECK_HIP_ERROR(hipFree(d_x1));
    CHECK_HIP_ERROR(hipFree(d_x2));
    CHECK_HIP_ERROR(hipFree(d_y));

    return array_y[0];
}

float binaryOpOnHostTest(float x1, float x2, hiptensorOperator_t op_type)
{
    HiptensorBinaryOp op(op_type);
    float             y;
    op(y, x1, x2);
    return y;
}

TEST(UnaryOpTest, HostAndDeviceUnaryOpTest)
{
    float x   = 3.3F;
    float ref = x;
    float y   = unaryOpOnHostTest(x, HIPTENSOR_OP_IDENTITY);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_IDENTITY);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 1.816590212458F; // sqrt(3.3)
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_SQRT);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_SQRT);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = x; // since 3.3 > 0 is true
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_RELU);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_RELU);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = x; // since 3.3 is real number
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_CONJ);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_CONJ);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 1 / x; // y = 1 / 3.3
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_RCP);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_RCP);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 0.964428810727F; // y = 1 / (1 + e^-3.3)
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_SIGMOID);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_SIGMOID);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 0.997283F; // y = (e^3.3 - e^-3.3)/(e^3.3+e^-3.3)
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_TANH);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_TANH);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 27.112638920657F; // y = e^3.3
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_EXP);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_EXP);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 1.19392246847F; // y = ln(3.3)
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_LOG);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_LOG);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 3.3F; // y = abs(3.3)
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_ABS);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_ABS);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = -3.3F; // y = -(3.3)
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_NEG);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_NEG);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = -0.157745694F; // y = sin(3.3)
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_SIN);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_SIN);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = -0.9874797699F; // y = cos(3.3)
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_COS);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_COS);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 0.15974574766F; // y = tan(3.3)
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_TAN);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_TAN);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 13.537878F; // y = (e^3.3 - e^-3.3) / 2
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_SINH);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_SINH);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 13.57476F; // y = (e^3.3 + e^-3.3) / 2
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_COSH);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_COSH);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 0.7F;
    ref = 0.77539749661075F; // y = asin(0.7)
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_ASIN);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_ASIN);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 0.7F;
    ref = 0.79539883018414F; // y = acos(0.7)
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_ACOS);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_ACOS);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 1.2765617616837F; // y = atan(3.3)
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_ATAN);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_ATAN);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 1.9092740140163F;
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_ASINH);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_ASINH);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 1.8632793511534F;
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_ACOSH);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_ACOSH);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 0.7F;
    ref = 0.86730052769405F;
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_ATANH);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_ATANH);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 4.0F;
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_CEIL);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_CEIL);
    EXPECT_FLOAT_EQ(y, ref);

    x   = 3.3F;
    ref = 3.0F;
    y   = unaryOpOnHostTest(x, HIPTENSOR_OP_FLOOR);
    EXPECT_FLOAT_EQ(y, ref);
    y = unaryOpOnDeviceTest(x, HIPTENSOR_OP_FLOOR);
    EXPECT_FLOAT_EQ(y, ref);
}

TEST(BinaryOpTest, HostAndDeviceBinaryOpTest)
{
    float x1  = 3.3F;
    float x2  = 1.2F;
    float ref = 4.5F; // 3.3 + 1.2
    float y   = binaryOpOnHostTest(x1, x2, HIPTENSOR_OP_ADD);
    EXPECT_FLOAT_EQ(y, ref);
    y = binaryOpOnDeviceTest(x1, x2, HIPTENSOR_OP_ADD);
    EXPECT_FLOAT_EQ(y, ref);

    x1  = 3.3F;
    x2  = 1.2F;
    ref = 3.96F; // 3.3 * 1.2
    y   = binaryOpOnHostTest(x1, x2, HIPTENSOR_OP_MUL);
    EXPECT_FLOAT_EQ(y, ref);
    y = binaryOpOnDeviceTest(x1, x2, HIPTENSOR_OP_MUL);
    EXPECT_FLOAT_EQ(y, ref);

    x1  = 3.3F;
    x2  = 1.2F;
    ref = 3.3F; // max(3.3, 1.2)
    y   = binaryOpOnHostTest(x1, x2, HIPTENSOR_OP_MAX);
    EXPECT_FLOAT_EQ(y, ref);
    y = binaryOpOnDeviceTest(x1, x2, HIPTENSOR_OP_MAX);
    EXPECT_FLOAT_EQ(y, ref);

    x1  = 3.3F;
    x2  = 1.2F;
    ref = 1.2F; // min(3.3, 1.2)
    y   = binaryOpOnHostTest(x1, x2, HIPTENSOR_OP_MIN);
    EXPECT_FLOAT_EQ(y, ref);
    y = binaryOpOnDeviceTest(x1, x2, HIPTENSOR_OP_MIN);
    EXPECT_FLOAT_EQ(y, ref);
}
