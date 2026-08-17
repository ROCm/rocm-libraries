// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "HipblasltMatmulDesc.hpp"
#include <gtest/gtest.h>
#include <hipblaslt/hipblaslt.h>

using namespace hipblaslt_plugin;

TEST(TestHipblasltMatmulDesc, CanCreateAndDestroy)
{
    EXPECT_NO_THROW({
        HipblasltMatmulDesc const desc(HIPBLAS_OP_N, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F);
        EXPECT_NE(desc.matmulDesc(), nullptr);
    });
}

namespace
{

template <typename T>
T getDescAttribute(const HipblasltMatmulDesc& desc, hipblasLtMatmulDescAttributes_t attr)
{
    T value{};
    size_t sizeWritten = 0;
    EXPECT_EQ(hipblasLtMatmulDescGetAttribute(
                  desc.matmulDesc(), attr, static_cast<void*>(&value), sizeof(value), &sizeWritten),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(sizeWritten, sizeof(value));
    return value;
}

} // anonymous namespace

TEST(TestHipblasltMatmulDesc, SetAScaleMode)
{
    HipblasltMatmulDesc desc(HIPBLAS_OP_N, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    desc.setAScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
    EXPECT_EQ(
        getDescAttribute<hipblasLtMatmulMatrixScale_t>(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE),
        HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
}

TEST(TestHipblasltMatmulDesc, SetBScaleMode)
{
    HipblasltMatmulDesc desc(HIPBLAS_OP_N, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    desc.setBScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
    EXPECT_EQ(
        getDescAttribute<hipblasLtMatmulMatrixScale_t>(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE),
        HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
}

TEST(TestHipblasltMatmulDesc, SetAScalePointer)
{
    HipblasltMatmulDesc desc(HIPBLAS_OP_N, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    const void* ptr = reinterpret_cast<const void*>(0x1234);
    desc.setAScalePointer(ptr);
    EXPECT_EQ(getDescAttribute<const void*>(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER), ptr);
}

TEST(TestHipblasltMatmulDesc, SetBScalePointer)
{
    HipblasltMatmulDesc desc(HIPBLAS_OP_N, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    const void* ptr = reinterpret_cast<const void*>(0x5678);
    desc.setBScalePointer(ptr);
    EXPECT_EQ(getDescAttribute<const void*>(desc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER), ptr);
}

// clone() reproduces the transposes and scale modes onto a new, distinct
// descriptor. Per-call scale pointers are deliberately not copied.
TEST(TestHipblasltMatmulDesc, CloneReproducesConfiguration)
{
    HipblasltMatmulDesc desc(HIPBLAS_OP_T, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    desc.setAScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
    desc.setBScaleMode(HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);

    HipblasltMatmulDesc const copy = desc.clone();

    // Independent descriptor with its own underlying handle.
    EXPECT_NE(copy.matmulDesc(), nullptr);
    EXPECT_NE(copy.matmulDesc(), desc.matmulDesc());

    EXPECT_EQ(getDescAttribute<hipblasOperation_t>(copy, HIPBLASLT_MATMUL_DESC_TRANSA),
              HIPBLAS_OP_T);
    EXPECT_EQ(getDescAttribute<hipblasOperation_t>(copy, HIPBLASLT_MATMUL_DESC_TRANSB),
              HIPBLAS_OP_N);
    EXPECT_EQ(
        getDescAttribute<hipblasLtMatmulMatrixScale_t>(copy, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE),
        HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
    EXPECT_EQ(
        getDescAttribute<hipblasLtMatmulMatrixScale_t>(copy, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE),
        HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0);
}

// Cloning an empty (default-constructed) descriptor must return another empty
// descriptor via the _desc==nullptr guard, without reading its unset mirror
// fields. Without the guard, clone() would build a non-null descriptor here.
TEST(TestHipblasltMatmulDesc, CloneOfEmptyDescriptorIsEmpty)
{
    EXPECT_EQ(HipblasltMatmulDesc{}.clone().matmulDesc(), nullptr);
}

// The clone is independent: mutating it must not affect the original. This is the
// property that lets execute() clone per call instead of mutating a shared desc.
TEST(TestHipblasltMatmulDesc, CloneIsIndependentOfOriginal)
{
    HipblasltMatmulDesc desc(HIPBLAS_OP_N, HIPBLAS_OP_N, HIPBLAS_COMPUTE_32F, HIP_R_32F);
    const void* origPtr = reinterpret_cast<const void*>(0x1111);
    desc.setAScalePointer(origPtr);

    HipblasltMatmulDesc copy = desc.clone();
    const void* newPtr = reinterpret_cast<const void*>(0x2222);
    copy.setAScalePointer(newPtr);

    EXPECT_EQ(getDescAttribute<const void*>(desc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER), origPtr);
    EXPECT_EQ(getDescAttribute<const void*>(copy, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER), newPtr);
}
