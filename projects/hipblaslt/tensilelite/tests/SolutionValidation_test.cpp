/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
 *******************************************************************************
 * Unit tests for TensileLite::validateSolutionParamsForSelection (no file I/O).
 * Library YAML walking and calling this API is done by hipBLASLt test.
 ******************************************************************************/

#include <gtest/gtest.h>
#include <Tensile/SolutionValidation.hpp>

using namespace TensileLite;

static SolutionParamsForValidation makeParams(int cuCount,
                                             int workGroupMappingXCC,
                                             int wg0,
                                             int wg1,
                                             int wg2,
                                             int wavefrontSize)
{
    SolutionParamsForValidation p;
    p.cuCount              = cuCount;
    p.workGroupMappingXCC  = workGroupMappingXCC;
    p.workGroup            = {wg0, wg1, wg2};
    p.wavefrontSize        = wavefrontSize;
    return p;
}

TEST(SolutionValidation, WorkGroupMappingXCC_38CU_4_Fails)
{
    auto p = makeParams(38, 4, 8, 8, 4, 64);
    std::string reason;
    EXPECT_FALSE(validateSolutionParamsForSelection(p, &reason));
    EXPECT_FALSE(reason.empty());
}

TEST(SolutionValidation, WorkGroupMappingXCC_38CU_1_Passes)
{
    auto p = makeParams(38, 1, 8, 8, 4, 64);
    EXPECT_TRUE(validateSolutionParamsForSelection(p, nullptr));
}

TEST(SolutionValidation, WorkGroupMappingXCC_Minus1_AlwaysPasses)
{
    auto p = makeParams(38, -1, 8, 8, 4, 64);
    EXPECT_TRUE(validateSolutionParamsForSelection(p, nullptr));
}

TEST(SolutionValidation, WorkGroupMappingXCC_80CU_4_Passes)
{
    auto p = makeParams(80, 4, 8, 8, 4, 64);
    EXPECT_TRUE(validateSolutionParamsForSelection(p, nullptr));
}

TEST(SolutionValidation, WorkGroup_ProductTooSmall_Fails)
{
    auto p = makeParams(80, 1, 2, 2, 2, 64); // 8 < 32
    std::string reason;
    EXPECT_FALSE(validateSolutionParamsForSelection(p, &reason));
}

TEST(SolutionValidation, WorkGroup_ProductTooLarge_Fails)
{
    auto p = makeParams(80, 1, 32, 32, 2, 64); // 2048 > 1024
    std::string reason;
    EXPECT_FALSE(validateSolutionParamsForSelection(p, &reason));
}

TEST(SolutionValidation, WorkGroup_Valid_Passes)
{
    auto p = makeParams(80, 1, 8, 8, 4, 64); // 256 in [32, 1024]
    EXPECT_TRUE(validateSolutionParamsForSelection(p, nullptr));
}

TEST(SolutionValidation, WavefrontSize_Invalid_Fails)
{
    auto p = makeParams(80, 1, 8, 8, 4, 16);
    std::string reason;
    EXPECT_FALSE(validateSolutionParamsForSelection(p, &reason));
}

TEST(SolutionValidation, WavefrontSize_32_Passes)
{
    auto p = makeParams(80, 1, 8, 8, 4, 32);
    EXPECT_TRUE(validateSolutionParamsForSelection(p, nullptr));
}

TEST(SolutionValidation, WavefrontSize_64_Passes)
{
    auto p = makeParams(80, 1, 8, 8, 4, 64);
    EXPECT_TRUE(validateSolutionParamsForSelection(p, nullptr));
}
