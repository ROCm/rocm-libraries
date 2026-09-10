/*******************************************************************************
 *
 * MIT License
 *
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
 *******************************************************************************/

// Regression test for https://github.com/ROCm/rocm-libraries/issues/9543:
// hiptensorPermute must honor the output tensor's strides. Previously the output
// was always written packed/contiguously, ignoring the strides set on the output
// descriptor. These tests execute a real GPU permute into a non-packed (gapped)
// output buffer and verify the physical device layout matches the requested
// strides.

#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>
#include <hiptensor/internal/hiptensor_utility.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace
{
    // Column-major packed strides for the given extents (stride[0] == 1).
    std::vector<int64_t> packedColMajorStrides(std::vector<int64_t> const& extents)
    {
        std::vector<int64_t> strides(extents.size(), 1);
        for(size_t i = 1; i < extents.size(); ++i)
        {
            strides[i] = strides[i - 1] * extents[i - 1];
        }
        return strides;
    }

    // Runs a permute with the supplied (possibly non-packed) output strides and
    // returns the full physical output buffer read back from the device. The
    // output buffer is sized to the span implied by strides/extents so that gaps
    // are preserved, and pre-filled with a sentinel to prove the kernel wrote to
    // the strided positions rather than packing.
    std::vector<float> runPermute(std::vector<int64_t> const& extentIn,
                                  std::vector<int32_t> const&  modeIn,
                                  std::vector<int64_t> const& stridesIn,
                                  std::vector<int64_t> const& extentOut,
                                  std::vector<int32_t> const&  modeOut,
                                  std::vector<int64_t> const& stridesOut,
                                  std::vector<float> const&   hostInput,
                                  size_t                      outElemSpan)
    {
        float* dInput  = nullptr;
        float* dOutput = nullptr;
        CHECK_HIP_ERROR(hipMalloc(&dInput, hostInput.size() * sizeof(float)));
        CHECK_HIP_ERROR(hipMalloc(&dOutput, outElemSpan * sizeof(float)));

        CHECK_HIP_ERROR(hipMemcpy(dInput,
                                  hostInput.data(),
                                  hostInput.size() * sizeof(float),
                                  hipMemcpyHostToDevice));

        // Sentinel fill: any gap left untouched stays -1 so we can assert the
        // kernel honored the strides instead of writing packed.
        std::vector<float> sentinel(outElemSpan, -1.0f);
        CHECK_HIP_ERROR(hipMemcpy(dOutput,
                                  sentinel.data(),
                                  outElemSpan * sizeof(float),
                                  hipMemcpyHostToDevice));

        hiptensorHandle_t handle;
        CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle));

        hiptensorTensorDescriptor_t descIn = nullptr;
        CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                              &descIn,
                                                              extentIn.size(),
                                                              extentIn.data(),
                                                              stridesIn.data(),
                                                              HIPTENSOR_R_32F,
                                                              0));

        hiptensorTensorDescriptor_t descOut = nullptr;
        CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                              &descOut,
                                                              extentOut.size(),
                                                              extentOut.data(),
                                                              stridesOut.data(),
                                                              HIPTENSOR_R_32F,
                                                              0));

        hiptensorOperationDescriptor_t descOp;
        CHECK_HIPTENSOR_ERROR(hiptensorCreatePermutation(handle,
                                                         &descOp,
                                                         descIn,
                                                         modeIn.data(),
                                                         HIPTENSOR_OP_IDENTITY,
                                                         descOut,
                                                         modeOut.data(),
                                                         HIPTENSOR_COMPUTE_DESC_32F));

        hiptensorPlanPreference_t planPref;
        CHECK_HIPTENSOR_ERROR(hiptensorCreatePlanPreference(
            handle, &planPref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));

        hiptensorPlan_t plan;
        CHECK_HIPTENSOR_ERROR(hiptensorCreatePlan(handle, &plan, descOp, planPref, 0));

        float alpha = 1.0f;
        CHECK_HIPTENSOR_ERROR(
            hiptensorPermute(handle, plan, &alpha, dInput, dOutput, hipStreamPerThread));
        CHECK_HIP_ERROR(hipStreamSynchronize(hipStreamPerThread));

        std::vector<float> hostOutput(outElemSpan);
        CHECK_HIP_ERROR(hipMemcpy(hostOutput.data(),
                                  dOutput,
                                  outElemSpan * sizeof(float),
                                  hipMemcpyDeviceToHost));

        CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlan(plan));
        CHECK_HIPTENSOR_ERROR(hiptensorDestroyPlanPreference(planPref));
        CHECK_HIPTENSOR_ERROR(hiptensorDestroyOperationDescriptor(descOp));
        CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(descIn));
        CHECK_HIPTENSOR_ERROR(hiptensorDestroyTensorDescriptor(descOut));
        CHECK_HIPTENSOR_ERROR(hiptensorDestroy(handle));

        CHECK_HIP_ERROR(hipFree(dInput));
        CHECK_HIP_ERROR(hipFree(dOutput));

        return hostOutput;
    }

    // CPU reference: scatter each input element into the output using the exact
    // output strides, matching the semantics the issue expects.
    std::vector<float> expectedStridedOutput(std::vector<int64_t> const& extentIn,
                                             std::vector<int32_t> const&  modeIn,
                                             std::vector<int64_t> const& stridesIn,
                                             std::vector<int32_t> const&  modeOut,
                                             std::vector<int64_t> const& stridesOut,
                                             std::vector<float> const&   hostInput,
                                             size_t                      outElemSpan)
    {
        std::vector<float> expected(outElemSpan, -1.0f);
        // Only rank-2 is exercised here.
        for(int64_t i = 0; i < extentIn[0]; ++i)
        {
            for(int64_t j = 0; j < extentIn[1]; ++j)
            {
                int64_t idx[2]      = {i, j};
                int64_t inOffset    = i * stridesIn[0] + j * stridesIn[1];
                // Output axis k iterates modeOut[k]; find which input axis carries
                // that mode to pick the right index.
                int64_t outOffset = 0;
                for(size_t k = 0; k < modeOut.size(); ++k)
                {
                    int axis = (modeOut[k] == modeIn[0]) ? 0 : 1;
                    outOffset += idx[axis] * stridesOut[k];
                }
                expected[outOffset] = hostInput[inOffset];
            }
        }
        return expected;
    }
}

// 2x2 transpose into a column-major buffer with a 2-element gap after every
// value (the exact scenario from issue #9543's reproducer).
TEST(PermutationOutputStridesTest, Rank2GappedColumnMajor)
{
    constexpr int64_t nskip = 2;

    std::vector<int64_t> extentIn{2, 2};
    std::vector<int32_t>  modeIn{0, 1};
    std::vector<int64_t> stridesIn{1, 2};

    std::vector<int32_t>  modeOut{1, 0};
    std::vector<int64_t> extentOut{extentIn[modeOut[0]], extentIn[modeOut[1]]};
    std::vector<int64_t> stridesOut{1 * (1 + nskip), 2 * (1 + nskip)};

    std::vector<float> hostInput{1.0f, 2.0f, 3.0f, 4.0f};
    size_t             outElemSpan = extentIn[0] * extentIn[1] * (1 + nskip);

    auto actual = runPermute(extentIn,
                             modeIn,
                             stridesIn,
                             extentOut,
                             modeOut,
                             stridesOut,
                             hostInput,
                             outElemSpan);
    auto expected = expectedStridedOutput(
        extentIn, modeIn, stridesIn, modeOut, stridesOut, hostInput, outElemSpan);

    EXPECT_EQ(actual, expected)
        << "hiptensorPermute did not honor the output tensor strides (issue #9543).";
}

// Same transpose with packed output strides must remain correct (guards against
// a regression in the common path from the fix).
TEST(PermutationOutputStridesTest, Rank2PackedColumnMajor)
{
    std::vector<int64_t> extentIn{2, 2};
    std::vector<int32_t>  modeIn{0, 1};
    std::vector<int64_t> stridesIn{1, 2};

    std::vector<int32_t>  modeOut{1, 0};
    std::vector<int64_t> extentOut{extentIn[modeOut[0]], extentIn[modeOut[1]]};
    std::vector<int64_t> stridesOut = packedColMajorStrides(extentOut);

    std::vector<float> hostInput{1.0f, 2.0f, 3.0f, 4.0f};
    size_t             outElemSpan = extentIn[0] * extentIn[1];

    auto actual = runPermute(extentIn,
                             modeIn,
                             stridesIn,
                             extentOut,
                             modeOut,
                             stridesOut,
                             hostInput,
                             outElemSpan);
    auto expected = expectedStridedOutput(
        extentIn, modeIn, stridesIn, modeOut, stridesOut, hostInput, outElemSpan);

    EXPECT_EQ(actual, expected);
}
