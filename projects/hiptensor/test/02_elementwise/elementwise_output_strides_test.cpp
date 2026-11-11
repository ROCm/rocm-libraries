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
// hiptensorPermute and the element-wise binary/trinary execute paths must honor
// the output tensor's strides. Previously the output was always written
// packed/contiguously, ignoring the strides set on the output descriptor. These
// tests execute a real GPU op into a non-packed (gapped) output buffer and
// verify the physical device layout matches the requested strides. All three
// public entry points share the same output-stride code path in
// ElementwiseSolutionImpl::initArgs, so each is exercised here.

#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>
#include <hiptensor/internal/hiptensor_utility.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

namespace
{
    // Exact-equality note (applies to every EXPECT_EQ below): all inputs are
    // small integers held in float, alpha/gamma/beta are 1, and the only op is
    // ADD/IDENTITY — every value is exactly representable and the arithmetic is
    // exact, so bit-exact comparison is deliberate and safe. It also lets the
    // sentinel (-1) in the gaps be checked exactly. Do not introduce fractional
    // scalars or lossy ops here without switching to a tolerance-based compare.

    // ---- RAII guards: cleanup runs on every exit path (incl. a failed CHECK) --

    template <typename T>
    struct DeviceBuffer
    {
        T* ptr = nullptr;
        explicit DeviceBuffer(size_t count)
        {
            CHECK_HIP_ERROR(hipMalloc(&ptr, count * sizeof(T)));
        }
        ~DeviceBuffer()
        {
            if(ptr)
            {
                static_cast<void>(hipFree(ptr));
            }
        }
        DeviceBuffer(DeviceBuffer const&)            = delete;
        DeviceBuffer& operator=(DeviceBuffer const&) = delete;
        DeviceBuffer(DeviceBuffer&& other) noexcept
            : ptr(other.ptr)
        {
            other.ptr = nullptr;
        }
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
        {
            if(this != &other)
            {
                if(ptr)
                {
                    static_cast<void>(hipFree(ptr));
                }
                ptr       = other.ptr;
                other.ptr = nullptr;
            }
            return *this;
        }
        T* get() const { return ptr; }
    };

    struct HandleGuard
    {
        hiptensorHandle_t handle = nullptr;
        HandleGuard() { CHECK_HIPTENSOR_ERROR(hiptensorCreate(&handle)); }
        ~HandleGuard()
        {
            if(handle)
            {
                static_cast<void>(hiptensorDestroy(handle));
            }
        }
        HandleGuard(HandleGuard const&)            = delete;
        HandleGuard& operator=(HandleGuard const&) = delete;
    };

    struct TensorDescGuard
    {
        hiptensorTensorDescriptor_t desc = nullptr;
        ~TensorDescGuard()
        {
            if(desc)
            {
                static_cast<void>(hiptensorDestroyTensorDescriptor(desc));
            }
        }
    };

    struct OpDescGuard
    {
        hiptensorOperationDescriptor_t desc = nullptr;
        ~OpDescGuard()
        {
            if(desc)
            {
                static_cast<void>(hiptensorDestroyOperationDescriptor(desc));
            }
        }
    };

    struct PlanPrefGuard
    {
        hiptensorPlanPreference_t pref = nullptr;
        ~PlanPrefGuard()
        {
            if(pref)
            {
                static_cast<void>(hiptensorDestroyPlanPreference(pref));
            }
        }
    };

    struct PlanGuard
    {
        hiptensorPlan_t plan = nullptr;
        ~PlanGuard()
        {
            if(plan)
            {
                static_cast<void>(hiptensorDestroyPlan(plan));
            }
        }
    };

    // ---- generic helpers (rank-N) --------------------------------------------

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

    // Element span (highest addressable index + 1) implied by extents/strides,
    // so a gapped buffer is allocated large enough to hold every written value.
    size_t elementSpan(std::vector<int64_t> const& extents, std::vector<int64_t> const& strides)
    {
        size_t span = 1;
        for(size_t d = 0; d < extents.size(); ++d)
        {
            span += static_cast<size_t>((extents[d] - 1) * strides[d]);
        }
        return span;
    }

    // Maps output axis k (mode modeOut[k]) to the input axis carrying that mode.
    std::vector<int> outAxisToInAxis(std::vector<int32_t> const& modeIn,
                                     std::vector<int32_t> const& modeOut)
    {
        std::vector<int> axis(modeOut.size(), 0);
        for(size_t k = 0; k < modeOut.size(); ++k)
        {
            for(size_t a = 0; a < modeIn.size(); ++a)
            {
                if(modeIn[a] == modeOut[k])
                {
                    axis[k] = static_cast<int>(a);
                    break;
                }
            }
        }
        return axis;
    }

    // CPU reference: scatter each input element into the output using the exact
    // input and output strides, for arbitrary rank. Gaps keep the -1 sentinel.
    std::vector<float> expectedStridedOutput(std::vector<int64_t> const& extentIn,
                                             std::vector<int32_t> const& modeIn,
                                             std::vector<int64_t> const& stridesIn,
                                             std::vector<int32_t> const& modeOut,
                                             std::vector<int64_t> const& stridesOut,
                                             std::vector<float> const&   hostInput,
                                             size_t                      outElemSpan)
    {
        std::vector<float> expected(outElemSpan, -1.0f);
        auto const         outAxis = outAxisToInAxis(modeIn, modeOut);

        size_t total = 1;
        for(auto e : extentIn)
        {
            total *= static_cast<size_t>(e);
        }

        std::vector<int64_t> idx(extentIn.size(), 0); // multi-index in input order
        for(size_t linear = 0; linear < total; ++linear)
        {
            int64_t inOffset = 0, outOffset = 0;
            for(size_t d = 0; d < extentIn.size(); ++d)
            {
                inOffset += idx[d] * stridesIn[d];
            }
            for(size_t k = 0; k < modeOut.size(); ++k)
            {
                outOffset += idx[outAxis[k]] * stridesOut[k];
            }
            expected[outOffset] = hostInput[inOffset];

            // increment the input-order multi-index (rightmost fastest)
            for(int d = static_cast<int>(extentIn.size()) - 1; d >= 0; --d)
            {
                if(++idx[d] < extentIn[d])
                {
                    break;
                }
                idx[d] = 0;
            }
        }
        return expected;
    }

    DeviceBuffer<float> makeInput(std::vector<float> const& host)
    {
        DeviceBuffer<float> d(host.size());
        CHECK_HIP_ERROR(
            hipMemcpy(d.get(), host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice));
        return d;
    }

    // Output buffer pre-filled with a -1 sentinel: any position the kernel leaves
    // untouched (i.e. a gap) stays -1, proving strides were honored, not packed.
    DeviceBuffer<float> makeSentinelOutput(size_t outElemSpan)
    {
        DeviceBuffer<float> d(outElemSpan);
        std::vector<float>  sentinel(outElemSpan, -1.0f);
        CHECK_HIP_ERROR(hipMemcpy(
            d.get(), sentinel.data(), outElemSpan * sizeof(float), hipMemcpyHostToDevice));
        return d;
    }

    std::vector<float> readback(DeviceBuffer<float> const& d, size_t outElemSpan)
    {
        std::vector<float> host(outElemSpan);
        CHECK_HIP_ERROR(hipMemcpy(
            host.data(), d.get(), outElemSpan * sizeof(float), hipMemcpyDeviceToHost));
        return host;
    }

    hiptensorTensorDescriptor_t makeDesc(hiptensorHandle_t           handle,
                                         TensorDescGuard&            guard,
                                         std::vector<int64_t> const& extent,
                                         std::vector<int64_t> const& strides)
    {
        CHECK_HIPTENSOR_ERROR(hiptensorCreateTensorDescriptor(handle,
                                                              &guard.desc,
                                                              extent.size(),
                                                              extent.data(),
                                                              strides.data(),
                                                              HIPTENSOR_R_32F,
                                                              0));
        return guard.desc;
    }

    hiptensorPlan_t makePlan(hiptensorHandle_t              handle,
                             PlanPrefGuard&                 prefGuard,
                             PlanGuard&                     planGuard,
                             hiptensorOperationDescriptor_t descOp)
    {
        CHECK_HIPTENSOR_ERROR(hiptensorCreatePlanPreference(
            handle, &prefGuard.pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE));
        CHECK_HIPTENSOR_ERROR(
            hiptensorCreatePlan(handle, &planGuard.plan, descOp, prefGuard.pref, 0));
        return planGuard.plan;
    }

    // ---- permute: D = alpha * A (identity), strided output -------------------
    std::vector<float> runPermute(std::vector<int64_t> const& extentIn,
                                  std::vector<int32_t> const& modeIn,
                                  std::vector<int64_t> const& stridesIn,
                                  std::vector<int64_t> const& extentOut,
                                  std::vector<int32_t> const& modeOut,
                                  std::vector<int64_t> const& stridesOut,
                                  std::vector<float> const&   hostInput,
                                  size_t                      outElemSpan)
    {
        DeviceBuffer<float> dInput  = makeInput(hostInput);
        DeviceBuffer<float> dOutput = makeSentinelOutput(outElemSpan);

        HandleGuard     h;
        TensorDescGuard descInG, descOutG;
        auto            descIn  = makeDesc(h.handle, descInG, extentIn, stridesIn);
        auto            descOut = makeDesc(h.handle, descOutG, extentOut, stridesOut);

        OpDescGuard opG;
        CHECK_HIPTENSOR_ERROR(hiptensorCreatePermutation(h.handle,
                                                         &opG.desc,
                                                         descIn,
                                                         modeIn.data(),
                                                         HIPTENSOR_OP_IDENTITY,
                                                         descOut,
                                                         modeOut.data(),
                                                         HIPTENSOR_COMPUTE_DESC_32F));

        PlanPrefGuard prefG;
        PlanGuard     planG;
        auto          plan = makePlan(h.handle, prefG, planG, opG.desc);

        float alpha = 1.0f;
        CHECK_HIPTENSOR_ERROR(hiptensorPermute(
            h.handle, plan, &alpha, dInput.get(), dOutput.get(), hipStreamPerThread));
        CHECK_HIP_ERROR(hipStreamSynchronize(hipStreamPerThread));

        return readback(dOutput, outElemSpan);
    }

    // ---- binary: D = (alpha*A) + (gamma*C), strided output D -----------------
    // A packed, C == D layout (strided), so D is written through non-packed
    // strides. With C all-zero and gamma=1, D == A scattered into D's strides.
    std::vector<float> runBinary(std::vector<int64_t> const& extentA,
                                 std::vector<int32_t> const& modeA,
                                 std::vector<int64_t> const& stridesA,
                                 std::vector<int64_t> const& extentD,
                                 std::vector<int32_t> const& modeD,
                                 std::vector<int64_t> const& stridesD,
                                 std::vector<float> const&   hostA,
                                 size_t                      outElemSpan)
    {
        DeviceBuffer<float> dA = makeInput(hostA);
        DeviceBuffer<float> dC = makeSentinelOutput(outElemSpan); // C shares D's layout
        // C must be 0 in the written positions so D == A; overwrite sentinel with 0.
        std::vector<float> zeros(outElemSpan, 0.0f);
        CHECK_HIP_ERROR(hipMemcpy(
            dC.get(), zeros.data(), outElemSpan * sizeof(float), hipMemcpyHostToDevice));
        DeviceBuffer<float> dD = makeSentinelOutput(outElemSpan);

        HandleGuard     h;
        TensorDescGuard descAG, descCG, descDG;
        auto            descA = makeDesc(h.handle, descAG, extentA, stridesA);
        auto            descC = makeDesc(h.handle, descCG, extentD, stridesD);
        auto            descD = makeDesc(h.handle, descDG, extentD, stridesD);

        OpDescGuard opG;
        CHECK_HIPTENSOR_ERROR(hiptensorCreateElementwiseBinary(h.handle,
                                                               &opG.desc,
                                                               descA,
                                                               modeA.data(),
                                                               HIPTENSOR_OP_IDENTITY,
                                                               descC,
                                                               modeD.data(),
                                                               HIPTENSOR_OP_IDENTITY,
                                                               descD,
                                                               modeD.data(),
                                                               HIPTENSOR_OP_ADD,
                                                               HIPTENSOR_COMPUTE_DESC_32F));

        PlanPrefGuard prefG;
        PlanGuard     planG;
        auto          plan = makePlan(h.handle, prefG, planG, opG.desc);

        float alpha = 1.0f, gamma = 1.0f;
        CHECK_HIPTENSOR_ERROR(hiptensorElementwiseBinaryExecute(
            h.handle, plan, &alpha, dA.get(), &gamma, dC.get(), dD.get(), hipStreamPerThread));
        CHECK_HIP_ERROR(hipStreamSynchronize(hipStreamPerThread));

        return readback(dD, outElemSpan);
    }

    // ---- trinary: D = ((alpha*A)+(beta*B))+(gamma*C), strided output D -------
    // B and C all-zero (beta=gamma=1), so D == A scattered into D's strides.
    std::vector<float> runTrinary(std::vector<int64_t> const& extentA,
                                  std::vector<int32_t> const& modeA,
                                  std::vector<int64_t> const& stridesA,
                                  std::vector<int64_t> const& extentD,
                                  std::vector<int32_t> const& modeD,
                                  std::vector<int64_t> const& stridesD,
                                  std::vector<float> const&   hostA,
                                  size_t                      outElemSpan)
    {
        DeviceBuffer<float> dA = makeInput(hostA);
        DeviceBuffer<float> dB = makeSentinelOutput(outElemSpan);
        DeviceBuffer<float> dC = makeSentinelOutput(outElemSpan);
        std::vector<float>  zeros(outElemSpan, 0.0f);
        CHECK_HIP_ERROR(hipMemcpy(
            dB.get(), zeros.data(), outElemSpan * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(
            dC.get(), zeros.data(), outElemSpan * sizeof(float), hipMemcpyHostToDevice));
        DeviceBuffer<float> dD = makeSentinelOutput(outElemSpan);

        HandleGuard     h;
        TensorDescGuard descAG, descBG, descCG, descDG;
        auto            descA = makeDesc(h.handle, descAG, extentA, stridesA);
        auto            descB = makeDesc(h.handle, descBG, extentD, stridesD);
        auto            descC = makeDesc(h.handle, descCG, extentD, stridesD);
        auto            descD = makeDesc(h.handle, descDG, extentD, stridesD);

        OpDescGuard opG;
        CHECK_HIPTENSOR_ERROR(hiptensorCreateElementwiseTrinary(h.handle,
                                                                &opG.desc,
                                                                descA,
                                                                modeA.data(),
                                                                HIPTENSOR_OP_IDENTITY,
                                                                descB,
                                                                modeD.data(),
                                                                HIPTENSOR_OP_IDENTITY,
                                                                descC,
                                                                modeD.data(),
                                                                HIPTENSOR_OP_IDENTITY,
                                                                descD,
                                                                modeD.data(),
                                                                HIPTENSOR_OP_ADD,
                                                                HIPTENSOR_OP_ADD,
                                                                HIPTENSOR_COMPUTE_DESC_32F));

        PlanPrefGuard prefG;
        PlanGuard     planG;
        auto          plan = makePlan(h.handle, prefG, planG, opG.desc);

        float alpha = 1.0f, beta = 1.0f, gamma = 1.0f;
        CHECK_HIPTENSOR_ERROR(hiptensorElementwiseTrinaryExecute(h.handle,
                                                                 plan,
                                                                 &alpha,
                                                                 dA.get(),
                                                                 &beta,
                                                                 dB.get(),
                                                                 &gamma,
                                                                 dC.get(),
                                                                 dD.get(),
                                                                 hipStreamPerThread));
        CHECK_HIP_ERROR(hipStreamSynchronize(hipStreamPerThread));

        return readback(dD, outElemSpan);
    }
}

// 2x2 transpose into a column-major buffer with a 2-element gap after every
// value (the exact scenario from issue #9543's reproducer).
TEST(PermutationOutputStridesTest, Rank2GappedColumnMajor)
{
    constexpr int64_t nskip = 2;

    std::vector<int64_t> extentIn{2, 2};
    std::vector<int32_t> modeIn{0, 1};
    std::vector<int64_t> stridesIn{1, 2};

    std::vector<int32_t> modeOut{1, 0};
    std::vector<int64_t> extentOut{extentIn[modeOut[0]], extentIn[modeOut[1]]};
    std::vector<int64_t> stridesOut{1 * (1 + nskip), 2 * (1 + nskip)};

    std::vector<float> hostInput{1.0f, 2.0f, 3.0f, 4.0f};
    size_t             outElemSpan = elementSpan(extentOut, stridesOut);

    auto actual   = runPermute(
        extentIn, modeIn, stridesIn, extentOut, modeOut, stridesOut, hostInput, outElemSpan);
    auto expected = expectedStridedOutput(
        extentIn, modeIn, stridesIn, modeOut, stridesOut, hostInput, outElemSpan);

    EXPECT_EQ(actual, expected)
        << "hiptensorPermute did not honor the output tensor strides (issue #9543).";
}

// Rank-3 permute into a gapped buffer, to cover more than the rank-2 corner.
TEST(PermutationOutputStridesTest, Rank3GappedColumnMajor)
{
    std::vector<int64_t> extentIn{2, 3, 4};
    std::vector<int32_t> modeIn{0, 1, 2};
    std::vector<int64_t> stridesIn = packedColMajorStrides(extentIn);

    std::vector<int32_t> modeOut{2, 0, 1};
    std::vector<int64_t> extentOut{extentIn[2], extentIn[0], extentIn[1]};
    // packed col-major for extentOut, then double every stride to introduce gaps.
    std::vector<int64_t> stridesOut = packedColMajorStrides(extentOut);
    for(auto& s : stridesOut)
    {
        s *= 2;
    }

    std::vector<float> hostInput(2 * 3 * 4);
    for(size_t i = 0; i < hostInput.size(); ++i)
    {
        hostInput[i] = static_cast<float>(i + 1);
    }
    size_t outElemSpan = elementSpan(extentOut, stridesOut);

    auto actual   = runPermute(
        extentIn, modeIn, stridesIn, extentOut, modeOut, stridesOut, hostInput, outElemSpan);
    auto expected = expectedStridedOutput(
        extentIn, modeIn, stridesIn, modeOut, stridesOut, hostInput, outElemSpan);

    EXPECT_EQ(actual, expected)
        << "hiptensorPermute did not honor rank-3 output strides (issue #9543).";
}

// Same transpose with packed output strides must remain correct (guards against
// a regression in the common path from the fix).
TEST(PermutationOutputStridesTest, Rank2PackedColumnMajor)
{
    std::vector<int64_t> extentIn{2, 2};
    std::vector<int32_t> modeIn{0, 1};
    std::vector<int64_t> stridesIn{1, 2};

    std::vector<int32_t> modeOut{1, 0};
    std::vector<int64_t> extentOut{extentIn[modeOut[0]], extentIn[modeOut[1]]};
    std::vector<int64_t> stridesOut = packedColMajorStrides(extentOut);

    std::vector<float> hostInput{1.0f, 2.0f, 3.0f, 4.0f};
    size_t             outElemSpan = elementSpan(extentOut, stridesOut);

    auto actual   = runPermute(
        extentIn, modeIn, stridesIn, extentOut, modeOut, stridesOut, hostInput, outElemSpan);
    auto expected = expectedStridedOutput(
        extentIn, modeIn, stridesIn, modeOut, stridesOut, hostInput, outElemSpan);

    EXPECT_EQ(actual, expected);
}

// Binary execute path: D = A + C with C == 0, D written through gapped strides.
TEST(ElementwiseBinaryOutputStridesTest, Rank2GappedColumnMajor)
{
    constexpr int64_t nskip = 2;

    std::vector<int64_t> extentA{2, 2};
    std::vector<int32_t> modeA{0, 1};
    std::vector<int64_t> stridesA = packedColMajorStrides(extentA);

    std::vector<int32_t> modeD{1, 0};
    std::vector<int64_t> extentD{extentA[modeD[0]], extentA[modeD[1]]};
    std::vector<int64_t> stridesD{1 * (1 + nskip), 2 * (1 + nskip)};

    std::vector<float> hostA{1.0f, 2.0f, 3.0f, 4.0f};
    size_t             outElemSpan = elementSpan(extentD, stridesD);

    auto actual = runBinary(
        extentA, modeA, stridesA, extentD, modeD, stridesD, hostA, outElemSpan);
    auto expected = expectedStridedOutput(
        extentA, modeA, stridesA, modeD, stridesD, hostA, outElemSpan);

    EXPECT_EQ(actual, expected)
        << "hiptensorElementwiseBinaryExecute did not honor the output strides (issue #9543).";
}

// Trinary execute path: D = A + B + C with B == C == 0, D written through gapped strides.
TEST(ElementwiseTrinaryOutputStridesTest, Rank2GappedColumnMajor)
{
    constexpr int64_t nskip = 2;

    std::vector<int64_t> extentA{2, 2};
    std::vector<int32_t> modeA{0, 1};
    std::vector<int64_t> stridesA = packedColMajorStrides(extentA);

    std::vector<int32_t> modeD{1, 0};
    std::vector<int64_t> extentD{extentA[modeD[0]], extentA[modeD[1]]};
    std::vector<int64_t> stridesD{1 * (1 + nskip), 2 * (1 + nskip)};

    std::vector<float> hostA{1.0f, 2.0f, 3.0f, 4.0f};
    size_t             outElemSpan = elementSpan(extentD, stridesD);

    auto actual = runTrinary(
        extentA, modeA, stridesA, extentD, modeD, stridesD, hostA, outElemSpan);
    auto expected = expectedStridedOutput(
        extentA, modeA, stridesA, modeD, stridesD, hostA, outElemSpan);

    EXPECT_EQ(actual, expected)
        << "hiptensorElementwiseTrinaryExecute did not honor the output strides (issue #9543).";
}
