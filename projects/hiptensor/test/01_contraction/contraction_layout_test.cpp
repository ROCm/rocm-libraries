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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *******************************************************************************/

// End-to-end checks that binary (D = alpha * A * B + beta * C) and trinary
// contractions honour every tensor's own modes and strides: permuted mode orders
// (patterns P1-P5), packed column-major, packed row-major and padded layouts,
// operands with mixed layouts, an additive input laid out differently from the
// output, and padded strides under HIPTENSOR_ALGO_DEFAULT. The 01_contraction
// suite compares the kernel against hipTensor's CPU reference, which shares the
// library's mode normalization and stride handling, so a fault in that code passes
// there unnoticed. The expected values here come from a plain loop over every index
// with 64-bit offsets.
//
// All inputs are small integers and alpha/beta are integers, so every product and
// partial sum is exact in f32 and the comparison is bit-exact. Output positions
// that no index maps to (the gaps of a padded layout) must keep their sentinel.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "hiptensor_options.hpp"

namespace
{
    enum class Layout
    {
        ColMajor, // packed, first mode fastest
        RowMajor, // packed, last mode fastest
        Padded, // column-major with one unused element after every mode
        WidePadded, // column-major with a large gap before the last mode
        Default // strides = nullptr, the library's default packed layout
    };

    enum class Pattern
    {
        Canonical, // A[M,K]  B[N,K]   D[M,N]
        P1, // A[M,K] and B[N,K] interleaved, D[M,N] interleaved
        P2, // A and B interleaved, D canonical
        P3, // A and B canonical, D interleaved
        P4, // B's contracted modes reversed
        P5, // A interleaved, B and D canonical
    };

    constexpr float    sentinel   = -1000.0f;
    constexpr int64_t  wideStride = int64_t(1) << 20;
    constexpr uint32_t alignment  = 4;
    constexpr float    alphaValue = 2.0f;
    constexpr float    betaValue  = -1.0f;

    // Extents differ within each mode group, so no two modes of a group are interchangeable.
    constexpr int32_t mModes[] = {'a', 'b', 'c'};
    constexpr int32_t nModes[] = {'p', 'q', 'r'};
    constexpr int32_t kModes[] = {'x', 'y', 'z'};
    // Output modes carried by the third operand of a trinary contraction.
    constexpr int32_t sModes[] = {'s', 't'};

    int64_t extentOf(int32_t mode)
    {
        switch(mode)
        {
        case 'a':
            return 5;
        case 'b':
            return 3;
        case 'c':
            return 2;
        case 'p':
            return 4;
        case 'q':
            return 6;
        case 'r':
            return 3;
        case 'x':
            return 3;
        case 'y':
            return 4;
        case 'z':
            return 2;
        case 's':
            return 3;
        case 't':
            return 5;
        default:
            return 1;
        }
    }

    char const* toString(Layout layout)
    {
        switch(layout)
        {
        case Layout::ColMajor:
            return "Col";
        case Layout::RowMajor:
            return "Row";
        case Layout::Padded:
            return "Pad";
        case Layout::WidePadded:
            return "WidePad";
        case Layout::Default:
            return "Default";
        }
        return "";
    }

    char const* toString(Pattern pattern)
    {
        switch(pattern)
        {
        case Pattern::Canonical:
            return "Canonical";
        case Pattern::P1:
            return "P1";
        case Pattern::P2:
            return "P2";
        case Pattern::P3:
            return "P3";
        case Pattern::P4:
            return "P4";
        case Pattern::P5:
            return "P5";
        }
        return "";
    }

    struct LayoutCase
    {
        Pattern pattern;
        int     rank; // modes per group: |M| = |N| = |K| = rank
        Layout  layoutA;
        Layout  layoutB;
        Layout  layoutC;
        Layout  layoutD;
        bool    reverseModesC; // C carries D's modes in reverse order
    };

    std::string caseName(LayoutCase const& c)
    {
        std::string name = std::string(toString(c.pattern)) + "_Rank" + std::to_string(c.rank)
                           + "_A" + toString(c.layoutA) + "_B" + toString(c.layoutB) + "_C"
                           + toString(c.layoutC) + (c.reverseModesC ? "Reversed" : "") + "_D"
                           + toString(c.layoutD);
        return name;
    }

    struct PatternModes
    {
        std::vector<int32_t> a;
        std::vector<int32_t> b;
        std::vector<int32_t> d;
    };

    PatternModes patternModes(Pattern pattern, int rank)
    {
        std::vector<int32_t> m(mModes, mModes + rank);
        std::vector<int32_t> n(nModes, nModes + rank);
        std::vector<int32_t> k(kModes, kModes + rank);

        auto concat = [](std::vector<int32_t> x, std::vector<int32_t> const& y) {
            x.insert(x.end(), y.cbegin(), y.cend());
            return x;
        };
        auto interleave = [](std::vector<int32_t> const& x, std::vector<int32_t> const& y) {
            std::vector<int32_t> r;
            for(std::size_t i = 0; i < x.size(); i++)
            {
                r.push_back(x[i]);
                r.push_back(y[i]);
            }
            return r;
        };
        std::vector<int32_t> kReversed(k.crbegin(), k.crend());

        switch(pattern)
        {
        case Pattern::P1:
            return {interleave(m, k), interleave(n, k), interleave(m, n)};
        case Pattern::P2:
            return {interleave(m, k), interleave(n, k), concat(m, n)};
        case Pattern::P3:
            return {concat(m, k), concat(n, k), interleave(m, n)};
        case Pattern::P4:
            return {concat(m, k), concat(n, kReversed), concat(m, n)};
        case Pattern::P5:
            return {interleave(m, k), concat(n, k), concat(m, n)};
        case Pattern::Canonical:
        default:
            return {concat(m, k), concat(n, k), concat(m, n)};
        }
    }

    std::vector<int64_t> stridesFor(Layout layout, std::vector<int64_t> const& extents)
    {
        auto const           rank = extents.size();
        std::vector<int64_t> strides(rank, 1);

        bool colMajor = (layout != Layout::RowMajor);
        if(layout == Layout::Default)
        {
            colMajor = hiptensor::HiptensorOptions::instance()->isColMajorStrides();
        }

        if(colMajor)
        {
            int64_t const pad = (layout == Layout::Padded) ? 1 : 0;
            for(std::size_t i = 1; i < rank; i++)
            {
                strides[i] = strides[i - 1] * (extents[i - 1] + pad);
            }
            if(layout == Layout::WidePadded && rank > 1)
            {
                strides[rank - 1] = wideStride;
            }
        }
        else
        {
            for(std::size_t i = rank - 1; i > 0; i--)
            {
                strides[i - 1] = strides[i] * extents[i];
            }
        }
        return strides;
    }

    struct HostTensor
    {
        std::vector<int32_t> modes;
        std::vector<int64_t> extents;
        std::vector<int64_t> strides;
        Layout               layout;
        std::vector<float>   data; // one past the largest offset, gaps included
    };

    HostTensor makeTensor(std::vector<int32_t> const& modes, Layout layout)
    {
        HostTensor t;
        t.modes  = modes;
        t.layout = layout;
        for(auto mode : modes)
        {
            t.extents.push_back(extentOf(mode));
        }
        t.strides = stridesFor(layout, t.extents);

        int64_t span = 1;
        for(std::size_t i = 0; i < modes.size(); i++)
        {
            span += (t.extents[i] - 1) * t.strides[i];
        }
        t.data.assign(static_cast<std::size_t>(span), sentinel);
        return t;
    }

    // Fills every element, gaps included, so a read from a gap changes the result.
    void fillRandomIntegers(HostTensor& t, int lo, int hi, unsigned seed)
    {
        std::mt19937                       gen(seed);
        std::uniform_int_distribution<int> dist(lo, hi);
        for(auto& v : t.data)
        {
            v = static_cast<float>(dist(gen));
        }
    }

    // Advances idx[begin, end) like an odometer, first position fastest.
    // Returns false once every combination has been visited.
    bool nextIndex(std::vector<int64_t>&       idx,
                   std::vector<int64_t> const& extents,
                   std::size_t                 begin,
                   std::size_t                 end)
    {
        for(std::size_t i = begin; i < end; i++)
        {
            if(++idx[i] < extents[i])
            {
                return true;
            }
            idx[i] = 0;
        }
        return false;
    }

    // Independent reference: out = alpha * sum(product of factors) + beta * addend, summed
    // over every mode the factors carry and the output does not. Every offset is
    // sum(index * stride) over the tensor's own modes, in 64-bit arithmetic.
    std::vector<float> reference(std::vector<HostTensor const*> const& factors,
                                 HostTensor const&                     addend,
                                 HostTensor const&                     out)
    {
        std::vector<int32_t> all = out.modes;
        for(auto const* f : factors)
        {
            for(auto mode : f->modes)
            {
                if(std::find(all.cbegin(), all.cend(), mode) == all.cend())
                {
                    all.push_back(mode);
                }
            }
        }
        auto const nOut = out.modes.size();

        std::vector<int64_t> extents;
        for(auto mode : all)
        {
            extents.push_back(extentOf(mode));
        }

        // Stride of each mode in `all` for one tensor, 0 where the tensor lacks the mode.
        auto strideTable = [&all](HostTensor const& t) {
            std::vector<int64_t> s(all.size(), 0);
            for(std::size_t g = 0; g < all.size(); g++)
            {
                for(std::size_t i = 0; i < t.modes.size(); i++)
                {
                    if(t.modes[i] == all[g])
                    {
                        s[g] = t.strides[i];
                    }
                }
            }
            return s;
        };
        std::vector<std::vector<int64_t>> factorStrides;
        for(auto const* f : factors)
        {
            factorStrides.push_back(strideTable(*f));
        }
        auto const addendStrides = strideTable(addend);
        auto const outStrides    = strideTable(out);

        auto offset = [](std::vector<int64_t> const& idx, std::vector<int64_t> const& s) {
            int64_t off = 0;
            for(std::size_t g = 0; g < idx.size(); g++)
            {
                off += idx[g] * s[g];
            }
            return static_cast<std::size_t>(off);
        };

        std::vector<float>   expected(out.data.size(), sentinel);
        std::vector<int64_t> idx(all.size(), 0);
        do
        {
            double acc = 0.0;
            do
            {
                double product = 1.0;
                for(std::size_t f = 0; f < factors.size(); f++)
                {
                    product *= static_cast<double>(factors[f]->data[offset(idx, factorStrides[f])]);
                }
                acc += product;
            } while(nextIndex(idx, extents, nOut, all.size()));

            expected[offset(idx, outStrides)] = static_cast<float>(
                alphaValue * acc
                + betaValue * static_cast<double>(addend.data[offset(idx, addendStrides)]));
        } while(nextIndex(idx, extents, 0, nOut));

        return expected;
    }

    // ---- RAII guards: cleanup runs on every exit path, including a failed ASSERT --

    struct HandleGuard
    {
        hiptensorHandle_t handle = nullptr;
        ~HandleGuard()
        {
            if(handle)
            {
                static_cast<void>(hiptensorDestroy(handle));
            }
        }
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

    struct DeviceBuffer
    {
        void* ptr = nullptr;
        ~DeviceBuffer()
        {
            if(ptr)
            {
                static_cast<void>(hipFree(ptr));
            }
        }
    };

    hiptensorStatus_t createDesc(hiptensorHandle_t   handle,
                                 TensorDescGuard&    guard,
                                 HostTensor const&   t,
                                 hiptensorDataType_t type = HIPTENSOR_R_32F)
    {
        return hiptensorCreateTensorDescriptor(handle,
                                               &guard.desc,
                                               t.modes.size(),
                                               t.extents.data(),
                                               t.layout == Layout::Default ? nullptr
                                                                           : t.strides.data(),
                                               type,
                                               alignment);
    }

    void upload(DeviceBuffer& buffer, std::vector<float> const& host)
    {
        ASSERT_EQ(hipMalloc(&buffer.ptr, host.size() * sizeof(float)), hipSuccess);
        ASSERT_EQ(
            hipMemcpy(buffer.ptr, host.data(), host.size() * sizeof(float), hipMemcpyHostToDevice),
            hipSuccess);
    }

    // Bit-exact comparison of the whole output footprint, gaps included.
    void expectMatches(DeviceBuffer const&       device,
                       std::vector<float> const& expected,
                       std::string const&        name)
    {
        std::vector<float> result(expected.size());
        ASSERT_EQ(
            hipMemcpy(
                result.data(), device.ptr, result.size() * sizeof(float), hipMemcpyDeviceToHost),
            hipSuccess);

        std::size_t        mismatches = 0;
        std::ostringstream firstMismatches;
        for(std::size_t i = 0; i < result.size(); i++)
        {
            if(result[i] != expected[i])
            {
                if(mismatches < 4)
                {
                    firstMismatches << " [offset " << i << ": got " << result[i] << ", expected "
                                    << expected[i] << "]";
                }
                mismatches++;
            }
        }
        EXPECT_EQ(mismatches, 0u) << name << ":" << firstMismatches.str();
    }

    void runAndCheck(LayoutCase const& tc)
    {
        auto const modes = patternModes(tc.pattern, tc.rank);
        auto       modesC
            = tc.reverseModesC ? std::vector<int32_t>(modes.d.crbegin(), modes.d.crend()) : modes.d;

        auto a = makeTensor(modes.a, tc.layoutA);
        auto b = makeTensor(modes.b, tc.layoutB);
        auto c = makeTensor(modesC, tc.layoutC);
        auto d = makeTensor(modes.d, tc.layoutD);
        fillRandomIntegers(a, -3, 3, 1u);
        fillRandomIntegers(b, -3, 3, 2u);
        fillRandomIntegers(c, -4, 4, 3u);

        auto const expected = reference({&a, &b}, c, d);

        HandleGuard handle;
        ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);

        TensorDescGuard descA, descB, descC, descD;
        ASSERT_EQ(createDesc(handle.handle, descA, a), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descB, b), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descC, c), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descD, d), HIPTENSOR_STATUS_SUCCESS);

        OpDescGuard op;
        auto        status = hiptensorCreateContraction(handle.handle,
                                                 &op.desc,
                                                 descA.desc,
                                                 a.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descB.desc,
                                                 b.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descC.desc,
                                                 c.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descD.desc,
                                                 d.modes.data(),
                                                 HIPTENSOR_COMPUTE_DESC_32F);
        if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            GTEST_SKIP() << "This contraction is not supported on this device";
        }
        ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

        // ALGO_DEFAULT runs every candidate kernel on scratch buffers at plan time.
        PlanPrefGuard pref;
        ASSERT_EQ(hiptensorCreatePlanPreference(
                      handle.handle, &pref.pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
                  HIPTENSOR_STATUS_SUCCESS);

        uint64_t workspaceSize = 0;
        ASSERT_EQ(
            hiptensorEstimateWorkspaceSize(
                handle.handle, op.desc, pref.pref, HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize),
            HIPTENSOR_STATUS_SUCCESS);

        PlanGuard plan;
        ASSERT_EQ(hiptensorCreatePlan(handle.handle, &plan.plan, op.desc, pref.pref, workspaceSize),
                  HIPTENSOR_STATUS_SUCCESS);

        DeviceBuffer devA, devB, devC, devD, workspace;
        ASSERT_NO_FATAL_FAILURE(upload(devA, a.data));
        ASSERT_NO_FATAL_FAILURE(upload(devB, b.data));
        ASSERT_NO_FATAL_FAILURE(upload(devC, c.data));
        ASSERT_NO_FATAL_FAILURE(upload(devD, d.data)); // all sentinel
        if(workspaceSize > 0)
        {
            ASSERT_EQ(hipMalloc(&workspace.ptr, workspaceSize), hipSuccess);
        }

        ASSERT_EQ(hiptensorContract(handle.handle,
                                    plan.plan,
                                    &alphaValue,
                                    devA.ptr,
                                    devB.ptr,
                                    &betaValue,
                                    devC.ptr,
                                    devD.ptr,
                                    workspace.ptr,
                                    workspaceSize,
                                    0 /* stream */),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        ASSERT_NO_FATAL_FAILURE(expectMatches(devD, expected, caseName(tc)));
    }

    std::vector<LayoutCase> layoutCases()
    {
        std::vector<LayoutCase> cases;

        // Every tensor in the same layout, every pattern, ranks 2 and 3.
        for(auto pattern :
            {Pattern::Canonical, Pattern::P1, Pattern::P2, Pattern::P3, Pattern::P4, Pattern::P5})
        {
            for(int rank : {2, 3})
            {
                for(auto layout :
                    {Layout::Default, Layout::ColMajor, Layout::RowMajor, Layout::Padded})
                {
                    cases.push_back({pattern, rank, layout, layout, layout, layout, false});
                }
            }
        }

        for(auto pattern : {Pattern::Canonical, Pattern::P1})
        {
            for(int rank : {2, 3})
            {
                auto const col = Layout::ColMajor;
                auto const row = Layout::RowMajor;
                auto const pad = Layout::Padded;

                // Operands in different layouts.
                cases.push_back({pattern, rank, col, row, col, col, false});
                cases.push_back({pattern, rank, row, col, col, col, false});
                cases.push_back({pattern, rank, pad, col, col, col, false});
                cases.push_back({pattern, rank, col, col, row, row, false});

                // C laid out differently from D.
                cases.push_back({pattern, rank, col, col, row, col, false});
                cases.push_back({pattern, rank, col, col, col, row, false});
                cases.push_back({pattern, rank, col, col, pad, col, false});
                cases.push_back({pattern, rank, col, col, col, col, true});
            }
        }

        // Padded strides whose footprint is far larger than the element count, so a
        // buffer sized from the lengths alone is overrun at plan time.
        for(auto pattern : {Pattern::Canonical, Pattern::P3})
        {
            auto const wide = Layout::WidePadded;
            cases.push_back({pattern, 2, wide, wide, wide, wide, false});
        }

        return cases;
    }

    // E = alpha * A * B * C + beta * D, run as T = A * B then E = T * C + D. Two mode
    // groups per operand: M and S are output modes, K is contracted by the first step
    // and N by the second.
    struct TrinaryCase
    {
        bool   interleaved; // A and E interleave their mode groups
        Layout layoutInputs; // A, B and C
        Layout layoutD;
        Layout layoutE;
    };

    std::string caseName(TrinaryCase const& c)
    {
        return std::string(c.interleaved ? "Interleaved" : "Canonical") + "_ABC"
               + toString(c.layoutInputs) + "_D" + toString(c.layoutD) + "_E" + toString(c.layoutE);
    }

    void runAndCheck(TrinaryCase const& tc)
    {
        std::vector<int32_t> m(mModes, mModes + 2);
        std::vector<int32_t> k(kModes, kModes + 2);
        std::vector<int32_t> n(nModes, nModes + 2);
        std::vector<int32_t> o(sModes, sModes + 2);

        // Canonical:   A[M,K]          B[N,K]  C[S,N]  E[M,S]
        // Interleaved: A[m0,k0,m1,k1]  B[K,N]  C[N,S]  E[m0,s0,m1,s1]
        std::vector<int32_t> modesA, modesB, modesC, modesE;
        if(tc.interleaved)
        {
            modesA = {m[0], k[0], m[1], k[1]};
            modesB = {k[0], k[1], n[0], n[1]};
            modesC = {n[0], n[1], o[0], o[1]};
            modesE = {m[0], o[0], m[1], o[1]};
        }
        else
        {
            modesA = {m[0], m[1], k[0], k[1]};
            modesB = {n[0], n[1], k[0], k[1]};
            modesC = {o[0], o[1], n[0], n[1]};
            modesE = {m[0], m[1], o[0], o[1]};
        }

        auto a = makeTensor(modesA, tc.layoutInputs);
        auto b = makeTensor(modesB, tc.layoutInputs);
        auto c = makeTensor(modesC, tc.layoutInputs);
        auto d = makeTensor(modesE, tc.layoutD);
        auto e = makeTensor(modesE, tc.layoutE);
        fillRandomIntegers(a, -3, 3, 4u);
        fillRandomIntegers(b, -3, 3, 5u);
        fillRandomIntegers(c, -3, 3, 6u);
        fillRandomIntegers(d, -4, 4, 7u);

        auto const expected = reference({&a, &b, &c}, d, e);

        HandleGuard handle;
        ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);

        TensorDescGuard descA, descB, descC, descD, descE;
        ASSERT_EQ(createDesc(handle.handle, descA, a), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descB, b), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descC, c), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descD, d), HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(createDesc(handle.handle, descE, e), HIPTENSOR_STATUS_SUCCESS);

        OpDescGuard op;
        auto        status = hiptensorCreateContractionTrinary(handle.handle,
                                                        &op.desc,
                                                        descA.desc,
                                                        a.modes.data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descB.desc,
                                                        b.modes.data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descC.desc,
                                                        c.modes.data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descD.desc,
                                                        d.modes.data(),
                                                        HIPTENSOR_OP_IDENTITY,
                                                        descE.desc,
                                                        e.modes.data(),
                                                        HIPTENSOR_COMPUTE_DESC_32F);
        if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            GTEST_SKIP() << "This contraction is not supported on this device";
        }
        ASSERT_EQ(status, HIPTENSOR_STATUS_SUCCESS);

        PlanPrefGuard pref;
        ASSERT_EQ(hiptensorCreatePlanPreference(
                      handle.handle, &pref.pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE),
                  HIPTENSOR_STATUS_SUCCESS);

        uint64_t workspaceSize = 0;
        ASSERT_EQ(
            hiptensorEstimateWorkspaceSize(
                handle.handle, op.desc, pref.pref, HIPTENSOR_WORKSPACE_DEFAULT, &workspaceSize),
            HIPTENSOR_STATUS_SUCCESS);

        PlanGuard plan;
        ASSERT_EQ(hiptensorCreatePlan(handle.handle, &plan.plan, op.desc, pref.pref, workspaceSize),
                  HIPTENSOR_STATUS_SUCCESS);

        DeviceBuffer devA, devB, devC, devD, devE, workspace;
        ASSERT_NO_FATAL_FAILURE(upload(devA, a.data));
        ASSERT_NO_FATAL_FAILURE(upload(devB, b.data));
        ASSERT_NO_FATAL_FAILURE(upload(devC, c.data));
        ASSERT_NO_FATAL_FAILURE(upload(devD, d.data));
        ASSERT_NO_FATAL_FAILURE(upload(devE, e.data)); // all sentinel
        ASSERT_GT(workspaceSize, 0u); // holds the intermediate T
        ASSERT_EQ(hipMalloc(&workspace.ptr, workspaceSize), hipSuccess);
        // The intermediate's initial contents are not under test here; start it finite.
        ASSERT_EQ(hipMemset(workspace.ptr, 0, workspaceSize), hipSuccess);

        ASSERT_EQ(hiptensorContractTrinary(handle.handle,
                                           plan.plan,
                                           &alphaValue,
                                           devA.ptr,
                                           devB.ptr,
                                           devC.ptr,
                                           &betaValue,
                                           devD.ptr,
                                           devE.ptr,
                                           workspace.ptr,
                                           workspaceSize,
                                           0 /* stream */),
                  HIPTENSOR_STATUS_SUCCESS);
        ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

        ASSERT_NO_FATAL_FAILURE(expectMatches(devE, expected, caseName(tc)));
    }

    std::vector<TrinaryCase> trinaryCases()
    {
        auto const col = Layout::ColMajor;
        auto const row = Layout::RowMajor;
        auto const pad = Layout::Padded;
        auto const def = Layout::Default;
        return {
            {false, col, col, col},
            {true, def, def, def},
            {true, col, col, col},
            {true, row, row, row},
            {true, pad, pad, pad},
            {true, col, row, col},
            {false, col, row, col},
        };
    }

    class ContractionLayoutTest : public ::testing::TestWithParam<LayoutCase>
    {
    };

    class TrinaryContractionLayoutTest : public ::testing::TestWithParam<TrinaryCase>
    {
    };

} // namespace

TEST_P(ContractionLayoutTest, MatchesIndependentReference)
{
    runAndCheck(GetParam());
}

INSTANTIATE_TEST_SUITE_P(ContractionLayout,
                         ContractionLayoutTest,
                         ::testing::ValuesIn(layoutCases()),
                         [](::testing::TestParamInfo<LayoutCase> const& info) {
                             return caseName(info.param);
                         });

TEST_P(TrinaryContractionLayoutTest, MatchesIndependentReference)
{
    runAndCheck(GetParam());
}

INSTANTIATE_TEST_SUITE_P(TrinaryContractionLayout,
                         TrinaryContractionLayoutTest,
                         ::testing::ValuesIn(trinaryCases()),
                         [](::testing::TestParamInfo<TrinaryCase> const& info) {
                             return caseName(info.param);
                         });

namespace
{
    // Creates D = A * B + C with f32 data and compute, A, B and D canonical and column-major.
    hiptensorStatus_t createWithC(HostTensor const& c, HostTensor const& d)
    {
        auto const modes = patternModes(Pattern::Canonical, 2);
        auto       a     = makeTensor(modes.a, Layout::ColMajor);
        auto       b     = makeTensor(modes.b, Layout::ColMajor);

        HandleGuard handle;
        EXPECT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);

        TensorDescGuard descA, descB, descC, descD;
        EXPECT_EQ(createDesc(handle.handle, descA, a), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(handle.handle, descB, b), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(handle.handle, descC, c), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(handle.handle, descD, d), HIPTENSOR_STATUS_SUCCESS);

        OpDescGuard op;
        return hiptensorCreateContraction(handle.handle,
                                          &op.desc,
                                          descA.desc,
                                          a.modes.data(),
                                          HIPTENSOR_OP_IDENTITY,
                                          descB.desc,
                                          b.modes.data(),
                                          HIPTENSOR_OP_IDENTITY,
                                          descC.desc,
                                          c.modes.data(),
                                          HIPTENSOR_OP_IDENTITY,
                                          descD.desc,
                                          d.modes.data(),
                                          HIPTENSOR_COMPUTE_DESC_32F);
    }

    // Creates E = A * B * C + D with f32 data and compute, every tensor column-major, in the
    // canonical mode order of TrinaryCase.
    hiptensorStatus_t createTrinaryWithD(HostTensor const& d, HostTensor const& e)
    {
        std::vector<int32_t> const modesA = {mModes[0], mModes[1], kModes[0], kModes[1]};
        std::vector<int32_t> const modesB = {nModes[0], nModes[1], kModes[0], kModes[1]};
        std::vector<int32_t> const modesC = {sModes[0], sModes[1], nModes[0], nModes[1]};

        auto a = makeTensor(modesA, Layout::ColMajor);
        auto b = makeTensor(modesB, Layout::ColMajor);
        auto c = makeTensor(modesC, Layout::ColMajor);

        HandleGuard handle;
        EXPECT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);

        TensorDescGuard descA, descB, descC, descD, descE;
        EXPECT_EQ(createDesc(handle.handle, descA, a), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(handle.handle, descB, b), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(handle.handle, descC, c), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(handle.handle, descD, d), HIPTENSOR_STATUS_SUCCESS);
        EXPECT_EQ(createDesc(handle.handle, descE, e), HIPTENSOR_STATUS_SUCCESS);

        OpDescGuard op;
        return hiptensorCreateContractionTrinary(handle.handle,
                                                 &op.desc,
                                                 descA.desc,
                                                 a.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descB.desc,
                                                 b.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descC.desc,
                                                 c.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descD.desc,
                                                 d.modes.data(),
                                                 HIPTENSOR_OP_IDENTITY,
                                                 descE.desc,
                                                 e.modes.data(),
                                                 HIPTENSOR_COMPUTE_DESC_32F);
    }

    std::vector<int32_t> trinaryOutputModes()
    {
        return {mModes[0], mModes[1], sModes[0], sModes[1]};
    }
} // namespace

// C must carry exactly D's modes, each with D's extent, since it is read through its own modes
// and strides. A mode with another extent is an invalid value, as in the elementwise
// operations; other modes are not supported.
TEST(ContractionLayoutValidationTest, CWithDifferentExtentsIsInvalid)
{
    auto const modes = patternModes(Pattern::Canonical, 2);
    auto       d     = makeTensor(modes.d, Layout::ColMajor);
    auto       c     = d;
    c.extents[0] += 1;
    c.strides = stridesFor(Layout::ColMajor, c.extents);

    auto status = createWithC(c, d);
    if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "This contraction is not supported on this device";
    }
    EXPECT_EQ(status, HIPTENSOR_STATUS_INVALID_VALUE);
}

TEST(ContractionLayoutValidationTest, CWithOtherModesIsNotSupported)
{
    auto const modes  = patternModes(Pattern::Canonical, 2);
    auto       d      = makeTensor(modes.d, Layout::ColMajor);
    auto       modesC = modes.d;
    // A contracted mode of the same extent in place of the first output mode.
    ASSERT_EQ(extentOf(kModes[1]), extentOf(nModes[0]));
    ASSERT_EQ(modesC[2], nModes[0]);
    modesC[2] = kModes[1];
    auto c    = makeTensor(modesC, Layout::ColMajor);

    auto status = createWithC(c, d);
    if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "This contraction is not supported on this device";
    }
    EXPECT_EQ(status, HIPTENSOR_STATUS_NOT_SUPPORTED);
}

TEST(ContractionLayoutValidationTest, TrinaryDWithDifferentExtentsIsInvalid)
{
    auto e = makeTensor(trinaryOutputModes(), Layout::ColMajor);
    auto d = e;
    d.extents[3] += 1;
    d.strides = stridesFor(Layout::ColMajor, d.extents);

    auto status = createTrinaryWithD(d, e);
    if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "This contraction is not supported on this device";
    }
    EXPECT_EQ(status, HIPTENSOR_STATUS_INVALID_VALUE);
}

TEST(ContractionLayoutValidationTest, TrinaryDWithOtherModesIsNotSupported)
{
    auto e      = makeTensor(trinaryOutputModes(), Layout::ColMajor);
    auto modesD = trinaryOutputModes();
    // A contracted mode of the same extent in place of an output mode.
    ASSERT_EQ(extentOf(kModes[0]), extentOf(sModes[0]));
    modesD[2] = kModes[0];
    auto d    = makeTensor(modesD, Layout::ColMajor);

    auto status = createTrinaryWithD(d, e);
    if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "This contraction is not supported on this device";
    }
    EXPECT_EQ(status, HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// The complex path combines C and D element by element in D's layout, so a C laid
// out differently cannot be honoured and is rejected.
TEST(ContractionLayoutValidationTest, ComplexCLaidOutDifferentlyFromDIsNotSupported)
{
    auto const modes = patternModes(Pattern::Canonical, 2);
    auto       a     = makeTensor(modes.a, Layout::ColMajor);
    auto       b     = makeTensor(modes.b, Layout::ColMajor);
    auto       c     = makeTensor(modes.d, Layout::RowMajor);
    auto       d     = makeTensor(modes.d, Layout::ColMajor);

    HandleGuard handle;
    ASSERT_EQ(hiptensorCreate(&handle.handle), HIPTENSOR_STATUS_SUCCESS);

    TensorDescGuard descA, descB, descC, descD;
    ASSERT_EQ(createDesc(handle.handle, descA, a, HIPTENSOR_C_32F), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(createDesc(handle.handle, descB, b, HIPTENSOR_C_32F), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(createDesc(handle.handle, descC, c, HIPTENSOR_C_32F), HIPTENSOR_STATUS_SUCCESS);
    ASSERT_EQ(createDesc(handle.handle, descD, d, HIPTENSOR_C_32F), HIPTENSOR_STATUS_SUCCESS);

    OpDescGuard op;
    auto        status = hiptensorCreateContraction(handle.handle,
                                             &op.desc,
                                             descA.desc,
                                             a.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             descB.desc,
                                             b.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             descC.desc,
                                             c.modes.data(),
                                             HIPTENSOR_OP_IDENTITY,
                                             descD.desc,
                                             d.modes.data(),
                                             HIPTENSOR_COMPUTE_DESC_C32F);
    if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
    {
        GTEST_SKIP() << "This contraction is not supported on this device";
    }
    EXPECT_EQ(status, HIPTENSOR_STATUS_NOT_SUPPORTED);
}
