/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hiptensor/hiptensor.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <vector>

// D[m0,m1,n] = alpha * sum_{k0,k1} A[m0,m1,k0,k1] * B[n,k0,k1] (+ beta * C[m0,m1,n]),
// over every listing order and memory order of each operand. The reference is
// computed from mode labels and the strides given to the descriptors, so it
// shares no code with hipTensor's host path (unlike the YAML suites).

namespace
{
    constexpr int32_t M0 = 'm', M1 = 'p', N0 = 'n', K0 = 'k', K1 = 'q';
    constexpr double  kTolerance = 1e-4;

    struct Layout
    {
        bool reversed; // modes listed in reverse order
        bool rowMajor; // packed with the last listed mode contiguous
    };

    constexpr Layout kLayouts[] = {{false, false}, {false, true}, {true, false}, {true, true}};

    std::string name(Layout l)
    {
        return std::string(l.reversed ? "rev" : "nat") + (l.rowMajor ? "/row" : "/col");
    }

    using Extents = std::map<int32_t, int64_t>;

    Extents extents(bool unitMode)
    {
        return {{M0, 3}, {M1, unitMode ? 1 : 4}, {N0, 5}, {K0, 2}, {K1, 3}};
    }

    struct Tensor
    {
        std::vector<int32_t> modes; // listed order
        std::vector<int64_t> lengths; // listed order
        std::vector<int64_t> strides; // listed order
        int64_t              elements = 1;
    };

    Tensor makeTensor(std::vector<int32_t> modes, Extents const& ext, Layout l)
    {
        if(l.reversed)
            std::reverse(modes.begin(), modes.end());
        Tensor t;
        t.modes = modes;
        for(auto m : modes)
            t.lengths.push_back(ext.at(m));
        t.strides.resize(modes.size());
        for(size_t j = 0; j < modes.size(); j++)
        {
            size_t i     = l.rowMajor ? modes.size() - 1 - j : j;
            t.strides[i] = t.elements;
            t.elements *= t.lengths[i];
        }
        return t;
    }

    int64_t offset(Tensor const& t, Extents const& idx)
    {
        int64_t off = 0;
        for(size_t i = 0; i < t.modes.size(); i++)
            off += t.strides[i] * idx.at(t.modes[i]);
        return off;
    }

    std::vector<float> randomData(int64_t n, unsigned seed)
    {
        std::mt19937                          gen(seed);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        std::vector<float>                    v(n);
        for(auto& x : v)
            x = dist(gen);
        return v;
    }

    // Independent reference: loops over mode labels, addresses memory with the
    // descriptor strides, accumulates in double.
    std::vector<double> reference(Extents const&            ext,
                                  float                     alpha,
                                  float                     beta,
                                  Tensor const&             a,
                                  std::vector<float> const& A,
                                  Tensor const&             b,
                                  std::vector<float> const& B,
                                  Tensor const*             c,
                                  std::vector<float> const* C,
                                  Tensor const&             d)
    {
        std::vector<double> D(d.elements, 0.0);
        Extents             i;
        for(i[M0] = 0; i[M0] < ext.at(M0); i[M0]++)
        {
            for(i[M1] = 0; i[M1] < ext.at(M1); i[M1]++)
            {
                for(i[N0] = 0; i[N0] < ext.at(N0); i[N0]++)
                {
                    double acc = 0.0;
                    for(i[K0] = 0; i[K0] < ext.at(K0); i[K0]++)
                    {
                        for(i[K1] = 0; i[K1] < ext.at(K1); i[K1]++)
                        {
                            acc += double(A[offset(a, i)]) * double(B[offset(b, i)]);
                        }
                    }
                    double out = double(alpha) * acc;
                    if(c)
                        out += double(beta) * double((*C)[offset(*c, i)]);
                    D[offset(d, i)] = out;
                }
            }
        }
        return D;
    }

    double maxRelError(std::vector<float> const& got, std::vector<double> const& ref)
    {
        double scale = 1.0, err = 0.0;
        for(double x : ref)
            scale = std::max(scale, std::abs(x));
        for(size_t i = 0; i < ref.size(); i++)
        {
            double e = std::abs(double(got[i]) - ref[i]);
            if(std::isnan(e))
                return std::numeric_limits<double>::infinity(); // element never written
            err = std::max(err, e);
        }
        return err / scale;
    }

    hiptensorTensorDescriptor_t describe(hiptensorHandle_t h, Tensor const& t)
    {
        hiptensorTensorDescriptor_t desc = nullptr;
        EXPECT_EQ(hiptensorCreateTensorDescriptor(h,
                                                  &desc,
                                                  static_cast<uint32_t>(t.modes.size()),
                                                  t.lengths.data(),
                                                  t.strides.data(),
                                                  HIPTENSOR_R_32F,
                                                  sizeof(float)),
                  HIPTENSOR_STATUS_SUCCESS);
        return desc;
    }

    struct Result
    {
        hiptensorStatus_t  status = HIPTENSOR_STATUS_SUCCESS;
        std::vector<float> D;
    };

    Result run(hiptensorHandle_t         h,
               float                     alpha,
               float                     beta,
               Tensor const&             a,
               std::vector<float> const& A,
               Tensor const&             b,
               std::vector<float> const& B,
               Tensor const*             c,
               std::vector<float> const* C,
               Tensor const&             d)
    {
        Result r;
        auto   descA = describe(h, a);
        auto   descB = describe(h, b);
        auto   descD = describe(h, d);
        auto   descC = c ? describe(h, *c) : nullptr; // no C = scale contraction

        hiptensorOperationDescriptor_t op   = nullptr;
        hiptensorPlanPreference_t      pref = nullptr;
        hiptensorPlan_t                plan = nullptr;
        uint64_t                       ws   = 0;
        void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dW = nullptr;

        r.status = hiptensorCreateContraction(h,
                                              &op,
                                              descA,
                                              a.modes.data(),
                                              HIPTENSOR_OP_IDENTITY,
                                              descB,
                                              b.modes.data(),
                                              HIPTENSOR_OP_IDENTITY,
                                              descC,
                                              c ? c->modes.data() : nullptr,
                                              HIPTENSOR_OP_IDENTITY,
                                              descD,
                                              d.modes.data(),
                                              HIPTENSOR_COMPUTE_DESC_32F);
        if(r.status == HIPTENSOR_STATUS_SUCCESS)
            r.status = hiptensorCreatePlanPreference(
                h, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE);
        if(r.status == HIPTENSOR_STATUS_SUCCESS)
            r.status
                = hiptensorEstimateWorkspaceSize(h, op, pref, HIPTENSOR_WORKSPACE_DEFAULT, &ws);
        if(r.status == HIPTENSOR_STATUS_SUCCESS)
            r.status = hiptensorCreatePlan(h, &plan, op, pref, ws);

        if(r.status == HIPTENSOR_STATUS_SUCCESS)
        {
            auto upload = [](void** dst, std::vector<float> const& src) {
                EXPECT_EQ(hipMalloc(dst, src.size() * sizeof(float)), hipSuccess);
                EXPECT_EQ(
                    hipMemcpy(*dst, src.data(), src.size() * sizeof(float), hipMemcpyHostToDevice),
                    hipSuccess);
            };
            upload(&dA, A);
            upload(&dB, B);
            if(C)
                upload(&dC, *C);
            // NaN in D: an element the kernel never writes cannot pass the check.
            upload(&dD, std::vector<float>(d.elements, std::numeric_limits<float>::quiet_NaN()));
            if(ws > 0)
            {
                EXPECT_EQ(hipMalloc(&dW, ws), hipSuccess);
            }

            r.status = hiptensorContract(h, plan, &alpha, dA, dB, &beta, dC, dD, dW, ws, nullptr);
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

            r.D.resize(d.elements);
            EXPECT_EQ(hipMemcpy(r.D.data(), dD, d.elements * sizeof(float), hipMemcpyDeviceToHost),
                      hipSuccess);
        }

        for(void* p : {dA, dB, dC, dD, dW})
            (void)hipFree(p);
        if(plan)
            hiptensorDestroyPlan(plan);
        if(pref)
            hiptensorDestroyPlanPreference(pref);
        if(op)
            hiptensorDestroyOperationDescriptor(op);
        for(auto desc : {descA, descB, descC, descD})
            if(desc)
                hiptensorDestroyTensorDescriptor(desc);
        return r;
    }

    struct Tally
    {
        int total = 0, wrong = 0, skipped = 0;
    };

    void check(hiptensorHandle_t         h,
               Extents const&            ext,
               float                     alpha,
               float                     beta,
               Tensor const&             a,
               std::vector<float> const& A,
               Tensor const&             b,
               std::vector<float> const& B,
               Tensor const*             c,
               std::vector<float> const* C,
               Tensor const&             d,
               std::string const&        label,
               Tally&                    t)
    {
        t.total++;
        auto r = run(h, alpha, beta, a, A, b, B, c, C, d);
        if(r.status == HIPTENSOR_STATUS_ARCH_MISMATCH || r.status == HIPTENSOR_STATUS_NOT_SUPPORTED)
        {
            t.skipped++;
            return;
        }
        ASSERT_EQ(r.status, HIPTENSOR_STATUS_SUCCESS) << label;
        auto err = maxRelError(r.D, reference(ext, alpha, beta, a, A, b, B, c, C, d));
        if(err > kTolerance)
        {
            t.wrong++;
            ADD_FAILURE() << label << ": max relative error " << err;
        }
    }

    void report(Tally const& t)
    {
        std::cout << "[layouts] " << t.wrong << " of " << (t.total - t.skipped) << " wrong, "
                  << t.skipped << " skipped\n";
        if(t.skipped == t.total)
            GTEST_SKIP() << "no layout ran on this device";
        EXPECT_EQ(t.wrong, 0);
    }

    class ContractionLayoutTest : public ::testing::TestWithParam<bool> // true: m1 has length 1
    {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(hiptensorCreate(&mHandle), HIPTENSOR_STATUS_SUCCESS);
        }
        void TearDown() override
        {
            if(mHandle)
                hiptensorDestroy(mHandle);
        }
        hiptensorHandle_t mHandle = nullptr;
    };
} // namespace

TEST_P(ContractionLayoutTest, Scale)
{
    auto const ext = extents(GetParam());
    Tally      t;
    for(auto la : kLayouts)
        for(auto lb : kLayouts)
            for(auto ld : kLayouts)
            {
                auto a = makeTensor({M0, M1, K0, K1}, ext, la);
                auto b = makeTensor({N0, K0, K1}, ext, lb);
                auto d = makeTensor({M0, M1, N0}, ext, ld);
                auto A = randomData(a.elements, 1);
                auto B = randomData(b.elements, 2);
                check(mHandle,
                      ext,
                      1.5f,
                      0.f,
                      a,
                      A,
                      b,
                      B,
                      nullptr,
                      nullptr,
                      d,
                      "A " + name(la) + " B " + name(lb) + " D " + name(ld),
                      t);
            }
    report(t);
}

TEST_P(ContractionLayoutTest, Bilinear)
{
    auto const ext = extents(GetParam());
    Tally      t;
    for(auto la : kLayouts)
        for(auto lb : kLayouts)
            for(auto ld : kLayouts)
            {
                auto a = makeTensor({M0, M1, K0, K1}, ext, la);
                auto b = makeTensor({N0, K0, K1}, ext, lb);
                auto d = makeTensor({M0, M1, N0}, ext, ld);
                auto A = randomData(a.elements, 1);
                auto B = randomData(b.elements, 2);
                auto C = randomData(d.elements, 3);
                check(mHandle,
                      ext,
                      1.5f,
                      0.5f,
                      a,
                      A,
                      b,
                      B,
                      &d,
                      &C,
                      d,
                      "A " + name(la) + " B " + name(lb) + " C=D " + name(ld),
                      t);
            }
    report(t);
}

// C laid out differently from D: develop normalises C with D's strides.
TEST_P(ContractionLayoutTest, BilinearCLayoutDiffersFromD)
{
    auto const   ext = extents(GetParam());
    Layout const canonical{false, false};
    Tally        t;
    for(auto lc : kLayouts)
    {
        auto a = makeTensor({M0, M1, K0, K1}, ext, canonical);
        auto b = makeTensor({N0, K0, K1}, ext, canonical);
        auto d = makeTensor({M0, M1, N0}, ext, canonical);
        auto c = makeTensor({M0, M1, N0}, ext, lc);
        auto A = randomData(a.elements, 1);
        auto B = randomData(b.elements, 2);
        auto C = randomData(c.elements, 3);
        check(mHandle, ext, 1.5f, 0.5f, a, A, b, B, &c, &C, d, "C " + name(lc), t);
    }
    report(t);
}

INSTANTIATE_TEST_SUITE_P(Layouts,
                         ContractionLayoutTest,
                         ::testing::Values(false, true),
                         [](auto const& info) { return info.param ? "UnitMode" : "Full"; });
