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
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

// D[m0,m1,n] = alpha * sum_{k0,k1} A[m0,m1,k0,k1] * B[n,k0,k1] (+ beta * C[m0,m1,n]),
// for every listing order and memory order of each operand. The expected values come
// from a reference computed from the mode labels and the strides given to the
// descriptors, so it shares no code with hipTensor's host path (unlike the YAML suites,
// whose reference goes through the same mode normalization as the library).

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

    // Part of a gtest name, e.g. "nat_col".
    std::string tag(Layout l)
    {
        return std::string(l.reversed ? "rev" : "nat") + (l.rowMajor ? "_row" : "_col");
    }

    // Lets gtest print a Layout parameter readably in failure messages.
    void PrintTo(Layout l, std::ostream* os)
    {
        *os << tag(l);
    }

    // Maps a mode label to a value: its extent, or its current index.
    using ModeMap = std::map<int32_t, int64_t>;

    // unitMode: m1 has length 1, so normalization meets a unit mode inside a group.
    ModeMap extents(bool unitMode)
    {
        return {{M0, 3}, {M1, unitMode ? 1 : 4}, {N0, 5}, {K0, 2}, {K1, 3}};
    }

    struct Tensor
    {
        std::vector<int32_t> modes; // listed order
        std::vector<int64_t> lengths; // listed order
        std::vector<int64_t> strides; // listed order
        int64_t              elements = 1;
        std::vector<float>   data; // host values; empty for the output D
    };

    // A packed tensor with the given layout and no data.
    Tensor makeTensor(std::vector<int32_t> modes, ModeMap const& ext, Layout l)
    {
        if(l.reversed)
        {
            std::ranges::reverse(modes);
        }
        Tensor t;
        t.modes = modes;
        for(auto m : modes)
        {
            t.lengths.push_back(ext.at(m));
        }
        t.strides.resize(modes.size());
        for(size_t j = 0; j < modes.size(); j++)
        {
            size_t i     = l.rowMajor ? modes.size() - 1 - j : j;
            t.strides[i] = t.elements;
            t.elements *= t.lengths[i];
        }
        return t;
    }

    // An input tensor: the layout from makeTensor, filled with values in [-1, 1).
    Tensor makeInput(std::vector<int32_t> modes, ModeMap const& ext, Layout l, unsigned seed)
    {
        Tensor                                t = makeTensor(std::move(modes), ext, l);
        std::mt19937                          gen(seed);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        t.data.resize(t.elements);
        for(auto& x : t.data)
        {
            x = dist(gen);
        }
        return t;
    }

    // D = alpha * sum(A * B) (+ beta * C)
    struct Contraction
    {
        float                 alpha;
        float                 beta;
        Tensor                a;
        Tensor                b;
        std::optional<Tensor> c; // absent: scale contraction
        Tensor                d;
    };

    int64_t offset(Tensor const& t, ModeMap const& idx)
    {
        int64_t off = 0;
        for(size_t i = 0; i < t.modes.size(); i++)
        {
            off += t.strides[i] * idx.at(t.modes[i]);
        }
        return off;
    }

    // Sets idx to the index values of the given modes at linear position n,
    // last mode fastest.
    void setIndex(std::vector<int32_t> const& modes, ModeMap const& ext, int64_t n, ModeMap& idx)
    {
        for(auto m = modes.rbegin(); m != modes.rend(); ++m)
        {
            idx[*m] = n % ext.at(*m);
            n /= ext.at(*m);
        }
    }

    // Calls f once for every combination of index values of the given modes,
    // with the current values in idx.
    template <typename F>
    void forEachIndex(std::vector<int32_t> const& modes, ModeMap const& ext, ModeMap& idx, F&& f)
    {
        int64_t count = 1;
        for(auto m : modes)
        {
            count *= ext.at(m);
        }
        for(int64_t n = 0; n < count; n++)
        {
            setIndex(modes, ext, n, idx);
            f();
        }
    }

    // Independent reference: D = alpha * sum(A * B) (+ beta * C), computed from the mode
    // labels and the descriptor strides only, accumulated in double.
    std::vector<double> reference(Contraction const& p, ModeMap const& ext)
    {
        std::vector<int32_t> contracted; // modes of A that D does not have
        for(auto m : p.a.modes)
        {
            if(std::ranges::find(p.d.modes, m) == p.d.modes.end())
            {
                contracted.push_back(m);
            }
        }

        std::vector<double> out(p.d.elements, 0.0);
        ModeMap             idx;
        forEachIndex(p.d.modes, ext, idx, [&] {
            double sum = 0.0;
            forEachIndex(contracted, ext, idx, [&] {
                sum += double(p.a.data[offset(p.a, idx)]) * double(p.b.data[offset(p.b, idx)]);
            });
            double value = double(p.alpha) * sum;
            if(p.c)
            {
                value += double(p.beta) * double(p.c->data[offset(*p.c, idx)]);
            }
            out[offset(p.d, idx)] = value;
        });
        return out;
    }

    // Largest absolute error relative to the largest reference value (at least 1).
    // An element the kernel never wrote is still NaN and counts as infinitely wrong.
    double maxRelError(std::vector<float> const& got, std::vector<double> const& ref)
    {
        double scale = 1.0;
        double err   = 0.0;
        for(double x : ref)
        {
            scale = std::max(scale, std::abs(x));
        }
        for(size_t i = 0; i < ref.size(); i++)
        {
            double e = std::abs(double(got[i]) - ref[i]);
            if(std::isnan(e))
            {
                return std::numeric_limits<double>::infinity();
            }
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

    // Runs the contraction on the device and returns D, or the first failing status.
    Result run(hiptensorHandle_t h, Contraction const& p)
    {
        Result r;
        auto   descA = describe(h, p.a);
        auto   descB = describe(h, p.b);
        auto   descD = describe(h, p.d);
        auto   descC = p.c ? describe(h, *p.c) : nullptr;

        hiptensorOperationDescriptor_t op   = nullptr;
        hiptensorPlanPreference_t      pref = nullptr;
        hiptensorPlan_t                plan = nullptr;
        uint64_t                       ws   = 0;
        void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr, *dW = nullptr;

        r.status = hiptensorCreateContraction(h,
                                              &op,
                                              descA,
                                              p.a.modes.data(),
                                              HIPTENSOR_OP_IDENTITY,
                                              descB,
                                              p.b.modes.data(),
                                              HIPTENSOR_OP_IDENTITY,
                                              descC,
                                              p.c ? p.c->modes.data() : nullptr,
                                              HIPTENSOR_OP_IDENTITY,
                                              descD,
                                              p.d.modes.data(),
                                              HIPTENSOR_COMPUTE_DESC_32F);
        if(r.status == HIPTENSOR_STATUS_SUCCESS)
        {
            r.status = hiptensorCreatePlanPreference(
                h, &pref, HIPTENSOR_ALGO_DEFAULT, HIPTENSOR_JIT_MODE_NONE);
        }
        if(r.status == HIPTENSOR_STATUS_SUCCESS)
        {
            r.status
                = hiptensorEstimateWorkspaceSize(h, op, pref, HIPTENSOR_WORKSPACE_DEFAULT, &ws);
        }
        if(r.status == HIPTENSOR_STATUS_SUCCESS)
        {
            r.status = hiptensorCreatePlan(h, &plan, op, pref, ws);
        }

        if(r.status == HIPTENSOR_STATUS_SUCCESS)
        {
            auto upload = [](void** dst, std::vector<float> const& src) {
                EXPECT_EQ(hipMalloc(dst, src.size() * sizeof(float)), hipSuccess);
                EXPECT_EQ(
                    hipMemcpy(*dst, src.data(), src.size() * sizeof(float), hipMemcpyHostToDevice),
                    hipSuccess);
            };
            upload(&dA, p.a.data);
            upload(&dB, p.b.data);
            if(p.c)
            {
                upload(&dC, p.c->data);
            }
            // NaN in D: an element the kernel never writes cannot pass the check.
            upload(&dD, std::vector<float>(p.d.elements, std::numeric_limits<float>::quiet_NaN()));
            if(ws > 0)
            {
                EXPECT_EQ(hipMalloc(&dW, ws), hipSuccess);
            }

            r.status
                = hiptensorContract(h, plan, &p.alpha, dA, dB, &p.beta, dC, dD, dW, ws, nullptr);
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

            r.D.resize(p.d.elements);
            EXPECT_EQ(
                hipMemcpy(r.D.data(), dD, p.d.elements * sizeof(float), hipMemcpyDeviceToHost),
                hipSuccess);
        }

        for(void* ptr : {dA, dB, dC, dD, dW})
        {
            (void)hipFree(ptr);
        }
        if(plan)
        {
            hiptensorDestroyPlan(plan);
        }
        if(pref)
        {
            hiptensorDestroyPlanPreference(pref);
        }
        if(op)
        {
            hiptensorDestroyOperationDescriptor(op);
        }
        for(auto desc : {descA, descB, descC, descD})
        {
            if(desc)
            {
                hiptensorDestroyTensorDescriptor(desc);
            }
        }
        return r;
    }

    // Runs the contraction and compares D with the reference. Skips when the device has
    // no kernel for it.
    void expectCorrect(hiptensorHandle_t h, ModeMap const& ext, Contraction const& p)
    {
        auto r = run(h, p);
        if(r.status == HIPTENSOR_STATUS_ARCH_MISMATCH || r.status == HIPTENSOR_STATUS_NOT_SUPPORTED)
        {
            GTEST_SKIP() << "no kernel for this contraction on this device";
        }
        ASSERT_EQ(r.status, HIPTENSOR_STATUS_SUCCESS);
        double err = maxRelError(r.D, reference(p, ext));
        EXPECT_LE(err, kTolerance) << "max relative error " << err;
    }

    class ContractionTestBase : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            ASSERT_EQ(hiptensorCreate(&mHandle), HIPTENSOR_STATUS_SUCCESS);
        }
        void TearDown() override
        {
            if(mHandle)
            {
                hiptensorDestroy(mHandle);
            }
        }
        hiptensorHandle_t mHandle = nullptr;
    };

    // Layouts of A, B and D; whether m1 has length 1.
    using LayoutCase = std::tuple<Layout, Layout, Layout, bool>;

    class ContractionLayoutTest : public ContractionTestBase,
                                  public ::testing::WithParamInterface<LayoutCase>
    {
    };

    // Layout of C (A, B and D are canonical); whether m1 has length 1.
    using CLayoutCase = std::tuple<Layout, bool>;

    class ContractionCLayoutTest : public ContractionTestBase,
                                   public ::testing::WithParamInterface<CLayoutCase>
    {
    };

    std::string sizeTag(bool unitMode)
    {
        return unitMode ? "UnitMode" : "Full";
    }

    // gtest case names, e.g. "A_nat_col_B_rev_row_D_nat_col_Full". Named functions, because
    // the commas of a structured binding would split the INSTANTIATE_TEST_SUITE_P macro.
    std::string layoutCaseName(::testing::TestParamInfo<LayoutCase> const& info)
    {
        auto const& [la, lb, ld, unitMode] = info.param;
        return "A_" + tag(la) + "_B_" + tag(lb) + "_D_" + tag(ld) + "_" + sizeTag(unitMode);
    }

    std::string cLayoutCaseName(::testing::TestParamInfo<CLayoutCase> const& info)
    {
        auto const& [lc, unitMode] = info.param;
        return "C_" + tag(lc) + "_" + sizeTag(unitMode);
    }
} // namespace

TEST_P(ContractionLayoutTest, Scale)
{
    auto const [la, lb, ld, unitMode] = GetParam();
    auto const ext                    = extents(unitMode);
    expectCorrect(mHandle,
                  ext,
                  {.alpha = 1.5f,
                   .beta  = 0.f,
                   .a     = makeInput({M0, M1, K0, K1}, ext, la, 1),
                   .b     = makeInput({N0, K0, K1}, ext, lb, 2),
                   .d     = makeTensor({M0, M1, N0}, ext, ld)});
}

// C is laid out like D.
TEST_P(ContractionLayoutTest, Bilinear)
{
    auto const [la, lb, ld, unitMode] = GetParam();
    auto const ext                    = extents(unitMode);
    expectCorrect(mHandle,
                  ext,
                  {.alpha = 1.5f,
                   .beta  = 0.5f,
                   .a     = makeInput({M0, M1, K0, K1}, ext, la, 1),
                   .b     = makeInput({N0, K0, K1}, ext, lb, 2),
                   .c     = makeInput({M0, M1, N0}, ext, ld, 3),
                   .d     = makeTensor({M0, M1, N0}, ext, ld)});
}

// C laid out differently from D: develop reads C with D's strides.
TEST_P(ContractionCLayoutTest, Bilinear)
{
    auto const [lc, unitMode] = GetParam();
    auto const   ext          = extents(unitMode);
    Layout const canonical{false, false};
    expectCorrect(mHandle,
                  ext,
                  {.alpha = 1.5f,
                   .beta  = 0.5f,
                   .a     = makeInput({M0, M1, K0, K1}, ext, canonical, 1),
                   .b     = makeInput({N0, K0, K1}, ext, canonical, 2),
                   .c     = makeInput({M0, M1, N0}, ext, lc, 3),
                   .d     = makeTensor({M0, M1, N0}, ext, canonical)});
}

INSTANTIATE_TEST_SUITE_P(Layouts,
                         ContractionLayoutTest,
                         ::testing::Combine(::testing::ValuesIn(kLayouts),
                                            ::testing::ValuesIn(kLayouts),
                                            ::testing::ValuesIn(kLayouts),
                                            ::testing::Bool()),
                         layoutCaseName);

INSTANTIATE_TEST_SUITE_P(Layouts,
                         ContractionCLayoutTest,
                         ::testing::Combine(::testing::ValuesIn(kLayouts), ::testing::Bool()),
                         cLayoutCaseName);
