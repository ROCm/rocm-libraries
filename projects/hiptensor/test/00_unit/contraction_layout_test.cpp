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
#include <array>
#include <cmath>
#include <complex>
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
//
// The trinary tests use
// E[m0,m1,n] = alpha * sum_{k0,k1} A[m0,m1,k0] * B[k0,k1] * C[k1,n] (+ beta * D[m0,m1,n]).

namespace
{
    constexpr int32_t M0 = 'm', M1 = 'p', N0 = 'n', K0 = 'k', K1 = 'q';
    constexpr double  kTolerance = 1e-4;

    // Floats per element.
    constexpr int kReal = 1, kComplex = 2;

    struct Layout
    {
        bool reversed; // modes listed in reverse order
        bool rowMajor; // packed with the last listed mode contiguous
        bool padded = false; // every stride doubled, which leaves gaps between elements
    };

    constexpr Layout kLayouts[] = {{false, false}, {false, true}, {true, false}, {true, true}};

    // The layout every contraction kernel supports: modes in their natural order, col-major.
    constexpr Layout kCanonical{false, false};

    constexpr Layout kRowMajor{false, true};
    constexpr Layout kPadded{false, false, true};

    // Part of a gtest name, e.g. "nat_col".
    std::string tag(Layout l)
    {
        return std::string(l.reversed ? "rev" : "nat") + (l.rowMajor ? "_row" : "_col")
               + (l.padded ? "_pad" : "");
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
        std::vector<int64_t> strides; // listed order, in elements
        int64_t              elements = 1;
        int64_t              span       = 1; // positions from the first element to the last
        int                  components = kReal;
        std::vector<float>   data; // host values; empty for the output D
    };

    // A tensor with the given layout and no data.
    Tensor
        makeTensor(std::vector<int32_t> modes, ModeMap const& ext, Layout l, int components = kReal)
    {
        if(l.reversed)
        {
            std::ranges::reverse(modes);
        }
        Tensor t;
        t.modes      = modes;
        t.components = components;
        for(auto m : modes)
        {
            t.lengths.push_back(ext.at(m));
        }
        t.strides.resize(modes.size());
        for(size_t j = 0; j < modes.size(); j++)
        {
            size_t i     = l.rowMajor ? modes.size() - 1 - j : j;
            t.strides[i] = t.elements * (l.padded ? 2 : 1);
            t.elements *= t.lengths[i];
            t.span += (t.lengths[i] - 1) * t.strides[i];
        }
        return t;
    }

    // An input tensor: the layout from makeTensor, filled with values in [-1, 1).
    Tensor makeInput(std::vector<int32_t> modes,
                     ModeMap const&       ext,
                     Layout               l,
                     unsigned             seed,
                     int                  components = kReal)
    {
        Tensor                                t = makeTensor(std::move(modes), ext, l, components);
        std::mt19937                          gen(seed);
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        t.data.resize(t.span * t.components);
        for(auto& x : t.data)
        {
            x = dist(gen);
        }
        return t;
    }

    // D = alpha * sum(A * B) (+ beta * C)
    //
    // With a third factor it is the trinary contraction, where the API calls the factors
    // A, B and C, the added tensor D and the result E. Here they are a, b, third, c and d.
    struct Contraction
    {
        std::complex<float>   alpha;
        std::complex<float>   beta;
        Tensor                a;
        Tensor                b;
        std::optional<Tensor> third; // present: trinary contraction
        std::optional<Tensor> c; // absent: scale contraction
        Tensor                d;
        bool                  inPlace = false; // C and D are one buffer
        std::optional<int64_t> cOffset; // C starts this many elements after D, in D's buffer
        bool                  negateC = false; // C is read through HIPTENSOR_OP_NEG
        hiptensorAlgo_t       algo    = HIPTENSOR_ALGO_DEFAULT;
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

    // The element of t at idx. A real element has a zero imaginary part.
    std::complex<double> element(Tensor const& t, ModeMap const& idx)
    {
        float const* x = &t.data[offset(t, idx) * t.components];
        return {x[0], t.components == kComplex ? x[1] : 0.f};
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

    // Adds to contracted the modes of t that D does not have.
    void addContracted(Tensor const& t, Tensor const& d, std::vector<int32_t>& contracted)
    {
        for(auto m : t.modes)
        {
            if(std::ranges::find(d.modes, m) == d.modes.end()
               && std::ranges::find(contracted, m) == contracted.end())
            {
                contracted.push_back(m);
            }
        }
    }

    // Independent reference: D = alpha * sum(A * B) (+ beta * C), computed from the mode
    // labels and the descriptor strides only, accumulated in double. One value per
    // position of D; a gap of a padded D holds NaN.
    std::vector<std::complex<double>> reference(Contraction const& p, ModeMap const& ext)
    {
        std::vector<int32_t> contracted;
        addContracted(p.a, p.d, contracted);
        addContracted(p.b, p.d, contracted);
        if(p.third)
        {
            addContracted(*p.third, p.d, contracted);
        }

        std::vector<std::complex<double>> out(p.d.span, std::numeric_limits<double>::quiet_NaN());
        ModeMap                           idx;
        forEachIndex(p.d.modes, ext, idx, [&] {
            std::complex<double> sum = 0.0;
            forEachIndex(contracted, ext, idx, [&] {
                auto product = element(p.a, idx) * element(p.b, idx);
                if(p.third)
                {
                    product *= element(*p.third, idx);
                }
                sum += product;
            });
            auto value = std::complex<double>(p.alpha) * sum;
            if(p.c)
            {
                value
                    += (p.negateC ? -1.0 : 1.0) * std::complex<double>(p.beta) * element(*p.c, idx);
            }
            out[offset(p.d, idx)] = value;
        });
        return out;
    }

    // Largest absolute error relative to the largest reference value (at least 1).
    // An element the kernel never wrote is still NaN and counts as infinitely wrong,
    // and so does a gap of a padded D that the kernel wrote to.
    double maxRelError(std::vector<float> const&                got,
                       int                                      components,
                       std::vector<std::complex<double>> const& ref)
    {
        double scale = 1.0;
        double err   = 0.0;
        for(auto const& x : ref)
        {
            if(!std::isnan(x.real()))
            {
                scale = std::max(scale, std::abs(x));
            }
        }
        for(size_t i = 0; i < ref.size(); i++)
        {
            float const*         x = &got[i * components];
            std::complex<double> value{x[0], components == kComplex ? x[1] : 0.f};
            bool                 gap = std::isnan(ref[i].real());
            if(gap != std::isnan(value.real()) || std::isnan(value.imag()))
            {
                return std::numeric_limits<double>::infinity();
            }
            if(!gap)
            {
                err = std::max(err, std::abs(value - ref[i]));
            }
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
                                                  t.components == kComplex ? HIPTENSOR_C_32F
                                                                           : HIPTENSOR_R_32F,
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
        auto   descT = p.third ? describe(h, *p.third) : nullptr;
        auto   descD = describe(h, p.d);
        auto   descC = p.c ? describe(h, *p.c) : nullptr;

        auto const compute
            = p.d.components == kComplex ? HIPTENSOR_COMPUTE_DESC_C32F : HIPTENSOR_COMPUTE_DESC_32F;
        auto const opC = p.negateC ? HIPTENSOR_OP_NEG : HIPTENSOR_OP_IDENTITY;

        hiptensorOperationDescriptor_t op   = nullptr;
        hiptensorPlanPreference_t      pref = nullptr;
        hiptensorPlan_t                plan = nullptr;
        uint64_t                       ws   = 0;
        void *dA = nullptr, *dB = nullptr, *dT = nullptr, *dC = nullptr, *dD = nullptr,
             *dW = nullptr;

        if(p.third)
        {
            r.status = hiptensorCreateContractionTrinary(h,
                                                         &op,
                                                         descA,
                                                         p.a.modes.data(),
                                                         HIPTENSOR_OP_IDENTITY,
                                                         descB,
                                                         p.b.modes.data(),
                                                         HIPTENSOR_OP_IDENTITY,
                                                         descT,
                                                         p.third->modes.data(),
                                                         HIPTENSOR_OP_IDENTITY,
                                                         descC,
                                                         p.c ? p.c->modes.data() : nullptr,
                                                         opC,
                                                         descD,
                                                         p.d.modes.data(),
                                                         compute);
        }
        else
        {
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
                                                  opC,
                                                  descD,
                                                  p.d.modes.data(),
                                                  compute);
        }
        if(r.status == HIPTENSOR_STATUS_SUCCESS)
        {
            r.status = hiptensorCreatePlanPreference(h, &pref, p.algo, HIPTENSOR_JIT_MODE_NONE);
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
            size_t const sizeD = p.d.span * p.d.components;
            upload(&dA, p.a.data);
            upload(&dB, p.b.data);
            if(p.third)
            {
                upload(&dT, p.third->data);
            }
            if(p.c && !p.inPlace && !p.cOffset)
            {
                upload(&dC, p.c->data);
            }
            // NaN in D: an element the kernel never writes cannot pass the check.
            // In place, D starts with C's values. With an offset, C follows NaN in D's buffer.
            std::vector<float> initialD(sizeD, std::numeric_limits<float>::quiet_NaN());
            if(p.inPlace)
            {
                initialD = p.c->data;
            }
            if(p.cOffset)
            {
                initialD.assign(*p.cOffset * p.d.components,
                                std::numeric_limits<float>::quiet_NaN());
                initialD.insert(initialD.end(), p.c->data.begin(), p.c->data.end());
            }
            upload(&dD, initialD);
            if(ws > 0)
            {
                EXPECT_EQ(hipMalloc(&dW, ws), hipSuccess);
            }

            void* added = p.inPlace ? dD : dC;
            if(p.cOffset)
            {
                added = static_cast<float*>(dD) + *p.cOffset * p.d.components;
            }
            if(p.third)
            {
                r.status = hiptensorContractTrinary(
                    h, plan, &p.alpha, dA, dB, dT, &p.beta, added, dD, dW, ws, nullptr);
            }
            else
            {
                r.status = hiptensorContract(
                    h, plan, &p.alpha, dA, dB, &p.beta, added, dD, dW, ws, nullptr);
            }
            EXPECT_EQ(hipDeviceSynchronize(), hipSuccess);

            r.D.resize(sizeD);
            EXPECT_EQ(hipMemcpy(r.D.data(), dD, sizeD * sizeof(float), hipMemcpyDeviceToHost),
                      hipSuccess);
        }

        for(void* ptr : {dA, dB, dT, dC, dD, dW})
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
        for(auto desc : {descA, descB, descT, descC, descD})
        {
            if(desc)
            {
                hiptensorDestroyTensorDescriptor(desc);
            }
        }
        return r;
    }

    // Runs the contraction and compares D with the reference.
    // A refused layout is a failure. The test skips only when the canonical layout is
    // refused with the same status: then the device has no kernel for the contraction.
    void expectCorrect(hiptensorHandle_t  h,
                       ModeMap const&     ext,
                       Contraction const& p,
                       Contraction const& canonical)
    {
        auto r = run(h, p);
        if((r.status == HIPTENSOR_STATUS_ARCH_MISMATCH
            || r.status == HIPTENSOR_STATUS_NOT_SUPPORTED)
           && run(h, canonical).status == r.status)
        {
            GTEST_SKIP() << "no kernel for this contraction on this device";
        }
        ASSERT_EQ(r.status, HIPTENSOR_STATUS_SUCCESS);
        double err = maxRelError(r.D, p.d.components, reference(p, ext));
        EXPECT_LE(err, kTolerance) << "max relative error " << err;
    }

    // Expects hipTensor to refuse the contraction with the given status. Skips when
    // the device cannot run this element type at all, which create reports first.
    void expectRefused(hiptensorHandle_t h, Contraction const& p, hiptensorStatus_t expected)
    {
        auto status = run(h, p).status;
        if(status == HIPTENSOR_STATUS_ARCH_MISMATCH)
        {
            GTEST_SKIP() << "no kernel for this element type on this device";
        }
        EXPECT_EQ(status, expected);
    }

    // How a test differs from the plain f32 contraction.
    struct Variant
    {
        int             components = kReal;
        hiptensorAlgo_t algo       = HIPTENSOR_ALGO_DEFAULT;
        bool            negateC    = false;
    };

    // The bilinear contraction of the layout tests, with one layout per tensor.
    Contraction bilinear(ModeMap const& ext, Layout a, Layout b, Layout c, Layout d, Variant v = {})
    {
        return Contraction{.alpha   = {1.5f, v.components == kComplex ? 0.25f : 0.f},
                           .beta    = {0.5f, v.components == kComplex ? -0.75f : 0.f},
                           .a       = makeInput({M0, M1, K0, K1}, ext, a, 1, v.components),
                           .b       = makeInput({N0, K0, K1}, ext, b, 2, v.components),
                           .c       = makeInput({M0, M1, N0}, ext, c, 3, v.components),
                           .d       = makeTensor({M0, M1, N0}, ext, d, v.components),
                           .negateC = v.negateC,
                           .algo    = v.algo};
    }

    // Positions of the tensors in the layout array of a trinary contraction, by API name.
    enum TrinaryTensor
    {
        kA,
        kB,
        kC,
        kD,
        kE,
        kAll // not a tensor: every tensor gets the layout
    };

    using TrinaryLayouts = std::array<Layout, 5>;

    constexpr TrinaryLayouts kTrinaryCanonical{
        kCanonical, kCanonical, kCanonical, kCanonical, kCanonical};

    // The canonical layouts with one tensor, or all of them, laid out as l.
    TrinaryLayouts trinaryLayouts(int which, Layout l)
    {
        TrinaryLayouts layouts = kTrinaryCanonical;
        for(int i = 0; i < 5; i++)
        {
            if(which == i || which == kAll)
            {
                layouts[i] = l;
            }
        }
        return layouts;
    }

    // The trinary contraction of the layout tests, with layouts in the order A, B, C, D, E.
    Contraction trinary(ModeMap const& ext, TrinaryLayouts const& l, Variant v = {})
    {
        return Contraction{.alpha   = {1.5f, v.components == kComplex ? 0.25f : 0.f},
                           .beta    = {0.5f, v.components == kComplex ? -0.75f : 0.f},
                           .a       = makeInput({M0, M1, K0}, ext, l[kA], 1, v.components),
                           .b       = makeInput({K0, K1}, ext, l[kB], 2, v.components),
                           .third   = makeInput({K1, N0}, ext, l[kC], 4, v.components),
                           .c       = makeInput({M0, M1, N0}, ext, l[kD], 3, v.components),
                           .d       = makeTensor({M0, M1, N0}, ext, l[kE], v.components),
                           .negateC = v.negateC,
                           .algo    = v.algo};
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

    // Trinary: the tensor that gets the layout (a TrinaryTensor; the others are
    // canonical); the layout; whether m1 has length 1.
    using TrinaryLayoutCase = std::tuple<int, Layout, bool>;

    class ContractionTrinaryLayoutTest : public ContractionTestBase,
                                         public ::testing::WithParamInterface<TrinaryLayoutCase>
    {
    };

    // The padded tensor of a binary contraction: 0 to 3 for A, B, C, D.
    class ContractionPaddedTest : public ContractionTestBase,
                                  public ::testing::WithParamInterface<int>
    {
    };

    // The checks on a contraction: what hipTensor must refuse, and what it must still accept.
    class ContractionCheckTest : public ContractionTestBase
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

    std::string trinaryLayoutCaseName(::testing::TestParamInfo<TrinaryLayoutCase> const& info)
    {
        constexpr char const* names[]    = {"A", "B", "C", "D", "E", "All"};
        auto const& [which, l, unitMode] = info.param;
        return std::string(names[which]) + "_" + tag(l) + "_" + sizeTag(unitMode);
    }

    std::string paddedCaseName(::testing::TestParamInfo<int> const& info)
    {
        constexpr char const* names[] = {"A", "B", "C", "D"};
        return names[info.param];
    }
} // namespace

TEST_P(ContractionLayoutTest, Scale)
{
    auto const [la, lb, ld, unitMode] = GetParam();
    auto const ext                    = extents(unitMode);
    auto const contraction            = [&](Layout a, Layout b, Layout d) {
        return Contraction{.alpha = 1.5f,
                                      .beta  = 0.f,
                                      .a     = makeInput({M0, M1, K0, K1}, ext, a, 1),
                                      .b     = makeInput({N0, K0, K1}, ext, b, 2),
                                      .d     = makeTensor({M0, M1, N0}, ext, d)};
    };
    expectCorrect(
        mHandle, ext, contraction(la, lb, ld), contraction(kCanonical, kCanonical, kCanonical));
}

// C is laid out like D.
TEST_P(ContractionLayoutTest, Bilinear)
{
    auto const [la, lb, ld, unitMode] = GetParam();
    auto const ext                    = extents(unitMode);
    auto const contraction            = [&](Layout a, Layout b, Layout d) {
        return Contraction{.alpha = 1.5f,
                                      .beta  = 0.5f,
                                      .a     = makeInput({M0, M1, K0, K1}, ext, a, 1),
                                      .b     = makeInput({N0, K0, K1}, ext, b, 2),
                                      .c     = makeInput({M0, M1, N0}, ext, d, 3),
                                      .d     = makeTensor({M0, M1, N0}, ext, d)};
    };
    expectCorrect(
        mHandle, ext, contraction(la, lb, ld), contraction(kCanonical, kCanonical, kCanonical));
}

// A unary operator on C selects the kernels that apply operators.
TEST_P(ContractionLayoutTest, BilinearUnary)
{
    auto const [la, lb, ld, unitMode] = GetParam();
    auto const    ext                 = extents(unitMode);
    Variant const v{.negateC = true};
    expectCorrect(mHandle,
                  ext,
                  bilinear(ext, la, lb, ld, ld, v),
                  bilinear(ext, kCanonical, kCanonical, kCanonical, kCanonical, v));
}

// The actor-critic kernel selection picks other kernels than the default one.
TEST_P(ContractionLayoutTest, BilinearActorCritic)
{
    auto const [la, lb, ld, unitMode] = GetParam();
    auto const    ext                 = extents(unitMode);
    Variant const v{.algo = HIPTENSOR_ALGO_ACTOR_CRITIC};
    expectCorrect(mHandle,
                  ext,
                  bilinear(ext, la, lb, ld, ld, v),
                  bilinear(ext, kCanonical, kCanonical, kCanonical, kCanonical, v));
}

// The complex path needs C laid out like D.
TEST_P(ContractionLayoutTest, BilinearComplex)
{
    auto const [la, lb, ld, unitMode] = GetParam();
    auto const    ext                 = extents(unitMode);
    Variant const v{.components = kComplex};
    expectCorrect(mHandle,
                  ext,
                  bilinear(ext, la, lb, ld, ld, v),
                  bilinear(ext, kCanonical, kCanonical, kCanonical, kCanonical, v));
}

// C laid out differently from D: develop reads C with D's strides.
TEST_P(ContractionCLayoutTest, Bilinear)
{
    auto const [lc, unitMode] = GetParam();
    auto const ext            = extents(unitMode);
    auto const contraction    = [&](Layout c) {
        return Contraction{.alpha = 1.5f,
                              .beta  = 0.5f,
                              .a     = makeInput({M0, M1, K0, K1}, ext, kCanonical, 1),
                              .b     = makeInput({N0, K0, K1}, ext, kCanonical, 2),
                              .c     = makeInput({M0, M1, N0}, ext, c, 3),
                              .d     = makeTensor({M0, M1, N0}, ext, kCanonical)};
    };
    expectCorrect(mHandle, ext, contraction(lc), contraction(kCanonical));
}

TEST_P(ContractionCLayoutTest, BilinearUnary)
{
    auto const [lc, unitMode] = GetParam();
    auto const    ext         = extents(unitMode);
    Variant const v{.negateC = true};
    expectCorrect(mHandle,
                  ext,
                  bilinear(ext, kCanonical, kCanonical, lc, kCanonical, v),
                  bilinear(ext, kCanonical, kCanonical, kCanonical, kCanonical, v));
}

TEST_P(ContractionCLayoutTest, BilinearActorCritic)
{
    auto const [lc, unitMode] = GetParam();
    auto const    ext         = extents(unitMode);
    Variant const v{.algo = HIPTENSOR_ALGO_ACTOR_CRITIC};
    expectCorrect(mHandle,
                  ext,
                  bilinear(ext, kCanonical, kCanonical, lc, kCanonical, v),
                  bilinear(ext, kCanonical, kCanonical, kCanonical, kCanonical, v));
}

// One padded tensor: its strides leave gaps, so no kernel may treat it as packed.
TEST_P(ContractionPaddedTest, Bilinear)
{
    auto const            ext = extents(false);
    std::array<Layout, 4> l{kCanonical, kCanonical, kCanonical, kCanonical};
    l[GetParam()] = kPadded;
    expectCorrect(mHandle,
                  ext,
                  bilinear(ext, l[0], l[1], l[2], l[3]),
                  bilinear(ext, kCanonical, kCanonical, kCanonical, kCanonical));
}

TEST_P(ContractionTrinaryLayoutTest, Bilinear)
{
    auto const [which, l, unitMode] = GetParam();
    auto const ext                  = extents(unitMode);
    expectCorrect(
        mHandle, ext, trinary(ext, trinaryLayouts(which, l)), trinary(ext, kTrinaryCanonical));
}

// The complex path needs D laid out like E, so D always gets E's layout here.
TEST_P(ContractionTrinaryLayoutTest, BilinearComplex)
{
    auto const [which, l, unitMode] = GetParam();
    auto const    ext               = extents(unitMode);
    Variant const v{.components = kComplex};
    auto          layouts = trinaryLayouts(which, l);
    layouts[kD]           = layouts[kE];
    expectCorrect(mHandle, ext, trinary(ext, layouts, v), trinary(ext, kTrinaryCanonical, v));
}

// C must have D's modes: here C lacks n.
TEST_F(ContractionCheckTest, CLacksMode)
{
    auto const  ext = extents(false);
    Contraction p{.alpha = 1.5f,
                  .beta  = 0.5f,
                  .a     = makeInput({M0, M1, K0, K1}, ext, kCanonical, 1),
                  .b     = makeInput({N0, K0, K1}, ext, kCanonical, 2),
                  .c     = makeInput({M0, M1}, ext, kCanonical, 3),
                  .d     = makeTensor({M0, M1, N0}, ext, kCanonical)};
    expectRefused(mHandle, p, HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// C must give each mode D's extent: here n is 4 in C and 5 in D.
TEST_F(ContractionCheckTest, CExtentDiffers)
{
    auto const ext  = extents(false);
    auto       extC = ext;
    extC[N0]        = 4;
    Contraction p{.alpha = 1.5f,
                  .beta  = 0.5f,
                  .a     = makeInput({M0, M1, K0, K1}, ext, kCanonical, 1),
                  .b     = makeInput({N0, K0, K1}, ext, kCanonical, 2),
                  .c     = makeInput({M0, M1, N0}, extC, kCanonical, 3),
                  .d     = makeTensor({M0, M1, N0}, ext, kCanonical)};
    expectRefused(mHandle, p, HIPTENSOR_STATUS_INVALID_VALUE);
}

// C and D in one buffer with the same layout: each element is read
// and written in place.
TEST_F(ContractionCheckTest, InPlaceSameLayout)
{
    auto const  ext = extents(false);
    Contraction p{.alpha   = 1.5f,
                  .beta    = 0.5f,
                  .a       = makeInput({M0, M1, K0, K1}, ext, kCanonical, 1),
                  .b       = makeInput({N0, K0, K1}, ext, kCanonical, 2),
                  .c       = makeInput({M0, M1, N0}, ext, kCanonical, 3),
                  .d       = makeTensor({M0, M1, N0}, ext, kCanonical),
                  .inPlace = true};
    // The skip reference is the same contraction out of place, so that a refused
    // in-place call fails the test.
    auto outOfPlace    = p;
    outOfPlace.inPlace = false;
    expectCorrect(mHandle, ext, p, outOfPlace);
}

// C and D in one buffer with different strides: the kernel would read elements
// of C that it has already overwritten.
TEST_F(ContractionCheckTest, InPlaceStridesDiffer)
{
    auto const  ext = extents(false);
    Contraction p{.alpha   = 1.5f,
                  .beta    = 0.5f,
                  .a       = makeInput({M0, M1, K0, K1}, ext, kCanonical, 1),
                  .b       = makeInput({N0, K0, K1}, ext, kCanonical, 2),
                  .c       = makeInput({M0, M1, N0}, ext, kCanonical, 3),
                  .d       = makeTensor({M0, M1, N0}, ext, Layout{false, true}),
                  .inPlace = true};
    expectRefused(mHandle, p, HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// The complex path does not use C's own layout.
TEST_F(ContractionCheckTest, ComplexCLayoutDiffers)
{
    auto const ext = extents(false);
    expectRefused(
        mHandle,
        bilinear(ext, kCanonical, kCanonical, kRowMajor, kCanonical, {.components = kComplex}),
        HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// The complex path sizes its buffers by the number of elements, so it cannot
// hold a tensor with gaps.
TEST_F(ContractionCheckTest, ComplexPadded)
{
    auto const ext = extents(false);
    for(int i = 0; i < 4; i++)
    {
        SCOPED_TRACE("padded tensor " + std::to_string(i) + " of A, B, C, D");
        std::array<Layout, 4> l{kCanonical, kCanonical, kCanonical, kCanonical};
        l[i] = kPadded;
        expectRefused(mHandle,
                      bilinear(ext, l[0], l[1], l[2], l[3], {.components = kComplex}),
                      HIPTENSOR_STATUS_NOT_SUPPORTED);
    }
}

// C lists its modes in reverse and is row-major, so each mode has D's stride:
// the same layout under another description.
TEST_F(ContractionCheckTest, InPlaceModesReordered)
{
    auto const ext = extents(false);
    auto       p   = bilinear(ext, kCanonical, kCanonical, Layout{true, true}, kCanonical);
    p.inPlace      = true;
    // The skip reference is the same contraction out of place, so that a refused
    // in-place call fails the test.
    auto outOfPlace    = p;
    outOfPlace.inPlace = false;
    expectCorrect(mHandle, ext, p, outOfPlace);
}

// The stride of a mode of extent 1 moves no position, so it may differ between
// C and D. Here C lists the unit mode p first: its stride is 1 in C and 3 in D.
TEST_F(ContractionCheckTest, InPlaceUnitModeStrideDiffers)
{
    auto const ext = extents(true);
    auto       p   = bilinear(ext, kCanonical, kCanonical, kCanonical, kCanonical);
    p.c            = makeInput({M1, M0, N0}, ext, kCanonical, 3);
    p.inPlace      = true;
    // The skip reference is the same contraction out of place, so that a refused
    // in-place call fails the test.
    auto outOfPlace    = p;
    outOfPlace.inPlace = false;
    expectCorrect(mHandle, ext, p, outOfPlace);
}

TEST_F(ContractionCheckTest, ComplexUnitModeStrideDiffers)
{
    auto const    ext = extents(true);
    Variant const v{.components = kComplex};
    auto const    canonical = bilinear(ext, kCanonical, kCanonical, kCanonical, kCanonical, v);
    auto          p         = canonical;
    p.c                     = makeInput({M1, M0, N0}, ext, kCanonical, 3, kComplex);
    expectCorrect(mHandle, ext, p, canonical);
}

// The complex path splits its tensors into planes. In place, it must read all
// of C before it writes D.
TEST_F(ContractionCheckTest, ComplexInPlaceSameLayout)
{
    auto const ext = extents(false);
    auto       p
        = bilinear(ext, kCanonical, kCanonical, kCanonical, kCanonical, {.components = kComplex});
    p.inPlace = true;
    // The skip reference is the same contraction out of place, so that a refused
    // in-place call fails the test.
    auto outOfPlace    = p;
    outOfPlace.inPlace = false;
    expectCorrect(mHandle, ext, p, outOfPlace);
}

// Two modes of A share stride 1, so elements overlap. The last element still
// sits at position "number of elements - 1", which must not pass as packed.
TEST_F(ContractionCheckTest, ComplexOverlappingStrides)
{
    auto const ext = extents(false);
    auto       p
        = bilinear(ext, kCanonical, kCanonical, kCanonical, kCanonical, {.components = kComplex});
    p.a.strides = {1, 1, 2, 32};
    expectRefused(mHandle, p, HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// C begins one element after D in the same buffer: the kernel would read
// elements of C that it has already overwritten.
TEST_F(ContractionCheckTest, COverlapsD)
{
    auto const ext = extents(false);
    auto       p   = bilinear(ext, kCanonical, kCanonical, kCanonical, kCanonical);
    p.cOffset      = 1;
    expectRefused(mHandle, p, HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// C begins where D ends: one buffer, but no shared element.
TEST_F(ContractionCheckTest, CFollowsD)
{
    auto const ext      = extents(false);
    auto const separate = bilinear(ext, kCanonical, kCanonical, kCanonical, kCanonical);
    auto       p        = separate;
    p.cOffset           = p.d.span;
    expectCorrect(mHandle, ext, p, separate);
}

// Trinary: D must have E's modes. Here D lacks n.
TEST_F(ContractionCheckTest, TrinaryDLacksMode)
{
    auto const ext = extents(false);
    auto       p   = trinary(ext, kTrinaryCanonical);
    p.c            = makeInput({M0, M1}, ext, kCanonical, 3);
    expectRefused(mHandle, p, HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// Trinary: D must give each mode E's extent. Here n is 4 in D and 5 in E.
TEST_F(ContractionCheckTest, TrinaryDExtentDiffers)
{
    auto const ext  = extents(false);
    auto       extD = ext;
    extD[N0]        = 4;
    auto p          = trinary(ext, kTrinaryCanonical);
    p.c             = makeInput({M0, M1, N0}, extD, kCanonical, 3);
    expectRefused(mHandle, p, HIPTENSOR_STATUS_INVALID_VALUE);
}

// Trinary: D and E in one buffer with the same layout.
TEST_F(ContractionCheckTest, TrinaryInPlaceSameLayout)
{
    auto const ext = extents(false);
    auto       p   = trinary(ext, kTrinaryCanonical);
    p.inPlace      = true;
    // The skip reference is the same contraction out of place, so that a refused
    // in-place call fails the test.
    auto outOfPlace    = p;
    outOfPlace.inPlace = false;
    expectCorrect(mHandle, ext, p, outOfPlace);
}

// Trinary: D and E in one buffer with different strides.
TEST_F(ContractionCheckTest, TrinaryInPlaceStridesDiffer)
{
    auto const ext = extents(false);
    auto       p   = trinary(ext, trinaryLayouts(kE, kRowMajor));
    p.inPlace      = true;
    expectRefused(mHandle, p, HIPTENSOR_STATUS_NOT_SUPPORTED);
}

TEST_F(ContractionCheckTest, TrinaryComplexInPlaceSameLayout)
{
    auto const ext = extents(false);
    auto       p   = trinary(ext, kTrinaryCanonical, {.components = kComplex});
    p.inPlace      = true;
    // The skip reference is the same contraction out of place, so that a refused
    // in-place call fails the test.
    auto outOfPlace    = p;
    outOfPlace.inPlace = false;
    expectCorrect(mHandle, ext, p, outOfPlace);
}

TEST_F(ContractionCheckTest, TrinaryDOverlapsE)
{
    auto const ext = extents(false);
    auto       p   = trinary(ext, kTrinaryCanonical);
    p.cOffset      = 1;
    expectRefused(mHandle, p, HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// Trinary: the complex path does not use D's own layout.
TEST_F(ContractionCheckTest, TrinaryComplexDLayoutDiffers)
{
    auto const ext = extents(false);
    expectRefused(mHandle,
                  trinary(ext, trinaryLayouts(kD, kRowMajor), {.components = kComplex}),
                  HIPTENSOR_STATUS_NOT_SUPPORTED);
}

// Trinary: the complex path cannot hold a tensor with gaps.
TEST_F(ContractionCheckTest, TrinaryComplexPadded)
{
    auto const ext = extents(false);
    for(int i = kA; i <= kE; i++)
    {
        SCOPED_TRACE("padded tensor " + std::to_string(i) + " of A, B, C, D, E");
        expectRefused(mHandle,
                      trinary(ext, trinaryLayouts(i, kPadded), {.components = kComplex}),
                      HIPTENSOR_STATUS_NOT_SUPPORTED);
    }
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

INSTANTIATE_TEST_SUITE_P(Layouts,
                         ContractionTrinaryLayoutTest,
                         ::testing::Combine(::testing::Range(0, 6),
                                            ::testing::ValuesIn(kLayouts),
                                            ::testing::Bool()),
                         trinaryLayoutCaseName);

INSTANTIATE_TEST_SUITE_P(Padded, ContractionPaddedTest, ::testing::Range(0, 4), paddedCaseName);
