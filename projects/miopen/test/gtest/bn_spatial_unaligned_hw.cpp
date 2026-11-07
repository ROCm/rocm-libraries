/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

// Regression test for NCHW spatial batch norm with H*W not a multiple of the
// kernel vector width (variant 1). The float4 paths read 4 consecutive elements
// of the flattened N*H*W range; when H*W % 4 != 0 a read near the end of a
// channel row crosses into the next channel. The existing BN tests miss this
// because (a) most shapes have H*W % 4 == 0 or C == 1, and (b) every channel is
// drawn from the same distribution, so mixing in a neighbour's elements changes
// the result only by ~1e-3, below the 4e-3 tolerance. Here odd channels get a
// large offset, so any cross-channel read is a gross error.
//
// Shapes (NCHW, fp32 scale/bias):
//  - fwd training, HW >= 4096, HW % 4 != 0           -> FwdTrain variant 1 float4 stats path
//  - bwd, saved stats, 60 < HW < 200, fp16 input      -> Bwd variant 1 (fixed by #6765)
//  - bwd, saved stats, HW > 1024, HW % 4 != 0         -> Bwd variant 1 (fixed by #6765)
//  - bwd, no saved stats, HW >= 4096, HW % 4 != 0     -> Bwd variant 1 CalcStats float4 path

#include <gtest/gtest.h>
#include <hip/hip_runtime_api.h>
#include <half/half.hpp>
#include <miopen/miopen.h>

#include <cmath>
#include <random>
#include <vector>

#include "get_handle.hpp"

namespace {

struct UnalignedCase
{
    int n, c, hw;
    bool half_input;
    bool backward;
    bool use_saved;
};

std::ostream& operator<<(std::ostream& os, const UnalignedCase& p)
{
    return os << "n" << p.n << "c" << p.c << "hw" << p.hw << (p.half_input ? "_fp16" : "_fp32")
              << (p.backward ? "_bwd" : "_fwd") << (p.use_saved ? "_saved" : "_recalc");
}

template <typename T>
struct DevBuf
{
    T* p = nullptr;
    explicit DevBuf(const std::vector<T>& h)
    {
        EXPECT_EQ(hipMalloc(reinterpret_cast<void**>(&p), h.size() * sizeof(T)), hipSuccess);
        EXPECT_EQ(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), hipSuccess);
    }
    std::vector<T> get(size_t n) const
    {
        std::vector<T> h(n);
        EXPECT_EQ(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), hipSuccess);
        return h;
    }
    ~DevBuf() { (void)hipFree(p); }
};

double RelErr(const std::vector<double>& got, const std::vector<double>& ref)
{
    double num = 0, den = 0;
    for(size_t i = 0; i < ref.size(); ++i)
    {
        num += (got[i] - ref[i]) * (got[i] - ref[i]);
        den += ref[i] * ref[i];
    }
    return std::sqrt(num / den);
}

template <typename TIn>
void Run(const UnalignedCase& p)
{
    auto&& handle = get_handle();
    const int n = p.n, c = p.c, hw = p.hw;
    const size_t total = static_cast<size_t>(n) * c * hw;
    const double eps   = 1e-5;
    const double M     = static_cast<double>(n) * hw;

    std::mt19937 gen(42);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<TIn> x(total), dy(total);
    std::vector<double> xd(total), dyd(total);
    for(int in = 0; in < n; ++in)
        for(int ic = 0; ic < c; ++ic)
            for(int k = 0; k < hw; ++k)
            {
                const size_t i = (static_cast<size_t>(in) * c + ic) * hw + k;
                x[i]           = static_cast<TIn>(nd(gen) + (ic % 2) * 100.0);
                dy[i]          = static_cast<TIn>(nd(gen));
                xd[i]          = static_cast<double>(x[i]);
                dyd[i]         = static_cast<double>(dy[i]);
            }
    std::vector<float> scale(c), bias(c, 0.f), zeros(c, 0.f), ones(c, 1.f);
    for(int ic = 0; ic < c; ++ic)
        scale[ic] = 0.5f + static_cast<float>(ic) / c;

    // fp64 reference
    std::vector<double> mean(c), inv(c), dscale(c), dbias(c), y(total), dx(total);
    for(int ic = 0; ic < c; ++ic)
    {
        double s = 0, s2 = 0;
        for(int in = 0; in < n; ++in)
            for(int k = 0; k < hw; ++k)
                s += xd[(static_cast<size_t>(in) * c + ic) * hw + k];
        mean[ic] = s / M;
        for(int in = 0; in < n; ++in)
            for(int k = 0; k < hw; ++k)
            {
                const double d = xd[(static_cast<size_t>(in) * c + ic) * hw + k] - mean[ic];
                s2 += d * d;
            }
        inv[ic]   = 1.0 / std::sqrt(s2 / M + eps);
        double db = 0, ds = 0;
        for(int in = 0; in < n; ++in)
            for(int k = 0; k < hw; ++k)
            {
                const size_t i  = (static_cast<size_t>(in) * c + ic) * hw + k;
                const double xh = (xd[i] - mean[ic]) * inv[ic];
                y[i]            = scale[ic] * xh + bias[ic];
                db += dyd[i];
                ds += dyd[i] * xh;
            }
        dbias[ic]  = db;
        dscale[ic] = ds;
        for(int in = 0; in < n; ++in)
            for(int k = 0; k < hw; ++k)
            {
                const size_t i  = (static_cast<size_t>(in) * c + ic) * hw + k;
                const double xh = (xd[i] - mean[ic]) * inv[ic];
                dx[i]           = scale[ic] * inv[ic] / M * (M * dyd[i] - db - xh * ds);
            }
    }

    const miopenDataType_t dt = p.half_input ? miopenHalf : miopenFloat;
    miopenTensorDescriptor_t xdesc, wdesc;
    ASSERT_EQ(miopenCreateTensorDescriptor(&xdesc), miopenStatusSuccess);
    ASSERT_EQ(miopenCreateTensorDescriptor(&wdesc), miopenStatusSuccess);
    ASSERT_EQ(miopenSet4dTensorDescriptor(xdesc, dt, n, c, hw, 1), miopenStatusSuccess);
    ASSERT_EQ(miopenSet4dTensorDescriptor(wdesc, miopenFloat, 1, c, 1, 1), miopenStatusSuccess);

    DevBuf<TIn> x_d(x), dy_d(dy), y_d(x), dx_d(x);
    DevBuf<float> sc_d(scale), bi_d(bias), rm_d(zeros), rv_d(ones), sm_d(zeros), si_d(zeros),
        ds_d(zeros), db_d(zeros);
    float alpha = 1.f, beta = 0.f;

    // Forward (also produces the saved statistics used by the backward "saved" cases).
    ASSERT_EQ(miopenBatchNormalizationForwardTraining(&handle,
                                                      miopenBNSpatial,
                                                      &alpha,
                                                      &beta,
                                                      xdesc,
                                                      x_d.p,
                                                      xdesc,
                                                      y_d.p,
                                                      wdesc,
                                                      sc_d.p,
                                                      bi_d.p,
                                                      1.0,
                                                      rm_d.p,
                                                      rv_d.p,
                                                      eps,
                                                      sm_d.p,
                                                      si_d.p),
              miopenStatusSuccess);

    auto to_d = [](const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); };
    if(!p.backward)
    {
        const auto ym = y_d.get(total);
        std::vector<double> yg(ym.begin(), ym.end());
        EXPECT_LT(RelErr(to_d(sm_d.get(c)), mean), 1e-5) << "saved mean";
        EXPECT_LT(RelErr(to_d(si_d.get(c)), inv), 1e-4) << "saved inv variance";
        EXPECT_LT(RelErr(yg, y), p.half_input ? 2e-3 : 1e-4) << "y";
    }
    else
    {
        // Exact saved statistics so the backward check is independent of the forward kernel.
        std::vector<float> mf(mean.begin(), mean.end()), vf(inv.begin(), inv.end());
        DevBuf<float> m_d(mf), v_d(vf);
        ASSERT_EQ(miopenBatchNormalizationBackward(&handle,
                                                   miopenBNSpatial,
                                                   &alpha,
                                                   &beta,
                                                   &alpha,
                                                   &beta,
                                                   xdesc,
                                                   x_d.p,
                                                   xdesc,
                                                   dy_d.p,
                                                   xdesc,
                                                   dx_d.p,
                                                   wdesc,
                                                   sc_d.p,
                                                   ds_d.p,
                                                   db_d.p,
                                                   eps,
                                                   p.use_saved ? m_d.p : nullptr,
                                                   p.use_saved ? v_d.p : nullptr),
                  miopenStatusSuccess);
        const auto dxm = dx_d.get(total);
        std::vector<double> dxg(dxm.begin(), dxm.end());
        EXPECT_LT(RelErr(to_d(ds_d.get(c)), dscale), 1e-3) << "dscale";
        EXPECT_LT(RelErr(to_d(db_d.get(c)), dbias), 1e-4) << "dbias";
        EXPECT_LT(RelErr(dxg, dx), p.half_input ? 2e-3 : 1e-3) << "dx";
    }
    miopenDestroyTensorDescriptor(xdesc);
    miopenDestroyTensorDescriptor(wdesc);
}

} // namespace

struct GPU_BNSpatialUnalignedHW : testing::TestWithParam<UnalignedCase>
{
};

TEST_P(GPU_BNSpatialUnalignedHW, MatchesFp64Reference)
{
    const auto p = GetParam();
    if(p.half_input)
        Run<half_float::half>(p);
    else
        Run<float>(p);
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_BNSpatialUnalignedHW,
                         testing::Values(
                             // forward training, variant 1 float4 stats path
                             UnalignedCase{2, 8, 4096, false, false, true}, // aligned control
                             UnalignedCase{2, 8, 4097, false, false, true},
                             UnalignedCase{2, 8, 4098, false, false, true},
                             UnalignedCase{2, 8, 65 * 65, true, false, true},
                             // backward with saved stats, variant 1 (fp16, 60 < HW < 200)
                             UnalignedCase{4, 64, 148, true, true, true}, // aligned control
                             UnalignedCase{4, 64, 150, true, true, true},
                             UnalignedCase{4, 64, 151, true, true, true},
                             // backward with saved stats, variant 1 (HW > 1024)
                             UnalignedCase{2, 16, 1030, false, true, true},
                             // backward recomputing stats, variant 1 float4 CalcStats path
                             UnalignedCase{2, 8, 4100, false, true, false}, // aligned control
                             UnalignedCase{2, 8, 4097, false, true, false},
                             UnalignedCase{2, 8, 4098, false, true, false}));
