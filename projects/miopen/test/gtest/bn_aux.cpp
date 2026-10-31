/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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

#include <fusionHost.hpp>
#include <miopen/batch_norm.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <gtest/gtest.h>
#include <array>
#include <cmath>
#include <vector>

namespace {

enum class BNDeriveMode
{
    Spatial,
    PerActivation
};

struct BNDeriveTestCase
{
    BNDeriveMode mode;
};

std::vector<BNDeriveTestCase> GetBNDeriveTestCases()
{
    return {{BNDeriveMode::Spatial}, {BNDeriveMode::PerActivation}};
}

struct GPU_BnDerive_FP32 : public ::testing::TestWithParam<BNDeriveTestCase>
{
    void SetUp() override
    {
        miopenCreateTensorDescriptor(&ctensor);
        miopenCreateTensorDescriptor(&derivedTensor);
        miopenSet4dTensorDescriptor(ctensor, miopenFloat, 100, 32, 8, 16);
    }

    void TearDown() override
    {
        miopenDestroyTensorDescriptor(ctensor);
        miopenDestroyTensorDescriptor(derivedTensor);
    }

    miopenTensorDescriptor_t ctensor{};
    miopenTensorDescriptor_t derivedTensor{};
};

struct CPU_BnSpatialHostBwd_NONE : public ::testing::TestWithParam<miopenTensorLayout_t>
{
    void SetUp() override
    {
        x  = tensor<double>{GetParam(), 2, 2, 1, 2};
        dy = tensor<double>{GetParam(), 2, 2, 1, 2};
        dx = tensor<double>{GetParam(), 2, 2, 1, 2};
        for(int c = 0; c < 2; ++c)
        {
            scale(0, c, 0, 0) = c == 0 ? 2.0 : -1.0;
            for(int n = 0; n < 2; ++n)
            {
                x(n, c, 0, 0) = 2 * c - 1;
                x(n, c, 0, 1) = 2 * c + 1;
            }
        }
    }

    void RunBackward(miopenActivationMode_t mode, double epsilon)
    {
        for(int c = 0; c < 2; ++c)
            for(int n = 0; n < 2; ++n)
                for(int w = 0; w < 2; ++w)
                    dy(n, c, 0, w) = 2 * n + w + 1;

        batchNormSpatialHostBwdTrain(
            x, dy, dx, scale, bias, dscale, dbias, epsilon, saved_mean, saved_inv_var, mode, 0, 0);
    }

    void ExpectChannel(int c,
                       const std::array<double, 4>& expected_dy,
                       const std::array<double, 4>& expected_dx,
                       double expected_dscale,
                       double expected_dbias)
    {
        SCOPED_TRACE(c);
        constexpr double tolerance = 1e-12;
        EXPECT_NEAR(dscale(0, c, 0, 0), expected_dscale, tolerance);
        EXPECT_NEAR(dbias(0, c, 0, 0), expected_dbias, tolerance);
        for(int i = 0; i < 4; ++i)
        {
            SCOPED_TRACE(i);
            EXPECT_NEAR(dy(i / 2, c, 0, i % 2), expected_dy[i], tolerance);
            EXPECT_NEAR(dx(i / 2, c, 0, i % 2), expected_dx[i], tolerance);
        }
    }

    tensor<double> x;
    tensor<double> dy;
    tensor<double> dx;
    tensor<double> scale{1, 2, 1, 1};
    tensor<double> bias{1, 2, 1, 1};
    tensor<double> dscale{1, 2, 1, 1};
    tensor<double> dbias{1, 2, 1, 1};
    tensor<double> saved_mean;
    tensor<double> saved_inv_var;
};

} // namespace

TEST_P(GPU_BnDerive_FP32, Test)
{
    const auto& test_case = this->GetParam();
    std::array<int, 4> lens{};
    miopenDataType_t dt;

    miopenBatchNormMode_t bn_mode =
        (test_case.mode == BNDeriveMode::Spatial) ? miopenBNSpatial : miopenBNPerActivation;

    miopenDeriveBNTensorDescriptor(derivedTensor, ctensor, bn_mode);
    miopenGetTensorDescriptor(derivedTensor, &dt, lens.data(), nullptr);

    EXPECT_EQ(dt, miopenFloat);
    EXPECT_EQ(lens.size(), 4);
    EXPECT_EQ(lens[0], 1);
    EXPECT_EQ(lens[1], 32);

    if(test_case.mode == BNDeriveMode::Spatial)
    {
        EXPECT_EQ(lens[2], 1);
        EXPECT_EQ(lens[3], 1);
    }
    else
    {
        EXPECT_EQ(lens[2], 8);
        EXPECT_EQ(lens[3], 16);
    }
}

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_BnDerive_FP32, testing::ValuesIn(GetBNDeriveTestCases()));

TEST_P(CPU_BnSpatialHostBwd_NONE, RecomputedPassthroughIncludesEpsilon)
{
    // Each channel has x = mean + [-1, 1, -1, 1], so population variance is 1.
    // For dy = [1, 2, 3, 4], dscale = 2r and dbias = 10, with r = 1/sqrt(1+eps).
    // Differentiating the two-valued input gives dx below; e is the epsilon correction.
    for(double epsilon : {0.25, 1e-3})
    {
        SCOPED_TRACE(epsilon);
        RunBackward(miopenActivationPASTHRU, epsilon);
        const double r = 1 / std::sqrt(1 + epsilon);
        const double e = epsilon / (1 + epsilon);
        for(int c = 0; c < 2; ++c)
        {
            const double s = scale(0, c, 0, 0) * r;
            ExpectChannel(c,
                          {1, 2, 3, 4},
                          {s * (-1 - e / 2), s * (-1 + e / 2), s * (1 - e / 2), s * (1 + e / 2)},
                          2 * r,
                          10);
        }
    }
}

TEST_P(CPU_BnSpatialHostBwd_NONE, RecomputedSoftReluIncludesEpsilon)
{
    // SoftReLU'(z) = sigmoid(z). The BN outputs are [-scale*r, scale*r] twice.
    // Writing q = sigmoid(scale*r), the activation gradients are [1-q, 2q, 3(1-q), 4q].
    // Their signed sum is b = 10q-4, hence dscale = r*b and dbias = 4+2q.
    // Closed-form dx separates the within-sign differences from the epsilon correction.
    for(double epsilon : {0.25, 1e-3})
    {
        SCOPED_TRACE(epsilon);
        RunBackward(miopenActivationSOFTRELU, epsilon);
        const double r = 1 / std::sqrt(1 + epsilon);
        const double e = epsilon / (1 + epsilon);
        for(int c = 0; c < 2; ++c)
        {
            const double s          = scale(0, c, 0, 0) * r;
            const double q          = 1 / (1 + std::exp(-s));
            const double p          = 1 - q;
            const double b          = 10 * q - 4;
            const double correction = e * b / 4;
            ExpectChannel(c,
                          {p, 2 * q, 3 * p, 4 * q},
                          {s * (-p - correction),
                           s * (-q + correction),
                           s * (p - correction),
                           s * (q + correction)},
                          r * b,
                          4 + 2 * q);
        }
    }
}

TEST_P(CPU_BnSpatialHostBwd_NONE, RecomputedZeroVarianceStaysFinite)
{
    for(int c = 0; c < 2; ++c)
        for(int n = 0; n < 2; ++n)
            for(int w = 0; w < 2; ++w)
                x(n, c, 0, w) = 2 * c + 1;

    // xhat is zero; only centering dy and multiplying by scale/sqrt(eps) remain.
    // SoftReLU's derivative at the zero BN output is exactly 1/2.
    for(double epsilon : {0.25, 1e-3})
    {
        SCOPED_TRACE(epsilon);
        for(auto mode : {miopenActivationPASTHRU, miopenActivationSOFTRELU})
        {
            SCOPED_TRACE(mode);
            RunBackward(mode, epsilon);
            const double a = mode == miopenActivationPASTHRU ? 1.0 : 0.5;
            for(int c = 0; c < 2; ++c)
            {
                const double s = a * scale(0, c, 0, 0) / std::sqrt(epsilon);
                ExpectChannel(
                    c, {a, 2 * a, 3 * a, 4 * a}, {-1.5 * s, -0.5 * s, 0.5 * s, 1.5 * s}, 0, 10 * a);
            }
        }
    }
}

TEST_P(CPU_BnSpatialHostBwd_NONE, SavedStatisticsOverrideRecomputation)
{
    saved_mean    = tensor<double>{1, 2, 1, 1};
    saved_inv_var = tensor<double>{1, 2, 1, 1};
    for(int c = 0; c < 2; ++c)
    {
        // Both statistics deliberately differ from the input's mean and inverse stddev.
        saved_mean(0, c, 0, 0)    = 2 * c + 0.5;
        saved_inv_var(0, c, 0, 0) = 0.5;
    }

    // Supplied statistics fix xhat = [-3/4, 1/4, -3/4, 1/4], independent of epsilon.
    // With activation derivatives a and b at the two BN outputs, dbias = 4a+6b
    // and dscale = -3a+3b/2. Substitution of these fixed rational xhat values
    // gives the four dx values below, including the non-centered supplied mean.
    for(double epsilon : {0.25, 1e-3})
    {
        SCOPED_TRACE(epsilon);
        for(auto mode : {miopenActivationPASTHRU, miopenActivationSOFTRELU})
        {
            SCOPED_TRACE(mode);
            RunBackward(mode, epsilon);
            for(int c = 0; c < 2; ++c)
            {
                const double scale_value = scale(0, c, 0, 0);
                const double a =
                    mode == miopenActivationPASTHRU ? 1.0 : 1 / (1 + std::exp(0.75 * scale_value));
                const double b =
                    mode == miopenActivationPASTHRU ? 1.0 : 1 / (1 + std::exp(-0.25 * scale_value));
                const double ds = -3 * a + 1.5 * b;
                const double s  = scale_value / 2;
                ExpectChannel(c,
                              {a, 2 * b, 3 * a, 4 * b},
                              {s * (-1.5 * b + 3 * ds / 16),
                               s * (-a + 0.5 * b - ds / 16),
                               s * (2 * a - 1.5 * b + 3 * ds / 16),
                               s * (-a + 2.5 * b - ds / 16)},
                              ds,
                              4 * a + 6 * b);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         CPU_BnSpatialHostBwd_NONE,
                         testing::Values(miopenTensorNCHW, miopenTensorNHWC),
                         [](const testing::TestParamInfo<miopenTensorLayout_t>& info) {
                             return info.param == miopenTensorNCHW ? "NCHW" : "NHWC";
                         });
