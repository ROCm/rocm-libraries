// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipdnn_data_sdk/utilities/Constants.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceLayernorm.hpp>
#include <hipdnn_test_sdk/utilities/TestTolerances.hpp>
#include <hipdnn_test_sdk/utilities/detail/CpuFpReferenceUtilities.hpp>

using namespace hipdnn_test_sdk::utilities;
using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;
using hipdnn_test_sdk::detail::safeTestTypeCast;

// ============================================================================
// Hand-computed golden reference tests
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, BpropSanityValidation2D)
{
    // Input shape: [2, 4] — normalize over last dim (4 features per sample)
    // dy = [[-1, 2, -3, 4],
    //       [2, -4, 6, -8]]
    // x  = [[1, 2, 3, 4],
    //       [2, 4, 6, 8]]
    //
    // Sample 0: mean=2.5, var=1.25, rstd=1/sqrt(1.25+1e-5)=0.894423613312618
    //   a      = 1.7888329159619083
    //   b      = -3.5776586765921525
    //   dx     = [-4.7998926705705713e-05, 3.577687297918808, -7.15538175116928, 3.5777159192454633]
    //
    // Sample 1: mean=5.0, var=5.0, rstd=1/sqrt(5.0+1e-5)=0.4472131482870333
    //   a      = -0.8944245077250512
    //   b      = 3.5776962420511893
    //   dx     = [2.3999731674440028e-05, -3.5777033974472507, 7.155408583743517, -3.577710552843312]
    //
    // dscale = [-1.341640786446209, 0.8944271909641394, 1.341640786446209, -5.366563145784836]
    // dbias  = [1, -2, 3, -4]

    Tensor<double> dy({2, 4});
    Tensor<double> x({2, 4});
    Tensor<double> scale({4});
    Tensor<double> mean({2});
    Tensor<double> rstd({2});
    Tensor<double> dx({2, 4});
    Tensor<double> dscale({4});
    Tensor<double> dbias({4});

    dy.setHostValue(-1.0, 0, 0);
    dy.setHostValue(2.0, 0, 1);
    dy.setHostValue(-3.0, 0, 2);
    dy.setHostValue(4.0, 0, 3);
    dy.setHostValue(2.0, 1, 0);
    dy.setHostValue(-4.0, 1, 1);
    dy.setHostValue(6.0, 1, 2);
    dy.setHostValue(-8.0, 1, 3);

    x.setHostValue(1.0, 0, 0);
    x.setHostValue(2.0, 0, 1);
    x.setHostValue(3.0, 0, 2);
    x.setHostValue(4.0, 0, 3);
    x.setHostValue(2.0, 1, 0);
    x.setHostValue(4.0, 1, 1);
    x.setHostValue(6.0, 1, 2);
    x.setHostValue(8.0, 1, 3);

    mean.setHostValue(2.5, 0);
    mean.setHostValue(5.0, 1);

    rstd.setHostValue(1.0 / std::sqrt(1.25 + LAYERNORM_DEFAULT_EPSILON), 0);
    rstd.setHostValue(1.0 / std::sqrt(5.0 + LAYERNORM_DEFAULT_EPSILON), 1);

    scale.fillWithValue(2.0);

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1);

    auto tolerance = layernorm::getTolerance<double>();

    // Sample 0 outputs
    EXPECT_NEAR(dx.getHostValue(0, 0), -2.1465994991753945e-05, tolerance);
    EXPECT_NEAR(dx.getHostValue(0, 1), 3.577687297918808, tolerance);
    EXPECT_NEAR(dx.getHostValue(0, 2), -7.15538175116928, tolerance);
    EXPECT_NEAR(dx.getHostValue(0, 3), 3.5777159192454633, tolerance);

    // Sample 1 outputs
    EXPECT_NEAR(dx.getHostValue(1, 0), 5.366547046303793e-06, tolerance);
    EXPECT_NEAR(dx.getHostValue(1, 1), -3.5777033974472507, tolerance);
    EXPECT_NEAR(dx.getHostValue(1, 2), 7.155408583743517, tolerance);
    EXPECT_NEAR(dx.getHostValue(1, 3), -3.577710552843312, tolerance);

    // dscale
    EXPECT_NEAR(dscale.getHostValue(0), -1.3416434697532726, tolerance);
    EXPECT_NEAR(dscale.getHostValue(1), 0.8944289798355152, tolerance);
    EXPECT_NEAR(dscale.getHostValue(2), 1.3416434697532726, tolerance);
    EXPECT_NEAR(dscale.getHostValue(3), -5.366573879013091, tolerance);

    // dbias
    EXPECT_NEAR(dbias.getHostValue(0), 1.0, tolerance);
    EXPECT_NEAR(dbias.getHostValue(1), -2.0, tolerance);
    EXPECT_NEAR(dbias.getHostValue(2), 3.0, tolerance);
    EXPECT_NEAR(dbias.getHostValue(3), -4.0, tolerance);
}

TEST(TestCpuFpReferenceLayernormBackwardFp64, BpropSanityValidation3D)
{
    // Input shape: [1, 2, 3] — normalize over last 2 dims (2x3 = 6 features)
    // dy = [[[-1, 2, -3], [4, -5, 6]]]
    // x  = [[[1, 2, 3], [4, 5, 6]]]
    //
    // mean = (1+2+3+4+5+6)/6 = 3.5
    // var = ((1-3.5)^2 + (2-3.5)^2 + (3-3.5)^2 + (4-3.5)^2 + (5-3.5)^2 + (6-3.5)^2) / 6
    //     = (6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25) / 6 = 17.5 / 6 = 2.9166666666666665
    // rstd = 1/sqrt(2.9166666666666665 + 1e-5) = 0.5855390399887689
    //
    // With scale=1, bias=0 (identity affine):
    //   a      = rstd³ * (sum(dy * scale * x) - sum(dy * scale) * mean) / 6
    //   b      = rstd * sum(dy * scale) / 6 - a * mean
    //   dx     = rstd * dy * scale - a * x - b
    //   dscale = sum(dy * (x - mean) * rstd)
    //   dbias  = sum(dy)

    Tensor<double> dy({1, 2, 3});
    Tensor<double> x({1, 2, 3});
    Tensor<double> scale({2, 3});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});
    Tensor<double> dx({1, 2, 3});
    Tensor<double> dscale({2, 3});
    Tensor<double> dbias({2, 3});

    dy.setHostValue(-1.0, 0, 0, 0);
    dy.setHostValue(2.0, 0, 0, 1);
    dy.setHostValue(-3.0, 0, 0, 2);
    dy.setHostValue(4.0, 0, 1, 0);
    dy.setHostValue(-5.0, 0, 1, 1);
    dy.setHostValue(6.0, 0, 1, 2);

    x.setHostValue(1.0, 0, 0, 0);
    x.setHostValue(2.0, 0, 0, 1);
    x.setHostValue(3.0, 0, 0, 2);
    x.setHostValue(4.0, 0, 1, 0);
    x.setHostValue(5.0, 0, 1, 1);
    x.setHostValue(6.0, 0, 1, 2);

    scale.fillWithValue(1.0);

    mean.setHostValue(3.5, 0);
    rstd.setHostValue(0.5855390399887689, 0);

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 2);

    Tensor<double> dxRef({1, 2, 3});
    Tensor<double> dscaleRef({2, 3});
    Tensor<double> dbiasRef({2, 3});

    dxRef.setHostValue(-3.0113333097103734e-06, 0, 0, 0);
    dxRef.setHostValue(1.4052918891730595, 0, 0, 1);
    dxRef.setHostValue(-1.8737255302307223, 0, 0, 2);
    dxRef.setHostValue(1.8737255302307223, 0, 1, 0);
    dxRef.setHostValue(-3.7474480491281357, 0, 1, 1);
    dxRef.setHostValue(2.342159171288385, 0, 1, 2);

    dscaleRef.setHostValue(1.4638475999719223, 0, 0);
    dscaleRef.setHostValue(-1.7566171199663065, 0, 1);
    dscaleRef.setHostValue(0.8783085599831533, 0, 2);
    dscaleRef.setHostValue(1.1710780799775378, 1, 0);
    dscaleRef.setHostValue(-4.391542799915767, 1, 1);
    dscaleRef.setHostValue(8.783085599831534, 1, 2);

    dbiasRef.setHostValue(-1.0, 0, 0);
    dbiasRef.setHostValue(2.0, 0, 1);
    dbiasRef.setHostValue(-3.0, 0, 2);
    dbiasRef.setHostValue(4.0, 1, 0);
    dbiasRef.setHostValue(-5.0, 1, 1);
    dbiasRef.setHostValue(6.0, 1, 2);

    auto tolerance = layernorm::getTolerance<double>();

    // Verify each output element: y = (x - mean) * rstd
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            EXPECT_NEAR(dx.getHostValue(0, i, j), dxRef.getHostValue(0, i, j), tolerance);
            EXPECT_NEAR(dscale.getHostValue(i, j), dscaleRef.getHostValue(i, j), tolerance);
            EXPECT_NEAR(dbias.getHostValue(i, j), dbiasRef.getHostValue(i, j), tolerance);
        }
    }
}

// ============================================================================
// Corner case: all zeros input
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, BpropAllZeros)
{
    // All-zero input: mean=0, var=0, rstd=1/sqrt(epsilon)
    //   a      = rstd³ * (sum(dy * scale * x) - sum(dy * scale) * mean) / 4 = 0
    //   b      = rstd * sum(dy * scale) / 4 - a * mean = 0
    //   dx     = rstd * dy * scale - a * x - b = 0
    //   dscale = sum(dy * (x - mean) * rstd) = 0
    //   dbias  = sum(dy) = 0

    Tensor<double> dy({2, 4});
    Tensor<double> x({2, 4});
    Tensor<double> scale({4});
    Tensor<double> mean({2});
    Tensor<double> rstd({2});
    Tensor<double> dx({2, 4});
    Tensor<double> dscale({4});
    Tensor<double> dbias({4});

    dy.fillWithValue(0.0);
    x.fillWithValue(0.0);
    mean.fillWithValue(0.0);
    rstd.fillWithValue(1.0 / std::sqrt(LAYERNORM_DEFAULT_EPSILON));

    for(int i = 0; i < 4; i++)
    {
        scale.setHostValue(2.0, i);
    }

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1);

    auto tolerance = layernorm::getTolerance<double>();

    for(int f = 0; f < 4; f++)
    {
        EXPECT_NEAR(dscale.getHostValue(f), 0.0, tolerance);
        EXPECT_NEAR(dbias.getHostValue(f), 0.0, tolerance);
        for(int b = 0; b < 2; b++)
        {
            EXPECT_NEAR(dx.getHostValue(b, f), 0.0, tolerance);
        }
    }
}

// ============================================================================
// Corner case: all ones input
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, BpropAllOnes)
{
    // All-one input: mean=1, var=0, rstd=1/sqrt(epsilon)
    //   a      = rstd³ * (sum(dy * scale * x) - sum(dy * scale) * mean) / 5 = 0
    //   b      = rstd * sum(dy * scale) / 5 - a * mean = sum(scale) / std::sqrt(epsilon) / 5 = 1.5 / std::sqrt(epsilon)
    //   dx     = rstd * dy * scale - a * x - b = scale / std::sqrt(epsilon) - sum(scale) / std::sqrt(epsilon) / 5 = 0
    //   dscale = sum(dy * (x - mean) * rstd) = 0
    //   dbias  = sum(dy) = 3

    Tensor<double> dy({3, 5});
    Tensor<double> x({3, 5});
    Tensor<double> scale({5});
    Tensor<double> mean({3});
    Tensor<double> rstd({3});
    Tensor<double> dx({3, 5});
    Tensor<double> dscale({5});
    Tensor<double> dbias({5});

    dy.fillWithValue(1.0);
    x.fillWithValue(1.0);
    scale.fillWithValue(1.5);
    mean.fillWithValue(1.0);
    rstd.fillWithValue(1.0 / std::sqrt(LAYERNORM_DEFAULT_EPSILON));

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1);

    auto tolerance = layernorm::getTolerance<double>();

    for(int f = 0; f < 5; f++)
    {
        EXPECT_NEAR(dscale.getHostValue(f), 0.0, tolerance);
        EXPECT_NEAR(dbias.getHostValue(f), 3.0, tolerance);
        for(int b = 0; b < 3; b++)
        {
            EXPECT_NEAR(dx.getHostValue(b, f), 0.0, tolerance);
        }
    }
}

// ============================================================================
// Corner case: constant input (all same non-trivial value)
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, BpropConstantInput)
{
    // All elements = 7.0, scale = 1.0: mean=7, var=0
    //   a      = rstd³ * (sum(dy * scale * x) - sum(dy * scale) * mean) / 3 = 0
    //   b      = rstd * sum(dy * scale) / 3 - a * mean = sum(dy) / std::sqrt(epsilon) / 3 = 7.0 / std::sqrt(epsilon)
    //   dx     = rstd * dy * scale - a * x - b = dy / std::sqrt(epsilon) - sum(dy) / std::sqrt(epsilon) / 3 = 0
    //   dscale = sum(dy * (x - mean) * rstd) = 0
    //   dbias  = sum(dy) = 14.0

    Tensor<double> dy({2, 3});
    Tensor<double> x({2, 3});
    Tensor<double> scale({3});
    Tensor<double> mean({2});
    Tensor<double> rstd({2});
    Tensor<double> dx({2, 3});
    Tensor<double> dscale({3});
    Tensor<double> dbias({3});

    dy.fillWithValue(7.0);
    x.fillWithValue(7.0);
    scale.fillWithValue(1.0);
    mean.fillWithValue(7.0);
    rstd.fillWithValue(1.0 / std::sqrt(LAYERNORM_DEFAULT_EPSILON));

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1);

    auto tolerance = layernorm::getTolerance<double>();

    for(int f = 0; f < 3; f++)
    {
        EXPECT_NEAR(dscale.getHostValue(f), 0.0, tolerance);
        EXPECT_NEAR(dbias.getHostValue(f), 14.0, tolerance);
        for(int b = 0; b < 2; b++)
        {
            EXPECT_NEAR(dx.getHostValue(b, f), 0.0, tolerance);
        }
    }
}

// ============================================================================
// Corner case: single element per normalized group
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, BpropSingleFeature)
{
    // Shape [3, 1]: normalize over dim of size 1
    // mean = x, var = 0
    //   a      = rstd³ * (dy * scale * x - dy * scale * mean) = 0
    //   b      = rstd * dy * scale - a * mean = scale / std::sqrt(epsilon) = 2.0 * dy / std::sqrt(epsilon)
    //   dx     = rstd * dy * scale - a * x - b = 2.0 * dy / std::sqrt(epsilon) - 2.0 * dy / std::sqrt(epsilon) = 0
    //   dscale = sum(dy * (x - mean) * rstd) = 0
    //   dbias  = sum(dy) = -2

    Tensor<double> dy({3, 1});
    Tensor<double> x({3, 1});
    Tensor<double> scale({1});
    Tensor<double> mean({3});
    Tensor<double> rstd({3});
    Tensor<double> dx({3, 1});
    Tensor<double> dscale({1});
    Tensor<double> dbias({1});

    dy.setHostValue(-5.0, 0, 0);
    dy.setHostValue(3.0, 1, 0);
    dy.setHostValue(0.0, 2, 0);

    x.setHostValue(5.0, 0, 0);
    x.setHostValue(-3.0, 1, 0);
    x.setHostValue(0.0, 2, 0);

    scale.setHostValue(2.0, 0);

    mean.setHostValue(5.0, 0);
    mean.setHostValue(-3.0, 1);
    mean.setHostValue(0.0, 2);

    rstd.fillWithValue(1.0 / std::sqrt(LAYERNORM_DEFAULT_EPSILON));

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1);

    auto tolerance = layernorm::getTolerance<double>();

    EXPECT_NEAR(dx.getHostValue(0, 0), 0.0, tolerance);
    EXPECT_NEAR(dx.getHostValue(1, 0), 0.0, tolerance);
    EXPECT_NEAR(dx.getHostValue(2, 0), 0.0, tolerance);

    EXPECT_NEAR(dscale.getHostValue(0), 0.0, tolerance);

    EXPECT_NEAR(dbias.getHostValue(0), -2.0, tolerance);
}

// ============================================================================
// Corner case: negative values and mixed signs
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, BpropNegativeAndMixedValues)
{
    // Test with negative and mixed-sign values to ensure correct handling
    //   dy     = [-1, -3, 3, 1]
    //   x      = [-3, 1, -1, 3]
    //   mean   = 0
    //   var    = 5
    //   rstd   = 1/sqrt(5 + 1e-5)
    //   a      = rstd³ * (sum(dy * scale * x) - sum(dy * scale) * mean) / 4 = 0
    //   b      = rstd * sum(dy * scale) / 4 - a * mean = sum(scale) / std::sqrt(epsilon) / 4 = 0
    //   dx     = rstd * dy * scale - a * x - b = dy / std::sqrt(5 + epsilon) = [-1, -3, 3, 1] / std::sqrt(5 + epsilon)
    //   dscale = dy * (x - mean) * rstd = [3, -3, -3, 3] / std::sqrt(5 + epsilon)
    //   dbias  = dy = [-1, -3, 3, 1]

    Tensor<double> dy({1, 4});
    Tensor<double> x({1, 4});
    Tensor<double> scale({4});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});
    Tensor<double> dx({1, 4});
    Tensor<double> dscale({4});
    Tensor<double> dbias({4});

    dy.setHostValue(-1.0, 0, 0);
    dy.setHostValue(-3.0, 0, 1);
    dy.setHostValue(3.0, 0, 2);
    dy.setHostValue(1.0, 0, 3);

    x.setHostValue(-3.0, 0, 0);
    x.setHostValue(1.0, 0, 1);
    x.setHostValue(-1.0, 0, 2);
    x.setHostValue(3.0, 0, 3);

    scale.fillWithValue(1.0);
    mean.fillWithValue(0.0);
    rstd.fillWithValue(1.0 / std::sqrt(5.0 + LAYERNORM_DEFAULT_EPSILON));

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1);

    auto tolerance = layernorm::getTolerance<double>();

    // dx and dbias should be antisymmetric: dx[0]=-dx[3], dx[1]=-dx[2]
    EXPECT_NEAR(dx.getHostValue(0, 0), -dx.getHostValue(0, 3), tolerance);
    EXPECT_NEAR(dx.getHostValue(0, 1), -dx.getHostValue(0, 2), tolerance);
    EXPECT_NEAR(dbias.getHostValue(0), -dbias.getHostValue(3), tolerance);
    EXPECT_NEAR(dbias.getHostValue(1), -dbias.getHostValue(2), tolerance);

    // dscale should be symmetric: dscale[0]=dscale[3], dscale[1]=dscale[2]
    EXPECT_NEAR(dscale.getHostValue(0), dscale.getHostValue(3), tolerance);
    EXPECT_NEAR(dscale.getHostValue(1), dscale.getHostValue(2), tolerance);

    // And the actual values
    EXPECT_NEAR(dx.getHostValue(0, 0), -1.0 * rstd.getHostValue(0), tolerance);
    EXPECT_NEAR(dx.getHostValue(0, 1), -3.0 * rstd.getHostValue(0), tolerance);
    EXPECT_NEAR(dx.getHostValue(0, 2), 3.0 * rstd.getHostValue(0), tolerance);
    EXPECT_NEAR(dx.getHostValue(0, 3), 1.0 * rstd.getHostValue(0), tolerance);

    EXPECT_NEAR(dscale.getHostValue(0), 3.0 * rstd.getHostValue(0), tolerance);
    EXPECT_NEAR(dscale.getHostValue(1), -3.0 * rstd.getHostValue(0), tolerance);
    EXPECT_NEAR(dscale.getHostValue(2), -3.0 * rstd.getHostValue(0), tolerance);
    EXPECT_NEAR(dscale.getHostValue(3), 3.0 * rstd.getHostValue(0), tolerance);

    EXPECT_NEAR(dbias.getHostValue(0), -1.0, tolerance);
    EXPECT_NEAR(dbias.getHostValue(1), -3.0, tolerance);
    EXPECT_NEAR(dbias.getHostValue(2), 3.0, tolerance);
    EXPECT_NEAR(dbias.getHostValue(3), 1.0, tolerance);
}

// ============================================================================
// Numerical stability: large values with small variance
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, BpropLargeValueNumericalStability)
{
    // Values clustered around 1e15 with small relative differences.
    // Naive one-pass (E[x²] - E[x]²) would suffer catastrophic cancellation here
    // because E[x²] ≈ 1e30 and E[x]² ≈ 1e30, so their difference loses all precision.
    // Welford's algorithm computes variance from centered deltas, avoiding this.
    //
    // dy     = [-1e15 - 1, -1e15 - 2, -1e15 - 3, -1e15 - 4]
    // x      = [1e15 + 1, 1e15 + 2, 1e15 + 3, 1e15 + 4]
    // mean   = 1e15 + 2.5
    // var    = 1.25 (same as small-value case, independent of offset)
    // rstd   = 1/sqrt(1.25 + 1e-5)
    // a      = rstd³ * (sum(dy * scale * x) - sum(dy * scale) * mean) / 4 = 0
    // b      = rstd * sum(dy * scale) / 4 - a * mean = (-1e15 - 2.5) / std::sqrt(1.25 + 1e-5) = -894423613312620.2
    // dx     = rstd * dy * scale - a * x - b = [1.375, 0.5, -0.375, -1.375]
    // dscale = dy * (x - mean) * rstd = [1.5e+15 + 1.5, 5.0e+14 + 1.0, -5.0e+14 - 1.5, -1.5e+15 - 6.0] / std::sqrt(1.25 + 1e-5) = [1.34163542e+15, 4.47211807e+14, -4.47211807e+14, -1.34163542e+15]
    // dbias  = dy = [-1e15 - 1, -1e15 - 2, -1e15 - 3, -1e15 - 4]

    Tensor<double> dy({1, 4});
    Tensor<double> x({1, 4});
    Tensor<double> scale({4});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});
    Tensor<double> dx({1, 4});
    Tensor<double> dscale({4});
    Tensor<double> dbias({4});

    const double offset = 1.0e15;
    dy.setHostValue(-offset - 1.0, 0, 0);
    dy.setHostValue(-offset - 2.0, 0, 1);
    dy.setHostValue(-offset - 3.0, 0, 2);
    dy.setHostValue(-offset - 4.0, 0, 3);

    x.setHostValue(offset + 1.0, 0, 0);
    x.setHostValue(offset + 2.0, 0, 1);
    x.setHostValue(offset + 3.0, 0, 2);
    x.setHostValue(offset + 4.0, 0, 3);

    scale.fillWithValue(1.0);
    mean.fillWithValue(offset + 2.5);
    rstd.fillWithValue(1.0 / std::sqrt(1.25 + LAYERNORM_DEFAULT_EPSILON));

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1);

    auto tolerance = layernorm::getTolerance<double>();

    // Normalized output should match the small-value case exactly
    EXPECT_NEAR(dx.getHostValue(0, 0), 1.375, tolerance);
    EXPECT_NEAR(dx.getHostValue(0, 1), 0.5, tolerance);
    EXPECT_NEAR(dx.getHostValue(0, 2), -0.375, tolerance);
    EXPECT_NEAR(dx.getHostValue(0, 3), -1.375, tolerance);

    EXPECT_NEAR(dscale.getHostValue(0), (1.5e15 + 1.5) * rstd.getHostValue(0), tolerance);
    EXPECT_NEAR(dscale.getHostValue(1), (0.5e15 + 1.0) * rstd.getHostValue(0), tolerance);
    EXPECT_NEAR(dscale.getHostValue(2), (-0.5e15 - 1.5) * rstd.getHostValue(0), tolerance);
    EXPECT_NEAR(dscale.getHostValue(3), (-1.5e15 - 6.0) * rstd.getHostValue(0), tolerance);

    EXPECT_NEAR(dbias.getHostValue(0), -1.0e15 - 1.0, tolerance);
    EXPECT_NEAR(dbias.getHostValue(1), -1.0e15 - 2.0, tolerance);
    EXPECT_NEAR(dbias.getHostValue(2), -1.0e15 - 3.0, tolerance);
    EXPECT_NEAR(dbias.getHostValue(3), -1.0e15 - 4.0, tolerance);
}

// ============================================================================
// Realistic shape: typical transformer hidden dim
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp32, BpropTypicalTransformerShape)
{
    // Batch=2, SeqLen=8, Hidden=64 — normalize over last dim (hidden)
    Tensor<float> dy({2, 8, 64});
    Tensor<float> x({2, 8, 64});
    Tensor<float> scale({64});
    Tensor<float> mean({2, 8});
    Tensor<float> rstd({2, 8});
    Tensor<float> dx({2, 8, 64});
    Tensor<float> dscale({64});
    Tensor<float> dbias({64});

    dy.fillWithRandomValues(-1.0f, 1.0f, 42);
    x.fillWithRandomValues(-1.0f, 1.0f, 43);
    scale.fillWithValue(1.0f);
    for(size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 8; ++j)
        {
            float batchMean = 0.0f;
            for(size_t k = 0; k < 64; ++k)
            {
                batchMean += x.getHostValue(i, j, k);
            }
            batchMean /= 64.0f;
            mean.setHostValue(batchMean, i, j);
            float batchVariance = 0.0f;
            for(size_t k = 0; k < 64; ++k)
            {
                const float diff = x.getHostValue(i, j, k) - batchMean;
                batchVariance += diff * diff;
            }
            batchVariance /= 64.0f;
            rstd.setHostValue(
                1.0f / std::sqrt(batchVariance + static_cast<float>(LAYERNORM_DEFAULT_EPSILON)),
                i,
                j);
        }
    }

    CpuFpReferenceLayernorm::bprop(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1);

    auto tolerance = layernorm::getTolerance<float>();

    for(size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 8; ++j)
        {
            const float batchRstd = rstd.getHostValue(i, j);
            float sumDyScaleX = 0.0f;
            float sumDyScale = 0.0f;
            for(size_t k = 0; k < 64; ++k)
            {
                sumDyScaleX
                    += dy.getHostValue(i, j, k) * scale.getHostValue(k) * x.getHostValue(i, j, k);
                sumDyScale += dy.getHostValue(i, j, k) * scale.getHostValue(k);
            }
            const float a = batchRstd * batchRstd * batchRstd
                            * (sumDyScaleX - sumDyScale * mean.getHostValue(i, j)) / 64.0f;
            const float b = batchRstd * sumDyScale / 64.0f - a * mean.getHostValue(i, j);
            for(size_t k = 0; k < 64; ++k)
            {
                const float outDx = batchRstd * dy.getHostValue(i, j, k) * scale.getHostValue(k)
                                    - a * x.getHostValue(i, j, k) - b;
                EXPECT_NEAR(outDx, dx.getHostValue(i, j, k), tolerance);
            }
        }
    }
    for(size_t k = 0; k < 64; ++k)
    {
        float batchDscale = 0.0f;
        float batchDbias = 0.0f;
        for(size_t i = 0; i < 2; ++i)
        {
            for(size_t j = 0; j < 8; ++j)
            {
                batchDscale += dy.getHostValue(i, j, k)
                               * (x.getHostValue(i, j, k) - mean.getHostValue(i, j))
                               * rstd.getHostValue(i, j);
                batchDbias += dy.getHostValue(i, j, k);
            }
        }
        EXPECT_NEAR(batchDscale, dscale.getHostValue(k), tolerance);
        EXPECT_NEAR(batchDbias, dbias.getHostValue(k), tolerance);
    }
}

// ============================================================================
// Type compatibility: various type combinations
// ============================================================================

template <typename T1, typename T2>
struct TypePair
{
    using First = T1;
    using Second = T2;
};

using LayernormBpropTypes = ::testing::Types<TypePair<float, float>,
                                             TypePair<half, float>,
                                             TypePair<bfloat16, float>,
                                             TypePair<double, double>>;

template <class T>
class CpuFpReferenceLayernormBpropTyped : public ::testing::Test
{
};

TYPED_TEST_SUITE(CpuFpReferenceLayernormBpropTyped, LayernormBpropTypes, );

TYPED_TEST(CpuFpReferenceLayernormBpropTyped, BpropRunsWithoutError)
{
    using DyType = typename TypeParam::First;
    using ScaleType = typename TypeParam::Second;

    Tensor<DyType> dy({2, 8, 32});
    Tensor<DyType> x({2, 8, 32});
    Tensor<ScaleType> scale({32});
    Tensor<DyType> mean({2, 8});
    Tensor<DyType> rstd({2, 8});
    Tensor<DyType> dx({2, 8, 32});
    Tensor<ScaleType> dscale({32});
    Tensor<ScaleType> dbias({32});

    dy.fillWithValue(safeTestTypeCast<DyType>(0.0));
    x.fillWithValue(safeTestTypeCast<DyType>(1.0));
    scale.fillWithValue(safeTestTypeCast<ScaleType>(1.0));
    mean.fillWithValue(safeTestTypeCast<DyType>(1.0));
    rstd.fillWithValue(safeTestTypeCast<DyType>(1.0 / std::sqrt(LAYERNORM_DEFAULT_EPSILON)));

    // Should not throw
    CpuFpReferenceLayernorm::bprop(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1);

    auto tolerance = layernorm::getTolerance<double>();

    // All zeroes dy -> all dx, dscale and dbias are zero
    for(size_t k = 0; k < 32; ++k)
    {
        for(size_t i = 0; i < 2; ++i)
        {
            for(size_t j = 0; j < 8; ++j)
            {
                EXPECT_NEAR(static_cast<float>(dx.getHostValue(i, j, k)), 0.0f, tolerance);
            }
        }
        EXPECT_NEAR(static_cast<float>(dscale.getHostValue(k)), 0.0f, tolerance);
        EXPECT_NEAR(static_cast<float>(dbias.getHostValue(k)), 0.0f, tolerance);
    }
}

// ============================================================================
// Multi-dim normalization: normalize over last 2 dims
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, BpropNormalizeLastTwoDims)
{
    // Shape [2, 3, 4], normalize over last 2 dims (3x4 = 12 features)
    // Each of the 2 batches is independently normalized over 12 elements

    Tensor<double> dy({2, 3, 4});
    Tensor<double> x({2, 3, 4});
    Tensor<double> scale({3, 4});
    Tensor<double> mean({2});
    Tensor<double> rstd({2});
    Tensor<double> dx({2, 3, 4});
    Tensor<double> dscale({3, 4});
    Tensor<double> dbias({3, 4});

    // Batch 0: dy values 0.1..1.2, x values 1..12, mean = (1+2+...+12)/12 = 78/12 = 6.5
    double batchVar = 0.0;
    for(int i = 0; i < 3; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            dy.setHostValue(static_cast<double>(i * 4 + j + 1) / 10.0, 0, i, j);
            auto xVal = static_cast<double>(i * 4 + j + 1);
            x.setHostValue(xVal, 0, i, j);
            batchVar += (xVal - 6.5) * (xVal - 6.5);
        }
    }
    mean.setHostValue(6.5, 0);
    rstd.setHostValue(1.0 / std::sqrt(batchVar + LAYERNORM_DEFAULT_EPSILON), 0);

    // Batch 1: dy values 1.3..2.4, x values 13..24, mean = (13+14+...+24)/12 = 222/12 = 18.5
    batchVar = 0.0;
    for(int i = 0; i < 2; i++)
    {
        for(int j = 0; j < 3; j++)
        {
            dy.setHostValue(static_cast<double>(i * 4 + j + 13) / 10.0, 1, i, j);
            auto xVal = static_cast<double>(i * 4 + j + 13);
            x.setHostValue(xVal, 1, i, j);
            batchVar += (xVal - 18.5) * (xVal - 18.5);
        }
    }
    mean.setHostValue(18.5, 1);
    rstd.setHostValue(1.0 / std::sqrt(batchVar + LAYERNORM_DEFAULT_EPSILON), 1);

    scale.fillWithValue(1.0);

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 2);

    auto tolerance = layernorm::getTolerance<double>();

    for(size_t i = 0; i < 2; ++i)
    {
        const double batchMean = mean.getHostValue(i);
        const double batchRstd = rstd.getHostValue(i);
        double sumDyScaleX = 0.0;
        double sumDyScale = 0.0;
        for(size_t j = 0; j < 3; ++j)
        {
            for(size_t k = 0; k < 4; ++k)
            {
                sumDyScaleX += dy.getHostValue(i, j, k) * scale.getHostValue(j, k)
                               * x.getHostValue(i, j, k);
                sumDyScale += dy.getHostValue(i, j, k) * scale.getHostValue(j, k);
            }
        }
        const double a
            = batchRstd * batchRstd * batchRstd * (sumDyScaleX - sumDyScale * batchMean) / 12.0;
        const double b = batchRstd * sumDyScale / 12.0 - a * batchMean;
        for(size_t j = 0; j < 3; ++j)
        {
            for(size_t k = 0; k < 4; ++k)
            {
                const double outDx = batchRstd * dy.getHostValue(i, j, k) * scale.getHostValue(j, k)
                                     - a * x.getHostValue(i, j, k) - b;
                EXPECT_NEAR(outDx, dx.getHostValue(i, j, k), tolerance);
            }
        }
    }
    for(size_t k = 0; k < 4; ++k)
    {
        for(size_t j = 0; j < 3; ++j)
        {
            double outDscale = 0.0;
            double outDbias = 0.0;
            for(size_t i = 0; i < 2; ++i)
            {
                outDscale += dy.getHostValue(i, j, k)
                             * (x.getHostValue(i, j, k) - mean.getHostValue(i))
                             * rstd.getHostValue(i);
                outDbias += dy.getHostValue(i, j, k);
            }
            EXPECT_NEAR(outDscale, dscale.getHostValue(j, k), tolerance);
            EXPECT_NEAR(outDbias, dbias.getHostValue(j, k), tolerance);
        }
    }
}

// ============================================================================
// Corner case: "diagonal-like" pattern — one hot-per-row
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, BpropOneHotRows)
{
    // Shape [3, 3], each row is a one-hot vector
    // Row 0: [1, 0, 0], mean=1/3, var=(4/9+1/9+1/9)/3=2/9
    // Row 1: [0, 1, 0], same statistics by symmetry
    // Row 2: [0, 0, 1], same statistics by symmetry
    //
    // All rows should produce the same set of output values (permuted)

    Tensor<double> dy({3, 3});
    Tensor<double> x({3, 3});
    Tensor<double> scale({3});
    Tensor<double> mean({3});
    Tensor<double> rstd({3});
    Tensor<double> dx({3, 3});
    Tensor<double> dscale({3});
    Tensor<double> dbias({3});

    dy.fillWithValue(0.0);
    dy.setHostValue(0.1, 0, 0);
    dy.setHostValue(0.1, 1, 1);
    dy.setHostValue(0.1, 2, 2);

    x.fillWithValue(0.0);
    x.setHostValue(1.0, 0, 0);
    x.setHostValue(1.0, 1, 1);
    x.setHostValue(1.0, 2, 2);

    scale.fillWithValue(1.0);
    mean.fillWithValue(1.0 / 3.0);
    rstd.fillWithValue(1.0 / std::sqrt(2.0 / 9.0 + LAYERNORM_DEFAULT_EPSILON));

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1);

    auto tolerance = layernorm::getTolerance<double>();

    // The "hot" position output should be identical across rows
    // dx[0,0] should equal dx[1,1] and dx[2,2]
    EXPECT_NEAR(dx.getHostValue(0, 0), dx.getHostValue(1, 1), tolerance);
    EXPECT_NEAR(dx.getHostValue(1, 1), dx.getHostValue(2, 2), tolerance);

    // The "cold" position output should be identical across rows
    EXPECT_NEAR(dx.getHostValue(0, 1), dx.getHostValue(0, 2), tolerance);
    EXPECT_NEAR(dx.getHostValue(0, 1), dx.getHostValue(1, 0), tolerance);
}

// ============================================================================
// Verify per-feature scale and bias
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, BpropPerFeatureScale)
{
    // Verify that different scale per feature is applied correctly
    // dy = [[0.1, 0.2, 0.3, 0.4]]
    // x = [[1, 2, 3, 4]]
    // scale = [1, 2, 3, 4]

    Tensor<double> dy({1, 4});
    Tensor<double> x({1, 4});
    Tensor<double> scale({4});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});
    Tensor<double> dx({1, 4});
    Tensor<double> dscale({4});
    Tensor<double> dbias({4});

    dy.setHostValue(0.1, 0, 0);
    dy.setHostValue(0.2, 0, 1);
    dy.setHostValue(0.3, 0, 2);
    dy.setHostValue(0.4, 0, 3);

    x.setHostValue(1.0, 0, 0);
    x.setHostValue(2.0, 0, 1);
    x.setHostValue(3.0, 0, 2);
    x.setHostValue(4.0, 0, 3);

    scale.setHostValue(1.0, 0);
    scale.setHostValue(2.0, 1);
    scale.setHostValue(3.0, 2);
    scale.setHostValue(4.0, 3);

    // mean=2.5, var=1.25, rstd=1/sqrt(1.25+1e-5)
    mean.setHostValue(2.5, 0);
    rstd.setHostValue(1.0 / std::sqrt(1.25 + LAYERNORM_DEFAULT_EPSILON), 0);

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1);

    auto tolerance = layernorm::getTolerance<double>();

    // a         = rstd³ * (sum(dy[i] * scale[i] * x[i]) - sum(dy[i] * scale[i]) * mean)
    // b         = rstd * sum(dy[i] * scale[i]) - a * mean
    // dx[i]     = rstd * dy[i] * scale[i] - a * x[i] - b
    // dscale[i] = dy[i] * (x[i] - mean) * rstd
    // dbias[i]  = dy[i]
    const double batchRstd = rstd.getHostValue(0);
    double sumDyScaleX = 0.0;
    double sumDyScale = 0.0;
    for(size_t k = 0; k < 4; ++k)
    {
        sumDyScaleX += dy.getHostValue(0, k) * scale.getHostValue(k) * x.getHostValue(0, k);
        sumDyScale += dy.getHostValue(0, k) * scale.getHostValue(k);
    }
    const double a = batchRstd * batchRstd * batchRstd
                     * (sumDyScaleX - sumDyScale * mean.getHostValue(0)) / 4.0;
    const double b = batchRstd * sumDyScale / 4.0 - a * mean.getHostValue(0);
    for(size_t k = 0; k < 4; ++k)
    {
        const double outDx = batchRstd * dy.getHostValue(0, k) * scale.getHostValue(k)
                             - a * x.getHostValue(0, k) - b;
        EXPECT_NEAR(outDx, dx.getHostValue(0, k), tolerance);
    }
    for(size_t k = 0; k < 4; ++k)
    {
        const double outDscale = dy.getHostValue(0, k)
                                 * (x.getHostValue(0, k) - mean.getHostValue(0))
                                 * rstd.getHostValue(0);
        const double outDbias = dy.getHostValue(0, k);
        EXPECT_NEAR(outDscale, dscale.getHostValue(k), tolerance);
        EXPECT_NEAR(outDbias, dbias.getHostValue(k), tolerance);
    }
}

// ============================================================================
// Error handling: invalid dimensions
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp32, BpropThrowsOnInvalidNormalizedDimCount)
{
    const Tensor<float> dy({2, 4});
    const Tensor<float> x({2, 4});
    const Tensor<float> scale({4});
    const Tensor<float> mean({2});
    const Tensor<float> rstd({2});
    Tensor<float> dx({2, 4});
    Tensor<float> dscale({4});
    Tensor<float> dbias({4});

    // normalizedDimCount = 0 is invalid
    EXPECT_THROW(CpuFpReferenceLayernorm::bprop(
                     dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 0),
                 std::runtime_error);

    // normalizedDimCount > ndim is invalid
    EXPECT_THROW(CpuFpReferenceLayernorm::bprop(
                     dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 3),
                 std::runtime_error);
}

// ============================================================================
// Error handling: scalar (0D) tensor
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp32, BpropThrowsOnScalarTensor)
{
    const Tensor<float> dy({});
    const Tensor<float> x({});
    const Tensor<float> scale({});
    const Tensor<float> mean({});
    const Tensor<float> rstd({});
    Tensor<float> dx({});
    Tensor<float> dscale({});
    Tensor<float> dbias({});

    EXPECT_THROW(CpuFpReferenceLayernorm::bprop(
                     dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1),
                 std::runtime_error);
}

// ============================================================================
// Dimensionality coverage: 1D tensor
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, Bprop1D)
{
    // Shape [5] — normalize entire 1D tensor (single group, no batch dim)
    // dy   = [-1, -2, -3, -4, -5]
    // x    = [2, 4, 6, 8, 10]
    // mean = 6, var = (16+4+0+4+16)/5 = 8
    // rstd = 1/sqrt(8 + 1e-5)

    Tensor<double> dy({5});
    Tensor<double> x({5});
    Tensor<double> scale({5});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});
    Tensor<double> dx({5});
    Tensor<double> dscale({5});
    Tensor<double> dbias({5});

    dy.setHostValue(-1, 0);
    dy.setHostValue(-2, 1);
    dy.setHostValue(-3, 2);
    dy.setHostValue(-4, 3);
    dy.setHostValue(-5, 4);

    x.setHostValue(2.0, 0);
    x.setHostValue(4.0, 1);
    x.setHostValue(6.0, 2);
    x.setHostValue(8.0, 3);
    x.setHostValue(10.0, 4);

    scale.fillWithValue(1.0);
    mean.fillWithValue(6.0);
    rstd.fillWithValue(1.0 / std::sqrt(8.0 + LAYERNORM_DEFAULT_EPSILON));

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1);

    auto tolerance = layernorm::getTolerance<double>();

    const double batchRstd = rstd.getHostValue(0);
    double sumDyScaleX = 0.0;
    double sumDyScale = 0.0;
    for(size_t k = 0; k < 5; ++k)
    {
        sumDyScaleX += dy.getHostValue(k) * scale.getHostValue(k) * x.getHostValue(k);
        sumDyScale += dy.getHostValue(k) * scale.getHostValue(k);
    }
    const double a = batchRstd * batchRstd * batchRstd
                     * (sumDyScaleX - sumDyScale * mean.getHostValue(0)) / 5.0;
    const double b = batchRstd * sumDyScale / 5.0 - a * mean.getHostValue(0);
    for(size_t k = 0; k < 5; ++k)
    {
        const double outDx
            = batchRstd * dy.getHostValue(k) * scale.getHostValue(k) - a * x.getHostValue(k) - b;
        EXPECT_NEAR(outDx, dx.getHostValue(k), tolerance);
    }
    for(size_t k = 0; k < 5; ++k)
    {
        const double outDscale = dy.getHostValue(k) * (x.getHostValue(k) - mean.getHostValue(0))
                                 * rstd.getHostValue(0);
        const double outDbias = dy.getHostValue(k);
        EXPECT_NEAR(outDscale, dscale.getHostValue(k), tolerance);
        EXPECT_NEAR(outDbias, dbias.getHostValue(k), tolerance);
    }
}

// ============================================================================
// Dimensionality coverage: 4D tensor
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, Bprop4DNormalizeLast1)
{
    // Shape [2, 3, 2, 4], normalizedDimCount=1: normalize over last dim (4)
    // Batch dims = [2, 3, 2], 12 independent groups of 4 elements each

    Tensor<double> dy({2, 3, 2, 4});
    Tensor<double> x({2, 3, 2, 4});
    Tensor<double> scale({4});
    Tensor<double> mean({2, 3, 2});
    Tensor<double> rstd({2, 3, 2});
    Tensor<double> dx({2, 3, 2, 4});
    Tensor<double> dscale({4});
    Tensor<double> dbias({4});

    // Fill with sequential values so each group has a known pattern
    double val = 1.0;
    for(int b = 0; b < 2; b++)
    {
        for(int c = 0; c < 3; c++)
        {
            for(int h = 0; h < 2; h++)
            {
                for(int w = 0; w < 4; w++)
                {
                    dy.setHostValue(-val / 10.0, b, c, h, w);
                    x.setHostValue(val, b, c, h, w);
                    val += 1.0;
                }
            }
        }
    }

    scale.fillWithValue(1.0);
    mean.fillWithValue(2.5);
    rstd.fillWithValue(1.0 / std::sqrt(1.25 + LAYERNORM_DEFAULT_EPSILON));

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 1);

    auto tolerance = layernorm::getTolerance<double>();

    for(size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 3; ++j)
        {
            for(size_t k = 0; k < 2; ++k)
            {
                const double batchRstd = rstd.getHostValue(i, j, k);
                double sumDyScaleX = 0.0;
                double sumDyScale = 0.0;
                for(size_t l = 0; l < 4; ++l)
                {
                    sumDyScaleX += dy.getHostValue(i, j, k, l) * scale.getHostValue(l)
                                   * x.getHostValue(i, j, k, l);
                    sumDyScale += dy.getHostValue(i, j, k, l) * scale.getHostValue(l);
                }
                const double a = batchRstd * batchRstd * batchRstd
                                 * (sumDyScaleX - sumDyScale * mean.getHostValue(i, j, k)) / 4.0;
                const double b = batchRstd * sumDyScale / 4.0 - a * mean.getHostValue(i, j, k);
                for(size_t l = 0; l < 4; ++l)
                {
                    const double outDx
                        = batchRstd * dy.getHostValue(i, j, k, l) * scale.getHostValue(l)
                          - a * x.getHostValue(i, j, k, l) - b;
                    EXPECT_NEAR(outDx, dx.getHostValue(i, j, k, l), tolerance);
                }
            }
        }
    }
    for(size_t l = 0; l < 4; ++l)
    {
        double outDscale = 0.0;
        double outDbias = 0.0;
        for(size_t i = 0; i < 2; ++i)
        {
            for(size_t j = 0; j < 3; ++j)
            {
                for(size_t k = 0; k < 2; ++k)
                {
                    outDscale += dy.getHostValue(i, j, k, l)
                                 * (x.getHostValue(i, j, k, l) - mean.getHostValue(i, j, k))
                                 * rstd.getHostValue(i, j, k);
                    outDbias += dy.getHostValue(i, j, k, l);
                }
            }
        }
        EXPECT_NEAR(outDscale, dscale.getHostValue(l), tolerance);
        EXPECT_NEAR(outDbias, dbias.getHostValue(l), tolerance);
    }
}

TEST(TestCpuFpReferenceLayernormBackwardFp64, Bprop4DNormalizeLast3)
{
    // Shape [2, 3, 2, 4], normalizedDimCount=3: normalize over last 3 dims (3*2*4=24)
    // Batch dims = [2], 2 independent groups of 24 elements each

    Tensor<double> dy({2, 3, 2, 4});
    Tensor<double> x({2, 3, 2, 4});
    Tensor<double> scale({3, 2, 4});
    Tensor<double> mean({2});
    Tensor<double> rstd({2});
    Tensor<double> dx({2, 3, 2, 4});
    Tensor<double> dscale({3, 2, 4});
    Tensor<double> dbias({3, 2, 4});

    // Batch 0: values 1..24, Batch 1: values 25..48
    double val = 1.0;
    for(int b = 0; b < 2; b++)
    {
        for(int c = 0; c < 3; c++)
        {
            for(int h = 0; h < 2; h++)
            {
                for(int w = 0; w < 4; w++)
                {
                    dy.setHostValue(-val / 10.0, b, c, h, w);
                    x.setHostValue(val, b, c, h, w);
                    val += 1.0;
                }
            }
        }
    }

    scale.fillWithValue(1.0);
    mean.fillWithValue(12.5);
    rstd.fillWithValue(1.0 / std::sqrt(47.91666666667 + LAYERNORM_DEFAULT_EPSILON));

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 3);

    auto tolerance = layernorm::getTolerance<double>();

    for(size_t i = 0; i < 2; ++i)
    {
        const double batchRstd = rstd.getHostValue(i);
        double sumDyScaleX = 0.0;
        double sumDyScale = 0.0;
        for(size_t j = 0; j < 3; ++j)
        {
            for(size_t k = 0; k < 2; ++k)
            {
                for(size_t l = 0; l < 4; ++l)
                {
                    sumDyScaleX += dy.getHostValue(i, j, k, l) * scale.getHostValue(j, k, l)
                                   * x.getHostValue(i, j, k, l);
                    sumDyScale += dy.getHostValue(i, j, k, l) * scale.getHostValue(j, k, l);
                }
            }
        }
        const double a = batchRstd * batchRstd * batchRstd
                         * (sumDyScaleX - sumDyScale * mean.getHostValue(i)) / 24.0;
        const double b = batchRstd * sumDyScale / 24.0 - a * mean.getHostValue(i);
        for(size_t j = 0; j < 3; ++j)
        {
            for(size_t k = 0; k < 2; ++k)
            {
                for(size_t l = 0; l < 4; ++l)
                {
                    const double outDx
                        = batchRstd * dy.getHostValue(i, j, k, l) * scale.getHostValue(j, k, l)
                          - a * x.getHostValue(i, j, k, l) - b;
                    EXPECT_NEAR(outDx, dx.getHostValue(i, j, k, l), tolerance);
                }
            }
        }
    }
    for(size_t j = 0; j < 3; ++j)
    {
        for(size_t k = 0; k < 2; ++k)
        {
            for(size_t l = 0; l < 4; ++l)
            {
                double outDscale = 0.0;
                double outDbias = 0.0;
                for(size_t i = 0; i < 2; ++i)
                {
                    outDscale += dy.getHostValue(i, j, k, l)
                                 * (x.getHostValue(i, j, k, l) - mean.getHostValue(i))
                                 * rstd.getHostValue(i);
                    outDbias += dy.getHostValue(i, j, k, l);
                }
                EXPECT_NEAR(outDscale, dscale.getHostValue(j, k, l), tolerance);
                EXPECT_NEAR(outDbias, dbias.getHostValue(j, k, l), tolerance);
            }
        }
    }
}

// ============================================================================
// Full-tensor normalization (normalizedDimCount == ndim)
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, BpropFullTensorNormalization)
{
    // Shape [2, 3], normalizedDimCount=2 means normalize entire tensor per "batch"
    // But there are no batch dims here, so entire tensor is one group
    // x = [[-1, 2, -3], [4, -5, 6]]
    // x = [[1, 2, 3], [4, 5, 6]]
    // mean = 3.5, var = 17.5/6 = 2.9166...

    Tensor<double> dy({2, 3});
    Tensor<double> x({2, 3});
    Tensor<double> scale({2, 3});
    Tensor<double> mean({1});
    Tensor<double> rstd({1});
    Tensor<double> dx({2, 3});
    Tensor<double> dscale({2, 3});
    Tensor<double> dbias({2, 3});

    dy.setHostValue(-1.0, 0, 0);
    dy.setHostValue(2.0, 0, 1);
    dy.setHostValue(-3.0, 0, 2);
    dy.setHostValue(4.0, 1, 0);
    dy.setHostValue(-5.0, 1, 1);
    dy.setHostValue(6.0, 1, 2);

    x.setHostValue(1.0, 0, 0);
    x.setHostValue(2.0, 0, 1);
    x.setHostValue(3.0, 0, 2);
    x.setHostValue(4.0, 1, 0);
    x.setHostValue(5.0, 1, 1);
    x.setHostValue(6.0, 1, 2);

    scale.fillWithValue(1.0);
    mean.fillWithValue(3.5);
    rstd.fillWithValue(1.0 / std::sqrt(2.916666667 + LAYERNORM_DEFAULT_EPSILON));

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 2);

    auto tolerance = layernorm::getTolerance<double>();

    const double batchRstd = rstd.getHostValue(0);
    double sumDyScaleX = 0.0;
    double sumDyScale = 0.0;
    for(size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 3; ++j)
        {
            sumDyScaleX += dy.getHostValue(i, j) * scale.getHostValue(i, j) * x.getHostValue(i, j);
            sumDyScale += dy.getHostValue(i, j) * scale.getHostValue(i, j);
        }
    }
    const double a = batchRstd * batchRstd * batchRstd
                     * (sumDyScaleX - sumDyScale * mean.getHostValue(0)) / 6.0;
    const double b = batchRstd * sumDyScale / 6.0 - a * mean.getHostValue(0);
    for(size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 3; ++j)
        {
            const double outDx = batchRstd * dy.getHostValue(i, j) * scale.getHostValue(i, j)
                                 - a * x.getHostValue(i, j) - b;
            EXPECT_NEAR(outDx, dx.getHostValue(i, j), tolerance);
        }
    }
    for(size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 3; ++j)
        {
            double outDscale = 0.0;
            double outDbias = 0.0;
            outDscale += dy.getHostValue(i, j) * (x.getHostValue(i, j) - mean.getHostValue(0))
                         * rstd.getHostValue(0);
            outDbias += dy.getHostValue(i, j);
            EXPECT_NEAR(outDscale, dscale.getHostValue(i, j), tolerance);
            EXPECT_NEAR(outDbias, dbias.getHostValue(i, j), tolerance);
        }
    }
}

// ============================================================================
// Dimensionality coverage: 5D tensor
// ============================================================================

TEST(TestCpuFpReferenceLayernormBackwardFp64, Bprop5DNormalizeLast2)
{
    // Shape [2, 2, 3, 4, 5], normalizedDimCount=2: normalize over last 2 dims (4*5=20)
    // Batch dims = [2, 2, 3], 12 independent groups of 20 elements each

    Tensor<double> dy({2, 2, 3, 4, 5});
    Tensor<double> x({2, 2, 3, 4, 5});
    Tensor<double> scale({4, 5});
    Tensor<double> mean({2, 2, 3});
    Tensor<double> rstd({2, 2, 3});
    Tensor<double> dx({2, 2, 3, 4, 5});
    Tensor<double> dscale({4, 5});
    Tensor<double> dbias({4, 5});

    dy.fillWithRandomValues(-5.0, 5.0, 123);
    x.fillWithRandomValues(-5.0, 5.0, 123);
    scale.fillWithValue(1.0);
    for(size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 2; ++j)
        {
            for(size_t k = 0; k < 3; ++k)
            {
                double batchMean = 0.0;
                for(size_t l = 0; l < 4; ++l)
                {
                    for(size_t m = 0; m < 5; ++m)
                    {
                        batchMean += x.getHostValue(i, j, k, l, m);
                    }
                }
                batchMean /= 12.0;
                mean.setHostValue(batchMean, i, j, k);
                double batchVar = 0.0;
                for(size_t l = 0; l < 4; ++l)
                {
                    for(size_t m = 0; m < 5; ++m)
                    {
                        const double diff = x.getHostValue(i, j, k, l, m) - batchMean;
                        batchVar += diff * diff;
                    }
                }
                batchVar /= 12.0;
                rstd.setHostValue(1.0 / std::sqrt(batchVar + LAYERNORM_DEFAULT_EPSILON), i, j, k);
            }
        }
    }

    CpuFpReferenceLayernorm::bprop<double, double, double, double, double>(
        dy, x, scale, mean, rstd, dx, dscale, dbias, LAYERNORM_DEFAULT_EPSILON, 2);

    auto tolerance = layernorm::getTolerance<double>();

    for(size_t i = 0; i < 2; ++i)
    {
        for(size_t j = 0; j < 2; ++j)
        {
            for(size_t k = 0; k < 3; ++k)
            {
                const double batchRstd = rstd.getHostValue(i, j, k);
                double sumDyScaleX = 0.0;
                double sumDyScale = 0.0;
                for(size_t l = 0; l < 4; ++l)
                {
                    for(size_t m = 0; m < 5; ++m)
                    {
                        sumDyScaleX += dy.getHostValue(i, j, k, l, m) * scale.getHostValue(l, m)
                                       * x.getHostValue(i, j, k, l, m);
                        sumDyScale += dy.getHostValue(i, j, k, l, m) * scale.getHostValue(l, m);
                    }
                }
                const double a = batchRstd * batchRstd * batchRstd
                                 * (sumDyScaleX - sumDyScale * mean.getHostValue(i, j, k)) / 20.0;
                const double b = batchRstd * sumDyScale / 20.0 - a * mean.getHostValue(i, j, k);
                for(size_t l = 0; l < 4; ++l)
                {
                    for(size_t m = 0; m < 5; ++m)
                    {
                        const double outDx
                            = batchRstd * dy.getHostValue(i, j, k, l, m) * scale.getHostValue(l, m)
                              - a * x.getHostValue(i, j, k, l, m) - b;
                        EXPECT_NEAR(outDx, dx.getHostValue(i, j, k, l, m), tolerance);
                    }
                }
            }
        }
    }
    for(size_t l = 0; l < 4; ++l)
    {
        for(size_t m = 0; m < 5; ++m)
        {
            double outDscale = 0.0;
            double outDbias = 0.0;
            for(size_t i = 0; i < 2; ++i)
            {
                for(size_t j = 0; j < 2; ++j)
                {
                    for(size_t k = 0; k < 3; ++k)
                    {
                        outDscale += dy.getHostValue(i, j, k, l, m)
                                     * (x.getHostValue(i, j, k, l, m) - mean.getHostValue(i, j, k))
                                     * rstd.getHostValue(i, j, k);
                        outDbias += dy.getHostValue(i, j, k, l, m);
                    }
                }
            }
            EXPECT_NEAR(outDscale, dscale.getHostValue(l, m), tolerance);
            EXPECT_NEAR(outDbias, dbias.getHostValue(l, m), tolerance);
        }
    }
}
