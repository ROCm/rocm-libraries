// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <roc/host_validation/adapters/hipblaslt/GroupedGemmDataInitialization.hpp>
#include <roc/host_validation/adapters/hipblaslt/HipblasltDataInitialization.hpp>
#include <roc/host_validation/adapters/hipblaslt/HipblasltReferenceGemm.hpp>
#include <roc/host_validation/adapters/hipblaslt/HostComparison.hpp>
#include <roc/host_validation/adapters/hipblaslt/hipblaslt_init.hpp>

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

TEST(HostValidationDataInitializationBridge, GeneratesComplexTrigonometricValues)
{
    std::array<std::complex<float>, 4> values{};
    roc::host_validation::hipblaslt_adapter::initialize(std::span<std::complex<float>>(values),
                                                        hipblaslt_initialization::trig_float,
                                                        roc::host_validation::DataPattern::Sine);

    for(size_t index = 0; index < values.size(); ++index)
    {
        EXPECT_FLOAT_EQ(values[index].real(), std::sin(static_cast<float>(index)));
        EXPECT_FLOAT_EQ(values[index].imag(), std::cos(static_cast<float>(index)));
    }
}

TEST(HostValidationDataInitializationBridge, GroupedGemmUsesStableRoleStreams)
{
    std::vector<float> a(5);
    std::vector<float> b(7);
    std::vector<float> c(4);
    std::vector<float> bias(3);

    roc::host_validation::hipblaslt_adapter::initializeGroupedGemm(
        a,
        static_cast<int64_t>(a.size()),
        b,
        static_cast<int64_t>(b.size()),
        c,
        static_cast<int64_t>(c.size()),
        bias,
        static_cast<int64_t>(bias.size()),
        hipblaslt_initialization::rand_int);

    constexpr uint64_t seed = 69069;
    for(size_t index = 0; index < a.size(); ++index)
        EXPECT_EQ(a[index], roc::host_validation::indexedUniformInteger(seed, 0, index, 1, 10));
    for(size_t index = 0; index < b.size(); ++index)
    {
        const int magnitude = roc::host_validation::indexedUniformInteger(seed, 1, index, 1, 10);
        EXPECT_EQ(b[index], (index & 1U) == 0 ? -magnitude : magnitude);
    }
    for(size_t index = 0; index < c.size(); ++index)
        EXPECT_EQ(c[index], roc::host_validation::indexedUniformInteger(seed, 2, index, 1, 10));
    for(size_t index = 0; index < bias.size(); ++index)
        EXPECT_EQ(bias[index], roc::host_validation::indexedUniformInteger(seed, 3, index, 1, 10));
}

TEST(HostValidationComparisonBridge, FindsAllcloseToleranceAcrossBatches)
{
    const std::array<float, 4> expected{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> observed{1.0f, 2.00009f, 3.0f, 4.0f};
    roc::host_validation::hipblaslt_adapter::HostComparisonRequest request;
    request.rows                  = 2;
    request.columns               = 1;
    request.leadingDimension      = 2;
    request.batchStride           = 2;
    request.batchCount            = 2;
    request.expected              = expected.data();
    request.observed              = observed.data();
    request.type                  = HIP_R_32F;
    request.findAllCloseTolerance = true;

    const auto report = roc::host_validation::hipblaslt_adapter::compareHost(request);
    ASSERT_TRUE(report.allCloseTolerance);
    EXPECT_EQ(report.allCloseTolerance->absolute, 1e-6);
    EXPECT_EQ(report.allCloseTolerance->relative, 1e-4);
}

TEST(HostValidationComparisonBridge, ComputesRelativeFrobeniusEvidence)
{
    std::array<double, 2>                                          expected{3.0, 4.0};
    std::array<double, 2>                                          observed{0.0, 4.0};
    roc::host_validation::hipblaslt_adapter::HostComparisonRequest request;
    request.rows                          = 2;
    request.columns                       = 1;
    request.leadingDimension              = 2;
    request.batchStride                   = 2;
    request.batchCount                    = 1;
    request.expected                      = expected.data();
    request.observed                      = observed.data();
    request.type                          = HIP_R_64F;
    request.computeRelativeFrobeniusError = true;
    EXPECT_DOUBLE_EQ(
        roc::host_validation::hipblaslt_adapter::compareHost(request).relativeFrobeniusError, 0.6);
}

TEST(HostValidationComparisonBridge, UnitNearAndSpecialValuePolicies)
{
    const float          oneUlp = std::nextafter(1.0f, 2.0f);
    std::array<float, 2> expected{1.0f, std::numeric_limits<float>::infinity()};
    std::array<float, 2> observed{oneUlp, std::numeric_limits<float>::infinity()};

    roc::host_validation::hipblaslt_adapter::HostComparisonRequest request;
    request.rows             = 2;
    request.columns          = 1;
    request.leadingDimension = 2;
    request.batchStride      = 2;
    request.batchCount       = 1;
    request.expected         = expected.data();
    request.observed         = observed.data();
    request.type             = HIP_R_32F;

    request.pointwise = roc::host_validation::hipblaslt_adapter::HostPointwiseComparison::Unit;
    EXPECT_TRUE(roc::host_validation::hipblaslt_adapter::compareHost(request).comparison.passed());

    request.pointwise = roc::host_validation::hipblaslt_adapter::HostPointwiseComparison::Near;
    request.absoluteTolerance = 1e-6;
    EXPECT_TRUE(roc::host_validation::hipblaslt_adapter::compareHost(request).comparison.passed());

    request.pointwise = roc::host_validation::hipblaslt_adapter::HostPointwiseComparison::Disabled;
    request.requireSpecialValueConsistency = true;
    EXPECT_EQ(roc::host_validation::hipblaslt_adapter::compareHost(request)
                  .comparison.nonFiniteMismatches,
              0);
}

TEST(HostValidationComparisonBridge, RunsTheCombinedHostComparisonProgram)
{
    const float                oneUlp = std::nextafter(1.0f, 2.0f);
    const std::array<float, 4> expected{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> observed{oneUlp, 2.0f, 3.0f, 4.0f};

    roc::host_validation::hipblaslt_adapter::HostComparisonRequest request;
    request.rows             = 2;
    request.columns          = 1;
    request.leadingDimension = 2;
    request.batchStride      = 2;
    request.batchCount       = 2;
    request.expected         = expected.data();
    request.observed         = observed.data();
    request.type             = HIP_R_32F;
    request.pointwise = roc::host_validation::hipblaslt_adapter::HostPointwiseComparison::Unit;
    request.requireSpecialValueConsistency = true;
    request.computeRelativeFrobeniusError  = true;
    request.findAllCloseTolerance          = true;
    request.computeUnitsInLastPlace        = true;

    const auto report = roc::host_validation::hipblaslt_adapter::compareHost(request);
    EXPECT_TRUE(report.comparison.passed());
    EXPECT_EQ(report.comparison.nonFiniteMismatches, 0);
    EXPECT_DOUBLE_EQ(report.comparison.maximumUlp, 1.0);
    EXPECT_DOUBLE_EQ(report.unitsInLastPlaceComparison.maximumUlp, 1.0);
    EXPECT_DOUBLE_EQ(report.unitsInLastPlaceComparison.sumUlp, 1.0);
    EXPECT_EQ(report.unitsInLastPlaceComparison.ulpCompared, 4);
    EXPECT_GT(report.relativeFrobeniusError, 0.0);
    ASSERT_TRUE(report.allCloseTolerance);
    EXPECT_DOUBLE_EQ(report.allCloseTolerance->absolute, 1e-6);
    EXPECT_DOUBLE_EQ(report.allCloseTolerance->relative, 1e-6);
}

TEST(HostValidationComparisonBridge, KeepsReportedUlpNonFinitePolicySeparate)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();

    roc::host_validation::hipblaslt_adapter::HostComparisonRequest request;
    request.rows             = 1;
    request.columns          = 1;
    request.leadingDimension = 1;
    request.batchStride      = 1;
    request.batchCount       = 1;
    request.expected         = &nan;
    request.observed         = &nan;
    request.type             = HIP_R_32F;
    request.pointwise = roc::host_validation::hipblaslt_adapter::HostPointwiseComparison::Unit;
    request.requireSpecialValueConsistency = true;
    request.computeUnitsInLastPlace        = true;

    const auto report = roc::host_validation::hipblaslt_adapter::compareHost(request);
    EXPECT_TRUE(report.comparison.passed());
    EXPECT_EQ(report.comparison.nonFiniteMismatches, 0);
    EXPECT_TRUE(std::isinf(report.unitsInLastPlaceComparison.maximumUlp));
}

TEST(HostValidationComparisonBridge, EmptyPointwiseRequestsStillValidateTheProductType)
{
    using namespace roc::host_validation::hipblaslt_adapter;

    HostComparisonRequest request;
    request.type      = HIPBLASLT_DATATYPE_INVALID;
    request.pointwise = HostPointwiseComparison::Unit;
    EXPECT_THROW(compareHost(request), std::invalid_argument);

    request.pointwise                     = HostPointwiseComparison::Disabled;
    request.computeRelativeFrobeniusError = true;
    EXPECT_NO_THROW(compareHost(request));
}

TEST(HostValidationDataInitializationBridge, CounterBasedGenerationIsRepeatable)
{
    std::array<float, 16> first{};
    std::array<float, 16> second{};
    roc::host_validation::hipblaslt_adapter::initialize(std::span<float>(first),
                                                        hipblaslt_initialization::norm_dist);
    roc::host_validation::hipblaslt_adapter::initialize(std::span<float>(second),
                                                        hipblaslt_initialization::norm_dist);
    EXPECT_EQ(first, second);
}

TEST(HostValidationDataInitializationBridge, DeviceNormalGenerationIsRepeatable)
{
    constexpr size_t elements = 16;
    void*            firstDevice{};
    void*            secondDevice{};
    ASSERT_EQ(hipMalloc(&firstDevice, elements * sizeof(float)), hipSuccess);
    ASSERT_EQ(hipMalloc(&secondDevice, elements * sizeof(float)), hipSuccess);

    auto initializeDevice = [](void* device) {
        hipblaslt_init_device(ABC_dims::A,
                              hipblaslt_initialization::norm_dist,
                              false,
                              device,
                              4,
                              4,
                              4,
                              HIP_R_32F,
                              16,
                              1);
    };
    initializeDevice(firstDevice);
    initializeDevice(secondDevice);

    std::array<float, elements> first{};
    std::array<float, elements> second{};
    EXPECT_EQ(hipMemcpy(first.data(),
                        firstDevice,
                        first.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(hipMemcpy(second.data(),
                        secondDevice,
                        second.size() * sizeof(float),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(first, second);

    EXPECT_EQ(hipFree(firstDevice), hipSuccess);
    EXPECT_EQ(hipFree(secondDevice), hipSuccess);
}

TEST(HostValidationDataInitializationBridge, LegacyHostEntryPointsUseTensorLayouts)
{
    using Complex = std::complex<float>;
    std::array<Complex, 8> values;
    values.fill(Complex(-99, -99));

    hipblaslt_init_sin(values.data(), 2, 2, 3);
    EXPECT_EQ(values[0], Complex(std::sin(0.0f), std::cos(0.0f)));
    EXPECT_EQ(values[1], Complex(std::sin(1.0f), std::cos(1.0f)));
    EXPECT_EQ(values[3], Complex(std::sin(2.0f), std::cos(2.0f)));
    EXPECT_EQ(values[4], Complex(std::sin(3.0f), std::cos(3.0f)));
    EXPECT_EQ(values[2], Complex(-99, -99));

    hipblaslt_init_zero(values.data(), 2, 2, 3);
    EXPECT_EQ(values[0], Complex(0, 0));
    EXPECT_EQ(values[1], Complex(0, 0));
    EXPECT_EQ(values[3], Complex(0, 0));
    EXPECT_EQ(values[4], Complex(0, 0));
    EXPECT_EQ(values[2], Complex(-99, -99));
}

TEST(HostValidationDataInitializationBridge, LegacyRandomHelpersUseComponentRecipes)
{
    std::array<float, 8> values;
    values.fill(-99);

    hipblaslt_init(values.data(), 2, 2, 3);
    for(const size_t index : {size_t{0}, size_t{1}, size_t{3}, size_t{4}})
    {
        EXPECT_EQ(values[index], std::trunc(values[index]));
        EXPECT_GE(values[index], 1);
        EXPECT_LE(values[index], 10);
    }
    EXPECT_EQ(values[2], -99);

    values.fill(-99);
    hipblaslt_init_small(values.data(), 2, 2, 3);
    for(const size_t index : {size_t{0}, size_t{1}, size_t{3}, size_t{4}})
    {
        EXPECT_GE(values[index], 0.1f);
        EXPECT_LE(values[index], 1.0f);
        EXPECT_FLOAT_EQ(values[index] * 10,
                        std::round(values[index] * 10));
    }
    EXPECT_EQ(values[2], -99);

    values.fill(-99);
    hipblaslt_init_alternating_sign(values.data(), 2, 2, 3);
    EXPECT_LT(values[0], 0);
    EXPECT_GT(values[1], 0);
    EXPECT_GT(values[3], 0);
    EXPECT_LT(values[4], 0);
    EXPECT_EQ(values[2], -99);

    values.fill(-99);
    hipblaslt_init_hpl(values.data(), 2, 2, 3);
    for(const size_t index : {size_t{0}, size_t{1}, size_t{3}, size_t{4}})
    {
        EXPECT_GE(values[index], -0.5f);
        EXPECT_LE(values[index], 0.5f);
    }
    EXPECT_EQ(values[2], -99);

    hipblaslt_init_nan(values.data(), values.size());
    for(const float value : values)
        EXPECT_TRUE(std::isnan(value));
}

TEST(HostValidationDataInitializationBridge, GeneratesProblemLevelMatrixRecipes)
{
    using namespace roc::host_validation;
    using namespace roc::host_validation::hipblaslt_adapter;

    MatrixStorageInitialization exact;
    exact.role             = MatrixRole::B;
    exact.initialization   = hipblaslt_initialization::integer_exact;
    exact.type             = HIP_R_32F;
    exact.rows             = 2;
    exact.columns          = 3;
    exact.leadingDimension = 4;
    exact.batchStride      = 12;
    exact.batchCount       = 2;
    std::vector<std::byte> exactStorage = generateMatrixStorage(exact);
    TensorView exactView(
        ScalarType::Float32,
        Layout(Shape{2, 3, 2}, {1, 4, 12}),
        exactStorage);
    for(size_t batch = 0; batch < 2; ++batch)
        for(size_t column = 0; column < 3; ++column)
            for(size_t row = 0; row < 2; ++row)
            {
                const float value
                    = exactView.loadAs<float>({row, column, batch});
                EXPECT_EQ(value, std::trunc(value));
                EXPECT_LE(std::abs(value), 2);
                if(value != 0)
                    EXPECT_EQ(value > 0, ((row ^ column) & 1U) != 0);
            }
    TensorView exactAllocation(
        ScalarType::Float32,
        Layout(Shape{4, 3, 2}, {1, 4, 12}),
        exactStorage);
    for(size_t batch = 0; batch < 2; ++batch)
        for(size_t column = 0; column < 3; ++column)
            for(size_t row = 2; row < 4; ++row)
                EXPECT_EQ(
                    exactAllocation.loadAs<float>({row, column, batch}), 0);

    MatrixStorageInitialization probe;
    probe.role             = MatrixRole::B;
    probe.initialization   = hipblaslt_initialization::fp16_accumulator_probe;
    probe.type             = HIP_R_16F;
    probe.rows             = 4;
    probe.columns          = 2;
    probe.leadingDimension = 4;
    std::vector<std::byte> probeStorage = generateMatrixStorage(probe);
    TensorView probeView(
        ScalarType::Float16,
        Layout(Shape{4, 2, 1}, {1, 4, 0}),
        probeStorage);
    for(size_t column = 0; column < 2; ++column)
        for(size_t row = 0; row < 4; ++row)
            EXPECT_EQ(probeView.loadAs<float>({row, column, 0}),
                      row % 2 == 0 ? 2 : -2);

    MatrixStorageInitialization oneSpecial;
    oneSpecial.role             = MatrixRole::A;
    oneSpecial.initialization
        = hipblaslt_initialization::norm_dist_one_special;
    oneSpecial.specialValueType = 0;
    oneSpecial.type             = HIP_R_32F;
    oneSpecial.rows             = 4;
    oneSpecial.columns          = 3;
    oneSpecial.leadingDimension = 4;
    oneSpecial.batchStride      = 12;
    oneSpecial.batchCount       = 2;
    std::vector<std::byte> specialStorage
        = generateMatrixStorage(oneSpecial);
    TensorView specialView(
        ScalarType::Float32,
        Layout(Shape{4, 3, 2}, {1, 4, 12}),
        specialStorage);
    size_t infinityCount = 0;
    for(size_t batch = 0; batch < 2; ++batch)
        for(size_t column = 0; column < 3; ++column)
            for(size_t row = 0; row < 4; ++row)
                infinityCount += std::isinf(
                    specialView.loadAs<float>({row, column, batch}));
    EXPECT_EQ(infinityCount, 1);
}

TEST(HostValidationDataInitializationBridge, HostSideDeviceFillCopiesComponentStorage)
{
    using namespace roc::host_validation::hipblaslt_adapter;

    MatrixStorageInitialization initialization;
    initialization.role             = MatrixRole::B;
    initialization.initialization   = hipblaslt_initialization::integer_exact;
    initialization.type             = HIP_R_32F;
    initialization.rows             = 2;
    initialization.columns          = 3;
    initialization.leadingDimension = 4;
    initialization.batchStride      = 12;
    initialization.batchCount       = 2;
    const std::vector<std::byte> expected
        = generateMatrixStorage(initialization);

    void* device = nullptr;
    ASSERT_EQ(hipMalloc(&device, expected.size()), hipSuccess);
    struct HostFillStateGuard
    {
        HostFillStateGuard()
        {
            set_host_side_fill_kernel_state(true);
        }
        ~HostFillStateGuard()
        {
            set_host_side_fill_kernel_state(false);
        }
    } guard;

    hipblaslt_init_device(ABC_dims::B,
                          initialization.initialization,
                          false,
                          device,
                          initialization.rows,
                          initialization.columns,
                          initialization.leadingDimension,
                          initialization.type,
                          initialization.batchStride,
                          initialization.batchCount);
    std::vector<std::byte> observed(expected.size());
    EXPECT_EQ(hipMemcpy(observed.data(),
                        device,
                        observed.size(),
                        hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_EQ(observed, expected);
    EXPECT_EQ(hipFree(device), hipSuccess);
}

TEST(HostValidationCblasBridge, DistinctHalfCAndFloatD)
{
    const std::array<float, 6>   a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6>   b{7, 9, 11, 8, 10, 12};
    std::array<hipblasLtHalf, 6> c{1, 2, -99, 3, 4, -99};
    const auto                   originalC = c;
    std::array<float, 4>         d{-1, -2, -3, -4};

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    2,
                                    2,
                                    3,
                                    2,
                                    a.data(),
                                    2,
                                    b.data(),
                                    3,
                                    3,
                                    c.data(),
                                    3,
                                    d.data(),
                                    2,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    1,
                                    false,
                                    false,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_16F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F);

    for(size_t index = 0; index < c.size(); ++index)
        EXPECT_FLOAT_EQ(static_cast<float>(c[index]), static_cast<float>(originalC[index]));
    EXPECT_FLOAT_EQ(d[0], 2 * 58 + 3 * static_cast<float>(originalC[0]));
    EXPECT_FLOAT_EQ(d[1], 2 * 139 + 3 * static_cast<float>(originalC[1]));
    EXPECT_FLOAT_EQ(d[2], 2 * 64 + 3 * static_cast<float>(originalC[3]));
    EXPECT_FLOAT_EQ(d[3], 2 * 154 + 3 * static_cast<float>(originalC[4]));
}

TEST(HostValidationCblasBridge, MixedHalfInputs)
{
    const std::array<hipblasLtHalf, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<hipblasLtHalf, 6> b{7, 9, 11, 8, 10, 12};
    std::array<float, 4>               d{};

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    2,
                                    2,
                                    3,
                                    1,
                                    a.data(),
                                    2,
                                    b.data(),
                                    3,
                                    0,
                                    d.data(),
                                    2,
                                    d.data(),
                                    2,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    1,
                                    false,
                                    false,
                                    HIP_R_16F,
                                    HIP_R_16F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_16F,
                                    HIP_R_16F);

    EXPECT_FLOAT_EQ(d[0], 58);
    EXPECT_FLOAT_EQ(d[1], 139);
    EXPECT_FLOAT_EQ(d[2], 64);
    EXPECT_FLOAT_EQ(d[3], 154);
}

TEST(HostValidationCblasBridge, QuantizesCombinedOperandScaleAndAlphaVector)
{
    const std::array<float, 1> a{0.3f};
    const std::array<float, 1> b{1.0f};
    std::array<float, 1>       d{};
    const std::array<float, 1> alphaVector{0.6f};
    const std::array<float, 1> scaleA{0.7f};

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    1,
                                    1,
                                    1,
                                    1.0f,
                                    a.data(),
                                    1,
                                    b.data(),
                                    1,
                                    0.0f,
                                    d.data(),
                                    1,
                                    d.data(),
                                    1,
                                    alphaVector.data(),
                                    scaleA.data(),
                                    nullptr,
                                    1.0f,
                                    false,
                                    false,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_8F_E4M3,
                                    HIP_R_32F);

    const float expected =
        static_cast<float>(hipblaslt_f8(a[0] * scaleA[0] * alphaVector[0]));
    EXPECT_FLOAT_EQ(d[0], expected);
}

TEST(HostValidationCblasBridge, AppliesOutputScaleBeforeNarrowConversion)
{
    const std::array<float, 1> a{0.3333f};
    const std::array<float, 1> b{3.0f};
    std::array<hipblasLtHalf, 1> d{hipblasLtHalf(0.0f)};

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    1,
                                    1,
                                    1,
                                    1.0f,
                                    a.data(),
                                    1,
                                    b.data(),
                                    1,
                                    0.0f,
                                    d.data(),
                                    1,
                                    d.data(),
                                    1,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    0.1f,
                                    false,
                                    false,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_16F,
                                    HIP_R_16F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F);

    const hipblasLtHalf expected((a[0] * b[0]) * 0.1f);
    EXPECT_FLOAT_EQ(static_cast<float>(d[0]), static_cast<float>(expected));
}

TEST(HostValidationCblasBridge, ConvertsFnuzOutputWithComponentCodec)
{
    const std::array<float, 1> a{1.3f};
    const std::array<float, 1> b{1.0f};
    std::array<hipblaslt_f8_fnuz, 1> d{hipblaslt_f8_fnuz(0.0f)};

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    1,
                                    1,
                                    1,
                                    1.0f,
                                    a.data(),
                                    1,
                                    b.data(),
                                    1,
                                    0.0f,
                                    d.data(),
                                    1,
                                    d.data(),
                                    1,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    1.0f,
                                    false,
                                    false,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_8F_E4M3_FNUZ,
                                    HIP_R_8F_E4M3_FNUZ,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F);

    EXPECT_EQ(d[0], hipblaslt_f8_fnuz(a[0]));
}

TEST(HostValidationCblasBridge, SaturatesRoundedInt8Output)
{
    const std::array<float, 1> a{63.75f};
    const std::array<float, 1> b{2.0f};
    std::array<int8_t, 1>      d{};

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    1,
                                    1,
                                    1,
                                    1.0f,
                                    a.data(),
                                    1,
                                    b.data(),
                                    1,
                                    0.0f,
                                    d.data(),
                                    1,
                                    d.data(),
                                    1,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    1.0f,
                                    false,
                                    false,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_8I,
                                    HIP_R_8I,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F);

    EXPECT_EQ(d[0], 127);
}

TEST(HostValidationCblasBridge, IntegerComputeUsesWideReferenceAndSaturatingOutput)
{
    const std::array<int8_t, 2> a{100, 100};
    const std::array<int8_t, 2> b{1, 1};
    std::array<int8_t, 1>       d{};

    hipblaslt_reference_gemm<int32_t>(HIPBLAS_OP_N,
                                      HIPBLAS_OP_N,
                                      1,
                                      1,
                                      2,
                                      1,
                                      a.data(),
                                      1,
                                      b.data(),
                                      2,
                                      0,
                                      d.data(),
                                      1,
                                      d.data(),
                                      1,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      1,
                                      false,
                                      false,
                                      HIP_R_8I,
                                      HIP_R_8I,
                                      HIP_R_8I,
                                      HIP_R_8I,
                                      HIP_R_32I,
                                      HIP_R_32I,
                                      HIP_R_32I);

    EXPECT_EQ(d[0], 127);
}

TEST(HostValidationCblasBridge, TransposedPaddedScaleUsesLogicalRows)
{
    // Stored A is K x M with one padding element after each column.
    const std::array<float, 6> a{1.0f, 2.0f, -99.0f, 3.0f, 4.0f, -99.0f};
    const std::array<float, 2> b{5.0f, 6.0f};
    const std::array<float, 2> scaleA{2.0f, 3.0f};
    std::array<float, 2>       d{};

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_T,
                                    HIPBLAS_OP_N,
                                    2,
                                    1,
                                    2,
                                    1.0f,
                                    a.data(),
                                    3,
                                    b.data(),
                                    2,
                                    0.0f,
                                    d.data(),
                                    2,
                                    d.data(),
                                    2,
                                    nullptr,
                                    scaleA.data(),
                                    nullptr,
                                    1.0f,
                                    true,
                                    false,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F);

    EXPECT_FLOAT_EQ(d[0], (1.0f * 5.0f + 2.0f * 6.0f) * 2.0f);
    EXPECT_FLOAT_EQ(d[1], (3.0f * 5.0f + 4.0f * 6.0f) * 3.0f);
}

#if defined(HIPBLASLT_USE_FP4)
TEST(HostValidationCblasBridge, PackedFloat4InputUsesLogicalElementLayout)
{
    const std::array<hipblaslt_f4x2, 1> a{hipblaslt_f4x2(1.0f, 2.0f)};
    const std::array<float, 2>           b{3.0f, 4.0f};
    std::array<float, 1>                 d{};

    hipblaslt_reference_gemm<float>(
        HIPBLAS_OP_N,
        HIPBLAS_OP_N,
        1,
        1,
        2,
        1.0f,
        a.data(),
        1,
        b.data(),
        2,
        0.0f,
        d.data(),
        1,
        d.data(),
        1,
        nullptr,
        nullptr,
        nullptr,
        1.0f,
        false,
        false,
        static_cast<hipDataType>(HIP_R_4F_E2M1),
        HIP_R_32F,
        HIP_R_32F,
        HIP_R_32F,
        HIP_R_32F,
        static_cast<hipDataType>(HIP_R_4F_E2M1),
        HIP_R_32F);

    EXPECT_FLOAT_EQ(d[0], 11.0f);
}
#endif

TEST(HostValidationCblasBridge, ComplexConjugateTranspose)
{
    using Complex = std::complex<float>;

    const std::array<Complex, 2> a{Complex(1, 2), Complex(3, -1)};
    const std::array<Complex, 2> b{Complex(2, -1), Complex(-4, 3)};
    std::array<Complex, 1>       d{Complex(0, 0)};

    hipblaslt_reference_gemm<Complex>(HIPBLAS_OP_C,
                                      HIPBLAS_OP_N,
                                      1,
                                      1,
                                      2,
                                      Complex(1, 0),
                                      a.data(),
                                      2,
                                      b.data(),
                                      2,
                                      Complex(0, 0),
                                      d.data(),
                                      1,
                                      d.data(),
                                      1,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      Complex(1, 0),
                                      false,
                                      false,
                                      HIP_C_32F,
                                      HIP_C_32F,
                                      HIP_C_32F,
                                      HIP_C_32F,
                                      HIP_C_32F,
                                      HIP_C_32F,
                                      HIP_C_32F);

    const Complex expected = std::conj(a[0]) * b[0] + std::conj(a[1]) * b[1];
    EXPECT_FLOAT_EQ(d[0].real(), expected.real());
    EXPECT_FLOAT_EQ(d[0].imag(), expected.imag());
}

TEST(HostValidationCblasBridge, LargeProblemUsesAcceleratedBackend)
{
    constexpr int64_t  m = 601;
    std::vector<float> a(m);
    for(int64_t row = 0; row < m; ++row)
        a[row] = static_cast<float>(row % 7);
    const std::array<float, 1> b{2};
    std::vector<float>         d(m, 1);

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    m,
                                    1,
                                    1,
                                    3,
                                    a.data(),
                                    m,
                                    b.data(),
                                    1,
                                    4,
                                    d.data(),
                                    m,
                                    d.data(),
                                    m,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    1,
                                    false,
                                    false,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F);

    for(int64_t row = 0; row < m; ++row)
        EXPECT_FLOAT_EQ(d[row], 6 * a[row] + 4);
}

TEST(HostValidationCblasBridge, ZeroReductionDoesNotRequireBlasOperands)
{
    std::array<float, 1> d{2.0f};

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    1,
                                    1,
                                    0,
                                    0.0f,
                                    nullptr,
                                    601,
                                    nullptr,
                                    601,
                                    3.0f,
                                    d.data(),
                                    1,
                                    d.data(),
                                    1,
                                    nullptr,
                                    nullptr,
                                    nullptr,
                                    1.0f,
                                    false,
                                    false,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F);

    EXPECT_FLOAT_EQ(d[0], 6.0f);
}
