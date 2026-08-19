// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipBuffer.hpp>
#include <hipblaslt/host_validation/GroupedGemmDataInitialization.hpp>
#include <hipblaslt/host_validation/HipblasltDataInitialization.hpp>
#include <hipblaslt/host_validation/HipblasltReferenceGemm.hpp>
#include <hipblaslt/host_validation/HostComparison.hpp>
#include <hipblaslt/host_validation/MatrixTransformReference.hpp>
#include <hipblaslt/host_validation/hipblaslt_init.hpp>
#include <hipblaslt/host_validation/near.hpp>

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <span>
#include <utility>
#include <vector>

TEST(HostValidationTensorStorage, PooledPinnedAllocatorBacksTensorAliases)
{
    using namespace roc::host_validation;

    Tensor tensor(ScalarType::Float32, Shape{2}, HipHostBuffer::tensorAllocator());
    tensor.storeFrom({1}, 7.0f);
    Tensor alias = tensor;
    EXPECT_EQ(alias.storage().data(), tensor.storage().data());
    EXPECT_EQ(alias.loadAs<float>({1}), 7.0f);

    Tensor clone = tensor.clone(HipHostBuffer::tensorAllocator());
    EXPECT_NE(clone.storage().data(), tensor.storage().data());
    clone.storeFrom({1}, 11.0f);
    EXPECT_EQ(tensor.loadAs<float>({1}), 7.0f);
    EXPECT_EQ(clone.loadAs<float>({1}), 11.0f);
}

TEST(HostValidationTensorStorage, HipHostBufferTensorRetainsAndMutatesThePinnedAllocation)
{
    using namespace roc::host_validation;

    Tensor tensor = [] {
        HipHostBuffer buffer(HIP_R_32F, 2);
        Tensor        wrapped = buffer.tensor(ScalarType::Float32, Layout::contiguous(Shape{2}));
        wrapped.storeFrom({1}, 7.0f);
        EXPECT_EQ(buffer.as<float>()[1], 7.0f);
        return wrapped;
    }();

    EXPECT_EQ(tensor.loadAs<float>({1}), 7.0f);
    Tensor alias = tensor;
    alias.storeFrom({0}, 3.0f);
    EXPECT_EQ(tensor.loadAs<float>({0}), 3.0f);
}

TEST(HostValidationTypeBridge, UsesScalarTypeAsTheExternalTypeConversionHub)
{
    using hipblaslt::host_validation::hipDataTypeForScalarType;
    using hipblaslt::host_validation::scalarType;
    using roc::host_validation::ScalarType;

    constexpr std::array mappings{
        std::pair{ScalarType::Float4E2M1, static_cast<hipDataType>(HIP_R_4F_E2M1_EXT)},
        std::pair{ScalarType::Float6E2M3, static_cast<hipDataType>(HIP_R_6F_E2M3_EXT)},
        std::pair{ScalarType::Float6E3M2, static_cast<hipDataType>(HIP_R_6F_E3M2_EXT)},
        std::pair{ScalarType::Float8E4M3, HIP_R_8F_E4M3},
        std::pair{ScalarType::Float8E5M2, HIP_R_8F_E5M2},
        std::pair{ScalarType::E8M0, HIP_R_8F_UE8M0},
        std::pair{ScalarType::E5M3, static_cast<hipDataType>(HIP_R_8F_E5M3_EXT)},
    };
    for(const auto& [scalar, hip] : mappings)
    {
        EXPECT_EQ(hipDataTypeForScalarType(scalar), hip);
        EXPECT_EQ(scalarType(hip), scalar);
    }

    EXPECT_EQ(hipDataTypeForScalarType(ScalarType::E4M3), HIP_R_8F_E4M3);
    EXPECT_THROW(hipDataTypeForScalarType(ScalarType::Int12), std::invalid_argument);
}

TEST(HostValidationDataInitializationBridge, GeneratesComplexTrigonometricValues)
{
    std::array<std::complex<float>, 4> values{};
    hipblaslt::host_validation::initialize(
        std::span<std::complex<float>>(values),
        hipblaslt_initialization::trig_float,
        hipblaslt::host_validation::TrigonometricComponent::Sine);

    for(size_t index = 0; index < values.size(); ++index)
    {
        EXPECT_FLOAT_EQ(values[index].real(), std::sin(static_cast<float>(index)));
        EXPECT_FLOAT_EQ(values[index].imag(), std::cos(static_cast<float>(index)));
    }
}

TEST(HostValidationDataInitializationBridge, ComplexRandomUsesTypedCartesianDomains)
{
    using namespace roc::host_validation;

    std::array<std::complex<float>, 8> first{};
    std::array<std::complex<float>, 8> second{};
    hipblaslt::host_validation::initialize(std::span<std::complex<float>>(first),
                                           hipblaslt_initialization::rand_int);
    hipblaslt::host_validation::initialize(std::span<std::complex<float>>(second),
                                           hipblaslt_initialization::rand_int);
    EXPECT_EQ(first, second);

    for(size_t index = 0; index < first.size(); ++index)
    {
        EXPECT_EQ(first[index].real(),
                  indexedUniformInteger(hipblaslt::host_validation::defaultInitializationSeed,
                                        generation_random_domain_version_1::realComponent,
                                        index,
                                        1,
                                        10));
        EXPECT_EQ(first[index].imag(),
                  indexedUniformInteger(hipblaslt::host_validation::defaultInitializationSeed,
                                        generation_random_domain_version_1::imaginaryComponent,
                                        index,
                                        1,
                                        10));
    }
}

TEST(HostValidationDataInitializationBridge, GroupedGemmUsesStableRoleSequencesAndDefaultSeed)
{
    std::vector<float> a(5);
    std::vector<float> b(7);
    std::vector<float> c(4);
    std::vector<float> bias(3);

    hipblaslt::host_validation::initializeGroupedGemm(a,
                                                      static_cast<int64_t>(a.size()),
                                                      b,
                                                      static_cast<int64_t>(b.size()),
                                                      c,
                                                      static_cast<int64_t>(c.size()),
                                                      bias,
                                                      static_cast<int64_t>(bias.size()),
                                                      hipblaslt_initialization::rand_int);

    constexpr uint64_t seed = hipblaslt::host_validation::defaultInitializationSeed;
    const auto expected = [seed](uint64_t sequence, size_t index) {
        return roc::host_validation::indexedUniformInteger(
            hipblaslt::host_validation::initialization::seedForSequence(seed, sequence),
            roc::host_validation::generation_random_domain_version_1::realComponent,
            index,
            1,
            10);
    };
    for(size_t index = 0; index < a.size(); ++index)
        EXPECT_EQ(a[index], expected(0, index));
    for(size_t index = 0; index < b.size(); ++index)
    {
        const int magnitude = expected(1, index);
        EXPECT_EQ(b[index], (index & 1U) == 0 ? -magnitude : magnitude);
    }
    for(size_t index = 0; index < c.size(); ++index)
        EXPECT_EQ(c[index], expected(2, index));
    for(size_t index = 0; index < bias.size(); ++index)
        EXPECT_EQ(bias[index], expected(3, index));
}

TEST(HostValidationDataInitializationBridge, InitializationSeedDependsOnSeedAndSequence)
{
    using hipblaslt::host_validation::initialization::seedForSequence;

    constexpr uint64_t seed     = 0x123456789abcdef0ULL;
    constexpr uint64_t sequence = 0x1020304050607080ULL;
    EXPECT_EQ(seedForSequence(seed, sequence), seedForSequence(seed, sequence));
    EXPECT_NE(seedForSequence(seed, sequence), seedForSequence(seed, sequence + 1));
    EXPECT_NE(seedForSequence(seed, sequence), seedForSequence(seed + 1, sequence));
}

TEST(HostValidationDataInitializationBridge, GroupedGemmPropagatesCallerSeed)
{
    std::vector<float> a(2);
    std::vector<float> b(2);
    std::vector<float> c(2);
    std::vector<float> bias(2);
    constexpr uint64_t seed = 0x123456789abcdef0ULL;

    hipblaslt::host_validation::initializeGroupedGemm(a,
                                                      static_cast<int64_t>(a.size()),
                                                      b,
                                                      static_cast<int64_t>(b.size()),
                                                      c,
                                                      static_cast<int64_t>(c.size()),
                                                      bias,
                                                      static_cast<int64_t>(bias.size()),
                                                      hipblaslt_initialization::rand_int,
                                                      seed);

    const auto expected = [seed](uint64_t sequence, size_t index) {
        return roc::host_validation::indexedUniformInteger(
            hipblaslt::host_validation::initialization::seedForSequence(seed, sequence),
            roc::host_validation::generation_random_domain_version_1::realComponent,
            index,
            1,
            10);
    };
    for(size_t index = 0; index < a.size(); ++index)
        EXPECT_EQ(a[index], expected(0, index));
    for(size_t index = 0; index < b.size(); ++index)
    {
        const int magnitude = expected(1, index);
        EXPECT_EQ(b[index], (index & 1U) == 0 ? -magnitude : magnitude);
    }
    for(size_t index = 0; index < c.size(); ++index)
        EXPECT_EQ(c[index], expected(2, index));
    for(size_t index = 0; index < bias.size(); ++index)
        EXPECT_EQ(bias[index], expected(3, index));
}

TEST(HostValidationDataInitializationBridge, GroupedGemmDefinesHplAndSpecialRecipes)
{
    std::vector<float> a(4);
    std::vector<float> b(4);
    std::vector<float> c(4);
    std::vector<float> bias(4);

    const auto initialize = [&](hipblaslt_initialization initialization) {
        hipblaslt::host_validation::initializeGroupedGemm(a,
                                                          static_cast<int64_t>(a.size()),
                                                          b,
                                                          static_cast<int64_t>(b.size()),
                                                          c,
                                                          static_cast<int64_t>(c.size()),
                                                          bias,
                                                          static_cast<int64_t>(bias.size()),
                                                          initialization);
    };
    const auto expectHplRange = [](const auto& values) {
        for(const float value : values)
        {
            EXPECT_GE(value, -0.5f);
            EXPECT_LE(value, 0.5f);
        }
    };

    initialize(hipblaslt_initialization::hpl);
    expectHplRange(a);
    expectHplRange(b);
    expectHplRange(c);
    expectHplRange(bias);

    initialize(hipblaslt_initialization::special);
    for(const float value : a)
        EXPECT_EQ(value, hipblaslt::host_validation::specialInitializationAValue);
    for(const float value : b)
        EXPECT_EQ(value, hipblaslt::host_validation::specialInitializationBValue);
    expectHplRange(c);
    expectHplRange(bias);
}

TEST(HostValidationMatrixTransformBridge, MapsLayoutsAndTransposes)
{
    constexpr size_t      rows        = 2;
    constexpr size_t      columns     = 3;
    constexpr size_t      batches     = 2;
    constexpr size_t      batchStride = 12;
    std::array<float, 20> a{};
    std::array<float, 20> b{};
    std::array<float, 20> observed{};

    for(size_t batch = 0; batch < batches; ++batch)
    {
        for(size_t row = 0; row < rows; ++row)
        {
            for(size_t column = 0; column < columns; ++column)
            {
                const float aValue = static_cast<float>(1 + row + 2 * column + 3 * batch);
                const float bValue = static_cast<float>(2 - static_cast<int>(row) + column + batch);
                a[batch * batchStride + 2 * column + row]        = aValue;
                b[batch * batchStride + 4 * row + column]        = bValue;
                observed[batch * batchStride + row + 3 * column] = 2.0f * aValue - bValue;
            }
        }
    }

    hipblaslt::host_validation::MatrixTransformReferenceArguments arguments;
    arguments.observed               = observed.data();
    arguments.observedStorageBytes   = sizeof(observed);
    arguments.a                      = a.data();
    arguments.aStorageBytes          = sizeof(a);
    arguments.b                      = b.data();
    arguments.bStorageBytes          = sizeof(b);
    arguments.type                   = HIP_R_32F;
    arguments.rows                   = rows;
    arguments.columns                = columns;
    arguments.batchCount             = batches;
    arguments.leadingDimensionA      = 2;
    arguments.leadingDimensionB      = 4;
    arguments.leadingDimensionOutput = 3;
    arguments.batchStride            = batchStride;
    arguments.rowMajorA              = true;
    arguments.rowMajorB              = true;
    arguments.rowMajorOutput         = false;
    arguments.transposeA             = true;
    arguments.alpha                  = 2.0;
    arguments.beta                   = -1.0;

    const auto result = hipblaslt::host_validation::referenceMatrixTransform(arguments);
    EXPECT_EQ(result.runInfo.outputElementsWritten, rows * columns * batches);
    EXPECT_TRUE(result.comparison.passed());
}

TEST(HostValidationComparisonBridge, FindsAllcloseToleranceAcrossBatches)
{
    const std::array<float, 4>                        expected{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4>                        observed{1.0f, 2.00009f, 3.0f, 4.0f};
    hipblaslt::host_validation::HostComparisonRequest request;
    request.rows                  = 2;
    request.columns               = 1;
    request.leadingDimension      = 2;
    request.batchStride           = 2;
    request.batchCount            = 2;
    request.expected              = expected.data();
    request.observed              = observed.data();
    request.type                  = HIP_R_32F;
    request.findAllCloseTolerance = true;

    const auto report = hipblaslt::host_validation::compareHost(request);
    ASSERT_TRUE(report.allCloseTolerance);
    EXPECT_EQ(report.allCloseTolerance->absolute, 1e-6);
    EXPECT_EQ(report.allCloseTolerance->relative, 1e-4);
}

TEST(HostValidationComparisonBridge, UsesMagnitudeForComplexAllcloseToleranceSearch)
{
    const std::complex<float> expected{0.0f, 0.0f};
    const std::complex<float> observed{0.09f, 0.09f};

    hipblaslt::host_validation::HostComparisonRequest request;
    request.rows                  = 1;
    request.columns               = 1;
    request.leadingDimension      = 1;
    request.batchStride           = 1;
    request.batchCount            = 1;
    request.expected              = &expected;
    request.observed              = &observed;
    request.type                  = HIP_C_32F;
    request.findAllCloseTolerance = true;

    EXPECT_FALSE(hipblaslt::host_validation::compareHost(request).allCloseTolerance);
}

TEST(HostValidationComparisonBridge, ComputesRelativeFrobeniusEvidence)
{
    std::array<double, 2>                             expected{3.0, 4.0};
    std::array<double, 2>                             observed{0.0, 4.0};
    hipblaslt::host_validation::HostComparisonRequest request;
    request.rows                          = 2;
    request.columns                       = 1;
    request.leadingDimension              = 2;
    request.batchStride                   = 2;
    request.batchCount                    = 1;
    request.expected                      = expected.data();
    request.observed                      = observed.data();
    request.type                          = HIP_R_64F;
    request.computeRelativeFrobeniusError = true;
    EXPECT_DOUBLE_EQ(hipblaslt::host_validation::compareHost(request).relativeFrobeniusError, 0.6);
}

TEST(HostValidationComparisonBridge, UnitNearAndSpecialValuePolicies)
{
    const float          oneUlp = std::nextafter(1.0f, 2.0f);
    std::array<float, 2> expected{1.0f, std::numeric_limits<float>::infinity()};
    std::array<float, 2> observed{oneUlp, std::numeric_limits<float>::infinity()};

    hipblaslt::host_validation::HostComparisonRequest request;
    request.rows             = 2;
    request.columns          = 1;
    request.leadingDimension = 2;
    request.batchStride      = 2;
    request.batchCount       = 1;
    request.expected         = expected.data();
    request.observed         = observed.data();
    request.type             = HIP_R_32F;

    request.pointwise = hipblaslt::host_validation::HostPointwiseComparison::Unit;
    EXPECT_TRUE(hipblaslt::host_validation::compareHost(request).comparison.passed());

    request.pointwise         = hipblaslt::host_validation::HostPointwiseComparison::Near;
    request.absoluteTolerance = 1e-6;
    EXPECT_TRUE(hipblaslt::host_validation::compareHost(request).comparison.passed());

    expected[0]       = 0.0f;
    observed[0]       = 2.0f * std::numeric_limits<float>::epsilon();
    request.pointwise = hipblaslt::host_validation::HostPointwiseComparison::SymmetricRelative;
    request.symmetricRelativeTolerance = 3.0f * std::numeric_limits<float>::epsilon();
    EXPECT_TRUE(hipblaslt::host_validation::compareHost(request).comparison.passed());
    request.symmetricRelativeTolerance = std::numeric_limits<float>::epsilon();
    EXPECT_FALSE(hipblaslt::host_validation::compareHost(request).comparison.passed());

    request.pointwise = hipblaslt::host_validation::HostPointwiseComparison::Disabled;
    request.requireSpecialValueConsistency = true;
    EXPECT_EQ(hipblaslt::host_validation::compareHost(request).comparison.nonFiniteMismatches, 0);
}

TEST(HostValidationComparisonBridge, RunsTheCombinedHostComparisonProgram)
{
    const float                oneUlp = std::nextafter(1.0f, 2.0f);
    const std::array<float, 4> expected{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> observed{oneUlp, 2.0f, 3.0f, 4.0f};

    hipblaslt::host_validation::HostComparisonRequest request;
    request.rows             = 2;
    request.columns          = 1;
    request.leadingDimension = 2;
    request.batchStride      = 2;
    request.batchCount       = 2;
    request.expected         = expected.data();
    request.observed         = observed.data();
    request.type             = HIP_R_32F;
    request.pointwise        = hipblaslt::host_validation::HostPointwiseComparison::Unit;
    request.requireSpecialValueConsistency = true;
    request.computeRelativeFrobeniusError  = true;
    request.findAllCloseTolerance          = true;
    request.computeUnitsInLastPlace        = true;

    const auto report = hipblaslt::host_validation::compareHost(request);
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

    hipblaslt::host_validation::HostComparisonRequest request;
    request.rows             = 1;
    request.columns          = 1;
    request.leadingDimension = 1;
    request.batchStride      = 1;
    request.batchCount       = 1;
    request.expected         = &nan;
    request.observed         = &nan;
    request.type             = HIP_R_32F;
    request.pointwise        = hipblaslt::host_validation::HostPointwiseComparison::Unit;
    request.requireSpecialValueConsistency = true;
    request.computeUnitsInLastPlace        = true;

    const auto report = hipblaslt::host_validation::compareHost(request);
    EXPECT_TRUE(report.comparison.passed());
    EXPECT_EQ(report.comparison.nonFiniteMismatches, 0);
    EXPECT_TRUE(std::isinf(report.unitsInLastPlaceComparison.maximumUlp));
}

TEST(HostValidationComparisonBridge, EmptyPointwiseRequestsStillValidateTheProductType)
{
    using namespace hipblaslt::host_validation;

    HostComparisonRequest request;
    request.type      = HIPBLASLT_DATATYPE_INVALID;
    request.pointwise = HostPointwiseComparison::Unit;
    EXPECT_THROW(compareHost(request), std::invalid_argument);

    request.pointwise                     = HostPointwiseComparison::Disabled;
    request.computeRelativeFrobeniusError = true;
    EXPECT_NO_THROW(compareHost(request));
}

TEST(HostValidationTolerancePolicy, Gfx11ScalesComputeTypeEpsilon)
{
    EXPECT_DOUBLE_EQ(sum_error_tolerance_for_compute_type(HIP_R_32F),
                     std::numeric_limits<float>::epsilon());
    EXPECT_DOUBLE_EQ(sum_error_tolerance_for_compute_type(HIP_R_16F),
                     std::numeric_limits<hipblasLtHalf>::epsilon());
    EXPECT_DOUBLE_EQ(gfx11_low_precision_accumulation_tolerance_coefficient(HIP_R_32F, 8),
                     64.0 * std::numeric_limits<float>::epsilon());
}

TEST(HostValidationDataInitializationBridge, CounterBasedGenerationIsRepeatable)
{
    std::array<float, 16> first{};
    std::array<float, 16> second{};
    hipblaslt::host_validation::initialize(std::span<float>(first),
                                           hipblaslt_initialization::norm_dist);
    hipblaslt::host_validation::initialize(std::span<float>(second),
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
    EXPECT_EQ(
        hipMemcpy(first.data(), firstDevice, first.size() * sizeof(float), hipMemcpyDeviceToHost),
        hipSuccess);
    EXPECT_EQ(
        hipMemcpy(
            second.data(), secondDevice, second.size() * sizeof(float), hipMemcpyDeviceToHost),
        hipSuccess);
    EXPECT_EQ(first, second);

    EXPECT_EQ(hipFree(firstDevice), hipSuccess);
    EXPECT_EQ(hipFree(secondDevice), hipSuccess);
}

TEST(HostValidationDataInitializationBridge, HostEntryPointsUseTensorLayouts)
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

TEST(HostValidationDataInitializationBridge, RandomHelpersUseComponentRecipes)
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
        EXPECT_FLOAT_EQ(values[index] * 10, std::round(values[index] * 10));
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

TEST(HostValidationDataInitializationBridge, RuntimeDispatchSupportsEveryFp8Encoding)
{
    constexpr std::array<hipDataType, 4> fp8Types{
        HIP_R_8F_E4M3_FNUZ,
        HIP_R_8F_E5M2_FNUZ,
        HIP_R_8F_E4M3,
        HIP_R_8F_E5M2,
    };

    for(const hipDataType type : fp8Types)
    {
        uint8_t value = 0xff;
        hipblaslt_init_zero(static_cast<void*>(&value), 1, 1, 1, type);
        EXPECT_EQ(value, 0) << "hipDataType=" << static_cast<int>(type);
    }
}

TEST(HostValidationDataInitializationBridge, GeneratesProblemLevelMatrixRecipes)
{
    using namespace roc::host_validation;
    using namespace hipblaslt::host_validation;

    MatrixStorageInitialization exact;
    exact.role                          = MatrixRole::B;
    exact.initialization                = hipblaslt_initialization::integer_exact;
    exact.type                          = HIP_R_32F;
    exact.rows                          = 2;
    exact.columns                       = 3;
    exact.leadingDimension              = 4;
    exact.batchStride                   = 12;
    exact.batchCount                    = 2;
    std::vector<std::byte> exactStorage = generateMatrixStorage(exact);
    Tensor exactView(ScalarType::Float32, Layout(Shape{2, 3, 2}, {1, 4, 12}), exactStorage);
    for(size_t batch = 0; batch < 2; ++batch)
        for(size_t column = 0; column < 3; ++column)
            for(size_t row = 0; row < 2; ++row)
            {
                const float  value        = exactView.loadAs<float>({row, column, batch});
                const size_t logicalIndex = row + 2 * (column + 3 * batch);
                const uint64_t sequenceSeed = initialization::seedForSequence(
                    defaultInitializationSeed, initialization::integerExactMatrixBSequence);
                const int magnitude = indexedUniformInteger(
                    sequenceSeed,
                    generation_random_domain_version_1::realComponent,
                    logicalIndex,
                    0,
                    2);
                const int expected = ((row ^ column) & 1U) == 0 ? -magnitude : magnitude;
                EXPECT_EQ(value, expected);
                EXPECT_EQ(value, std::trunc(value));
                EXPECT_LE(std::abs(value), 2);
                if(value != 0)
                    EXPECT_EQ(value > 0, ((row ^ column) & 1U) != 0);
            }
    Tensor exactAllocation(ScalarType::Float32, Layout(Shape{4, 3, 2}, {1, 4, 12}), exactStorage);
    for(size_t batch = 0; batch < 2; ++batch)
        for(size_t column = 0; column < 3; ++column)
            for(size_t row = 2; row < 4; ++row)
                EXPECT_EQ(exactAllocation.loadAs<float>({row, column, batch}), 0);

    MatrixStorageInitialization probe;
    probe.role                          = MatrixRole::B;
    probe.initialization                = hipblaslt_initialization::fp16_accumulator_probe;
    probe.type                          = HIP_R_16F;
    probe.rows                          = 4;
    probe.columns                       = 2;
    probe.leadingDimension              = 4;
    std::vector<std::byte> probeStorage = generateMatrixStorage(probe);
    Tensor probeView(ScalarType::Float16, Layout(Shape{4, 2, 1}, {1, 4, 0}), probeStorage);
    for(size_t column = 0; column < 2; ++column)
        for(size_t row = 0; row < 4; ++row)
            EXPECT_EQ(probeView.loadAs<float>({row, column, 0}), row % 2 == 0 ? 2 : -2);

    MatrixStorageInitialization oneSpecial;
    oneSpecial.role                       = MatrixRole::A;
    oneSpecial.initialization             = hipblaslt_initialization::norm_dist_one_special;
    oneSpecial.specialValueType           = 0;
    oneSpecial.type                       = HIP_R_32F;
    oneSpecial.rows                       = 4;
    oneSpecial.columns                    = 3;
    oneSpecial.leadingDimension           = 4;
    oneSpecial.batchStride                = 12;
    oneSpecial.batchCount                 = 2;
    std::vector<std::byte> specialStorage = generateMatrixStorage(oneSpecial);
    Tensor specialView(ScalarType::Float32, Layout(Shape{4, 3, 2}, {1, 4, 12}), specialStorage);
    size_t infinityCount = 0;
    for(size_t batch = 0; batch < 2; ++batch)
        for(size_t column = 0; column < 3; ++column)
            for(size_t row = 0; row < 4; ++row)
                infinityCount += std::isinf(specialView.loadAs<float>({row, column, batch}));
    EXPECT_EQ(infinityCount, 1);
}

TEST(HostValidationDataInitializationBridge, HostSideDeviceFillCopiesComponentStorage)
{
    using namespace hipblaslt::host_validation;

    MatrixStorageInitialization initialization;
    initialization.role                   = MatrixRole::B;
    initialization.initialization         = hipblaslt_initialization::integer_exact;
    initialization.type                   = HIP_R_32F;
    initialization.rows                   = 2;
    initialization.columns                = 3;
    initialization.leadingDimension       = 4;
    initialization.batchStride            = 12;
    initialization.batchCount             = 2;
    const std::vector<std::byte> expected = generateMatrixStorage(initialization);

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
    EXPECT_EQ(hipMemcpy(observed.data(), device, observed.size(), hipMemcpyDeviceToHost),
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

    const float expected = static_cast<float>(hipblaslt_f8(a[0] * scaleA[0] * alphaVector[0]));
    EXPECT_FLOAT_EQ(d[0], expected);
}

TEST(HostValidationCblasBridge, AppliesSameWidthCrossFormatComputeQuantization)
{
    const std::array<hipblaslt_f8, 1> a{hipblaslt_f8(1.125f)};
    const std::array<float, 1>        b{1.0f};
    std::array<float, 1>              d{};

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
                                    HIP_R_8F_E4M3,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_32F,
                                    HIP_R_8F_E5M2,
                                    HIP_R_32F);

    EXPECT_FLOAT_EQ(d[0], static_cast<float>(hipblaslt_bf8(static_cast<float>(a[0]))));
}

TEST(HostValidationCblasBridge, AppliesOutputScaleBeforeNarrowConversion)
{
    const std::array<float, 1>   a{0.3333f};
    const std::array<float, 1>   b{3.0f};
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
    const std::array<float, 1>       a{1.3f};
    const std::array<float, 1>       b{1.0f};
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

TEST(HostValidationCblasBridge, ZeroScalarsSuppressNonFiniteInputs)
{
    const float          nan      = std::numeric_limits<float>::quiet_NaN();
    const float          infinity = std::numeric_limits<float>::infinity();
    const float          finiteC  = 3.0f;
    std::array<float, 1> output{};

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    1,
                                    1,
                                    1,
                                    0.0f,
                                    &nan,
                                    1,
                                    &infinity,
                                    1,
                                    2.0f,
                                    &finiteC,
                                    1,
                                    output.data(),
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
    EXPECT_EQ(output[0], 6.0f);

    const float a = 2.0f;
    const float b = 4.0f;
    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
                                    HIPBLAS_OP_N,
                                    1,
                                    1,
                                    1,
                                    1.0f,
                                    &a,
                                    1,
                                    &b,
                                    1,
                                    0.0f,
                                    &infinity,
                                    1,
                                    output.data(),
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
    EXPECT_EQ(output[0], 8.0f);
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
    const std::array<float, 2>          b{3.0f, 4.0f};
    std::array<float, 1>                d{};

    hipblaslt_reference_gemm<float>(HIPBLAS_OP_N,
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
