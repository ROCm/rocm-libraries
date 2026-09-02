// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipBuffer.hpp>
#include <hipblaslt/host_numerics/GroupedGemmDataInitialization.hpp>
#include <hipblaslt/host_numerics/HipblasltDataInitialization.hpp>
#include <hipblaslt/host_numerics/HipblasltReferenceGemm.hpp>
#include <hipblaslt/host_numerics/HostComparison.hpp>
#include <hipblaslt/host_numerics/MatrixTransformReference.hpp>
#include <hipblaslt/host_numerics/hipblaslt_init.hpp>
#include <hipblaslt/host_numerics/near.hpp>

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
    template <typename Compute>
    computeTypeInterface computeInterfaceValue(Compute value)
    {
        computeTypeInterface result{};
        if constexpr(std::is_same_v<Compute, hipblasLtHalf>)
            result.f16 = value;
        else if constexpr(std::is_same_v<Compute, float>)
            result.f32 = value;
        else if constexpr(std::is_same_v<Compute, double>)
            result.f64 = value;
        else if constexpr(std::is_same_v<Compute, int32_t>)
            result.i32 = value;
        else if constexpr(std::is_same_v<Compute, std::complex<float>>)
            result.cf = value;
        else if constexpr(std::is_same_v<Compute, std::complex<double>>)
            result.cd = value;
        return result;
    }

    template <typename Compute>
    void testHipblasltReferenceGemm(hipblasOperation_t transA,
                                    hipblasOperation_t transB,
                                    int64_t            m,
                                    int64_t            n,
                                    int64_t            k,
                                    Compute            alpha,
                                    const void*        a,
                                    int64_t            lda,
                                    const void*        b,
                                    int64_t            ldb,
                                    Compute            beta,
                                    const void*        c,
                                    int64_t            ldc,
                                    void*              d,
                                    int64_t            ldd,
                                    const void*        alphaVector,
                                    const void*        scaleA,
                                    const void*        scaleB,
                                    Compute            scaleD,
                                    bool               scaleAIsVector,
                                    bool               scaleBIsVector,
                                    hipDataType        typeA,
                                    hipDataType        typeB,
                                    hipDataType        typeC,
                                    hipDataType        typeD,
                                    hipDataType        computeInputTypeA,
                                    hipDataType        computeInputTypeB,
                                    bool               scaleAIsMx = false,
                                    bool               scaleBIsMx = false,
                                    Compute            scaleC     = Compute{1})
    {
        constexpr hipDataType coefficientType = [] {
            if constexpr(std::is_same_v<Compute, hipblasLtHalf>)
                return HIP_R_16F;
            else if constexpr(std::is_same_v<Compute, float>)
                return HIP_R_32F;
            else if constexpr(std::is_same_v<Compute, double>)
                return HIP_R_64F;
            else if constexpr(std::is_same_v<Compute, int32_t>)
                return HIP_R_32I;
            else if constexpr(std::is_same_v<Compute, std::complex<float>>)
                return HIP_C_32F;
            else
                return HIP_C_64F;
        }();
        const auto scaleCValue = computeInterfaceValue(scaleC);
        const auto scaleDValue = computeInterfaceValue(scaleD);
        const auto matrixLayout = [](size_t               rows,
                                     size_t               columns,
                                     int64_t              leadingDimension,
                                     hipblasOperation_t   operation) {
            return roc::host_numerics::Layout(
                roc::host_numerics::Shape{rows, columns},
                {operation == HIPBLAS_OP_N ? 1 : leadingDimension,
                 operation == HIPBLAS_OP_N ? leadingDimension : 1});
        };
        const auto layoutA = matrixLayout(m, k, lda, transA);
        const auto layoutB = matrixLayout(k, n, ldb, transB);
        const auto layoutC = matrixLayout(m, n, ldc, HIPBLAS_OP_N);
        const auto layoutD = matrixLayout(m, n, ldd, HIPBLAS_OP_N);
        const auto matrix = [](hipDataType type, const roc::host_numerics::Layout& layout) {
            return hipblaslt::client::MatmulMatrix{
                type,
                hipblaslt::host_numerics::scalarType(type),
                roc::host_numerics::Layout(
                    roc::host_numerics::Shape{
                        layout.shape()[0], layout.shape()[1], 1},
                    {layout.stride(0), layout.stride(1), 0}),
                layout.shape()[1] * static_cast<size_t>(layout.stride(1)),
            };
        };
        const hipblaslt::client::MatmulProblem problem{
            .m          = m,
            .n          = n,
            .k          = k,
            .operationA = transA,
            .operationB = transB,
            .batchMode  = HIPBLASLT_BATCH_MODE_STRIDED,
            .batchCount = 1,
            .a          = matrix(typeA, layoutA),
            .b          = matrix(typeB, layoutB),
            .c          = matrix(typeC, layoutC),
            .d          = matrix(typeD, layoutD),
            .auxiliary  = std::nullopt,
            .cEqualsD   = false,
        };
        const hipblaslt::client::MatmulDataTypes dataTypes{
            .computeScalar = coefficientType,
            .computeInputA = computeInputTypeA,
            .computeInputB = computeInputTypeB,
            .coefficient   = coefficientType,
            .bias          = HIP_R_32F,
            .biasStorage   = HIP_R_32F,
            .auxiliary     = HIP_R_32F,
        };
        hipblaslt::client::PreparedMatmulProblem preparation;
        preparation.alpha = computeInterfaceValue(alpha);
        preparation.beta  = computeInterfaceValue(beta);

        auto output = hipblaslt::host_numerics::copyTensorFromEncodedStorage(
            d, hipblaslt::host_numerics::scalarType(typeD), layoutD);
        hipblaslt::host_numerics::MatmulReferenceInputs inputs(
            hipblaslt::host_numerics::copyTensorFromEncodedStorage(
                a, hipblaslt::host_numerics::scalarType(typeA), layoutA),
            hipblaslt::host_numerics::copyTensorFromEncodedStorage(
                b, hipblaslt::host_numerics::scalarType(typeB), layoutB),
            hipblaslt::host_numerics::copyTensorFromEncodedStorage(
                c, hipblaslt::host_numerics::scalarType(typeC), layoutC),
            output);
        const auto vector = [&](const void* values, size_t elements) {
            return hipblaslt::host_numerics::copyTensorFromEncodedStorage(
                values,
                hipblaslt::host_numerics::scalarType(coefficientType),
                roc::host_numerics::Layout::contiguousLastDimensionFastest(
                    roc::host_numerics::Shape{elements}));
        };
        if(alphaVector)
            inputs.alphaVector = vector(alphaVector, static_cast<size_t>(m));
        if(scaleA)
            inputs.scaleA = vector(scaleA, scaleAIsVector ? static_cast<size_t>(m) : 1);
        if(scaleB)
            inputs.scaleB = vector(scaleB, scaleBIsVector ? static_cast<size_t>(n) : 1);
        inputs.scaleC
            = hipblaslt::host_numerics::realOnlyScalarValue(&scaleCValue, coefficientType);
        inputs.scaleD = hipblaslt::host_numerics::scalarValue(&scaleDValue, coefficientType);

        const auto scaleMode = [](const void* scale, bool vectorScale, bool mxScale) {
            if(mxScale)
                return hipblaslt_scaling_format::Block_32_UE8M0;
            if(vectorScale)
                return hipblaslt_scaling_format::Vector;
            return scale ? hipblaslt_scaling_format::Scalar : hipblaslt_scaling_format::none;
        };
        (void)hipblaslt::host_numerics::referenceMatmulGemm(problem,
                                                            dataTypes,
                                                            preparation,
                                                            std::move(inputs),
                                                            scaleMode(scaleA,
                                                                      scaleAIsVector,
                                                                      scaleAIsMx),
                                                            scaleMode(scaleB,
                                                                      scaleBIsVector,
                                                                      scaleBIsMx));
        hipblaslt::host_numerics::copyTensorEncodedBackingStorageToBuffer(
            d, roc::host_numerics::storageBytesForLayout(output.type(), output.layout()), output);
    }
} // namespace

TEST(HostNumericsTensorStorage, HipHostBufferTensorRetainsAndMutatesThePinnedAllocation)
{
    using namespace roc::host_numerics;

    Tensor tensor = [] {
        HipHostBuffer buffer(HIP_R_32F, 2);
        Tensor        wrapped
            = buffer.tensor(ScalarType::Float32, Layout::contiguousLastDimensionFastest(Shape{2}));
        wrapped.storeFrom({1}, 7.0f);
        EXPECT_EQ(buffer.as<float>()[1], 7.0f);
        return wrapped;
    }();

    EXPECT_EQ(tensor.loadAs<float>({1}), 7.0f);
    Tensor alias = tensor;
    alias.storeFrom({0}, 3.0f);
    EXPECT_EQ(tensor.loadAs<float>({0}), 3.0f);
}

TEST(HostNumericsTensorStorage, EncodedStorageBridgeCopiesOwnershipAndBackingPadding)
{
    using namespace hipblaslt::host_numerics;
    using namespace roc::host_numerics;

    std::array<float, 3> storage{1.0f, -7.0f, 2.0f};
    Tensor               tensor
        = copyTensorFromEncodedStorage(storage.data(), storage.size(), Layout(Shape{2}, {2}));

    storage[0] = 9.0f;
    tensor.storeFrom({1}, 4.0f);
    EXPECT_EQ(tensor.loadAs<float>({0}), 1.0f);
    EXPECT_EQ(storage, (std::array<float, 3>{9.0f, -7.0f, 2.0f}));

    copyTensorEncodedBackingStorageToBuffer(storage.data(), storage.size(), tensor);
    EXPECT_EQ(storage, (std::array<float, 3>{1.0f, -7.0f, 4.0f}));
}

TEST(HostNumericsTypeBridge, UsesScalarTypeAsTheExternalTypeConversionHub)
{
    using hipblaslt::host_numerics::scalarType;
    using roc::host_numerics::ScalarType;

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
        EXPECT_EQ(scalarType(hip), scalar);
}

#if HIPBLASLT_ENABLE_MXDATAGENERATOR
TEST(HostNumericsMxGenerationBridge, MapsScaleLayoutsAndGeneratesTypedData)
{
    using namespace hipblaslt::host_numerics;
    using namespace roc::host_numerics;
    using amd_gpu_layout::MxScaleStorageLayout;

    EXPECT_EQ(mxScaleStorageLayoutForArchName("gfx950"), MxScaleStorageLayout::Gfx950);
    EXPECT_EQ(mxScaleStorageLayoutForArchName("gfx1250"), MxScaleStorageLayout::Gfx1250);
    EXPECT_EQ(mxScaleStorageLayoutForArchName("gfx942"), MxScaleStorageLayout::Natural);
    EXPECT_EQ(
        mxScaleStorageLayoutForFormat(hipblaslt_scaling_format::Block_32_UE8M0_32_8_EXT, "gfx950"),
        MxScaleStorageLayout::Gfx950);

    const MxTensor generated = generateMxData((hipDataType)HIP_R_4F_E2M1,
                                              HIP_R_8F_UE8M0,
                                              Shape{8, 4},
                                              8,
                                              0,
                                              4,
                                              hipblaslt_initialization::hpl,
                                              17);
    EXPECT_EQ(generated.data.type(), ScalarType::Float4E2M1);
    EXPECT_EQ(generated.scales.type(), ScalarType::E8M0);
    EXPECT_EQ(generated.reference.shape(), (Shape{8, 4}));
}
#endif

TEST(HostNumericsDataInitializationBridge, GeneratesComplexTrigonometricValues)
{
    std::array<std::complex<float>, 4> values{};
    hipblaslt::host_numerics::initialize(std::span<std::complex<float>>(values),
                                         hipblaslt_initialization::trig_float,
                                         hipblaslt::host_numerics::TrigonometricComponent::Sine);

    for(size_t index = 0; index < values.size(); ++index)
    {
        EXPECT_FLOAT_EQ(values[index].real(), std::sin(static_cast<float>(index)));
        EXPECT_FLOAT_EQ(values[index].imag(), std::cos(static_cast<float>(index)));
    }
}

TEST(HostNumericsDataInitializationBridge, ComplexRandomUsesTypedCartesianDomains)
{
    using namespace roc::host_numerics;

    std::array<std::complex<float>, 8> first{};
    std::array<std::complex<float>, 8> second{};
    hipblaslt::host_numerics::initialize(std::span<std::complex<float>>(first),
                                         hipblaslt_initialization::rand_int);
    hipblaslt::host_numerics::initialize(std::span<std::complex<float>>(second),
                                         hipblaslt_initialization::rand_int);
    EXPECT_EQ(first, second);

    constexpr std::array<float, 8> expectedReal{2, 6, 3, 9, 10, 8, 1, 9};
    constexpr std::array<float, 8> expectedImaginary{8, 5, 9, 5, 1, 5, 6, 7};
    for(size_t index = 0; index < first.size(); ++index)
    {
        EXPECT_EQ(first[index].real(), expectedReal[index]);
        EXPECT_EQ(first[index].imag(), expectedImaginary[index]);
    }
}

TEST(HostNumericsDataInitializationBridge, GroupedGemmUsesStableRoleSequencesAndDefaultSeed)
{
    using namespace hipblaslt::host_numerics;
    using namespace roc::host_numerics;

    std::vector<float> a(5);
    std::vector<float> b(7);
    std::vector<float> c(4);
    std::vector<float> bias(3);

    hipblaslt::host_numerics::initializeGroupedGemm(a,
                                                    static_cast<int64_t>(a.size()),
                                                    b,
                                                    static_cast<int64_t>(b.size()),
                                                    c,
                                                    static_cast<int64_t>(c.size()),
                                                    bias,
                                                    static_cast<int64_t>(bias.size()),
                                                    hipblaslt_initialization::rand_int);

    const auto expected = [](size_t size, initialization::OperandSequence sequence) {
        std::vector<float>          values(size);
        GenerationRecipe::Component component
            = GenerationRecipe::uniformInteger({.lower = 1, .upper = 10});
        if(sequence == initialization::OperandSequence::MatrixB)
            component
                = component.withAlternatingSign({.dimensions = {0}, .negativeWhenOdd = false});
        initializeTensor(values.data(),
                         Layout::contiguousLastDimensionFastest(Shape{size}),
                         GenerationRecipe::realOnly(std::move(component),
                                                    {.seed = initialization::seedForSequence(
                                                         defaultInitializationSeed, sequence)}));
        return values;
    };

    EXPECT_EQ(a, expected(a.size(), initialization::OperandSequence::MatrixA));
    EXPECT_EQ(b, expected(b.size(), initialization::OperandSequence::MatrixB));
    EXPECT_EQ(c, expected(c.size(), initialization::OperandSequence::MatrixC));
    EXPECT_EQ(bias, expected(bias.size(), initialization::OperandSequence::Bias));
}

TEST(HostNumericsDataInitializationBridge, InitializationSeedDependsOnSeedAndSequence)
{
    using hipblaslt::host_numerics::initialization::seedForSequence;

    constexpr uint64_t seed     = 0x123456789abcdef0ULL;
    constexpr uint64_t sequence = 0x1020304050607080ULL;
    EXPECT_EQ(seedForSequence(seed, sequence), seed + sequence);
}

TEST(HostNumericsDataInitializationBridge, GroupedGemmPropagatesCallerSeed)
{
    constexpr size_t   elements = 32;
    constexpr uint64_t seed     = 0x123456789abcdef0ULL;

    const auto generated = [](uint64_t callerSeed) {
        std::array<std::vector<float>, 4> operands;
        for(auto& operand : operands)
            operand.resize(elements);
        hipblaslt::host_numerics::initializeGroupedGemm(operands[0],
                                                        static_cast<int64_t>(operands[0].size()),
                                                        operands[1],
                                                        static_cast<int64_t>(operands[1].size()),
                                                        operands[2],
                                                        static_cast<int64_t>(operands[2].size()),
                                                        operands[3],
                                                        static_cast<int64_t>(operands[3].size()),
                                                        hipblaslt_initialization::rand_int,
                                                        callerSeed);
        return operands;
    };

    const auto first       = generated(seed);
    const auto replay      = generated(seed);
    const auto changedSeed = generated(seed + 10);
    EXPECT_EQ(first, replay);
    EXPECT_NE(first, changedSeed);

    for(size_t operand = 0; operand < first.size(); ++operand)
        for(size_t index = 0; index < first[operand].size(); ++index)
        {
            const float value = first[operand][index];
            EXPECT_EQ(value, std::trunc(value));
            EXPECT_GE(std::abs(value), 1);
            EXPECT_LE(std::abs(value), 10);
            if(operand == 1)
                EXPECT_EQ(value < 0, index % 2 == 0);
            else
                EXPECT_GT(value, 0);
        }
}

TEST(HostNumericsDataInitializationBridge, GroupedGemmDefinesHplAndSpecialRecipes)
{
    std::vector<float> a(4);
    std::vector<float> b(4);
    std::vector<float> c(4);
    std::vector<float> bias(4);

    const auto initialize = [&](hipblaslt_initialization initialization) {
        hipblaslt::host_numerics::initializeGroupedGemm(a,
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
        EXPECT_EQ(value, hipblaslt::host_numerics::specialInitializationAValue);
    for(const float value : b)
        EXPECT_EQ(value, hipblaslt::host_numerics::specialInitializationBValue);
    expectHplRange(c);
    expectHplRange(bias);
}

TEST(HostNumericsDataInitializationBridge,
     GroupedGemmHandlesZeroAndRejectsUnsupportedInitialization)
{
    std::vector<float> a(1, 1.0f);
    std::vector<float> b(1, 1.0f);
    std::vector<float> c(1, 1.0f);
    std::vector<float> bias(1, 1.0f);

    hipblaslt::host_numerics::initializeGroupedGemm(a,
                                                    static_cast<int64_t>(a.size()),
                                                    b,
                                                    static_cast<int64_t>(b.size()),
                                                    c,
                                                    static_cast<int64_t>(c.size()),
                                                    bias,
                                                    static_cast<int64_t>(bias.size()),
                                                    hipblaslt_initialization::zero);
    EXPECT_EQ(a[0], 0.0f);
    EXPECT_EQ(b[0], 0.0f);
    EXPECT_EQ(c[0], 0.0f);
    EXPECT_EQ(bias[0], 0.0f);

    EXPECT_THROW(
        hipblaslt::host_numerics::initializeGroupedGemm(a,
                                                        static_cast<int64_t>(a.size()),
                                                        b,
                                                        static_cast<int64_t>(b.size()),
                                                        c,
                                                        static_cast<int64_t>(c.size()),
                                                        bias,
                                                        static_cast<int64_t>(bias.size()),
                                                        hipblaslt_initialization::norm_dist),
        std::invalid_argument);
}

TEST(HostNumericsMatrixTransformBridge, MapsLayoutsAndTransposes)
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

    hipblaslt::host_numerics::MatrixTransformReferenceArguments arguments;
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

    const auto comparison = hipblaslt::host_numerics::referenceMatrixTransform(arguments);
    EXPECT_EQ(comparison.compared, rows * columns * batches);
    EXPECT_TRUE(comparison.passed());
}

TEST(HostNumericsComparisonBridge, FindsAllcloseToleranceAcrossBatches)
{
    const std::array<float, 4>                      expected{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4>                      observed{1.0f, 2.00009f, 3.0f, 4.0f};
    hipblaslt::host_numerics::HostComparisonRequest request;
    request.rows                  = 2;
    request.columns               = 1;
    request.leadingDimension      = 2;
    request.batchStride           = 2;
    request.batchCount            = 2;
    request.expected              = expected.data();
    request.observed              = observed.data();
    request.type                  = HIP_R_32F;
    request.findAllCloseTolerance = true;

    const auto report = hipblaslt::host_numerics::compareHost(request);
    ASSERT_TRUE(report.allCloseTolerance);
    EXPECT_EQ(report.allCloseTolerance->absolute, 1e-6);
    EXPECT_EQ(report.allCloseTolerance->relative, 1e-4);
}

TEST(HostNumericsComparisonBridge, UsesMagnitudeForComplexAllcloseToleranceSearch)
{
    const std::complex<float> expected{0.0f, 0.0f};
    const std::complex<float> observed{0.09f, 0.09f};

    hipblaslt::host_numerics::HostComparisonRequest request;
    request.rows                  = 1;
    request.columns               = 1;
    request.leadingDimension      = 1;
    request.batchStride           = 1;
    request.batchCount            = 1;
    request.expected              = &expected;
    request.observed              = &observed;
    request.type                  = HIP_C_32F;
    request.findAllCloseTolerance = true;

    EXPECT_FALSE(hipblaslt::host_numerics::compareHost(request).allCloseTolerance);
}

TEST(HostNumericsComparisonBridge, ComputesRelativeFrobeniusEvidence)
{
    std::array<double, 2>                           expected{3.0, 4.0};
    std::array<double, 2>                           observed{0.0, 4.0};
    hipblaslt::host_numerics::HostComparisonRequest request;
    request.rows                          = 2;
    request.columns                       = 1;
    request.leadingDimension              = 2;
    request.batchStride                   = 2;
    request.batchCount                    = 1;
    request.expected                      = expected.data();
    request.observed                      = observed.data();
    request.type                          = HIP_R_64F;
    request.computeRelativeFrobeniusError = true;
    EXPECT_DOUBLE_EQ(hipblaslt::host_numerics::compareHost(request).relativeFrobeniusError, 0.6);
}

TEST(HostNumericsComparisonBridge, UnitNearAndSpecialValuePolicies)
{
    const float          oneUlp = std::nextafter(1.0f, 2.0f);
    std::array<float, 2> expected{1.0f, std::numeric_limits<float>::infinity()};
    std::array<float, 2> observed{oneUlp, std::numeric_limits<float>::infinity()};

    hipblaslt::host_numerics::HostComparisonRequest request;
    request.rows             = 2;
    request.columns          = 1;
    request.leadingDimension = 2;
    request.batchStride      = 2;
    request.batchCount       = 1;
    request.expected         = expected.data();
    request.observed         = observed.data();
    request.type             = HIP_R_32F;

    request.pointwise = hipblaslt::host_numerics::HostPointwiseComparison::Unit;
    EXPECT_TRUE(hipblaslt::host_numerics::compareHost(request).comparison.passed());

    request.pointwise         = hipblaslt::host_numerics::HostPointwiseComparison::Near;
    request.absoluteTolerance = 1e-6;
    EXPECT_TRUE(hipblaslt::host_numerics::compareHost(request).comparison.passed());

    expected[0]       = 0.0f;
    observed[0]       = 2.0f * std::numeric_limits<float>::epsilon();
    request.pointwise = hipblaslt::host_numerics::HostPointwiseComparison::SymmetricRelative;
    request.symmetricRelativeTolerance = 3.0f * std::numeric_limits<float>::epsilon();
    EXPECT_TRUE(hipblaslt::host_numerics::compareHost(request).comparison.passed());
    request.symmetricRelativeTolerance = std::numeric_limits<float>::epsilon();
    EXPECT_FALSE(hipblaslt::host_numerics::compareHost(request).comparison.passed());

    request.pointwise = hipblaslt::host_numerics::HostPointwiseComparison::Disabled;
    request.requireSpecialValueConsistency = true;
    EXPECT_EQ(hipblaslt::host_numerics::compareHost(request).comparison.nonFiniteMismatches, 0);
}

TEST(HostNumericsComparisonBridge, RunsTheCombinedHostComparisonProgram)
{
    const float                oneUlp = std::nextafter(1.0f, 2.0f);
    const std::array<float, 4> expected{1.0f, 2.0f, 3.0f, 4.0f};
    const std::array<float, 4> observed{oneUlp, 2.0f, 3.0f, 4.0f};

    hipblaslt::host_numerics::HostComparisonRequest request;
    request.rows             = 2;
    request.columns          = 1;
    request.leadingDimension = 2;
    request.batchStride      = 2;
    request.batchCount       = 2;
    request.expected         = expected.data();
    request.observed         = observed.data();
    request.type             = HIP_R_32F;
    request.pointwise        = hipblaslt::host_numerics::HostPointwiseComparison::Unit;
    request.requireSpecialValueConsistency = true;
    request.computeRelativeFrobeniusError  = true;
    request.findAllCloseTolerance          = true;
    request.computeUnitsInLastPlace        = true;

    const auto report = hipblaslt::host_numerics::compareHost(request);
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

TEST(HostNumericsComparisonBridge, KeepsReportedUlpNonFinitePolicySeparate)
{
    const float nan = std::numeric_limits<float>::quiet_NaN();

    hipblaslt::host_numerics::HostComparisonRequest request;
    request.rows             = 1;
    request.columns          = 1;
    request.leadingDimension = 1;
    request.batchStride      = 1;
    request.batchCount       = 1;
    request.expected         = &nan;
    request.observed         = &nan;
    request.type             = HIP_R_32F;
    request.pointwise        = hipblaslt::host_numerics::HostPointwiseComparison::Unit;
    request.requireSpecialValueConsistency = true;
    request.computeUnitsInLastPlace        = true;

    const auto report = hipblaslt::host_numerics::compareHost(request);
    EXPECT_TRUE(report.comparison.passed());
    EXPECT_EQ(report.comparison.nonFiniteMismatches, 0);
    EXPECT_TRUE(std::isinf(report.unitsInLastPlaceComparison.maximumUlp));
}

TEST(HostNumericsComparisonBridge, EmptyPointwiseRequestsStillValidateTheProductType)
{
    using namespace hipblaslt::host_numerics;

    HostComparisonRequest request;
    request.type      = HIPBLASLT_DATATYPE_INVALID;
    request.pointwise = HostPointwiseComparison::Unit;
    EXPECT_THROW(compareHost(request), std::invalid_argument);

    request.pointwise                     = HostPointwiseComparison::Disabled;
    request.computeRelativeFrobeniusError = true;
    EXPECT_NO_THROW(compareHost(request));
}

TEST(HostNumericsTolerancePolicy, Gfx11ScalesComputeTypeEpsilon)
{
    EXPECT_DOUBLE_EQ(sum_error_tolerance_for_compute_type(HIP_R_32F),
                     std::numeric_limits<float>::epsilon());
    EXPECT_DOUBLE_EQ(sum_error_tolerance_for_compute_type(HIP_R_16F),
                     std::numeric_limits<hipblasLtHalf>::epsilon());
    EXPECT_DOUBLE_EQ(gfx11_low_precision_accumulation_tolerance_coefficient(HIP_R_32F, 8),
                     64.0 * std::numeric_limits<float>::epsilon());
    EXPECT_DOUBLE_EQ(bfloat16_output_rounding_tolerance_coefficient(), 0x1p-8);
}

TEST(HostNumericsDataInitializationBridge, CounterBasedGenerationIsRepeatable)
{
    std::array<float, 16> first{};
    std::array<float, 16> second{};
    hipblaslt::host_numerics::initialize(std::span<float>(first),
                                         hipblaslt_initialization::norm_dist);
    hipblaslt::host_numerics::initialize(std::span<float>(second),
                                         hipblaslt_initialization::norm_dist);
    EXPECT_EQ(first, second);
}

TEST(HostNumericsDataInitializationBridge, DeviceNormalGenerationIsRepeatable)
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

TEST(HostNumericsDataInitializationBridge, RandomHelpersUseComponentRecipes)
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
    hipblaslt_init_small(static_cast<void*>(values.data()), 2, 2, 3, HIP_R_32F);
    for(const size_t index : {size_t{0}, size_t{1}, size_t{3}, size_t{4}})
    {
        EXPECT_GE(values[index], 0.1f);
        EXPECT_LE(values[index], 1.0f);
        EXPECT_FLOAT_EQ(values[index] * 10, std::round(values[index] * 10));
    }
    EXPECT_EQ(values[2], -99);

    hipblaslt_init_nan(values.data(), values.size());
    for(const float value : values)
        EXPECT_TRUE(std::isnan(value));
}

TEST(HostNumericsDataInitializationBridge, RuntimeDispatchSupportsEveryFp8Encoding)
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

TEST(HostNumericsDataInitializationBridge, GeneratesProblemLevelMatrixRecipes)
{
    using namespace roc::host_numerics;
    using namespace hipblaslt::host_numerics;

    MatrixInitialization exact;
    exact.role             = MatrixRole::B;
    exact.initialization   = hipblaslt_initialization::integer_exact;
    exact.type             = HIP_R_32F;
    exact.rows             = 2;
    exact.columns          = 3;
    exact.leadingDimension = 4;
    exact.batchStride      = 12;
    exact.batchCount       = 2;
    Tensor exactMatrix     = generateMatrix(exact);
    Tensor exactReplay     = generateMatrix(exact);
    EXPECT_EQ(exactMatrix.layout(), (Layout(Shape{2, 3, 2}, {1, 4, 12})));
    EXPECT_EQ(exactMatrix.rawEncodedBackingStorage().size(), 24 * sizeof(float));
    for(size_t batch = 0; batch < 2; ++batch)
        for(size_t column = 0; column < 3; ++column)
            for(size_t row = 0; row < 2; ++row)
            {
                const float value = exactMatrix.loadAs<float>({row, column, batch});
                EXPECT_EQ(value, exactReplay.loadAs<float>({row, column, batch}));
                EXPECT_EQ(value, std::trunc(value));
                EXPECT_LE(std::abs(value), 2);
                if(value != 0)
                    EXPECT_EQ(value > 0, ((row ^ column) & 1U) != 0);
            }
    Tensor exactAllocation = exactMatrix.shareStorageWithLayout(Layout(Shape{4, 3, 2}, {1, 4, 12}));
    for(size_t batch = 0; batch < 2; ++batch)
        for(size_t column = 0; column < 3; ++column)
            for(size_t row = 2; row < 4; ++row)
                EXPECT_EQ(exactAllocation.loadAs<float>({row, column, batch}), 0);

    MatrixInitialization probe;
    probe.role             = MatrixRole::B;
    probe.initialization   = hipblaslt_initialization::fp16_accumulator_probe;
    probe.type             = HIP_R_16F;
    probe.rows             = 4;
    probe.columns          = 2;
    probe.leadingDimension = 4;
    Tensor probeMatrix     = generateMatrix(probe);
    for(size_t column = 0; column < 2; ++column)
        for(size_t row = 0; row < 4; ++row)
            EXPECT_EQ(probeMatrix.loadAs<float>({row, column, 0}), row % 2 == 0 ? 2 : -2);
}

TEST(HostNumericsDataInitializationBridge,
     OneSpecialInitializationSelectsALogicalElementAndPreservesGaps)
{
    using namespace roc::host_numerics;
    using namespace hipblaslt::host_numerics;

    MatrixInitialization configuration;
    configuration.role             = MatrixRole::A;
    configuration.initialization   = hipblaslt_initialization::norm_dist_one_special;
    configuration.type             = HIP_R_32F;
    configuration.rows             = 2;
    configuration.columns          = 3;
    configuration.leadingDimension = 4;
    configuration.batchStride      = 16;
    configuration.batchCount       = 2;

    const Layout expectedLayout(Shape{2, 3, 2}, {1, 4, 16});
    const Tensor actual = generateMatrix(configuration);
    EXPECT_EQ(actual.layout(), expectedLayout);
    EXPECT_EQ(actual.rawEncodedBackingStorage().size(), 28 * sizeof(float));

    Tensor baselineStorage(ScalarType::Float32, Layout::contiguousLastDimensionFastest(Shape{28}));
    Tensor baseline = baselineStorage.shareStorageWithLayout(expectedLayout);
    generate(
        baseline,
        normalRecipe(ScalarType::Float32,
                     ComplexGenerationPolicy::RealOnly,
                     initialization::seedForSequence(oneSpecialInitializationSeed,
                                                     initialization::OperandSequence::MatrixA)));

    constexpr size_t  expectedSpecialLogicalIndex = 6;
    std::vector<bool> logicalStorageElements(28, false);
    size_t            infinityCount = 0;
    for(size_t batch = 0; batch < configuration.batchCount; ++batch)
        for(size_t column = 0; column < configuration.columns; ++column)
            for(size_t row = 0; row < configuration.rows; ++row)
            {
                const size_t logicalIndex
                    = row + configuration.rows * (column + configuration.columns * batch);
                const size_t storageIndex = row + configuration.leadingDimension * column
                                            + configuration.batchStride * batch;
                logicalStorageElements[storageIndex] = true;

                const float value = actual.loadAs<float>({row, column, batch});
                if(logicalIndex == expectedSpecialLogicalIndex)
                {
                    EXPECT_TRUE(std::isinf(value));
                    EXPECT_FALSE(std::signbit(value));
                    ++infinityCount;
                }
                else
                {
                    EXPECT_EQ(value, baseline.loadAs<float>({row, column, batch}));
                }
            }
    EXPECT_EQ(infinityCount, 1);

    const Tensor allocation = actual.shareStorageWithLayout(Layout(Shape{28}, {1}));
    for(size_t storageIndex = 0; storageIndex < logicalStorageElements.size(); ++storageIndex)
        if(!logicalStorageElements[storageIndex])
            EXPECT_EQ(allocation.loadAs<float>({storageIndex}), 0.0f);

    configuration.oneSpecialValue = OneSpecialValue::NegativeInfinity;
    const float negativeInfinity  = generateMatrix(configuration).loadAs<float>({0, 0, 1});
    EXPECT_TRUE(std::isinf(negativeInfinity));
    EXPECT_TRUE(std::signbit(negativeInfinity));

    configuration.oneSpecialValue = OneSpecialValue::NaN;
    EXPECT_TRUE(std::isnan(generateMatrix(configuration).loadAs<float>({0, 0, 1})));

    configuration.oneSpecialValue.reset();
    configuration.rows             = 5;
    configuration.columns          = 1;
    configuration.leadingDimension = 7;
    configuration.batchStride      = 0;
    configuration.batchCount       = 1;
    const Tensor fiveElements      = generateMatrix(configuration);
    for(size_t row = 0; row < configuration.rows; ++row)
        EXPECT_EQ(std::isinf(fiveElements.loadAs<float>({row, 0, 0})), row == 4);
}

TEST(HostNumericsDataInitializationBridge, StochasticMatrixModesUseRoleSpecificSequences)
{
    using namespace hipblaslt::host_numerics;

    constexpr std::array modes{
        hipblaslt_initialization::rand_int,
        hipblaslt_initialization::hpl,
        hipblaslt_initialization::uniform_low_precision,
        hipblaslt_initialization::norm_dist,
        hipblaslt_initialization::norm_dist_one_special,
        hipblaslt_initialization::uniform_01,
        hipblaslt_initialization::integer_exact,
    };

    const auto values = [](const Tensor& tensor) {
        return std::vector<std::byte>(tensor.rawEncodedBackingStorage().begin(),
                                      tensor.rawEncodedBackingStorage().end());
    };

    for(const hipblaslt_initialization mode : modes)
    {
        MatrixInitialization initialization;
        initialization.initialization   = mode;
        initialization.type             = HIP_R_32F;
        initialization.rows             = 8;
        initialization.columns          = 8;
        initialization.leadingDimension = 8;

        initialization.role = MatrixRole::A;
        const auto matrixA  = values(generateMatrix(initialization));
        EXPECT_EQ(matrixA, values(generateMatrix(initialization)));

        initialization.role = MatrixRole::B;
        const auto matrixB  = values(generateMatrix(initialization));
        EXPECT_EQ(matrixB, values(generateMatrix(initialization)));

        initialization.role = MatrixRole::C;
        const auto matrixC  = values(generateMatrix(initialization));
        EXPECT_EQ(matrixC, values(generateMatrix(initialization)));

        EXPECT_NE(matrixA, matrixB) << "initialization=" << static_cast<int>(mode);
        EXPECT_NE(matrixA, matrixC) << "initialization=" << static_cast<int>(mode);
        EXPECT_NE(matrixB, matrixC) << "initialization=" << static_cast<int>(mode);
    }
}

TEST(HostNumericsDataInitializationBridge, PositiveOnlyMatrixPolicyIsExplicit)
{
    using namespace hipblaslt::host_numerics;

    MatrixInitialization initialization;
    initialization.role             = MatrixRole::A;
    initialization.initialization   = hipblaslt_initialization::hpl;
    initialization.type             = HIP_R_32F;
    initialization.rows             = 8;
    initialization.columns          = 8;
    initialization.leadingDimension = 8;
    const Tensor signedMatrix       = generateMatrix(initialization);

    initialization.positiveOnly = true;
    const Tensor positiveMatrix = generateMatrix(initialization);
    for(size_t column = 0; column < initialization.columns; ++column)
        for(size_t row = 0; row < initialization.rows; ++row)
        {
            const float signedValue   = signedMatrix.loadAs<float>({row, column, 0});
            const float positiveValue = positiveMatrix.loadAs<float>({row, column, 0});
            EXPECT_FLOAT_EQ(positiveValue, std::abs(signedValue));
        }
}

TEST(HostNumericsDataInitializationBridge, UnsupportedModesThrowInsteadOfProducingFallbackData)
{
    using namespace hipblaslt::host_numerics;

    MatrixInitialization initialization;
    initialization.role             = MatrixRole::A;
    initialization.initialization   = hipblaslt_initialization::fp16_accumulator_probe;
    initialization.type             = HIP_R_32F;
    initialization.rows             = 1;
    initialization.columns          = 1;
    initialization.leadingDimension = 1;
    EXPECT_THROW(generateMatrix(initialization), std::invalid_argument);

    initialization.initialization
        = static_cast<hipblaslt_initialization>(std::numeric_limits<int>::max());
    EXPECT_THROW(generateMatrix(initialization), std::invalid_argument);

    std::array<float, 1> values{};
    EXPECT_THROW(
        initialize(std::span<float>(values), hipblaslt_initialization::fp16_accumulator_probe),
        std::invalid_argument);

    uint8_t unsupportedRuntimeStorage = 0x5a;
    EXPECT_THROW(hipblaslt_init_small(
                     static_cast<void*>(&unsupportedRuntimeStorage), 1, 1, 1, HIP_R_8F_E4M3),
                 std::invalid_argument);
    EXPECT_EQ(unsupportedRuntimeStorage, 0x5a);
}

TEST(HostNumericsDataInitializationBridge, DeviceInitializationUploadsGeneratedTensor)
{
    using namespace hipblaslt::host_numerics;

    MatrixInitialization initialization;
    initialization.role             = MatrixRole::B;
    initialization.initialization   = hipblaslt_initialization::integer_exact;
    initialization.type             = HIP_R_32F;
    initialization.rows             = 2;
    initialization.columns          = 3;
    initialization.leadingDimension = 4;
    initialization.batchStride      = 12;
    initialization.batchCount       = 2;
    const Tensor expected           = generateMatrix(initialization);

    void* device = nullptr;
    ASSERT_EQ(hipMalloc(&device, expected.rawEncodedBackingStorage().size()), hipSuccess);

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
    std::vector<std::byte> observed(expected.rawEncodedBackingStorage().size());
    EXPECT_EQ(hipMemcpy(observed.data(), device, observed.size(), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_TRUE(
        std::equal(observed.begin(), observed.end(), expected.rawEncodedBackingStorage().begin()));
    EXPECT_EQ(hipFree(device), hipSuccess);
}

TEST(HostNumericsCblasBridge, DistinctHalfCAndFloatD)
{
    const std::array<float, 6>   a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6>   b{7, 9, 11, 8, 10, 12};
    std::array<hipblasLtHalf, 6> c{1, 2, -99, 3, 4, -99};
    const auto                   originalC = c;
    std::array<float, 4>         d{-1, -2, -3, -4};

    testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
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
                                      false,
                                      false,
                                      2.0f);

    for(size_t index = 0; index < c.size(); ++index)
        EXPECT_FLOAT_EQ(static_cast<float>(c[index]), static_cast<float>(originalC[index]));
    EXPECT_FLOAT_EQ(d[0], 2 * 58 + 3 * 2 * static_cast<float>(originalC[0]));
    EXPECT_FLOAT_EQ(d[1], 2 * 139 + 3 * 2 * static_cast<float>(originalC[1]));
    EXPECT_FLOAT_EQ(d[2], 2 * 64 + 3 * 2 * static_cast<float>(originalC[3]));
    EXPECT_FLOAT_EQ(d[3], 2 * 154 + 3 * 2 * static_cast<float>(originalC[4]));
}

TEST(HostNumericsCblasBridge, ConsumesNormalizedMatmulProblemAndTensorBindings)
{
    using namespace roc::host_numerics;

    const Layout matrixLayout = Layout::contiguousLastDimensionFastest(Shape{2, 2});
    const Layout batchedLayout(Shape{2, 2, 1}, {1, 2, 4});
    const hipblaslt::client::MatmulMatrix matrix{
        HIP_R_32F, ScalarType::Float32, batchedLayout, 4};
    const hipblaslt::client::MatmulProblem problem{
        .m          = 2,
        .n          = 2,
        .k          = 2,
        .operationA = HIPBLAS_OP_N,
        .operationB = HIPBLAS_OP_N,
        .batchMode  = HIPBLASLT_BATCH_MODE_STRIDED,
        .batchCount = 1,
        .a          = matrix,
        .b          = matrix,
        .c          = matrix,
        .d          = matrix,
        .auxiliary  = std::nullopt,
        .cEqualsD   = false,
    };
    const hipblaslt::client::MatmulDataTypes dataTypes{
        .computeScalar = HIP_R_32F,
        .computeInputA = HIP_R_32F,
        .computeInputB = HIP_R_32F,
        .coefficient   = HIP_R_32F,
        .bias          = HIP_R_32F,
        .biasStorage   = HIP_R_32F,
        .auxiliary     = HIP_R_32F,
    };
    hipblaslt::client::PreparedMatmulProblem preparation;
    preparation.alpha.f32 = 2.0f;
    preparation.beta.f32  = 3.0f;

    const std::array<float, 4> a{1, 2, 3, 4};
    const std::array<float, 4> b{5, 6, 7, 8};
    const std::array<float, 4> c{1, 1, 1, 1};
    Tensor                     d(ScalarType::Float32, matrixLayout);
    hipblaslt::host_numerics::MatmulReferenceInputs inputs(
        Tensor::copyNativeValues<float>(Shape{2, 2}, a),
        Tensor::copyNativeValues<float>(Shape{2, 2}, b),
        Tensor::copyNativeValues<float>(Shape{2, 2}, c),
        d);

    hipblaslt::host_numerics::referenceMatmulGemm(problem,
                                                  dataTypes,
                                                  preparation,
                                                  std::move(inputs),
                                                  hipblaslt_scaling_format::none,
                                                  hipblaslt_scaling_format::none);
    EXPECT_FLOAT_EQ(d.loadAs<float>({0, 0}), 41.0f);
    EXPECT_FLOAT_EQ(d.loadAs<float>({0, 1}), 47.0f);
    EXPECT_FLOAT_EQ(d.loadAs<float>({1, 0}), 89.0f);
    EXPECT_FLOAT_EQ(d.loadAs<float>({1, 1}), 103.0f);
}

TEST(HostNumericsCblasBridge, AppliesScaleCInsideSharedGemm)
{
    const float          a      = 4.0f;
    const float          b      = 5.0f;
    const float          c      = 7.0f;
    float                d      = -1.0f;
    const float          scaleC = 2.0f;
    const float          scaleD = 1.0f;
    testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
                                      HIPBLAS_OP_N,
                                      1,
                                      1,
                                      1,
                                      2.0f,
                                      &a,
                                      1,
                                      &b,
                                      1,
                                      3.0f,
                                      &c,
                                      1,
                                      &d,
                                      1,
                                      nullptr,
                                      nullptr,
                                      nullptr,
                                      scaleD,
                                      false,
                                      false,
                                      HIP_R_32F,
                                      HIP_R_32F,
                                      HIP_R_32F,
                                      HIP_R_32F,
                                      HIP_R_32F,
                                      HIP_R_32F,
                                      false,
                                      false,
                                      scaleC);

    EXPECT_FLOAT_EQ(d, 82.0f);
}

TEST(HostNumericsCblasBridge, RejectsUnsupportedCoefficientType)
{
    computeTypeInterface value{};

    EXPECT_THROW(hipblaslt::host_numerics::scalarValue(value, HIP_R_8I),
                 std::invalid_argument);
}

TEST(HostNumericsCblasBridge, MixedHalfInputs)
{
    const std::array<hipblasLtHalf, 6> a{1, 4, 2, 5, 3, 6};
    const std::array<hipblasLtHalf, 6> b{7, 9, 11, 8, 10, 12};
    std::array<float, 4>               d{};

    testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
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
                                      HIP_R_16F,
                                      HIP_R_16F);

    EXPECT_FLOAT_EQ(d[0], 58);
    EXPECT_FLOAT_EQ(d[1], 139);
    EXPECT_FLOAT_EQ(d[2], 64);
    EXPECT_FLOAT_EQ(d[3], 154);
}

TEST(HostNumericsCblasBridge, QuantizesCombinedOperandScaleAndAlphaVector)
{
    const std::array<float, 1> a{0.3f};
    const std::array<float, 1> b{1.0f};
    std::array<float, 1>       d{};
    const std::array<float, 1> alphaVector{0.6f};
    const std::array<float, 1> scaleA{0.7f};

    testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
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
                                      HIP_R_8F_E4M3,
                                      HIP_R_32F);

    const float expected = static_cast<float>(hipblaslt_f8(a[0] * scaleA[0] * alphaVector[0]));
    EXPECT_FLOAT_EQ(d[0], expected);
}

TEST(HostNumericsCblasBridge, AppliesSameWidthCrossFormatComputeQuantization)
{
    const std::array<hipblaslt_f8, 1> a{hipblaslt_f8(1.125f)};
    const std::array<float, 1>        b{1.0f};
    std::array<float, 1>              d{};

    testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
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
                                      HIP_R_8F_E5M2,
                                      HIP_R_32F);

    EXPECT_FLOAT_EQ(d[0], static_cast<float>(hipblaslt_bf8(static_cast<float>(a[0]))));
}

TEST(HostNumericsCblasBridge, AppliesOutputScaleBeforeNarrowConversion)
{
    const std::array<float, 1>   a{0.3333f};
    const std::array<float, 1>   b{3.0f};
    std::array<hipblasLtHalf, 1> d{hipblasLtHalf(0.0f)};

    testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
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
                                      HIP_R_32F);

    const hipblasLtHalf expected((a[0] * b[0]) * 0.1f);
    EXPECT_FLOAT_EQ(static_cast<float>(d[0]), static_cast<float>(expected));
}

TEST(HostNumericsCblasBridge, ConvertsFnuzOutputWithComponentCodec)
{
    const std::array<float, 1>       a{1.3f};
    const std::array<float, 1>       b{1.0f};
    std::array<hipblaslt_f8_fnuz, 1> d{hipblaslt_f8_fnuz(0.0f)};

    testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
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
                                      HIP_R_32F);

    EXPECT_EQ(d[0], hipblaslt_f8_fnuz(a[0]));
}

TEST(HostNumericsCblasBridge, SaturatesRoundedInt8Output)
{
    const std::array<float, 1> a{63.75f};
    const std::array<float, 1> b{2.0f};
    std::array<int8_t, 1>      d{};

    testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
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
                                      HIP_R_32F);

    EXPECT_EQ(d[0], 127);
}

TEST(HostNumericsCblasBridge, ZeroScalarsSuppressNonFiniteInputs)
{
    const float          nan      = std::numeric_limits<float>::quiet_NaN();
    const float          infinity = std::numeric_limits<float>::infinity();
    const float          finiteC  = 3.0f;
    std::array<float, 1> output{};

    testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
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
                                      HIP_R_32F);
    EXPECT_EQ(output[0], 6.0f);

    const float a = 2.0f;
    const float b = 4.0f;
    testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
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
                                      HIP_R_32F);
    EXPECT_EQ(output[0], 8.0f);
}

TEST(HostNumericsCblasBridge, IntegerComputeUsesWideReferenceAndSaturatingOutput)
{
    const std::array<int8_t, 2> a{100, 100};
    const std::array<int8_t, 2> b{1, 1};
    std::array<int8_t, 1>       d{};

    testHipblasltReferenceGemm<int32_t>(HIPBLAS_OP_N,
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
                                        HIP_R_32I);

    EXPECT_EQ(d[0], 127);
}

TEST(HostNumericsCblasBridge, TransposedPaddedScaleUsesLogicalRows)
{
    // Stored A is K x M with one padding element after each column.
    const std::array<float, 6> a{1.0f, 2.0f, -99.0f, 3.0f, 4.0f, -99.0f};
    const std::array<float, 2> b{5.0f, 6.0f};
    const std::array<float, 2> scaleA{2.0f, 3.0f};
    std::array<float, 2>       d{};

    testHipblasltReferenceGemm<float>(HIPBLAS_OP_T,
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
                                      HIP_R_32F);

    EXPECT_FLOAT_EQ(d[0], (1.0f * 5.0f + 2.0f * 6.0f) * 2.0f);
    EXPECT_FLOAT_EQ(d[1], (3.0f * 5.0f + 4.0f * 6.0f) * 3.0f);
}

#if defined(HIPBLASLT_USE_FP4)
TEST(HostNumericsCblasBridge, PackedFloat4InputUsesLogicalElementLayout)
{
    const std::array<hipblaslt_f4x2, 1> a{hipblaslt_f4x2(1.0f, 2.0f)};
    const std::array<float, 2>          b{3.0f, 4.0f};
    std::array<float, 1>                d{};

    testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
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
                                      static_cast<hipDataType>(HIP_R_4F_E2M1),
                                      HIP_R_32F);

    EXPECT_FLOAT_EQ(d[0], 11.0f);
}
#endif

TEST(HostNumericsCblasBridge, ComplexConjugateTranspose)
{
    using Complex = std::complex<float>;

    const std::array<Complex, 2> a{Complex(1, 2), Complex(3, -1)};
    const std::array<Complex, 2> b{Complex(2, -1), Complex(-4, 3)};
    std::array<Complex, 1>       d{Complex(0, 0)};

    testHipblasltReferenceGemm<Complex>(HIPBLAS_OP_C,
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
                                        HIP_C_32F);

    const Complex expected = std::conj(a[0]) * b[0] + std::conj(a[1]) * b[1];
    EXPECT_FLOAT_EQ(d[0].real(), expected.real());
    EXPECT_FLOAT_EQ(d[0].imag(), expected.imag());
}

TEST(HostNumericsCblasBridge, UsesResolvedComplexReferenceTypes)
{
    using namespace roc::host_numerics;
    using Complex = std::complex<float>;

    Arguments arguments{};
    arguments.init();
    arguments.a_type     = HIP_C_32F;
    arguments.b_type     = HIP_C_32F;
    arguments.c_type     = HIP_C_32F;
    arguments.d_type     = HIP_C_32F;
    const auto dataTypes = hipblaslt::client::resolveMatmulDataTypes(arguments);

    const Layout matrixLayout = Layout::contiguousLastDimensionFastest(Shape{1, 1});
    const Layout batchedLayout(Shape{1, 1, 1}, {1, 1, 1});
    const hipblaslt::client::MatmulMatrix matrix{
        HIP_C_32F, ScalarType::ComplexFloat32, batchedLayout, 1};
    const hipblaslt::client::MatmulProblem problem{
        .m          = 1,
        .n          = 1,
        .k          = 1,
        .operationA = HIPBLAS_OP_N,
        .operationB = HIPBLAS_OP_N,
        .batchMode  = HIPBLASLT_BATCH_MODE_STRIDED,
        .batchCount = 1,
        .a          = matrix,
        .b          = matrix,
        .c          = matrix,
        .d          = matrix,
        .auxiliary  = std::nullopt,
        .cEqualsD   = false,
    };
    hipblaslt::client::PreparedMatmulProblem preparation;
    preparation.alpha.cf = Complex{2.0f, 3.0f};
    preparation.beta.cf  = Complex{4.0f, 5.0f};

    const Complex                                   a{1.0f, 2.0f};
    const Complex                                   b{3.0f, -1.0f};
    const Complex                                   c{-2.0f, 4.0f};
    Tensor                                          d(ScalarType::ComplexFloat32, matrixLayout);
    hipblaslt::host_numerics::MatmulReferenceInputs inputs(
        Tensor::copyNativeValues<Complex>(Shape{1, 1}, std::span<const Complex>(&a, 1)),
        Tensor::copyNativeValues<Complex>(Shape{1, 1}, std::span<const Complex>(&b, 1)),
        Tensor::copyNativeValues<Complex>(Shape{1, 1}, std::span<const Complex>(&c, 1)),
        d);

    (void)hipblaslt::host_numerics::referenceMatmulGemm(problem,
                                                        dataTypes,
                                                        preparation,
                                                        std::move(inputs),
                                                        hipblaslt_scaling_format::none,
                                                        hipblaslt_scaling_format::none);

    const Complex expected = preparation.alpha.cf * a * b + preparation.beta.cf * c;
    EXPECT_FLOAT_EQ(d.loadAs<Complex>({0, 0}).real(), expected.real());
    EXPECT_FLOAT_EQ(d.loadAs<Complex>({0, 0}).imag(), expected.imag());
}

TEST(HostNumericsCblasBridge, BuildsEmptyBatchLayoutsWithoutAddressingAnElement)
{
    using namespace roc::host_numerics;

    const hipblaslt::client::MatmulMatrix matrix{
        HIP_R_32F, ScalarType::Float32, Layout(Shape{4, 0, 3}, {1, 4, 17}), 51};

    const Layout emptyColumns
        = hipblaslt::host_numerics::referenceBatchLayout(matrix, 4, 0, HIPBLAS_OP_N, 2, false);
    EXPECT_EQ(emptyColumns, Layout(Shape{4, 0}, {1, 4}, 34));

    const Layout emptyRows
        = hipblaslt::host_numerics::referenceBatchLayout(matrix, 0, 4, HIPBLAS_OP_T, 2, false);
    EXPECT_EQ(emptyRows, Layout(Shape{0, 4}, {4, 1}, 34));

    const Layout separate
        = hipblaslt::host_numerics::referenceBatchLayout(matrix, 4, 0, HIPBLAS_OP_N, 2, true);
    EXPECT_EQ(separate, Layout(Shape{4, 0}, {1, 4}));
}

TEST(HostNumericsCblasBridge, EmptyOutputShapesAreNoOps)
{
    const std::array<float, 6> values{1, 2, 3, 4, 5, 6};

    EXPECT_NO_THROW(testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
                                                      HIPBLAS_OP_N,
                                                      0,
                                                      3,
                                                      2,
                                                      1.0f,
                                                      nullptr,
                                                      1,
                                                      values.data(),
                                                      2,
                                                      0.0f,
                                                      nullptr,
                                                      1,
                                                      nullptr,
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
                                                      HIP_R_32F));

    EXPECT_NO_THROW(testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
                                                      HIPBLAS_OP_N,
                                                      3,
                                                      0,
                                                      2,
                                                      1.0f,
                                                      values.data(),
                                                      3,
                                                      nullptr,
                                                      2,
                                                      0.0f,
                                                      nullptr,
                                                      3,
                                                      nullptr,
                                                      3,
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
                                                      HIP_R_32F));
}

TEST(HostNumericsCblasBridge, ComplexScaleCUsesOnlyItsRealComponent)
{
    using Complex = std::complex<float>;

    const Complex a{0.0f, 0.0f};
    const Complex b{0.0f, 0.0f};
    const Complex c{3.0f, 4.0f};
    Complex       d{-1.0f, -1.0f};

    testHipblasltReferenceGemm<Complex>(HIPBLAS_OP_N,
                                        HIPBLAS_OP_N,
                                        1,
                                        1,
                                        1,
                                        Complex(0.0f, 0.0f),
                                        &a,
                                        1,
                                        &b,
                                        1,
                                        Complex(1.0f, 0.0f),
                                        &c,
                                        1,
                                        &d,
                                        1,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        Complex(1.0f, 0.0f),
                                        false,
                                        false,
                                        HIP_C_32F,
                                        HIP_C_32F,
                                        HIP_C_32F,
                                        HIP_C_32F,
                                        HIP_C_32F,
                                        HIP_C_32F,
                                        false,
                                        false,
                                        Complex(2.0f, 9.0f));

    EXPECT_EQ(d, Complex(6.0f, 8.0f));
}

TEST(HostNumericsCblasBridge, LargeProblemUsesAcceleratedBackend)
{
    constexpr int64_t  m = 601;
    std::vector<float> a(m);
    for(int64_t row = 0; row < m; ++row)
        a[row] = static_cast<float>(row % 7);
    const std::array<float, 1> b{2};
    std::vector<float>         d(m, 1);

    testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
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
                                      HIP_R_32F);

    for(int64_t row = 0; row < m; ++row)
        EXPECT_FLOAT_EQ(d[row], 6 * a[row] + 4);
}

TEST(HostNumericsCblasBridge, ZeroReductionDoesNotRequireBlasOperands)
{
    std::array<float, 1> d{2.0f};

    testHipblasltReferenceGemm<float>(HIPBLAS_OP_N,
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
                                      HIP_R_32F);

    EXPECT_FLOAT_EQ(d[0], 6.0f);
}
