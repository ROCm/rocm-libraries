// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <hipBuffer.hpp>
#include <hipblaslt/client/MatmulProblem.hpp>
#include <hipblaslt/host_numerics/HipblasltDataInitialization.hpp>
#include <hipblaslt/host_numerics/MatrixTransformReference.hpp>
#include <hipblaslt/host_numerics/Types.hpp>
#include <hipblaslt/host_numerics/near.hpp>
#include <hipblaslt/host_numerics/norm.hpp>

#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <roc/host_numerics/backends/blas.hpp>
#include <roc/host_numerics/epilogue.hpp>
#include <roc/host_numerics/tensor_operations.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{
    uint64_t seedForMatrixRole(uint64_t seed, hipblaslt::host_numerics::MatrixRole role)
    {
        using hipblaslt::host_numerics::MatrixRole;
        using hipblaslt::host_numerics::initialization::OperandSequence;
        using hipblaslt::host_numerics::initialization::seedForSequence;

        switch(role)
        {
        case MatrixRole::A:
            return seedForSequence(seed, OperandSequence::MatrixA);
        case MatrixRole::B:
            return seedForSequence(seed, OperandSequence::MatrixB);
        case MatrixRole::C:
            return seedForSequence(seed, OperandSequence::MatrixC);
        }
        throw std::invalid_argument("Unsupported hipBLASLt matrix role.");
    }

    std::vector<float>
        groupedValues(size_t                                                    size,
                      hipblaslt_initialization                                  mode,
                      hipblaslt::host_numerics::initialization::OperandSequence operand,
                      uint64_t seed = hipblaslt::host_numerics::defaultInitializationSeed)
    {
        using namespace roc::host_numerics;
        using namespace hipblaslt::host_numerics;
        const Tensor generated = generate(
            ScalarType::Float32,
            Shape{size},
            groupedGemmInitializationRecipe(ScalarType::Float32,
                                            mode,
                                            operand,
                                            initialization::seedForSequence(seed, operand)));
        std::vector<float> result(size);
        generated.copyLogicalElementsToEncodedStorage(std::as_writable_bytes(std::span(result)));
        return result;
    }

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
    void testTensorMatmulOperations(hipblasOperation_t transA,
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
        const auto scaleCValue  = computeInterfaceValue(scaleC);
        const auto scaleDValue  = computeInterfaceValue(scaleD);
        const auto matrixLayout = [](size_t             rows,
                                     size_t             columns,
                                     int64_t            leadingDimension,
                                     hipblasOperation_t operation) {
            return roc::host_numerics::Layout(roc::host_numerics::Shape{rows, columns},
                                              {operation == HIPBLAS_OP_N ? 1 : leadingDimension,
                                               operation == HIPBLAS_OP_N ? leadingDimension : 1});
        };
        const auto layoutA = matrixLayout(m, k, lda, transA);
        const auto layoutB = matrixLayout(k, n, ldb, transB);
        const auto layoutC = matrixLayout(m, n, ldc, HIPBLAS_OP_N);
        const auto layoutD = matrixLayout(m, n, ldd, HIPBLAS_OP_N);
        auto       output  = hipblaslt::host_numerics::copyTensorFromEncodedStorage(
            d, hipblaslt::host_numerics::scalarType(typeD), layoutD);
        const auto inputA = hipblaslt::host_numerics::copyTensorFromEncodedStorage(
            a, hipblaslt::host_numerics::scalarType(typeA), layoutA);
        const auto inputB = hipblaslt::host_numerics::copyTensorFromEncodedStorage(
            b, hipblaslt::host_numerics::scalarType(typeB), layoutB);
        const auto inputC = hipblaslt::host_numerics::copyTensorFromEncodedStorage(
            c, hipblaslt::host_numerics::scalarType(typeC), layoutC);

        using namespace roc::host_numerics;
        const ScalarType accumulatorType
            = hipblaslt::host_numerics::referenceAccumulatorType(coefficientType);
        MatmulOptions options(accumulatorType);
        options.conjugateA = transA == HIPBLAS_OP_C;
        options.conjugateB = transB == HIPBLAS_OP_C;
        const ScalarType inputComputeTypeA
            = scaleAIsMx ? inputA.type()
                         : hipblaslt::host_numerics::referenceComputeType(computeInputTypeA);
        const ScalarType inputComputeTypeB
            = scaleBIsMx ? inputB.type()
                         : hipblaslt::host_numerics::referenceComputeType(computeInputTypeB);
        if(inputComputeTypeA != inputA.type())
            options.computeTypeA = inputComputeTypeA;
        if(inputComputeTypeB != inputB.type())
            options.computeTypeB = inputComputeTypeB;

        const auto vector = [&](const void* values, size_t elements) {
            return hipblaslt::host_numerics::copyTensorFromEncodedStorage(
                values,
                hipblaslt::host_numerics::scalarType(coefficientType),
                Layout::contiguousLastDimensionFastest(Shape{elements}));
        };
        if(scaleA && !scaleAIsMx)
            options.preQuantizationScalesA.push_back(
                vector(scaleA, scaleAIsVector ? static_cast<size_t>(m) : 1).expandDims(1));
        if(alphaVector)
            options.preQuantizationScalesA.push_back(
                vector(alphaVector, static_cast<size_t>(m)).expandDims(1));
        if(scaleB && !scaleBIsMx)
            options.preQuantizationScalesB.push_back(
                vector(scaleB, scaleBIsVector ? static_cast<size_t>(n) : 1).expandDims(0));

        const Tensor alphaTensor
            = hipblaslt::host_numerics::scalarValue(computeInterfaceValue(alpha), coefficientType);
        const Tensor betaTensor
            = hipblaslt::host_numerics::scalarValue(computeInterfaceValue(beta), coefficientType);
        const Tensor scaleCTensor
            = hipblaslt::host_numerics::realOnlyScalarValue(&scaleCValue, coefficientType);
        const Tensor scaleDTensor
            = hipblaslt::host_numerics::scalarValue(&scaleDValue, coefficientType);

        std::optional<Tensor> referenceAccumulator;
        if(alphaTensor.item<std::complex<double>>() != std::complex<double>(0.0, 0.0)
           && inputA.shape()[1] != 0)
        {
            const Tensor product = matmulWithBlasBackend(inputA, inputB, accumulatorType, options);
            referenceAccumulator = multiply(product, alphaTensor, accumulatorType, accumulatorType);
        }
        if(betaTensor.item<std::complex<double>>() != std::complex<double>(0.0, 0.0))
        {
            const Tensor cScale
                = multiply(betaTensor, scaleCTensor, accumulatorType, accumulatorType);
            Tensor addend = multiply(inputC, cScale, accumulatorType, accumulatorType);
            referenceAccumulator
                = referenceAccumulator
                      ? add(*referenceAccumulator, addend, accumulatorType, accumulatorType)
                      : std::move(addend);
        }
        if(!referenceAccumulator)
            referenceAccumulator.emplace(accumulatorType,
                                         Shape{inputA.shape()[0], inputB.shape()[1]});

        EpilogueOptions conversion(accumulatorType);
        conversion.outputScale = scaleDTensor;
        if(output.type() == ScalarType::Int8)
            conversion.outputConversion = OutputConversion::SaturatingInt8;
        referenceEpilogueInto(*referenceAccumulator, {.output = output}, conversion);

        hipblaslt::host_numerics::copyTensorEncodedBackingStorageToBuffer(
            d, storageBytesForLayout(output.type(), output.layout()), output);
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

TEST(HostNumericsDataInitializationBridge, GeneratesComplexTrigonometricValues)
{
    using namespace roc::host_numerics;
    using namespace hipblaslt::host_numerics;

    const Tensor values = generate(ScalarType::ComplexFloat32,
                                   Shape{4},
                                   initializationRecipe(ScalarType::ComplexFloat32,
                                                        hipblaslt_initialization::trig_float,
                                                        17,
                                                        TrigonometricComponent::Sine));

    for(size_t index = 0; index < values.elementCount(); ++index)
    {
        const auto value = values.loadAs<std::complex<float>>({index});
        EXPECT_FLOAT_EQ(value.real(), std::sin(static_cast<float>(index)));
        EXPECT_FLOAT_EQ(value.imag(), std::cos(static_cast<float>(index)));
    }
}

TEST(HostNumericsDataInitializationBridge, ComplexRandomUsesTypedCartesianDomains)
{
    using namespace roc::host_numerics;

    const GenerationRecipe recipe = hipblaslt::host_numerics::initializationRecipe(
        ScalarType::ComplexFloat32,
        hipblaslt_initialization::rand_int,
        hipblaslt::host_numerics::defaultInitializationSeed,
        hipblaslt::host_numerics::TrigonometricComponent::Cosine);
    const Tensor first  = generate(ScalarType::ComplexFloat32, Shape{8}, recipe);
    const Tensor second = generate(ScalarType::ComplexFloat32, Shape{8}, recipe);
    EXPECT_TRUE(
        std::ranges::equal(first.rawEncodedBackingStorage(), second.rawEncodedBackingStorage()));

    constexpr std::array<float, 8> expectedReal{2, 6, 3, 9, 10, 8, 1, 9};
    constexpr std::array<float, 8> expectedImaginary{8, 5, 9, 5, 1, 5, 6, 7};
    for(size_t index = 0; index < first.elementCount(); ++index)
    {
        const auto value = first.loadAs<std::complex<float>>({index});
        EXPECT_EQ(value.real(), expectedReal[index]);
        EXPECT_EQ(value.imag(), expectedImaginary[index]);
    }
}

TEST(HostNumericsDataInitializationBridge, GroupedGemmUsesStableRoleSequencesAndDefaultSeed)
{
    using namespace hipblaslt::host_numerics;
    using namespace roc::host_numerics;

    const auto a = groupedValues(
        5, hipblaslt_initialization::rand_int, initialization::OperandSequence::MatrixA);
    const auto b = groupedValues(
        7, hipblaslt_initialization::rand_int, initialization::OperandSequence::MatrixB);
    const auto c = groupedValues(
        4, hipblaslt_initialization::rand_int, initialization::OperandSequence::MatrixC);
    const auto bias = groupedValues(
        3, hipblaslt_initialization::rand_int, initialization::OperandSequence::Bias);

    const auto expected = [](size_t size, initialization::OperandSequence sequence) {
        std::vector<float>          values(size);
        GenerationRecipe::Component component
            = GenerationRecipe::uniformInteger({.lower = 1, .upper = 10});
        if(sequence == initialization::OperandSequence::MatrixB)
            component
                = component.withAlternatingSign({.dimensions = {0}, .negativeWhenOdd = false});
        const Tensor generated = generate(
            ScalarType::Float32,
            Shape{size},
            GenerationRecipe::realOnly(
                std::move(component),
                {.seed = initialization::seedForSequence(defaultInitializationSeed, sequence)}));
        generated.copyLogicalElementsToEncodedStorage(std::as_writable_bytes(std::span(values)));
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
        using hipblaslt::host_numerics::initialization::OperandSequence;
        return std::array{
            groupedValues(
                elements, hipblaslt_initialization::rand_int, OperandSequence::MatrixA, callerSeed),
            groupedValues(
                elements, hipblaslt_initialization::rand_int, OperandSequence::MatrixB, callerSeed),
            groupedValues(
                elements, hipblaslt_initialization::rand_int, OperandSequence::MatrixC, callerSeed),
            groupedValues(
                elements, hipblaslt_initialization::rand_int, OperandSequence::Bias, callerSeed),
        };
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
    const auto initialize = [&](hipblaslt_initialization initialization) {
        using hipblaslt::host_numerics::initialization::OperandSequence;
        return std::tuple{
            groupedValues(4, initialization, OperandSequence::MatrixA),
            groupedValues(4, initialization, OperandSequence::MatrixB),
            groupedValues(4, initialization, OperandSequence::MatrixC),
            groupedValues(4, initialization, OperandSequence::Bias),
        };
    };
    const auto expectHplRange = [](const auto& values) {
        for(const float value : values)
        {
            EXPECT_GE(value, -0.5f);
            EXPECT_LE(value, 0.5f);
        }
    };

    auto [a, b, c, bias] = initialize(hipblaslt_initialization::hpl);
    expectHplRange(a);
    expectHplRange(b);
    expectHplRange(c);
    expectHplRange(bias);

    std::tie(a, b, c, bias) = initialize(hipblaslt_initialization::special);
    for(const float value : a)
        EXPECT_EQ(value, 65'280.0f);
    for(const float value : b)
        EXPECT_EQ(value, 0.0000607967376708984375f);
    expectHplRange(c);
    expectHplRange(bias);
}

TEST(HostNumericsDataInitializationBridge,
     GroupedGemmHandlesZeroAndRejectsUnsupportedInitialization)
{
    using hipblaslt::host_numerics::initialization::OperandSequence;
    const auto a    = groupedValues(1, hipblaslt_initialization::zero, OperandSequence::MatrixA);
    const auto b    = groupedValues(1, hipblaslt_initialization::zero, OperandSequence::MatrixB);
    const auto c    = groupedValues(1, hipblaslt_initialization::zero, OperandSequence::MatrixC);
    const auto bias = groupedValues(1, hipblaslt_initialization::zero, OperandSequence::Bias);
    EXPECT_EQ(a[0], 0.0f);
    EXPECT_EQ(b[0], 0.0f);
    EXPECT_EQ(c[0], 0.0f);
    EXPECT_EQ(bias[0], 0.0f);

    EXPECT_THROW(hipblaslt::host_numerics::groupedGemmInitializationRecipe(
                     roc::host_numerics::ScalarType::Float32,
                     hipblaslt_initialization::norm_dist,
                     OperandSequence::MatrixA,
                     hipblaslt::host_numerics::defaultInitializationSeed),
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

    using namespace roc::host_numerics;
    const Tensor observedTensor = hipblaslt::host_numerics::copyTensorFromEncodedStorage(
        observed.data(),
        observed.size(),
        hipblaslt::host_numerics::matrixTransformLayout(
            rows, columns, batches, 3, batchStride, false, false));
    const Tensor aTensor = hipblaslt::host_numerics::copyTensorFromEncodedStorage(
        a.data(),
        a.size(),
        hipblaslt::host_numerics::matrixTransformLayout(
            rows, columns, batches, 2, batchStride, true, true));
    const Tensor bTensor = hipblaslt::host_numerics::copyTensorFromEncodedStorage(
        b.data(),
        b.size(),
        hipblaslt::host_numerics::matrixTransformLayout(
            rows, columns, batches, 4, batchStride, true, false));

    const auto comparison = hipblaslt::host_numerics::referenceMatrixTransform(
        observedTensor, aTensor, bTensor, 2.0, -1.0);
    EXPECT_EQ(comparison.compared, rows * columns * batches);
    EXPECT_TRUE(comparison.passed());
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
    EXPECT_THROW(sum_error_tolerance_for_compute_type(HIP_R_8I), std::invalid_argument);
    EXPECT_THROW(norm_tolerance(roc::host_numerics::ScalarType::Count), std::invalid_argument);
}

TEST(HostNumericsDataInitializationBridge, CounterBasedGenerationIsRepeatable)
{
    using namespace roc::host_numerics;
    using namespace hipblaslt::host_numerics;

    const GenerationRecipe recipe = initializationRecipe(ScalarType::Float32,
                                                         hipblaslt_initialization::norm_dist,
                                                         17,
                                                         TrigonometricComponent::Cosine);
    const Tensor           first  = generate(ScalarType::Float32, Shape{16}, recipe);
    const Tensor           second = generate(ScalarType::Float32, Shape{16}, recipe);
    EXPECT_TRUE(
        std::ranges::equal(first.rawEncodedBackingStorage(), second.rawEncodedBackingStorage()));
}

TEST(HostNumericsDataInitializationBridge, DirectMatrixNormalGenerationIsRepeatable)
{
    using namespace roc::host_numerics;
    using namespace hipblaslt::host_numerics;

    const Layout layout(Shape{4, 4}, {1, 4});
    auto         initialize = [&] {
        Tensor result(ScalarType::Float32, layout);
        initializeMatrix(result,
                         MatrixRole::A,
                         hipblaslt_initialization::norm_dist,
                         seedForMatrixRole(defaultInitializationSeed, MatrixRole::A));
        return result;
    };
    const Tensor first  = initialize();
    const Tensor second = initialize();

    EXPECT_TRUE(
        std::ranges::equal(first.rawEncodedBackingStorage(), second.rawEncodedBackingStorage()));
}

TEST(HostNumericsDataInitializationBridge, TranslatesInitializationModesToComponentRecipes)
{
    using namespace roc::host_numerics;
    using namespace hipblaslt::host_numerics;

    const Layout layout(Shape{2, 2}, {1, 3});
    const Tensor values = generate(
        ScalarType::Float32,
        layout,
        initializationRecipe(ScalarType::Float32,
                             hipblaslt_initialization::rand_int,
                             defaultInitializationSeed));

    for(const size_t index : {size_t{0}, size_t{1}, size_t{3}, size_t{4}})
    {
        const float value
            = values.shareStorageWithLayout(Layout(Shape{5}, {1})).loadAs<float>({index});
        EXPECT_EQ(value, std::trunc(value));
        EXPECT_GE(value, 1);
        EXPECT_LE(value, 10);
    }
    EXPECT_EQ(values.shareStorageWithLayout(Layout(Shape{5}, {1})).loadAs<float>({2}), 0);

    const Tensor nanValues = generate(
        ScalarType::Float32,
        Shape{8},
        initializationRecipe(ScalarType::Float32,
                             hipblaslt_initialization::nan,
                             defaultInitializationSeed));
    for(size_t index = 0; index < nanValues.elementCount(); ++index)
        EXPECT_TRUE(std::isnan(nanValues.loadAs<float>({index})));

    constexpr std::array integerTypes{
        ScalarType::Boolean,
        ScalarType::UInt8,
        ScalarType::Int8,
        ScalarType::UInt16,
        ScalarType::Int16,
        ScalarType::UInt32,
        ScalarType::Int32,
        ScalarType::UInt64,
        ScalarType::Int64,
        ScalarType::Int4,
    };
    for(const ScalarType type : integerTypes)
    {
        const Tensor zeroOrOne = generate(
            type,
            Shape{64},
            initializationRecipe(type, hipblaslt_initialization::uniform_01, 17));
        for(size_t index = 0; index < zeroOrOne.elementCount(); ++index)
        {
            const int64_t value = zeroOrOne.loadAs<int64_t>({index});
            EXPECT_TRUE(value == 0 || value == 1) << scalarTypeName(type);
        }
    }
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
        using namespace roc::host_numerics;
        const Tensor value = generate(hipblaslt::host_numerics::scalarType(type),
                                      Shape{1},
                                      GenerationRecipe::realOnly(GenerationRecipe::zero()));
        EXPECT_EQ(value.rawEncodedBackingStorage().front(), std::byte{0})
            << "hipDataType=" << static_cast<int>(type);
    }
}

TEST(HostNumericsDataInitializationBridge, GeneratesProblemLevelMatrixRecipes)
{
    using namespace roc::host_numerics;
    using namespace hipblaslt::host_numerics;

    const Layout exactLayout(Shape{2, 3, 2}, {1, 4, 12});
    const auto   initializeExact = [&] {
        Tensor result(ScalarType::Float32, exactLayout);
        initializeMatrix(result,
                         MatrixRole::B,
                         hipblaslt_initialization::integer_exact,
                         seedForMatrixRole(defaultInitializationSeed, MatrixRole::B));
        return result;
    };
    const Tensor exactMatrix = initializeExact();
    const Tensor exactReplay = initializeExact();
    EXPECT_EQ(exactMatrix.layout(), exactLayout);
    EXPECT_EQ(exactMatrix.rawEncodedBackingStorage().size(),
              storageBytesForLayout(ScalarType::Float32, exactLayout));
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
    const size_t exactStorageElements
        = exactMatrix.rawEncodedBackingStorage().size() / sizeof(float);
    Tensor exactAllocation
        = exactMatrix.shareStorageWithLayout(Layout(Shape{exactStorageElements}, {1}));
    for(const size_t index : {size_t{2},
                              size_t{3},
                              size_t{6},
                              size_t{7},
                              size_t{10},
                              size_t{11},
                              size_t{14},
                              size_t{15},
                              size_t{18},
                              size_t{19}})
        EXPECT_EQ(exactAllocation.loadAs<float>({index}), 0);

    Tensor probeMatrix(ScalarType::Float16, Layout(Shape{4, 2}, {1, 4}));
    initializeMatrix(probeMatrix,
                     MatrixRole::B,
                     hipblaslt_initialization::fp16_accumulator_probe,
                     seedForMatrixRole(defaultInitializationSeed, MatrixRole::B));
    for(size_t column = 0; column < 2; ++column)
        for(size_t row = 0; row < 4; ++row)
            EXPECT_EQ(probeMatrix.loadAs<float>({row, column}), row % 2 == 0 ? 2 : -2);
}

TEST(HostNumericsDataInitializationBridge,
     OneSpecialInitializationSelectsALogicalElementAndPreservesGaps)
{
    using namespace roc::host_numerics;
    using namespace hipblaslt::host_numerics;

    const Layout expectedLayout(Shape{2, 3, 2}, {1, 4, 16});
    Tensor       actual(ScalarType::Float32, expectedLayout);
    initializeMatrix(actual,
                     MatrixRole::A,
                     hipblaslt_initialization::norm_dist_one_special,
                     seedForMatrixRole(oneSpecialInitializationSeed, MatrixRole::A));
    EXPECT_EQ(actual.layout(), expectedLayout);
    EXPECT_EQ(actual.rawEncodedBackingStorage().size(),
              storageBytesForLayout(ScalarType::Float32, expectedLayout));

    Tensor baseline(ScalarType::Float32, expectedLayout);
    generate(baseline,
             initializationRecipe(
                 ScalarType::Float32,
                 hipblaslt_initialization::norm_dist,
                 seedForMatrixRole(oneSpecialInitializationSeed, MatrixRole::A)));

    constexpr size_t  expectedSpecialLogicalIndex = 6;
    const size_t      storageElements = actual.rawEncodedBackingStorage().size() / sizeof(float);
    std::vector<bool> logicalStorageElements(storageElements, false);
    size_t            infinityCount = 0;
    for(size_t batch = 0; batch < expectedLayout.shape()[2]; ++batch)
        for(size_t column = 0; column < expectedLayout.shape()[1]; ++column)
            for(size_t row = 0; row < expectedLayout.shape()[0]; ++row)
            {
                const size_t logicalIndex
                    = row
                      + expectedLayout.shape()[0] * (column + expectedLayout.shape()[1] * batch);
                const size_t storageIndex            = row + 4 * column + 16 * batch;
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

    const Tensor allocation = actual.shareStorageWithLayout(Layout(Shape{storageElements}, {1}));
    for(size_t storageIndex = 0; storageIndex < logicalStorageElements.size(); ++storageIndex)
        if(!logicalStorageElements[storageIndex])
            EXPECT_EQ(allocation.loadAs<float>({storageIndex}), 0.0f);

    Tensor negative(ScalarType::Float32, expectedLayout);
    initializeMatrix(negative,
                     MatrixRole::A,
                     hipblaslt_initialization::norm_dist_one_special,
                     seedForMatrixRole(oneSpecialInitializationSeed, MatrixRole::A),
                     false,
                     OneSpecialValue::NegativeInfinity);
    const float negativeInfinity = negative.loadAs<float>({0, 0, 1});
    EXPECT_TRUE(std::isinf(negativeInfinity));
    EXPECT_TRUE(std::signbit(negativeInfinity));

    Tensor nan(ScalarType::Float32, expectedLayout);
    initializeMatrix(nan,
                     MatrixRole::A,
                     hipblaslt_initialization::norm_dist_one_special,
                     seedForMatrixRole(oneSpecialInitializationSeed, MatrixRole::A),
                     false,
                     OneSpecialValue::NaN);
    EXPECT_TRUE(std::isnan(nan.loadAs<float>({0, 0, 1})));

    Tensor fiveElements(ScalarType::Float32, Layout(Shape{5, 1}, {1, 7}));
    initializeMatrix(fiveElements,
                     MatrixRole::A,
                     hipblaslt_initialization::norm_dist_one_special,
                     seedForMatrixRole(oneSpecialInitializationSeed, MatrixRole::A));
    for(size_t row = 0; row < 5; ++row)
        EXPECT_EQ(std::isinf(fiveElements.loadAs<float>({row, 0})), row == 4);
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
        const auto initialized = [&](MatrixRole role) {
            Tensor         result(ScalarType::Float32, Shape{8, 8});
            const uint64_t baseSeed = mode == hipblaslt_initialization::norm_dist_one_special
                                          ? oneSpecialInitializationSeed
                                          : defaultInitializationSeed;
            initializeMatrix(result, role, mode, seedForMatrixRole(baseSeed, role));
            return result;
        };

        const auto matrixA = values(initialized(MatrixRole::A));
        EXPECT_EQ(matrixA, values(initialized(MatrixRole::A)));
        const auto matrixB = values(initialized(MatrixRole::B));
        EXPECT_EQ(matrixB, values(initialized(MatrixRole::B)));
        const auto matrixC = values(initialized(MatrixRole::C));
        EXPECT_EQ(matrixC, values(initialized(MatrixRole::C)));

        EXPECT_NE(matrixA, matrixB) << "initialization=" << static_cast<int>(mode);
        EXPECT_NE(matrixA, matrixC) << "initialization=" << static_cast<int>(mode);
        EXPECT_NE(matrixB, matrixC) << "initialization=" << static_cast<int>(mode);
    }
}

TEST(HostNumericsDataInitializationBridge, MatrixInitializationUsesTheExactCallerSeed)
{
    using namespace roc::host_numerics;
    using namespace hipblaslt::host_numerics;

    const auto initialized = [](MatrixRole role, uint64_t seed) {
        Tensor result(ScalarType::Float32, Shape{8, 8});
        initializeMatrix(result, role, hipblaslt_initialization::norm_dist, seed);
        return result;
    };

    const Tensor first  = initialized(MatrixRole::A, 17);
    const Tensor replay = initialized(MatrixRole::A, 17);
    const Tensor next   = initialized(MatrixRole::A, 18);
    const Tensor c      = initialized(MatrixRole::C, 17);

    EXPECT_TRUE(
        std::ranges::equal(first.rawEncodedBackingStorage(), replay.rawEncodedBackingStorage()));
    EXPECT_FALSE(
        std::ranges::equal(first.rawEncodedBackingStorage(), next.rawEncodedBackingStorage()));
    EXPECT_TRUE(std::ranges::equal(first.rawEncodedBackingStorage(), c.rawEncodedBackingStorage()));
}

TEST(HostNumericsDataInitializationBridge, PositiveOnlyMatrixPolicyIsExplicit)
{
    using namespace hipblaslt::host_numerics;

    Tensor         signedMatrix(ScalarType::Float32, Shape{8, 8});
    Tensor         positiveMatrix(ScalarType::Float32, Shape{8, 8});
    const uint64_t seed = seedForMatrixRole(defaultInitializationSeed, MatrixRole::A);
    initializeMatrix(signedMatrix, MatrixRole::A, hipblaslt_initialization::hpl, seed);
    initializeMatrix(positiveMatrix,
                     MatrixRole::A,
                     hipblaslt_initialization::hpl,
                     seed,
                     false,
                     std::nullopt,
                     true);
    for(size_t column = 0; column < 8; ++column)
        for(size_t row = 0; row < 8; ++row)
        {
            const float signedValue   = signedMatrix.loadAs<float>({row, column});
            const float positiveValue = positiveMatrix.loadAs<float>({row, column});
            EXPECT_FLOAT_EQ(positiveValue, std::abs(signedValue));
        }
}

TEST(HostNumericsDataInitializationBridge, UnsupportedModesThrowInsteadOfProducingFallbackData)
{
    using namespace hipblaslt::host_numerics;

    Tensor value(ScalarType::Float32, Shape{1});
    EXPECT_THROW(initializeMatrix(value,
                                  MatrixRole::A,
                                  hipblaslt_initialization::fp16_accumulator_probe,
                                  defaultInitializationSeed),
                 std::invalid_argument);

    EXPECT_THROW(
        initializeMatrix(value,
                         MatrixRole::A,
                         static_cast<hipblaslt_initialization>(std::numeric_limits<int>::max()),
                         defaultInitializationSeed),
        std::invalid_argument);

    EXPECT_THROW(initializationRecipe(ScalarType::Float32,
                                      hipblaslt_initialization::fp16_accumulator_probe,
                                      defaultInitializationSeed,
                                      TrigonometricComponent::Cosine),
                 std::invalid_argument);
}

TEST(HostNumericsDataInitializationBridge, PinnedTensorInitializationUploadsToDevice)
{
    using namespace hipblaslt::host_numerics;

    const Layout     layout(Shape{2, 3, 2}, {1, 4, 12});
    constexpr size_t elements = 22;
    HipHostBuffer    host(HIP_R_32F, elements);
    Tensor           expected = host.tensor(ScalarType::Float32, layout);
    std::ranges::fill(expected.rawEncodedBackingStorage(), std::byte{0});
    initializeMatrix(expected,
                     MatrixRole::B,
                     hipblaslt_initialization::integer_exact,
                     seedForMatrixRole(defaultInitializationSeed, MatrixRole::B));

    HipDeviceBuffer device(HIP_R_32F, elements);
    ASSERT_EQ(device.memcheck(), hipSuccess);
    EXPECT_EQ(synchronize(device, host), hipSuccess);
    std::vector<std::byte> observed(expected.rawEncodedBackingStorage().size());
    EXPECT_EQ(hipMemcpy(observed.data(), device.buf(), observed.size(), hipMemcpyDeviceToHost),
              hipSuccess);
    EXPECT_TRUE(
        std::equal(observed.begin(), observed.end(), expected.rawEncodedBackingStorage().begin()));
}

TEST(HostNumericsCblasBridge, DistinctHalfCAndFloatD)
{
    const std::array<float, 6>   a{1, 4, 2, 5, 3, 6};
    const std::array<float, 6>   b{7, 9, 11, 8, 10, 12};
    std::array<hipblasLtHalf, 6> c{1, 2, -99, 3, 4, -99};
    const auto                   originalC = c;
    std::array<float, 4>         d{-1, -2, -3, -4};

    testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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

TEST(HostNumericsCblasBridge, ComposesAlphaMatmulAndBetaCWithTensorOperations)
{
    using namespace roc::host_numerics;

    const std::array<float, 4> a{1, 2, 3, 4};
    const std::array<float, 4> b{5, 6, 7, 8};
    const std::array<float, 4> c{1, 1, 1, 1};
    const Tensor               inputA = Tensor::copyNativeValues<float>(Shape{2, 2}, a);
    const Tensor               inputB = Tensor::copyNativeValues<float>(Shape{2, 2}, b);
    const Tensor               inputC = Tensor::copyNativeValues<float>(Shape{2, 2}, c);
    const Tensor output = matmul(inputA, inputB, ScalarType::Float32) * 2.0f + inputC * 3.0f;

    EXPECT_FLOAT_EQ(output.loadAs<float>({0, 0}), 41.0f);
    EXPECT_FLOAT_EQ(output.loadAs<float>({0, 1}), 47.0f);
    EXPECT_FLOAT_EQ(output.loadAs<float>({1, 0}), 89.0f);
    EXPECT_FLOAT_EQ(output.loadAs<float>({1, 1}), 103.0f);
}

TEST(HostNumericsCblasBridge, AppliesScaleCInsideSharedGemm)
{
    const float a      = 4.0f;
    const float b      = 5.0f;
    const float c      = 7.0f;
    float       d      = -1.0f;
    const float scaleC = 2.0f;
    const float scaleD = 1.0f;
    testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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

    testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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

    testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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

    testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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

    testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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

    testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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

    testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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

    testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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
    testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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

    testTensorMatmulOperations<int32_t>(HIPBLAS_OP_N,
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

    testTensorMatmulOperations<float>(HIPBLAS_OP_T,
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

    testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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

    testTensorMatmulOperations<Complex>(HIPBLAS_OP_C,
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
    using Complex = std::complex<float>;

    Arguments arguments{};
    arguments.init();
    arguments.a_type     = HIP_C_32F;
    arguments.b_type     = HIP_C_32F;
    arguments.c_type     = HIP_C_32F;
    arguments.d_type     = HIP_C_32F;
    const auto dataTypes = hipblaslt::client::resolveMatmulDataTypes(arguments);

    const Complex a{1.0f, 2.0f};
    const Complex b{3.0f, -1.0f};
    const Complex c{-2.0f, 4.0f};
    const Complex alpha{2.0f, 3.0f};
    const Complex beta{4.0f, 5.0f};
    Complex       d{};

    testTensorMatmulOperations<Complex>(HIPBLAS_OP_N,
                                        HIPBLAS_OP_N,
                                        1,
                                        1,
                                        1,
                                        alpha,
                                        &a,
                                        1,
                                        &b,
                                        1,
                                        beta,
                                        &c,
                                        1,
                                        &d,
                                        1,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        Complex{1.0f, 0.0f},
                                        false,
                                        false,
                                        HIP_C_32F,
                                        HIP_C_32F,
                                        HIP_C_32F,
                                        HIP_C_32F,
                                        dataTypes.computeInputA,
                                        dataTypes.computeInputB);

    const Complex expected = alpha * a * b + beta * c;
    EXPECT_FLOAT_EQ(d.real(), expected.real());
    EXPECT_FLOAT_EQ(d.imag(), expected.imag());
}

TEST(HostNumericsCblasBridge, BuildsEmptyBatchLayoutsWithoutAddressingAnElement)
{
    using namespace roc::host_numerics;

    const hipblaslt::client::MatmulMatrix matrix{HIP_R_32F, Layout(Shape{4, 0, 3}, {1, 4, 17}), 51};

    const Layout emptyColumns = matrix.logicalBatchLayout(HIPBLAS_OP_N, 2, false);
    EXPECT_EQ(emptyColumns, Layout(Shape{4, 0}, {1, 4}, 34));

    const Layout emptyRows = matrix.logicalBatchLayout(HIPBLAS_OP_T, 2, false);
    EXPECT_EQ(emptyRows, Layout(Shape{0, 4}, {4, 1}, 34));

    const Layout separate = matrix.logicalBatchLayout(HIPBLAS_OP_N, 2, true);
    EXPECT_EQ(separate, Layout(Shape{4, 0}, {1, 4}));
}

TEST(HostNumericsCblasBridge, EmptyOutputShapesAreNoOps)
{
    const std::array<float, 6> values{1, 2, 3, 4, 5, 6};

    EXPECT_NO_THROW(testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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

    EXPECT_NO_THROW(testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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

    testTensorMatmulOperations<Complex>(HIPBLAS_OP_N,
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

    testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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

    testTensorMatmulOperations<float>(HIPBLAS_OP_N,
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
