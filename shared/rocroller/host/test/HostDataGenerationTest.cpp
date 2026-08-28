// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/HostNumerics/HostDataGeneration.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <roc/host_numerics/mx.hpp>
#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/Utilities/Settings.hpp>

namespace
{
    using namespace rocRoller;
    using namespace rocRoller::HostNumerics;
    using roc::host_numerics::GenerationRecipe;
    using roc::host_numerics::IndexOrder;
    using roc::host_numerics::Layout;
    using roc::host_numerics::MxDataGeneration;
    using roc::host_numerics::MxGenerationProblem;
    using roc::host_numerics::MxScaleGenerationMode;
    using roc::host_numerics::ScalarType;
    using roc::host_numerics::Shape;
    using roc::host_numerics::Tensor;

    void require(bool condition, std::string const& message)
    {
        if(!condition)
            throw std::runtime_error(message);
    }

    std::vector<uint8_t> bytes(Tensor const& tensor)
    {
        std::vector<uint8_t> result(tensor.rawEncodedBackingStorage().size());
        std::transform(tensor.rawEncodedBackingStorage().begin(),
                       tensor.rawEncodedBackingStorage().end(),
                       result.begin(),
                       [](std::byte value) { return std::to_integer<uint8_t>(value); });
        return result;
    }

    bool sameStorage(Tensor const& first, Tensor const& second)
    {
        return first.type() == second.type() && first.layout() == second.layout()
               && bytes(first) == bytes(second);
    }

    GeneratedGEMMInputs generate(TensorDescriptor const&   descriptorA,
                                 TensorDescriptor const&   descriptorB,
                                 TensorDescriptor const&   descriptorC,
                                 DataInitialization const& initialization,
                                 DataType                  scaleTypeA     = DataType::None,
                                 DataType                  scaleTypeB     = DataType::None,
                                 size_t                    scaleBlockSize = 1,
                                 uint32_t                  seed           = 31415u)
    {
        return generateGEMMInputs(descriptorA,
                                  descriptorB,
                                  descriptorC,
                                  initialization,
                                  initialization,
                                  initialization,
                                  scaleTypeA,
                                  scaleTypeB,
                                  scaleBlockSize,
                                  -1.0f,
                                  1.0f,
                                  seed);
    }

    Tensor generateC(TensorDescriptor const& descriptor,
                     DataInitialization      initialization,
                     uint32_t                seed = 31415u)
    {
        return generate(descriptor,
                        descriptor,
                        descriptor,
                        initialization,
                        DataType::None,
                        DataType::None,
                        1,
                        seed)
            .c;
    }

    void testLayoutAndSeedOffsets()
    {
        TensorDescriptor   descriptor(DataType::Float, {4, 3}, "N");
        DataInitialization bounded{DataInitializationMode::Bounded};

        auto first       = generate(descriptor, descriptor, descriptor, bounded);
        auto repeated    = generate(descriptor, descriptor, descriptor, bounded);
        auto seedPlusOne = generate(
            descriptor, descriptor, descriptor, bounded, DataType::None, DataType::None, 1, 31416u);
        auto seedPlusTwo = generate(
            descriptor, descriptor, descriptor, bounded, DataType::None, DataType::None, 1, 31417u);

        require(first.a.layout() == Layout(Shape{4, 3}, {1, 4}),
                "A descriptor layout was not preserved.");
        require(first.b.layout() == Layout(Shape{4, 3}, {1, 4}),
                "B descriptor layout was not preserved.");
        require(first.c.layout() == Layout(Shape{4, 3}, {1, 4}),
                "C descriptor layout was not preserved.");
        require(sameStorage(first.a, repeated.a) && sameStorage(first.b, repeated.b)
                    && sameStorage(first.c, repeated.c),
                "Generation is not reproducible for a fixed seed.");
        require(sameStorage(first.a, seedPlusOne.c), "A did not use the base seed plus one.");
        require(sameStorage(first.b, seedPlusTwo.c), "B did not use the base seed plus two.");
        require(!sameStorage(first.c, seedPlusOne.c),
                "Different generation seeds produced identical C storage.");
    }

    void testAllInitializationModes()
    {
        TensorDescriptor descriptor(DataType::Float, {4, 4}, "N");

        auto bounded = generateC(descriptor, DataInitialization{DataInitializationMode::Bounded});
        auto boundedView = bounded;
        for(size_t column = 0; column < 4; ++column)
            for(size_t row = 0; row < 4; ++row)
            {
                auto const value = boundedView.loadAs<float>({row, column});
                require(value >= -1.0f && value <= 1.0f, "Bounded generation exceeded [-1, 1].");
            }

        auto alternating = generateC(
            descriptor, DataInitialization{DataInitializationMode::BoundedAlternatingSign});
        auto alternatingView = alternating;
        for(size_t column = 0; column < 4; ++column)
            for(size_t row = 0; row < 4; ++row)
            {
                auto const value = alternatingView.loadAs<float>({row, column});
                auto const index = descriptor.index(row, column);
                require(index % 2 == 0 ? value >= 0.0f : value <= 0.0f,
                        "Alternating-sign generation did not follow physical storage order.");
                require(std::abs(value) <= 1.0f, "Alternating-sign generation exceeded [-1, 1].");
            }

        auto unbounded
            = generateC(descriptor, DataInitialization{DataInitializationMode::Unbounded});
        auto unboundedView        = unbounded;
        bool hasMagnitudeAboveOne = false;
        for(size_t column = 0; column < 4; ++column)
            for(size_t row = 0; row < 4; ++row)
            {
                auto const value = unboundedView.loadAs<float>({row, column});
                require(std::isfinite(value), "Unbounded generation produced a non-finite value.");
                hasMagnitudeAboveOne = hasMagnitudeAboveOne || std::abs(value) > 1.0f;
            }
        require(hasMagnitudeAboveOne,
                "Unbounded generation did not exercise values outside [-1, 1].");

        auto identity = generateC(descriptor, DataInitialization{DataInitializationMode::Identity});
        auto identityView = identity;
        for(size_t column = 0; column < 4; ++column)
            for(size_t row = 0; row < 4; ++row)
                require(identityView.loadAs<float>({row, column}) == (row == column ? 1.0f : 0.0f),
                        "Identity generation mismatch.");

        auto ones  = generateC(descriptor, DataInitialization{DataInitializationMode::Ones});
        auto zeros = generateC(descriptor, DataInitialization{DataInitializationMode::Zeros});
        for(size_t column = 0; column < 4; ++column)
            for(size_t row = 0; row < 4; ++row)
            {
                require(ones.loadAs<float>({row, column}) == 1.0f, "Ones generation mismatch.");
                require(zeros.loadAs<float>({row, column}) == 0.0f, "Zeros generation mismatch.");
            }

        auto trigonometric = generateC(
            descriptor, DataInitialization{DataInitializationMode::TrigonometricFromFloat});
        for(size_t column = 0; column < 4; ++column)
            for(size_t row = 0; row < 4; ++row)
            {
                auto const value = trigonometric.loadAs<float>({row, column});
                require(value >= -1.0f && value <= 1.0f,
                        "Trigonometric generation exceeded [-1, 1].");
            }

        DataInitialization normal{DataInitializationMode::NormalFromFloat, 2.0, 0.0};
        auto               normalValues = generateC(descriptor, normal);
        for(size_t column = 0; column < 4; ++column)
            for(size_t row = 0; row < 4; ++row)
                require(normalValues.loadAs<float>({row, column}) == 2.0f,
                        "Normal generation did not preserve its mean and deviation.");
    }

    void testPackedStorageBytes()
    {
        auto const ones = DataInitialization{DataInitializationMode::Ones};

        TensorDescriptor halfDescriptor(DataType::Half, {2, 1}, "N");
        auto             half = generateC(halfDescriptor, ones);
        require(bytes(half) == std::vector<uint8_t>({0x00, 0x3c, 0x00, 0x3c}),
                "Half unity bytes mismatch.");
        require(copyTensorStorage<Half>(half).size() == 2,
                "Half typed storage copy size mismatch.");

        TensorDescriptor bfloat16Descriptor(DataType::BFloat16, {2, 1}, "N");
        auto             bfloat16 = generateC(bfloat16Descriptor, ones);
        require(bytes(bfloat16) == std::vector<uint8_t>({0x80, 0x3f, 0x80, 0x3f}),
                "BFloat16 unity bytes mismatch.");
        require(copyTensorStorage<BFloat16>(bfloat16).size() == 2,
                "BFloat16 typed storage copy size mismatch.");

        TensorDescriptor fp4Descriptor(DataType::FP4, {8, 1}, "N");
        auto             fp4 = generateC(fp4Descriptor, ones);
        require(bytes(fp4) == std::vector<uint8_t>({0x22, 0x22, 0x22, 0x22}),
                "FP4 packed bytes mismatch.");
        require(copyTensorStorage<FP4x8>(fp4).size() == 1, "FP4 typed storage copy size mismatch.");

        TensorDescriptor fp6Descriptor(DataType::FP6, {16, 1}, "N");
        auto             fp6 = generateC(fp6Descriptor, ones);
        require(bytes(fp6)
                    == std::vector<uint8_t>(
                        {0x08, 0x82, 0x20, 0x08, 0x82, 0x20, 0x08, 0x82, 0x20, 0x08, 0x82, 0x20}),
                "FP6 packed bytes mismatch.");
        require(copyTensorStorage<FP6x16>(fp6).size() == 1,
                "FP6 typed storage copy size mismatch.");

        TensorDescriptor bf6Descriptor(DataType::BF6, {16, 1}, "N");
        auto             bf6 = generateC(bf6Descriptor, ones);
        require(bytes(bf6)
                    == std::vector<uint8_t>(
                        {0x0c, 0xc3, 0x30, 0x0c, 0xc3, 0x30, 0x0c, 0xc3, 0x30, 0x0c, 0xc3, 0x30}),
                "BF6 packed bytes mismatch.");
        require(copyTensorStorage<BF6x16>(bf6).size() == 1,
                "BF6 typed storage copy size mismatch.");

        TensorDescriptor partialFp4Descriptor(DataType::FP4, {5, 1}, "N");
        auto             partialFp4        = generateC(partialFp4Descriptor, ones);
        auto             partialFp4Storage = copyTensorStorage<FP4x8>(partialFp4);
        require(bytes(partialFp4) == std::vector<uint8_t>({0x22, 0x22, 0x02}),
                "Partial FP4 packed bytes mismatch.");
        require(partialFp4Storage.size() == 1,
                "Partial FP4 did not fit in one rocRoller upload container.");
        std::array<uint8_t, sizeof(FP4x8)> partialFp4Bytes{};
        std::memcpy(partialFp4Bytes.data(), partialFp4Storage.data(), partialFp4Bytes.size());
        require(partialFp4Bytes == std::array<uint8_t, sizeof(FP4x8)>{0x22, 0x22, 0x02, 0x00},
                "Partial FP4 upload container was not zero-padded.");
    }

    void testPaddedOrdinaryStorage()
    {
        TensorDescriptor descriptor(DataType::Float, {3, 2}, {1, 5}, 2);
        auto tensor  = generateC(descriptor, DataInitialization{DataInitializationMode::Ones});
        auto storage = copyTensorStorage<float>(tensor);

        require(tensor.layout() == Layout(Shape{3, 2}, {1, 5}, 2),
                "Padded ordinary descriptor layout was not preserved.");
        require(storage.size() == descriptor.totalAllocatedElements(),
                "Padded ordinary storage size differs from the descriptor allocation.");
        require(
            storage
                == std::vector<float>({0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}),
            "Padded ordinary storage values mismatch.");
    }

    MxGenerationProblem expectedMxProblem(ScalarType dataType,
                                          ScalarType scaleType,
                                          Shape      shape,
                                          ptrdiff_t  leadingDimension,
                                          size_t     blockAxis,
                                          uint32_t   seed)
    {
        MxDataGeneration data
            = MxDataGeneration::preserveGeneratedEncoding(GenerationRecipe::realOnly(
                GenerationRecipe::uniformFiniteEncodedValue(),
                {
                    .seed       = seed,
                    .indexOrder = IndexOrder::FirstDimensionFastest,
                    .randomDomain
                    = roc::host_numerics::mx_generation_random_domain_version_1::unboundedData,
                }));
        MxGenerationProblem problem(std::move(shape), std::move(data));
        problem.dataType         = dataType;
        problem.scaleType        = scaleType;
        problem.leadingDimension = leadingDimension;
        problem.blockAxis        = blockAxis;
        problem.blockSize        = 4;
        problem.scale            = MxScaleGenerationMode::RandomFinite;
        return problem;
    }

    void testScaledTypeBlockAndNaturalOrder()
    {
        TensorDescriptor descriptorA(DataType::FP4, {3, 8}, "T");
        TensorDescriptor descriptorB(DataType::FP4, {8, 5}, "N");
        TensorDescriptor descriptorC(DataType::Float, {3, 5}, "N");
        auto const       unbounded = DataInitialization{DataInitializationMode::Unbounded};
        auto             generated = generate(
            descriptorA, descriptorB, descriptorC, unbounded, DataType::E4M3, DataType::E5M3, 4);

        require(generated.a.layout() == Layout(Shape{3, 8}, {8, 1}),
                "Scaled A descriptor layout was not preserved.");
        require(generated.b.layout() == Layout(Shape{8, 5}, {1, 8}),
                "Scaled B descriptor layout was not preserved.");
        require(generated.scaleA && generated.scaleB,
                "Scaled generation did not return scale tensors.");
        require(generated.scaleA->type() == ScalarType::E4M3
                    && generated.scaleB->type() == ScalarType::E5M3,
                "Scale type translation mismatch.");
        require(generated.scaleA->layout() == Layout(Shape{3, 2}, {2, 1})
                    && generated.scaleB->layout() == Layout(Shape{5, 2}, {2, 1}),
                "K-contiguous canonical scale layout mismatch.");

        auto expectedA = roc::host_numerics::generateMx(
            expectedMxProblem(ScalarType::Float4E2M1, ScalarType::E4M3, Shape{8, 3}, 8, 0, 31416u));
        auto expectedB = roc::host_numerics::generateMx(
            expectedMxProblem(ScalarType::Float4E2M1, ScalarType::E5M3, Shape{8, 5}, 8, 0, 31417u));
        require(bytes(generated.a) == bytes(expectedA.data)
                    && bytes(*generated.scaleA) == bytes(expectedA.scales),
                "A data or natural scale order differs from generateMx.");
        require(bytes(generated.b) == bytes(expectedB.data)
                    && bytes(*generated.scaleB) == bytes(expectedB.scales),
                "B data or natural scale order differs from generateMx.");

        TensorDescriptor noncontiguousA(DataType::FP4, {3, 8}, "N");
        TensorDescriptor noncontiguousB(DataType::FP4, {8, 5}, "T");
        auto             noncontiguous = generate(noncontiguousA,
                                      noncontiguousB,
                                      descriptorC,
                                      unbounded,
                                      DataType::E4M3,
                                      DataType::E5M3,
                                      4);
        require(noncontiguous.scaleA && noncontiguous.scaleB,
                "K-strided scaled generation did not return scale tensors.");
        require(noncontiguous.scaleA->layout() == Layout(Shape{3, 2}, {1, 3})
                    && noncontiguous.scaleB->layout() == Layout(Shape{5, 2}, {1, 5}),
                "K-strided canonical scale layout mismatch.");

        auto expectedNoncontiguousA = roc::host_numerics::generateMx(
            expectedMxProblem(ScalarType::Float4E2M1, ScalarType::E4M3, Shape{3, 8}, 3, 1, 31416u));
        auto expectedNoncontiguousB = roc::host_numerics::generateMx(
            expectedMxProblem(ScalarType::Float4E2M1, ScalarType::E5M3, Shape{5, 8}, 5, 1, 31417u));
        require(bytes(noncontiguous.a) == bytes(expectedNoncontiguousA.data)
                    && bytes(*noncontiguous.scaleA) == bytes(expectedNoncontiguousA.scales),
                "K-strided A generation did not use logical K blocks.");
        require(bytes(noncontiguous.b) == bytes(expectedNoncontiguousB.data)
                    && bytes(*noncontiguous.scaleB) == bytes(expectedNoncontiguousB.scales),
                "K-strided B generation did not use logical K blocks.");

        TensorDescriptor fp6A(DataType::FP6, {8, 4}, "N");
        TensorDescriptor fp6B(DataType::FP6, {4, 2}, "N");
        TensorDescriptor fp6C(DataType::Float, {8, 2}, "N");
        auto             fp6 = generate(fp6A,
                            fp6B,
                            fp6C,
                            DataInitialization{DataInitializationMode::Ones},
                            DataType::E8M0,
                            DataType::E8M0,
                            4);
        require(fp6.scaleA && fp6.scaleA->type() == ScalarType::E8M0,
                "E8M0 scale type translation mismatch.");
    }

    void testUnscaledF8ModeSemantics()
    {
        auto       settings = Settings::getInstance();
        auto const ones     = DataInitialization{DataInitializationMode::Ones};

        TensorDescriptor fp8Descriptor(DataType::FP8, {1, 1}, "N");
        settings->set(Settings::F8ModeOption, F8Mode::OCP);
        auto fp8Ocp = generateC(fp8Descriptor, ones);
        require(fp8Ocp.type() == ScalarType::Float8E4M3
                    && bytes(fp8Ocp) == std::vector<uint8_t>{0x38},
                "Unscaled FP8 did not use rocRoller OCP semantics.");

        settings->set(Settings::F8ModeOption, F8Mode::NaNoo);
        auto fp8Fnuz = generateC(fp8Descriptor, ones);
        require(fp8Fnuz.type() == ScalarType::Float8E4M3Fnuz
                    && bytes(fp8Fnuz) == std::vector<uint8_t>{0x40},
                "Unscaled FP8 did not use rocRoller NaNoo semantics.");

        TensorDescriptor bf8Descriptor(DataType::BF8, {1, 1}, "N");
        settings->set(Settings::F8ModeOption, F8Mode::OCP);
        auto bf8Ocp = generateC(bf8Descriptor, ones);
        require(bf8Ocp.type() == ScalarType::Float8E5M2
                    && bytes(bf8Ocp) == std::vector<uint8_t>{0x3c},
                "Unscaled BF8 did not use rocRoller OCP semantics.");

        settings->set(Settings::F8ModeOption, F8Mode::NaNoo);
        auto bf8Fnuz = generateC(bf8Descriptor, ones);
        require(bf8Fnuz.type() == ScalarType::Float8E5M2Fnuz
                    && bytes(bf8Fnuz) == std::vector<uint8_t>{0x40},
                "Unscaled BF8 did not use rocRoller NaNoo semantics.");

        TensorDescriptor scaledFp8A(DataType::FP8, {1, 4}, "N");
        TensorDescriptor scaledFp8B(DataType::FP8, {4, 1}, "N");
        TensorDescriptor cDescriptor(DataType::Float, {1, 1}, "N");
        auto             scaled = generate(
            scaledFp8A, scaledFp8B, cDescriptor, ones, DataType::E8M0, DataType::E8M0, 4);
        require(scaled.a.type() == ScalarType::Float8E4M3
                    && bytes(scaled.a) == std::vector<uint8_t>({0x38, 0x38, 0x38, 0x38}),
                "Scaled FP8 did not retain the OCP MX encoding.");
        require(scaled.scaleA && bytes(*scaled.scaleA) == std::vector<uint8_t>{0x7f},
                "Scaled FP8 unity scale encoding mismatch.");

        settings->set(Settings::F8ModeOption, F8Mode::OCP);
    }
}

int main()
{
    // Random byte sequences intentionally differ from the legacy mt19937/OpenMP
    // generator. The migration contract is stable indexed seeds, modes, bounds,
    // layouts, packed encodings, and natural scale order.
    testLayoutAndSeedOffsets();
    testAllInitializationModes();
    testPackedStorageBytes();
    testPaddedOrdinaryStorage();
    testScaledTypeBlockAndNaturalOrder();
    testUnscaledF8ModeSemantics();
    return 0;
}
