// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "DataInitialization.hpp" // isMXTensor / Problem
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <TensileLite/Client/HostNumerics/DataInitializationHelpers.hpp>
#include <TensileLite/Client/HostNumerics/HostNumericsBridge.hpp>
#include <TensileLite/Client/HostNumerics/TensileDataGeneration.hpp>
#include <roc/host_numerics/validation.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if HIPBLASLT_ENABLE_MXDATAGENERATOR
#include <hip/hip_runtime.h>
#endif

using TensileLite::ContractionProblemGemm;
using TensileLite::DataTypeInfo;
using TensileLite::TensorDescriptor;
using TensileLite::Client::DataInitialization;
using TensileLite::Client::DataInitializationKey;
using TensileLite::Client::initCPUSparseInput;
using TensileLite::Client::InitMode;
using TensileLite::Client::isMXProblem;
using TensileLite::Client::isMXTensor;
using TensileLite::Client::PruneSparseMode;
using TensileLite::Client::toHostNumericsScalarType;
using TensileLite::Client::initializeHostBufferWithHostNumerics;

// Shorthand for the production helper namespace under test (MX builds only).
#if HIPBLASLT_ENABLE_MXDATAGENERATOR
namespace dt = TensileLite::Client::HostNumerics::detail;
#endif
namespace
{
    constexpr DataInitializationKey fixedInitializationKey{0x12345678ULL, 0x01020304ULL};
    constexpr DataInitializationKey firstRandomInitializationKey{0x12345678ULL, 101};
    constexpr DataInitializationKey secondRandomInitializationKey{0x12345678ULL, 202};
    constexpr DataInitializationKey otherSeedInitializationKey{0x87654321ULL, 101};

    // -----------------------------------------------------------------------
    // Helper: build a ContractionProblemGemm with the requested A/B dtypes.
    // Mirrors tests/MXScalePadding_test.cpp::makeMXProblem so the geometry
    // matches what the real client produces, then enables MX scaling on each
    // side independently. mxBlock==0 means "do NOT call setMXScale*", so the
    // problem's mxBlockA() / mxBlockB() stays 0 and isMXTensor returns
    // false on that side. This is exactly the lever needed to drive every
    // branch of isMXProblem.
    // -----------------------------------------------------------------------
    ContractionProblemGemm makeProblem(rocisa::DataType aType,
                                       rocisa::DataType bType,
                                       int              mxBlockA,
                                       int              mxBlockB,
                                       size_t           M      = 128,
                                       size_t           N      = 128,
                                       size_t           K      = 256,
                                       size_t           batch  = 1,
                                       bool             transA = true,
                                       bool             transB = false)
    {
        auto problem = ContractionProblemGemm::GEMM_Strides(transA,
                                                            transB,
                                                            aType,
                                                            bType,
                                                            rocisa::DataType::BFloat16,
                                                            rocisa::DataType::BFloat16,
                                                            M,
                                                            N,
                                                            K,
                                                            batch,
                                                            transA ? K : M, // lda
                                                            transA ? K * M : M * K, // strideA
                                                            transB ? N : K, // ldb
                                                            transB ? N * K : K * N, // strideB
                                                            M,
                                                            M * N, // ldc, strideC
                                                            M,
                                                            M * N, // ldd, strideD
                                                            0.0); // beta
        if(mxBlockA > 0)
            problem.setMXScaleA(rocisa::DataType::E8, mxBlockA);
        if(mxBlockB > 0)
            problem.setMXScaleB(rocisa::DataType::E8, mxBlockB);
        return problem;
    }
} // namespace

TEST(HostNumericsDataInitialization, GeneratesStridedProblemDependentPatterns)
{
    TensorDescriptor   descriptor("t", rocisa::DataType::Float, {2, 3}, {1, 4});
    std::vector<float> values(descriptor.totalAllocatedElements(), -99.0f);

    initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                         InitMode::SerialIdx,
                                         values.data(),
                                         descriptor,
                                         fixedInitializationKey);
    EXPECT_EQ(values[0], 0);
    EXPECT_EQ(values[1], 1);
    EXPECT_EQ(values[4], 2);
    EXPECT_EQ(values[5], 3);
    EXPECT_EQ(values[8], 4);
    EXPECT_EQ(values[9], 5);
    EXPECT_EQ(values[2], -99);
    EXPECT_EQ(values[3], -99);

    TensorDescriptor   identityDescriptor("identity", rocisa::DataType::Float, {3, 4}, {1, 3});
    std::vector<float> identity(identityDescriptor.totalAllocatedElements(), -1.0f);
    initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                         InitMode::Identity,
                                         identity.data(),
                                         identityDescriptor,
                                         fixedInitializationKey);
    for(size_t column = 0; column < 4; ++column)
        for(size_t row = 0; row < 3; ++row)
            EXPECT_EQ(identity[row + column * 3], row == column ? 1.0f : 0.0f);

    TensorDescriptor halfDescriptor("half-raw-dimension", rocisa::DataType::Half, {2, 3}, {1, 4});
    std::vector<uint16_t> halfBits(halfDescriptor.totalAllocatedElements(), 0xffffU);
    initializeHostBufferWithHostNumerics(rocisa::DataType::Half,
                                         InitMode::SerialDim1,
                                         halfBits.data(),
                                         halfDescriptor,
                                         fixedInitializationKey);
    for(size_t column = 0; column < 3; ++column)
        for(size_t row = 0; row < 2; ++row)
            EXPECT_EQ(halfBits[row + column * 4], column);
    EXPECT_EQ(halfBits[2], 0xffffU);
    EXPECT_EQ(halfBits[3], 0xffffU);
}

TEST(HostNumericsDataInitialization, SizesDescriptorStorageForTheRequestedType)
{
    TensorDescriptor bfloat16Descriptor(
        "cross-type-bias", rocisa::DataType::BFloat16, {3, 1, 1}, {1, 3, 0});
    std::array<float, 3> floatValues{-1.0f, -1.0f, -1.0f};

    initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                         InitMode::Zero,
                                         floatValues.data(),
                                         bfloat16Descriptor,
                                         fixedInitializationKey);
    EXPECT_EQ(floatValues, (std::array<float, 3>{0.0f, 0.0f, 0.0f}));

    TensorDescriptor halfDescriptor(
        "cross-type-bias", rocisa::DataType::Half, {3, 1, 1}, {1, 3, 0});
    initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                         InitMode::Random,
                                         floatValues.data(),
                                         halfDescriptor,
                                         fixedInitializationKey);
}

TEST(HostNumericsDataInitialization, ThrowsForUnsupportedTypeOrRecipe)
{
    std::array<float, 1> value{};
    EXPECT_THROW(initializeHostBufferWithHostNumerics(rocisa::DataType::None,
                                                      InitMode::Zero,
                                                      value.data(),
                                                      value.size(),
                                                      fixedInitializationKey),
                 std::invalid_argument);
    EXPECT_THROW(initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                                      InitMode::UniformLowPrecision,
                                                      value.data(),
                                                      value.size(),
                                                      fixedInitializationKey),
                 std::invalid_argument);
}

TEST(HostNumericsDataInitialization, RejectsDescriptorStrideOutsidePtrdiff)
{
    if constexpr(std::numeric_limits<size_t>::digits <= std::numeric_limits<ptrdiff_t>::digits)
        GTEST_SKIP() << "size_t has no values outside ptrdiff_t range";

    const size_t oversizedStride
        = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) + 1;
    TensorDescriptor descriptor(
        "oversized-stride", rocisa::DataType::Float, {1, 1}, {1, oversizedStride});
    std::array<float, 1> value{};
    EXPECT_THROW(initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                                      InitMode::Zero,
                                                      value.data(),
                                                      descriptor,
                                                      fixedInitializationKey),
                 std::overflow_error);
}

TEST(HostNumericsDataInitialization, HandlesIndexedAndEncodedRandomModes)
{
    std::array<float, 4> values{-1, -1, -1, -1};
    initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                         InitMode::RandomNarrow,
                                         values.data(),
                                         values.size(),
                                         fixedInitializationKey);
    for(float value : values)
    {
        const uint32_t exponent = (std::bit_cast<uint32_t>(value) >> 23) & 0xffU;
        EXPECT_GE(exponent, 27U);
        EXPECT_LE(exponent, 127U);
    }

    initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                         InitMode::Two,
                                         values.data(),
                                         values.size(),
                                         fixedInitializationKey);
    EXPECT_EQ(values, (std::array<float, 4>{2, 2, 2, 2}));

    std::array<float, 4> randomFirst{};
    std::array<float, 4> randomSecond{};
    std::array<float, 4> randomFirstRepeat{};
    std::array<float, 4> randomOtherSeed{};
    initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                         InitMode::Random,
                                         randomFirst.data(),
                                         randomFirst.size(),
                                         firstRandomInitializationKey);
    initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                         InitMode::Random,
                                         randomSecond.data(),
                                         randomSecond.size(),
                                         secondRandomInitializationKey);
    initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                         InitMode::Random,
                                         randomFirstRepeat.data(),
                                         randomFirstRepeat.size(),
                                         firstRandomInitializationKey);
    initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                         InitMode::Random,
                                         randomOtherSeed.data(),
                                         randomOtherSeed.size(),
                                         otherSeedInitializationKey);
    EXPECT_EQ(randomFirst, randomFirstRepeat);
    EXPECT_NE(randomFirst, randomSecond);
    EXPECT_NE(randomFirst, randomOtherSeed);
    for(float value : randomFirst)
    {
        EXPECT_GE(value, -100);
        EXPECT_LE(value, 100);
    }

    std::array<uint8_t, 64> e8Values{};
    initializeHostBufferWithHostNumerics(rocisa::DataType::E8,
                                         InitMode::RandomNegPosLimited,
                                         e8Values.data(),
                                         e8Values.size(),
                                         fixedInitializationKey);

    std::array<TensileLite::E5M3, 64> unsignedScale{};
    initializeHostBufferWithHostNumerics(rocisa::DataType::E5M3,
                                         InitMode::Random,
                                         unsignedScale.data(),
                                         unsignedScale.size(),
                                         fixedInitializationKey);
    for(const TensileLite::E5M3 value : unsignedScale)
    {
        EXPECT_FALSE(value.is_nan());
        EXPECT_GE(static_cast<float>(value), 0);
        EXPECT_LE(static_cast<float>(value), 3);
    }

#ifndef _WIN32
#ifdef TENSILE_USE_FP4
    constexpr size_t                                  logicalFP4Elements = 65;
    std::array<uint8_t, (logicalFP4Elements + 1) / 2> packedFP4{};
    initializeHostBufferWithHostNumerics(rocisa::DataType::Float4,
                                         InitMode::RandomNarrow,
                                         packedFP4.data(),
                                         logicalFP4Elements,
                                         fixedInitializationKey);
    for(size_t index = 0; index < logicalFP4Elements; ++index)
    {
        const uint8_t byte = packedFP4[index / 2];
        const uint8_t raw  = index % 2 == 0 ? byte & 0xfU : byte >> 4;
        EXPECT_LE(raw, 14);
    }
#endif
#endif
}

TEST(HostNumericsDataInitialization, ComplexRandomValuesAreRepeatableAndIndependent)
{
    std::array<std::complex<float>, 32> first{};
    std::array<std::complex<float>, 32> repeat{};
    std::array<std::complex<float>, 32> other{};

    initializeHostBufferWithHostNumerics(rocisa::DataType::ComplexFloat,
                                         InitMode::Random,
                                         first.data(),
                                         first.size(),
                                         firstRandomInitializationKey);
    initializeHostBufferWithHostNumerics(rocisa::DataType::ComplexFloat,
                                         InitMode::Random,
                                         repeat.data(),
                                         repeat.size(),
                                         firstRandomInitializationKey);
    initializeHostBufferWithHostNumerics(rocisa::DataType::ComplexFloat,
                                         InitMode::Random,
                                         other.data(),
                                         other.size(),
                                         secondRandomInitializationKey);

    EXPECT_EQ(first, repeat);
    EXPECT_NE(first, other);
    EXPECT_TRUE(std::ranges::any_of(
        first, [](const std::complex<float>& value) { return value.real() != value.imag(); }));
    for(const std::complex<float>& value : first)
    {
        EXPECT_GE(value.real(), -3);
        EXPECT_LE(value.real(), 3);
        EXPECT_GE(value.imag(), -3);
        EXPECT_LE(value.imag(), 3);
    }
}

namespace
{
    template <typename T>
    void expectComponentInitializationBytes(rocisa::DataType dataType,
                                            InitMode         mode,
                                            const T&         expected)
    {
        constexpr size_t                 logicalElements = TensileLite::TypeInfo<T>::Packing;
        std::array<std::byte, sizeof(T)> observed{};
        initializeHostBufferWithHostNumerics(
            dataType, mode, observed.data(), logicalElements, fixedInitializationKey);
        EXPECT_EQ(std::memcmp(observed.data(), &expected, sizeof(T)), 0)
            << "dataType=" << dataType << " mode=" << mode;
    }
}

TEST(HostNumericsDataInitialization, TypeDerivedSpecialValuesMatchTensileEncoding)
{
    expectComponentInitializationBytes<int8_t>(
        rocisa::DataType::Int8, InitMode::Max, std::numeric_limits<int8_t>::max());
    expectComponentInitializationBytes<int32_t>(
        rocisa::DataType::Int32, InitMode::Max, std::numeric_limits<int32_t>::max());
    expectComponentInitializationBytes<int8_t>(
        rocisa::DataType::Int8, InitMode::BadInput, std::numeric_limits<int8_t>::max());
    expectComponentInitializationBytes<int8_t>(
        rocisa::DataType::Int8, InitMode::BadOutput, std::numeric_limits<int8_t>::min());
    expectComponentInitializationBytes<int32_t>(
        rocisa::DataType::Int32, InitMode::BadInput, std::numeric_limits<int32_t>::max());
    expectComponentInitializationBytes<int32_t>(
        rocisa::DataType::Int32, InitMode::BadOutput, std::numeric_limits<int32_t>::min());

    expectComponentInitializationBytes<TensileLite::BFloat16>(
        rocisa::DataType::BFloat16,
        InitMode::Max,
        TensileLite::BFloat16(std::numeric_limits<float>::max()));
#ifndef _WIN32
#ifdef TENSILE_USE_BF6
    expectComponentInitializationBytes<TensileLite::BFloat6x32>(
        rocisa::DataType::BFloat6, InitMode::Max, TensileLite::BFloat6x32(7.5f));
    expectComponentInitializationBytes<TensileLite::BFloat6x32>(
        rocisa::DataType::BFloat6, InitMode::DenormMin, TensileLite::BFloat6x32(0.125f));
    expectComponentInitializationBytes<TensileLite::BFloat6x32>(
        rocisa::DataType::BFloat6, InitMode::DenormMax, TensileLite::BFloat6x32(0.875f));
#endif
#endif

    expectComponentInitializationBytes<TensileLite::E8>(
        rocisa::DataType::E8, InitMode::Zero, TensileLite::E8(uint8_t{0}));
    expectComponentInitializationBytes<TensileLite::E8>(
        rocisa::DataType::E8, InitMode::One, TensileLite::E8(1.0f));
    expectComponentInitializationBytes<TensileLite::E8>(
        rocisa::DataType::E8, InitMode::Two, TensileLite::E8(2.0f));
    expectComponentInitializationBytes<TensileLite::E8>(
        rocisa::DataType::E8, InitMode::Max, TensileLite::E8(static_cast<uint8_t>(0xfe)));
    for(const InitMode mode : {InitMode::NaN, InitMode::BadInput, InitMode::BadOutput})
        expectComponentInitializationBytes<TensileLite::E8>(
            rocisa::DataType::E8, mode, TensileLite::E8(static_cast<uint8_t>(0xff)));
}

TEST(HostNumericsStructuredSparsity, TensileAdapterMatchesStandaloneComponent)
{
    auto problem = makeProblem(rocisa::DataType::Int8,
                               rocisa::DataType::Int8,
                               /*mxBlockA=*/0,
                               /*mxBlockB=*/0,
                               /*M=*/2,
                               /*N=*/2,
                               /*K=*/8,
                               /*batch=*/1,
                               /*transA=*/false,
                               /*transB=*/false);
    problem.setSparse(1, 0);

    const TensorDescriptor& denseDescriptor      = problem.a();
    const TensorDescriptor& compressedDescriptor = problem.compressed();
    const TensorDescriptor& metadataDescriptor   = problem.metadata();
    const size_t            sparseAxis           = problem.boundIndices()[0].a;

    std::vector<int8_t> original(denseDescriptor.totalAllocatedElements());
    for(size_t index = 0; index < original.size(); ++index)
        original[index] = static_cast<int8_t>(index + 1);

    std::vector<int8_t>  adapterPruned = original;
    std::vector<int8_t>  adapterCompressed(compressedDescriptor.totalAllocatedElements());
    std::vector<uint8_t> adapterMetadata(metadataDescriptor.totalAllocatedElements());
    initCPUSparseInput(PruneSparseMode::PruneXX00,
                       adapterPruned.data(),
                       adapterCompressed.data(),
                       adapterMetadata.data(),
                       denseDescriptor,
                       compressedDescriptor,
                       metadataDescriptor,
                       sparseAxis,
                       problem.metadataLayout());

    auto layout = [](const TensorDescriptor& descriptor) {
        return TensileLite::Client::hostNumericsLayout(descriptor);
    };
    using namespace roc::host_numerics;
    const ScalarType     scalarType = toHostNumericsScalarType(denseDescriptor.dataType());
    std::vector<int8_t>  componentPruned(original.size());
    std::vector<int8_t>  componentCompressed(adapterCompressed.size());
    std::vector<uint8_t> componentMetadata(adapterMetadata.size());
    const Shape          logicalMetadataShape{
        denseDescriptor.sizes()[0], denseDescriptor.sizes()[1] / 8, denseDescriptor.sizes()[2]};
    const Layout metadataLayout = layout(metadataDescriptor);
    const Layout logicalMetadataLayout(logicalMetadataShape,
                                       {metadataLayout.stride(1),
                                        metadataLayout.stride(0),
                                        metadataLayout.stride(2)});

    StructuredSparsityPattern pattern;
    pattern.axis           = sparseAxis;
    pattern.fixedPositions = {0, 1};
    Tensor componentPrunedTensor = Tensor::copyEncodedBackingStorage(
        scalarType, layout(denseDescriptor),
        std::as_writable_bytes(std::span<int8_t>(componentPruned)));
    Tensor componentCompressedTensor = Tensor::copyEncodedBackingStorage(
        scalarType,
        layout(compressedDescriptor),
        std::as_writable_bytes(std::span<int8_t>(componentCompressed)));
    Tensor componentMetadataTensor = Tensor::copyEncodedBackingStorage(
        ScalarType::UInt8, logicalMetadataLayout,
        std::as_writable_bytes(std::span<uint8_t>(componentMetadata)));
    StructuredSparsityRequest componentRequest(
        Tensor::copyEncodedBackingStorage(
            scalarType, layout(denseDescriptor), std::as_bytes(std::span<const int8_t>(original))),
        componentPrunedTensor,
        componentCompressedTensor,
        std::nullopt,
        componentMetadataTensor,
        pattern);
    applyStructuredSparsity(componentRequest);
    std::memcpy(componentPruned.data(),
                componentPrunedTensor.rawEncodedBackingStorage().data(),
                componentPrunedTensor.rawEncodedBackingStorage().size());
    std::memcpy(componentCompressed.data(),
                componentCompressedTensor.rawEncodedBackingStorage().data(),
                componentCompressedTensor.rawEncodedBackingStorage().size());
    std::memcpy(componentMetadata.data(),
                componentMetadataTensor.rawEncodedBackingStorage().data(),
                componentMetadataTensor.rawEncodedBackingStorage().size());

    EXPECT_EQ(componentPruned, adapterPruned);
    EXPECT_EQ(componentCompressed, adapterCompressed);
    EXPECT_EQ(componentMetadata, adapterMetadata);
}

TEST(HostNumericsStructuredSparsity, ValidationFailureDoesNotMutateCallerStorage)
{
    const TensorDescriptor dense("dense", rocisa::DataType::Int8, {2, 4}, {1, 2});
    const TensorDescriptor compressed("compressed", rocisa::DataType::Int8, {2, 2}, {1, 4});
    const TensorDescriptor metadata("metadata", rocisa::DataType::Int8, {2, 2}, {1, 4});

    std::vector<int8_t>  pruned(dense.totalAllocatedElements(), 11);
    std::vector<int8_t>  compressedOutput(compressed.totalAllocatedElements(), 22);
    std::vector<uint8_t> metadataOutput(metadata.totalAllocatedElements(), 33);
    const auto           originalPruned     = pruned;
    const auto           originalCompressed = compressedOutput;
    const auto           originalMetadata   = metadataOutput;

    ASSERT_GT(compressed.totalAllocatedElements(), compressed.totalLogicalElements());
    ASSERT_GT(metadata.totalAllocatedElements(), metadata.totalLogicalElements());
    EXPECT_THROW(initCPUSparseInput(PruneSparseMode::PruneXX00,
                                    pruned.data(),
                                    compressedOutput.data(),
                                    metadataOutput.data(),
                                    dense,
                                    compressed,
                                    metadata,
                                    dense.dimensions(),
                                    false),
                 std::out_of_range);

    EXPECT_EQ(pruned, originalPruned);
    EXPECT_EQ(compressedOutput, originalCompressed);
    EXPECT_EQ(metadataOutput, originalMetadata);
}

TEST(HostNumericsStructuredSparsity, TensileAdapterCoversModesLayoutsAndSparseSides)
{
    const std::array<PruneSparseMode, 7> modes{
        PruneSparseMode::PruneRandom,
        PruneSparseMode::PruneXX00,
        PruneSparseMode::PruneX0X0,
        PruneSparseMode::Prune0XX0,
        PruneSparseMode::PruneX00X,
        PruneSparseMode::Prune0X0X,
        PruneSparseMode::Prune00XX,
    };
    const std::array<std::array<size_t, 2>, 7> retainedPositionSets{{
        {0, 0},
        {0, 1},
        {0, 2},
        {1, 2},
        {0, 3},
        {1, 3},
        {2, 3},
    }};
    struct SparseCase
    {
        int  sparseSide;
        bool transA;
        bool transB;
    };
    const std::array<SparseCase, 4> sparseCases{{
        {1, false, false},
        {1, true, false},
        {2, false, false},
        {2, false, true},
    }};

    for(const SparseCase& sparseCase : sparseCases)
    {
        for(const int metadataLayout : {0, 1})
        {
            for(const PruneSparseMode mode : modes)
            {
                SCOPED_TRACE(::testing::Message()
                             << "sparseSide=" << sparseCase.sparseSide
                             << " transA=" << sparseCase.transA << " transB=" << sparseCase.transB
                             << " metadataLayout=" << metadataLayout
                             << " mode=" << static_cast<int>(mode));
                auto problem = makeProblem(rocisa::DataType::Int8,
                                           rocisa::DataType::Int8,
                                           0,
                                           0,
                                           8,
                                           8,
                                           16,
                                           2,
                                           sparseCase.transA,
                                           sparseCase.transB);
                problem.setSparse(sparseCase.sparseSide, metadataLayout);

                const TensorDescriptor& denseDescriptor
                    = sparseCase.sparseSide == 1 ? problem.a() : problem.b();
                const TensorDescriptor& compressedDescriptor = problem.compressed();
                const TensorDescriptor& metadataDescriptor   = problem.metadata();
                const size_t sparseAxis = sparseCase.sparseSide == 1 ? problem.boundIndices()[0].a
                                                                     : problem.boundIndices()[0].b;

                std::vector<int8_t> original(denseDescriptor.totalAllocatedElements());
                for(size_t index = 0; index < original.size(); ++index)
                    original[index] = static_cast<int8_t>(index % 127 + 1);
                std::vector<int8_t>  pruned = original;
                std::vector<int8_t>  compressed(compressedDescriptor.totalAllocatedElements(),
                                                static_cast<int8_t>(-101));
                std::vector<uint8_t> metadata(metadataDescriptor.totalAllocatedElements(), 0xff);

                initCPUSparseInput(mode,
                                   pruned.data(),
                                   compressed.data(),
                                   metadata.data(),
                                   denseDescriptor,
                                   compressedDescriptor,
                                   metadataDescriptor,
                                   sparseAxis,
                                   problem.metadataLayout());

                const size_t groupsPerSlice = denseDescriptor.sizes()[sparseAxis] / 4;
                const size_t sliceCount
                    = denseDescriptor.totalLogicalElements() / denseDescriptor.sizes()[sparseAxis];
                std::vector<size_t> denseCoordinates(denseDescriptor.dimensions(), 0);
                std::vector<size_t> compressedCoordinates(compressedDescriptor.dimensions(), 0);
                std::vector<size_t> metadataCoordinates(metadataDescriptor.dimensions(), 0);
                for(size_t slice = 0; slice < sliceCount; ++slice)
                {
                    TensileLite::CoordNumberedExclude(slice,
                                                      denseCoordinates.begin(),
                                                      denseCoordinates.end(),
                                                      denseDescriptor.sizes().begin(),
                                                      denseDescriptor.sizes().end(),
                                                      sparseAxis);
                    TensileLite::CoordNumberedExclude(slice,
                                                      metadataCoordinates.begin(),
                                                      metadataCoordinates.end(),
                                                      metadataDescriptor.sizes().begin(),
                                                      metadataDescriptor.sizes().end(),
                                                      problem.metadataLayout());
                    compressedCoordinates = denseCoordinates;
                    for(size_t group = 0; group < groupsPerSlice; ++group)
                    {
                        std::array<size_t, 2>        randomRetainedPositions{};
                        const std::array<size_t, 2>* retained
                            = &retainedPositionSets[static_cast<uint32_t>(mode)];
                        if(mode == PruneSparseMode::PruneRandom)
                        {
                            size_t retainedCount = 0;
                            for(size_t position = 0; position < 4; ++position)
                            {
                                denseCoordinates[sparseAxis] = group * 4 + position;
                                if(pruned[denseDescriptor.index(denseCoordinates)] != 0)
                                {
                                    ASSERT_LT(retainedCount, randomRetainedPositions.size());
                                    randomRetainedPositions[retainedCount++] = position;
                                }
                            }
                            ASSERT_EQ(retainedCount, randomRetainedPositions.size());
                            retained = &randomRetainedPositions;
                        }
                        const uint8_t expectedMetadata
                            = static_cast<uint8_t>((*retained)[0] | ((*retained)[1] << 2));

                        for(size_t position = 0; position < 4; ++position)
                        {
                            denseCoordinates[sparseAxis] = group * 4 + position;
                            const size_t denseIndex      = denseDescriptor.index(denseCoordinates);
                            const bool   isRetained
                                = position == (*retained)[0] || position == (*retained)[1];
                            EXPECT_EQ(pruned[denseIndex], isRetained ? original[denseIndex] : 0);
                        }
                        for(size_t retainedIndex = 0; retainedIndex < retained->size();
                            ++retainedIndex)
                        {
                            denseCoordinates[sparseAxis] = group * 4 + (*retained)[retainedIndex];
                            compressedCoordinates[sparseAxis] = group * 2 + retainedIndex;
                            EXPECT_EQ(compressed[compressedDescriptor.index(compressedCoordinates)],
                                      original[denseDescriptor.index(denseCoordinates)]);
                        }

                        metadataCoordinates[problem.metadataLayout()] = group / 2;
                        const size_t metadataIndex
                            = TensileLite::CoordFlattenIndex(metadataCoordinates.begin(),
                                                             metadataCoordinates.end(),
                                                             metadataDescriptor.sizes().begin(),
                                                             metadataDescriptor.sizes().end());
                        const uint8_t observedMetadata
                            = static_cast<uint8_t>(metadata[metadataIndex] >> ((group % 2) * 4));
                        EXPECT_EQ(observedMetadata & 0xfU, expectedMetadata);
                    }
                }
            }
        }
    }
}

// =============================================================================
//   Section 1 - TensileLite::Client::isMXTensor
//
//       bool isMXTensor(t, mxBlock) {
//           if(mxBlock == 0) return false;            // (a) short-circuit
//           return dt in {Float4, Float6, BFloat6, Float8, BFloat8}; // (b) dtype gate
//       }
// =============================================================================
struct TensorParam
{
    rocisa::DataType dtype;
    size_t           mxBlock;
    bool             expected;
    char const*      name;
};
class IsMXTensorTest : public ::testing::TestWithParam<TensorParam>
{
};
TEST_P(IsMXTensorTest, MatchesContract)
{
    auto const& p = GetParam();
    // 1x1 descriptor is enough; the helper only inspects .dataType().
    TensorDescriptor t("t", p.dtype, {1, 1}, {1, 1});
    EXPECT_EQ(isMXTensor(t, p.mxBlock), p.expected)
        << "case=" << p.name << " dtype=" << static_cast<int>(p.dtype) << " mxBlock=" << p.mxBlock;
}

INSTANTIATE_TEST_SUITE_P(MXFP4OrFP8Coverage,
                         IsMXTensorTest,
                         ::testing::Values(
                             // ----- (a) mxBlock==0 must short-circuit even for MX dtypes --------
                             TensorParam{rocisa::DataType::Float4, 0, false, "Float4_block0"},
                             TensorParam{rocisa::DataType::Float6, 0, false, "Float6_block0"},
                             TensorParam{rocisa::DataType::BFloat6, 0, false, "BFloat6_block0"},
                             TensorParam{rocisa::DataType::Float8, 0, false, "Float8_block0"},
                             TensorParam{rocisa::DataType::BFloat8, 0, false, "BFloat8_block0"},
                             // ----- (b) supported MX dtypes with mxBlock>0 -> true --------------
                             TensorParam{rocisa::DataType::Float4, 32, true, "Float4_block32"},
                             TensorParam{rocisa::DataType::Float6, 32, true, "Float6_block32"},
                             TensorParam{rocisa::DataType::BFloat6, 32, true, "BFloat6_block32"},
                             TensorParam{rocisa::DataType::Float8, 32, true, "Float8_block32"},
                             TensorParam{rocisa::DataType::BFloat8, 32, true, "BFloat8_block32"},
                             // ----- (b') unsupported dtypes with mxBlock>0 -> false -------------
                             TensorParam{rocisa::DataType::Float, 32, false, "Float_block32"},
                             TensorParam{rocisa::DataType::Half, 32, false, "Half_block32"},
                             TensorParam{rocisa::DataType::BFloat16, 32, false, "BFloat16_block32"},
                             TensorParam{rocisa::DataType::Int8, 32, false, "Int8_block32"},
                             TensorParam{rocisa::DataType::Int32, 32, false, "Int32_block32"},
                             // ----- mxBlock not equal to 32 (any positive value works) ----------
                             TensorParam{rocisa::DataType::Float8, 1, true, "Float8_block1"},
                             TensorParam{rocisa::DataType::BFloat8, 128, true, "BFloat8_block128"}),
                         [](::testing::TestParamInfo<TensorParam> const& info) {
                             return std::string(info.param.name);
                         });

// =============================================================================
//   Section 2 - TensileLite::Client::isMXProblem
//
//   Contract:
//       isMXProblem(P)
//         = isMXTensor(P.a, P.mxBlockA)
//            || isMXTensor(P.b, P.mxBlockB)
// =============================================================================
TEST(IsMXProblem, BothFP4)
{
    auto p = makeProblem(rocisa::DataType::Float4,
                         rocisa::DataType::Float4,
                         /*mxBlockA=*/32,
                         /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, BothFP6)
{
    auto p = makeProblem(rocisa::DataType::Float6,
                         rocisa::DataType::Float6,
                         /*mxBlockA=*/32,
                         /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, BothBFloat6)
{
    auto p = makeProblem(rocisa::DataType::BFloat6,
                         rocisa::DataType::BFloat6,
                         /*mxBlockA=*/32,
                         /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, BothFP8)
{
    auto p = makeProblem(rocisa::DataType::Float8,
                         rocisa::DataType::Float8,
                         /*mxBlockA=*/32,
                         /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, BothBFloat8)
{
    auto p = makeProblem(rocisa::DataType::BFloat8,
                         rocisa::DataType::BFloat8,
                         /*mxBlockA=*/32,
                         /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, MixedFP4AandFP8B)
{
    auto p = makeProblem(rocisa::DataType::Float4,
                         rocisa::DataType::Float8,
                         /*mxBlockA=*/32,
                         /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, MixedBFloat8AandFP4B)
{
    auto p = makeProblem(rocisa::DataType::BFloat8,
                         rocisa::DataType::Float4,
                         /*mxBlockA=*/32,
                         /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, OnlyA_isMX_BIsBF16)
{
    // First disjunct true, second disjunct short-circuits false (mxBlockB=0).
    auto p = makeProblem(rocisa::DataType::Float8,
                         rocisa::DataType::BFloat16,
                         /*mxBlockA=*/32,
                         /*mxBlockB=*/0);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, OnlyB_isMX_AIsBF16)
{
    auto p = makeProblem(rocisa::DataType::BFloat16,
                         rocisa::DataType::Float4,
                         /*mxBlockA=*/0,
                         /*mxBlockB=*/32);
    EXPECT_TRUE(isMXProblem(p));
}
TEST(IsMXProblem, NeitherIsMX)
{
    auto p = makeProblem(rocisa::DataType::BFloat16,
                         rocisa::DataType::BFloat16,
                         /*mxBlockA=*/0,
                         /*mxBlockB=*/0);
    EXPECT_FALSE(isMXProblem(p));
}
TEST(IsMXProblem, FloatABIsFalse)
{
    auto p = makeProblem(rocisa::DataType::Float,
                         rocisa::DataType::Float,
                         /*mxBlockA=*/0,
                         /*mxBlockB=*/0);
    EXPECT_FALSE(isMXProblem(p));
}

// =============================================================================
//   Section 3 — Byte-stride formula
//
//   For FP8 / BFloat8 the OCP standard packs one element per byte. The
//   DataTypeInfo for these dtypes therefore reports elementSize == 1, and the
//   formula must be the identity on strides[2]. These tests pin BOTH facts:
//   if anyone ever changes elementSize for FP8, or breaks multiplyElementSize,
//   the failure surfaces here instead of as a silent multi-batch FP8 bug.
// =============================================================================
TEST(InitializeMXDataForFP4OrFP8_BatchStrideFormula, FP8_OneBytePerElement)
{
    auto const info = DataTypeInfo::Get(rocisa::DataType::Float8);
    ASSERT_EQ(info.elementSize, 1u)
        << "OCP E4M3 must pack 1 byte per element; if this assertion fires the "
           "patch 3/3 batch-stride formula needs to be revisited.";
    constexpr size_t kStrideElems = 12345; // arbitrary, prime-ish
    size_t const     bytes
        = TensileLite::multiplyElementSize(kStrideElems, static_cast<float>(info.elementSize));
    EXPECT_EQ(bytes, kStrideElems);
}

TEST(InitializeMXDataForFP4OrFP8_BatchStrideFormula, BFloat8_OneBytePerElement)
{
    auto const info = DataTypeInfo::Get(rocisa::DataType::BFloat8);
    ASSERT_EQ(info.elementSize, 1u) << "OCP E5M2 must pack 1 byte per element.";
    constexpr size_t kStrideElems = 1u << 20; // 1 Mi elements
    size_t const     bytes
        = TensileLite::multiplyElementSize(kStrideElems, static_cast<float>(info.elementSize));
    EXPECT_EQ(bytes, kStrideElems);
}

// =============================================================================
//   Section 4 — direct calls into TensileLite::Client::detail (MX builds only)
// =============================================================================
#if HIPBLASLT_ENABLE_MXDATAGENERATOR
TEST(TensileMxGenerationTranslation, MapsTypesAndInitializationPolicy)
{
    using namespace roc::host_numerics;

    const MxGenerationProblem problem = dt::makeMxGenerationProblem(rocisa::DataType::Float4,
                                                                    rocisa::DataType::E5M3,
                                                                    Shape{8, 4},
                                                                    8,
                                                                    0,
                                                                    4,
                                                                    InitMode::Random,
                                                                    InitMode::One,
                                                                    17);
    EXPECT_EQ(problem.dataType, ScalarType::Float4E2M1);
    EXPECT_EQ(problem.scaleType, ScalarType::E5M3);
    EXPECT_EQ(problem.scale, MxScaleGenerationMode::One);
    EXPECT_EQ(problem.data.recipe().seed(), 17);
}

TEST(TensileMxGenerationTranslation, MapsArchitectureToPhysicalScaleLayout)
{
    using roc::host_numerics::amd_gpu_layout::MxScaleStorageLayout;

    EXPECT_EQ(dt::mxScaleStorageLayoutForArchName("gfx950"), MxScaleStorageLayout::Gfx950);
    EXPECT_EQ(dt::mxScaleStorageLayoutForArchName("gfx950:sramecc+:xnack-"),
              MxScaleStorageLayout::Gfx950);
    EXPECT_EQ(dt::mxScaleStorageLayoutForArchName("gfx1250"), MxScaleStorageLayout::Gfx1250);
    EXPECT_EQ(dt::mxScaleStorageLayoutForArchName("gfx942"), MxScaleStorageLayout::Natural);
}

TEST(TensileMxGenerationTranslation, RejectsUnsupportedTypesAndModes)
{
    using namespace roc::host_numerics;

    EXPECT_THROW(generateMx(dt::makeMxGenerationProblem(rocisa::DataType::Float,
                                                        rocisa::DataType::E8,
                                                        Shape{8, 4},
                                                        8,
                                                        0,
                                                        4,
                                                        InitMode::Random,
                                                        InitMode::One,
                                                        17)),
                 std::invalid_argument);
    EXPECT_THROW(dt::makeMxGenerationProblem(rocisa::DataType::Float4,
                                             rocisa::DataType::E8,
                                             Shape{8, 4},
                                             8,
                                             0,
                                             4,
                                             InitMode::Free,
                                             InitMode::One,
                                             17),
                 std::invalid_argument);
}

#endif // HIPBLASLT_ENABLE_MXDATAGENERATOR
