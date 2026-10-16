// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ClientProblemFactory.hpp"
#include "DataInitialization.hpp" // isMXTensor / Problem
#include "ProgramOptions.hpp"
#include "ReferenceValidator.hpp"
#include "ResultReporter.hpp"
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/PerformanceMetricTypes.hpp>
#include <Tensile/TensorDescriptor.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipUtils.hpp>
#include <TensileLite/Client/hostnumerics/DataInitializationHelpers.hpp>
#include <TensileLite/Client/hostnumerics/HostNumericsBridge.hpp>
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
#include <hip/hip_runtime.h>
#include <limits>
#include <numeric>
#include <roc/hostnumerics/validation.hpp>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using TensileLite::ContractionProblemGemm;
using TensileLite::ContractionSolution;
using TensileLite::DataTypeInfo;
using TensileLite::KernelLanguage;
using TensileLite::PerformanceMetric;
using TensileLite::TensorDescriptor;
using TensileLite::Client::DataInitialization;
using TensileLite::Client::DataInitializationKey;
using TensileLite::Client::initCPUSparseInput;
using TensileLite::Client::initializeHostBufferWithHostNumerics;
using TensileLite::Client::InitMode;
using TensileLite::Client::isMXProblem;
using TensileLite::Client::isMXTensor;
using TensileLite::Client::PruneSparseMode;
using TensileLite::Client::toHostNumericsScalarType;

// Shorthand for the production helper namespace under test.
namespace dt = TensileLite::Client::hostnumerics::detail;
namespace
{
    template <typename T>
    void setOption(TensileLite::Client::po::variables_map& args, std::string const& name, T value)
    {
        args[name].value() = std::move(value);
    }

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
                                       size_t           M            = 128,
                                       size_t           N            = 128,
                                       size_t           K            = 256,
                                       size_t           batch        = 1,
                                       bool             transA       = true,
                                       bool             transB       = false,
                                       int              mxBlockFreeA = 1,
                                       int              mxBlockFreeB = 1)
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
            problem.setMXScaleA(rocisa::DataType::E8, mxBlockA, {}, true, mxBlockFreeA);
        if(mxBlockB > 0)
            problem.setMXScaleB(rocisa::DataType::E8, mxBlockB, {}, true, mxBlockFreeB);
        return problem;
    }

    TensileLite::Client::po::variables_map makeMinimalClientArgs(int mxScaleFormat)
    {
        using namespace TensileLite::Client;

        po::variables_map args;
        setOption(args, "problem-identifier", std::string("Contraction_l_Ailk_Bljk_Cijk_Dijk"));
        setOption(args, "problem-size", std::vector<std::vector<size_t>>{{128, 128, 1, 512}});
        setOption(args, "strided-batched", true);
        setOption(args, "batch-mode", 0);
        setOption(args, "grouped-gemm", false);
        setOption(args, "sparse", 0);
        setOption(args, "high-precision-accumulate", false);
        setOption(args, "kernel-language", KernelLanguage::Any);
        setOption(args, "performance-metric", PerformanceMetric::DeviceEfficiency);
        setOption(args, "deterministic-mode", false);
        setOption(args, "c-equal-d", false);
        setOption(args, "type", rocisa::DataType::Float8);
        setOption(args, "a-type", rocisa::DataType::Float8);
        setOption(args, "b-type", rocisa::DataType::Float8);
        setOption(args, "c-type", rocisa::DataType::Float);
        setOption(args, "d-type", rocisa::DataType::Float);
        setOption(args, "alpha-type", rocisa::DataType::Float);
        setOption(args, "beta-type", rocisa::DataType::Float);
        setOption(args, "compute-input-type-A", rocisa::DataType::Float8);
        setOption(args, "compute-input-type-B", rocisa::DataType::Float8);
        setOption(args, "f32-xdl-math-op", rocisa::DataType::Float);
        setOption(args, "activation-compute-type", rocisa::DataType::Float);
        // Deliberately 0 here (not 32): these only control whether
        // ClientProblemFactory calls setMXScaleA/B on the *internal dummy*
        // problem it builds for sizing purposes (see ClientProblemFactory.cpp),
        // which is unrelated to the `problem` object this test passes directly
        // into referenceNeedsPerSolutionRecompute(). When non-zero here, the
        // DataInitialization ctor's storage-geometry computation for the
        // swizzled MX-scale tensor indexes problem.freeIndicesA()[0] /
        // freeIndicesB()[0] on that dummy problem without a bounds check; for
        // the minimal synthetic problem-size/identifier used here those
        // indices can be empty, which segfaults (only reachable on real
        // hardware, so it stays hidden behind GTEST_SKIP() on any other arch).
        // m_mxScaleFormat (checked by referenceNeedsPerSolutionRecompute) is
        // read directly from "mx-scale-format" below, independent of these.
        setOption(args, "mx-a-block", 0);
        setOption(args, "mx-b-block", 0);
        setOption(args, "mx-a-type", rocisa::DataType::E8);
        setOption(args, "mx-b-type", rocisa::DataType::E8);
        setOption(args, "mx-scale-format", mxScaleFormat);
        setOption(args, "fused-gemm-a2a", false);
        setOption(args, "metadata-layout", 0);
        setOption(args, "a-ops", TensileLite::TensorOps{});
        setOption(args, "b-ops", TensileLite::TensorOps{});
        setOption(args, "c-ops", TensileLite::TensorOps{});
        setOption(args, "d-ops", TensileLite::TensorOps{});
        setOption(args, "use-gradient", false);
        setOption(args, "output-amaxD", false);
        setOption(args, "use-scaleAB", std::string());
        setOption(args, "use-scaleCD", false);
        // ClientProblemFactory::m_useBias / m_biasSrc are plain `int` members
        // with NO default member initializer (see ClientProblemFactory.hpp);
        // they're only assigned when args.count("use-bias") /
        // args.count("bias-source") is true. Without these two options,
        // m_useBias holds whatever garbage was left on the stack/heap when
        // ClientProblemFactory was constructed. If that garbage is non-zero,
        // ContractionProblemGemm::setBias() takes a branch that indexes
        // m_tensors[m_biasSrc].sizes()[batchIdx] with batchIdx left at its
        // default of 2 for this non-batched dummy problem (m_batchIndices is
        // empty) -- an out-of-bounds read that segfaults. This is why the
        // crash only showed up when the test ran in isolation (a fresh
        // process, as CI's per-test ctest invocation does): the leftover
        // stack/heap contents differ from running the whole 733-test binary
        // in one process, where prior tests happened to leave zeroed memory
        // behind and masked the bug. Setting these explicitly makes the
        // behavior deterministic regardless of process/memory history.
        setOption(args, "use-bias", 0);
        setOption(args, "bias-source", static_cast<int>(ContractionProblemGemm::TENSOR::D));
        setOption(args, "use-scaleAlphaVec", 0);
        setOption(args, "device-idx", 0);
        setOption(args, "num-elements-to-validate", 0);
        setOption(args, "pristine-on-gpu", true);
        setOption(args, "prune-mode", TensileLite::Client::PruneSparseMode::PruneRandom);
        setOption(args, "rotating-buffer-size", 0);
        setOption(args, "rotating-buffer-mode", 0);
        setOption(args, "bounds-check", TensileLite::Client::BoundsCheckMode::Disable);
        return args;
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
                 std::runtime_error);
    EXPECT_THROW(initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                                      InitMode::UniformLowPrecision,
                                                      value.data(),
                                                      value.size(),
                                                      fixedInitializationKey),
                 std::runtime_error);
}

TEST(HostNumericsDataInitialization, RejectsDescriptorStrideOutsidePtrdiff)
{
    if constexpr(std::numeric_limits<size_t>::digits <= std::numeric_limits<ptrdiff_t>::digits)
        GTEST_SKIP() << "size_t has no values outside ptrdiff_t range";

    const size_t oversizedStride = static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()) + 1;
    TensorDescriptor descriptor(
        "oversized-stride", rocisa::DataType::Float, {1, 1}, {1, oversizedStride});
    std::array<float, 1> value{};
    EXPECT_THROW(initializeHostBufferWithHostNumerics(rocisa::DataType::Float,
                                                      InitMode::Zero,
                                                      value.data(),
                                                      descriptor,
                                                      fixedInitializationKey),
                 std::runtime_error);
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
    constexpr size_t                                  logicalFP4Elements = 1024;
    std::array<uint8_t, (logicalFP4Elements + 1) / 2> packedFP4{};
    initializeHostBufferWithHostNumerics(rocisa::DataType::Float4,
                                         InitMode::RandomNarrow,
                                         packedFP4.data(),
                                         logicalFP4Elements,
                                         fixedInitializationKey);
    bool observedEncodingFifteen = false;
    for(size_t index = 0; index < logicalFP4Elements; ++index)
    {
        const uint8_t byte = packedFP4[index / 2];
        const uint8_t raw  = index % 2 == 0 ? byte & 0xfU : byte >> 4;
        EXPECT_LE(raw, 15);
        observedEncodingFifteen |= raw == 15;
    }
    EXPECT_TRUE(observedEncodingFifteen);
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
    expectComponentInitializationBytes<TensileLite::BFloat6x32>(
        rocisa::DataType::BFloat6, InitMode::Max, TensileLite::BFloat6x32(7.5f));
    expectComponentInitializationBytes<TensileLite::BFloat6x32>(
        rocisa::DataType::BFloat6, InitMode::DenormMin, TensileLite::BFloat6x32(0.125f));
    expectComponentInitializationBytes<TensileLite::BFloat6x32>(
        rocisa::DataType::BFloat6, InitMode::DenormMax, TensileLite::BFloat6x32(0.875f));
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
        return TensileLite::Client::hostnumericsLayout(descriptor);
    };
    using namespace roc::hostnumerics;
    const ScalarType     scalarType = toHostNumericsScalarType(denseDescriptor.dataType());
    std::vector<int8_t>  componentPruned(original.size());
    std::vector<int8_t>  componentCompressed(adapterCompressed.size());
    std::vector<uint8_t> componentMetadata(adapterMetadata.size());
    const Shape          logicalMetadataShape{
        denseDescriptor.sizes()[0], denseDescriptor.sizes()[1] / 8, denseDescriptor.sizes()[2]};
    const Layout metadataLayout = layout(metadataDescriptor);
    const Layout logicalMetadataLayout(
        logicalMetadataShape,
        {metadataLayout.stride(1), metadataLayout.stride(0), metadataLayout.stride(2)});

    StructuredSparsityPattern pattern;
    pattern.axis                 = sparseAxis;
    pattern.fixedPositions       = {0, 1};
    Tensor componentPrunedTensor = Tensor::copyEncodedBackingStorage(
        scalarType,
        layout(denseDescriptor),
        std::as_writable_bytes(std::span<int8_t>(componentPruned)));
    Tensor componentCompressedTensor = Tensor::copyEncodedBackingStorage(
        scalarType,
        layout(compressedDescriptor),
        std::as_writable_bytes(std::span<int8_t>(componentCompressed)));
    Tensor componentMetadataTensor = Tensor::copyEncodedBackingStorage(
        ScalarType::UInt8,
        logicalMetadataLayout,
        std::as_writable_bytes(std::span<uint8_t>(componentMetadata)));
    applyStructuredSparsityInto(
        Tensor::copyEncodedBackingStorage(
            scalarType, layout(denseDescriptor), std::as_bytes(std::span<const int8_t>(original))),
        {.pruned            = componentPrunedTensor,
         .compressed        = componentCompressedTensor,
         .twoOfFourMetadata = componentMetadataTensor},
        pattern);
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
                 std::runtime_error);

    EXPECT_EQ(pruned, originalPruned);
    EXPECT_EQ(compressedOutput, originalCompressed);
    EXPECT_EQ(metadataOutput, originalMetadata);
}

TEST(HostNumericsStructuredSparsity, ZeroExtentIsANoOp)
{
    const TensorDescriptor dense("dense", rocisa::DataType::Int8, {0, 4}, {1, 1});
    const TensorDescriptor compressed("compressed", rocisa::DataType::Int8, {0, 4}, {1, 1});
    const TensorDescriptor metadata("metadata", rocisa::DataType::Int8, {0, 4}, {1, 1});

    EXPECT_NO_THROW(initCPUSparseInput(PruneSparseMode::PruneXX00,
                                       nullptr,
                                       nullptr,
                                       nullptr,
                                       dense,
                                       compressed,
                                       metadata,
                                       0,
                                       false));
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
TEST(IsMXProblem, BothFP8Block128)
{
    // 1x128 block scaling: mxBlock only has to be positive, so widening the
    // block must not change MX-problem detection or the scale dtypes.
    auto p = makeProblem(rocisa::DataType::Float8,
                         rocisa::DataType::Float8,
                         /*mxBlockA=*/128,
                         /*mxBlockB=*/128);
    EXPECT_TRUE(isMXProblem(p));
    EXPECT_EQ(p.mxBlockA(), 128);
    EXPECT_EQ(p.mxBlockB(), 128);
}
TEST(IsMXProblem, BothFP8FreeTile128)
{
    // 128x128 block scaling. isMXProblem keys off mxBlock{A,B} only, so adding a
    // free-dimension extent must not change detection either way.
    auto p = makeProblem(rocisa::DataType::Float8,
                         rocisa::DataType::Float8,
                         /*mxBlockA=*/128,
                         /*mxBlockB=*/128,
                         /*M=*/128,
                         /*N=*/128,
                         /*K=*/256,
                         /*batch=*/1,
                         /*transA=*/true,
                         /*transB=*/false,
                         /*mxBlockFreeA=*/128,
                         /*mxBlockFreeB=*/128);
    EXPECT_TRUE(isMXProblem(p));
    EXPECT_EQ(p.mxBlockA(), 128);
    EXPECT_EQ(p.mxBlockB(), 128);
    EXPECT_EQ(p.mxBlockFreeA(), 128);
    EXPECT_EQ(p.mxBlockFreeB(), 128);
}
TEST(IsMXProblem, FreeTileDefaultsToOne)
{
    // Every problem built before MXBlockFree* existed must keep reporting 1,
    // which is the divide-by-nothing 1xmxBlock layout.
    auto p = makeProblem(rocisa::DataType::Float8,
                         rocisa::DataType::Float8,
                         /*mxBlockA=*/128,
                         /*mxBlockB=*/128);
    EXPECT_EQ(p.mxBlockFreeA(), 1);
    EXPECT_EQ(p.mxBlockFreeB(), 1);
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

TEST(DataInitializationReferenceRecompute, Gfx950RequiresPerSolutionRecomputeForSwizzledMX)
{
    hipDeviceProp_t prop{};
    ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
    if(std::string(prop.gcnArchName).find("gfx950") == std::string::npos)
        GTEST_SKIP() << "gfx950-specific recompute behavior is only observable on gfx950";

    auto                                      problem = makeProblem(rocisa::DataType::Float8,
                               rocisa::DataType::Float8,
                               /*mxBlockA=*/32,
                               /*mxBlockB=*/32);
    auto                                      args    = makeMinimalClientArgs(/*mxScaleFormat=*/1);
    TensileLite::Client::ClientProblemFactory factory(args);
    TensileLite::Client::DataInitialization   dataInit(args, factory);

    ContractionSolution noSwizzleSolution;
    noSwizzleSolution.problemType.mxScaleFormat = 0;
    ContractionSolution swizzleSolution;
    swizzleSolution.problemType.mxScaleFormat = 1;

    EXPECT_FALSE(dataInit.referenceNeedsPerSolutionRecompute(problem, &noSwizzleSolution));
    EXPECT_TRUE(dataInit.referenceNeedsPerSolutionRecompute(problem, &swizzleSolution));
}

TEST(DataInitializationReferenceRecompute, Gfx1250ReusesPreswizzledMXAcrossSolutions)
{
    hipDeviceProp_t prop{};
    ASSERT_EQ(hipGetDeviceProperties(&prop, 0), hipSuccess);
    if(std::string(prop.gcnArchName).find("gfx1250") == std::string::npos)
        GTEST_SKIP() << "gfx1250-specific reuse behavior is only observable on gfx1250";

    auto                                      problem = makeProblem(rocisa::DataType::Float8,
                               rocisa::DataType::Float8,
                               /*mxBlockA=*/32,
                               /*mxBlockB=*/32);
    auto                                      args    = makeMinimalClientArgs(/*mxScaleFormat=*/1);
    TensileLite::Client::ClientProblemFactory factory(args);
    TensileLite::Client::DataInitialization   dataInit(args, factory);

    ContractionSolution swizzleSolution;
    swizzleSolution.problemType.mxScaleFormat = 1;

    EXPECT_FALSE(dataInit.referenceNeedsPerSolutionRecompute(problem, &swizzleSolution));
}

// =============================================================================
//   Section 4 — direct calls into TensileLite::Client::detail
// =============================================================================
TEST(TensileMxGenerationTranslation, MapsTypesAndInitializationPolicy)
{
    using namespace roc::hostnumerics;

    const MxTensor generated = dt::generateMxData(rocisa::DataType::Float4,
                                                  rocisa::DataType::E5M3,
                                                  Shape{8, 4},
                                                  8,
                                                  0,
                                                  4,
                                                  InitMode::Random,
                                                  InitMode::One,
                                                  17);
    EXPECT_EQ(generated.data.type(), ScalarType::Float4E2M1);
    EXPECT_EQ(generated.scales.type(), ScalarType::E5M3);
}

TEST(TensileMxGenerationTranslation, PreservesTheFullCallerSeed)
{
    using namespace roc::hostnumerics;

    const auto generateWithSeed = [](uint64_t seed) {
        return dt::generateMxData(rocisa::DataType::Float4,
                                  rocisa::DataType::E8,
                                  Shape{32, 4},
                                  32,
                                  0,
                                  32,
                                  InitMode::Random,
                                  InitMode::One,
                                  seed);
    };
    const MxTensor first  = generateWithSeed(17);
    const MxTensor replay = generateWithSeed(17);
    const MxTensor other  = generateWithSeed(17 + (uint64_t{1} << 40));

    EXPECT_TRUE(std::ranges::equal(first.data.rawEncodedBackingStorage(),
                                   replay.data.rawEncodedBackingStorage()));
    EXPECT_FALSE(std::ranges::equal(first.data.rawEncodedBackingStorage(),
                                    other.data.rawEncodedBackingStorage()));
}

TEST(TensileMxGenerationTranslation, MapsArchitectureToPhysicalScaleLayout)
{
    using roc::hostnumerics::amd_gpu_layout::MxScaleStorageLayout;

    EXPECT_EQ(dt::mxScaleStorageLayoutForArchName("gfx950"), MxScaleStorageLayout::Gfx950);
    EXPECT_EQ(dt::mxScaleStorageLayoutForArchName("gfx950:sramecc+:xnack-"),
              MxScaleStorageLayout::Gfx950);
    EXPECT_EQ(dt::mxScaleStorageLayoutForArchName("gfx1250"), MxScaleStorageLayout::Gfx1250);
    EXPECT_EQ(dt::mxScaleStorageLayoutForArchName("gfx942"), MxScaleStorageLayout::Natural);
}

// One scale must cover a whole MX block of K. Every element of a block shares its scale
// index, and its reference value divided by that scale lies on the FP4 E2M1 grid.
class TensileMxBlockScaleGranularity : public ::testing::TestWithParam<size_t>
{
};

TEST_P(TensileMxBlockScaleGranularity, BlockSharesOneScale)
{
    using namespace roc::hostnumerics;

    const size_t   mxBlock   = GetParam();
    const size_t   K         = 512;
    const size_t   M         = 64;
    const MxTensor generated = dt::generateMxData(rocisa::DataType::Float4,
                                                  rocisa::DataType::E8,
                                                  Shape{K, M},
                                                  K,
                                                  0,
                                                  mxBlock,
                                                  InitMode::Random,
                                                  InitMode::Random,
                                                  29,
                                                  false);
    ASSERT_EQ(generated.scales.elementCount(), (K / mxBlock) * M);

    const std::array<double, 8> grid{0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};
    const std::array<size_t, 2> order{0, 1};
    for(size_t m = 0; m < M; ++m)
        for(size_t k = 0; k < K; ++k)
        {
            const auto index      = generated.scaleIndices.loadAs<uint32_t>({k, m});
            const auto blockStart = generated.scaleIndices.loadAs<uint32_t>({k - k % mxBlock, m});
            ASSERT_EQ(index, blockStart) << "k=" << k << " m=" << m;
            const std::array<size_t, 1> selected{index};
            const double                scale
                = generated.scales.copySelectedElements(selected, order).loadAs<double>({0});
            ASSERT_GT(scale, 0.0);
            const double quotient = std::abs(generated.reference.loadAs<double>({k, m})) / scale;
            EXPECT_TRUE(std::ranges::any_of(
                grid, [&](double value) { return std::abs(quotient - value) <= 1e-6; }))
                << quotient << " at k=" << k << " m=" << m;
        }
}

INSTANTIATE_TEST_SUITE_P(MxBlockSizes,
                         TensileMxBlockScaleGranularity,
                         ::testing::Values(size_t{32}, size_t{128}));

namespace
{
    roc::hostnumerics::Tensor mxScales(roc::hostnumerics::ScalarType type,
                                       roc::hostnumerics::Shape      shape,
                                       std::vector<uint8_t>          bytes)
    {
        return roc::hostnumerics::Tensor::copyEncodedBackingStorage(
            type,
            roc::hostnumerics::Layout::contiguousLastDimensionFastest(std::move(shape)),
            std::as_bytes(std::span<const uint8_t>(bytes)));
    }

    std::vector<uint8_t> mxScaleBytes(const roc::hostnumerics::Tensor& scales)
    {
        std::vector<size_t> all(scales.elementCount());
        std::iota(all.begin(), all.end(), size_t{0});
        const std::array<size_t, 2> order{0, 1};
        const auto                  flat
            = scales.copySelectedElements(all, std::span(order).first(scales.shape().rank()));
        std::vector<uint8_t> result;
        for(const std::byte byte : flat.rawEncodedBackingStorage())
            result.push_back(std::to_integer<uint8_t>(byte));
        return result;
    }

    constexpr auto kE8 = roc::hostnumerics::ScalarType::E8M0Zero;
}

// A K-fast operand's scales are [free, K block]; a 2-row tile keeps each column's maximum.
TEST(TileMxScalesAlongFreeDimension, KFastTakesGroupMaximumPerKBlock)
{
    const auto tiled = dt::tileMxScalesAlongFreeDimension(
        mxScales(kE8, {4, 2}, {10, 1, 3, 20, 7, 7, 7, 8}), 0, 2);
    EXPECT_EQ(tiled.shape(), (roc::hostnumerics::Shape{2, 2}));
    EXPECT_EQ(mxScaleBytes(tiled), (std::vector<uint8_t>{10, 20, 7, 8}));
}

TEST(TileMxScalesAlongFreeDimension, KSlowTakesGroupMaximumPerKBlock)
{
    const auto tiled = dt::tileMxScalesAlongFreeDimension(
        mxScales(kE8, {2, 4}, {10, 3, 7, 7, 1, 20, 7, 8}), 1, 2);
    EXPECT_EQ(tiled.shape(), (roc::hostnumerics::Shape{2, 2}));
    EXPECT_EQ(mxScaleBytes(tiled), (std::vector<uint8_t>{10, 7, 20, 8}));
}

TEST(TileMxScalesAlongFreeDimension, PartialLastTileStopsAtTheFreeDimension)
{
    const auto tiled = dt::tileMxScalesAlongFreeDimension(mxScales(kE8, {3, 1}, {5, 9, 4}), 0, 2);
    EXPECT_EQ(mxScaleBytes(tiled), (std::vector<uint8_t>{9, 4}));
}

// The maximum keeps NaN, whose 0xFF byte orders above every finite E8M0 scale.
TEST(TileMxScalesAlongFreeDimension, NaNDominatesItsTile)
{
    const auto tiled
        = dt::tileMxScalesAlongFreeDimension(mxScales(kE8, {2, 1}, {0x7F, 0xFF}), 0, 2);
    EXPECT_EQ(mxScaleBytes(tiled), (std::vector<uint8_t>{0xFF}));
}

// Without an exponent-ordered encoding, each tile keeps its first scale.
TEST(TileMxScalesAlongFreeDimension, NonExponentScaleKeepsFirstOfEachTile)
{
    const auto tiled = dt::tileMxScalesAlongFreeDimension(
        mxScales(roc::hostnumerics::ScalarType::E4M3, {4, 1}, {5, 9, 4, 1}), 0, 2);
    EXPECT_EQ(mxScaleBytes(tiled), (std::vector<uint8_t>{5, 4}));
}

TEST(TileMxScalesAlongFreeDimension, TileExtentOneKeepsEveryScale)
{
    const auto tiled
        = dt::tileMxScalesAlongFreeDimension(mxScales(kE8, {2, 2}, {5, 9, 4, 1}), 0, 1);
    EXPECT_EQ(mxScaleBytes(tiled), (std::vector<uint8_t>{5, 9, 4, 1}));
}

// The offset is (tile + 2 * K block) % 5 - 2, so neighbouring tiles always differ.
TEST(VaryMxScaleTiles, KFastNeighbouringTilesDiffer)
{
    const auto varied = mxScaleBytes(
        dt::varyMxScaleTiles(mxScales(kE8, {4, 2}, std::vector<uint8_t>(8, 0x7F)), 0));
    EXPECT_EQ(varied, (std::vector<uint8_t>{0x7D, 0x7F, 0x7E, 0x80, 0x7F, 0x81, 0x80, 0x7D}));
}

TEST(VaryMxScaleTiles, KSlowMatchesKFastPerTile)
{
    const auto fast = mxScaleBytes(
        dt::varyMxScaleTiles(mxScales(kE8, {3, 2}, std::vector<uint8_t>(6, 0x7F)), 0));
    const auto slow = mxScaleBytes(
        dt::varyMxScaleTiles(mxScales(kE8, {2, 3}, std::vector<uint8_t>(6, 0x7F)), 1));
    for(size_t tile = 0; tile < 3; ++tile)
        for(size_t block = 0; block < 2; ++block)
            EXPECT_EQ(fast[tile * 2 + block], slow[block * 3 + tile]);
}

TEST(VaryMxScaleTiles, KeepsZeroAndNaNAndClampsToFiniteRange)
{
    const auto varied
        = mxScaleBytes(dt::varyMxScaleTiles(mxScales(kE8, {4, 1}, {0x00, 0xFF, 0x01, 0xFE}), 0));
    EXPECT_EQ(varied, (std::vector<uint8_t>{0x00, 0xFF, 0x01, 0xFE}));
}

TEST(TensileMxGenerationTranslation, RejectsUnsupportedTypesAndModes)
{
    using namespace roc::hostnumerics;

    EXPECT_THROW(dt::generateMxData(rocisa::DataType::Float,
                                    rocisa::DataType::E8,
                                    Shape{8, 4},
                                    8,
                                    0,
                                    4,
                                    InitMode::Random,
                                    InitMode::One,
                                    17),
                 std::runtime_error);
    EXPECT_THROW(dt::generateMxData(rocisa::DataType::Float4,
                                    rocisa::DataType::E8,
                                    Shape{8, 4},
                                    8,
                                    0,
                                    4,
                                    InitMode::Free,
                                    InitMode::One,
                                    17),
                 std::runtime_error);
}

TEST(DataInitializationFreivalds, RestoresOutputAndChecksRequestedGuards)
{
    using namespace TensileLite::Client;
    class Reporter : public ResultReporter
    {
    public:
        std::vector<std::string> verdicts;
        void reportValue_string(const std::string& key, const std::string& value) override
        {
            if(key == ResultKey::Validation)
                verdicts.push_back(value);
        }
        void reportValue_uint(const std::string&, uint64_t) override {}
        void reportValue_int(const std::string&, int64_t) override {}
        void reportValue_double(const std::string&, double) override {}
        void reportValue_sizes(const std::string&, const std::vector<size_t>&) override {}
        void reportValue_vecOfSizes(const std::string&,
                                    const std::vector<std::vector<size_t>>&) override
        {
        }
        void finalizeReport() override {}
    };
    auto args = makeMinimalClientArgs(0);
    setOption(args, "problem-size", std::vector<std::vector<size_t>>{{2, 3, 2, 4}});
    setOption(args, "d-strides", std::vector<std::vector<size_t>>{{1, 4, 16}});
    for(const char* name : {"type",
                            "a-type",
                            "b-type",
                            "c-type",
                            "d-type",
                            "compute-input-type-A",
                            "compute-input-type-B"})
        setOption(args, name, rocisa::DataType::Float);
    setOption(args, "init-seed", 0u);
    for(const char* name : {"init-a", "init-b", "init-c", "init-alpha", "init-beta"})
        setOption(args, name, InitMode::One);
    setOption(args, "init-d", InitMode::Two);
    setOption(args, "freivalds", true);
    for(const char* name : {"print-valids",
                            "print-tensor-a",
                            "print-tensor-b",
                            "print-tensor-c",
                            "print-tensor-d",
                            "print-tensor-ref",
                            "print-tensor-bias",
                            "print-tensor-gate",
                            "print-tensor-scale-alpha-vec",
                            "print-tensor-amaxd"})
        setOption(args, name, false);
    setOption(args, "print-max", 8);
    hipStream_t stream = nullptr;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    const std::shared_ptr<void> streamOwner(
        stream, [](void* s) { (void)hipStreamDestroy(hipStream_t(s)); });
    for(const auto bounds : {BoundsCheckMode::Disable, BoundsCheckMode::NaN})
    {
        SCOPED_TRACE(static_cast<int>(bounds));
        setOption(args, "bounds-check", bounds);
        ClientProblemFactory factory(args);
        auto                 dataInit = std::make_shared<DataInitialization>(args, factory);
        auto*                problem  = factory.problems().front().get();
        auto                 reporter = std::make_shared<Reporter>();
        ReferenceValidator   validator(args, dataInit, stream);
        validator.setReporter(reporter);
        dataInit->preProblem(problem);
        validator.preProblem(problem);
        const TimingEvents events(0, 0);
        for(int attempt = 0; attempt < 4; ++attempt)
        {
            SCOPED_TRACE(attempt);
            validator.preSolution(nullptr);
            EXPECT_EQ(validator.numWarmupRuns(), 1);
            auto               inputs = dataInit->prepareGPUInputs(problem);
            auto&              output = dynamic_cast<TensileLite::ContractionInputs&>(*inputs);
            std::vector<float> observed(26);
            ASSERT_EQ(hipMemcpy(observed.data(),
                                output.d,
                                observed.size() * sizeof(float),
                                hipMemcpyDeviceToHost),
                      hipSuccess);
            for(size_t batch = 0; batch < 2; ++batch)
                for(size_t column = 0; column < 3; ++column)
                    for(size_t row = 0; row < 2; ++row)
                    {
                        const size_t index = batch * 16 + column * 4 + row;
                        EXPECT_EQ(observed[index], 2);
                        observed[index] = 5; // four 1*1 products plus C=1
                    }
            if(bounds == BoundsCheckMode::NaN)
                EXPECT_FALSE(std::isfinite(observed[2]));
            if(attempt == 1)
                observed[17] += 1; // finite fault in the second batch
            if(attempt == 2)
                observed[2] = 5; // padding is checked only when requested
            ASSERT_EQ(hipMemcpyAsync(output.d,
                                     observed.data(),
                                     observed.size() * sizeof(float),
                                     hipMemcpyHostToDevice,
                                     stream),
                      hipSuccess);
            validator.postWarmup(events, events, stream);
            validator.validateWarmups(inputs, events, events);
            EXPECT_EQ(validator.numWarmupRuns(), 0);
            validator.postSolution();
        }
        EXPECT_EQ(reporter->verdicts,
                  (std::vector<std::string>{"PASSED",
                                            "FAILED",
                                            bounds == BoundsCheckMode::NaN ? "FAILED" : "PASSED",
                                            "PASSED"}));
        EXPECT_EQ(validator.error(), bounds == BoundsCheckMode::NaN ? 2 : 1);
    }
}

TEST(CopyTensorVoid, CopiesSource)
{
    TensorDescriptor   desc("t", rocisa::DataType::Float, {3, 2});
    std::vector<float> src(desc.totalAllocatedElements());
    std::iota(src.begin(), src.end(), 1.0f);
    std::vector<float> dst(src.size(), 0.0f);

    TensileLite::hip::CopyTensorVoid(dst.data(), src.data(), desc, hipMemcpyHostToHost);
    ASSERT_EQ(hipStreamSynchronize(nullptr), hipSuccess);

    EXPECT_EQ(dst, src);
}
