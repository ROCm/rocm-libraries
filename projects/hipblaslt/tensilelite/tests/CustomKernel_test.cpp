/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

#include <algorithm>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/FusedA2AKernArg.hpp>
#ifndef TENSILE_YAML
#include <Tensile/msgpack/MessagePack.hpp>
#endif

#include "FallbackTestUtils.hpp"

using namespace TensileLite;
using TensileLite::testing::dummyProblem;
using TensileLite::testing::makeDevice;

// generateKernelCall rejects a solution whose metadata was never populated by
// testing threads.x / macrotile.x against zero. Every field is deserialized with
// mapOptional and vector3 leaves its components uninitialized by default, so that
// guard only means anything if an unpopulated CustomKernel really does hold zeros.
TEST(CustomKernelTest, DefaultsTripTheUninitializedMetadataGuard)
{
    CustomKernel kernel;

    EXPECT_EQ(kernel.threads.x, 0u);
    EXPECT_EQ(kernel.macrotile.x, 0u);
}

#ifndef TENSILE_YAML
// A kernel's uniformSummationOrder declaration reaches hipBLASLt only through
// this mapping; lost, every declaring kernel would quietly be refused again
// under uniform summation order.
TEST(CustomKernelTest, UniformSummationOrderDeserializes)
{
    for(const bool declared : {false, true})
    {
        msgpack::zone                          zone;
        std::map<std::string, msgpack::object> fields{
            {"name", msgpack::object(std::string("probe"), zone)}};
        if(declared)
            fields["uniformSummationOrder"] = msgpack::object(true, zone);

        msgpack::sbuffer packed;
        msgpack::pack(packed, fields);
        auto unpacked = msgpack::unpack(packed.data(), packed.size());

        CustomKernel                    kernel;
        Serialization::MessagePackInput reader(unpacked.get());
        reader.input(kernel);
        ASSERT_TRUE(reader.error.empty()) << reader.error.front();
        EXPECT_EQ(kernel.name, "probe");
        EXPECT_EQ(kernel.uniformSummationOrder, declared);
    }
}

// Descriptors written before post kernels existed carry neither key and must
// keep loading as tree-only kernels without post kernels.
TEST(CustomKernelTest, PostKernelsDefaultWhenAbsent)
{
    msgpack::zone                          zone;
    std::map<std::string, msgpack::object> fields{
        {"name", msgpack::object(std::string("probe"), zone)}};
    msgpack::sbuffer packed;
    msgpack::pack(packed, fields);
    auto unpacked = msgpack::unpack(packed.data(), packed.size());

    CustomKernel                    kernel;
    Serialization::MessagePackInput reader(unpacked.get());
    reader.input(kernel);
    ASSERT_TRUE(reader.error.empty()) << reader.error.front();
    EXPECT_EQ(kernel.reductions, std::vector<CustomReductionMode>{CustomReductionMode::Tree});
    EXPECT_TRUE(kernel.postKernels.empty());
}

TEST(CustomKernelTest, PostKernelsRoundTrip)
{
    CustomKernel kernel;
    kernel.name       = "probe";
    kernel.reductions = {CustomReductionMode::Tree, CustomReductionMode::Parallel};

    CustomPostKernel post;
    post.name    = "probe_streamKReduce";
    post.args    = {{CustomArgType::address, CustomArgSemantic::AddressWorkspace},
                    {CustomArgType::uint32, CustomArgSemantic::SKSplitFactor},
                    {CustomArgType::uint32, CustomArgSemantic::NumTiles},
                    {CustomArgType::uint32, CustomArgSemantic::WorkspaceOffsetPartials}};
    post.threads = TensileLite::dim3(256, 1, 1);
    post.grid    = {CustomGridSize::TilesBands, CustomGridSize::One, CustomGridSize::One};
    post.bands   = 4;

    kernel.postKernels = {post};

    Serialization::MessagePackOutput writer;
    writer.output(kernel);
    ASSERT_TRUE(writer.error.empty()) << writer.error.front();
    auto unpacked = msgpack::unpack(writer.buffer.data(), writer.buffer.size());

    CustomKernel                    loaded;
    Serialization::MessagePackInput reader(unpacked.get());
    reader.input(loaded);
    ASSERT_TRUE(reader.error.empty()) << reader.error.front();
    EXPECT_EQ(loaded.reductions, kernel.reductions);
    ASSERT_EQ(loaded.postKernels.size(), 1u);
    auto const& got = loaded.postKernels[0];
    EXPECT_EQ(got.kind, CustomPostKernelKind::StreamKReduce);
    EXPECT_EQ(got.name, post.name);
    ASSERT_EQ(got.args.size(), post.args.size());
    for(size_t i = 0; i < post.args.size(); ++i)
    {
        EXPECT_EQ(got.args[i].type, post.args[i].type);
        EXPECT_EQ(got.args[i].semantic, post.args[i].semantic);
    }
    EXPECT_EQ(got.threads.x, 256u);
    EXPECT_EQ(got.grid.x, CustomGridSize::TilesBands);
    EXPECT_EQ(got.bands, 4u);
}
#endif

namespace
{
    // Just enough metadata to get past the guards in generateCustomCall, carrying a
    // single argument so the emitted buffer holds exactly that value.
    void configureProbeKernel(ContractionSolution& solution, CustomArgDefinition arg)
    {
        solution.sizeMapping.macroTile = TensileLite::dim3(128, 128, 1);
        solution.sizeMapping.depthU    = 64;
        // Pin the work-group mapping: the auto path reads it off a HipAMDGPU, which a
        // plain AMDGPU test device is not.
        solution.sizeMapping.workGroupMapping    = 1;
        solution.sizeMapping.workGroupMappingXCC = 1;

        solution.customKernel.name      = "probe";
        solution.customKernel.macrotile = TensileLite::dim3(128, 128, 64);
        solution.customKernel.threads   = TensileLite::dim3(256, 1, 1);
        solution.customKernel.grid
            = {CustomGridSize::TilesX, CustomGridSize::TilesY, CustomGridSize::One};
        solution.customKernel.args = {arg};
    }

    AMDGPU probeDevice()
    {
        return makeDevice(TensileLite::testing::_MI350_CHIP_ID,
                          TensileLite::testing::_SPX_CU,
                          "mi350spx");
    }

    KernelInvocation splitKProbeCall(int16_t        compiledGsu,
                                     int16_t        runtimeGsu,
                                     CustomGridSize gridY = CustomGridSize::TilesYGSU)
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::SplitK});
        solution.customKernel.grid.y      = gridY;
        solution.sizeMapping.globalSplitU = compiledGsu;

        auto problem = dummyProblem();
        problem.setParams().setGSU(runtimeGsu);

        auto              device = probeDevice();
        ContractionInputs inputs;
        StreamKSettings   sk;

        return solution.generateCustomCall<false>(problem, inputs, device, sk);
    }

    uint32_t emittedSplitK(int16_t compiledGsu, int16_t runtimeGsu)
    {
        auto invocation = splitKProbeCall(compiledGsu, runtimeGsu);

        EXPECT_EQ(invocation.args.size(), sizeof(uint32_t));
        uint32_t splitK = 0;
        std::memcpy(&splitK, invocation.args.data(), sizeof(splitK));
        return splitK;
    }

    std::vector<uint8_t> emittedScalar(CustomArgSemantic semantic,
                                       rocisa::DataType  computeType,
                                       ConstantVariant   value)
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::float32, semantic});
        solution.sizeMapping.globalSplitU = 1;

        auto problem = dummyProblem();
        problem.setAlphaType(computeType);
        problem.setBetaType(computeType);

        auto              device = probeDevice();
        ContractionInputs inputs;
        inputs.alpha = value;
        inputs.beta  = value;
        StreamKSettings sk;

        auto invocation = solution.generateCustomCall<false>(problem, inputs, device, sk);

        auto const* bytes = static_cast<uint8_t const*>(invocation.args.data());
        return std::vector<uint8_t>(bytes, bytes + invocation.args.size());
    }
}

// The kernel wants log2(GSU). Taking it from the solution's compiled-in
// globalSplitU disagrees with the grid and workspace sizing, which are built from
// the runtime-effective GSU, so a user-supplied GSU has to win here too.
TEST(CustomKernelTest, SplitKFollowsTheRuntimeGsu)
{
    EXPECT_EQ(emittedSplitK(/*compiled*/ 1, /*runtime*/ 8), 3u);
    EXPECT_EQ(emittedSplitK(/*compiled*/ 1, /*runtime*/ 4), 2u);
    EXPECT_EQ(emittedSplitK(/*compiled*/ 8, /*runtime*/ 1), 0u);
}

// With no runtime override the effective GSU falls back to the compiled value, so
// the emitted argument is unchanged for kernels that never set one.
TEST(CustomKernelTest, SplitKFallsBackToTheCompiledGsu)
{
    EXPECT_EQ(emittedSplitK(/*compiled*/ 1, /*runtime*/ 0), 0u);
    EXPECT_EQ(emittedSplitK(/*compiled*/ 16, /*runtime*/ 0), 4u);
}

// A split-K kernel reduces into D only once every GSU slice of a tile has arrived,
// so TilesYGSU has to launch one row of work groups per slice, following the
// runtime-effective GSU like the SplitK argument does.
TEST(CustomKernelTest, TilesYGsuLaunchesEveryGsuSlice)
{
    auto const unsplit = splitKProbeCall(/*compiled*/ 1, /*runtime*/ 0).numWorkGroups;

    auto const compiled = splitKProbeCall(/*compiled*/ 16, /*runtime*/ 0).numWorkGroups;
    EXPECT_EQ(compiled.x, unsplit.x);
    EXPECT_EQ(compiled.y, 16 * unsplit.y);
    EXPECT_EQ(compiled.z, unsplit.z);

    EXPECT_EQ(splitKProbeCall(/*compiled*/ 1, /*runtime*/ 4).numWorkGroups.y, 4 * unsplit.y);
    EXPECT_EQ(splitKProbeCall(/*compiled*/ 0, /*runtime*/ 0).numWorkGroups.y, unsplit.y);
    EXPECT_EQ(
        splitKProbeCall(/*compiled*/ 1, /*runtime*/ 0, CustomGridSize::TilesY).numWorkGroups.y,
        unsplit.y);
}

// A grid with no GSU term would launch one slice per tile and report success with
// D unwritten, so a split-K launch on one has to fail instead.
TEST(CustomKernelTest, GridWithoutGsuTermRejectsSplitK)
{
    EXPECT_THROW(splitKProbeCall(/*compiled*/ 16, /*runtime*/ 0, CustomGridSize::TilesY),
                 std::runtime_error);
    EXPECT_THROW(splitKProbeCall(/*compiled*/ 1, /*runtime*/ 4, CustomGridSize::TilesY),
                 std::runtime_error);
    EXPECT_NO_THROW(splitKProbeCall(/*compiled*/ 1, /*runtime*/ 0, CustomGridSize::TilesY));
}

// A custom kernel declares alpha and beta as 32-bit slots, so a narrower compute
// type has to be widened before it is written or every argument after it shifts.
// KernelArguments only widens an argument spelled exactly "alpha" or "beta".
TEST(CustomKernelTest, ScalarsFillTheDeclaredThirtyTwoBitSlot)
{
    for(auto semantic : {CustomArgSemantic::Alpha, CustomArgSemantic::Beta})
    {
        EXPECT_EQ(emittedScalar(semantic, rocisa::DataType::Float, 1.5f).size(),
                  sizeof(float));

        auto const fromHalf
            = emittedScalar(semantic, rocisa::DataType::Half, static_cast<Half>(1.5f));
        ASSERT_EQ(fromHalf.size(), sizeof(float));

        float widened = 0.0f;
        std::memcpy(&widened, fromHalf.data(), sizeof(widened));
        EXPECT_EQ(widened, 1.5f);

        EXPECT_EQ(
            emittedScalar(semantic, rocisa::DataType::BFloat16, static_cast<BFloat16>(1.5f))
                .size(),
            sizeof(float));
    }
}

namespace
{
    // hipBLASLt passes compact scales; the TensileLite client uses the padded
    // descriptor. Keep both caller layouts available to the argument probes.
    ContractionProblemGemm mxScaleProblem(size_t M,
                                          size_t N,
                                          size_t K,
                                          bool   transA,
                                          bool   transB,
                                          int    mxBlock      = 32,
                                          bool   packedScales = true)
    {
        auto problem = ContractionProblemGemm::GEMM_Strides(transA,
                                                            transB,
                                                            rocisa::DataType::Float4,
                                                            rocisa::DataType::Float4,
                                                            rocisa::DataType::BFloat16,
                                                            rocisa::DataType::BFloat16,
                                                            M,
                                                            N,
                                                            K,
                                                            1,
                                                            transA ? K : M,
                                                            transA ? K * M : M * K,
                                                            transB ? N : K,
                                                            transB ? N * K : K * N,
                                                            M,
                                                            M * N,
                                                            M,
                                                            M * N,
                                                            0.0);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        problem.setMXScaleA(rocisa::DataType::E8, mxBlock);
        problem.setMXScaleB(rocisa::DataType::E8, mxBlock);
        problem.setCustomKernelMXScalesArePacked(packedScales);
        return problem;
    }

    // The int64 this problem emits into a probe kernel's single scale-stride slot.
    int64_t emittedScaleStride(CustomArgSemantic             semantic,
                               ContractionProblemGemm const& problem,
                               int                           argsVersion       = 0,
                               bool                          useInitialStrides = false)
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::int64, semantic});
        solution.sizeMapping.globalSplitU                      = 1;
        solution.internalArgsSupport.persistentLoopArgsVersion = argsVersion;
        solution.problemType.useInitialStridesAB               = useInitialStrides;
        if(argsVersion == 1)
        {
            // Version 1 is valid only for a persistent data-parallel solution
            // with KernArgsVersion 3.
            solution.internalArgsSupport.version        = 3;
            solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
            solution.customKernel.generated             = true;
        }

        auto              device = probeDevice();
        ContractionInputs inputs;
        StreamKSettings   sk;

        auto invocation = solution.generateCustomCall<false>(problem, inputs, device, sk);

        EXPECT_EQ(invocation.args.size(), sizeof(int64_t));
        int64_t stride = 0;
        std::memcpy(&stride, invocation.args.data(), sizeof(stride));
        return stride;
    }
}

// A compact buffer needs the unpadded pitch when ceil(K/mxBlock) is not a
// multiple of 8; using the descriptor pitch would misaddress later rows.
TEST(CustomKernelTest, ScaleStrideIsCompactNotPadded)
{
    // TN: K is the fast (bound) dim of both A and B, so the padded K-block count
    // would leak straight into the stride.
    struct
    {
        size_t  K;
        int64_t compact;
        int64_t padded;
    } const tnCases[] = {
        {128, 4, 8}, // ceil(K/32) = 4  -> padded to 8
        {384, 12, 16}, // ceil(K/32) = 12 -> padded to 16
        {256, 8, 8}, // ceil(K/32) = 8  -> already aligned, the two agree
    };

    for(auto const& c : tnCases)
    {
        auto problem = mxScaleProblem(64, 64, c.K, /*transA*/ true, /*transB*/ false);

        // The descriptor really does carry the padding; we are diverging from it.
        EXPECT_EQ(problem.mxsa().strides()[1], static_cast<size_t>(c.padded)) << "K=" << c.K;
        EXPECT_EQ(problem.mxsb().strides()[1], static_cast<size_t>(c.padded)) << "K=" << c.K;

        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, problem), c.compact)
            << "K=" << c.K;
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, problem), c.compact)
            << "K=" << c.K;
    }
}

// Both argument versions must preserve the caller's batch layout, including
// padding in the free dimension as well as the K-block dimension.
TEST(CustomKernelTest, ScaleBatchStrideFollowsTheCallerBuffer)
{
    auto packedProblem = mxScaleProblem(60, 60, 128, true, false);
    EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA1, packedProblem), 4 * 60);
    EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB1, packedProblem), 4 * 60);
    for(bool packed : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "packed=" << packed);
        auto          problem  = mxScaleProblem(60, 60, 128, true, false, 32, packed);
        const int64_t expected = packed ? 4 * 60 : 8 * 64;
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA1, problem, 1), expected);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB1, problem, 1), expected);
    }
}

TEST(CustomKernelTest, ScaleStridesIncludeTheInitialStrideForDataParallelArgs)
{
    for(bool packed : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "packed=" << packed);
        auto problem = mxScaleProblem(60, 60, 128, true, false, 32, packed);
        // With initial strides enabled, slots 0 and 1 describe tensor dimensions 0 and 1.
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, problem, 1, true), 1);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, problem, 1, true), 1);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA1, problem, 1, true),
                  packed ? 4 : 8);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB1, problem, 1, true),
                  packed ? 4 : 8);
    }
}

TEST(CustomKernelTest, ScaleStrideRoundsUpPartialBlocks)
{
    for(int block : {16, 32})
    {
        SCOPED_TRACE(::testing::Message() << "block=" << block);
        auto          problem  = mxScaleProblem(60, 60, 129, true, false, block);
        const int64_t expected = block == 16 ? 9 : 5;
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, problem), expected);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, problem), expected);
    }
}

// The TensileLite client allocates and fills the descriptor's padded layout.
// Compact strides would make some rows read padding instead of their scales.
TEST(CustomKernelTest, ScaleStrideFollowsDescriptorWhenCallerDoesNotPack)
{
    struct
    {
        size_t  K;
        int64_t padded; // what the descriptor says, and what must be emitted
    } const tnCases[] = {
        {128, 8}, // ceil(K/32) = 4  -> padded to 8;  packed would have said 4
        {384, 16}, // ceil(K/32) = 12 -> padded to 16; packed would have said 12
        {256, 8}, // ceil(K/32) = 8  -> already aligned, the two agree
    };

    for(auto const& c : tnCases)
    {
        auto problem = mxScaleProblem(
            64, 64, c.K, /*transA*/ true, /*transB*/ false, /*mxBlock*/ 32, /*packedScales*/ false);

        EXPECT_EQ(problem.mxsa().strides()[1], static_cast<size_t>(c.padded)) << "K=" << c.K;
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, problem), c.padded)
            << "K=" << c.K;
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, problem), c.padded)
            << "K=" << c.K;
    }

    // Same for the K-slow orientation the packed path handles specially.
    auto nn = mxScaleProblem(
        60, 60, 128, /*transA*/ false, /*transB*/ false, /*mxBlock*/ 32, /*packedScales*/ false);
    EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, nn), 64);
    EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, nn), 8);
}

// hipBLASLt takes the A and B scale formats independently, so a pre-swizzled
// operand keeps the padded strides while the other operand gets compact ones.
TEST(CustomKernelTest, ScaleStrideFollowsEachOperandsBuffer)
{
    for(bool packedA : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "packedA=" << packedA);
        auto problem = mxScaleProblem(60, 60, 128, /*transA*/ true, /*transB*/ false);
        problem.setCustomKernelMXScaleAIsPacked(packedA);
        problem.setCustomKernelMXScaleBIsPacked(!packedA);

        // ceil(128/32) = 4 K-blocks pad to 8, and M = N = 60 pad to 64.
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, problem), packedA ? 4 : 8);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, problem), packedA ? 8 : 4);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA1, problem, 1),
                  packedA ? 4 * 60 : 8 * 64);
        EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB1, problem, 1),
                  packedA ? 8 * 64 : 4 * 60);
    }
}

// NN places K on different stored axes in A and B. Unaligned M and K make
// compact and padded pitches differ for both operands.
TEST(CustomKernelTest, ScaleStrideIsCompactForEitherBoundDimension)
{
    auto problem = mxScaleProblem(60, 60, 128, /*transA*/ false, /*transB*/ false);

    // A: bound dim is index 1, so the stride is A's free dim, padded to 32.
    EXPECT_EQ(problem.mxsa().strides()[1], 64u);
    EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleA0, problem), 60);

    // B: bound dim is index 0, so the stride is the K-block count, padded to 8.
    EXPECT_EQ(problem.mxsb().strides()[1], 8u);
    EXPECT_EQ(emittedScaleStride(CustomArgSemantic::StrideScaleB0, problem), 4);
}

namespace
{
    // Grid x and ComputeUnits kernarg of a persistent probe kernel.
    std::pair<size_t, int32_t> computeUnitsLaunch(int smCountTarget, int persistentMaxCUs)
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::int32, CustomArgSemantic::ComputeUnits});
        solution.customKernel.grid
            = {CustomGridSize::ComputeUnits, CustomGridSize::One, CustomGridSize::One};

        auto problem = dummyProblem();
        problem.setParams().setSmCountTarget(smCountTarget);
        auto device             = probeDevice();
        device.persistentMaxCUs = persistentMaxCUs;
        ContractionInputs inputs;
        StreamKSettings   sk;

        auto invocation = solution.generateCustomCall<false>(problem, inputs, device, sk);

        EXPECT_EQ(invocation.numWorkGroups.y, 1u);
        EXPECT_EQ(invocation.numWorkGroups.z, 1u);
        EXPECT_EQ(invocation.args.size(), sizeof(int32_t));
        int32_t cuCount = 0;
        std::memcpy(&cuCount, invocation.args.data(), sizeof(cuCount));
        return {invocation.numWorkGroups.x, cuCount};
    }
}

// Persistent skinny-GEMM kernels stride by CU count; the launch grid and the
// ComputeUnits kernarg have to be the same value.
TEST(CustomKernelTest, ComputeUnitsGridAndArgFollowHardware)
{
    constexpr int cus = TensileLite::testing::_SPX_CU;
    EXPECT_EQ(computeUnitsLaunch(0, 0), std::make_pair(size_t{cus}, int32_t{cus}));
}

// The SM-count target and persistentMaxCUs cap that value as they cap Tensile's
// persistent grids, and the tighter cap wins.
TEST(CustomKernelTest, ComputeUnitsHonorTheCuBudget)
{
    constexpr int cus = TensileLite::testing::_SPX_CU;
    EXPECT_EQ(computeUnitsLaunch(64, 0), std::make_pair(size_t{64}, int32_t{64}));
    EXPECT_EQ(computeUnitsLaunch(64, 48), std::make_pair(size_t{48}, int32_t{48}));
    EXPECT_EQ(computeUnitsLaunch(0, 96), std::make_pair(size_t{96}, int32_t{96}));
    EXPECT_EQ(computeUnitsLaunch(4 * cus, 0), std::make_pair(size_t{cus}, int32_t{cus}));
}

namespace
{
    char const fusedPeerFlag = 0;
    char const fusedCounter  = 0;

    constexpr size_t peerGroupBytes     = FUSED_A2A_SLOT_COUNT * sizeof(void*);
    constexpr size_t peer0FlagOffset    = 0;
    constexpr size_t peer1FlagOffset    = peer0FlagOffset + peerGroupBytes;
    constexpr size_t counterOffset      = peer0FlagOffset + FUSED_A2A_MAX_RANKS * peerGroupBytes;
    constexpr size_t fusedMyRankOffset  = counterOffset + sizeof(void*);
    constexpr size_t fusedWOffset       = fusedMyRankOffset + sizeof(uint32_t);
    constexpr size_t fusedDrainOffset   = fusedWOffset + sizeof(uint32_t);
    constexpr size_t fusedAMOffset      = fusedDrainOffset + sizeof(uint32_t);

    template <typename T>
    T packedAt(KernelArguments const& args, size_t offset)
    {
        T result{};
        std::memcpy(&result, static_cast<uint8_t const*>(args.data()) + offset, sizeof(T));
        return result;
    }

    KernelInvocation fusedCustomCall()
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::FusedA2ASegment});

        auto problem = dummyProblem();
        problem.setFusedGemmA2A(true);
        problem.setWorld(2);
        problem.setFusedA2AExtent(256);

        FusedA2APeerFields peer0{};
        peer0[FUSED_A2A_SLOT_FLAG_PTR] = const_cast<char*>(&fusedPeerFlag);

        ContractionInputs inputs;
        inputs.fusedA2APeers   = {peer0, FusedA2APeerFields{}};
        inputs.fusedA2ACounter = const_cast<char*>(&fusedCounter);
        inputs.fusedA2AMyRank  = 1;
        inputs.fusedA2ADrain   = FUSED_A2A_DRAIN_RECV;

        return solution.generateCustomCall<false>(
            problem, inputs, probeDevice(), StreamKSettings{});
    }
}

TEST(CustomKernelTest, FusedSegmentPacksEveryField)
{
    auto fused = fusedCustomCall();
    ASSERT_EQ(fused.args.size(), FUSED_A2A_SEGMENT_BYTES);

    EXPECT_EQ(packedAt<void*>(fused.args, peer0FlagOffset), const_cast<char*>(&fusedPeerFlag));
    EXPECT_EQ(packedAt<void*>(fused.args, peer1FlagOffset), nullptr);
    EXPECT_EQ(packedAt<void*>(fused.args, counterOffset), const_cast<char*>(&fusedCounter));
    EXPECT_EQ(packedAt<uint32_t>(fused.args, fusedMyRankOffset), 1u);
    EXPECT_EQ(packedAt<uint32_t>(fused.args, fusedWOffset), 2u);
    EXPECT_EQ(packedAt<uint32_t>(fused.args, fusedDrainOffset), FUSED_A2A_DRAIN_RECV);
    EXPECT_EQ(packedAt<uint32_t>(fused.args, fusedAMOffset), 256u);
}

namespace
{
    void configurePostKernels(ContractionSolution& solution)
    {
        solution.internalArgsSupport.version                   = 3;
        solution.internalArgsSupport.persistentLoopArgsVersion = 3;
        solution.customKernel.name                             = "probe";
        solution.customKernel.threads                          = TensileLite::dim3(256, 1, 1);
        solution.customKernel.reductions
            = {CustomReductionMode::Tree, CustomReductionMode::Parallel};

        CustomPostKernel post;
        post.name    = "probe_streamKReduce";
        post.threads = TensileLite::dim3(256, 1, 1);

        solution.customKernel.postKernels = {post};
    }
}

// Layout 3 is what keeps a runtime without post-kernel support from running
// the main kernel alone, so the two must always come together.
TEST(CustomKernelTest, PostKernelsValidate)
{
    {
        ContractionSolution solution;
        configurePostKernels(solution);
        EXPECT_NO_THROW(solution.validatePersistentLoopArgs());
    }

    using Solution = ContractionSolution;
    const std::vector<std::function<void(Solution&)>> invalid{
        [](Solution& s) { s.internalArgsSupport.persistentLoopArgsVersion = 0; },
        [](Solution& s) { s.internalArgsSupport.version = 2; },
        [](Solution& s) {
            s.customKernel.postKernels.clear();
            s.customKernel.reductions = {CustomReductionMode::Tree};
        },
        [](Solution& s) { s.customKernel.reductions = {CustomReductionMode::Tree}; },
        [](Solution& s) {
            s.customKernel.postKernels.clear();
            s.customKernel.reductions                       = {CustomReductionMode::Parallel};
            s.internalArgsSupport.persistentLoopArgsVersion = 0;
        },
        [](Solution& s) { s.customKernel.reductions.push_back(CustomReductionMode::Tree); },
        [](Solution& s) { s.customKernel.reductions.clear(); },
        [](Solution& s) { s.customKernel.postKernels[0].name = "other_streamKReduce"; },
        [](Solution& s) { s.customKernel.postKernels.push_back(s.customKernel.postKernels[0]); },
        [](Solution& s) { s.customKernel.postKernels[0].bands = 0; },
        [](Solution& s) { s.customKernel.generated = true; },
    };
    for(size_t i = 0; i < invalid.size(); ++i)
    {
        ContractionSolution solution;
        configurePostKernels(solution);
        invalid[i](solution);
        EXPECT_THROW(solution.validatePersistentLoopArgs(), std::runtime_error) << "case " << i;
    }
}

TEST(CustomKernelTest, PostKernelsAreNotLaunchedYet)
{
    ContractionSolution solution;
    configurePostKernels(solution);
    EXPECT_THROW(solution.solve(dummyProblem(), ContractionInputs(), probeDevice()),
                 std::runtime_error);
}

namespace
{
    // FNV-1a over the packed arguments and the launch geometry.
    struct CallDigest
    {
        uint64_t hash = 0xcbf29ce484222325ull;

        void add(void const* data, size_t size)
        {
            auto const* bytes = static_cast<uint8_t const*>(data);
            for(size_t i = 0; i < size; ++i)
                hash = (hash ^ bytes[i]) * 0x100000001b3ull;
        }

        void add(TensileLite::dim3 const& d)
        {
            for(size_t v : {size_t(d.x), size_t(d.y), size_t(d.z)})
                add(&v, sizeof(v));
        }
    };

    uint64_t digest(KernelInvocation const& call)
    {
        CallDigest d;
        d.add(call.args.data(), call.args.size());
        d.add(call.workGroupSize);
        d.add(call.numWorkGroups);
        d.add(call.numWorkItems);
        d.add(call.clusterDim);
        d.add(call.kernelName.data(), call.kernelName.size());
        return d.hash;
    }

    // Every semantic the probe problem can pack deterministically.
    std::vector<CustomArgDefinition> representativeArgs()
    {
        using S = CustomArgSemantic;
        using T = CustomArgType;
        return {
            {T::uint32, S::GemmInfo},
            {T::uint32, S::NumWorkGroups},
            {T::uint32, S::TensileInternalArg0},
            {T::uint32, S::TensileInternalArg1},
            {T::uint32, S::InternalArgs1},
            {T::uint32, S::SizeFree0},
            {T::uint32, S::SizeFree1},
            {T::uint32, S::SizeFree2},
            {T::uint32, S::SizeSum},
            {T::uint32, S::SizeSumDiv2},
            {T::uint32, S::SizeSum1},
            {T::uint32, S::SizeSum2},
            {T::address, S::AddressA},
            {T::address, S::AddressB},
            {T::address, S::AddressC},
            {T::address, S::AddressD},
            {T::uint32, S::StrideA0},
            {T::uint32, S::StrideA1},
            {T::uint32, S::StrideB0},
            {T::uint32, S::StrideB1},
            {T::uint32, S::StrideC0},
            {T::uint32, S::StrideC1},
            {T::uint32, S::StrideD0},
            {T::uint64, S::StrideD1},
            {T::uint32, S::StrideCK},
            {T::uint64, S::StrideA0Bytes},
            {T::uint64, S::StrideD0Bytes},
            {T::float32, S::Alpha},
            {T::float32, S::Beta},
            {T::int32, S::ComputeUnits},
            {T::uint32, S::SplitK},
            {T::uint32, S::OutputBF16},
            {T::uint32, S::Padding},
            {T::uint32, S::ConstantZero, 4},
            {T::uint32, S::ConstantOne},
            {T::uint32, S::DebugPattern},
            {T::uint32, S::DebugPattern},
            {T::address, S::AddressWorkspace},
            {T::address, S::AddressFlags},
            {T::address, S::AddressSynchronizer},
            {T::address, S::Synchronizer},
            {T::address, S::DebugBuffer},
            {T::uint32, S::GemmCount},
            {T::uint32, S::ItersPerTile},
            {T::uint32, S::MagicNumberItersPerTile},
            {T::uint32, S::MagicShiftItersPerTile},
            {T::uint32, S::TotalIters},
            {T::uint32, S::SKItersPerWG},
            {T::uint32, S::SKGrid},
            {T::uint32, S::SKTilesAndSplit},
            {T::uint32, S::MagicNumberSize, 0, 1},
            {T::uint32, S::MagicShiftSize, 0, 1},
            {T::uint32, S::FactorDim},
            {T::uint32, S::ActivationTypeArg},
            {T::float32, S::ActivationArg, 0, 0},
            {T::uint32, S::ActivationArg, 0, 1},
            {T::uint32, S::GSUSync},
            {T::int32, S::ESMRuntimeSupported},
            {T::int64, S::BatchOffsetD},
            {T::int64, S::BatchOffsetA},
            {T::int32, S::RR_WorkgroupMapping},
            {T::uint32, S::RR_MagicMultipleWgm},
            {T::int32, S::RR_QuotientTilesMByWgm},
            {T::uint32, S::RR_MagicShiftAndSignWgmMainBlock},
        };
    }

    ContractionInputs representativeInputs()
    {
        auto              fake = [](uintptr_t address) { return reinterpret_cast<void*>(address); };
        ContractionInputs inputs;
        inputs.a              = fake(0xa000);
        inputs.b              = fake(0xb000);
        inputs.c              = fake(0xc000);
        inputs.d              = fake(0xd000);
        inputs.ws             = fake(0xe000);
        inputs.Synchronizer   = fake(0xf000);
        inputs.alpha          = 2.5f;
        inputs.beta           = -1.25f;
        inputs.activationArgs = {0.5f};
        inputs.batchOffsetA   = 0x300000005LL;
        inputs.batchOffsetD   = 0x100000001LL;
        return inputs;
    }

    template <typename T>
    T namedArg(KernelInvocation const& call, std::string const& name)
    {
        auto it = KernelArguments::const_iterator(call.args, name);
        if(it == call.args.end())
            throw std::runtime_error("Missing packed argument " + name);
        auto field = *it;
        EXPECT_EQ(field.second, sizeof(T)) << name;
        T result{};
        std::memcpy(&result, field.first, std::min(sizeof(T), field.second));
        return result;
    }

    using Configure = std::function<void(
        ContractionSolution&, ContractionProblemGemm&, PersistentLaunchSettings&)>;

    struct RepresentativeCall
    {
        char const*                                  name;
        Configure                                    configure;
        std::function<void(KernelInvocation const&)> expect;
        size_t                                       argBytes;
        uint64_t                                     digest;
    };

    void configureStreamK(ContractionSolution& solution, PersistentLaunchSettings& launch)
    {
        solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::StreamK;
        solution.customKernel.workspaceType         = CustomWorkspaceType::StreamK;
        solution.customKernel.grid
            = {CustomGridSize::StreamKWithBatch, CustomGridSize::One, CustomGridSize::One};
        launch.tileProcessingStrategy = TileProcessingStrategy::StreamK;
    }

    void* address(KernelInvocation const& call, char const* name)
    {
        return namedArg<void*>(call, name);
    }

    void* const inputWs   = reinterpret_cast<void*>(0xe000);
    void* const inputSync = reinterpret_cast<void*>(0xf000);

    // Golden digests pinning the byte layout of generateCustomCall's arguments
    // and launch.
    std::vector<RepresentativeCall> const& representativeCalls()
    {
        static const std::vector<RepresentativeCall> calls{
            {"tiles",
             [](auto& solution, auto&, auto&) {
                 solution.customKernel.grid
                     = {CustomGridSize::TilesX, CustomGridSize::TilesY, CustomGridSize::Batch};
             },
             [](auto const& call) {
                 EXPECT_EQ(call.numWorkGroups.x, 8u);
                 EXPECT_EQ(call.numWorkGroups.y, 8u);
                 EXPECT_EQ(address(call, "AddressD"), reinterpret_cast<void*>(0xd000));
             },
             316,
             0x77e20b50bcd2df0aull},
            {"tilesFlattened",
             [](auto& solution, auto&, auto&) {
                 solution.internalArgsSupport.version = 1;
                 solution.customKernel.grid
                     = {CustomGridSize::TilesX, CustomGridSize::TilesY, CustomGridSize::Batch};
             },
             [](auto const& call) {
                 EXPECT_EQ(call.numWorkGroups.x, 64u);
                 EXPECT_EQ(call.numWorkGroups.y, 1u);
             },
             316,
             0xf5ea9f5546ac715full},
            {"splitKMultipleBuffer",
             [](auto& solution, auto& problem, auto&) {
                 solution.sizeMapping.globalAccumulation = 2;
                 solution.customKernel.grid
                     = {CustomGridSize::TilesXYBatchGSU, CustomGridSize::One, CustomGridSize::One};
                 problem.setParams().setGSU(4);
             },
             [](auto const& call) {
                 EXPECT_EQ(call.numWorkGroups.x, 256u);
                 EXPECT_EQ(namedArg<uint32_t>(call, "SplitK"), 2u);
                 EXPECT_EQ(address(call, "AddressC"), inputWs);
                 EXPECT_EQ(address(call, "AddressD"), inputWs);
             },
             316,
             0x7d33ab2fe5036458ull},
            {"computeUnits",
             [](auto& solution, auto& problem, auto&) {
                 solution.customKernel.grid
                     = {CustomGridSize::ComputeUnits, CustomGridSize::One, CustomGridSize::One};
                 problem.setParams().setSmCountTarget(96);
             },
             [](auto const& call) {
                 EXPECT_EQ(call.numWorkGroups.x, 96u);
                 EXPECT_EQ(namedArg<int32_t>(call, "ComputeUnits"), 96);
             },
             316,
             0x1841435f58750687ull},
            {"streamKTree",
             [](auto& solution, auto&, auto& launch) {
                 configureStreamK(solution, launch);
                 launch.grid      = 37;
                 launch.reduction = origami::reduction_t::tree;
             },
             [](auto const& call) {
                 EXPECT_EQ(call.numWorkGroups.x, 37u);
                 EXPECT_EQ(namedArg<uint32_t>(call, "SKGrid"), 37u);
                 EXPECT_EQ(address(call, "AddressFlags"), inputSync);
                 EXPECT_EQ(address(call, "AddressD"), reinterpret_cast<void*>(0xd000));
             },
             316,
             0x4f9b2824978a5ef9ull},
            {"streamKParallel",
             [](auto& solution, auto&, auto& launch) {
                 configureStreamK(solution, launch);
                 launch.grid      = 128;
                 launch.reduction = origami::reduction_t::parallel;
             },
             [](auto const& call) {
                 EXPECT_EQ(namedArg<uint32_t>(call, "SKTilesAndSplit"), 2u);
                 EXPECT_EQ(address(call, "AddressFlags"), nullptr);
                 EXPECT_EQ(address(call, "AddressC"), inputWs);
                 EXPECT_EQ(address(call, "AddressD"), inputWs);
             },
             316,
             0x89b5fa721fdaf378ull},
            {"streamKParallelPostKernels",
             [](auto& solution, auto&, auto& launch) {
                 configureStreamK(solution, launch);
                 configurePostKernels(solution);
                 launch.grid      = 192;
                 launch.reduction = origami::reduction_t::parallel;
             },
             [](auto const& call) {
                 EXPECT_EQ(namedArg<uint32_t>(call, "SKTilesAndSplit"), 3u);
                 EXPECT_NE(namedArg<uint32_t>(call, "TotalIters"), 0u);
                 EXPECT_EQ(address(call, "AddressD"), inputWs);
             },
             316,
             0x9a053eb3611e583dull},
            {"dataParallelV1Cluster",
             [](auto& solution, auto&, auto& launch) {
                 solution.internalArgsSupport.version                   = 3;
                 solution.internalArgsSupport.persistentLoopArgsVersion = 1;
                 solution.sizeMapping.clusterDim             = TensileLite::dim3(2, 2, 1);
                 solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
                 solution.customKernel.grid
                     = {CustomGridSize::PersistentGrid, CustomGridSize::One, CustomGridSize::One};
                 launch.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
                 launch.grid                   = 64;
             },
             [](auto const& call) {
                 EXPECT_EQ(call.numWorkGroups.x, 32u);
                 EXPECT_EQ(call.numWorkGroups.y, 2u);
                 EXPECT_EQ(call.clusterDim.x, 2u);
             },
             316,
             0x03f7667a96fa1a6full},
        };
        return calls;
    }

    template <bool T_Debug>
    KernelInvocation representativeCall(RepresentativeCall const& call)
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {});
        solution.kernelName               = "probe";
        solution.sizeMapping.globalSplitU = 1;
        solution.customKernel.args        = representativeArgs();
        auto                     problem  = dummyProblem();
        PersistentLaunchSettings launch;
        call.configure(solution, problem, launch);
        auto device = probeDevice();
        return solution.generateCustomCall<T_Debug>(
            problem, representativeInputs(), device, launch);
    }
}

// Pins the packed arguments and launch geometry of custom kernels covering the
// tile, split-K, persistent, DataParallel cluster and Stream-K tree/parallel paths.
TEST(CustomKernelTest, CustomCallMatchesGoldenDigest)
{
    for(auto const& call : representativeCalls())
    {
        SCOPED_TRACE(call.name);
        auto invocation = representativeCall<false>(call);
        EXPECT_EQ(invocation.args.size(), call.argBytes);
        EXPECT_EQ(digest(invocation), call.digest) << std::hex << digest(invocation);

        ::testing::internal::CaptureStdout();
        auto logged = representativeCall<true>(call);
        ::testing::internal::GetCapturedStdout();
        EXPECT_EQ(digest(logged), call.digest);
        call.expect(logged);
    }
}
