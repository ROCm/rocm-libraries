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
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/FusedA2AKernArg.hpp>
#include <Tensile/MagicDivision.hpp>
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
    constexpr size_t fusedNumCuOffset   = fusedAMOffset + sizeof(uint32_t);

    template <typename T>
    T packedAt(KernelArguments const& args, size_t offset)
    {
        T result{};
        std::memcpy(&result, static_cast<uint8_t const*>(args.data()) + offset, sizeof(T));
        return result;
    }

    KernelInvocation fusedCustomCall(FusedA2A mode     = FusedA2A::GemmA2A,
                                     int      fixedWGM = std::numeric_limits<int>::max(),
                                     void*    myBuf    = nullptr)
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::FusedA2ASegment});
        solution.problemType.fusedA2A = mode;

        auto problem = dummyProblem();
        problem.setFusedA2A(mode);
        problem.setWorld(2);
        problem.setFusedA2AExtent(256);

        FusedA2APeerFields peer0{};
        peer0[FUSED_A2A_SLOT_FLAG_PTR] = const_cast<char*>(&fusedPeerFlag);

        FusedA2APeerFields peer1{};
        peer1[FUSED_A2A_SLOT_BUF_PTR] = myBuf;

        ContractionInputs inputs;
        inputs.fusedA2APeers   = {peer0, peer1};
        inputs.fusedA2ACounter = const_cast<char*>(&fusedCounter);
        inputs.fusedA2AMyRank  = 1;
        inputs.fusedA2ADrain   = FUSED_A2A_DRAIN_RECV;

        auto device     = probeDevice();
        device.fixedWGM = fixedWGM;
        return solution.generateCustomCall<false>(problem, inputs, device, StreamKSettings{});
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
    EXPECT_EQ(packedAt<uint32_t>(fused.args, fusedNumCuOffset),
              static_cast<uint32_t>(TensileLite::testing::_SPX_CU));
}

TEST(CustomKernelTest, A2AGemmRejectsANegativeRuntimeWgm)
{
    EXPECT_NO_THROW(fusedCustomCall(FusedA2A::A2AGemm));
    EXPECT_THROW(fusedCustomCall(FusedA2A::A2AGemm, -8), std::runtime_error);
    EXPECT_NO_THROW(fusedCustomCall(FusedA2A::GemmA2A, -8));
}

TEST(CustomKernelTest, A2AGemmRejectsAMisalignedA2ABuf)
{
    alignas(128) static char buf[256];
    constexpr int            defaultWGM = std::numeric_limits<int>::max();
    EXPECT_NO_THROW(fusedCustomCall(FusedA2A::A2AGemm, defaultWGM, buf));
    EXPECT_THROW(fusedCustomCall(FusedA2A::A2AGemm, defaultWGM, buf + 64), std::runtime_error);
    EXPECT_NO_THROW(fusedCustomCall(FusedA2A::GemmA2A, defaultWGM, buf + 64));
}

// The random seed takes exactly one 32-bit slot. Follow it with another argument
// to verify proper width is used.
TEST(CustomKernelTest, RndSeedDescriptorPacksThroughCustomCall)
{
    ContractionSolution solution;
    configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::RNDSeed});
    solution.customKernel.args.push_back({CustomArgType::uint32, CustomArgSemantic::Padding});

    auto              problem = dummyProblem();
    auto              device  = probeDevice();
    ContractionInputs inputs;
    StreamKSettings   sk;

    auto invocation = solution.generateCustomCall<false>(problem, inputs, device, sk);

    ASSERT_EQ(invocation.args.size(), 2 * sizeof(uint32_t));

    uint32_t padding = 1;
    std::memcpy(&padding,
                static_cast<uint8_t const*>(invocation.args.data()) + sizeof(uint32_t),
                sizeof(padding));
    EXPECT_EQ(padding, 0u);
}

static size_t integerArgBytes(CustomArgType type)
{
    switch(type)
    {
    case CustomArgType::int8:
    case CustomArgType::uint8:
        return 1;
    case CustomArgType::int16:
    case CustomArgType::uint16:
        return 2;
    case CustomArgType::int32:
    case CustomArgType::uint32:
        return 4;
    case CustomArgType::int64:
    case CustomArgType::uint64:
        return 8;
    default:
        throw std::logic_error("not an integer kernel argument type");
    }
}

// Same-typed kernargs are packed without padding, so slot index sits at index * width.
// Values are zero-extended, which assumes a little-endian host.
static uint64_t emittedInteger(KernelInvocation const& invocation, size_t index, size_t width)
{
    uint64_t value = 0;
    std::memcpy(&value, static_cast<uint8_t const*>(invocation.args.data()) + index * width, width);
    return value;
}

// The byte strides must be computed in integers: a float product rounds to 24 bits.
static std::array<uint64_t, 4> emittedByteStrides(rocisa::DataType abType,
                                                  rocisa::DataType cdType,
                                                  CustomArgType    type,
                                                  size_t           lda,
                                                  size_t           ldb,
                                                  size_t           ldc,
                                                  size_t           ldd)
{
    ContractionProblemGemm problem = ContractionProblemGemm::GEMM_Strides(true,
                                                                          false,
                                                                          abType,
                                                                          abType,
                                                                          cdType,
                                                                          cdType,
                                                                          4,
                                                                          2,
                                                                          64,
                                                                          1,
                                                                          lda,
                                                                          -1,
                                                                          ldb,
                                                                          -1,
                                                                          ldc,
                                                                          -1,
                                                                          ldd,
                                                                          -1,
                                                                          0.0);
    problem.setComputeInputTypeA(abType);
    problem.setComputeInputTypeB(abType);

    ContractionSolution solution;
    configureProbeKernel(solution, {type, CustomArgSemantic::StrideA0Bytes});
    solution.customKernel.args = {{type, CustomArgSemantic::StrideA0Bytes},
                                  {type, CustomArgSemantic::StrideB0Bytes},
                                  {type, CustomArgSemantic::StrideC0Bytes},
                                  {type, CustomArgSemantic::StrideD0Bytes}};

    AMDGPU            device = probeDevice();
    ContractionInputs inputs;
    StreamKSettings   sk;
    KernelInvocation  invocation = solution.generateCustomCall<false>(problem, inputs, device, sk);
    size_t width = integerArgBytes(type);
    EXPECT_EQ(invocation.args.size(), 4 * width);
    return {emittedInteger(invocation, 0, width),
            emittedInteger(invocation, 1, width),
            emittedInteger(invocation, 2, width),
            emittedInteger(invocation, 3, width)};
}

TEST(CustomKernelTest, ByteStridesAreExactBeyondTheFloatMantissa)
{
    // 2-byte C/D: 2 * 16777219 = 33554438, which a float cannot represent.
    std::array<uint64_t, 4> bf16 = emittedByteStrides(
        rocisa::DataType::Float8, rocisa::DataType::BFloat16, CustomArgType::int32, 64, 64, 16777219, 16777219);
    EXPECT_EQ(bf16[0], 64u);
    EXPECT_EQ(bf16[1], 64u);
    EXPECT_EQ(bf16[2], 33554438u);
    EXPECT_EQ(bf16[3], 33554438u);

    // 1-byte A/B: 2^26 + 4 needs 27 significant bits.
    constexpr size_t big = (size_t(1) << 26) + 4;
    std::array<uint64_t, 4> fp8
        = emittedByteStrides(rocisa::DataType::Float8,
                             rocisa::DataType::BFloat16,
                             CustomArgType::int32,
                             big,
                             big,
                             4,
                             4);
    EXPECT_EQ(fp8[0], 67108868u);
    EXPECT_EQ(fp8[1], 67108868u);
    EXPECT_EQ(fp8[2], 8u);
    EXPECT_EQ(fp8[3], 8u);
}

// Byte strides of A, B, C and D when only the given tensor gets a non-trivial leading dimension.
static std::array<uint64_t, 4> emittedByteStridesWithLd(CustomArgType type, size_t tensor, size_t ld)
{
    std::array<size_t, 4> lds = {64, 64, 4, 4};
    lds[tensor]               = ld;
    return emittedByteStrides(rocisa::DataType::Float8,
                              rocisa::DataType::BFloat16,
                              type,
                              lds[0],
                              lds[1],
                              lds[2],
                              lds[3]);
}

// A byte stride must fit the integer type its kernel declares: the largest value is accepted and
// read back exactly, and one more is rejected.
TEST(CustomKernelTest, ByteStridesMustFitTheirKernelArgumentType)
{
    struct Limit
    {
        CustomArgType type;
        uint64_t      max;
    };
    std::array<Limit, 8> const limits = {{{CustomArgType::int8, std::numeric_limits<int8_t>::max()},
                                          {CustomArgType::uint8, std::numeric_limits<uint8_t>::max()},
                                          {CustomArgType::int16, std::numeric_limits<int16_t>::max()},
                                          {CustomArgType::uint16, std::numeric_limits<uint16_t>::max()},
                                          {CustomArgType::int32, std::numeric_limits<int32_t>::max()},
                                          {CustomArgType::uint32, std::numeric_limits<uint32_t>::max()},
                                          {CustomArgType::int64, std::numeric_limits<int64_t>::max()},
                                          {CustomArgType::uint64, std::numeric_limits<uint64_t>::max()}}};

    for(Limit const& limit : limits)
    {
        SCOPED_TRACE("argument type " + std::to_string(static_cast<int>(limit.type)));
        // A and B have 1-byte elements, C and D have 2-byte elements, and the limits are odd.
        for(size_t tensor = 0; tensor < 4; tensor++)
        {
            SCOPED_TRACE("tensor " + std::to_string(tensor));
            size_t bytesPerElement = tensor < 2 ? 1 : 2;
            // size_t(-1) asks GEMM_Strides for the default leading dimension, so it is not usable.
            size_t largestLd = std::min<uint64_t>(limit.max / bytesPerElement,
                                                  std::numeric_limits<size_t>::max() - 1);
            std::array<uint64_t, 4> strides = emittedByteStridesWithLd(limit.type, tensor, largestLd);
            EXPECT_EQ(strides[tensor], largestLd * bytesPerElement);

            // No larger value fits in size_t for uint64.
            if(limit.max == std::numeric_limits<uint64_t>::max())
                continue;
            size_t overLd = largestLd + 1;
            EXPECT_GT(overLd * bytesPerElement, limit.max);
            EXPECT_THROW(emittedByteStridesWithLd(limit.type, tensor, overLd), std::runtime_error);
        }
    }
}

// A byte stride bound to a floating point or boolean kernarg cannot be checked and must be rejected.
TEST(CustomKernelTest, ByteStridesNeedAnIntegerKernelArgument)
{
    for(CustomArgType type : {CustomArgType::float16,
                              CustomArgType::float32,
                              CustomArgType::float64,
                              CustomArgType::boolean})
    {
        SCOPED_TRACE("argument type " + std::to_string(static_cast<int>(type)));
        try
        {
            emittedByteStridesWithLd(type, 0, 64);
            ADD_FAILURE() << "expected std::runtime_error";
        }
        catch(std::runtime_error const& e)
        {
            EXPECT_NE(std::string(e.what()).find("needs an integer kernel argument"),
                      std::string::npos)
                << e.what();
        }
    }
}

// ---------------------------------------------------------------------------
// StreamK scheduling arguments of a handwritten custom kernel.
//
// generateCustomCall() and generateSingleCall() pack the same words from
// StreamK::Static / StreamK::Dynamic::calculateKernelArgs(), so a descriptor in
// the right family has to receive exactly the bytes a generated kernel of that
// mode receives.
// ---------------------------------------------------------------------------
namespace
{
    enum class StreamKMode
    {
        Static,
        Dynamic,
    };

    char const* toString(StreamKMode mode)
    {
        switch(mode)
        {
        case StreamKMode::Static:
            return "Static";
        case StreamKMode::Dynamic:
            return "Dynamic";
        }
        return "?";
    }

    StreamK::WorkAssignment assignmentOf(StreamKMode mode)
    {
        switch(mode)
        {
        case StreamKMode::Static:
            return StreamK::WorkAssignment::StaticGrid;
        case StreamKMode::Dynamic:
            return StreamK::WorkAssignment::DynamicWorkQueue;
        }
        return StreamK::WorkAssignment::StaticGrid;
    }

    // The scheduling words of one mode, in kernel-argument order: the name the
    // generated packer gives the word, and the semantic a descriptor asks for.
    struct SchedulingWord
    {
        char const*       generatedName;
        CustomArgSemantic semantic;
        char const*       customName;
    };

    std::vector<SchedulingWord> schedulingWords(StreamKMode mode)
    {
        using S = CustomArgSemantic;
        switch(mode)
        {
        case StreamKMode::Static:
            return {{"itersPerTile", S::ItersPerTile, "ItersPerTile"},
                    {"magicNumberItersPerTile", S::MagicNumberItersPerTile, "MagicNumberItersPerTile"},
                    {"magicShiftItersPerTile", S::MagicShiftItersPerTile, "MagicShiftItersPerTile"},
                    {"SKItersPerWG", S::SKItersPerWG, "SKItersPerWG"},
                    {"skGrid", S::SKGrid, "SKGrid"},
                    {"skTiles", S::SKTilesAndSplit, "SKTilesAndSplit"}};
        case StreamKMode::Dynamic:
            return {{"ItersPerTile", S::ItersPerTile, "ItersPerTile"},
                    {"TotalItems", S::TotalItems, "TotalItems"},
                    {"SKTiles", S::SKTiles, "SKTiles"},
                    {"SKSplit", S::SKSplit, "SKSplit"},
                    {"SKItersPerWI", S::SKItersPerWI, "SKItersPerWI"},
                    {"SKGrid", S::SKGrid, "SKGrid"}};
        }
        return {};
    }

    void configureStreamK(ContractionSolution& solution,
                          StreamK::WorkAssignment       assignment,
                          bool                 perTileExtraIters = false)
    {
        solution.kernelName                                 = "streamk_probe";
        solution.sizeMapping.tileProcessingStrategy         = TileProcessingStrategy::StreamK;
        solution.sizeMapping.workAssignment                 = assignment;
        solution.sizeMapping.macroTile                      = TensileLite::dim3(128, 128, 1);
        solution.sizeMapping.depthU                         = 64;
        solution.sizeMapping.workGroupSize                  = TensileLite::dim3(256, 1, 1);
        solution.sizeMapping.waveNum                        = 4;
        solution.sizeMapping.matrixInstruction              = {16, 16, 4, 1};
        solution.sizeMapping.workGroupMapping               = 1;
        solution.sizeMapping.workGroupMappingXCC            = 1;
        solution.sizeMapping.CUOccupancy                    = 1;
        solution.internalArgsSupport.version                = 3;
        solution.internalArgsSupport.useUniversalArgs       = true;
        solution.internalArgsSupport.persistentLoopArgsVersion = 0;
        solution.internalArgsSupport.perTileExtraIters      = perTileExtraIters;
    }

    // A handwritten twin of a generated StreamK solution: same sizeMapping, a
    // descriptor that asks for `words` and no others.
    void makeHandwritten(ContractionSolution&             solution,
                         std::vector<CustomArgSemantic> const& words,
                         TensileLite::dim3                macrotile = TensileLite::dim3(128, 128, 64))
    {
        solution.customKernel.name      = "streamk_probe";
        solution.customKernel.macrotile = macrotile;
        solution.customKernel.threads   = TensileLite::dim3(256, 1, 1);
        solution.customKernel.grid
            = {CustomGridSize::StreamKNoBatch, CustomGridSize::One, CustomGridSize::One};
        solution.customKernel.args.clear();
        for(auto semantic : words)
            solution.customKernel.args.push_back({CustomArgType::uint32, semantic});
    }

    std::vector<CustomArgSemantic> semanticsOf(StreamKMode mode)
    {
        std::vector<CustomArgSemantic> out;
        for(auto const& word : schedulingWords(mode))
            out.push_back(word.semantic);
        return out;
    }

    ContractionProblemGemm streamKProblem(size_t m = 257, size_t n = 385, size_t k = 129, size_t batches = 3)
    {
        auto problem = ContractionProblemGemm::GEMM(
            false, false, m, n, k, m, std::max(size_t{1}, k), m, 1.0, false, batches);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        problem.setParams().setUniformSummationOrder(0);
        return problem;
    }

    uint32_t word(KernelArguments const& args, std::string const& name)
    {
        auto it = KernelArguments::const_iterator(args, name);
        if(it == args.end())
            throw std::runtime_error("Missing packed argument " + name);
        auto field = *it;
        EXPECT_EQ(field.second, sizeof(uint32_t)) << name;
        uint32_t result = 0;
        std::memcpy(&result, field.first, sizeof(result));
        return result;
    }

    size_t tileCount(ContractionProblemGemm const& problem, size_t mt0 = 128, size_t mt1 = 128)
    {
        const size_t m = (problem.freeSizeA(0) + mt0 - 1) / mt0;
        const size_t n = (problem.freeSizeB(0) + mt1 - 1) / mt1;
        size_t       b = 1;
        for(size_t i = 0; i < problem.batchIndices().size(); ++i)
            b *= problem.batchSize(i);
        return m * n * b;
    }
}

// (a) A handwritten descriptor in the right family receives, word for word, what a
// generated kernel of the same mode receives.
TEST(CustomKernelTest, StreamKCustomWordsMatchTheGeneratedPacker)
{
    struct Shape
    {
        size_t k, batches;
    };
    struct Launch
    {
        size_t               tilesMultiple; // grid = tilesMultiple * tiles, or a fixed grid when 0
        size_t               fixedGrid;
        origami::reduction_t reduction;
    };

    for(auto mode : {StreamKMode::Static, StreamKMode::Dynamic})
    for(bool uniformSummationOrder : {false, true})
    for(bool kUnit : {false, true})
    for(auto shape : {Shape{129, 3}, Shape{65, 1}, Shape{0, 2}})
    for(auto launchSpec : {Launch{0, 5, origami::reduction_t::tree},
                           Launch{0, 7, origami::reduction_t::tree},
                           Launch{1, 0, origami::reduction_t::tree},
                           Launch{2, 0, origami::reduction_t::parallel}})
    {
        // The parallel split factor is a static StreamK notion; the dynamic queue is tree-only.
        if(launchSpec.reduction == origami::reduction_t::parallel && mode == StreamKMode::Dynamic)
            continue;

        SCOPED_TRACE(::testing::Message()
                     << toString(mode) << " uso=" << uniformSummationOrder << " kUnit=" << kUnit
                     << " K=" << shape.k << " batches=" << shape.batches << " grid="
                     << launchSpec.fixedGrid << "+" << launchSpec.tilesMultiple << "xTiles"
                     << " reduction=" << static_cast<int>(launchSpec.reduction));

        ContractionSolution generated;
        configureStreamK(generated, assignmentOf(mode), /*perTileExtraIters*/ true);
        ContractionSolution custom;
        configureStreamK(custom, assignmentOf(mode), /*perTileExtraIters*/ true);
        makeHandwritten(custom, semanticsOf(mode));
        if(kUnit)
        {
            // A K unit wider than one iteration pads every tile.
            generated.problemType.mxScaleFormat = static_cast<int>(MXScaleFormat::HostPreSwizzle);
            custom.problemType.mxScaleFormat    = static_cast<int>(MXScaleFormat::HostPreSwizzle);
            ASSERT_GT(generated.skKGranularity(), size_t{1});
        }

        auto problem = streamKProblem(257, 385, shape.k, shape.batches);
        problem.setParams().setUniformSummationOrder(uniformSummationOrder ? 1 : 0);

        auto device = probeDevice();

        const size_t tiles = tileCount(problem);
        StreamKSettings launch;
        launch.grid = launchSpec.tilesMultiple ? launchSpec.tilesMultiple * tiles
                                               : launchSpec.fixedGrid;
        launch.reduction  = launchSpec.reduction;
        launch.totalTiles = tiles;

        auto const generatedCall
            = generated.generateSingleCall<true>(problem, ContractionInputs{}, device, launch, GSUSettings{});
        auto const customCall
            = custom.generateCustomCall<true>(problem, ContractionInputs{}, device, launch);

        for(auto const& w : schedulingWords(mode))
            EXPECT_EQ(word(customCall.args, w.customName), word(generatedCall.args, w.generatedName))
                << w.customName;
    }
}

namespace
{
    // The static packing as it was before calculateStreamKArgs, written out
    // independently so a change to the shared packer cannot move both sides.
    struct LegacyStatic
    {
        uint32_t magicNumber = 0, magicShift = 0, totalIters = 0, itersPerWG = 0, skTiles = 0;
    };

    LegacyStatic legacyStatic(size_t               numTiles,
                        size_t               itersPerTile,
                        size_t               grid,
                        origami::reduction_t reduction,
                        int                  fullTiles)
    {
        LegacyStatic out;
        if(grid == 0)
            return out;
        itersPerTile    = std::max(itersPerTile, size_t{1});
        out.totalIters  = static_cast<uint32_t>(numTiles * itersPerTile);
        out.magicNumber = magicNumberAlg2(static_cast<uint32_t>(itersPerTile), &out.magicShift);
        if(reduction == origami::reduction_t::parallel)
        {
            const uint32_t skSplit = static_cast<uint32_t>(grid / numTiles);
            out.itersPerWG = static_cast<uint32_t>(itersPerTile) / std::max(skSplit, uint32_t(1));
            out.skTiles    = skSplit;
        }
        else
        {
            size_t skTotalTiles = grid;
            if(numTiles % grid != 0)
            {
                const size_t skT = numTiles > grid ? grid * fullTiles + numTiles % grid : numTiles;
                skTotalTiles     = std::min(skT, numTiles);
            }
            out.itersPerWG = static_cast<uint32_t>(skTotalTiles * itersPerTile / grid);
            out.skTiles    = static_cast<uint32_t>(skTotalTiles);
        }
        return out;
    }
}

// (b) The static semantics keep the bytes they had before the shared packer. The
// descriptor's semantic declares the words, so a kernel whose sizeMapping is not StreamK
// receives them too.
TEST(CustomKernelTest, StreamKCustomStaticKeepsItsLegacyBytes)
{
    const std::vector<CustomArgSemantic> staticWords{CustomArgSemantic::ItersPerTile,
                                                  CustomArgSemantic::TotalIters,
                                                  CustomArgSemantic::MagicNumberItersPerTile,
                                                  CustomArgSemantic::MagicShiftItersPerTile,
                                                  CustomArgSemantic::SKItersPerWG,
                                                  CustomArgSemantic::SKGrid,
                                                  CustomArgSemantic::SKTilesAndSplit};

    for(bool streamKPolicy : {true, false})
    for(auto reduction : {origami::reduction_t::tree, origami::reduction_t::parallel})
    for(size_t gridMultiple : {size_t{0}, size_t{1}, size_t{2}})
    for(size_t fixedGrid : {size_t{0}, size_t{5}, size_t{7}})
    {
        const size_t grid = gridMultiple ? gridMultiple * 36 : fixedGrid;
        // Parallel reduction packs grid / tiles, which only means something above one split.
        if(reduction == origami::reduction_t::parallel && (grid < 36 || grid % 36 != 0))
            continue;

        SCOPED_TRACE(::testing::Message() << "streamKPolicy=" << streamKPolicy << " grid=" << grid
                                          << " reduction=" << static_cast<int>(reduction));

        ContractionSolution custom;
        configureStreamK(custom, StreamK::WorkAssignment::StaticGrid);
        if(!streamKPolicy)
            custom.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::None;
        makeHandwritten(custom, staticWords);

        auto problem = streamKProblem();
        ASSERT_EQ(tileCount(problem), 36u);
        auto device = probeDevice();

        StreamKSettings launch;
        launch.grid      = grid;
        launch.reduction = reduction;

        auto const call = custom.generateCustomCall<true>(problem, ContractionInputs{}, device, launch);

        const auto expected = legacyStatic(36, 3, grid, reduction, device.skFullTiles);
        EXPECT_EQ(word(call.args, "ItersPerTile"), grid ? 3u : 0u);
        EXPECT_EQ(word(call.args, "TotalIters"), expected.totalIters);
        EXPECT_EQ(word(call.args, "MagicNumberItersPerTile"), expected.magicNumber);
        EXPECT_EQ(word(call.args, "MagicShiftItersPerTile"), expected.magicShift);
        EXPECT_EQ(word(call.args, "SKItersPerWG"), expected.itersPerWG);
        EXPECT_EQ(word(call.args, "SKGrid"), static_cast<uint32_t>(grid));
        EXPECT_EQ(word(call.args, "SKTilesAndSplit"), expected.skTiles);
    }
}

// Uniform summation order is the one deliberate change to the static words: a kernel
// that declares the runtime gate now receives bit 29 of MagicShiftItersPerTile when
// the problem asks for the mode, as a generated kernel does. Without the capability
// or without the mode the word is the legacy one.
TEST(CustomKernelTest, StreamKCustomStaticUniformSummationOrderBitNeedsCapabilityAndMode)
{
    for(bool capable : {false, true})
    for(bool requested : {false, true})
    {
        SCOPED_TRACE(::testing::Message() << "capable=" << capable << " requested=" << requested);
        ContractionSolution custom;
        configureStreamK(custom, StreamK::WorkAssignment::StaticGrid, capable);
        makeHandwritten(custom, {CustomArgSemantic::MagicShiftItersPerTile});

        auto problem = streamKProblem();
        problem.setParams().setUniformSummationOrder(requested ? 1 : 0);
        auto device = probeDevice();

        StreamKSettings launch;
        launch.grid      = 36;
        launch.reduction = origami::reduction_t::tree;

        auto const call = custom.generateCustomCall<true>(problem, ContractionInputs{}, device, launch);
        const uint32_t shift = word(call.args, "MagicShiftItersPerTile");
        EXPECT_EQ((shift & StreamK::Slot2Bits::UniformSummationOrderBit) != 0, capable && requested);
        EXPECT_EQ(shift & ~StreamK::Slot2Bits::UniformSummationOrderBit,
                  legacyStatic(36, 3, 36, launch.reduction, device.skFullTiles).magicShift);
    }
}

// ItersPerTile and every packed word divide K by the DepthU the launch was sized
// from, sizeMapping.depthU, not by customKernel.macrotile.z, which a descriptor may
// leave at zero.
TEST(CustomKernelTest, StreamKCustomItersPerTileFollowsTheSizeMappingDepthU)
{
    for(size_t macrotileZ : {size_t{0}, size_t{64}, size_t{32}})
    {
        SCOPED_TRACE(macrotileZ);
        ContractionSolution custom;
        configureStreamK(custom, StreamK::WorkAssignment::StaticGrid);
        makeHandwritten(custom,
                        {CustomArgSemantic::ItersPerTile},
                        TensileLite::dim3(128, 128, macrotileZ));

        auto problem = streamKProblem();
        auto device  = probeDevice();
        StreamKSettings launch;
        launch.grid = 36;

        auto const call = custom.generateCustomCall<true>(problem, ContractionInputs{}, device, launch);
        EXPECT_EQ(word(call.args, "ItersPerTile"),
                  std::max<size_t>(1, problem.getItersPerTile(custom.sizeMapping)));
    }
}

// (c) A descriptor's StreamK words must all belong to one family, fixed by the first
// StreamK semantic packed. Anything else would be handed words packed for another
// mode and run the wrong schedule silently.
namespace
{
    void setUpHandwritten(ContractionSolution&                  solution,
                          StreamK::WorkAssignment                        assignment,
                          std::vector<CustomArgSemantic> const& words)
    {
        configureStreamK(solution, assignment);
        makeHandwritten(solution, words);
    }

    using S = CustomArgSemantic;

    // The family check runs while generateCustomCall() packs the descriptor, so this
    // is how a descriptor is accepted or refused.
    KernelInvocation packCustomCall(ContractionSolution& solution, size_t grid = 36)
    {
        StreamKSettings launch;
        launch.grid = grid;
        auto device = probeDevice();
        return solution.generateCustomCall<false>(
            streamKProblem(), ContractionInputs{}, device, launch);
    }
}

TEST(CustomKernelTest, StreamKFamilyMatchingDescriptorsAreAccepted)
{
    for(auto mode : {StreamKMode::Static, StreamKMode::Dynamic})
    {
        SCOPED_TRACE(toString(mode));
        ContractionSolution solution;
        setUpHandwritten(solution, assignmentOf(mode), semanticsOf(mode));
        EXPECT_NO_THROW(solution.validatePersistentLoopArgs());
        EXPECT_NO_THROW(packCustomCall(solution));
    }
    // Words every family shares, and no StreamK words at all, fit any assignment.
    for(auto assignment :
        {StreamK::WorkAssignment::StaticGrid, StreamK::WorkAssignment::DynamicWorkQueue, StreamK::WorkAssignment::Hybrid})
    {
        ContractionSolution solution;
        setUpHandwritten(
            solution, assignment, {S::ItersPerTile, S::SKGrid, S::AddressA});
        EXPECT_NO_THROW(packCustomCall(solution));
    }
}

TEST(CustomKernelTest, StreamKFamilyRejectsDescriptorsThatMixFamilies)
{
    const std::vector<std::vector<CustomArgSemantic>> mixed{
        {S::SKItersPerWG, S::SKTiles},
        {S::MagicNumberItersPerTile, S::SKSplit},
        {S::SKTiles, S::SKItersPerWG},
        // TotalIters is a static word.
        {S::TotalIters, S::SKTiles},
        {S::TotalItems, S::TotalIters},
    };
    for(auto assignment :
        {StreamK::WorkAssignment::StaticGrid, StreamK::WorkAssignment::DynamicWorkQueue, StreamK::WorkAssignment::Hybrid})
    for(auto const& words : mixed)
    {
        ContractionSolution solution;
        setUpHandwritten(solution, assignment, words);
        try
        {
            packCustomCall(solution);
            ADD_FAILURE() << "a descriptor that mixes StreamK families was accepted";
        }
        catch(std::runtime_error const& e)
        {
            EXPECT_NE(std::string(e.what()).find("mixes StreamK argument families"),
                      std::string::npos)
                << e.what();
        }
    }
}

// A static word on a launch without a grid is all zeros, but it still fixes the
// family, so a dynamic word after it is refused as on any other launch.
TEST(CustomKernelTest, StreamKFamilyIsCheckedOnALaunchWithoutAGrid)
{
    ContractionSolution solution;
    setUpHandwritten(solution, StreamK::WorkAssignment::StaticGrid, {S::SKItersPerWG, S::SKTiles});
    EXPECT_THROW(packCustomCall(solution, 0), std::runtime_error);
}

// ItersPerTile divides K by the DepthU the launch was sized from.
TEST(CustomKernelTest, StreamKCustomRejectsADescriptorWithoutDepthU)
{
    ContractionSolution solution;
    setUpHandwritten(
        solution, StreamK::WorkAssignment::StaticGrid, semanticsOf(StreamKMode::Static));
    solution.sizeMapping.depthU = 0;
    EXPECT_THROW(packCustomCall(solution), std::runtime_error);
}

// With a K unit wider than one iteration the flat iteration space counts tiles padded
// to a whole unit, and TotalIters is that extent, not tiles * ItersPerTile.
TEST(CustomKernelTest, StreamKCustomTotalItersMeasuresPaddedTiles)
{
    ContractionSolution custom;
    configureStreamK(custom, StreamK::WorkAssignment::StaticGrid);
    makeHandwritten(custom, {S::ItersPerTile, S::TotalIters});
    custom.problemType.mxScaleFormat = static_cast<int>(MXScaleFormat::HostPreSwizzle);
    const size_t unit = custom.skKGranularity();
    ASSERT_GT(unit, size_t{1});

    auto problem = streamKProblem(); // 36 tiles of 3 iterations
    auto device  = probeDevice();
    StreamKSettings launch;
    launch.grid = 7;

    auto const   call = custom.generateCustomCall<true>(problem, ContractionInputs{}, device, launch);
    const size_t padded = StreamK::paddedItersPerTile(3, unit);
    ASSERT_NE(padded, size_t{3});
    EXPECT_EQ(word(call.args, "ItersPerTile"), 3u);
    EXPECT_EQ(word(call.args, "TotalIters"), 36 * padded);
}
