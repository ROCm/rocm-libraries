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

#include <cstring>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/FusedA2AKernArg.hpp>

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
    // A GEMM with MX block scales attached the way tensile_host.cpp does on gfx950,
    // i.e. with setMXScaleA/B's default padScaleTensorFreeDim = true.
    //
    // packedScales mirrors the caller: hipBLASLt hands an external custom kernel the
    // user's own compactly packed scale buffer (true), while the TensileLite benchmark
    // client allocates the scale buffer from this padded descriptor and hands that very
    // buffer over (false).
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
    int64_t emittedScaleStride(CustomArgSemantic semantic, ContractionProblemGemm const& problem)
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::int64, semantic});
        solution.sizeMapping.globalSplitU = 1;

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

// setMXScaleA/B pad the scale tensor for the Tensile-generated gfx950 subtile
// kernels, which recompute the pitch from K anyway and ignore the value they are
// handed. An external custom kernel (rocRoller, aiter, triton) does not: it takes
// this slot verbatim as the leading dimension of the caller's scale buffer, and a
// caller that packs that buffer compactly (hipBLASLt -- it says so with
// setCustomKernelMXScalesArePacked) must be given the packed pitch. Handing it the
// padded pitch mis-addresses every scale row past the first whenever
// ceil(K/mxBlock) % 8 != 0.
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

// The packed pitch is for callers whose scale buffer really is packed. The
// TensileLite benchmark client is not one of them: DataInitialization allocates the
// MX scale tensor straight from the padded descriptor and restrides the canonical
// scales into it, zeroing the tail, then hands that same buffer to the custom
// kernel. Giving it the packed pitch made every odd scale row land in the zeroed
// padding -- an E8M0 0x00 is 2^-127 -- so tensilelite/Tests/custom/custom_rr_*.yaml
// came back with ~3/4 of the output tensor wrong. Default (unpacked) callers must
// get the descriptor's own stride back.
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

// The fix derives the stride from boundIndices() rather than assuming K sits at
// index 0, so it has to stay correct when the bound dim is the slow one. NN puts
// K slow in A (stride is the unpadded free dim M) and fast in B (stride is the
// unpadded K-block count), exercising both orientations in one problem. M=60 and
// K=128 are both unaligned, so compact and padded differ on each side.
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
        problem.setFusedA2AWorld(2);
        problem.setFusedA2AExtent(256);

        FusedA2APeerFields peer0{};
        peer0[FUSED_A2A_SLOT_FLAG_PTR] = const_cast<char*>(&fusedPeerFlag);

        ContractionInputs inputs;
        inputs.fusedA2APeers   = {peer0};
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
