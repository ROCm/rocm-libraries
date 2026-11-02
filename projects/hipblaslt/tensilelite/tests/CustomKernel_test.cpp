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

constexpr size_t tileGridK = 64;

// A wide extent of 1025 and a skinny extent of 9 give partial tiles on both axes of
// the macrotiles below, so an off-by-one in the tile count shows up.
static ContractionProblemGemm tileGridProblem(size_t m, size_t n, size_t batch)
{
    ContractionProblemGemm problem = ContractionProblemGemm::GEMM(
        false, false, m, n, tileGridK, m, tileGridK, m, 0.0, false, batch);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);
    return problem;
}

static std::vector<CustomArgSemantic> perDimensionSemantics()
{
    return {CustomArgSemantic::NumWorkGroupsX,
            CustomArgSemantic::NumWorkGroupsY,
            CustomArgSemantic::NumWorkGroupsZ};
}

// A probe kernel carrying one uint32 kernarg per semantic, on the given grid.
static KernelInvocation launchWithSemantics(ContractionSolution&                  solution,
                                            vector3<CustomGridSize>               grid,
                                            TensileLite::dim3                     macrotile,
                                            std::vector<CustomArgSemantic> const& semantics,
                                            ContractionProblemGemm const&         problem)
{
    configureProbeKernel(solution, {CustomArgType::uint32, semantics.front()});
    solution.customKernel.grid      = grid;
    solution.customKernel.macrotile = macrotile;
    solution.customKernel.args.clear();
    for(CustomArgSemantic semantic : semantics)
        solution.customKernel.args.push_back({CustomArgType::uint32, semantic});

    AMDGPU            device = probeDevice();
    ContractionInputs inputs;
    StreamKSettings   sk;

    KernelInvocation invocation = solution.generateCustomCall<false>(problem, inputs, device, sk);
    EXPECT_EQ(invocation.args.size(), semantics.size() * sizeof(uint32_t));
    return invocation;
}

static uint32_t emittedUint32(KernelInvocation const& invocation, size_t index)
{
    uint32_t value = 0;
    std::memcpy(&value,
                static_cast<uint8_t const*>(invocation.args.data()) + index * sizeof(value),
                sizeof(value));
    return value;
}

// The three kernargs starting at first must be the launched grid.
static void expectPerDimensionArgs(KernelInvocation const& invocation, size_t first)
{
    EXPECT_EQ(emittedUint32(invocation, first + 0), invocation.numWorkGroups.x);
    EXPECT_EQ(emittedUint32(invocation, first + 1), invocation.numWorkGroups.y);
    EXPECT_EQ(emittedUint32(invocation, first + 2), invocation.numWorkGroups.z);
}

// The skinny MXFP8 kernels want the wide tiles on grid x; with a skinny n that is
// the natural [TilesX, TilesY, Batch] order over a [BN, bm, depth] macrotile.
TEST(CustomKernelTest, NumWorkGroupsPerDimensionFollowTheTileGrid)
{
    ContractionSolution solution;

    KernelInvocation invocation
        = launchWithSemantics(solution,
                              {CustomGridSize::TilesX, CustomGridSize::TilesY, CustomGridSize::Batch},
                              TensileLite::dim3(256, 4, tileGridK),
                              perDimensionSemantics(),
                              tileGridProblem(1025, 9, 3));

    EXPECT_EQ(invocation.numWorkGroups.x, 5u);
    EXPECT_EQ(invocation.numWorkGroups.y, 3u);
    EXPECT_EQ(invocation.numWorkGroups.z, 3u);
    expectPerDimensionArgs(invocation, 0);
}

// With a skinny m the wide extent is n, so the macrotile and the grid axes swap and
// the kernargs must still report the launched grid, not the tile counts.
TEST(CustomKernelTest, NumWorkGroupsPerDimensionFollowTheSwappedTileGrid)
{
    ContractionSolution solution;

    KernelInvocation invocation
        = launchWithSemantics(solution,
                              {CustomGridSize::TilesY, CustomGridSize::TilesX, CustomGridSize::Batch},
                              TensileLite::dim3(4, 256, tileGridK),
                              perDimensionSemantics(),
                              tileGridProblem(9, 1025, 3));

    EXPECT_EQ(invocation.numWorkGroups.x, 5u);
    EXPECT_EQ(invocation.numWorkGroups.y, 3u);
    EXPECT_EQ(invocation.numWorkGroups.z, 3u);
    expectPerDimensionArgs(invocation, 0);
}

// One [BN, bm, depth] macrotile on [TilesWide, TilesSkinny, Batch] must launch the same
// grid, and report it through the kernargs, whichever of m and n is the skinny one.
TEST(CustomKernelTest, WideSkinnyGridIsIndependentOfTheOrientation)
{
    // 1025 / 256 and 9 / 4 both leave a tail tile.
    for(ContractionProblemGemm const& problem :
        {tileGridProblem(9, 1025, 3), tileGridProblem(1025, 9, 3)})
    {
        ContractionSolution solution;

        KernelInvocation invocation
            = launchWithSemantics(solution,
                                  {CustomGridSize::TilesWide,
                                   CustomGridSize::TilesSkinny,
                                   CustomGridSize::Batch},
                                  TensileLite::dim3(256, 4, tileGridK),
                                  perDimensionSemantics(),
                                  problem);

        EXPECT_EQ(invocation.numWorkGroups.x, 5u);
        EXPECT_EQ(invocation.numWorkGroups.y, 3u);
        EXPECT_EQ(invocation.numWorkGroups.z, 3u);
        expectPerDimensionArgs(invocation, 0);
    }
}

TEST(CustomKernelTest, WideSkinnyGridRoundTripsThroughStrings)
{
    for(std::string name : {"TilesWide", "TilesSkinny"})
        EXPECT_EQ(toString(fromStringCustomGridSize(name)), name);
}

TEST(CustomKernelTest, WideSkinnyGridRejectsSplitK)
{
    EXPECT_THROW(splitKProbeCall(/*compiled*/ 1, /*runtime*/ 4, CustomGridSize::TilesWide),
                 std::runtime_error);
    EXPECT_THROW(splitKProbeCall(/*compiled*/ 1, /*runtime*/ 4, CustomGridSize::TilesSkinny),
                 std::runtime_error);
}

// Skinny MXFP8 emitter launch grid (see skinny_gemm_support.Problem.grid).
static KernelInvocation launchSkinnyEmitter(ContractionSolution&          solution,
                                            std::string const&            kernelName,
                                            TensileLite::dim3             macrotile,
                                            ContractionProblemGemm const& problem)
{
    configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::NumWorkGroupsX});
    solution.customKernel.name = kernelName;
    solution.customKernel.macrotile = macrotile;
    solution.customKernel.grid      = {CustomGridSize::TilesWide,
                                  CustomGridSize::TilesSkinny,
                                  CustomGridSize::Batch};
    solution.customKernel.args      = {{CustomArgType::uint32, CustomArgSemantic::NumWorkGroupsX},
                                  {CustomArgType::uint32, CustomArgSemantic::NumWorkGroupsY},
                                  {CustomArgType::uint32, CustomArgSemantic::NumWorkGroupsZ}};

    AMDGPU            device = probeDevice();
    ContractionInputs inputs;
    StreamKSettings   sk;
    return solution.generateCustomCall<false>(problem, inputs, device, sk);
}

TEST(CustomKernelTest, SkinnyMxfp8EmitterPersistentGridUsesBlockCount)
{
    ContractionSolution solution;
    constexpr char kernelName[] = "gemm_BM16_TN2_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF4_"
                                  "GROUP1_NUM_XCDS8_GRID_3D0_USE_BETA1_MFMA1_LDS1";
    auto problem = ContractionProblemGemm::GEMM(false, false, 5120, 16, 25600, 25600, 25600, 5120, 0.0, false, 1);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);

    KernelInvocation invocation
        = launchSkinnyEmitter(solution, kernelName, TensileLite::dim3(16, 2, 4096), problem);

    EXPECT_EQ(invocation.numWorkGroups.x, 160u);
    EXPECT_EQ(invocation.numWorkGroups.y, 1u);
    EXPECT_EQ(invocation.numWorkGroups.z, 1u);
    expectPerDimensionArgs(invocation, 0);
}

TEST(CustomKernelTest, SkinnyMxfp8EmitterGrid3dUsesWideBlockColumns)
{
    ContractionSolution solution;
    constexpr char kernelName[] = "gemm_BM16_TN1_LANES_K64_WAVES1_UP_ITERS0_PARTIAL_STEPS0_PF6_"
                                  "GROUP1_NUM_XCDS1_GRID_3D1_USE_BETA1_MFMA1_LDS1";
    auto problem = ContractionProblemGemm::GEMM(false, false, 1, 10240, 5120, 5120, 5120, 1, 0.0, false, 1);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);

    KernelInvocation invocation
        = launchSkinnyEmitter(solution, kernelName, TensileLite::dim3(16, 1, 4096), problem);

    EXPECT_EQ(invocation.numWorkGroups.x, 640u);
    EXPECT_EQ(invocation.numWorkGroups.y, 1u);
    EXPECT_EQ(invocation.numWorkGroups.z, 1u);
    expectPerDimensionArgs(invocation, 0);
}

// The skinny MXFP8 custom.config declares KernArgsVersion 0, which leaves
// internalArgsSupport.version at 0: the 3-D grid the kernel needs is never flattened.
TEST(CustomKernelTest, NumWorkGroupsPerDimensionKeepTheGridUnflattened)
{
    ContractionSolution solution;
    solution.internalArgsSupport.version = 0;

    KernelInvocation invocation
        = launchWithSemantics(solution,
                              {CustomGridSize::TilesX, CustomGridSize::TilesY, CustomGridSize::Batch},
                              TensileLite::dim3(256, 4, tileGridK),
                              perDimensionSemantics(),
                              tileGridProblem(1025, 9, 7));

    EXPECT_EQ(invocation.numWorkGroups.x, 5u);
    EXPECT_EQ(invocation.numWorkGroups.y, 3u);
    EXPECT_EQ(invocation.numWorkGroups.z, 7u);
    expectPerDimensionArgs(invocation, 0);
}

// NumWorkGroups with internal args folds the grid into x; the per-dimension kernargs
// must report that launched grid, and NumWorkGroups keeps its own meaning.
TEST(CustomKernelTest, NumWorkGroupsPerDimensionSeeTheFlattenedGrid)
{
    ContractionSolution solution;
    solution.internalArgsSupport.version = 1;

    std::vector<CustomArgSemantic> semantics = {CustomArgSemantic::NumWorkGroups};
    for(CustomArgSemantic semantic : perDimensionSemantics())
        semantics.push_back(semantic);

    KernelInvocation invocation
        = launchWithSemantics(solution,
                              {CustomGridSize::TilesX, CustomGridSize::TilesY, CustomGridSize::Batch},
                              TensileLite::dim3(256, 4, tileGridK),
                              semantics,
                              tileGridProblem(1025, 9, 3));

    // The (5, 3, 3) tile grid is launched as 45 work groups along x.
    EXPECT_EQ(invocation.numWorkGroups.x, 45u);
    EXPECT_EQ(invocation.numWorkGroups.y, 1u);
    EXPECT_EQ(invocation.numWorkGroups.z, 1u);
    EXPECT_EQ(emittedUint32(invocation, 0), 45u);
    expectPerDimensionArgs(invocation, 1);
}

// The custom.config args are deserialized by name, so the new semantics need a
// string mapping in both directions.
TEST(CustomKernelTest, NumWorkGroupsPerDimensionRoundTripThroughStrings)
{
    for(std::string name : {"NumWorkGroupsX", "NumWorkGroupsY", "NumWorkGroupsZ"})
        EXPECT_EQ(toString(fromStringCustomArgSemantic(name)), name);
}
