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

static ContractionProblemGemm longShortProblem(size_t m, size_t n, size_t batch)
{
    constexpr size_t k = 64;

    ContractionProblemGemm problem
        = ContractionProblemGemm::GEMM(false, false, m, n, k, m, k, m, 0.0, false, batch);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);
    return problem;
}

namespace
{
    KernelInvocation longShortCall(size_t  m,
                                   size_t  n,
                                   size_t  batch,
                                   int16_t compiledGsu = 1,
                                   int16_t runtimeGsu  = 0)
    {
        ContractionSolution solution;
        configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::SizeFree0});
        solution.customKernel.macrotile = TensileLite::dim3(256, 4, 64);
        solution.customKernel.grid
            = {CustomGridSize::TilesLong, CustomGridSize::TilesShort, CustomGridSize::Batch};
        solution.sizeMapping.globalSplitU = compiledGsu;

        ContractionProblemGemm problem = longShortProblem(m, n, batch);
        problem.setParams().setGSU(runtimeGsu);

        AMDGPU            device = probeDevice();
        ContractionInputs inputs;
        StreamKSettings   sk;

        return solution.generateCustomCall<false>(problem, inputs, device, sk);
    }

    void expectLongShortGrid(size_t m, size_t n, size_t batch, TensileLite::dim3 expected)
    {
        TensileLite::dim3 const grid = longShortCall(m, n, batch).numWorkGroups;
        EXPECT_EQ(grid.x, expected.x);
        EXPECT_EQ(grid.y, expected.y);
        EXPECT_EQ(grid.z, expected.z);
    }
}

// The long extent is tiled by macrotile.x and the short one by macrotile.y, whichever
// of M and N is longer.
TEST(CustomKernelTest, LongShortGridIsOrientationAgnostic)
{
    expectLongShortGrid(5000, 3, 1, TensileLite::dim3(20, 1, 1));
    expectLongShortGrid(3, 5000, 1, TensileLite::dim3(20, 1, 1));
}

TEST(CustomKernelTest, LongShortGridRoundsTailsUp)
{
    expectLongShortGrid(1025, 9, 1, TensileLite::dim3(5, 3, 1));
    expectLongShortGrid(9, 1025, 1, TensileLite::dim3(5, 3, 1));
    expectLongShortGrid(1024, 8, 1, TensileLite::dim3(4, 2, 1));
}

TEST(CustomKernelTest, LongShortGridCarriesTheBatch)
{
    expectLongShortGrid(512, 4, 3, TensileLite::dim3(2, 1, 3));
}

// The long/short grids launch one slice per tile, so a split-K run would leave D unwritten.
TEST(CustomKernelTest, LongShortGridRejectsSplitK)
{
    EXPECT_THROW(longShortCall(512, 4, 1, /*compiled*/ 16), std::runtime_error);
    EXPECT_THROW(longShortCall(512, 4, 1, /*compiled*/ 1, /*runtime*/ 4), std::runtime_error);
    EXPECT_NO_THROW(longShortCall(512, 4, 1, /*compiled*/ 1, /*runtime*/ 0));
}

// Probe kernel with one uint32 kernarg per semantic, launched on the given grid.
static KernelInvocation launchWithSemantics(ContractionSolution&                 solution,
                                     vector3<CustomGridSize>               grid,
                                     TensileLite::dim3                     macrotile,
                                     std::vector<CustomArgSemantic> const& semantics,
                                     ContractionProblemGemm const&         problem,
                                     AMDGPU const&                         device)
{
    configureProbeKernel(solution, {CustomArgType::uint32, semantics.front()});
    solution.customKernel.grid      = grid;
    solution.customKernel.macrotile = macrotile;
    solution.customKernel.args.clear();
    for(CustomArgSemantic semantic : semantics)
        solution.customKernel.args.push_back({CustomArgType::uint32, semantic});

    ContractionInputs inputs;
    StreamKSettings   sk;
    KernelInvocation  invocation
        = solution.generateCustomCall<false>(problem, inputs, device, sk);
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

static std::vector<CustomArgSemantic> perDimensionSemantics()
{
    return {CustomArgSemantic::NumWorkGroupsX,
            CustomArgSemantic::NumWorkGroupsY,
            CustomArgSemantic::NumWorkGroupsZ};
}

static void expectComputeUnitsPerDimension(int32_t smCountTarget, int32_t persistentMaxCUs, uint32_t cus)
{
    ContractionSolution solution;
    ContractionProblemGemm problem = dummyProblem();
    problem.setParams().setSmCountTarget(smCountTarget);
    AMDGPU device           = probeDevice();
    device.persistentMaxCUs = persistentMaxCUs;

    KernelInvocation invocation
        = launchWithSemantics(solution,
                              {CustomGridSize::ComputeUnits,
                               CustomGridSize::One,
                               CustomGridSize::One},
                              TensileLite::dim3(128, 128, 64),
                              perDimensionSemantics(),
                              problem,
                              device);

    EXPECT_EQ(invocation.numWorkGroups.x, cus);
    EXPECT_EQ(invocation.numWorkGroups.y, 1u);
    EXPECT_EQ(invocation.numWorkGroups.z, 1u);
    expectPerDimensionArgs(invocation, 0);
}

static TensileLite::dim3 longShortMacrotile()
{
    return TensileLite::dim3(256, 4, 64);
}

static vector3<CustomGridSize> longShortBatchGrid()
{
    return {CustomGridSize::TilesLong, CustomGridSize::TilesShort, CustomGridSize::Batch};
}

// NumWorkGroupsX/Y/Z carry the launched grid, including a persistent ComputeUnits
// grid after the CU budget caps are applied.
TEST(CustomKernelTest, NumWorkGroupsPerDimensionFollowComputeUnitsGrid)
{
    constexpr uint32_t cus = TensileLite::testing::_SPX_CU;
    expectComputeUnitsPerDimension(0, 0, cus);
    expectComputeUnitsPerDimension(64, 0, 64);
    expectComputeUnitsPerDimension(64, 48, 48);
    expectComputeUnitsPerDimension(4 * cus, 0, cus);
}

TEST(CustomKernelTest, NumWorkGroupsPerDimensionFollowTileGrid)
{
    ContractionSolution solution;

    KernelInvocation invocation = launchWithSemantics(solution,
                                                      longShortBatchGrid(),
                                                      longShortMacrotile(),
                                                      perDimensionSemantics(),
                                                      longShortProblem(1025, 9, 3),
                                                      probeDevice());

    EXPECT_EQ(invocation.numWorkGroups.x, 5u);
    EXPECT_EQ(invocation.numWorkGroups.y, 3u);
    EXPECT_EQ(invocation.numWorkGroups.z, 3u);
    expectPerDimensionArgs(invocation, 0);
}

// With internal args the flattened NumWorkGroups folds the grid into x; the per-dimension
// kernargs must report that launched grid while NumWorkGroups keeps its meaning.
TEST(CustomKernelTest, NumWorkGroupsPerDimensionSeeTheFlattenedGrid)
{
    ContractionSolution solution;
    solution.internalArgsSupport.version = 1;

    std::vector<CustomArgSemantic> semantics = {CustomArgSemantic::NumWorkGroups};
    for(CustomArgSemantic semantic : perDimensionSemantics())
        semantics.push_back(semantic);

    KernelInvocation invocation = launchWithSemantics(solution,
                                                      longShortBatchGrid(),
                                                      longShortMacrotile(),
                                                      semantics,
                                                      longShortProblem(1025, 9, 3),
                                                      probeDevice());

    // The (5, 3, 3) tile grid is launched as 45 work groups along x.
    EXPECT_EQ(invocation.numWorkGroups.x, 45u);
    EXPECT_EQ(invocation.numWorkGroups.y, 1u);
    EXPECT_EQ(invocation.numWorkGroups.z, 1u);
    EXPECT_EQ(emittedUint32(invocation, 0), 45u);
    expectPerDimensionArgs(invocation, 1);
}

// Without a NumWorkGroups argument the grid is not flattened, even with internal args.
TEST(CustomKernelTest, NumWorkGroupsPerDimensionKeepTheGridUnflattened)
{
    ContractionSolution solution;
    solution.internalArgsSupport.version = 1;

    KernelInvocation invocation = launchWithSemantics(solution,
                                                      longShortBatchGrid(),
                                                      longShortMacrotile(),
                                                      perDimensionSemantics(),
                                                      longShortProblem(1025, 9, 3),
                                                      probeDevice());

    EXPECT_EQ(invocation.numWorkGroups.x, 5u);
    EXPECT_EQ(invocation.numWorkGroups.y, 3u);
    EXPECT_EQ(invocation.numWorkGroups.z, 3u);
    expectPerDimensionArgs(invocation, 0);
}

TEST(CustomKernelTest, NumWorkGroupsSemanticsRoundTripThroughStrings)
{
    for(std::string name : {"NumWorkGroupsX", "NumWorkGroupsY", "NumWorkGroupsZ"})
        EXPECT_EQ(toString(fromStringCustomArgSemantic(name)), name);
}

namespace
{
    constexpr char libraryCodeObject[] = "TensileLibrary_shard_gfx950.co";
    constexpr char customCodeObject[]  = "TensileLibrary_shard_Generated_gfx950.co";

    void configureCodeObjectProbe(ContractionSolution& solution, std::string const& customFile)
    {
        configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::SizeFree0});
        solution.sizeMapping.globalSplitU = 1;
        solution.codeObjectFilename       = std::string(libraryCodeObject);
        solution.customKernel.codeObjectFile = customFile;
    }

    std::string launchedCodeObject(ContractionSolution const& solution)
    {
        ContractionProblemGemm problem = dummyProblem();
        AMDGPU                 device  = probeDevice();
        ContractionInputs inputs;
        StreamKSettings   sk;

        return solution.generateCustomCall<false>(problem, inputs, device, sk).codeObjectFile;
    }
}

TEST(CustomKernelTest, CustomKernelDefaultsToTheLibraryCodeObject)
{
    ContractionSolution solution;
    configureCodeObjectProbe(solution, "");

    EXPECT_EQ(launchedCodeObject(solution), libraryCodeObject);
    EXPECT_EQ(solution.customKernelCodeObjectFile(), libraryCodeObject);
}

// Only the custom kernel's launch follows its own file; the library's helper kernels
// keep resolving in the library's code object.
TEST(CustomKernelTest, CustomKernelHonorsItsOwnCodeObject)
{
    ContractionSolution solution;
    configureCodeObjectProbe(solution, customCodeObject);

    EXPECT_EQ(launchedCodeObject(solution), customCodeObject);
    EXPECT_EQ(solution.customKernelCodeObjectFile(), customCodeObject);
    EXPECT_EQ(solution.codeObjectFilename.load(), libraryCodeObject);
}
