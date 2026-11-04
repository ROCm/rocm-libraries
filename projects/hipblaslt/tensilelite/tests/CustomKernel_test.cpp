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
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <Tensile/ContractionSolution.hpp>
#include <Tensile/ContractionTaskPredicates.hpp>
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

// Same extents as tileGridProblem, but C is transposed, which swaps the free extents back.
static ContractionProblemGemm transposedTileGridProblem(size_t m, size_t n, size_t batch)
{
    ContractionProblemGemm problem
        = ContractionProblemGemm::FromIndexSizes("Contraction_l_Ajlk_Blik_Cijk_Dijk",
                                                 {m, n, batch, tileGridK},
                                                 rocisa::DataType::Float,
                                                 {},
                                                 rocisa::DataType::Float,
                                                 {},
                                                 rocisa::DataType::Float,
                                                 {},
                                                 rocisa::DataType::Float,
                                                 {},
                                                 0.0);
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

static vector3<CustomGridSize> wideSkinny3dGrid()
{
    return {CustomGridSize::TilesWide, CustomGridSize::TilesSkinny, CustomGridSize::Batch};
}

static vector3<CustomGridSize> wideSkinnyFlatGrid()
{
    return {CustomGridSize::TilesWideSkinnyBatch, CustomGridSize::One, CustomGridSize::One};
}

// A probe kernel on the given grid and macrotile that reports the launched grid.
static void configureGridProbe(ContractionSolution&    solution,
                               vector3<CustomGridSize> grid,
                               TensileLite::dim3       macrotile)
{
    configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::NumWorkGroupsX});
    solution.customKernel.macrotile = macrotile;
    solution.customKernel.grid      = grid;
    solution.customKernel.args      = {{CustomArgType::uint32, CustomArgSemantic::NumWorkGroupsX},
                                  {CustomArgType::uint32, CustomArgSemantic::NumWorkGroupsY},
                                  {CustomArgType::uint32, CustomArgSemantic::NumWorkGroupsZ}};
}

// Launches a probe kernel on the skinny MXFP8 wide/skinny grid; the expected grids below were
// computed from the original name-derived emitter logic.
static KernelInvocation launchSkinnyEmitter(ContractionSolution&          solution,
                                            vector3<CustomGridSize>       grid,
                                            TensileLite::dim3             macrotile,
                                            ContractionProblemGemm const& problem)
{
    configureGridProbe(solution, grid, macrotile);

    AMDGPU            device = probeDevice();
    ContractionInputs inputs;
    StreamKSettings   sk;
    return solution.generateCustomCall<false>(problem, inputs, device, sk);
}

// A split-K launch on a tile-count grid would leave D unwritten, wide/skinny grids included.
TEST(CustomKernelTest, WideSkinnyGridRejectsSplitK)
{
    ContractionProblemGemm problem = tileGridProblem(1025, 9, 1);
    AMDGPU                 device  = probeDevice();
    ContractionInputs      inputs;
    StreamKSettings        sk;

    ContractionSolution solution;
    configureGridProbe(solution, wideSkinny3dGrid(), TensileLite::dim3(32, 16, tileGridK));
    solution.sizeMapping.globalSplitU = 1;
    EXPECT_NO_THROW(solution.generateCustomCall<false>(problem, inputs, device, sk));

    problem.setParams().setGSU(4);
    EXPECT_THROW(solution.generateCustomCall<false>(problem, inputs, device, sk),
                 std::runtime_error);
}

// Per-dimension NumWorkGroups kernargs must report the launched grid. Rows cover a 3-D
// grid in both axis orders, the wide/skinny grid in both orientations (one macrotile must
// launch the same grid whichever of m and n is skinny), a flat wide/skinny grid, the
// wide-block-column case, and the NumWorkGroups kernarg with and without internal args
// (flattening needs internalArgsSupport.version >= 1).
namespace
{
    struct NumWorkGroupsCase
    {
        char const*             name;
        vector3<CustomGridSize> grid;
        TensileLite::dim3       macrotile;
        size_t                  m, n, batch;
        bool                    withNumWorkGroups;
        int                     internalArgsVersion;
        uint32_t                x, y, z;
    };
}

TEST(CustomKernelTest, NumWorkGroupsPerDimensionReportTheLaunchedGrid)
{
    vector3<CustomGridSize> const tilesXY
        = {CustomGridSize::TilesX, CustomGridSize::TilesY, CustomGridSize::Batch};
    vector3<CustomGridSize> const tilesYX
        = {CustomGridSize::TilesY, CustomGridSize::TilesX, CustomGridSize::Batch};

    // The 1025 / 256 and 9 / 4 splits both leave a tail tile.
    std::vector<NumWorkGroupsCase> const cases = {
        {"tileGrid", tilesXY, {256, 4, tileGridK}, 1025, 9, 3, false, 1, 5, 3, 3},
        {"swappedTileGrid", tilesYX, {4, 256, tileGridK}, 9, 1025, 3, false, 1, 5, 3, 3},
        {"wideSkinnySkinnyN", wideSkinny3dGrid(), {256, 4, tileGridK}, 1025, 9, 3, false, 1, 5, 3, 3},
        {"wideSkinnySkinnyM", wideSkinny3dGrid(), {256, 4, tileGridK}, 9, 1025, 3, false, 1, 5, 3, 3},
        {"wideSkinnyFlat", wideSkinnyFlatGrid(), {32, 16, tileGridK}, 5120, 16, 1, false, 1, 160, 1, 1},
        {"wideSkinnyBlockColumns", wideSkinny3dGrid(), {16, 16, tileGridK}, 1, 10240, 1, false, 1, 640, 1, 1},
        {"unflattenedKernArgsVersion0", tilesXY, {256, 4, tileGridK}, 1025, 9, 7, true, 0, 5, 3, 7},
        {"flattenedKernArgsVersion1", tilesXY, {256, 4, tileGridK}, 1025, 9, 3, true, 1, 45, 1, 1},
    };

    for(NumWorkGroupsCase const& c : cases)
    {
        SCOPED_TRACE(c.name);
        ContractionSolution solution;
        solution.internalArgsSupport.version = c.internalArgsVersion;

        std::vector<CustomArgSemantic> semantics;
        if(c.withNumWorkGroups)
            semantics.push_back(CustomArgSemantic::NumWorkGroups);
        for(CustomArgSemantic semantic : perDimensionSemantics())
            semantics.push_back(semantic);

        KernelInvocation invocation = launchWithSemantics(
            solution, c.grid, c.macrotile, semantics, tileGridProblem(c.m, c.n, c.batch));

        EXPECT_EQ(invocation.numWorkGroups.x, c.x);
        EXPECT_EQ(invocation.numWorkGroups.y, c.y);
        EXPECT_EQ(invocation.numWorkGroups.z, c.z);
        expectPerDimensionArgs(invocation, c.withNumWorkGroups ? 1 : 0);
        // The NumWorkGroups kernarg keeps its own meaning: the flattened x extent.
        if(c.withNumWorkGroups && c.internalArgsVersion >= 1)
            EXPECT_EQ(emittedUint32(invocation, 0), c.x);
    }
}


// Spk is K rounded up to the declared k step, not to macrotile.z.
static uint32_t emittedSpk(size_t kStep, size_t k)
{
    ContractionSolution solution;
    auto problem = ContractionProblemGemm::GEMM(false, false, 5120, 2, k, 5120, k, 5120, 0.0, false, 1);
    problem.setComputeInputTypeA(rocisa::DataType::Float);
    problem.setComputeInputTypeB(rocisa::DataType::Float);

    configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::Spk});
    solution.customKernel.macrotile = TensileLite::dim3(8, 2, 4096);
    solution.customKernel.grid      = wideSkinny3dGrid();
    solution.customKernel.kStep     = kStep;

    AMDGPU            device = probeDevice();
    ContractionInputs inputs;
    StreamKSettings   sk;
    KernelInvocation invocation = solution.generateCustomCall<false>(problem, inputs, device, sk);
    return emittedUint32(invocation, 0);
}

TEST(CustomKernelTest, SpkRequiresKStep)
{
    EXPECT_THROW(emittedSpk(0, 5120), std::runtime_error);
}

// The host sizes the grid from the wide and skinny free extents, so a problem and
// its m <-> n swap must launch identically (equal extents are their own swap).
TEST(CustomKernelTest, WideSkinnyLaunchIsIndependentOfTheOrientation)
{
    for(vector3<CustomGridSize> const& grid : {wideSkinny3dGrid(), wideSkinnyFlatGrid()})
    {
        for(auto const& [wide, skinny] : std::vector<std::pair<size_t, size_t>>{
                {5120, 3}, {1025, 9}, {128256, 1}, {64, 64}})
        {
            ContractionSolution      solutionMN;
            ContractionSolution      solutionNM;
            ContractionProblemGemm   mn = tileGridProblem(wide, skinny, 2);
            ContractionProblemGemm   nm = tileGridProblem(skinny, wide, 2);
            TensileLite::dim3 const  macrotile(32, 16, 4096);

            KernelInvocation a = launchSkinnyEmitter(solutionMN, grid, macrotile, mn);
            KernelInvocation b = launchSkinnyEmitter(solutionNM, grid, macrotile, nm);

            SCOPED_TRACE("wide=" + std::to_string(wide) + " skinny=" + std::to_string(skinny));
            EXPECT_EQ(a.numWorkGroups.x, b.numWorkGroups.x);
            EXPECT_EQ(a.numWorkGroups.y, b.numWorkGroups.y);
            EXPECT_EQ(a.numWorkGroups.z, b.numWorkGroups.z);
            EXPECT_EQ(a.workGroupSize.x, b.workGroupSize.x);
            EXPECT_EQ(a.workGroupSize.y, b.workGroupSize.y);
            EXPECT_EQ(a.workGroupSize.z, b.workGroupSize.z);
            EXPECT_GT(a.numWorkGroups.x, 0u);
        }
    }
}

// calculateGrid (LaunchLimits) and the launch share the grid sizing; they must agree
// for both orientations and for a transposed C too.
TEST(CustomKernelTest, WideSkinnyLaunchLimitsAgreeWithTheLaunch)
{
    for(vector3<CustomGridSize> const& grid : {wideSkinny3dGrid(), wideSkinnyFlatGrid()})
    {
        for(auto const& [m, n] : std::vector<std::pair<size_t, size_t>>{{5120, 3}, {3, 5120}, {64, 64}})
        {
            for(bool transposed : {false, true})
            {
                SCOPED_TRACE("m=" + std::to_string(m) + " n=" + std::to_string(n)
                             + " transposed=" + std::to_string(transposed));
                ContractionSolution    solution;
                ContractionProblemGemm problem = transposed ? transposedTileGridProblem(m, n, 2)
                                                            : tileGridProblem(m, n, 2);
                ASSERT_EQ(problem.transposeC01(), transposed);
                KernelInvocation launch
                    = launchSkinnyEmitter(solution, grid, TensileLite::dim3(32, 16, 4096), problem);

                EXPECT_TRUE(solution.wideSkinnyGridObstacle().empty());
                TensileLite::dim3 workGroupSize;
                TensileLite::dim3 numWorkGroups;
                solution.calculateGrid(workGroupSize, numWorkGroups, problem);

                EXPECT_EQ(numWorkGroups.x, launch.numWorkGroups.x);
                EXPECT_EQ(numWorkGroups.y, launch.numWorkGroups.y);
                EXPECT_EQ(numWorkGroups.z, launch.numWorkGroups.z);
                EXPECT_EQ(workGroupSize.x, launch.workGroupSize.x);

                AMDGPU device = probeDevice();
                Task   task(device, problem, solution);
                EXPECT_TRUE(Predicates::Contraction::LaunchLimits()(task));
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Skinny MXFP8 launch sizing. The configurations below are representative grid
// configurations, not the shipped kernels' values (the Python metadata tests pin those
// from the .s files). The oracle is independent of the implementation:
//   tilesWide   = ceil(max(m, n) / macrotile[0])
//   tilesSkinny = ceil(min(m, n) / macrotile[1])
//   spk         = max(ceil(k / kStep), 1) * kStep
// ----------------------------------------------------------------------------

namespace
{
    struct SkinnyGridConfig
    {
        TensileLite::dim3       macrotile;
        size_t                  threads;
        vector3<CustomGridSize> grid;
        size_t                  kStep;
    };

    struct SkinnyShape
    {
        size_t m, n, k, batch;
    };

    size_t ceilDiv(size_t a, size_t b)
    {
        return (a + b - 1) / b;
    }

    void configureSkinnySolution(ContractionSolution&    solution,
                                 SkinnyGridConfig const& config,
                                 std::string const&      name)
    {
        configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::Spk});
        solution.customKernel.name      = name;
        solution.customKernel.macrotile = config.macrotile;
        solution.customKernel.threads   = TensileLite::dim3(config.threads, 1, 1);
        solution.customKernel.grid      = config.grid;
        solution.customKernel.kStep     = config.kStep;
        solution.customKernel.noSplitK  = true;
    }

    // C is either regular (A and B are column-major) or transposed (transposeC01).
    ContractionProblemGemm skinnyProblem(SkinnyShape const& s, bool transposed)
    {
        ContractionProblemGemm problem
            = transposed ? ContractionProblemGemm::FromIndexSizes("Contraction_l_Ajlk_Blik_Cijk_Dijk",
                                                                  {s.m, s.n, s.batch, s.k},
                                                                  rocisa::DataType::Float,
                                                                  {},
                                                                  rocisa::DataType::Float,
                                                                  {},
                                                                  rocisa::DataType::Float,
                                                                  {},
                                                                  rocisa::DataType::Float,
                                                                  {},
                                                                  0.0)
                         : ContractionProblemGemm::GEMM(
                             false, false, s.m, s.n, s.k, s.m, s.k, s.m, 0.0, false, s.batch);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        return problem;
    }

    void expectSkinnyLaunch(SkinnyGridConfig const& config,
                            SkinnyShape const&      s,
                            bool                    transposed)
    {
        SCOPED_TRACE("m=" + std::to_string(s.m) + " n=" + std::to_string(s.n)
                     + " k=" + std::to_string(s.k) + " batch=" + std::to_string(s.batch)
                     + " transposed=" + std::to_string(transposed));

        size_t tilesWide   = ceilDiv(std::max(s.m, s.n), config.macrotile.x);
        size_t tilesSkinny = ceilDiv(std::min(s.m, s.n), config.macrotile.y);
        size_t spk         = std::max<size_t>(ceilDiv(s.k, config.kStep), 1) * config.kStep;
        bool   flat        = config.grid.x == CustomGridSize::TilesWideSkinnyBatch;

        ContractionSolution solution;
        configureSkinnySolution(solution, config, "skinnyGridProbe");
        ContractionProblemGemm problem = skinnyProblem(s, transposed);
        ASSERT_EQ(problem.transposeC01(), transposed);

        size_t tw = 0;
        size_t ts = 0;
        size_t nb = 0;
        solution.calculateWideSkinnyTiles(tw, ts, nb, problem);
        EXPECT_EQ(tw, tilesWide);
        EXPECT_EQ(ts, tilesSkinny);
        EXPECT_EQ(nb, s.batch);

        AMDGPU            device = probeDevice();
        ContractionInputs inputs;
        StreamKSettings   sk;
        KernelInvocation  launch = solution.generateCustomCall<false>(problem, inputs, device, sk);

        EXPECT_EQ(launch.numWorkGroups.x, flat ? tilesWide * tilesSkinny * s.batch : tilesWide);
        EXPECT_EQ(launch.numWorkGroups.y, flat ? 1u : tilesSkinny);
        EXPECT_EQ(launch.numWorkGroups.z, flat ? 1u : s.batch);
        EXPECT_EQ(launch.workGroupSize.x, config.threads);
        EXPECT_EQ(launch.workGroupSize.y, 1u);
        EXPECT_EQ(launch.workGroupSize.z, 1u);
        EXPECT_EQ(emittedUint32(launch, 0), spk);

        // LaunchLimits sizes the grid through calculateGrid; it must agree.
        TensileLite::dim3 workGroupSize;
        TensileLite::dim3 numWorkGroups;
        solution.calculateGrid(workGroupSize, numWorkGroups, problem);
        EXPECT_EQ(numWorkGroups.x, launch.numWorkGroups.x);
        EXPECT_EQ(numWorkGroups.y, launch.numWorkGroups.y);
        EXPECT_EQ(numWorkGroups.z, launch.numWorkGroups.z);
        EXPECT_EQ(workGroupSize.x, config.threads);
    }
}

TEST(SkinnyMxfp8LaunchTest, GridWorkGroupSizeAndSpkFollowTheMetadata)
{
    std::vector<SkinnyGridConfig> const configs = {
        {TensileLite::dim3(16, 16, 4096), 128, wideSkinny3dGrid(), 256},
        {TensileLite::dim3(32, 16, 4096), 64, wideSkinny3dGrid(), 256},
        {TensileLite::dim3(48, 16, 4096), 64, wideSkinny3dGrid(), 256},
        {TensileLite::dim3(32, 16, 4096), 128, wideSkinnyFlatGrid(), 256},
        {TensileLite::dim3(8, 2, 4096), 64, wideSkinny3dGrid(), 1024},
        {TensileLite::dim3(8, 2, 4096), 64, wideSkinnyFlatGrid(), 1024},
    };
    // Each shape is run in both orientations; the extremes cover single-tile, partial-tile,
    // multi-batch, K not a multiple of kStep and a 2^20 wide extent.
    std::vector<SkinnyShape> const shapes = {
        {5120, 3, 5120, 1},
        {8191, 4, 2048, 1},
        {33, 17, 1024, 3},
        {29584, 1, 25600, 2},
        {48, 5, 5121, 1},
        {1048576, 2, 300, 1},
        {64, 64, 256, 2},
    };

    for(SkinnyGridConfig const& config : configs)
    {
        SCOPED_TRACE("macrotile.x=" + std::to_string(config.macrotile.x) + " macrotile.y="
                     + std::to_string(config.macrotile.y) + " threads="
                     + std::to_string(config.threads) + " kStep=" + std::to_string(config.kStep)
                     + " flat=" + std::to_string(config.grid.x
                                                 == CustomGridSize::TilesWideSkinnyBatch));
        for(SkinnyShape const& s : shapes)
        {
            for(bool transposed : {false, true})
            {
                expectSkinnyLaunch(config, s, transposed);
                expectSkinnyLaunch(config, {s.n, s.m, s.k, s.batch}, transposed);
            }
        }
    }
}


// A kernel without wide/skinny metadata keeps the generic grid sizing.
TEST(CustomKernelTest, GenericGridIsUnchangedWithoutWideSkinnyMetadata)
{
    ContractionSolution solution;
    configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::SizeFree0});
    solution.sizeMapping.workGroupSize = TensileLite::dim3(16, 16, 1);

    EXPECT_EQ(solution.customKernel.kStep, 0u);
    EXPECT_FALSE(solution.customKernel.noSplitK);

    ContractionProblemGemm problem = tileGridProblem(1025, 9, 3);

    TensileLite::dim3 workGroupSize;
    TensileLite::dim3 numWorkGroups;
    solution.calculateGrid(workGroupSize, numWorkGroups, problem);
    // sizeMapping.macroTile is 128 x 128.
    EXPECT_EQ(workGroupSize.x, 256u);
    EXPECT_EQ(numWorkGroups.x, 9u);
    EXPECT_EQ(numWorkGroups.y, 1u);
    EXPECT_EQ(numWorkGroups.z, 3u);

    AMDGPU            device = probeDevice();
    ContractionInputs inputs;
    StreamKSettings   sk;
    KernelInvocation  launch = solution.generateCustomCall<false>(problem, inputs, device, sk);
    EXPECT_EQ(launch.numWorkGroups.x, 9u);
    EXPECT_EQ(launch.numWorkGroups.y, 1u);
    EXPECT_EQ(launch.numWorkGroups.z, 1u);
    EXPECT_EQ(launch.workGroupSize.x, 256u);
}

// A generated kernel never takes the wide/skinny path, whatever its grid says.
TEST(CustomKernelTest, GeneratedKernelKeepsTheGenericCalculateGrid)
{
    ContractionSolution solution;
    configureProbeKernel(solution, {CustomArgType::uint32, CustomArgSemantic::SizeFree0});
    solution.sizeMapping.workGroupSize = TensileLite::dim3(16, 16, 1);
    solution.customKernel.generated    = true;
    solution.customKernel.threads      = TensileLite::dim3(64, 1, 1);
    solution.customKernel.grid = {CustomGridSize::TilesWide, CustomGridSize::TilesSkinny, CustomGridSize::Batch};

    TensileLite::dim3 workGroupSize;
    TensileLite::dim3 numWorkGroups;
    solution.calculateGrid(workGroupSize, numWorkGroups, tileGridProblem(1025, 9, 3));
    EXPECT_EQ(workGroupSize.x, 256u);
    EXPECT_EQ(numWorkGroups.x, 9u);
    EXPECT_EQ(numWorkGroups.y, 1u);
    EXPECT_EQ(numWorkGroups.z, 3u);
}

// TilesX and TilesY count tiles of an (m, n) macrotile, which a [wide, skinny] one does not
// describe, so the grid is rejected: selection must drop the solution without throwing, while
// the launch paths throw.
TEST(CustomKernelTest, MixedWideSkinnyGridIsRejectedWithoutAbortingSelection)
{
    for(vector3<CustomGridSize> const& grid :
        {vector3<CustomGridSize>{CustomGridSize::TilesWide, CustomGridSize::TilesXY, CustomGridSize::One},
         vector3<CustomGridSize>{CustomGridSize::TilesX, CustomGridSize::TilesWide, CustomGridSize::One},
         vector3<CustomGridSize>{CustomGridSize::TilesWide, CustomGridSize::One, CustomGridSize::TilesXY}})
    {
        ContractionSolution    solution;
        ContractionProblemGemm problem = tileGridProblem(1025, 9, 1);
        configureGridProbe(solution, grid, TensileLite::dim3(32, 16, tileGridK));

        EXPECT_NE(solution.wideSkinnyGridObstacle().find("combines wide/skinny tile counts"),
                  std::string::npos);

        AMDGPU device = probeDevice();
        Task   task(device, problem, solution);
        bool   accepted = true;
        EXPECT_NO_THROW(accepted = Predicates::Contraction::LaunchLimits()(task));
        EXPECT_FALSE(accepted);

        TensileLite::dim3 workGroupSize;
        TensileLite::dim3 numWorkGroups;
        std::string gridMessage;
        try
        {
            solution.calculateGrid(workGroupSize, numWorkGroups, problem);
        }
        catch(std::runtime_error const& error)
        {
            gridMessage = error.what();
        }
        EXPECT_NE(gridMessage.find("combines wide/skinny tile counts"), std::string::npos)
            << gridMessage;

        ContractionInputs inputs;
        StreamKSettings   sk;
        std::string       callMessage;
        try
        {
            solution.generateCustomCall<false>(problem, inputs, device, sk);
        }
        catch(std::runtime_error const& error)
        {
            callMessage = error.what();
        }
        EXPECT_NE(callMessage.find("combines wide/skinny tile counts"), std::string::npos)
            << callMessage;

        std::ostringstream stream;
        EXPECT_FALSE(Predicates::Contraction::LaunchLimits().debugEval(task, stream));
        EXPECT_NE(stream.str().find("combines wide/skinny tile counts"), std::string::npos)
            << stream.str();
    }
}

TEST(CustomKernelTest, WideSkinnyGridRejectsAZeroMacrotileOrThreadCount)
{
    // Zero in macrotile.x, macrotile.y or threads.x each divides by zero in the grid math.
    for(int32_t zeroed : {0, 1, 2})
    {
        SCOPED_TRACE("zeroed field " + std::to_string(zeroed));
        ContractionSolution solution;
        configureGridProbe(solution,
                           wideSkinny3dGrid(),
                           TensileLite::dim3(zeroed == 0 ? 0 : 32, zeroed == 1 ? 0 : 16, tileGridK));
        if(zeroed == 2)
            solution.customKernel.threads = TensileLite::dim3(0, 1, 1);

        EXPECT_NE(solution.wideSkinnyGridObstacle().find("zero macrotile or thread count"),
                  std::string::npos);

        AMDGPU                 device  = probeDevice();
        ContractionProblemGemm problem = tileGridProblem(1025, 9, 1);
        Task                   task(device, problem, solution);
        EXPECT_FALSE(Predicates::Contraction::LaunchLimits()(task));
    }
}

// Only handwritten kernels with a wide/skinny grid value have an obstacle to report.
TEST(CustomKernelTest, GridsWithoutWideSkinnyValuesHaveNoObstacle)
{
    ContractionSolution solution;
    configureGridProbe(solution,
                       {CustomGridSize::TilesX, CustomGridSize::TilesY, CustomGridSize::Batch},
                       TensileLite::dim3(0, 0, 0));
    EXPECT_TRUE(solution.wideSkinnyGridObstacle().empty());

    configureGridProbe(solution, wideSkinny3dGrid(), TensileLite::dim3(32, 16, tileGridK));
    solution.customKernel.generated = true;
    EXPECT_TRUE(solution.wideSkinnyGridObstacle().empty());
}

// The generic tile counts follow the output order whichever way C is transposed, and
// a packed batch is not a tile dimension.
TEST(CustomKernelTest, CalculateTilesFollowsTheMacrotileAndTheOutputOrder)
{
    ContractionSolution solution;
    configureGridProbe(solution,
                       {CustomGridSize::TilesX, CustomGridSize::TilesY, CustomGridSize::Batch},
                       TensileLite::dim3(256, 4, tileGridK));

    for(ContractionProblemGemm const& problem :
        {tileGridProblem(1025, 9, 3), transposedTileGridProblem(1025, 9, 3)})
    {
        TensileLite::dim3 tiles;
        solution.calculateTiles(tiles, problem);
        EXPECT_EQ(tiles.x, 5u);
        EXPECT_EQ(tiles.y, 3u);
        EXPECT_EQ(tiles.z, 3u);
    }

    solution.sizeMapping.packBatchDims = 1;
    TensileLite::dim3 tiles;
    solution.calculateTiles(tiles, tileGridProblem(1025, 9, 3));
    EXPECT_EQ(tiles.z, 1u);
}

// The kernels read Spk and the grid as unsigned 32-bit, so a value that does not fit must
// throw instead of truncating.
TEST(CustomKernelTest, SpkAtTheThirtyTwoBitLimit)
{
    constexpr size_t limit = size_t(1) << 32;
    EXPECT_EQ(emittedSpk(256, limit - 256), limit - 256);
    EXPECT_THROW(emittedSpk(256, limit - 255), std::runtime_error);
}

TEST(CustomKernelTest, NumWorkGroupsKernArgsThrowBeyondThirtyTwoBits)
{
    constexpr size_t over = size_t(1) << 32;
    struct Case
    {
        size_t      m, n, batch;
        char const* name;
    };
    for(Case const& c : {Case{over, 1, 1, "NumWorkGroupsX"},
                         Case{1, over, 1, "NumWorkGroupsY"},
                         Case{1, 1, over, "NumWorkGroupsZ"}})
    {
        ContractionSolution solution;
        std::string         message;
        try
        {
            launchWithSemantics(
                solution,
                {CustomGridSize::TilesX, CustomGridSize::TilesY, CustomGridSize::Batch},
                TensileLite::dim3(1, 1, tileGridK),
                perDimensionSemantics(),
                tileGridProblem(c.m, c.n, c.batch));
        }
        catch(std::runtime_error const& error)
        {
            message = error.what();
        }
        EXPECT_NE(message.find(std::string(c.name) + " 4294967296 does not fit"), std::string::npos)
            << c.name << ": " << message;
    }
}

// The largest launchable grid is 2^32 - 1 work-groups per dimension.
TEST(CustomKernelTest, NumWorkGroupsKernArgsAcceptTheThirtyTwoBitMaximum)
{
    constexpr size_t maxCount = (size_t(1) << 32) - 1;
    ContractionSolution solution;
    KernelInvocation    invocation = launchWithSemantics(
        solution,
        {CustomGridSize::TilesX, CustomGridSize::TilesY, CustomGridSize::Batch},
        TensileLite::dim3(1, 1, tileGridK),
        perDimensionSemantics(),
        tileGridProblem(maxCount, 1, 1));
    EXPECT_EQ(emittedUint32(invocation, 0), 4294967295u);
    EXPECT_EQ(invocation.numWorkGroups.x, 4294967295u);
}

// The byte strides must be computed in integers: a float product rounds to 24 bits.
static std::array<uint32_t, 4> emittedByteStrides(rocisa::DataType abType,
                                                  rocisa::DataType cdType,
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
    configureProbeKernel(solution, {CustomArgType::int32, CustomArgSemantic::StrideA0Bytes});
    solution.customKernel.args = {{CustomArgType::int32, CustomArgSemantic::StrideA0Bytes},
                                  {CustomArgType::int32, CustomArgSemantic::StrideB0Bytes},
                                  {CustomArgType::int32, CustomArgSemantic::StrideC0Bytes},
                                  {CustomArgType::int32, CustomArgSemantic::StrideD0Bytes}};

    AMDGPU            device = probeDevice();
    ContractionInputs inputs;
    StreamKSettings   sk;
    KernelInvocation  invocation = solution.generateCustomCall<false>(problem, inputs, device, sk);
    EXPECT_EQ(invocation.args.size(), 4 * sizeof(uint32_t));
    return {emittedUint32(invocation, 0),
            emittedUint32(invocation, 1),
            emittedUint32(invocation, 2),
            emittedUint32(invocation, 3)};
}

TEST(CustomKernelTest, ByteStridesAreExactBeyondTheFloatMantissa)
{
    // 2-byte C/D: 2 * 16777219 = 33554438, which a float cannot represent.
    std::array<uint32_t, 4> bf16 = emittedByteStrides(
        rocisa::DataType::Float8, rocisa::DataType::BFloat16, 64, 64, 16777219, 16777219);
    EXPECT_EQ(bf16[0], 64u);
    EXPECT_EQ(bf16[1], 64u);
    EXPECT_EQ(bf16[2], 33554438u);
    EXPECT_EQ(bf16[3], 33554438u);

    // 1-byte A/B: 2^26 + 4 needs 27 significant bits.
    constexpr size_t big = (size_t(1) << 26) + 4;
    std::array<uint32_t, 4> fp8
        = emittedByteStrides(rocisa::DataType::Float8, rocisa::DataType::BFloat16, big, big, 4, 4);
    EXPECT_EQ(fp8[0], 67108868u);
    EXPECT_EQ(fp8[1], 67108868u);
    EXPECT_EQ(fp8[2], 8u);
    EXPECT_EQ(fp8[3], 8u);
}

// A single-pass kernel has no split-K workspace and a launch-independent summation order;
// the same kernel without noSplitK needs both.
TEST(CustomKernelTest, NoSplitKSkipsTheWorkspaceAndUniformSummationOrderChecks)
{
    AMDGPU                 device  = probeDevice();
    ContractionProblemGemm problem = tileGridProblem(5120, 3, 1);
    problem.setBetaType(rocisa::DataType::Float);
    problem.setParams().setGSU(4);
    problem.setParams().setUniformSummationOrder(1);

    for(bool noSplitK : {true, false})
    {
        SCOPED_TRACE("noSplitK=" + std::to_string(noSplitK));
        ContractionSolution solution;
        configureGridProbe(solution, wideSkinny3dGrid(), TensileLite::dim3(32, 16, tileGridK));
        solution.sizeMapping.globalAccumulation = 1;
        solution.customKernel.noSplitK          = noSplitK;

        EXPECT_EQ(solution.requiredWorkspaceSize(problem, device) == 0, noSplitK);
        EXPECT_EQ(solution.uniformSummationOrderSupported(problem, device), noSplitK);
    }
}
