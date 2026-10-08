// Copyright (C) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Unit tests for the StreamK "launch summary" decision snapshot produced by
// ContractionSolution::computeStreamKDecisions(). This is the single source of
// truth solve() consumes to fill StreamKSettings, so asserting on it here is
// asserting on the real launch DECISIONS (mode / reduction / grid / tiles /
// split / workspace / partials / DP-only / fallbacks) made in the StreamK
// launch-parameter path -- without needing a GPU. Host-only: mock AMDGPU and
// hip::HipAMDGPU devices, no device library required.
//
// The partials-workspace behaviour asserted below: the dynamic work-queue path
// reserves one partial tile per part (skTiles*skSplit) when it splits tiles,
// and otherwise by tiles%grid divisibility.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

#include <Tensile/AMDGPU.hpp>
#include <Tensile/ContractionProblemProperties.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/ContractionTaskPredicates.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>
#include <origami/streamk.hpp>

#include "FallbackTestUtils.hpp"

using namespace TensileLite;
using namespace TensileLite::testing;

namespace
{
    constexpr size_t kGfx950AnalyticalCuCount = 256;

    // gfx950 analytical hardware advertising NUM_XCD=8 (matches the baked
    // per-XCD work-queue count), so the SK4 / SK5-dynamic work-stealing path is
    // supported. Mirrors makeGfx950AnalyticalHardware in CuCount_test.cpp.
    origami::hardware_t makeGfx950AnalyticalHardware()
    {
        using arch_t = origami::hardware_t::architecture_t;
        return origami::hardware_t(arch_t::gfx950,
                                   kGfx950AnalyticalCuCount,
                                   163840,
                                   262144,
                                   8, // NUM_XCD
                                   1.0,
                                   1.0,
                                   1.0,
                                   4000000,
                                   1.2,
                                   1,
                                   std::make_tuple(0.0, 0.008, 0.0));
    }

    hip::HipAMDGPU makeHipDeviceWithAnalytical(origami::hardware_t const& hw)
    {
        hip::HipAMDGPU device;
        device.processor          = AMDGPU::Processor::gfx950;
        device.computeUnitCount   = static_cast<int>(hw.N_CU);
        device.deviceName         = "test-gfx950-analytical";
        device.analyticalHardware = std::make_shared<origami::hardware_t>(hw);
        return device;
    }

    void initStreamKSolution(ContractionSolution& solution, int streamK)
    {
        solution.sizeMapping.tileProcessingStrategy = streamK == 0
            ? TensileLite::TileProcessingStrategy::None : TensileLite::TileProcessingStrategy::StreamK;
        solution.sizeMapping.workAssignment = (streamK == 4 ? TensileLite::WorkAssignment::DynamicWorkQueue : streamK == 5 ? TensileLite::WorkAssignment::Hybrid : TensileLite::WorkAssignment::StaticGrid);
        solution.sizeMapping.streamKAtomic         = 0;

        solution.sizeMapping.macroTile             = TensileLite::dim3(128, 128, 1);
        // SizeMapping's dim3 members have no default member initializer
        // (geom.hpp: vector3() = default with plain T x,y,z), so a
        // default-initialized ContractionSolution leaves workGroupSize
        // indeterminate. streamKUniformSummationOrderObstacle() reads
        // workGroupSize.z for its WaveSplitK check, so it must be set.
        solution.sizeMapping.workGroupSize         = TensileLite::dim3(256, 1, 1);
        solution.sizeMapping.threadTile            = TensileLite::dim3(1, 1, 1);
        solution.sizeMapping.depthU                = 64;
        solution.sizeMapping.matrixInstruction     = {16, 16, 32, 1};
        solution.sizeMapping.workGroupSize         = TensileLite::dim3(256, 1, 1);
        solution.sizeMapping.CUOccupancy           = 1;
        solution.sizeMapping.workspaceSizePerElemC = 4;
    }

    ContractionProblemGemm makeBatchedGemmProblem(size_t m, size_t n, size_t k, size_t batch)
    {
        auto problem
            = ContractionProblemGemm::GEMM(false, false, m, n, k, m, n, m, 1.0, false, batch);
        problem.setComputeInputTypeA(rocisa::DataType::Float);
        problem.setComputeInputTypeB(rocisa::DataType::Float);
        // Scheduling tests allow uneven K splits; USO-specific cases enable it explicitly.
        problem.setParams().setUniformSummationOrder(0);
        return problem;
    }

    ContractionProblemGemm makeGemmProblem(size_t m, size_t n, size_t k)
    {
        return makeBatchedGemmProblem(m, n, k, 1);
    }

    struct AnalyticalEnv
    {
        AnalyticalEnv()
            : hw(makeGfx950AnalyticalHardware())
            , device(makeHipDeviceWithAnalytical(hw))
        {
        }
        origami::hardware_t hw;
        hip::HipAMDGPU      device;
    };

    // The summary is a deeply-indented "key = value" block whose column widths are
    // chosen per-section for alignment. Collapsing runs of spaces to a single space
    // lets the assertions below match on the stable tokens ("changedBy = ...",
    // "source = ...", etc.) without pinning exact column widths. Newlines are left
    // intact so a token can never be matched across a line break.
    std::string collapseSpaces(std::string const& s)
    {
        std::string out;
        out.reserve(s.size());
        bool prevSpace = false;
        for(char c : s)
        {
            if(c == ' ')
            {
                if(!prevSpace)
                    out.push_back(' ');
                prevSpace = true;
            }
            else
            {
                out.push_back(c);
                prevSpace = false;
            }
        }
        return out;
    }

    // Sets (or clears) TENSILE_DB for the lifetime of the object and refreshes the
    // Debug singleton from it, restoring the PREVIOUS value -- not merely unsetting
    // -- on scope exit. Debug::Instance() is a process-wide function-local static
    // (see Singleton.hpp, LazySingleton), so an unrestored TENSILE_DB would leak
    // into every later test in this binary; the destructor still runs when an
    // ASSERT_* aborts the enclosing test, which a TearDown() could not be relied on
    // to do for a scoped state change.
    //
    // reloadDebugBitsForTest() refreshes exactly TENSILE_DB, TENSILE_DB2 and
    // TENSILE_STREAMK5_FORCE_MODE. It deliberately does NOT refresh
    // TENSILE_STREAMK_DATA_PARALLEL, so nothing here can toggle that flag.
    class ScopedTensileDb
    {
    public:
        // value == nullptr means "unset TENSILE_DB", i.e. the shipped default.
        explicit ScopedTensileDb(const char* value)
        {
            const char* prior = std::getenv("TENSILE_DB");
            m_had             = (prior != nullptr);
            if(m_had)
                m_saved = prior; // copy BEFORE setenv invalidates the pointer

            if(value)
                setenv("TENSILE_DB", value, /*overwrite=*/1);
            else
                unsetenv("TENSILE_DB");
            Debug::Instance().reloadDebugBitsForTest();
        }

        ~ScopedTensileDb()
        {
            if(m_had)
                setenv("TENSILE_DB", m_saved.c_str(), /*overwrite=*/1);
            else
                unsetenv("TENSILE_DB");
            Debug::Instance().reloadDebugBitsForTest();
        }

        ScopedTensileDb(ScopedTensileDb const&)            = delete;
        ScopedTensileDb& operator=(ScopedTensileDb const&) = delete;

    private:
        bool        m_had = false;
        std::string m_saved;
    };
} // namespace

TEST(StreamKLaunchSummaryTest, UniformSummationStaggerRespectsPersistentCapability)
{
    struct Case
    {
        TileProcessingStrategy strategy;
        bool                   perTile;
        bool                   runtimeStagger;
        size_t                 mapping;
        int                    uniformOrder;
        size_t                 expectedStagger;
    };
    const Case cases[] = {
        {TileProcessingStrategy::None, false, true, 1, 1, 16},
        {TileProcessingStrategy::DataParallel, false, true, 1, 1, 0},
        {TileProcessingStrategy::StreamK, false, true, 1, 1, 0},
        {TileProcessingStrategy::DataParallel, true, true, 1, 1, 16},
        {TileProcessingStrategy::StreamK, true, true, 1, 1, 16},
        {TileProcessingStrategy::StreamK, true, true, 0, 1, 0},
        {TileProcessingStrategy::None, false, false, 1, 1, 0},
        {TileProcessingStrategy::StreamK, false, true, 1, 0, 16},
    };
    AnalyticalEnv env;
    for(const auto& test : cases)
    {
        SCOPED_TRACE(::testing::Message() << toString(test.strategy) << " perTile=" << test.perTile
                                         << " runtimeStagger=" << test.runtimeStagger
                                         << " mapping=" << test.mapping
                                         << " uniformOrder=" << test.uniformOrder);
        ContractionSolution solution;
        initStreamKSolution(solution, 3);
        solution.sizeMapping.tileProcessingStrategy = test.strategy;
        solution.sizeMapping.workGroupMapping       = 1;
        solution.sizeMapping.workGroupMappingXCC    = 0;
        solution.sizeMapping.globalSplitU           = 1;
        solution.sizeMapping.globalAccumulation     = 4;
        solution.sizeMapping.staggerU               = 16;
        solution.sizeMapping.staggerUMapping        = test.mapping;
        solution.sizeMapping.staggerStrideShift     = 2;
        solution.internalArgsSupport.staggerU       = test.runtimeStagger;
        solution.internalArgsSupport.perTileExtraIters = test.perTile;
        solution.problemType.mxScaleFormat = 1;
        auto problem = makeGemmProblem(1024, 1024, 1024);
        problem.setParams().setUniformSummationOrder(test.uniformOrder);

        const auto [mapping, stagger, shift]
            = solution.calculateAutoStaggerU(problem, &env.device, 0, 1);
        EXPECT_EQ(stagger, test.expectedStagger);
        EXPECT_EQ(mapping, test.expectedStagger ? test.mapping : 0u);
        EXPECT_EQ(shift, test.expectedStagger ? 2u : 0u);
        if(test.uniformOrder)
            EXPECT_TRUE(solution.uniformSummationOrderSupported(problem, env.device));
    }
}

// ---------------------------------------------------------------------------
// No-drift contract: the snapshot fields equal what the individual production
// helpers report, so the summary reflects the REAL decisions (not a re-derivation
// that could drift). This is the property that makes the summary trustworthy.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, SnapshotMatchesHelpersForDynamicPartialTiles)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, 4); // SK4 = unconditionally dynamic (tree, work-queue)

    // 4096x4224 -> tiles = 32*33 = 1056; grid = min(1056, 256) = 256; 1056 % 256
    // != 0 -> partial tiles -> partials workspace required. K spans several
    // depthU iterations per tile, without which a workgroup's range would be a
    // whole number of tiles and no partials would exist to reserve for.
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto d = solution.computeStreamKDecisions(problem, env.device);

    const size_t tiles = problem.getNumTiles(solution.sizeMapping, 1);
    const auto   red   = solution.getSKReduction(problem, env.device);
    const size_t grid  = solution.getSKGrid(problem, env.device, tiles, red);

    EXPECT_TRUE(d.isDynamic);
    EXPECT_EQ(d.reduction, origami::reduction_t::tree);
    EXPECT_EQ(d.tiles, tiles);
    EXPECT_EQ(d.tiles, 1056u) << "4096/128 * 4224/128 = 32 * 33";
    // getSKGrid() reproduces the pre-fallback grid the snapshot records.
    EXPECT_EQ(d.skGridPreFallback, grid);
    EXPECT_EQ(d.skGridPreFallback, 256u) << "min(tiles, cuCount * CUOccupancy) = min(1056, 256)";
    ASSERT_NE(tiles % grid, 0u) << "test needs partial tiles";

    // No fallback fires here, so selected == pre-fallback == final launch grid.
    EXPECT_EQ(d.selectedGrid, d.skGridPreFallback);
    EXPECT_EQ(d.finalGrid, d.skGridPreFallback);
    EXPECT_EQ(d.skGrid, d.finalGrid);
    EXPECT_FALSE(d.workspaceDPFallbackFired);
    EXPECT_FALSE(d.treeBoundsFallbackFired);
    EXPECT_FALSE(d.fixedGridUsed);

    // The authoritative workspace query must agree with the snapshot byte-for-byte.
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, env.device));
    EXPECT_GT(d.requiredWorkspaceBytes, 0u);
    EXPECT_TRUE(d.workspaceAllocated);
    EXPECT_FALSE(d.dpOnly);
    EXPECT_EQ(d.numQueues, 8u) << "gfx950 bakes 8 per-XCD work queues (NUM_XCD)";
}

// ---------------------------------------------------------------------------
// SK5 resolves to the dynamic (SK4) sub-path when the API mode is ON, with tree
// reduction; SK5 OFF stays static (SK3) with no work-queue.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, Sk5OnResolvesDynamicTree)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, 5);

    // Several depthU iterations per tile, so the launch has partial tiles to
    // reserve for and the workspace assertions below have something to see.
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    problem.setParams().setStreamKTileSchedulingMode(1); // ON

    auto d = solution.computeStreamKDecisions(problem, env.device);

    EXPECT_TRUE(d.effectiveDynamic);
    EXPECT_TRUE(d.isDynamic);
    EXPECT_EQ(d.reduction, origami::reduction_t::tree);
    EXPECT_EQ(d.numQueues, 8u);
    // With plenty of workspace the partials block is taken (no DP fallback).
    EXPECT_TRUE(d.workspaceAllocated);
    EXPECT_FALSE(d.dpOnly);
}

TEST(StreamKLaunchSummaryTest, Sk5OffResolvesStaticSk3)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, 5);

    // Geometry choice matters: the k_split_aware grid selector (skDynamicGrid == 6,
    // AMDGPU's default) now collapses the grid to the tile count -- i.e. plain DP,
    // no partials -- whenever DP already fills the CUs efficiently or a tile is a
    // single k-iteration (see correct_sk_grid_for_partial_tiles in streamk.cpp).
    // To exercise the "SK5-OFF static path reserves a partial-tile workspace"
    // intent we need a geometry that KEEPS StreamK partials: depthU=64 with K=512
    // gives 8 iters/tile, and 2048x2176 -> 272 tiles over 256 CUs underfills DP
    // (2 waves, the 2nd nearly idle), so the selector keeps a sub-tile-count grid.
    auto problem = makeGemmProblem(2048, 2176, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    problem.setParams().setStreamKTileSchedulingMode(0); // OFF (static, smCountTarget=0)

    auto d = solution.computeStreamKDecisions(problem, env.device);

    EXPECT_FALSE(d.effectiveDynamic);
    EXPECT_FALSE(d.isDynamic) << "SK5-OFF must take the static (SK3) sub-path";
    EXPECT_EQ(d.numQueues, 8u); // baked count still reported (informational)
    // The grid must stay below the tile count, otherwise there are no partials and
    // the workspace sizing check below would silently assert nothing.
    ASSERT_NE(d.tiles % d.skGrid, 0u)
        << "geometry must produce partial tiles (grid < tiles) for this scenario";
    // SK3-static: no per-XCD work-queue region in the workspace it reserves.
    ASSERT_TRUE(d.workspaceAllocated)
        << "scenario must actually reserve a workspace, otherwise the sizing check below "
           "would silently assert nothing";
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.partialTileSize(d.skGrid))
        << "static SK3 workspace = partialTileSize(grid), no work-queue region";
}

// ---------------------------------------------------------------------------
// SK3 static with partial tiles: partials present, skTiles>0, workspace>0,
// not dynamic, not DP-only.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, Sk3StaticPartialTilesReserveWorkspace)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 3);

    // 4096x4224 -> tiles = 1056; grid = cuCount = 64; 1056 % 64 == 32 (!=0).
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;

    auto d = solution.computeStreamKDecisions(problem, device);

    EXPECT_FALSE(d.isDynamic);
    EXPECT_EQ(d.reduction, origami::reduction_t::tree);
    ASSERT_NE(d.tiles % d.skGrid, 0u) << "test needs partial tiles";
    EXPECT_TRUE(d.partialsPresent);
    EXPECT_GT(d.skTiles, 0u);
    EXPECT_TRUE(d.workspaceAllocated);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
    EXPECT_FALSE(d.dpOnly);
    EXPECT_EQ(d.numQueues, 0u) << "static SK3 mock device has no analytical work-queue count";
}

// itersPerTile == 1 is the whole-tile case: the iteration space Stream-K
// divides is tiles * 1, so a workgroup's range covers a whole number of tiles
// however the grid cuts it, and the kernel's store-partials and fixup paths are
// unreachable. No partial-tile workspace may be requested there, even though
// the grid is not a divisor of the tile count and the divisibility test alone
// would ask for one.
TEST(StreamKLaunchSummaryTest, Sk3WholeTileIterationReservesNoPartialsWorkspace)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 3);

    // 4096x4224 -> tiles = 1056; grid = cuCount = 64; 1056 % 64 == 32 (!=0).
    // K == depthU makes itersPerTile == 1 without disturbing either.
    auto problem = makeGemmProblem(4096, 4224, solution.sizeMapping.depthU);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device                  = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;

    auto d = solution.computeStreamKDecisions(problem, device);

    ASSERT_EQ(problem.getItersPerTile(solution.sizeMapping), 1u);
    ASSERT_EQ(d.reduction, origami::reduction_t::tree);
    ASSERT_NE(d.skGrid, d.tiles) << "case needs a grid that is not the tile count";
    ASSERT_NE(d.tiles % d.skGrid, 0u) << "case needs the divisibility test to fire";

    EXPECT_EQ(d.idealWorkspaceBytes, 0u);
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_FALSE(d.workspaceAllocated);
    // The query the caller sizes its allocation from must report the same.
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
}

// Complement: one more K iteration per tile and the reservation comes back, so
// the whole-tile skip cannot reach a launch that is able to stop inside a tile.
TEST(StreamKLaunchSummaryTest, Sk3TwoIterationTileStillReservesPartialsWorkspace)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 3);

    auto problem = makeGemmProblem(4096, 4224, solution.sizeMapping.depthU + 1);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device                  = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;

    auto d = solution.computeStreamKDecisions(problem, device);

    ASSERT_EQ(problem.getItersPerTile(solution.sizeMapping), 2u);
    ASSERT_NE(d.tiles % d.skGrid, 0u) << "case needs the divisibility test to fire";

    EXPECT_GT(d.requiredWorkspaceBytes, 0u);
    EXPECT_TRUE(d.workspaceAllocated);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
}

// With no partial tiles to hold, there is no workspace requirement for a given
// workspace to fall short of, so the workspace-insufficiency fallback to data
// parallel cannot fire and the launch keeps the grid that selection picked.
// This is the one launch behaviour the reservation rule carries with it.
TEST(StreamKLaunchSummaryTest, Sk3WholeTileIterationKeepsGridWithNoWorkspace)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 3);

    auto problem = makeGemmProblem(4096, 4224, solution.sizeMapping.depthU);
    problem.setWorkspaceSize(0);

    auto device                  = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;

    auto d = solution.computeStreamKDecisions(problem, device);

    ASSERT_EQ(problem.getItersPerTile(solution.sizeMapping), 1u);
    ASSERT_NE(d.skGridPreFallback, d.tiles) << "case needs a grid selection to preserve";

    EXPECT_FALSE(d.workspaceDPFallbackFired);
    EXPECT_EQ(d.skGrid, d.skGridPreFallback);
    EXPECT_FALSE(d.dpOnly);
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
}

namespace
{
    struct StreamKArgumentCase
    {
        const char* name;
        int mode;
        bool dynamic;
        bool parallel;
        std::array<uint32_t, 6> scheduling;
    };

    class StreamKArgumentLayoutTest
        : public ::testing::TestWithParam<std::tuple<int, StreamKArgumentCase>>
    {
    };
}

TEST_P(StreamKArgumentLayoutTest, PacksPartialWorkIntoKernelInvocation)
{
    auto const& [outerVersion, test] = GetParam();
    AnalyticalEnv env;
    ContractionSolution solution;
    initStreamKSolution(solution, test.mode);
    solution.kernelName = "generated_streamk_argument_contract";
    solution.customKernel.name = solution.kernelName;
    solution.customKernel.generated = true;
    solution.sizeMapping.globalSplitU = 0;
    solution.sizeMapping.globalAccumulation = 0;
    solution.sizeMapping.workGroupMapping = 1;
    solution.sizeMapping.workGroupMappingXCC = 1;
    solution.internalArgsSupport.version = outerVersion;
    solution.internalArgsSupport.persistentLoopArgsVersion = 0;
    solution.internalArgsSupport.useUniversalArgs = true;

    // Tree: 1056 tiles, eight K iterations each, grid 64. The two-tile
    // algorithm assigns 96 tiles to StreamK, giving each WG 12 K iterations.
    // Parallel: 64 tiles, 64 K iterations each, grid 256: four splits of 16.
    auto problem = test.parallel ? makeGemmProblem(256, 4096, 4096)
                                 : makeGemmProblem(4096, 4224, 512);
    problem.setAlphaType(rocisa::DataType::Float);
    problem.setBetaType(rocisa::DataType::Float);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    problem.setParams().setStreamKTileSchedulingMode(test.dynamic ? 1 : 0);
    env.device.persistentFixedGrid = test.parallel ? 256 : 64;
    env.device.persistentDynamicGrid = test.parallel
        ? static_cast<int>(origami::grid_selection_t::k_split_aware) : 0;
    env.device.skFullTiles = 1;
    // Dynamic: split 32 of the 1056 tiles into three items (3, 3, 2
    // iterations). This requires 1120 work items, including actual partials.
    env.device.skTiles = 32;
    env.device.skSplit = 3;
    ASSERT_FALSE(Debug::Instance().useStreamKDataParrallel());

    auto launch = solution.resolvePersistentSettings(problem, env.device);
    ASSERT_EQ(launch.tileProcessingStrategy, TileProcessingStrategy::StreamK);
    ASSERT_EQ(launch.reduction, test.parallel ? origami::reduction_t::parallel
                                              : origami::reduction_t::tree);
    ASSERT_EQ(launch.effectiveWorkAssignment, test.dynamic ? WorkAssignment::DynamicWorkQueue
                                                          : WorkAssignment::StaticGrid);
    ASSERT_EQ(launch.grid, test.parallel ? 256u : 64u);
    ASSERT_GT(launch.workspaceBytes, 0u);

    std::array<uint32_t, 4> workspace{}, flags{};
    ContractionInputs inputs;
    inputs.alpha = 1.0f;
    inputs.beta = 0.0f;
    inputs.ws = workspace.data();
    inputs.Synchronizer = flags.data();
    auto invocation = solution.generateSingleCall<true>(
        problem, inputs, env.device, launch, GSUSettings{});
    auto const& args = invocation.args;
    EXPECT_EQ(invocation.numWorkGroups.x, test.parallel ? 256u : 64u);
    EXPECT_EQ(invocation.numWorkGroups.y, 1u);
    EXPECT_EQ(invocation.numWorkGroups.z, 1u);
    EXPECT_EQ(KernelArguments::const_iterator(args, "PersistentGrid"), args.end());

    // Check the actual six transmitted words, not a recomputed launch summary.
    // Division by eight/64 uses multiply-high constants 2^29/2^26, shift zero.
    const char* firstName = test.mode == 3 ? "itersPerTile" : "ItersPerTile";
    auto first = KernelArguments::const_iterator(args, firstName);
    ASSERT_NE(first, args.end());
    auto start = static_cast<uint8_t const*>((*first).first);
    auto startOffset = start - static_cast<uint8_t const*>(args.data());
    ASSERT_GE(args.size(), startOffset + sizeof(test.scheduling));
    std::array<uint32_t, 6> packed{};
    std::memcpy(packed.data(), start, sizeof(packed));
    EXPECT_EQ(packed, test.scheduling);

    auto last = KernelArguments::const_iterator(args, test.dynamic ? "SKGrid" : "skTiles");
    ASSERT_NE(last, args.end());
    EXPECT_EQ(static_cast<uint8_t const*>((*last).first) - start, 20);
    if(outerVersion == 3)
    {
        auto alpha = KernelArguments::const_iterator(args, "alpha");
        ASSERT_NE(alpha, args.end());
        EXPECT_EQ(static_cast<uint8_t const*>((*alpha).first) - start, 24);
    }
    for(auto name : {"ws", "Flags"})
    {
        auto field = KernelArguments::const_iterator(args, name);
        ASSERT_NE(field, args.end()) << name;
        ASSERT_EQ((*field).second, sizeof(void*)) << name;
        void* pointer = nullptr;
        std::memcpy(&pointer, (*field).first, sizeof(pointer));
        void* expected = std::string(name) == "ws" ? inputs.ws
                       : test.parallel ? nullptr : inputs.Synchronizer;
        EXPECT_EQ(pointer, expected) << name;
    }
}

INSTANTIATE_TEST_SUITE_P(
    StreamKModesAndOuterVersions, StreamKArgumentLayoutTest,
    ::testing::Combine(
        ::testing::Values(0, 1, 2, 3),
        ::testing::Values(
            StreamKArgumentCase{"StaticTree", 3, false, false, {8, 0x20000000, 0, 12, 64, 96}},
            StreamKArgumentCase{"StaticParallel", 3, false, true, {64, 0x04000000, 0, 16, 256, 4}},
            StreamKArgumentCase{"Dynamic", 4, true, false, {8, 1120, 32, 3, 3, 64}},
            StreamKArgumentCase{"HybridStaticTree", 5, false, false, {8, 0x20000000, 0, 12, 64, 96}},
            StreamKArgumentCase{"HybridStaticParallel", 5, false, true, {64, 0x04000000, 0, 16, 256, 4}},
            StreamKArgumentCase{"HybridDynamic", 5, true, false, {8, 1120, 0x40000020, 3, 3, 64}})),
    [](::testing::TestParamInfo<StreamKArgumentLayoutTest::ParamType> const& info) {
        return "Outer" + std::to_string(std::get<0>(info.param)) + "_"
             + std::get<1>(info.param).name;
    });

// ---------------------------------------------------------------------------
// The dynamic work queues need a home workgroup for every non-empty queue
// (8 on gfx950). A fixed grid below that is raised to min(totalItems, 8), and
// the arg packers reject a grid that still leaves a queue unserved.
// ---------------------------------------------------------------------------
struct DynamicGridCase
{
    const char* name;
    int         mode; // 4 = DynamicWorkQueue, 5 = Hybrid (dynamic sub-mode)
    int         skTiles;
    int         skSplit;
    size_t      expectedGrid;
};

class StreamKDynamicGridCoversQueuesTest : public ::testing::TestWithParam<DynamicGridCase>
{
};

TEST_P(StreamKDynamicGridCoversQueuesTest, FixedGridIsRaisedToCoverEveryQueue)
{
    auto const&         test = GetParam();
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, test.mode);
    solution.kernelName                           = "generated_streamk_dynamic_grid";
    solution.customKernel.name                    = solution.kernelName;
    solution.customKernel.generated               = true;
    solution.sizeMapping.globalSplitU             = 0;
    solution.sizeMapping.globalAccumulation       = 0;
    solution.sizeMapping.workGroupMapping         = 1;
    solution.sizeMapping.workGroupMappingXCC      = 1;
    solution.internalArgsSupport.useUniversalArgs = true;

    // 4 tiles of 64 K iterations each.
    auto problem = makeGemmProblem(256, 256, 4096);
    problem.setAlphaType(rocisa::DataType::Float);
    problem.setBetaType(rocisa::DataType::Float);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    problem.setParams().setStreamKTileSchedulingMode(1);
    env.device.persistentFixedGrid   = 2;
    env.device.persistentDynamicGrid = 0;
    env.device.skTiles               = test.skTiles;
    env.device.skSplit               = test.skSplit;

    auto launch = solution.resolvePersistentSettings(problem, env.device);
    ASSERT_EQ(launch.effectiveWorkAssignment, WorkAssignment::DynamicWorkQueue);
    EXPECT_EQ(launch.grid, test.expectedGrid);

    std::array<uint32_t, 4> workspace{}, flags{};
    ContractionInputs       inputs;
    inputs.alpha        = 1.0f;
    inputs.beta         = 0.0f;
    inputs.ws           = workspace.data();
    inputs.Synchronizer = flags.data();
    EXPECT_NO_THROW(
        solution.generateSingleCall<true>(problem, inputs, env.device, launch, GSUSettings{}));

    // A grid that bypasses selection and leaves a queue unserved is rejected.
    launch.grid = 2;
    EXPECT_THROW(
        solution.generateSingleCall<true>(problem, inputs, env.device, launch, GSUSettings{}),
        std::runtime_error);
}

INSTANTIATE_TEST_SUITE_P(StreamKDynamicGrid,
                         StreamKDynamicGridCoversQueuesTest,
                         ::testing::Values(
                             // 4 tiles split 4 ways: 16 items, so all 8 queues are non-empty.
                             DynamicGridCase{"DynamicWorkQueueSplit", 4, 4, 4, 8},
                             DynamicGridCase{"HybridDynamicSplit", 5, 4, 4, 8},
                             // 3 whole tiles plus 1 tile split 2 ways: 5 items in 5 queues.
                             DynamicGridCase{"DynamicWorkQueueFewItems", 4, 1, 2, 5},
                             DynamicGridCase{"HybridDynamicFewItems", 5, 1, 2, 5}),
                         [](::testing::TestParamInfo<DynamicGridCase> const& info) {
                             return info.param.name;
                         });

// ---------------------------------------------------------------------------
// Force-DP-only (SK3): every tile stays data-parallel. skTiles==0, no partials,
// workspace==0, dpOnly reported and sourced from the compile-time PARAM.
// ---------------------------------------------------------------------------
TEST(PersistentLaunchSummaryTest, DataParallelHasNoReductionOrWorkspace)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 3);
    solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    auto device = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;
    auto launch = solution.resolvePersistentSettings(problem, device);
    EXPECT_EQ(launch.grid, static_cast<size_t>(_CPX_CU));
    EXPECT_EQ(launch.grid, launch.selectedGrid);
    EXPECT_EQ(launch.reduction, origami::reduction_t::none);
    EXPECT_EQ(launch.workspaceBytes, 0u);
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), 0u);
    EXPECT_EQ(launch.tileProcessingStrategy, TileProcessingStrategy::DataParallel);
    EXPECT_EQ(launch.workAssignment, WorkAssignment::StaticGrid);
    EXPECT_EQ(launch.effectiveWorkAssignment, WorkAssignment::StaticGrid);
}

// ---------------------------------------------------------------------------
// Workspace-starved SK4 falls back to a DP grid (grid=tiles, tree reduction):
// workspaceDPFallbackFired and dpOnly set, nothing reserved. This is the RUNTIME
// dp-only source, and it is the fallback that turns selectedGrid into finalGrid.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, WorkspaceDpFallbackFires)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, 4);

    // Several depthU iterations per tile, so the grid really does straddle
    // tiles and the launch wants a partials region it cannot have.
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(0); // no workspace at all -> must fall back to DP

    auto d = solution.computeStreamKDecisions(problem, env.device);

    ASSERT_NE(d.tiles % d.skGridPreFallback, 0u) << "test needs partial tiles pre-fallback";
    EXPECT_GT(d.idealWorkspaceBytes, 0u) << "the launch wanted a partials region";
    EXPECT_TRUE(d.workspaceDPFallbackFired);
    EXPECT_TRUE(d.dpOnly);
    // dp-only distinction: RUNTIME fallback, not the compile-time param.
    EXPECT_FALSE(d.forceDPOnly);
    // selected vs final: the fallback resets the grid to tiles.
    EXPECT_EQ(d.selectedGrid, d.skGridPreFallback) << "no tree-bounds fallback here";
    EXPECT_NE(d.selectedGrid, d.finalGrid) << "workspace-DP fallback changed the grid";
    EXPECT_EQ(d.finalGrid, d.tiles) << "DP fallback sets grid = tiles";
    EXPECT_EQ(d.skGrid, d.tiles);
    EXPECT_EQ(d.reduction, origami::reduction_t::tree);
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_FALSE(d.workspaceAllocated);
    // requiredWorkspaceSize agrees: nothing reserved when the workspace is too small.
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, env.device), 0u);
}

// ---------------------------------------------------------------------------
// Selected vs final grid + which-fallback attribution, exercised directly:
// the workspace-DP fallback is the mechanism that makes finalGrid != selectedGrid.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, SelectedVsFinalGridAttribution)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, 4);

    // Enough tiles for a real StreamK grid and several depthU iterations per
    // tile, but zero workspace forces the DP fallback that overwrites the
    // selected grid with tiles.
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(0);

    auto d = solution.computeStreamKDecisions(problem, env.device);

    // "selected" = what StreamK wanted (CU/config), "final" = what launched.
    EXPECT_GT(d.selectedGrid, 0u);
    EXPECT_EQ(d.finalGrid, d.tiles);
    EXPECT_NE(d.selectedGrid, d.finalGrid);
    // The workspace-DP fallback is the flag responsible for the change here, and
    // the fixed-grid override did not fire.
    EXPECT_TRUE(d.workspaceDPFallbackFired);
    EXPECT_FALSE(d.fixedGridUsed);
    EXPECT_FALSE(d.treeBoundsFallbackFired);
    EXPECT_FALSE(d.clusterDPGridClamped);
}

// ---------------------------------------------------------------------------
// DP-only source disambiguation: the snapshot distinguishes the compile-time
// PARAM (forceDPOnly) from the RUNTIME workspace-insufficient fallback. Both
// yield dpOnly, but only the runtime path sets workspaceDPFallbackFired and
// resets finalGrid to tiles.
// (The third source, the TENSILE_STREAMK_DATA_PARALLEL debug flag, is not
// toggleable in-process here: Debug caches it at construction and
// reloadDebugBitsForTest() intentionally does not refresh it. Its plumbing is
// still asserted-absent below via d.streamKDP.)
// ---------------------------------------------------------------------------
TEST(PersistentLaunchSummaryTest, CompiledDataParallelDiffersFromStreamKWorkspaceFallback)
{
    AnalyticalEnv env;
    env.device.persistentFixedGrid = 64;
    ContractionSolution dataParallel;
    initStreamKSolution(dataParallel, 3);
    dataParallel.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
    // Several depthU iterations per tile, so the StreamK half of the
    // comparison takes the workspace fallback it is here to demonstrate.
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(0);
    auto launch = dataParallel.resolvePersistentSettings(problem, env.device);
    EXPECT_EQ(launch.tileProcessingStrategy, TileProcessingStrategy::DataParallel);
    EXPECT_EQ(launch.reduction, origami::reduction_t::none);
    EXPECT_EQ(launch.workspaceBytes, 0u);
    EXPECT_NE(launch.grid, launch.totalTiles);

    ContractionSolution streamK;
    initStreamKSolution(streamK, 4);
    auto fallback = streamK.computeStreamKDecisions(problem, env.device);
    EXPECT_TRUE(fallback.dpOnly);
    EXPECT_TRUE(fallback.workspaceDPFallbackFired);
    EXPECT_EQ(fallback.finalGrid, fallback.tiles);
    EXPECT_TRUE(streamK.sizeMapping.isStreamK());
    EXPECT_FALSE(streamK.sizeMapping.isPersistentDataParallel());
}

// ---------------------------------------------------------------------------
// Dynamic-path partials-workspace reservation rule.
//
// A dynamic split (dynamicPartialsSlots > 0) reserves one partial tile per
// part, whatever the grid; otherwise the dynamic path (always tree) reserves
// iff tiles%grid!=0. The three tests below vary dynamicPartialsSlots (0 vs >0)
// against tiles%grid (==0 vs !=0); slots>0 with tiles%grid!=0 adds nothing.
//
// All three use a plain mock AMDGPU with no analyticalHardware, so
// streamKBakedQueueCount() is 0 and solve() would reject this device/solution pair
// at its dynamic-queue guard before ever sizing a workspace. Calling
// computeStreamKDecisions() directly is what lets the sizing rule be tested in
// isolation from that rejection.
// ---------------------------------------------------------------------------

// Case A: dynamicSlots == 0 (no split stream-k tiles) AND tiles % grid != 0.
// Workspace IS reserved (because tiles%grid!=0) even though the dynamic packing
// produced no partial tiles.
TEST(StreamKLaunchSummaryTest, DynamicNoSlotsButIndivisible_ReservesWorkspace)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 4); // dynamic

    // 4096x4224 -> tiles = 1056; grid = cuCount = 64; 1056 % 64 == 32 (!=0).
    // No skTiles override -> the dynamic packing yields skTiles == 0.
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;

    auto d = solution.computeStreamKDecisions(problem, device);

    ASSERT_TRUE(d.isDynamic);
    ASSERT_NE(d.tiles % d.skGrid, 0u) << "case needs tiles % grid != 0";
    EXPECT_EQ(d.skTiles, 0u) << "no override -> dynamic packing produces no split tiles";
    EXPECT_EQ(d.dynamicPartialsSlots, 0u) << "skTiles*skSplit == 0";

    // Workspace is reserved because tiles%grid != 0, independent of the zero
    // dynamic slot count.
    EXPECT_TRUE(d.workspaceAllocated);
    EXPECT_GT(d.requiredWorkspaceBytes, 0u);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
}

// Case B (complement): dynamicSlots == 0 AND tiles % grid == 0.
// No workspace reserved.
TEST(StreamKLaunchSummaryTest, DynamicNoSlotsAndDivisible_NoWorkspace)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 4); // dynamic

    // 4096x4096 -> tiles = 1024; grid = cuCount = 64; 1024 % 64 == 0.
    auto problem = makeGemmProblem(4096, 4096, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;

    auto d = solution.computeStreamKDecisions(problem, device);

    ASSERT_TRUE(d.isDynamic);
    ASSERT_EQ(d.tiles % d.skGrid, 0u) << "case needs tiles % grid == 0";
    EXPECT_EQ(d.dynamicPartialsSlots, 0u);
    EXPECT_FALSE(d.workspaceAllocated);
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
}

// Case C: dynamicSlots > 0 (skTiles override) AND tiles % grid == 0.
// Workspace IS reserved, one partial tile per part, even though tiles%grid==0.
TEST(StreamKLaunchSummaryTest, DynamicSlotsPositiveAndDivisible_ReservesPerPartWorkspace)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 4); // dynamic

    // 4096x4096 -> tiles = 1024; grid = cuCount = 64; 1024 % 64 == 0, so ONLY the
    // skTiles override creates partial (split) tiles.
    auto problem = makeGemmProblem(4096, 4096, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;
    device.skTiles       = 256; // override: number of split stream-k tiles
    device.skSplit       = 4; // override: k-split factor per tile

    auto d = solution.computeStreamKDecisions(problem, device);

    ASSERT_TRUE(d.isDynamic);
    ASSERT_EQ(d.tiles % d.skGrid, 0u) << "case needs tiles % grid == 0";
    EXPECT_EQ(d.skTiles, 256u);
    EXPECT_GE(d.skSplit, 1u);
    EXPECT_TRUE(d.partialsPresent) << "override produced split stream-k tiles";
    EXPECT_GT(d.dynamicPartialsSlots, 0u) << "skTiles*skSplit > 0";
    // totalItems = (tiles - skTiles) + skTiles*skSplit
    //            = (1024 - 256) + 256*4 = 1792.
    EXPECT_EQ(d.skSplit, 4u) << "itersPerTile=8, skSplit override 4 -> 4 work items per tile";
    EXPECT_EQ(d.totalItems, 1792u);

    // One partial tile per part despite tiles%grid==0; requiredWorkspaceSize agrees.
    EXPECT_TRUE(d.workspaceAllocated);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.partialTileSize(d.dynamicPartialsSlots));
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
}

// ---------------------------------------------------------------------------
// Non-StreamK solutions produce an inert (mode==0) snapshot.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, NonStreamKProducesInertSnapshot)
{
    ContractionSolution solution; // streamK defaults to 0
    auto                problem = makeGemmProblem(512, 512, 512);
    auto                device  = makeDevice(_MI350_CHIP_ID, _SPX_CU, "mi350spx");

    auto d = solution.computeStreamKDecisions(problem, device);
    EXPECT_FALSE(d.isDynamic);
    EXPECT_FALSE(d.dpOnly);
    EXPECT_EQ(d.skGrid, 0u);
    EXPECT_EQ(d.finalGrid, 0u);
    EXPECT_EQ(d.selectedGrid, 0u);
}

// ---------------------------------------------------------------------------
// The printed summary is well-formed and reports the key fields, including the
// selected-vs-final grid attribution. Exercises the formatting path.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, PrintSummaryEmitsFields)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    solution.kernelName = "test_streamk_kernel";
    initStreamKSolution(solution, 4);

    auto problem = makeGemmProblem(4096, 4224, 64);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto               d = solution.computeStreamKDecisions(problem, env.device);
    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());

    // Deeply-indented multi-line labeled block. The leading "LAUNCH SUMMARY" token
    // and the kernel name are emitted verbatim on the first line; the remaining
    // fields are aligned "key = value" pairs (whitespace-collapsed for matching).
    EXPECT_NE(line.find("LAUNCH SUMMARY"), std::string::npos);
    EXPECT_NE(line.find("test_streamk_kernel"), std::string::npos);
    EXPECT_NE(line.find("reduction = tree"), std::string::npos);
    // SK4 is unconditionally dynamic -> mode line reports it and the work-queue
    // line carries the real per-XCD counts (not NA).
    EXPECT_NE(line.find("isDynamic = yes"), std::string::npos);
    EXPECT_NE(line.find("selected = "), std::string::npos);
    EXPECT_NE(line.find("final = "), std::string::npos);
    EXPECT_NE(line.find("changedBy = "), std::string::npos);
    EXPECT_NE(line.find("source = "), std::string::npos);
    EXPECT_NE(line.find("numQueues(NUM_XCD) = 8"), std::string::npos);
    // Section headers are present on their own lines (multi-line block).
    EXPECT_NE(line.find("mode:"), std::string::npos);
    EXPECT_NE(line.find("grid:"), std::string::npos);
    EXPECT_NE(line.find("work-queue:"), std::string::npos);
}

TEST(StreamKLaunchSummaryTest, SummaryReportsNamedPolicyAndEffectiveAssignment)
{
    struct Case
    {
        TileProcessingStrategy strategy;
        WorkAssignment assignment;
        bool dynamic;
        const char* expectedPolicy;
    };
    const Case cases[] = {
        {TileProcessingStrategy::None, WorkAssignment::StaticGrid, false,
         "TileProcessingStrategy=None WorkAssignment=StaticGrid EffectiveWorkAssignment=StaticGrid"},
        {TileProcessingStrategy::DataParallel, WorkAssignment::StaticGrid, false,
         "TileProcessingStrategy=DataParallel WorkAssignment=StaticGrid EffectiveWorkAssignment=StaticGrid"},
        {TileProcessingStrategy::StreamK, WorkAssignment::StaticGrid, false,
         "TileProcessingStrategy=StreamK WorkAssignment=StaticGrid EffectiveWorkAssignment=StaticGrid"},
        {TileProcessingStrategy::StreamK, WorkAssignment::DynamicWorkQueue, true,
         "TileProcessingStrategy=StreamK WorkAssignment=DynamicWorkQueue EffectiveWorkAssignment=DynamicWorkQueue"},
        {TileProcessingStrategy::StreamK, WorkAssignment::Hybrid, false,
         "TileProcessingStrategy=StreamK WorkAssignment=Hybrid EffectiveWorkAssignment=StaticGrid"},
        {TileProcessingStrategy::StreamK, WorkAssignment::Hybrid, true,
         "TileProcessingStrategy=StreamK WorkAssignment=Hybrid EffectiveWorkAssignment=DynamicWorkQueue"},
    };
    auto problem = makeGemmProblem(4096, 4224, 64);
    for(auto const& test : cases)
    {
        SCOPED_TRACE(test.expectedPolicy);
        ContractionSolution solution;
        initStreamKSolution(solution, 3);
        solution.sizeMapping.tileProcessingStrategy = test.strategy;
        solution.sizeMapping.workAssignment = test.assignment;
        StreamKDecisions decisions;
        decisions.isDynamic = test.dynamic;
        std::ostringstream os;
        solution.printStreamKLaunchSummary(os, problem, decisions);
        const auto text = collapseSpaces(os.str());
        EXPECT_NE(text.find(test.expectedPolicy), std::string::npos);
        EXPECT_EQ(text.find("streamK ="), std::string::npos);
        EXPECT_EQ(text.find("SK3"), std::string::npos);
        EXPECT_EQ(text.find("SK4"), std::string::npos);
        EXPECT_EQ(text.find("SK5"), std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// The printed summary attributes the grid change to the workspace-DP fallback
// when it fires (selected vs final are both reported and differ).
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, PrintSummaryReportsFallbackGridChange)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    solution.kernelName = "test_streamk_fallback";
    initStreamKSolution(solution, 4);

    // Several depthU iterations per tile, so there are partials to be short of.
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(0); // force DP fallback

    auto               d = solution.computeStreamKDecisions(problem, env.device);
    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());

    EXPECT_NE(line.find("changedBy = workspaceDP"), std::string::npos);
    EXPECT_NE(line.find("source = workspaceDP(runtime)"), std::string::npos);
    EXPECT_NE(line.find("workspaceDPFallback = yes"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Work-queue fields are per-XCD dynamic-path only. On the SK5-static (SK3)
// sub-path (isDynamic == false) the summary must print the work-queue line as
// "NA (work-queues not used)" instead of a misleading numQueues value, while
// still reporting all the StreamK-wide fields (mode/grid/tiles/workspace).
// This is display-only: the struct still carries d.numQueues.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, PrintSummaryNaWorkQueueWhenNotDynamic)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    solution.kernelName = "test_streamk_static";
    initStreamKSolution(solution, 5);

    auto problem = makeGemmProblem(4096, 4224, 64);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    problem.setParams().setStreamKTileSchedulingMode(0); // OFF -> static (SK3), not dynamic

    auto d = solution.computeStreamKDecisions(problem, env.device);
    ASSERT_FALSE(d.isDynamic) << "SK5-OFF must resolve to the static (non-work-queue) path";
    // The struct still holds the baked count; only the DISPLAY is NA'd.
    EXPECT_EQ(d.numQueues, 8u);

    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());

    // Non-dynamic -> work-queue fields are NA, and the misleading numeric
    // per-XCD field is NOT printed. The "work-queue:" header is on its own line
    // with the NA note indented beneath it.
    EXPECT_NE(line.find("isDynamic = no"), std::string::npos);
    EXPECT_NE(line.find("work-queue:"), std::string::npos);
    EXPECT_NE(line.find("NA (work-queues not used)"), std::string::npos);
    EXPECT_EQ(line.find("numQueues(NUM_XCD)"), std::string::npos)
        << "static path must not print a per-XCD work-queue count";
    EXPECT_EQ(line.find("dynamicPartialsSlots"), std::string::npos)
        << "dynamicPartialsSlots is a dynamic-path-only field";
    // StreamK-wide fields are still reported (not NA'd).
    EXPECT_NE(line.find("reduction = "), std::string::npos);
    EXPECT_NE(line.find("selected = "), std::string::npos);
    EXPECT_NE(line.find("tiles:"), std::string::npos);
    EXPECT_NE(line.find("workspace:"), std::string::npos);
    EXPECT_NE(line.find("fallbacks:"), std::string::npos);
}

// ---------------------------------------------------------------------------
// DataParallel/StaticGrid spatial clusters walk whole Cs x Ck tile blocks, so
// the launch grid is the selected grid in whole clusters, with no reduction or
// partial workspace.
// ---------------------------------------------------------------------------
TEST(PersistentLaunchSummaryTest, ClusterDataParallelGridIsWholeClusters)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 3);
    solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
    solution.sizeMapping.clusterDim = TensileLite::dim3(2, 2, 1);
    auto problem = makeGemmProblem(4096, 4224, 512);
    auto device = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;
    auto launch = solution.resolvePersistentSettings(problem, device);
    EXPECT_FALSE(launch.clusterGridClamp);
    EXPECT_EQ(launch.selectedGrid, static_cast<size_t>(_CPX_CU));
    EXPECT_EQ(launch.totalTiles, 32u * 33u);
    EXPECT_EQ(launch.grid, static_cast<size_t>(_CPX_CU));
    EXPECT_EQ(launch.reduction, origami::reduction_t::none);
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, device), 0u);
}

// ---------------------------------------------------------------------------
// The selected grid rounds down to whole clusters, keeps at least one cluster,
// and never exceeds one cluster per tile block (16 x 17 blocks of 2 x 2 tiles).
// ---------------------------------------------------------------------------
TEST(PersistentLaunchSummaryTest, ClusterDataParallelGridRoundsToClustersAndBlocks)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 3);
    solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
    solution.sizeMapping.clusterDim = TensileLite::dim3(2, 2, 1);
    auto problem = makeGemmProblem(4096, 4224, 512);
    auto device = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;
    const std::vector<std::pair<int, size_t>> cases = {{30, 28u}, {1, 4u}, {5000, 16u * 17u * 4u}};
    for(auto [fixedGrid, expectedGrid] : cases)
    {
        device.persistentFixedGrid = fixedGrid;
        auto launch = solution.resolvePersistentSettings(problem, device);
        EXPECT_EQ(launch.grid, expectedGrid) << "fixed grid " << fixedGrid;
        EXPECT_FALSE(launch.clusterGridClamp);
    }
}

// ---------------------------------------------------------------------------
// PrefetchAcrossPersistent issues the next tile's multicast loads while cluster
// partners may still read LDS, so it keeps one cluster per tile block.
// ---------------------------------------------------------------------------
TEST(PersistentLaunchSummaryTest, ClusterDataParallelPrefetchAcrossPersistentUsesOneBlockPerCluster)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 3);
    solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
    solution.sizeMapping.clusterDim = TensileLite::dim3(2, 2, 1);
    solution.sizeMapping.prefetchAcrossPersistent = 1;
    auto problem = makeGemmProblem(4096, 4224, 512);
    auto device = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;
    auto launch = solution.resolvePersistentSettings(problem, device);
    EXPECT_TRUE(launch.clusterGridClamp);
    EXPECT_EQ(launch.grid, 16u * 17u * 4u);
}

// Padded peers count toward the kernel's 32-bit scheduling bound even when the
// actual tile count fits. No matrix storage is allocated for these host checks.
TEST(PersistentLaunchSummaryTest, ClusterDataParallelRejectsPaddedTileCountOverflow)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 3);
    solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
    solution.sizeMapping.clusterDim = TensileLite::dim3(2, 2, 1);
    auto device = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;
    device.persistentFixedGrid = 4;

    for(bool prefetch : {false, true})
    {
        SCOPED_TRACE(prefetch);
        solution.sizeMapping.prefetchAcrossPersistent = prefetch;
        // 65535 x 65535 real tiles become 65536 x 65536 padded tiles.
        auto overflow = makeGemmProblem(65535u * 128u, 65535u * 128u, 64);
        ASSERT_LT(overflow.getNumTiles(solution.sizeMapping, 1),
                  std::numeric_limits<uint32_t>::max());
        EXPECT_THROW(solution.resolvePersistentSettings(overflow, device), std::runtime_error);
        auto fitting = makeGemmProblem(65534u * 128u, 65535u * 128u, 64);
        EXPECT_NO_THROW(solution.resolvePersistentSettings(fitting, device));

        // One real tile per batch still consumes a complete four-peer block.
        auto batched = makeBatchedGemmProblem(128, 128, 64, size_t{1} << 30);
        EXPECT_THROW(solution.resolvePersistentSettings(batched, device), std::runtime_error);
    }
}

// Check the hardware dimensions and packed stride independently of each other:
// comparing generated and custom calls alone would miss a shared mapping error.
TEST(PersistentLaunchSummaryTest, ClusterDataParallelInvocationsUseWholeClusterRanks)
{
    for(auto cluster : {TensileLite::dim3(2, 1, 1), TensileLite::dim3(2, 2, 1),
                        TensileLite::dim3(2, 4, 1), TensileLite::dim3(1, 4, 1)})
    for(bool prefetch : {false, true})
    for(int selected : {1, 13, 1000})
    {
        SCOPED_TRACE(::testing::Message() << cluster.x << 'x' << cluster.y
                     << " prefetch=" << prefetch << " selected=" << selected);
        ContractionSolution solution;
        initStreamKSolution(solution, 3);
        solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
        solution.sizeMapping.clusterDim = cluster;
        solution.sizeMapping.prefetchAcrossPersistent = prefetch;
        solution.sizeMapping.globalSplitU = 0;
        solution.sizeMapping.workGroupMapping = 1;
        solution.sizeMapping.workGroupMappingXCC = 1;
        solution.internalArgsSupport.version = 3;
        solution.internalArgsSupport.persistentLoopArgsVersion = 1;
        solution.internalArgsSupport.useUniversalArgs = true;
        solution.customKernel.name = "cluster_data_parallel_probe";
        solution.customKernel.generated = true;
        solution.customKernel.macrotile = TensileLite::dim3(128, 128, 64);
        solution.customKernel.threads = TensileLite::dim3(256, 1, 1);
        solution.customKernel.grid = {CustomGridSize::PersistentGrid,
                                      CustomGridSize::One, CustomGridSize::One};
        solution.customKernel.args = {
            {CustomArgType::uint32, CustomArgSemantic::ItersPerTile},
            {CustomArgType::uint32, CustomArgSemantic::PersistentGrid}};
        // 5 x 3 real tiles in each of three batches: both spatial edges pad.
        auto problem = makeBatchedGemmProblem(513, 257, 129, 3);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        auto device = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
        device.persistentDynamicGrid = 0;
        device.persistentFixedGrid = selected;
        auto launch = solution.resolvePersistentSettings(problem, device);
        const size_t blockCount = cluster.x == 1 ? 15 : cluster.y == 1 ? 27 : cluster.y == 2 ? 18 : 9;
        const size_t clusterSize = cluster.x * cluster.y;
        const size_t clusters = prefetch ? blockCount
            : std::min(blockCount, std::max(size_t{1}, size_t(selected) / clusterSize));
        ASSERT_EQ(launch.grid, clusters * clusterSize);
        ASSERT_EQ(launch.totalTiles, 45u);
        ContractionInputs inputs;
        inputs.alpha = 1.0f;
        inputs.beta = 0.0f;
        auto generated = solution.generateSingleCall<true>(
            problem, inputs, device, launch, GSUSettings{});
        auto custom = solution.generateCustomCall<true>(problem, inputs, device, launch);
        for(auto const* invocation : {&generated, &custom})
        {
            EXPECT_EQ(invocation->numWorkGroups.x, clusters * cluster.x);
            EXPECT_EQ(invocation->numWorkGroups.y, cluster.y);
            EXPECT_EQ(invocation->numWorkGroups.z, 1u);
            auto arg = KernelArguments::const_iterator(invocation->args, "PersistentGrid");
            ASSERT_NE(arg, invocation->args.end());
            ASSERT_EQ((*arg).second, sizeof(uint32_t));
            uint32_t packedGrid = 0;
            std::memcpy(&packedGrid, (*arg).first, sizeof(packedGrid));
            EXPECT_EQ(packedGrid, clusters * clusterSize);
        }
    }
}

// ---------------------------------------------------------------------------
// Cluster grids require whole-tile processing and spatial cluster peers. An
// unsupported policy pair is rejected; partial StreamK and K-only clusters
// retain their selected grid.
// ---------------------------------------------------------------------------
TEST(PersistentLaunchSummaryTest, ClusterClampRequiresWholeTilesAndSpatialPeers)
{
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    auto device = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;
    ContractionSolution invalid;
    initStreamKSolution(invalid, 4);
    invalid.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
    invalid.sizeMapping.clusterDim = TensileLite::dim3(2, 2, 1);
    EXPECT_THROW(invalid.resolvePersistentSettings(problem, device), std::runtime_error);

    ContractionSolution streamK;
    initStreamKSolution(streamK, 3);
    streamK.sizeMapping.clusterDim = TensileLite::dim3(2, 2, 1);
    auto partial = streamK.computeStreamKDecisions(problem, device);
    EXPECT_FALSE(partial.clusterDPGridClamped);
    EXPECT_EQ(partial.finalGrid, partial.selectedGrid);

    ContractionSolution dataParallel;
    initStreamKSolution(dataParallel, 3);
    dataParallel.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
    dataParallel.sizeMapping.clusterDim = TensileLite::dim3(1, 1, 8);
    auto launch = dataParallel.resolvePersistentSettings(problem, device);
    EXPECT_FALSE(launch.clusterGridClamp);
    EXPECT_EQ(launch.grid, static_cast<size_t>(_CPX_CU));
}

// ---------------------------------------------------------------------------
// The printed summary uses canonical selectors, the final persistent grid,
// and zero workspace for both clustered and ordinary DataParallel launches.
// ---------------------------------------------------------------------------
TEST(PersistentLaunchSummaryTest, DataParallelSummaryUsesCanonicalFields)
{
    auto problem = makeGemmProblem(4096, 4224, 512);
    auto device = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;
    for(bool clustered : {false, true})
    {
        ContractionSolution solution;
        initStreamKSolution(solution, 3);
        solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
        solution.sizeMapping.clusterDim = clustered ? TensileLite::dim3(2, 2, 1) : TensileLite::dim3(1, 1, 1);
        auto launch = solution.resolvePersistentSettings(problem, device);
        std::ostringstream output;
        solution.printPersistentLaunchSummary(output, problem, launch);
        auto text = collapseSpaces(output.str());
        EXPECT_NE(text.find("TileProcessingStrategy=DataParallel"), std::string::npos);
        EXPECT_NE(text.find("WorkAssignment=StaticGrid"), std::string::npos);
        EXPECT_NE(text.find("WorkspaceBytes=0"), std::string::npos);
        EXPECT_NE(text.find("PersistentGrid=" + std::to_string(launch.grid)), std::string::npos);
        EXPECT_EQ(text.find("StreamK"), std::string::npos);
        EXPECT_EQ(text.find("forceDPOnly"), std::string::npos);
        EXPECT_FALSE(launch.clusterGridClamp);
    }
}

// ---------------------------------------------------------------------------
// A fixed-grid override selects 32 workgroups, which is already a whole number
// of 2 x 2 clusters, so the spatial cluster launches it as is. The summary
// exposes both values.
// ---------------------------------------------------------------------------
TEST(PersistentLaunchSummaryTest, SpatialClusterKeepsWholeClusterFixedGrid)
{
    ContractionSolution solution;
    initStreamKSolution(solution, 3);
    solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
    solution.sizeMapping.clusterDim = TensileLite::dim3(2, 2, 1);
    auto problem = makeGemmProblem(4096, 4224, 512);
    auto device = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;
    device.persistentFixedGrid = 32;
    auto launch = solution.resolvePersistentSettings(problem, device);
    EXPECT_EQ(launch.selectedGrid, 32u);
    EXPECT_FALSE(launch.clusterGridClamp);
    EXPECT_EQ(launch.grid, 32u);
    std::ostringstream output;
    solution.printPersistentLaunchSummary(output, problem, launch);
    const auto text = output.str();
    EXPECT_NE(text.find("ClusterGridClamp="), std::string::npos);
    EXPECT_NE(text.find("SelectedGrid=32"), std::string::npos);
}

// ---------------------------------------------------------------------------
// The parallel (split-K data-parallel) reduction path.
//
// The other StreamK tests above run tree-reduction launches. getSKReduction()
// delegates to origami's select_reduction() whenever the device's persistentDynamicGrid is
// k_split_aware (== 6, the AMDGPU default), and select_reduction returns parallel
// only on the narrow band "tiles < cuCount && itersPerTile >= 64 &&
// tiles <= cuCount/4". 256x4096x4096 with a 128x128x64 macro tile lands exactly
// there on the 256-CU analytical mock: tiles = 2*32 = 64 == cuCount/4, and
// itersPerTile = 4096/64 = 64. SK4 and SK5-dynamic are unconditionally tree, so
// this has to be SK3.
//
// Parallel is the one reduction that reserves a partials workspace unconditionally
// (the tree path only reserves when tiles % grid != 0). The snapshot and the
// caller-facing requiredWorkspaceSize() both size that reservation with
// partialTileSize(finalGrid); see the comment on
// StreamKDecisions::requiredWorkspaceBytes.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, Sk3ParallelReductionReservesPartialsWorkspace)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    solution.kernelName = "test_streamk_parallel";
    initStreamKSolution(solution, 3); // SK3 static; SK4 is unconditionally tree

    auto problem = makeGemmProblem(256, 4096, 4096);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    // TENSILE_STREAMK_DATA_PARALLEL is latched when Debug is constructed and is not
    // among the variables reloadDebugBitsForTest() refreshes, so it can only be
    // cleared before the process starts. Fail loudly rather than silently asserting
    // something else if it is set.
    ASSERT_FALSE(Debug::Instance().useStreamKDataParrallel())
        << "unset TENSILE_STREAMK_DATA_PARALLEL before running this suite";

    auto d = solution.computeStreamKDecisions(problem, env.device);

    // Anti-vacuity: the whole point of this test is the parallel branch.
    ASSERT_EQ(d.reduction, origami::reduction_t::parallel)
        << "scenario must actually select parallel reduction, otherwise every "
           "assertion below is about the already-covered tree path";
    // ...and it came from the real helper, not from a special case in the snapshot.
    EXPECT_EQ(solution.getSKReduction(problem, env.device), origami::reduction_t::parallel);

    EXPECT_FALSE(d.isDynamic) << "isDynamic implies tree reduction";
    EXPECT_EQ(d.tiles, 64u) << "256/128 * 4096/128 = 2 * 32";
    EXPECT_EQ(problem.getItersPerTile(solution.sizeMapping), 64u)
        << "4096/64; select_reduction needs itersPerTile >= 64";

    // No clamp fires: selected == preFallback == final.
    EXPECT_EQ(d.selectedGrid, 256u);
    EXPECT_EQ(d.skGridPreFallback, 256u);
    EXPECT_EQ(d.finalGrid, 256u);
    EXPECT_EQ(d.skGrid, d.finalGrid);
    EXPECT_EQ(solution.getSKGrid(problem, env.device, d.tiles, d.reduction), d.finalGrid);
    EXPECT_FALSE(d.fixedGridUsed);
    EXPECT_FALSE(d.treeBoundsFallbackFired) << "tree-bounds fixup is tree-reduction only";
    EXPECT_FALSE(d.clusterDPGridClamped);
    EXPECT_FALSE(d.workspaceDPFallbackFired);

    // Parallel packing: skSplit = grid/tiles, and skTiles mirrors skSplit.
    EXPECT_EQ(d.skSplit, 4u) << "grid/tiles = 256/64";
    EXPECT_EQ(d.skTiles, d.skSplit) << "the parallel path packs skTiles = skSplit";
    EXPECT_EQ(d.totalItems, d.tiles);
    EXPECT_TRUE(d.partialsPresent);
    EXPECT_FALSE(d.dpOnly);
    EXPECT_FALSE(d.forceDPOnly);
    EXPECT_FALSE(d.streamKDP);

    // Parallel reduction always reserves partials, sized by the FINAL grid.
    ASSERT_TRUE(d.workspaceAllocated)
        << "parallel reduction must reserve a partials region, otherwise the sizing "
           "checks below assert nothing";
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.partialTileSize(d.finalGrid));
    EXPECT_EQ(d.requiredWorkspaceBytes, 16777216u) << "128*128*4 bytes * 256 work-groups";
    EXPECT_EQ(d.idealWorkspaceBytes, d.requiredWorkspaceBytes) << "it fits, so nothing was trimmed";

    // The two independent sizings land on the same byte count.
    EXPECT_EQ(d.finalGrid, d.tiles * d.skSplit);
    EXPECT_GT(solution.requiredWorkspaceSize(problem, env.device), 0u);
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, env.device), d.requiredWorkspaceBytes);

    // The report names the reduction and attributes nothing to a fallback.
    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());
    EXPECT_NE(line.find("reduction = parallel(DP)"), std::string::npos);
    EXPECT_NE(line.find("changedBy = none"), std::string::npos);
    EXPECT_EQ(line.find("preFallback"), std::string::npos) << "nothing moved the grid";
    EXPECT_NE(line.find("isDynamic = no"), std::string::npos);
    EXPECT_NE(line.find("NA (work-queues not used)"), std::string::npos);
}

TEST(StreamKLaunchSummaryTest, GeneratedKernelNameKeepsOrigamiReduction)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, 3);
    env.device.persistentDynamicGrid = static_cast<int>(origami::grid_selection_t::k_split_aware);

    // This shape selects parallel reduction on the 256-CU analytical device.
    auto problem = makeGemmProblem(256, 4096, 4096);
    ASSERT_EQ(solution.getSKReduction(problem, env.device), origami::reduction_t::parallel);

    solution.customKernel.name      = "generated_streamk";
    solution.customKernel.generated = true;
    EXPECT_EQ(solution.getSKReduction(problem, env.device), origami::reduction_t::parallel);

    solution.customKernel.generated = false;
    EXPECT_EQ(solution.getSKReduction(problem, env.device), origami::reduction_t::tree);

    solution.customKernel.generated             = true;
    solution.sizeMapping.tileProcessingStrategy = TileProcessingStrategy::DataParallel;
    EXPECT_EQ(solution.getSKReduction(problem, env.device), origami::reduction_t::none);
}

// ---------------------------------------------------------------------------
// The batched parallel-reduction scenario, which is what separates the two
// sizings. getNumTiles() folds the batch count into its result for every
// streamK != 0 solution, so the batch is already carried by tiles, by the grid
// derived from it, and therefore by partialTileSize(grid). A sizing that
// multiplied by the batch a second time would over-report by exactly the batch
// count, push the reported requirement past the caller's workspace cap, and
// leave the launch running data-parallel on a zero-byte allocation.
//
// 128x128x4096 keeps one tile per batch entry, so batch entries and tiles are
// the same count and select_reduction's "tiles <= cuCount/4" band still admits
// parallel at a batch of 8.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, Sk3ParallelBatchedWorkspaceIsNotBatchScaled)
{
    constexpr size_t kBatch = 8;

    AnalyticalEnv       env;
    ContractionSolution solution;
    solution.kernelName = "test_streamk_parallel_batched";
    initStreamKSolution(solution, 3);

    auto problem = makeBatchedGemmProblem(128, 128, 4096, kBatch);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto d = solution.computeStreamKDecisions(problem, env.device);

    ASSERT_EQ(d.reduction, origami::reduction_t::parallel)
        << "scenario must actually select parallel reduction, otherwise the "
           "assertions below are about the already-covered tree path";
    ASSERT_EQ(d.tiles, kBatch) << "one 128x128 tile per batch entry";
    EXPECT_EQ(problem.getNumTiles(solution.sizeMapping, 1), kBatch)
        << "getNumTiles already folds the batch count in for streamK != 0";
    ASSERT_TRUE(d.workspaceAllocated);

    // Both sizings are partialTileSize(finalGrid); neither re-applies the batch.
    const size_t expected = solution.partialTileSize(d.finalGrid);
    EXPECT_EQ(d.requiredWorkspaceBytes, expected);
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, env.device), expected)
        << "the query must not scale the partials region by the batch count";

    // A budget that fits the partials exactly must survive the query followed
    // by allocation. The old extra batch factor made this query return zero.
    ASSERT_GT(expected, 0u);
    problem.setWorkspaceSize(expected);
    const size_t allocated = solution.requiredWorkspaceSize(problem, env.device);
    ASSERT_EQ(allocated, expected);
    problem.setWorkspaceSize(allocated);
    const auto launch = solution.computeStreamKDecisions(problem, env.device);
    EXPECT_EQ(launch.reduction, origami::reduction_t::parallel);
    EXPECT_EQ(launch.finalGrid, d.finalGrid);
    EXPECT_EQ(launch.requiredWorkspaceBytes, allocated);
    EXPECT_FALSE(launch.workspaceDPFallbackFired);

    // Hold the grid fixed to test the fit boundary itself: without this
    // override the grid selector can choose a smaller split that still fits.
    env.device.persistentFixedGrid = d.finalGrid;
    problem.setWorkspaceSize(expected - 1);
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, env.device), 0u);
    const auto starved = solution.computeStreamKDecisions(problem, env.device);
    ASSERT_TRUE(starved.fixedGridUsed);
    EXPECT_EQ(starved.idealWorkspaceBytes, expected);
    EXPECT_TRUE(starved.workspaceDPFallbackFired);
    EXPECT_EQ(starved.reduction, origami::reduction_t::tree);
    EXPECT_EQ(starved.finalGrid, starved.tiles);
    EXPECT_EQ(starved.requiredWorkspaceBytes, 0u);
}

// Preserve every auxiliary allocation from the old GSU-based query while
// counting StreamK partial tiles once. Fixed grids isolate the sizing rules
// from the workspace-aware grid selector. These are host sizing tests, not
// evidence that the corresponding auxiliary GPU kernels support StreamK.
TEST(StreamKLaunchSummaryTest, Sk3ParallelWorkspacePreservesAuxiliaryAllocations)
{
    using Tensor = ContractionProblemGemm::TENSOR;
    struct Case
    {
        const char* name;
        bool        gradient;
        Tensor      biasSrc;
        bool        biasEnabled;
        bool        amax;
        size_t      biasBytes;
    };
    const Case cases[] = {
        {"bias A", true, Tensor::A, true, false, 128 * 4 * 4},
        {"bias B", true, Tensor::B, true, false, 256 * 4 * 4},
        {"bias D uses partials", true, Tensor::D, true, false, 0},
        {"forward bias", false, Tensor::D, true, false, 0},
        {"disabled bias", true, Tensor::A, false, false, 0},
        {"amax", false, Tensor::D, false, true, 0},
        {"bias and amax", true, Tensor::A, true, true, 128 * 4 * 4},
    };
    for(size_t batch : {1u, 8u})
    {
        for(const auto& test : cases)
        {
            SCOPED_TRACE(test.name);
            SCOPED_TRACE(batch);
            AnalyticalEnv       env;
            ContractionSolution solution;
            initStreamKSolution(solution, 3);
            solution.problemType.useBias                  = 1;
            solution.problemType.useGradient              = test.gradient;
            solution.problemType.outputAmaxD              = test.amax;
            solution.sizeMapping.workspaceSizePerElemBias = 4;

            auto problem = makeBatchedGemmProblem(128, 256, 4096, batch);
            problem.setUseBias(1);
            problem.setUseGradient(test.gradient);
            problem.setBias(test.biasEnabled ? rocisa::DataType::Float : rocisa::DataType::None,
                            test.biasSrc == Tensor::B ? 256 : 128,
                            0, test.gradient, test.biasSrc);
            problem.setOutputAmaxD(test.amax);
            problem.setAmaxD(rocisa::DataType::Float, true);
            problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
            const size_t tiles = 2 * batch;
            env.device.persistentFixedGrid = tiles * 4;
            const auto decisions = solution.computeStreamKDecisions(problem, env.device);
            ASSERT_EQ(decisions.reduction, origami::reduction_t::parallel);
            ASSERT_EQ(decisions.finalGrid, tiles * 4);

            const size_t partials = 128 * 128 * 4 * tiles * 4;
            const size_t expected = partials + test.biasBytes + (test.amax ? tiles * 4 : 0);
            EXPECT_EQ(solution.requiredWorkspaceSize(problem, env.device), expected);
            // The previous query differs only by its extra batch factor on partials.
            EXPECT_EQ(solution.requiredWorkspaceSizeGsu(problem, env.device, 4),
                      expected + partials * (batch - 1));

            problem.setWorkspaceSize(expected);
            const size_t allocated = solution.requiredWorkspaceSize(problem, env.device);
            ASSERT_EQ(allocated, expected);
            problem.setWorkspaceSize(allocated);
            const auto launch = solution.computeStreamKDecisions(problem, env.device);
            EXPECT_EQ(launch.reduction, origami::reduction_t::parallel);
            EXPECT_FALSE(launch.workspaceDPFallbackFired);
            EXPECT_EQ(launch.requiredWorkspaceBytes, partials);

            problem.setWorkspaceSize(expected - 1);
            EXPECT_EQ(solution.requiredWorkspaceSize(problem, env.device), 0u);
        }
    }
}

TEST(StreamKLaunchSummaryTest, GsuWorkspacePreservesCustomMetadata)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, 0);
    solution.problemType.useBias                  = 1;
    solution.problemType.useGradient              = true;
    solution.sizeMapping.workspaceSizePerElemBias = 4;
    solution.customKernel.name                    = "workspace_probe";
    solution.customKernel.workspaceType           = CustomWorkspaceType::StreamKWithReduction;
    solution.customKernel.macrotile               = TensileLite::dim3(128, 128, 64);
    solution.customKernel.workspaceSizePerElemC    = 8;
    solution.customKernel.workspaceSizePerElemBias = 8;
    auto problem = makeBatchedGemmProblem(128, 128, 4096, 8);
    problem.setUseBias(1);
    problem.setUseGradient(true);
    problem.setBias(rocisa::DataType::Float, 128, 0, true, ContractionProblemGemm::TENSOR::A);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
    env.device.persistentFixedGrid = 8 * 4;
    ASSERT_EQ(solution.getSKReduction(problem, env.device), origami::reduction_t::tree);

    // Handwritten kernels use custom metadata; generated kernels use sizeMapping.
    solution.customKernel.generated = false;
    EXPECT_EQ(solution.requiredWorkspaceSizeGsu(problem, env.device, 4),
              128u * 128 * 8 * 32 + 128 * 8 * 4);
    solution.customKernel.generated = true;
    EXPECT_EQ(solution.requiredWorkspaceSizeGsu(problem, env.device, 4),
              128u * 128 * 4 * 32 + 128 * 4 * 4);
}

// The shared sizing helper must retain GSU's split-one bias-D allocation;
// parallel StreamK cannot reach that case because reconciliation demotes it.
TEST(StreamKLaunchSummaryTest, GsuWorkspacePreservesSplitOneBiasD)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    initStreamKSolution(solution, 0);
    solution.problemType.useBias = 1;
    solution.problemType.useGradient = true;
    solution.sizeMapping.workspaceSizePerElemBias = 4;
    auto problem = makeBatchedGemmProblem(128, 256, 4096, 8);
    problem.setBetaType(rocisa::DataType::Float);
    problem.setUseBias(1);
    problem.setUseGradient(true);
    problem.setBias(rocisa::DataType::Float, 128, 0, true, ContractionProblemGemm::TENSOR::D);
    EXPECT_EQ(solution.requiredWorkspaceSizeGsu(problem, env.device, 0), 0u);
    EXPECT_EQ(solution.requiredWorkspaceSizeGsu(problem, env.device, 1), 128u * 256 * 8 * 4);
    EXPECT_EQ(solution.requiredWorkspaceSizeGsu(problem, env.device, 4), 128u * 128 * 4 * 2 * 8 * 4);
}

// ---------------------------------------------------------------------------
// A workspace-starved parallel scenario, with uniform summation order explicitly OFF.
//
// Same shape as the test above, but with no workspace at all. getSKReduction()
// still says parallel -- it never looks at the workspace -- but origami's
// k_split_aware grid selection does: with zero workspace no split factor F >= 2
// is admissible, so it returns grid == tiles (64) instead of 256. That leaves a
// (parallel, F == 1) triple, which streamKReconcileReduction() demotes to tree.
//
// The demotion is not conditioned on the mode -- F < 2 is unlaunchable for
// parallel either way -- so this pins that the mode-off path reaches the same
// decision as the mode-on variant below: the reconcile runs BEFORE the
// workspace-fit guard, the guard then sees tree with tiles % grid == 0, decides
// no partials are needed, and never runs its body. No fallback is attributed.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, Sk3ParallelWorkspaceStarvedNoUniformOrderReconcilesToTree)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    solution.kernelName = "test_streamk_parallel_starved";
    initStreamKSolution(solution, 3);

    auto problem = makeGemmProblem(256, 4096, 4096);
    problem.setWorkspaceSize(0); // no workspace at all
    problem.setParams().setUniformSummationOrder(0);

    // Anti-vacuity: the mode must be off.
    ASSERT_EQ(problem.getParams().uniformSummationOrder(), 0)
        << "uniform summation order must be off for this scenario, otherwise this test is "
           "a duplicate of the variant below";

    ASSERT_FALSE(Debug::Instance().useStreamKDataParrallel())
        << "unset TENSILE_STREAMK_DATA_PARALLEL before running this suite";

    // Anti-vacuity: the pre-reconcile reduction really is parallel. getSKReduction
    // is workspace-independent, so it reports what the snapshot started from.
    ASSERT_EQ(solution.getSKReduction(problem, env.device), origami::reduction_t::parallel)
        << "scenario must start from parallel reduction, otherwise the reconcile "
           "below is not what demoted it";

    auto d = solution.computeStreamKDecisions(problem, env.device);

    EXPECT_EQ(d.reduction, origami::reduction_t::tree)
        << "streamKReconcileReduction demotes parallel at a split factor of 1, "
           "with the mode off as well as on";
    EXPECT_FALSE(d.workspaceDPFallbackFired)
        << "the reconcile ran first, so the workspace guard saw tree and divisible "
           "tiles and never fired";
    EXPECT_FALSE(d.dpOnly) << "no DP trigger is set: this is a plain tree launch";
    EXPECT_FALSE(d.forceDPOnly) << "not the compile-time param either";
    EXPECT_FALSE(d.streamKDP);

    EXPECT_EQ(d.tiles, 64u);
    EXPECT_EQ(d.finalGrid, d.tiles) << "grid selection, not a clamp, produced grid = tiles";
    // Nothing moved the grid: selection already landed on tiles.
    EXPECT_EQ(d.selectedGrid, d.tiles);
    EXPECT_EQ(d.selectedGrid, d.finalGrid);

    EXPECT_EQ(d.idealWorkspaceBytes, 0u)
        << "the partials guard never ran, so no ideal size was ever computed";
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_FALSE(d.workspaceAllocated);
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, env.device), 0u);

    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());
    EXPECT_NE(line.find("changedBy = none"), std::string::npos)
        << "no clamp produced the launch grid; grid selection did";
    EXPECT_NE(line.find("source = none"), std::string::npos);
    EXPECT_NE(line.find("reduction = tree"), std::string::npos);
    EXPECT_EQ(line.find("preFallback"), std::string::npos)
        << "selection already sat on tiles, so nothing moved";
}

// ---------------------------------------------------------------------------
// The same workspace-starved parallel scenario with uniform summation order ON.
//
// Grid selection behaves identically: the F-star snap in getSKGridImpl only fires for
// g0 != tiles, and here g0 == tiles == 64. So the snapshot again reaches the reconcile
// with a (parallel, F == 1) triple and streamKReconcileReduction() demotes it to tree
// BEFORE the workspace-fit guard. The guard then sees tree with tiles % grid == 0,
// decides no partials are needed, and never runs its body: idealWorkspaceBytes stays 0
// and workspaceDPFallbackFired stays false.
//
// This is NOT a fallback -- it is the selected launch, and the summary attributes the
// grid to nothing ("changedBy = none"). Paired with the variant above to pin that the
// mode does not change the outcome. The genuine parallel-side workspace-DP fallback,
// on a grid that is NOT already tiles, is covered by the test below, which uses a
// fixed grid to get past origami's own workspace clamp.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, Sk3ParallelWorkspaceStarvedUniformOrderReconcilesToTree)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    solution.kernelName = "test_streamk_parallel_starved_uniform";
    initStreamKSolution(solution, 3);

    auto problem = makeGemmProblem(256, 4096, 4096);
    problem.setWorkspaceSize(0); // no workspace at all
    problem.setParams().setUniformSummationOrder(1);

    ASSERT_FALSE(Debug::Instance().useStreamKDataParrallel())
        << "unset TENSILE_STREAMK_DATA_PARALLEL before running this suite";

    // Anti-vacuity: the pre-reconcile reduction really is parallel. With the mode on,
    // getSKReduction's static-two-tile-packing arm (SK3 is one) forces tree unless
    // origami still says parallel with streamKAtomic == 0 and an empty
    // streamKUniformSummationOrderObstacle(), so this also pins that no obstacle fires
    // for this solution.
    ASSERT_EQ(solution.getSKReduction(problem, env.device), origami::reduction_t::parallel)
        << "scenario must start from parallel reduction, otherwise the reconcile "
           "step below is not what demoted it";

    auto d = solution.computeStreamKDecisions(problem, env.device);

    EXPECT_EQ(d.reduction, origami::reduction_t::tree)
        << "streamKReconcileReduction demotes parallel at a split factor of 1";
    EXPECT_FALSE(d.workspaceDPFallbackFired)
        << "the reconcile ran first, so the workspace guard saw tree and divisible "
           "tiles and never fired";
    EXPECT_FALSE(d.dpOnly) << "no DP trigger is set: this is a plain tree launch";
    EXPECT_FALSE(d.forceDPOnly);

    EXPECT_EQ(d.tiles, 64u);
    EXPECT_EQ(d.finalGrid, d.tiles) << "grid selection, not a clamp, produced grid = tiles";
    // Nothing moved the grid: selection already landed on tiles.
    EXPECT_EQ(d.selectedGrid, d.tiles);
    EXPECT_EQ(d.selectedGrid, d.finalGrid);

    EXPECT_EQ(d.idealWorkspaceBytes, 0u)
        << "the partials guard never ran, so no ideal size was ever computed";
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_FALSE(d.workspaceAllocated);
    // Both sizings agree that nothing is reserved.
    EXPECT_EQ(solution.requiredWorkspaceSize(problem, env.device), 0u);

    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());
    EXPECT_NE(line.find("changedBy = none"), std::string::npos)
        << "no clamp produced the launch grid; grid selection did";
    EXPECT_NE(line.find("source = none"), std::string::npos);
    EXPECT_NE(line.find("reduction = tree"), std::string::npos);
    EXPECT_EQ(line.find("preFallback"), std::string::npos);
}

// ---------------------------------------------------------------------------
// The parallel branch's workspace-DP fallback moving the grid.
//
// Neither test above can reach that: origami applies the same workspace predicate
// during grid selection, so a zero-workspace parallel scenario is already sitting on
// grid == tiles by the time the guard runs, and the guard's grid = tiles assignment is
// a no-op even when it fires. persistentFixedGrid bypasses origami's selection entirely
// (getSKGridImpl takes the user-override branch before it consults persistentDynamicGrid), so
// a fixed grid of 2*tiles keeps the split factor at 2 -- which
// streamKReconcileReduction accepts -- while the workspace stays at zero. That is the
// one shape where the guard sees parallel, computes a non-zero partials size, finds it
// does not fit, and actually moves the grid.
//
// This is the parallel-side complement of WorkspaceDpFallbackFires (which covers
// the tree / indivisible-tiles side), and unlike that test the fallback here
// genuinely moves the grid: selected 128 -> final 64.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, Sk3ParallelFixedGridWorkspaceDpFallbackFires)
{
    AnalyticalEnv       env;
    ContractionSolution solution;
    solution.kernelName = "test_streamk_parallel_fixed_starved";
    initStreamKSolution(solution, 3);

    auto problem = makeGemmProblem(256, 4096, 4096);
    problem.setWorkspaceSize(0); // no workspace at all

    // 2 * tiles, so the split factor is exactly 2 -- the smallest value parallel
    // reduction can express, and enough for streamKReconcileReduction to leave it
    // alone.
    env.device.persistentFixedGrid = 128;

    ASSERT_FALSE(Debug::Instance().useStreamKDataParrallel())
        << "unset TENSILE_STREAMK_DATA_PARALLEL before running this suite";

    ASSERT_EQ(solution.getSKReduction(problem, env.device), origami::reduction_t::parallel)
        << "scenario must start from parallel reduction for the fallback to be the "
           "parallel-path fallback";

    auto d = solution.computeStreamKDecisions(problem, env.device);

    EXPECT_EQ(d.tiles, 64u);
    ASSERT_TRUE(d.fixedGridUsed) << "the fixed-grid override is what bypasses origami's clamp";
    ASSERT_EQ(d.selectedGrid, 128u) << "2 * tiles, so the reconcile keeps parallel";

    EXPECT_TRUE(d.workspaceDPFallbackFired);
    EXPECT_EQ(d.reduction, origami::reduction_t::tree) << "the fallback demotes to tree";
    EXPECT_TRUE(d.dpOnly);
    EXPECT_FALSE(d.forceDPOnly) << "runtime fallback, not the compile-time param";
    EXPECT_FALSE(d.treeBoundsFallbackFired) << "tree-bounds fixup is tree-reduction only";
    EXPECT_FALSE(d.clusterDPGridClamped);

    EXPECT_EQ(d.finalGrid, d.tiles) << "the DP fallback sets grid = tiles";
    EXPECT_EQ(d.skGridPreFallback, d.selectedGrid) << "nothing inside getSKGridImpl moved it";

    // Sized by the PARALLEL grid the launch wanted, not by the grid it settled for.
    EXPECT_EQ(d.idealWorkspaceBytes, solution.partialTileSize(d.selectedGrid));
    EXPECT_EQ(d.idealWorkspaceBytes, 8388608u) << "128*128*4 bytes * 128 work-groups";
    EXPECT_GT(d.idealWorkspaceBytes, d.givenWorkspaceBytes) << "why the fallback fired";
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u) << "the fallback reserves nothing";
    EXPECT_FALSE(d.workspaceAllocated);

    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());
    EXPECT_NE(line.find("changedBy = workspaceDP"), std::string::npos)
        << "the workspace fallback, not the fixed-grid override, produced the launch grid";
    EXPECT_EQ(line.find("changedBy = fixedGrid"), std::string::npos);
    EXPECT_NE(line.find("source = workspaceDP(runtime)"), std::string::npos);
    EXPECT_NE(line.find("reduction = tree"), std::string::npos);
    EXPECT_EQ(line.find("preFallback"), std::string::npos)
        << "preFallback only prints when a clamp INSIDE getSKGridImpl moved the grid";
    // Both are still reported as having fired in the fallbacks section.
    EXPECT_NE(line.find("fixedGrid = yes"), std::string::npos);
    EXPECT_NE(line.find("workspaceDPFallback = yes"), std::string::npos);
}

// ---------------------------------------------------------------------------
// The launch summary is OFF unless TENSILE_DB bit 0x200000 is set.
//
// That the diagnostic is opt-in is the feature's central safety claim, so pin it
// directly on the Debug accessor the production gate reads rather than inferring
// it. Three states are checked: TENSILE_DB unset (the shipped default), the
// summary bit, and the neighbouring StreamK mode-selection bit 0x100000 -- which
// must NOT enable the summary.
//
// What this test cannot cover: the production gate is a call site inside
// ContractionSolution::solve() ("if(Debug::Instance().printStreamKLaunchSummary())
// printStreamKLaunchSummary(std::cerr, ...)"), and solve() needs real device input
// pointers and passes a work-queue guard that throws on these mock devices, so a
// host-only unit test cannot reach it. The two host-reachable halves of the claim
// are asserted instead: the predicate the gate reads is false by default, and the
// snapshot builder this suite drives writes nothing to stderr on its own.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, LaunchSummaryDebugBitIsOffByDefault)
{
    // Whatever TENSILE_DB the suite was launched with, the guards below must put it
    // back. Compared against the ambient value rather than against false, so that
    // running the whole suite under TENSILE_DB=0x200000 to eyeball the summaries
    // does not fail this test.
    const bool ambientSummaryBit = Debug::Instance().printStreamKLaunchSummary();

    // (1) Unset -> off. The compile-time debug mask default is 0, so an unset
    // TENSILE_DB leaves every debug bit clear.
    {
        ScopedTensileDb noDb(nullptr);
        EXPECT_FALSE(Debug::Instance().printStreamKLaunchSummary())
            << "the StreamK launch summary must be opt-in";
    }

    // (2) The documented bit turns it on.
    {
        ScopedTensileDb withBit("0x200000");
        ASSERT_TRUE(Debug::Instance().printStreamKLaunchSummary())
            << "TENSILE_DB=0x200000 must enable the launch summary; if this fails the "
               "rest of the test proves nothing about the bit being distinct";
    }

    // (3) The neighbouring StreamK bit does NOT turn it on -- they are distinct
    // opt-ins, so enabling mode-selection tracing does not also spam launch
    // summaries.
    {
        ScopedTensileDb siblingBit("0x100000");
        EXPECT_TRUE(Debug::Instance().printStreamKModeSelection());
        EXPECT_FALSE(Debug::Instance().printStreamKLaunchSummary());
    }

    // (4) State is restored: the guards above must not leak the bit into later
    // tests sharing this process-wide singleton.
    EXPECT_EQ(Debug::Instance().printStreamKLaunchSummary(), ambientSummaryBit)
        << "ScopedTensileDb must restore the prior TENSILE_DB value and reload";

    // (5) Building the snapshot is silent -- nothing reaches stderr unless the
    // caller explicitly prints.
    {
        ScopedTensileDb noDb(nullptr);

        ContractionSolution solution;
        initStreamKSolution(solution, 3);
        auto problem = makeGemmProblem(4096, 4224, 512);
        problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
        auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
        device.persistentDynamicGrid = 0;

        std::ostringstream    captured;
        std::streambuf* const savedCerr = std::cerr.rdbuf(captured.rdbuf());
        auto                  d         = solution.computeStreamKDecisions(problem, device);
        std::cerr.rdbuf(savedCerr);

        EXPECT_TRUE(captured.str().empty())
            << "computeStreamKDecisions() must not write to stderr; got: " << captured.str();

        // ...and the printer is not self-gating: the bit gates the CALL SITE, so an
        // explicit call still emits. That is what lets every other test in this file
        // exercise the report with the bit clear.
        std::ostringstream os;
        solution.printStreamKLaunchSummary(os, problem, d);
        EXPECT_NE(os.str().find("LAUNCH SUMMARY"), std::string::npos);
    }
}

// ---------------------------------------------------------------------------
// changedBy == treeBounds, as the WINNER of the attribution.
//
// The tree fixup indexes with 24-bit divide/remainder arithmetic, so the grid
// selection resets the grid to tiles when itersPerTile >= 65536, itersPerWG >=
// 65536, or tiles*itersPerTile >= 2^24. Existing tests only ever assert that this
// fallback did NOT fire; this one makes it the credited clamp.
//
// Shape: 1152x131072x131072 with a 128x128x64 macro tile.
//   tiles              = ceil(1152/128) * ceil(131072/128) = 9 * 1024 = 9216
//   itersPerTile       = ceil(131072/64)                              = 2048
//   tiles*itersPerTile = 18874368 >= 16777216  -> bound tripped
// (itersPerWG = 9216*2048/64 = 294912 also exceeds 65536; on a device with at most
// 256 CUs the two bounds cannot be tripped independently, since tiles*itersPerTile
// >= 2^24 forces itersPerWG >= 2^24/256 == 65536. itersPerTile itself stays under
// 65536, which the test asserts, so the trigger is not that bound.)
//
// n == k == 131072 keeps ldb == n >= k, which the {k, n} B descriptor built by
// makeGemmProblem with strides {1, ldb} requires.
//
// The clamp sets grid = tiles, so tiles % grid == 0 and the later workspace-DP
// fallback cannot fire and steal the attribution -- asserted below rather than
// assumed. A plain mock AMDGPU with persistentDynamicGrid = 0 is used so the pre-clamp grid
// is exactly the CU count rather than a performance-model output.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, TreeBoundsFallbackWinsGridAttribution)
{
    ContractionSolution solution;
    solution.kernelName = "test_streamk_tree_bounds";
    initStreamKSolution(solution, 3);

    auto problem = makeGemmProblem(1152, 131072, 131072);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;

    ASSERT_FALSE(Debug::Instance().useStreamKDataParrallel())
        << "unset TENSILE_STREAMK_DATA_PARALLEL before running this suite";

    auto d = solution.computeStreamKDecisions(problem, device);

    // The arithmetic that trips the bound, spelled out so that a shape change fails
    // here with a legible message instead of silently un-tripping the fallback.
    const size_t itersPerTile = problem.getItersPerTile(solution.sizeMapping);
    ASSERT_EQ(d.tiles, 9216u) << "ceil(1152/128) * ceil(131072/128) = 9 * 1024";
    ASSERT_EQ(itersPerTile, 2048u) << "ceil(131072/64)";
    ASSERT_GE(d.tiles * itersPerTile, 16777216u) << "tiles*itersPerTile must reach 2^24";
    EXPECT_LT(itersPerTile, 65536u) << "itersPerTile alone is not what trips it";

    // Anti-vacuity: the fallback actually fired, and it is the credited one.
    ASSERT_TRUE(d.treeBoundsFallbackFired) << "scenario must actually trip the tree-bounds fixup";
    EXPECT_EQ(d.reduction, origami::reduction_t::tree) << "the fixup is tree-reduction only";
    EXPECT_FALSE(d.fixedGridUsed);
    EXPECT_FALSE(d.clusterDPGridClamped);
    // The clamp sets grid = tiles, which makes tiles % grid == 0, which is exactly
    // why the later workspace-DP fallback cannot fire and take the attribution.
    ASSERT_EQ(d.tiles % d.finalGrid, 0u) << "clamped grid must divide tiles";
    EXPECT_FALSE(d.workspaceDPFallbackFired)
        << "grid == tiles leaves no partial tiles, so nothing to reserve";

    // selected vs final: the clamp lives inside the grid selection, so the
    // pre-fallback grid moved with it.
    EXPECT_EQ(d.selectedGrid, static_cast<size_t>(_CPX_CU))
        << "selection picks the CU-count grid before the fixup";
    EXPECT_EQ(d.skGridPreFallback, d.tiles);
    EXPECT_EQ(d.finalGrid, d.tiles);
    EXPECT_NE(d.selectedGrid, d.finalGrid);
    EXPECT_EQ(d.skGrid, d.finalGrid);
    EXPECT_EQ(solution.getSKGrid(problem, device, d.tiles, d.reduction), d.finalGrid);

    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_EQ(d.idealWorkspaceBytes, 0u);
    EXPECT_FALSE(d.workspaceAllocated);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));

    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());

    EXPECT_NE(line.find("changedBy = treeBounds"), std::string::npos);
    EXPECT_NE(line.find("treeBoundsFallback = yes"), std::string::npos);
    // Unambiguous: no other clamp is credited or reported as fired.
    EXPECT_NE(line.find("fixedGrid = no"), std::string::npos);
    EXPECT_NE(line.find("workspaceDPFallback = no"), std::string::npos);
    EXPECT_NE(line.find("clusterDPMulticast = no"), std::string::npos);
    EXPECT_NE(line.find("selected = " + std::to_string(d.selectedGrid)), std::string::npos);
    EXPECT_NE(line.find("final = " + std::to_string(d.tiles)), std::string::npos);
    EXPECT_NE(line.find("preFallback = " + std::to_string(d.tiles)), std::string::npos);

    // Complement: the identical shape with a small K leaves the grid alone, so the
    // fallback is a property of the bound and not of the tile count.
    ContractionSolution small;
    small.kernelName = "test_streamk_tree_bounds_ok";
    initStreamKSolution(small, 3);
    auto smallProblem = makeGemmProblem(1152, 131072, 512);
    smallProblem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto ds = small.computeStreamKDecisions(smallProblem, device);
    EXPECT_LT(ds.tiles * smallProblem.getItersPerTile(small.sizeMapping), 16777216u);
    EXPECT_FALSE(ds.treeBoundsFallbackFired);
    EXPECT_EQ(ds.finalGrid, static_cast<size_t>(_CPX_CU));
    EXPECT_EQ(ds.selectedGrid, ds.finalGrid);
}

// ---------------------------------------------------------------------------
// changedBy == fixedGrid, as the WINNER of the attribution.
//
// fixedGridUsed is otherwise only exercised by
// ClusterDpClampWinsAttributionOverFixedGrid, where the override deliberately
// LOSES to a later clamp. Here it is the last clamp standing: SK3, persistentFixedGrid
// set, clusterDim {1, 1, 1} (so no multicast clamp), K small enough that the tree
// bounds are nowhere near, and persistentFixedGrid chosen to divide tiles so the
// workspace-DP fallback has nothing to reserve and cannot steal the attribution.
//
// The subtlety this test exists to pin: the selected grid is captured AFTER the
// persistentFixedGrid override is applied, so selectedGrid == finalGrid == 32 and the
// printed "changedBy = fixedGrid" is NOT derived from selected != final. The
// evidence that the override did something is that the launch grid is 32 rather
// than the CU count the same solution and problem pick without it -- asserted via
// the baseline run at the end.
// ---------------------------------------------------------------------------
TEST(StreamKLaunchSummaryTest, FixedGridOverrideWinsGridAttribution)
{
    ContractionSolution solution;
    solution.kernelName = "test_streamk_fixed_grid";
    initStreamKSolution(solution, 3); // clusterDim stays {1, 1, 1}

    // 4096x4224 -> tiles = 32*33 = 1056, and 1056 % 32 == 0, so the fixed grid
    // leaves no partial tiles. K=512 -> itersPerTile = 8, far below every tree bound.
    auto problem = makeGemmProblem(4096, 4224, 512);
    problem.setWorkspaceSize(std::numeric_limits<size_t>::max());

    auto device          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    device.persistentDynamicGrid = 0;
    device.persistentFixedGrid   = 32;

    ASSERT_FALSE(Debug::Instance().useStreamKDataParrallel())
        << "unset TENSILE_STREAMK_DATA_PARALLEL before running this suite";

    auto d = solution.computeStreamKDecisions(problem, device);

    // Anti-vacuity: the override actually ran.
    ASSERT_TRUE(d.fixedGridUsed) << "scenario must actually take the persistentFixedGrid override";
    EXPECT_FALSE(d.isDynamic);
    EXPECT_EQ(d.reduction, origami::reduction_t::tree);
    EXPECT_EQ(d.tiles, 1056u);

    // No later clamp supersedes it -- this is what makes fixedGrid the winner.
    EXPECT_FALSE(d.treeBoundsFallbackFired);
    EXPECT_FALSE(d.clusterDPGridClamped);
    ASSERT_EQ(d.tiles % d.finalGrid, 0u)
        << "the fixed grid must divide tiles, otherwise the workspace-DP fallback "
           "would reserve partials and could steal the attribution";
    EXPECT_FALSE(d.workspaceDPFallbackFired);
    EXPECT_EQ(d.requiredWorkspaceBytes, 0u);
    EXPECT_EQ(d.idealWorkspaceBytes, 0u);
    EXPECT_FALSE(d.workspaceAllocated);
    EXPECT_EQ(d.requiredWorkspaceBytes, solution.requiredWorkspaceSize(problem, device));
    EXPECT_FALSE(d.dpOnly);

    // The override is applied BEFORE the selected grid is captured, so "selected" is
    // already the overridden value: changedBy is attribution, not a diff.
    EXPECT_EQ(d.selectedGrid, 32u) << "selection reports the overridden grid";
    EXPECT_EQ(d.skGridPreFallback, 32u);
    EXPECT_EQ(d.finalGrid, 32u);
    EXPECT_EQ(d.selectedGrid, d.finalGrid)
        << "the fixed-grid override cannot make selected differ from final";
    EXPECT_EQ(d.skGrid, d.finalGrid);
    EXPECT_EQ(solution.getSKGrid(problem, device, d.tiles, d.reduction), 32u);

    std::ostringstream os;
    solution.printStreamKLaunchSummary(os, problem, d);
    const std::string line = collapseSpaces(os.str());

    EXPECT_NE(line.find("changedBy = fixedGrid"), std::string::npos);
    EXPECT_NE(line.find("fixedGrid = yes"), std::string::npos);
    EXPECT_NE(line.find("selected = 32"), std::string::npos);
    EXPECT_NE(line.find("final = 32"), std::string::npos);
    // preFallback is suppressed because no clamp inside the grid selection moved the
    // grid away from what selection produced.
    EXPECT_EQ(line.find("preFallback"), std::string::npos);
    EXPECT_NE(line.find("treeBoundsFallback = no"), std::string::npos);
    EXPECT_NE(line.find("workspaceDPFallback = no"), std::string::npos);
    EXPECT_NE(line.find("clusterDPMulticast = no"), std::string::npos);

    // Baseline: same solution and problem, no override. This is the evidence that 32
    // came from persistentFixedGrid -- without it the launch uses the CU count.
    ContractionSolution baseline;
    baseline.kernelName = "test_streamk_no_fixed_grid";
    initStreamKSolution(baseline, 3);
    auto baseDevice          = makeDevice(_MI350_CHIP_ID, _CPX_CU, "mi350cpx");
    baseDevice.persistentDynamicGrid = 0; // persistentFixedGrid stays 0

    auto db = baseline.computeStreamKDecisions(problem, baseDevice);
    EXPECT_FALSE(db.fixedGridUsed);
    EXPECT_EQ(db.selectedGrid, static_cast<size_t>(_CPX_CU));
    EXPECT_EQ(db.finalGrid, static_cast<size_t>(_CPX_CU));
    EXPECT_NE(db.finalGrid, d.finalGrid) << "the override is what produced grid 32";

    std::ostringstream osb;
    baseline.printStreamKLaunchSummary(osb, problem, db);
    const std::string baseLine = collapseSpaces(osb.str());
    EXPECT_NE(baseLine.find("changedBy = none"), std::string::npos);
    EXPECT_NE(baseLine.find("fixedGrid = no"), std::string::npos);
}

// AIHPBLAS-5190: scheduling, workspace queries, and packed arguments must agree.
class ClusterStreamKTest : public ::testing::Test
{
protected:
    AnalyticalEnv env;
    ContractionSolution solution;
    ContractionProblemGemm problem = makeGemmProblem(640, 896, 16384); // 35 tiles
    static constexpr size_t partialTileBytes = 128 * 128 * 4;

    void SetUp() override
    {
        initStreamKSolution(solution, 3);
        solution.sizeMapping.clusterDim = TensileLite::dim3(4, 1, 1);
        solution.sizeMapping.globalSplitU = 0;
        solution.sizeMapping.workGroupMapping = 1;
        solution.sizeMapping.workGroupMappingXCC = 1;
        solution.internalArgsSupport.version = 3;
        solution.internalArgsSupport.useUniversalArgs = true;
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        problem.setWorkspaceSize(std::numeric_limits<size_t>::max());
        env.device.persistentDynamicGrid = 0;
        env.device.persistentFixedGrid = 86;
    }

    void expectLaunch(size_t logicalGrid, size_t physicalGrid, size_t workspace)
    {
        auto launch = solution.resolvePersistentSettings(problem, env.device);
        ASSERT_EQ(launch.grid, logicalGrid);
        EXPECT_EQ(launch.workspaceBytes, workspace);
        EXPECT_EQ(solution.requiredWorkspaceSize(problem, env.device), workspace);
        ContractionInputs inputs;
        inputs.alpha = 1.0f;
        inputs.beta = 0.0f;
        auto invocation = solution.generateSingleCall<true>(problem, inputs, env.device, launch, GSUSettings{});
        EXPECT_EQ(invocation.numWorkGroups.x, physicalGrid);
        auto arg = KernelArguments::const_iterator(invocation.args, "skGrid");
        ASSERT_NE(arg, invocation.args.end());
        uint32_t packedGrid = 0;
        ASSERT_EQ((*arg).second, sizeof(packedGrid));
        std::memcpy(&packedGrid, (*arg).first, sizeof(packedGrid));
        EXPECT_EQ(packedGrid, logicalGrid);
    }
};

TEST_F(ClusterStreamKTest, AlignsWorkersBeforeScheduling)
{
    // Reported grid, aligned control, minimum cluster, and the later flag clamp.
    for(auto [requested, expected] : {std::pair<int, size_t>{86, 84}, {84, 84},
                                     {4, 4}, {100000, StreamKFlagElements}})
    {
        SCOPED_TRACE(requested);
        env.device.persistentFixedGrid = requested;
        expectLaunch(expected, expected, partialTileBytes * expected);
    }
}

TEST_F(ClusterStreamKTest, RejectsBudgetSmallerThanCluster)
{
    env.device.persistentFixedGrid = 3;
    problem.setWorkspaceSize(0); // Rejection must precede whole-tile fallback.
    for(int policy : {0, 6}) // Tree and parallel reduction.
    {
        SCOPED_TRACE(policy);
        env.device.persistentDynamicGrid = policy;
        EXPECT_THROW(solution.resolvePersistentSettings(problem, env.device), std::invalid_argument);
        Task task(env.device, problem, solution);
        Predicates::Contraction::WorkspaceCheck predicate;
        EXPECT_FALSE(predicate(task));
        std::ostringstream diagnostic;
        EXPECT_FALSE(predicate.debugEval(task, diagnostic));
        EXPECT_NE(diagnostic.str().find("budget 3 is smaller than ClusterDim=[4,1]; at least 4"),
                  std::string::npos);
    }
}

TEST_F(ClusterStreamKTest, WholeTileFallbackPadsOnlyPhysicalGrid)
{
    for(bool uniform : {false, true})
    {
        SCOPED_TRACE(uniform);
        problem.setWorkspaceSize(uniform ? std::numeric_limits<size_t>::max() : 0);
        problem.setParams().setUniformSummationOrder(uniform);
        expectLaunch(35, 36, 0);
    }
}

TEST_F(ClusterStreamKTest, ParallelAlignmentPreservesIntegralSplits)
{
    env.device.persistentDynamicGrid = 6;
    ASSERT_EQ(solution.getSKReduction(problem, env.device), origami::reduction_t::parallel);
    env.device.persistentFixedGrid = 400;
    expectLaunch(280, 280, partialTileBytes * 280); // Eight workers per tile, in full clusters.
    EXPECT_EQ(solution.resolvePersistentSettings(problem, env.device).reduction,
              origami::reduction_t::parallel);
    env.device.persistentFixedGrid = 86; // No integral clustered split fits; use whole tiles.
    expectLaunch(35, 36, 0);
    EXPECT_EQ(solution.resolvePersistentSettings(problem, env.device).reduction,
              origami::reduction_t::tree);
}

// ---------------------------------------------------------------------------
// Dynamic work-queue decomposition (streamKDynamicSplit), case by case.
// ---------------------------------------------------------------------------
namespace
{
    StreamKDynamicSplitInputs dynamicSplitInputs(size_t tiles, size_t itersPerTile)
    {
        StreamKDynamicSplitInputs in;
        in.tiles            = tiles;
        in.itersPerTile     = itersPerTile;
        in.maxGrid          = 256;
        in.splitCUs         = 256;
        in.flagSlots        = 1792;
        in.workspaceBytes   = size_t{1} << 30;
        in.partialTileBytes = 192 * 224 * 4;
        return in;
    }
} // namespace

TEST(StreamKDynamicSplitTest, ManyTilesStayWhole)
{
    const auto d = streamKDynamicSplit(dynamicSplitInputs(300, 128));
    EXPECT_EQ(d.skTiles, 0u);
    // Whole tiles are packed with split 2.
    EXPECT_EQ(d.skSplit, 2u);
    EXPECT_EQ(d.skItersPerWI, 64u);
    EXPECT_EQ(d.totalItems, 300u);
    EXPECT_EQ(d.grid, 256u);
    EXPECT_EQ(d.partialSlots(), 0u);

    const auto e = streamKDynamicSplit(dynamicSplitInputs(256, 128));
    EXPECT_EQ(e.skTiles, 0u) << "tiles == CUs already fills the grid";
    EXPECT_EQ(e.grid, 256u);
}

// Fewer tiles than CUs: the static grid's split (see streamKDynamicSplit).
TEST(StreamKDynamicSplitTest, SplitFollowsTheStaticFactors)
{
    const auto d = streamKDynamicSplit(dynamicSplitInputs(1, 20480));
    EXPECT_EQ(d.skTiles, 1u);
    EXPECT_EQ(d.skSplit, 16u) << "at most 16 parts, however long K is";
    EXPECT_EQ(d.skItersPerWI, 1280u);
    EXPECT_EQ(d.totalItems, 16u);
    EXPECT_EQ(d.grid, 16u);

    // 20 tiles: 16 parts would need 320 CUs, 12 fit in 240.
    const auto e = streamKDynamicSplit(dynamicSplitInputs(20, 128));
    EXPECT_EQ(e.skTiles, 20u);
    EXPECT_EQ(e.skSplit, 12u);
    EXPECT_EQ(e.skItersPerWI, 11u);
    EXPECT_EQ(e.totalItems, 240u);
    EXPECT_EQ(e.grid, 240u);
    EXPECT_GE(size_t{e.skItersPerWI} * e.skSplit, 128u) << "parts cover the tile";
    EXPECT_LT(size_t{e.skItersPerWI} * (e.skSplit - 1), 128u) << "no empty part";

    // The split is sized for the CUs, not for the grid.
    auto in      = dynamicSplitInputs(50, 128);
    in.maxGrid   = 768;
    const auto f = streamKDynamicSplit(in);
    EXPECT_EQ(f.skSplit, 4u) << "50 x 4 <= 256 CUs, 50 x 6 is not";
    EXPECT_EQ(f.grid, 200u);
}

TEST(StreamKDynamicSplitTest, FewTilesSplitWithinOneWave)
{
    // 100 tiles: two parts each still fit one wave of 256 workgroups.
    const auto d = streamKDynamicSplit(dynamicSplitInputs(100, 128));
    EXPECT_EQ(d.skTiles, 100u);
    EXPECT_EQ(d.skSplit, 2u);
    EXPECT_EQ(d.skItersPerWI, 64u);
    EXPECT_EQ(d.totalItems, 200u);
    EXPECT_EQ(d.grid, 200u);

    // 200 tiles: a second part would not fit, so they stay whole.
    const auto e = streamKDynamicSplit(dynamicSplitInputs(200, 128));
    EXPECT_EQ(e.skTiles, 0u);
    EXPECT_EQ(e.totalItems, 200u);
    EXPECT_EQ(e.grid, 200u);
}

TEST(StreamKDynamicSplitTest, MinimumItersPerPart)
{
    // I = 40: at most 5 parts of 8 iterations, so the largest allowed factor is 4.
    const auto d = streamKDynamicSplit(dynamicSplitInputs(1, 40));
    EXPECT_EQ(d.skSplit, 4u);
    EXPECT_EQ(d.skItersPerWI, 10u);
    EXPECT_GE(d.skItersPerWI, StreamKDynamicMinItersPerWI);

    const auto e = streamKDynamicSplit(dynamicSplitInputs(1, 15));
    EXPECT_EQ(e.skTiles, 0u) << "too short to split";
}

TEST(StreamKDynamicSplitTest, ShrinksToFitWorkspaceAndFlags)
{
    auto in           = dynamicSplitInputs(1, 262144);
    in.workspaceBytes = 10 * in.partialTileBytes + 7;
    const auto d      = streamKDynamicSplit(in);
    EXPECT_EQ(d.skSplit, 8u) << "the largest factor whose partials fit";
    EXPECT_LE(d.partialSlots() * in.partialTileBytes, in.workspaceBytes);

    in.workspaceBytes = in.partialTileBytes;
    EXPECT_EQ(streamKDynamicSplit(in).skTiles, 0u) << "room for one partial is no split at all";

    auto fl      = dynamicSplitInputs(7, 262144);
    fl.flagSlots = 20;
    const auto f = streamKDynamicSplit(fl);
    EXPECT_EQ(f.skSplit, 2u);
    EXPECT_LE(f.partialSlots(), fl.flagSlots);
}

TEST(StreamKDynamicSplitTest, SplitNotAllowedKeepsTilesWhole)
{
    auto in       = dynamicSplitInputs(1, 262144);
    in.allowSplit = false;
    const auto d  = streamKDynamicSplit(in);
    EXPECT_EQ(d.skTiles, 0u);
    EXPECT_EQ(d.skSplit, 2u);
    EXPECT_EQ(d.totalItems, 1u);
    EXPECT_EQ(d.grid, 1u);
}

TEST(StreamKDynamicSplitTest, DebugOverridesReplaceThePolicy)
{
    auto in          = dynamicSplitInputs(1, 262144);
    in.overrideTiles = 1;
    in.overrideSplit = 64;
    auto d           = streamKDynamicSplit(in);
    EXPECT_EQ(d.skTiles, 1u);
    EXPECT_EQ(d.skSplit, 64u);
    EXPECT_EQ(d.totalItems, 64u);
    EXPECT_EQ(d.grid, 64u);

    // Even when the split is not allowed by default.
    in.allowSplit = false;
    EXPECT_EQ(streamKDynamicSplit(in).skSplit, 64u);

    // Split alone keeps the default of no stream-k tiles.
    auto s          = dynamicSplitInputs(1, 262144);
    s.overrideSplit = 64;
    d               = streamKDynamicSplit(s);
    EXPECT_EQ(d.skTiles, 0u);
    EXPECT_EQ(d.skSplit, 64u);
    EXPECT_EQ(d.totalItems, 1u);

    // Partials that do not fit the workspace are not written past it.
    in.workspaceBytes = 10 * in.partialTileBytes;
    d                 = streamKDynamicSplit(in);
    EXPECT_EQ(d.skTiles, 0u);
    EXPECT_EQ(d.partialSlots(), 0u);
}

// The same decomposition through a StreamK Hybrid solution in its dynamic
// sub-mode (or a DynamicWorkQueue one), comparing the workspace query, launch
// settings, summary and launch.
namespace
{
    std::shared_ptr<ContractionSolution> dynamicSplitSolution(int streamK = 5)
    {
        auto solution = std::make_shared<ContractionSolution>();
        initStreamKSolution(*solution, streamK);
        solution->kernelName                            = "streamk_dynamic_split_probe";
        solution->sizeMapping.LocalSplitU               = 1;
        solution->sizeMapping.packBatchDims             = 0;
        solution->sizeMapping.workGroupMapping          = 1;
        solution->sizeMapping.workGroupMappingXCC       = 0;
        solution->sizeMapping.staggerU                  = 0;
        solution->sizeMapping.globalSplitU              = 1;
        solution->sizeMapping.globalAccumulation        = 4;
        solution->internalArgsSupport.staggerU          = true;
        solution->internalArgsSupport.perTileExtraIters = false;
        solution->internalArgsSupport.arrivalFixup      = true;
        solution->problemType.mxScaleFormat             = 1;
        return solution;
    }

    hip::HipAMDGPU dynamicSplitDevice()
    {
        auto device = makeHipDeviceWithAnalytical(makeGfx950AnalyticalHardware());
        device.persistentDynamicGrid
            = static_cast<int>(origami::grid_selection_t::k_split_aware);
        return device;
    }

    ContractionProblemGemm dynamicSplitGemm(size_t m, size_t n, size_t k, size_t batch)
    {
        auto problem = makeBatchedGemmProblem(m, n, k, batch);
        problem.setWorkspaceSize(size_t{1} << 30);
        problem.setParams().setStreamKTileSchedulingMode(1);
        // Off unless a test asks for uniform summation order.
        problem.setParams().setUniformSummationOrder(0);
        return problem;
    }

    void expectSameSplit(StreamKDynamicSplit const& a, StreamKDynamicSplit const& b)
    {
        EXPECT_EQ(a.skTiles, b.skTiles);
        EXPECT_EQ(a.skSplit, b.skSplit);
        EXPECT_EQ(a.skItersPerWI, b.skItersPerWI);
        EXPECT_EQ(a.totalItems, b.totalItems);
        EXPECT_EQ(a.grid, b.grid);
    }

    std::vector<KernelInvocation> solveWithWorkspace(ContractionSolution const&    solution,
                                                     ContractionProblemGemm const& problem,
                                                     Hardware const&               device)
    {
        // Dummy addresses: solve() only stores pointers into the kernel
        // argument block, it never dereferences them.
        void* const       fake = reinterpret_cast<void*>(0x1000);
        ContractionInputs inputs;
        inputs.a             = fake;
        inputs.b             = fake;
        inputs.c             = fake;
        inputs.d             = fake;
        inputs.ws            = fake;
        inputs.Synchronizer  = fake;
        inputs.alpha         = static_cast<float>(1);
        inputs.beta          = static_cast<float>(1);
        inputs.workspaceSize = problem.workspaceSize();
        return solution.solve(problem, inputs, device);
    }
} // namespace

// Only kernels that advertise the arrival fixup are split.
TEST(StreamKDynamicSplitTest, SplitNeedsTheArrivalFixup)
{
    auto         device  = dynamicSplitDevice();
    auto         problem = dynamicSplitGemm(128, 128, 262144, 1);
    auto         capable = dynamicSplitSolution();
    const size_t tiles   = problem.getNumTiles(capable->sizeMapping, 1);
    ASSERT_GT(capable->streamKDynamicDecomposition(problem, device, tiles).skTiles, 0u);

    auto spinFlag                              = dynamicSplitSolution();
    spinFlag->internalArgsSupport.arrivalFixup = false;
    EXPECT_EQ(spinFlag->streamKDynamicDecomposition(problem, device, tiles).skTiles, 0u);
}

// With the dynamic grid policy off and no fixed grid the launch uses one
// workgroup per CU (or TENSILE_STREAMK_MAX_CUS), which the split already fits.
TEST(StreamKDynamicSplitTest, NoDynamicGridKeepsTheSplitWithinTheGrid)
{
    auto         solution = dynamicSplitSolution();
    auto         device   = dynamicSplitDevice();
    auto         problem  = dynamicSplitGemm(128, 128, 262144, 1);
    const size_t tiles    = problem.getNumTiles(solution->sizeMapping, 1);
    const auto   policy   = solution->streamKDynamicDecomposition(problem, device, tiles);
    ASSERT_GT(policy.skTiles, 0u);

    device.persistentDynamicGrid = 0;
    ASSERT_TRUE(solution->streamK5EffectiveDynamic(problem, device));
    for(int maxCUs : {0, 8})
    {
        SCOPED_TRACE("max CUs " + std::to_string(maxCUs));
        device.persistentMaxCUs = maxCUs;
        const auto d = solution->streamKDynamicDecomposition(problem, device, tiles);
        if(maxCUs == 0)
            expectSameSplit(d, policy);
        EXPECT_GT(d.skTiles, 0u);
        EXPECT_GE(solution->getSKGrid(problem, device, tiles, origami::reduction_t::tree),
                  size_t{d.totalItems});
    }
}

// A fixed grid is what launches, so the split is sized for it: every part
// keeps a workgroup of its own and the grid is not raised for the queues.
TEST(StreamKDynamicSplitTest, FixedGridBoundsTheSplit)
{
    auto         solution  = dynamicSplitSolution();
    auto         device    = dynamicSplitDevice();
    auto         problem   = dynamicSplitGemm(128, 128, 262144, 1);
    const size_t tiles     = problem.getNumTiles(solution->sizeMapping, 1);
    const auto   unbounded = solution->streamKDynamicDecomposition(problem, device, tiles);
    ASSERT_GT(unbounded.totalItems, 8u);
    for(int fixed : {4, 8})
    {
        SCOPED_TRACE("fixed grid " + std::to_string(fixed));
        device.persistentFixedGrid = fixed;
        const auto d = solution->streamKDynamicDecomposition(problem, device, tiles);
        EXPECT_EQ(d.skTiles, tiles);
        EXPECT_EQ(d.totalItems, size_t(fixed));
        // 4 is below the 8 per-XCD queues: the items stay within the fixed
        // grid, so the queues do not raise it.
        EXPECT_EQ(solution->getSKGrid(problem, device, tiles, origami::reduction_t::tree),
                  size_t(fixed))
            << "no raise: the items fit the fixed grid";
    }
    // Only an upper bound: a larger fixed grid does not split further.
    device.persistentFixedGrid = 100000;
    expectSameSplit(solution->streamKDynamicDecomposition(problem, device, tiles), unbounded);
}

// A caller that allocates exactly the reported workspace gets the split the
// report was sized for, and the launch carries that same split.
TEST(StreamKDynamicSplitTest, QueriedWorkspaceReproducesTheSplit)
{
    auto solution = dynamicSplitSolution();
    auto device   = dynamicSplitDevice();
    for(size_t batch : {size_t{1}, size_t{4}})
    {
        SCOPED_TRACE("batch " + std::to_string(batch));
        auto problem = dynamicSplitGemm(128, 128, 262144, batch);
        ASSERT_TRUE(solution->streamK5EffectiveDynamic(problem, device));
        const size_t tiles = problem.getNumTiles(solution->sizeMapping, 1);
        ASSERT_EQ(tiles, batch);

        const auto wanted = solution->streamKDynamicDecomposition(problem, device, tiles);
        ASSERT_EQ(wanted.skTiles, tiles);
        ASSERT_GT(wanted.skSplit, 1u);

        const size_t required = solution->requiredWorkspaceSize(problem, device);
        EXPECT_EQ(required, solution->partialTileSize(wanted.partialSlots()));

        problem.setWorkspaceSize(required);
        expectSameSplit(solution->streamKDynamicDecomposition(problem, device, tiles), wanted);
        EXPECT_EQ(solution->requiredWorkspaceSize(problem, device), required);

        const auto launch = solution->resolvePersistentSettings(problem, device);
        EXPECT_EQ(launch.workspaceBytes, required);
        EXPECT_EQ(launch.grid, wanted.grid);

        const auto invocations = solveWithWorkspace(*solution, problem, device);
        ASSERT_FALSE(invocations.empty());
        EXPECT_EQ(invocations.front().numWorkGroups.x, wanted.grid);

        // One byte short: the split shrinks to what fits, never past it.
        problem.setWorkspaceSize(required - 1);
        const auto shorter = solution->streamKDynamicDecomposition(problem, device, tiles);
        EXPECT_LT(shorter.partialSlots(), wanted.partialSlots());
        EXPECT_LE(solution->partialTileSize(shorter.partialSlots()), required - 1);
    }
}

// A DynamicWorkQueue solution with the arrival fixup splits few-tile problems
// as Hybrid's dynamic sub-mode does, and the workspace query and the launch
// carry the split. Without the arrival fixup the tiles stay whole.
TEST(StreamKDynamicSplitTest, DynamicWorkQueueSplitsFewTiles)
{
    auto solution = dynamicSplitSolution(4);
    auto hybrid   = dynamicSplitSolution(5);
    auto device   = dynamicSplitDevice();
    ASSERT_TRUE(solution->sizeMapping.hasDynamicAssignment());
    auto spinFlag                              = dynamicSplitSolution(4);
    spinFlag->internalArgsSupport.arrivalFixup = false;
    for(size_t batch : {size_t{1}, size_t{4}})
    {
        SCOPED_TRACE("batch " + std::to_string(batch));
        auto         problem = dynamicSplitGemm(128, 128, 262144, batch);
        const size_t tiles   = problem.getNumTiles(solution->sizeMapping, 1);
        ASSERT_EQ(tiles, batch);

        const auto d = solution->streamKDynamicDecomposition(problem, device, tiles);
        EXPECT_EQ(d.skTiles, tiles);
        EXPECT_GT(d.skSplit, 1u);
        expectSameSplit(d, hybrid->streamKDynamicDecomposition(problem, device, tiles));

        const size_t required = solution->requiredWorkspaceSize(problem, device);
        EXPECT_EQ(required, solution->partialTileSize(d.partialSlots()));
        problem.setWorkspaceSize(required);
        expectSameSplit(solution->streamKDynamicDecomposition(problem, device, tiles), d);

        const auto launch = solution->resolvePersistentSettings(problem, device);
        EXPECT_EQ(launch.workspaceBytes, required);
        EXPECT_EQ(launch.grid, d.grid);
        const auto invocations = solveWithWorkspace(*solution, problem, device);
        ASSERT_FALSE(invocations.empty());
        EXPECT_EQ(invocations.front().numWorkGroups.x, d.grid);

        EXPECT_EQ(spinFlag->streamKDynamicDecomposition(problem, device, tiles).skTiles, 0u);
    }
}

// requiredWorkspaceSize(), the launch settings and the launch-summary snapshot
// implement the reserve-or-not rule separately; they must agree, split or not.
TEST(StreamKDynamicSplitTest, WorkspaceQueriesAgree)
{
    auto solution = dynamicSplitSolution();
    auto device   = dynamicSplitDevice();
    struct Shape
    {
        size_t m, n, k, batch;
        bool   split;
    };
    // 1 tile, 4 batched tiles, 64 tiles (split 4), 300 tiles (whole, and
    // 300 % 256 != 0 so the tiles % grid partials reservation applies).
    for(Shape sh : {Shape{128, 128, 262144, 1, true},
                    Shape{128, 128, 262144, 4, true},
                    Shape{1024, 1024, 4096, 1, true},
                    Shape{1280, 3840, 1024, 1, false}})
    {
        SCOPED_TRACE(std::to_string(sh.m) + "x" + std::to_string(sh.n) + "x"
                     + std::to_string(sh.k) + " batch " + std::to_string(sh.batch));
        auto         problem = dynamicSplitGemm(sh.m, sh.n, sh.k, sh.batch);
        const size_t tiles   = problem.getNumTiles(solution->sizeMapping, 1);
        EXPECT_EQ(solution->streamKDynamicDecomposition(problem, device, tiles).skTiles > 0,
                  sh.split);
        for(int pass = 0; pass < 2; ++pass)
        {
            const size_t required = solution->requiredWorkspaceSize(problem, device);
            const auto   launch   = solution->resolvePersistentSettings(problem, device);
            const auto   decided  = solution->computeStreamKDecisions(problem, device);
            EXPECT_EQ(launch.workspaceBytes, required) << "pass " << pass;
            EXPECT_EQ(decided.requiredWorkspaceBytes, required) << "pass " << pass;
            EXPECT_GT(required, 0u);
            // Second pass: exactly the reported size.
            problem.setWorkspaceSize(required);
        }
    }
}

// The split is fitted with slots * partialTileSize(1) and reserved as
// partialTileSize(slots): the two must be the same bytes.
TEST(StreamKDynamicSplitTest, FitMatchesTheReservation)
{
    auto solution = dynamicSplitSolution();
    for(size_t slots : {size_t{1}, size_t{2}, size_t{45}, size_t{256}, size_t{1792}})
        EXPECT_EQ(solution->partialTileSize(slots), slots * solution->partialTileSize(1));
}

// Uniform summation order: the dynamic split gives every tile the same parts
// (skTiles == tiles), so it splits as with the mode off, except where partials
// are summed in arrival order or widened by the fixup.
TEST(StreamKDynamicSplitTest, UniformSummationOrderSplitsEveryTile)
{
    auto device = dynamicSplitDevice();
    for(int streamK : {4, 5})
    {
        auto solution = dynamicSplitSolution(streamK);
        for(auto sh : {std::array<size_t, 4>{128, 128, 262144, 1},
                       std::array<size_t, 4>{128, 128, 262144, 4},
                       std::array<size_t, 4>{1024, 1024, 4096, 1}})
        {
            SCOPED_TRACE("StreamK " + std::to_string(streamK) + " " + std::to_string(sh[0])
                         + "x" + std::to_string(sh[1]) + "x" + std::to_string(sh[2])
                         + " batch " + std::to_string(sh[3]));
            auto         off   = dynamicSplitGemm(sh[0], sh[1], sh[2], sh[3]);
            const size_t tiles = off.getNumTiles(solution->sizeMapping, 1);
            auto         uso   = off;
            uso.setParams().setUniformSummationOrder(1);
            const auto d = solution->streamKDynamicDecomposition(uso, device, tiles);
            EXPECT_EQ(d.skTiles, tiles);
            EXPECT_GE(d.skSplit, 2u);
            expectSameSplit(d, solution->streamKDynamicDecomposition(off, device, tiles));
            EXPECT_TRUE(solution->uniformSummationOrderSupported(uso, device));
            EXPECT_NO_THROW(solveWithWorkspace(*solution, uso, device));
        }
    }

    auto         uso   = dynamicSplitGemm(128, 128, 262144, 1);
    const size_t tiles = uso.getNumTiles(dynamicSplitSolution()->sizeMapping, 1);
    uso.setParams().setUniformSummationOrder(1);
    auto off = uso;
    off.setParams().setUniformSummationOrder(0);

    auto atomic                       = dynamicSplitSolution();
    atomic->sizeMapping.streamKAtomic = 1;
    EXPECT_GT(atomic->streamKDynamicDecomposition(off, device, tiles).skTiles, 0u);
    EXPECT_EQ(atomic->streamKDynamicDecomposition(uso, device, tiles).skTiles, 0u);

    auto widened                                 = dynamicSplitSolution();
    widened->problemType.highPrecisionAccumulate = true;
    widened->problemType.computeType             = rocisa::DataType::Half;
    EXPECT_GT(widened->streamKDynamicDecomposition(off, device, tiles).skTiles, 0u);
    EXPECT_EQ(widened->streamKDynamicDecomposition(uso, device, tiles).skTiles, 0u);
}

// Uniform summation order ignores debug overrides that split only some tiles.
TEST(StreamKDynamicSplitTest, UniformSummationOrderIgnoresPartialOverrides)
{
    auto       in     = dynamicSplitInputs(4, 4096);
    const auto policy = streamKDynamicSplit(in);
    ASSERT_EQ(policy.skTiles, 4u);
    in.uniformSummationOrder = true;
    in.overrideTiles         = 1;
    in.overrideSplit         = 8;
    expectSameSplit(streamKDynamicSplit(in), policy);
    in.overrideTiles = 4;
    EXPECT_EQ(streamKDynamicSplit(in).skTiles, 4u);
    EXPECT_EQ(streamKDynamicSplit(in).skSplit, 8u);
    in.overrideTiles = 0;
    EXPECT_EQ(streamKDynamicSplit(in).skTiles, 0u);
    // Nor where the policy would not split.
    in.overrideTiles = 4;
    in.allowSplit    = false;
    EXPECT_EQ(streamKDynamicSplit(in).skTiles, 0u);

    auto         solution = dynamicSplitSolution();
    auto         device   = dynamicSplitDevice();
    auto         problem  = dynamicSplitGemm(128, 128, 262144, 4);
    const size_t tiles    = problem.getNumTiles(solution->sizeMapping, 1);
    device.skTiles        = 1;
    EXPECT_EQ(solution->streamKDynamicDecomposition(problem, device, tiles).skTiles, 1u);
    problem.setParams().setUniformSummationOrder(1);
    EXPECT_EQ(solution->streamKDynamicDecomposition(problem, device, tiles).skTiles, tiles);
    EXPECT_TRUE(solution->uniformSummationOrderSupported(problem, device));
}

// Parallel (PostGSU) reduction of the dynamic decomposition (see streamKDynamicSplit).
namespace
{
    StreamKDynamicSplitInputs parallelSplitInputs(size_t tiles, size_t itersPerTile)
    {
        auto in          = dynamicSplitInputs(tiles, itersPerTile);
        in.allowParallel = true;
        return in;
    }

    std::shared_ptr<ContractionSolution> dynamicParallelSolution()
    {
        auto solution                                 = dynamicSplitSolution();
        solution->internalArgsSupport.dynamicParallel = true;
        return solution;
    }
} // namespace

TEST(StreamKDynamicSplitTest, ParallelSplitFollowsTheStaticPath)
{
    // I = 20480: one part per CU, beyond the tree factors.
    const auto d = streamKDynamicSplit(parallelSplitInputs(1, 20480));
    EXPECT_TRUE(d.parallel);
    EXPECT_EQ(d.skTiles, 1u);
    EXPECT_EQ(d.skSplit, 256u);
    EXPECT_EQ(d.skItersPerWI, 80u);
    EXPECT_EQ(d.totalItems, 256u);
    EXPECT_EQ(d.grid, 256u);

    // 12 tiles: 21 parts each, one wave.
    const auto e = streamKDynamicSplit(parallelSplitInputs(12, 4096));
    EXPECT_TRUE(e.parallel);
    EXPECT_EQ(e.skTiles, 12u);
    EXPECT_EQ(e.skSplit, 21u);
    EXPECT_EQ(e.totalItems, 252u);

    // Without the capability the same problem takes the arrival fixup.
    auto in          = parallelSplitInputs(1, 20480);
    in.allowParallel = false;
    const auto a     = streamKDynamicSplit(in);
    EXPECT_FALSE(a.parallel);
    EXPECT_EQ(a.skSplit, 16u);
}

TEST(StreamKDynamicSplitTest, ParallelKeepsManyTilesWholeAndMinIters)
{
    const auto d = streamKDynamicSplit(parallelSplitInputs(256, 4096));
    EXPECT_EQ(d.skTiles, 0u);
    EXPECT_FALSE(d.parallel) << "whole tiles need no reduction";
    EXPECT_EQ(d.totalItems, 256u);

    // 200 tiles: a second part would not fit one wave.
    EXPECT_EQ(streamKDynamicSplit(parallelSplitInputs(200, 4096)).skTiles, 0u);

    // I = 64: at most 8 parts of 8 iterations.
    const auto m = streamKDynamicSplit(parallelSplitInputs(1, 64));
    EXPECT_TRUE(m.parallel);
    EXPECT_EQ(m.skSplit, 8u);
    EXPECT_GE(m.skItersPerWI, StreamKDynamicMinItersPerWI);
    EXPECT_EQ(streamKDynamicSplit(parallelSplitInputs(1, 15)).skTiles, 0u);
}

TEST(StreamKDynamicSplitTest, ParallelFitsTheWorkspaceNotTheFlags)
{
    // No per-part flags: 7 tiles x 36 parts with only 20 flag slots.
    auto fl      = parallelSplitInputs(7, 262144);
    fl.flagSlots = 20;
    const auto f = streamKDynamicSplit(fl);
    EXPECT_TRUE(f.parallel);
    EXPECT_EQ(f.skSplit, 36u);

    // One partial tile per part.
    auto in           = parallelSplitInputs(2, 262144);
    in.workspaceBytes = 50 * 2 * in.partialTileBytes;
    const auto d      = streamKDynamicSplit(in);
    EXPECT_TRUE(d.parallel);
    EXPECT_EQ(d.skSplit, 50u);
    in.workspaceBytes -= 1;
    EXPECT_EQ(streamKDynamicSplit(in).skSplit, 49u) << "shrink, not fall back";

    // Room for one part per tile only: no split fits.
    in.workspaceBytes = 2 * in.partialTileBytes;
    EXPECT_EQ(streamKDynamicSplit(in).skTiles, 0u);
}

TEST(StreamKDynamicSplitTest, ParallelWithoutTheArrivalFixup)
{
    // A kernel can support the parallel reduction without the arrival fixup.
    auto in       = parallelSplitInputs(1, 20480);
    in.allowSplit = false;
    const auto d  = streamKDynamicSplit(in);
    EXPECT_TRUE(d.parallel);
    EXPECT_EQ(d.skSplit, 256u);
}

TEST(StreamKDynamicSplitTest, ParallelOnlyWhereTheStaticPathTakesIt)
{
    // 64 tiles = 256 / 4: parallel, 4 parts each.
    const auto p = streamKDynamicSplit(parallelSplitInputs(64, 4096));
    EXPECT_TRUE(p.parallel);
    EXPECT_EQ(p.skSplit, 4u);

    // 65 and 128 tiles: the arrival fixup with the tree factors.
    const auto a = streamKDynamicSplit(parallelSplitInputs(65, 4096));
    EXPECT_FALSE(a.parallel);
    EXPECT_EQ(a.skSplit, 3u);
    auto       in = parallelSplitInputs(128, 4096);
    const auto b  = streamKDynamicSplit(in);
    EXPECT_FALSE(b.parallel);
    EXPECT_EQ(b.skTiles, 128u);
    EXPECT_EQ(b.skSplit, 2u);

    // Without the arrival fixup those tiles stay whole.
    in.allowSplit = false;
    EXPECT_EQ(streamKDynamicSplit(in).skTiles, 0u);

    // Fewer than 64 iterations per tile: the arrival fixup.
    const auto s = streamKDynamicSplit(parallelSplitInputs(1, 63));
    EXPECT_FALSE(s.parallel);
    EXPECT_EQ(s.skSplit, 6u);
}

TEST(StreamKDynamicSplitTest, ParallelOnlyWhenEveryTileIsSplit)
{
    // Debug overrides: every tile split -> parallel; a DP + SK mix keeps the
    // arrival fixup (the PostGSU kernel sums every tile).
    auto in          = parallelSplitInputs(4, 262144);
    in.overrideTiles = 4;
    in.overrideSplit = 16;
    auto d           = streamKDynamicSplit(in);
    EXPECT_TRUE(d.parallel);
    EXPECT_EQ(d.skTiles, 4u);
    EXPECT_EQ(d.skSplit, 16u);

    in.overrideTiles = 3;
    d                = streamKDynamicSplit(in);
    EXPECT_FALSE(d.parallel);
    EXPECT_EQ(d.skTiles, 3u);
    EXPECT_EQ(d.totalItems, 1u + 3u * 16u);

    // Split 1 is no split: whole tiles, no reduction.
    in.overrideTiles = 4;
    in.overrideSplit = 1;
    d                = streamKDynamicSplit(in);
    EXPECT_FALSE(d.parallel);
}

// Only kernels that advertise the parallel reduction get it.
TEST(StreamKDynamicSplitTest, ParallelNeedsTheCapability)
{
    auto         device  = dynamicSplitDevice();
    auto         problem = dynamicSplitGemm(128, 128, 262144, 1);
    auto         capable = dynamicParallelSolution();
    const size_t tiles   = problem.getNumTiles(capable->sizeMapping, 1);
    const auto   d       = capable->streamKDynamicDecomposition(problem, device, tiles);
    EXPECT_TRUE(d.parallel);
    EXPECT_EQ(d.skTiles, tiles);

    auto older = dynamicSplitSolution();
    EXPECT_FALSE(older->streamKDynamicDecomposition(problem, device, tiles).parallel)
        << "logic without the capability keeps the arrival fixup";

    // The parallel reduction is Hybrid-only: DynamicWorkQueue keeps the arrival fixup.
    auto queueOnly                                 = dynamicSplitSolution(4);
    queueOnly->internalArgsSupport.dynamicParallel = true;
    const auto q = queueOnly->streamKDynamicDecomposition(problem, device, tiles);
    EXPECT_FALSE(q.parallel);
    EXPECT_EQ(q.skTiles, tiles);

    // Every tile splits alike and PostGSU sums the parts in order, so uniform
    // summation order keeps the parallel reduction.
    auto uso = problem;
    uso.setParams().setUniformSummationOrder(1);
    const auto u = capable->streamKDynamicDecomposition(uso, device, tiles);
    EXPECT_TRUE(u.parallel);
    EXPECT_EQ(u.skTiles, tiles);
    EXPECT_TRUE(capable->uniformSummationOrderSupported(uso, device));
}

// The bias-gradient and amaxD stores of the parallel reduction assume one tile
// per workgroup, so those problems take the arrival fixup.
TEST(StreamKDynamicSplitTest, ParallelExcludesBiasGradientAndAmaxD)
{
    auto         device  = dynamicSplitDevice();
    auto         problem = dynamicSplitGemm(128, 128, 262144, 1);
    const size_t tiles   = problem.getNumTiles(dynamicParallelSolution()->sizeMapping, 1);

    auto gradient                     = dynamicParallelSolution();
    gradient->problemType.useGradient = true;
    gradient->problemType.useBias     = 1;
    const auto g = gradient->streamKDynamicDecomposition(problem, device, tiles);
    EXPECT_FALSE(g.parallel);
    EXPECT_EQ(g.skTiles, tiles) << "still split, with the arrival fixup";

    auto amax                     = dynamicParallelSolution();
    amax->problemType.outputAmaxD = true;
    const auto a = amax->streamKDynamicDecomposition(problem, device, tiles);
    EXPECT_FALSE(a.parallel);
    EXPECT_EQ(a.skTiles, tiles);
}

// Query, launch summary, launch and solve() agree on the parallel workspace,
// batched or not. The launch is the GEMM plus a PostGSU kernel.
TEST(StreamKDynamicSplitTest, ParallelWorkspaceAndLaunchAgree)
{
    auto solution = dynamicParallelSolution();
    auto device   = dynamicSplitDevice();
    for(size_t batch : {size_t{1}, size_t{4}})
    {
        SCOPED_TRACE("batch " + std::to_string(batch));
        auto problem = dynamicSplitGemm(128, 128, 262144, batch);
        ASSERT_TRUE(solution->streamK5EffectiveDynamic(problem, device));
        const size_t tiles = problem.getNumTiles(solution->sizeMapping, 1);
        ASSERT_EQ(tiles, batch);

        const auto wanted = solution->streamKDynamicDecomposition(problem, device, tiles);
        ASSERT_TRUE(wanted.parallel);
        ASSERT_EQ(wanted.skTiles, tiles);
        ASSERT_GT(wanted.skSplit, 1u);

        const size_t required = solution->requiredWorkspaceSize(problem, device);
        EXPECT_EQ(required, solution->partialTileSize(wanted.partialSlots()));
        const auto decided = solution->computeStreamKDecisions(problem, device);
        EXPECT_EQ(decided.requiredWorkspaceBytes, required);
        EXPECT_EQ(decided.reduction, origami::reduction_t::parallel);

        problem.setWorkspaceSize(required);
        expectSameSplit(solution->streamKDynamicDecomposition(problem, device, tiles), wanted);
        EXPECT_EQ(solution->requiredWorkspaceSize(problem, device), required);

        const auto launch = solution->resolvePersistentSettings(problem, device);
        EXPECT_EQ(launch.reduction, origami::reduction_t::parallel);
        EXPECT_EQ(launch.workspaceBytes, required);
        EXPECT_EQ(launch.grid, wanted.grid);

        const auto invocations = solveWithWorkspace(*solution, problem, device);
        ASSERT_EQ(invocations.size(), 2u) << "GEMM + PostGSU";
        EXPECT_EQ(invocations[0].numWorkGroups.x, wanted.grid);
        EXPECT_NE(invocations[1].kernelName.find("_PostGSU"), std::string::npos);

        // One byte short: the split shrinks to what fits.
        problem.setWorkspaceSize(required - 1);
        const auto shorter = solution->streamKDynamicDecomposition(problem, device, tiles);
        EXPECT_LT(shorter.skSplit, wanted.skSplit);
        EXPECT_LE(solution->partialTileSize(shorter.partialSlots()), required - 1);
    }
}

// More than CUs / 4 tiles on a kernel with both reductions take the arrival
// fixup, and the workspace query, launch summary and launch agree on it.
TEST(StreamKDynamicSplitTest, ManyTilesQueryAndLaunchTakeTheArrivalFixup)
{
    auto         solution = dynamicParallelSolution();
    auto         device   = dynamicSplitDevice();
    const size_t mtx      = solution->sizeMapping.macroTile.x;
    const size_t mty      = solution->sizeMapping.macroTile.y;
    auto         problem  = dynamicSplitGemm(16 * mtx, 8 * mty, 4096, 1);
    const size_t tiles    = problem.getNumTiles(solution->sizeMapping, 1);
    ASSERT_EQ(tiles, 128u);

    const auto wanted = solution->streamKDynamicDecomposition(problem, device, tiles);
    ASSERT_EQ(wanted.skTiles, tiles);
    ASSERT_EQ(wanted.skSplit, 2u);
    EXPECT_FALSE(wanted.parallel);
    const size_t required = solution->requiredWorkspaceSize(problem, device);
    EXPECT_EQ(required, solution->partialTileSize(wanted.partialSlots()));
    EXPECT_EQ(solution->computeStreamKDecisions(problem, device).requiredWorkspaceBytes, required);

    problem.setWorkspaceSize(required);
    expectSameSplit(solution->streamKDynamicDecomposition(problem, device, tiles), wanted);
    const auto launch = solution->resolvePersistentSettings(problem, device);
    EXPECT_NE(launch.reduction, origami::reduction_t::parallel);
    EXPECT_EQ(launch.workspaceBytes, required);
    EXPECT_EQ(solveWithWorkspace(*solution, problem, device).size(), 1u) << "no PostGSU";

    // Without the arrival fixup the same problem keeps its tiles whole.
    auto parallelOnly                              = dynamicParallelSolution();
    parallelOnly->internalArgsSupport.arrivalFixup = false;
    problem.setWorkspaceSize(size_t{1} << 30);
    const auto p = parallelOnly->streamKDynamicDecomposition(problem, device, tiles);
    EXPECT_FALSE(p.parallel);
    EXPECT_EQ(p.skTiles, 0u);
    EXPECT_EQ(solveWithWorkspace(*parallelOnly, problem, device).size(), 1u);
}

// The packed kernel arguments: bit 29 of the SKTiles slot set, Flags still the
// Synchronizer (work queues), and the PostGSU kernel's gsu the split.
TEST(StreamKDynamicSplitTest, ParallelKernelArguments)
{
    auto         solution = dynamicParallelSolution();
    auto         device   = dynamicSplitDevice();
    auto         problem  = dynamicSplitGemm(128, 128, 262144, 1);
    const size_t tiles    = problem.getNumTiles(solution->sizeMapping, 1);
    const auto   d        = solution->streamKDynamicDecomposition(problem, device, tiles);
    ASSERT_TRUE(d.parallel);

    void* const       fake = reinterpret_cast<void*>(0x1000);
    void* const       cPtr = reinterpret_cast<void*>(0x3000);
    void* const       dPtr = reinterpret_cast<void*>(0x4000);
    void* const       ws   = reinterpret_cast<void*>(0x5000);
    void* const       sync = reinterpret_cast<void*>(0x9000);
    ContractionInputs inputs;
    inputs.a = inputs.b  = fake;
    inputs.c             = cPtr;
    inputs.d             = dPtr;
    inputs.ws            = ws;
    inputs.Synchronizer  = sync;
    inputs.alpha         = static_cast<float>(1);
    inputs.beta          = static_cast<float>(1);
    inputs.workspaceSize = problem.workspaceSize();
    const auto inv = solution->solve(problem, inputs, device);
    ASSERT_EQ(inv.size(), 2u);

    auto words = [](KernelArguments const& args) {
        std::vector<uint32_t> w(args.size() / 4);
        std::memcpy(w.data(), args.data(), w.size() * 4);
        return w;
    };
    const auto gemm = words(inv[0].args);
    // SKTiles | mode bit (30) | parallel bit (29), then SKSplit.
    const uint32_t packed = d.skTiles | 0x40000000u | 0x20000000u;
    auto           it     = std::find(gemm.begin(), gemm.end(), packed);
    ASSERT_NE(it, gemm.end()) << "SKTiles slot carries the parallel bit";
    ASSERT_NE(std::next(it), gemm.end());
    EXPECT_EQ(*std::next(it), d.skSplit);
    EXPECT_NE(std::find(gemm.begin(), gemm.end(),
                        static_cast<uint32_t>(reinterpret_cast<uintptr_t>(sync))),
              gemm.end())
        << "Flags stay the Synchronizer for the work queues";

    // The GEMM kernel writes its parts to the workspace: D and C (ws_d, ws_c)
    // are both the workspace, and neither the user's D nor C is passed to it.
    const auto ptrs = [](KernelArguments const& args) {
        std::vector<uint64_t> p(args.size() / 8);
        std::memcpy(p.data(), args.data(), p.size() * 8);
        return p;
    };
    const auto asU64   = [](void* p) { return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)); };
    const auto gemmP   = ptrs(inv[0].args);
    const auto wsFirst = std::find(gemmP.begin(), gemmP.end(), asU64(ws));
    ASSERT_NE(wsFirst, gemmP.end()) << "GEMM ws_d is the workspace";
    ASSERT_NE(std::next(wsFirst), gemmP.end());
    EXPECT_EQ(*std::next(wsFirst), asU64(ws)) << "GEMM ws_c is the workspace";
    EXPECT_EQ(std::count(gemmP.begin(), gemmP.end(), asU64(dPtr)), 0) << "GEMM never sees D";
    EXPECT_EQ(std::count(gemmP.begin(), gemmP.end(), asU64(cPtr)), 0) << "GEMM never sees C";

    // The PostGSU kernel reads the slots from that workspace and writes the
    // user's D: its arguments start D, WS, C.
    const auto post = ptrs(inv[1].args);
    ASSERT_GE(post.size(), 3u);
    EXPECT_EQ(post[0], asU64(dPtr)) << "PostGSU D";
    EXPECT_EQ(post[1], asU64(ws)) << "PostGSU WS is the GEMM's D (ws_d)";
    EXPECT_EQ(post[2], asU64(cPtr)) << "PostGSU C";

    // It sums gsu = skSplit slots; gsu follows the output sizes (M, N, batch).
    // A debug-override split whose parts exceed the grid makes grid / tiles
    // differ from skSplit.
    const auto gsuOf = [&](KernelInvocation const& postGSU) {
        const auto                  w     = words(postGSU.args);
        const std::vector<uint32_t> sizes = {static_cast<uint32_t>(problem.d().sizes()[0]),
                                             static_cast<uint32_t>(problem.d().sizes()[1]),
                                             static_cast<uint32_t>(problem.d().sizes()[2])};
        auto at = std::search(w.begin(), w.end(), sizes.begin(), sizes.end());
        EXPECT_NE(at, w.end()) << "PostGSU output sizes";
        if(at == w.end() || at + 3 == w.end())
            return uint32_t{0};
        return *(at + 3);
    };
    EXPECT_EQ(gsuOf(inv[1]), d.skSplit);

    auto overridden    = device;
    overridden.skTiles = static_cast<int>(tiles);
    overridden.skSplit = 1000;
    const auto o       = solution->streamKDynamicDecomposition(problem, overridden, tiles);
    ASSERT_TRUE(o.parallel);
    ASSERT_NE(o.grid / tiles, size_t{o.skSplit});
    const auto invO = solution->solve(problem, inputs, overridden);
    ASSERT_EQ(invO.size(), 2u);
    EXPECT_EQ(invO[0].numWorkGroups.x, o.grid);
    EXPECT_EQ(gsuOf(invO[1]), o.skSplit) << "gsu is skSplit, not grid / tiles";

    // Without the capability (arrival fixup) the bit is never sent.
    auto       older = dynamicSplitSolution();
    const auto inv2  = older->solve(problem, inputs, device);
    ASSERT_EQ(inv2.size(), 1u) << "no PostGSU kernel for the arrival fixup";
    for(uint32_t w : words(inv2[0].args))
        if((w & 0x40000000u) != 0 && (w & 0x1FFFFFFFu) == tiles)
            EXPECT_EQ(w & 0x20000000u, 0u);
}

// The CU-count hint caps splitCUs (see streamKDynamicDecomposition). 14 x 2
// tiles of 144 iterations: 256 / 28 = 9 parts per tile, 192 / 28 = 6 hinted.
TEST(StreamKDynamicSplitTest, HintSizesTheSplit)
{
    auto         solution = dynamicParallelSolution();
    auto         device   = dynamicSplitDevice();
    auto         problem  = dynamicSplitGemm(1792, 256, 9216, 1);
    const size_t tiles    = problem.getNumTiles(solution->sizeMapping, 1);
    ASSERT_EQ(tiles, 28u);

    const auto open = solution->streamKDynamicDecomposition(problem, device, tiles);
    ASSERT_TRUE(open.parallel);
    EXPECT_EQ(open.skSplit, 9u);

    problem.setParams().setSmCountTarget(192);
    const auto hinted = solution->streamKDynamicDecomposition(problem, device, tiles);
    ASSERT_TRUE(hinted.parallel);
    EXPECT_EQ(hinted.skSplit, 6u);
    EXPECT_EQ(hinted.grid, 168u);
    EXPECT_LT(solution->partialTileSize(hinted.partialSlots()),
              solution->partialTileSize(open.partialSlots()));

    // A hint above the CU count, or none, sizes for the device.
    problem.setParams().setSmCountTarget(100000);
    expectSameSplit(solution->streamKDynamicDecomposition(problem, device, tiles), open);
    problem.setParams().setSmCountTarget(0);
    expectSameSplit(solution->streamKDynamicDecomposition(problem, device, tiles), open);

    // 14 x 4 tiles: parallel on the device (56 <= 256 / 4), the arrival
    // fixup on 192 hinted CUs (56 > 192 / 4), with 3 parts per tile.
    auto         wide      = dynamicSplitGemm(1792, 512, 9216, 1);
    const size_t wideTiles = wide.getNumTiles(solution->sizeMapping, 1);
    ASSERT_EQ(wideTiles, 56u);
    const auto w = solution->streamKDynamicDecomposition(wide, device, wideTiles);
    EXPECT_TRUE(w.parallel);
    EXPECT_EQ(w.skSplit, 4u);
    wide.setParams().setSmCountTarget(192);
    const auto wh = solution->streamKDynamicDecomposition(wide, device, wideTiles);
    EXPECT_FALSE(wh.parallel);
    EXPECT_EQ(wh.skSplit, 3u);
    EXPECT_EQ(wh.grid, 168u);
}

// The workspace query and the launch see the same hint, so they agree. A
// launch given the larger workspace an unhinted query asks for takes the
// hinted split; one given the hinted workspace without the hint shrinks to it.
TEST(StreamKDynamicSplitTest, HintedWorkspaceQueryAndLaunchAgree)
{
    auto         solution     = dynamicParallelSolution();
    auto         device       = dynamicSplitDevice();
    auto         problem      = dynamicSplitGemm(1792, 256, 9216, 1);
    const size_t tiles        = problem.getNumTiles(solution->sizeMapping, 1);
    const auto   open         = solution->streamKDynamicDecomposition(problem, device, tiles);
    const size_t openRequired = solution->requiredWorkspaceSize(problem, device);

    problem.setParams().setSmCountTarget(192);
    const auto   hinted   = solution->streamKDynamicDecomposition(problem, device, tiles);
    const size_t required = solution->requiredWorkspaceSize(problem, device);
    ASSERT_TRUE(hinted.parallel);
    EXPECT_EQ(required, solution->partialTileSize(hinted.partialSlots()));
    EXPECT_EQ(solution->computeStreamKDecisions(problem, device).requiredWorkspaceBytes, required);

    problem.setWorkspaceSize(required);
    expectSameSplit(solution->streamKDynamicDecomposition(problem, device, tiles), hinted);
    const auto launch = solution->resolvePersistentSettings(problem, device);
    EXPECT_EQ(launch.reduction, origami::reduction_t::parallel);
    EXPECT_EQ(launch.workspaceBytes, required);
    EXPECT_EQ(launch.grid, hinted.grid);
    const auto inv = solveWithWorkspace(*solution, problem, device);
    ASSERT_EQ(inv.size(), 2u) << "GEMM + PostGSU";
    EXPECT_EQ(inv[0].numWorkGroups.x, hinted.grid);

    problem.setWorkspaceSize(openRequired);
    expectSameSplit(solution->streamKDynamicDecomposition(problem, device, tiles), hinted);

    problem.setParams().setSmCountTarget(0);
    problem.setWorkspaceSize(required);
    const auto shrunk = solution->streamKDynamicDecomposition(problem, device, tiles);
    EXPECT_LT(shrunk.totalItems, open.totalItems);
    EXPECT_LE(solution->partialTileSize(shrunk.partialSlots()), required);
    EXPECT_LE(solution->resolvePersistentSettings(problem, device).workspaceBytes, required);
}

// Whole tiles keep the device-wide grid under the hint.
TEST(StreamKDynamicSplitTest, HintKeepsWholeTileGrid)
{
    auto         solution = dynamicParallelSolution();
    auto         device   = dynamicSplitDevice();
    auto         problem  = dynamicSplitGemm(4096, 2048, 9216, 1);
    const size_t tiles    = problem.getNumTiles(solution->sizeMapping, 1);
    const auto   open     = solution->streamKDynamicDecomposition(problem, device, tiles);
    ASSERT_EQ(open.skTiles, 0u);

    problem.setParams().setSmCountTarget(192);
    const auto hinted = solution->streamKDynamicDecomposition(problem, device, tiles);
    EXPECT_EQ(hinted.skTiles, 0u);
    EXPECT_EQ(hinted.grid, open.grid);
    EXPECT_EQ(hinted.grid, 256u);
}
