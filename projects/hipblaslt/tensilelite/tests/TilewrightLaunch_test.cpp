// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TilewrightLaunch: the launch plans TensileLite gives MLREC_v3 models. The
// expected plans are those of the training pipeline's launch mirror
// (training_pipeline/lib/hipblaslt_launch.py and its tests).
//
// TilewrightLaunchTest.DumpsPoolPlans writes the plans of a library's pool for
// training_pipeline/scripts/check_launch_plans.py to compare; it runs only
// when TILEWRIGHT_LAUNCH_PLANS_POOL and TILEWRIGHT_LAUNCH_PLANS_OUT are set,
// on the device and under the rule version dumpDevice() reads.

#include <gtest/gtest.h>

#if defined(TENSILELITE_HAS_TILEWRIGHT)

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/SolutionBlobCache.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/TilewrightRanker.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <origami/hardware.hpp>

using namespace TensileLite;

namespace
{
    using Launch = TilewrightLaunch<ContractionSolution>;
    using Plan   = std::tuple<bool, size_t, std::string>;

    // A device of `processor` with `cuCount` CUs. Launch plans read only the
    // architecture and the CU count of its analytical hardware; the other
    // values are the test's own.
    std::shared_ptr<hip::HipAMDGPU> makeDevice(AMDGPU::Processor processor, size_t cuCount)
    {
        using hardware_t = origami::hardware_t;
        auto const arch  = hardware_t::arch_name_to_enum(AMDGPU::toString(processor));
        if(arch == hardware_t::architecture_t::Count)
            throw std::invalid_argument("no analytical hardware for "
                                        + AMDGPU::toString(processor));
        auto device              = std::make_shared<hip::HipAMDGPU>();
        device->processor        = processor;
        device->computeUnitCount = static_cast<int>(cuCount);
        device->deviceName       = "tilewright-launch-test";
        device->analyticalHardware
            = std::make_shared<origami::hardware_t>(arch,
                                                    cuCount,
                                                    163840,
                                                    262144,
                                                    8,
                                                    1.0,
                                                    1.0,
                                                    1.0,
                                                    4000000,
                                                    1.2,
                                                    1,
                                                    std::make_tuple(0.0, 0.008, 0.0));
        return device;
    }

    struct Kernel
    {
        int    occupancy     = 1;
        int    strategy      = 0;
        int    assign        = 0;
        int    atomic        = 0;
        size_t wgX           = 256;
        size_t wgY           = 1;
        size_t packBatchDims = 0;
        size_t clusterX      = 1;
        bool   handwritten   = false;
        // The macro tile, per-tile extra iterations and a static
        // uniform-summation-order obstacle (UseInitialStridesCD).
        size_t mt               = 128;
        bool   perTile          = false;
        bool   initialStrides   = false;
        size_t workspacePerElem = 4;
    };

    std::shared_ptr<ContractionSolution> makeSolution(Kernel const& k)
    {
        auto  solution            = std::make_shared<ContractionSolution>();
        auto& sm                  = solution->sizeMapping;
        sm.tileProcessingStrategy = static_cast<TileProcessingStrategy>(k.strategy);
        sm.workAssignment         = static_cast<WorkAssignment>(k.assign);
        sm.streamKAtomic          = k.atomic;
        sm.macroTile              = TensileLite::dim3(k.mt, k.mt, 1);
        sm.workGroupSize          = TensileLite::dim3(k.wgX, k.wgY, 1);
        sm.threadTile             = TensileLite::dim3(1, 1, 1);
        sm.clusterDim             = TensileLite::dim3(k.clusterX, 1, 1);
        sm.depthU                 = 64;
        sm.matrixInstruction      = {16, 16, 32, 1};
        sm.CUOccupancy            = k.occupancy;
        sm.workspaceSizePerElemC  = k.workspacePerElem;
        sm.workGroupMapping       = 1;
        sm.globalSplitU           = 1;
        sm.packBatchDims          = k.packBatchDims;
        solution->internalArgsSupport.perTileExtraIters = k.perTile;
        solution->problemType.useInitialStridesCD       = k.initialStrides;
        if(k.handwritten)
            solution->customKernel.name = "tilewright-launch-test-kernel";
        return solution;
    }

    tilewright::Problem bf16Problem(size_t m, size_t n, size_t k, size_t batch = 1)
    {
        return {.size        = {m, n, k},
                .batch       = batch,
                .a_transpose = tilewright::Transpose::N,
                .b_transpose = tilewright::Transpose::N,
                .a_dtype     = tilewright::DataType::BFloat16,
                .b_dtype     = tilewright::DataType::BFloat16,
                .c_dtype     = tilewright::DataType::BFloat16,
                .d_dtype     = tilewright::DataType::BFloat16,
                .mi_dtype    = tilewright::DataType::BFloat16};
    }

    tilewright::ExecutionContext
        context(size_t cuBudget, tilewright::Schedule schedule = tilewright::Schedule::Default)
    {
        return {.cu_budget = cuBudget, .schedule = schedule};
    }

    constexpr size_t MiB       = size_t{1} << 20;
    constexpr size_t Unbounded = std::numeric_limits<size_t>::max();

    // A call with uniform summation order `uso` and `workspace` bytes.
    tilewright::ExecutionContext call(bool                 uso,
                                      size_t               workspace,
                                      size_t               cuBudget = 0,
                                      tilewright::Schedule schedule = tilewright::Schedule::Default)
    {
        return {.cu_budget               = cuBudget,
                .schedule                = schedule,
                .uniform_summation_order = uso,
                .workspace_bytes         = workspace};
    }

    std::string reductionName(tilewright::PartialReduction reduction)
    {
        switch(reduction)
        {
        case tilewright::PartialReduction::Tree:
            return "tree";
        case tilewright::PartialReduction::Parallel:
            return "parallel";
        case tilewright::PartialReduction::Atomic:
            return "atomic";
        default:
            return "none";
        }
    }

    Plan planOf(Kernel const&                       kernel,
                tilewright::Problem const&          problem,
                tilewright::ExecutionContext const& ctx,
                AMDGPU::Processor                   processor = AMDGPU::Processor::gfx950,
                int                                 version   = 0)
    {
        Launch launch("gfx950", {makeSolution(kernel)}, makeDevice(processor, 256), version);
        auto   plan = launch.plan(tilewright::Config{}, problem, ctx);
        // The plan from the kernel's record is the one of TensileLite's own
        // launch code.
        auto const resolved = launch.resolvedPlan(tilewright::Config{}, problem, ctx);
        EXPECT_EQ(plan.dynamic_queue, resolved.dynamic_queue);
        EXPECT_EQ(plan.grid_wgs, resolved.grid_wgs);
        EXPECT_EQ(plan.partial_reduction, resolved.partial_reduction);
        return {plan.dynamic_queue, plan.grid_wgs, reductionName(plan.partial_reduction)};
    }

    Kernel hybrid(int occupancy)
    {
        return {.occupancy = occupancy, .strategy = 2, .assign = 2};
    }

    // A static non-atomic Stream-K kernel of the spatial-cluster layout as
    // TensileLite's build gives it (PersistentLoopArgsVersion 2,
    // KernArgsVersion 3): ClusterDim [x, y, 1], a tileM x tileN macro tile of
    // depth depthU, 256 threads, four bytes per partial element.
    std::shared_ptr<ContractionSolution> makeClusterSolution(size_t x,
                                                             size_t y,
                                                             size_t tileM     = 256,
                                                             size_t tileN     = 256,
                                                             size_t depthU    = 64,
                                                             int    occupancy = 1,
                                                             bool   perTile   = false)
    {
        auto solution = makeSolution(
            {.occupancy = occupancy, .strategy = 2, .assign = 0, .perTile = perTile});
        auto& sm                              = solution->sizeMapping;
        sm.macroTile                          = TensileLite::dim3(tileM, tileN, 1);
        sm.clusterDim                         = TensileLite::dim3(x, y, 1);
        sm.depthU                             = depthU;
        solution->internalArgsSupport.version = 3;
        solution->internalArgsSupport.persistentLoopArgsVersion = 2;
        return solution;
    }

    std::string env(char const* name)
    {
        char const* value = std::getenv(name);
        return value ? value : "";
    }
}

TEST(TilewrightLaunchTest, IdNamesTheArchAndRuleVersion)
{
    EXPECT_EQ(Launch::RuleVersion, 4);
    EXPECT_EQ(Launch::TrainedRuleVersion, 3);
    EXPECT_EQ(Launch("gfx950", {}).id(), "hipblaslt:gfx950:3");
    EXPECT_EQ(Launch("gfx950", {}).version(), Launch::TrainedRuleVersion);
    EXPECT_NE(Launch("gfx942", {}).id(), Launch("gfx950", {}).id());
    EXPECT_EQ(Launch("gfx1250", {}).id(), "hipblaslt:gfx1250:4");
    EXPECT_EQ(Launch("gfx950", {}, nullptr, 1).id(), "hipblaslt:gfx950:1");
    EXPECT_EQ(Launch("gfx1250", {}, nullptr, 1).id(), "hipblaslt:gfx1250:1");
    EXPECT_EQ(Launch("gfx1250", {}, nullptr, 3).id(), "hipblaslt:gfx1250:3");
    EXPECT_EQ(Launch("gfx950", {}, nullptr, 3).id(), "hipblaslt:gfx950:3");
    EXPECT_EQ(Launch("gfx950", {}, nullptr, 3).version(), 3);
    EXPECT_EQ(Launch("gfx950", {}, nullptr, 4).id(), "hipblaslt:gfx950:4");
    // The error names the versions this class implements, the text the
    // pipeline's preflight looks for.
    EXPECT_EQ(std::string(Launch::RuleVersions),
              "tilewright launch: rule versions 1, 2, 3 and 4 only");
    for(int const bad : {-1, 5})
    {
        try
        {
            Launch("gfx950", {}, nullptr, bad);
            ADD_FAILURE() << bad;
        }
        catch(std::invalid_argument const& e)
        {
            EXPECT_STREQ(e.what(), Launch::RuleVersions) << bad;
        }
    }
}

// rule_version() of hipblaslt_launch.py: the tables its
// test_each_arch_trains_on_its_rule_version pins (RULE_1_ARCHS, RULE_4_ARCHS).
TEST(TilewrightLaunchTest, EachArchTrainsOnItsRuleVersion)
{
    for(char const* arch : {"gfx950", "gfx942", "gfx90a", "gfx1201"})
        EXPECT_EQ(Launch::ruleVersion(arch), Launch::TrainedRuleVersion) << arch;
    EXPECT_EQ(Launch::ruleVersion("gfx1250"), 4);
}

// served_rule_version() of hipblaslt_launch.py, for the same ids: the version a
// model records when this class implements it, else the arch's training one.
TEST(TilewrightLaunchTest, ServesAModelTheRuleVersionItRecords)
{
    EXPECT_EQ(Launch::servedRuleVersion("gfx950", "hipblaslt:gfx950:1"), 1);
    EXPECT_EQ(Launch::servedRuleVersion("gfx950", "hipblaslt:gfx950:2"), 2);
    EXPECT_EQ(Launch::servedRuleVersion("gfx950", ""), 3);
    EXPECT_EQ(Launch::servedRuleVersion("gfx950", "hipblaslt:gfx950:3"), 3);
    EXPECT_EQ(Launch::servedRuleVersion("gfx950", "hipblaslt:gfx950:4"), 4);
    for(char const* unknown :
        {"hipblaslt:gfx950:5", "hipblaslt:gfx950:0", "hipblaslt:gfx950:", "hipblaslt:gfx950:31"})
        EXPECT_EQ(Launch::servedRuleVersion("gfx950", unknown), 3) << unknown;
    EXPECT_EQ(Launch::servedRuleVersion("gfx1250", "hipblaslt:gfx1250:3"), 3);
    EXPECT_EQ(Launch::servedRuleVersion("gfx1250", "hipblaslt:gfx1250:4"), 4);
    EXPECT_EQ(Launch::servedRuleVersion("gfx950", "hipblaslt:gfx942:1"), 3);
    EXPECT_EQ(Launch::servedRuleVersion("gfx950", "other:gfx950:1"), 3);
    EXPECT_EQ(Launch::servedRuleVersion("gfx1250", "hipblaslt:gfx1250:1"), 1);
    EXPECT_EQ(Launch::servedRuleVersion("gfx1250", "hipblaslt:gfx1250:2"), 2);
    EXPECT_EQ(Launch::servedRuleVersion("gfx1250", "hipblaslt:gfx1250:5"), 4);
    EXPECT_EQ(Launch::servedRuleVersion("gfx1250", ""), 4);
    EXPECT_EQ(Launch::servedRuleVersion("gfx942", "hipblaslt:gfx942:1"), 1);
}

// The plans of hipblaslt_launch.py's test_one_k_iteration_launches_by_rule_version
// under rule version 2, which equal the pipeline's, and under version 1, which
// keep Origami's grid and report a partial reduction for every grid that does
// not divide the tiles.
TEST(TilewrightLaunchTest, OneKIterationLaunchesByRuleVersion)
{
    Kernel const sk{.occupancy = 1, .strategy = 2, .assign = 0};
    Kernel const sk2{.occupancy = 2, .strategy = 2, .assign = 0};
    Kernel const sk3{.occupancy = 3, .strategy = 2, .assign = 0};
    Kernel const busy{.occupancy = 4, .strategy = 2, .assign = 0};
    Kernel const atomic{.occupancy = 1, .strategy = 2, .assign = 0, .atomic = 1};
    Kernel const dp{.occupancy = 2, .strategy = 1, .assign = 0};
    struct Case
    {
        Kernel              kernel;
        tilewright::Problem problem;
        size_t              budget;
        Plan                v1;
        Plan                v2;
    };
    for(auto const& c : {
            Case{sk, bf16Problem(5120, 5120, 64), 0, {false, 246, "tree"}, {false, 246, "none"}},
            Case{busy,
                 bf16Problem(7680, 5120, 64),
                 64,
                 {false, 2400, "none"},
                 {false, 2400, "none"}},
            Case{sk, bf16Problem(7680, 5120, 64), 64, {false, 64, "tree"}, {false, 64, "none"}},
            Case{atomic,
                 bf16Problem(7680, 5120, 17),
                 64,
                 {false, 64, "atomic"},
                 {false, 64, "none"}},
            Case{
                sk, bf16Problem(7680, 5120, 128), 64, {false, 2400, "none"}, {false, 2400, "none"}},
            Case{sk,
                 bf16Problem(128 * 300, 128, 64 * 300),
                 96,
                 {false, 86, "tree"},
                 {false, 86, "tree"}},
            // One workgroup per tile exactly when the static grid has more than
            // two workgroups per usable CU, for a DataParallel kernel and a
            // Hybrid one on a static grid too.
            Case{sk2,
                 bf16Problem(128 * 1024, 128, 64),
                 0,
                 {false, 512, "none"},
                 {false, 512, "none"}},
            Case{sk3,
                 bf16Problem(128 * 9216, 128, 64),
                 0,
                 {false, 9216, "none"},
                 {false, 9216, "none"}},
            Case{dp,
                 bf16Problem(128 * 1100, 128, 64),
                 0,
                 {false, 440, "none"},
                 {false, 440, "none"}},
            Case{hybrid(2),
                 bf16Problem(128 * 1100, 128, 64),
                 0,
                 {false, 440, "tree"},
                 {false, 440, "none"}},
            Case{sk2,
                 bf16Problem(128 * 1100, 128, 64, 2),
                 0,
                 {false, 489, "tree"},
                 {false, 489, "none"}},
        })
    {
        auto const ctx = context(c.budget);
        EXPECT_EQ(planOf(c.kernel, c.problem, ctx, AMDGPU::Processor::gfx950, 1), c.v1)
            << c.problem.size.m << " x " << c.problem.size.n << " x " << c.problem.size.k;
        EXPECT_EQ(planOf(c.kernel, c.problem, ctx, AMDGPU::Processor::gfx950, 2), c.v2)
            << c.problem.size.m << " x " << c.problem.size.n << " x " << c.problem.size.k;
        EXPECT_EQ(planOf(c.kernel, c.problem, ctx), c.v2);
    }
}

TEST(TilewrightLaunchTest, HybridThresholds)
{
    auto const sm224 = context(224);
    EXPECT_EQ(planOf(hybrid(2), bf16Problem(128 * 480, 128, 512), sm224), Plan(false, 320, "tree"));
    EXPECT_EQ(planOf(hybrid(2), bf16Problem(128 * 481, 128, 512), sm224), Plan(true, 481, "none"));
    EXPECT_EQ(planOf(hybrid(3), bf16Problem(4096, 4096, 512), sm224), Plan(true, 768, "none"));
    EXPECT_EQ(planOf(hybrid(4), bf16Problem(4096, 4096, 512), sm224), Plan(false, 1024, "none"));
    EXPECT_EQ(planOf(hybrid(4), bf16Problem(128 * 1883, 128, 512), sm224),
              Plan(false, 1883, "none"));
    EXPECT_EQ(planOf(hybrid(4), bf16Problem(128 * 1884, 128, 512), sm224), Plan(true, 768, "none"));
}

TEST(TilewrightLaunchTest, SmTargetAndSchedules)
{
    using tilewright::Schedule;
    auto const plan = [](size_t            budget,
                         Schedule          schedule  = Schedule::Default,
                         AMDGPU::Processor processor = AMDGPU::Processor::gfx950) {
        return planOf(
            hybrid(2), bf16Problem(4096, 4096, 512), context(budget, schedule), processor);
    };
    EXPECT_EQ(plan(256), Plan(false, 512, "none"));
    for(size_t budget : {31, 224, 240, 255})
        EXPECT_EQ(plan(budget), Plan(true, 512, "none")) << budget;
    EXPECT_EQ(plan(0), Plan(false, 512, "none"));
    EXPECT_EQ(plan(224, Schedule::Auto), Plan(true, 512, "none"));
    EXPECT_EQ(plan(0, Schedule::Auto), Plan(false, 512, "none"));
    EXPECT_EQ(plan(0, Schedule::Dynamic), Plan(true, 512, "none"));
    EXPECT_EQ(plan(224, Schedule::Default, AMDGPU::Processor::gfx942), Plan(false, 1024, "none"));
    EXPECT_EQ(plan(224, Schedule::Dynamic, AMDGPU::Processor::gfx942), Plan(true, 512, "none"));
}

TEST(TilewrightLaunchTest, StreamKGrids)
{
    auto const excl   = context(0);
    auto       atomic = hybrid(1);
    atomic.atomic     = 1;
    auto huge         = hybrid(1);
    huge.wgX = huge.wgY = 1024;
    Kernel const dp{.occupancy = 2, .strategy = 1, .assign = 0};

    EXPECT_EQ(planOf(hybrid(1), bf16Problem(256, 256, 64 * 64), excl), Plan(false, 32, "parallel"));
    EXPECT_EQ(planOf(hybrid(1), bf16Problem(256, 256, 64 * 32), excl), Plan(false, 16, "tree"));
    EXPECT_EQ(planOf(atomic, bf16Problem(256, 256, 64 * 32), excl), Plan(false, 16, "atomic"));
    EXPECT_EQ(planOf(dp, bf16Problem(4096, 4096, 512), excl), Plan(false, 512, "none"));
    EXPECT_EQ(planOf({.occupancy = 2}, bf16Problem(4096, 4096, 512), excl),
              Plan(false, 1024, "none"));
    EXPECT_EQ(planOf(hybrid(1), bf16Problem(128 * 300, 128, 64 * 300), excl),
              Plan(false, 200, "tree"));
    EXPECT_EQ(planOf(huge, bf16Problem(128 * 300, 128, 64 * 300), excl), Plan(false, 300, "none"));
    huge.occupancy = 9;
    EXPECT_EQ(planOf(huge, bf16Problem(128 * 4500, 128, 512), excl), Plan(false, 2048, "tree"));
    EXPECT_EQ(planOf(hybrid(2), bf16Problem(1024, 1024, 512, 16), excl), Plan(false, 512, "none"));
}

TEST(TilewrightLaunchTest, DynamicQueueCountersShortenTheFlagRegion)
{
    auto const problem = bf16Problem(128 * 4096, 128 * 4096, 64);
    EXPECT_EQ(planOf(hybrid(9), problem, context(0)), Plan(false, 2048, "none"));
    EXPECT_EQ(planOf(hybrid(9), problem, context(0, tilewright::Schedule::Dynamic)),
              Plan(true, 2048 - 8 * 128 / 4, "none"));
}

TEST(TilewrightLaunchTest, NonPersistentKernelsLaunchOneWorkgroupPerTile)
{
    for(Kernel const& kernel : {Kernel{.occupancy = 2}, Kernel{.occupancy = 2, .assign = 2}})
    {
        EXPECT_EQ(planOf(kernel, bf16Problem(4096, 1000, 512), context(0)),
                  Plan(false, 256, "none"));
        EXPECT_EQ(planOf(kernel, bf16Problem(4096, 1000, 512, 3), context(0)),
                  Plan(false, 768, "none"));
    }
}

TEST(TilewrightLaunchTest, PlansTheScopeProblemOnTheScopeDevice)
{
    auto const scopeDevice = makeDevice(AMDGPU::Processor::gfx950, 256);
    Launch     launch("gfx950", {makeSolution(hybrid(2))});
    auto const problem = bf16Problem(4096, 4096, 512);
    EXPECT_THROW(launch.plan(tilewright::Config{}, problem, context(224)), std::logic_error);

    auto gemm = ContractionProblemGemm::GEMM(
        false, false, 4096, 4096, 512, 4096, 512, 4096, 1.0, false, 1);
    gemm.setComputeInputTypeA(rocisa::DataType::BFloat16);
    gemm.setComputeInputTypeB(rocisa::DataType::BFloat16);
    gemm.setWorkspaceSize(0);
    {
        TilewrightLaunchScope<ContractionProblemGemm> scope(gemm, *scopeDevice);
        auto plan = launch.plan(tilewright::Config{}, problem, context(224));
        EXPECT_TRUE(plan.dynamic_queue);
        EXPECT_EQ(plan.grid_wgs, 512u);
        plan = launch.plan(tilewright::Config{}, problem, context(0));
        EXPECT_FALSE(plan.dynamic_queue);
    }
    EXPECT_EQ(TilewrightLaunchScope<ContractionProblemGemm>::active(), nullptr);
    tilewright::Config outside{};
    outside.index = 1;
    EXPECT_THROW(
        Launch("gfx950", {makeSolution(hybrid(2))}, scopeDevice).plan(outside, problem, context(0)),
        std::out_of_range);
}

// Expected keys: lib/hipblaslt_launch.launch_key of the same kernels under
// rule version 2 (LaunchKeysOfRuleVersionThree pins version 3's). A
// hand-written Stream-K kernel that can run on a static grid, whose partial
// tiles TensileLite reduces in a tree only, hashes a sixth launch input, 1.
TEST(TilewrightLaunchTest, LaunchKeysMatchThePipeline)
{
    auto const key = [](Kernel const& kernel) {
        return Launch("gfx950", {makeSolution(kernel)}, nullptr, 2)
            .launch_key(tilewright::Config{});
    };
    auto const byHand = [](Kernel kernel) {
        kernel.handwritten = true;
        return kernel;
    };
    auto atomic   = hybrid(2);
    atomic.atomic = 1;
    EXPECT_EQ(key(hybrid(2)), 0x138281954b1a0008ull);
    EXPECT_EQ(key(atomic), 0x30a54cb3ed1d0e1dull);
    EXPECT_EQ(key({.occupancy = 2, .strategy = 1}), 0x862a415d97fba99dull);
    EXPECT_EQ(key({.occupancy = 2, .strategy = 1, .atomic = 1}), 0x862a415d97fba99dull);
    EXPECT_EQ(key({.occupancy = 2}), 0x40d69e0cf0f65c45ull);
    EXPECT_EQ(key({.occupancy = 5, .assign = 2, .atomic = 1}), 0x40d69e0cf0f65c45ull);
    EXPECT_EQ(key(hybrid(0)), 0x7072d6b06be7de6bull);
    EXPECT_EQ(key(hybrid(1)), 0x7072d6b06be7de6bull);

    Kernel const staticGrid{.occupancy = 2, .strategy = 2, .assign = 0};
    Kernel const queue{.occupancy = 2, .strategy = 2, .assign = 1};
    EXPECT_EQ(key(staticGrid), 0x9a1f0386bcc3a98eull);
    EXPECT_EQ(key(queue), 0xd890c4fa793a6183ull);
    EXPECT_EQ(key(byHand(hybrid(2))), 0xfd52fc8b94c39b29ull);
    EXPECT_EQ(key(byHand(atomic)), 0xac0da9653af6e99cull);
    EXPECT_EQ(key(byHand(staticGrid)), 0x76836540c6bd316full);
    EXPECT_EQ(key(byHand(hybrid(0))), 0xac57afd1b8244faaull);
    EXPECT_EQ(key(byHand(hybrid(1))), 0xac57afd1b8244faaull);
    // A queue hands out whole tiles, and the other kernels split none.
    EXPECT_EQ(key(byHand(queue)), 0xd890c4fa793a6183ull);
    EXPECT_EQ(key(byHand({.occupancy = 2, .strategy = 1})), 0x862a415d97fba99dull);
    EXPECT_EQ(key(byHand({.occupancy = 2})), 0x40d69e0cf0f65c45ull);

    // A kernel generated from a custom kernel's config is not hand-written.
    EXPECT_TRUE(Launch::treeOnly(*makeSolution(byHand(hybrid(2)))));
    auto generated                    = makeSolution(byHand(hybrid(2)));
    generated->customKernel.generated = true;
    EXPECT_FALSE(Launch::treeOnly(*generated));
    EXPECT_EQ(Launch("gfx950", {generated}, nullptr, 2).launch_key(tilewright::Config{}),
              0x138281954b1a0008ull);

    tilewright::Config outside{};
    outside.index = 1;
    EXPECT_THROW(Launch("gfx950", {makeSolution(hybrid(2))}).launch_key(outside),
                 std::out_of_range);
}

// The uniform-summation-order and workspace steps that ContractionSolution and
// resolveDefaultPersistentLaunch() share (DefaultPersistentLaunch.hpp).
TEST(TilewrightLaunchTest, UniformReductionKeepsParallelOnlyWithoutObstacles)
{
    using origami::reduction_t;
    using PersistentLaunch::streamKUniformReduction;
    int        asked    = 0;
    auto const obstacle = [&asked](bool value) {
        return [&asked, value] {
            ++asked;
            return value;
        };
    };
    EXPECT_EQ(streamKUniformReduction(reduction_t::parallel, true, false, obstacle(false)),
              reduction_t::parallel);
    EXPECT_EQ(streamKUniformReduction(reduction_t::parallel, true, false, obstacle(true)),
              reduction_t::tree);
    EXPECT_EQ(asked, 2);
    // An atomic kernel or a tree launch is not asked about obstacles.
    EXPECT_EQ(streamKUniformReduction(reduction_t::parallel, true, true, obstacle(false)),
              reduction_t::tree);
    EXPECT_EQ(streamKUniformReduction(reduction_t::tree, true, false, obstacle(false)),
              reduction_t::tree);
    // Without static two-tile packing the reduction is kept.
    for(reduction_t const r : {reduction_t::parallel, reduction_t::tree, reduction_t::none})
        EXPECT_EQ(streamKUniformReduction(r, false, true, obstacle(true)), r);
    EXPECT_EQ(asked, 2);
}

TEST(TilewrightLaunchTest, UniformGridSnapsOntoAdmissibleGrids)
{
    using origami::reduction_t;
    using PersistentLaunch::streamKUniformGrid;
    constexpr size_t any  = std::numeric_limits<size_t>::max();
    auto const       tree = reduction_t::tree, parallel = reduction_t::parallel;
    struct Row
    {
        size_t      g0, tiles, iters, cluster;
        bool        perTile;
        reduction_t reduction;
        size_t      bytes, workspace;
        bool        flags;
        size_t      grid;
    };
    Row const rows[] = {
        // Never upward: below the tiles one workgroup per tile, at the tiles unchanged.
        {200, 300, 64, 1, true, tree, 4, any, true, 300},
        {300, 300, 64, 1, true, tree, 4, any, true, 300},
        // No tiles or no grid: unchanged.
        {64, 0, 64, 1, true, tree, 4, any, true, 64},
        {0, 16, 64, 1, true, tree, 4, any, true, 0},
        // The largest factor up to 64 / 16 that passes every test.
        {64, 16, 64, 1, true, tree, 1, any, false, 64},
        // Whole clusters.
        {64, 16, 64, 3, true, tree, 1, any, false, 48},
        // F | I, without per-tile extras and for tree only.
        {64, 16, 42, 1, false, tree, 1, any, false, 48},
        {64, 16, 35, 1, false, tree, 1, any, false, 16},
        {64, 16, 35, 1, true, tree, 1, any, false, 64},
        {64, 16, 35, 1, false, parallel, 1, any, false, 64},
        // At least MinItersPerCU (8) iterations per workgroup.
        {64, 16, 24, 1, true, tree, 1, any, false, 48},
        {64, 16, 15, 1, true, tree, 1, any, false, 16},
        {256, 1, 64, 1, true, tree, 1, any, false, 8},
        // The partial tiles fit the workspace.
        {64, 16, 64, 1, true, tree, 1000, 48000, false, 48},
        {64, 16, 64, 1, true, tree, 1000, 0, false, 16},
        // Inside the flag region, when it binds.
        {4096, 1024, 64, 1, true, tree, 1, any, true, 2048},
        {4096, 1024, 64, 1, true, tree, 1, any, false, 4096},
    };
    for(Row const& r : rows)
        EXPECT_EQ(streamKUniformGrid(r.g0,
                                     r.tiles,
                                     r.iters,
                                     r.cluster,
                                     r.perTile,
                                     r.reduction,
                                     r.bytes,
                                     r.workspace,
                                     r.flags),
                  r.grid)
            << r.g0 << " " << r.tiles << " " << r.iters << " " << r.cluster << " " << r.perTile
            << " " << static_cast<int>(r.reduction) << " " << r.bytes << " " << r.workspace << " "
            << r.flags;
}

TEST(TilewrightLaunchTest, WorkspaceFitFallsBackToOneWorkgroupPerTile)
{
    using origami::reduction_t;
    using PersistentLaunch::streamKWorkspaceFit;
    auto const   tree = reduction_t::tree, parallel = reduction_t::parallel;
    size_t const tile = 256 * 256 * 4;
    struct Row
    {
        size_t      grid;
        reduction_t reduction;
        size_t      tiles, iters, workspace;
        bool        streamKDP, dpOnly;
        size_t      wantGrid;
        reduction_t wantReduction;
    };
    Row const rows[] = {
        // A grid that straddles tiles needs a partial tile per workgroup.
        {200, tree, 300, 64, 200 * tile, false, false, 200, tree},
        {200, tree, 300, 64, 200 * tile - 1, false, false, 300, tree},
        // Unless StreamKDP or DataParallel, the tiles split evenly or a tile
        // takes one iteration.
        {200, tree, 300, 64, 0, true, false, 200, tree},
        {200, tree, 300, 64, 0, false, true, 200, tree},
        {150, tree, 300, 64, 0, false, false, 150, tree},
        {200, tree, 300, 1, 0, false, false, 200, tree},
        // A parallel reduction always needs them.
        {256, parallel, 16, 128, 256 * tile, false, false, 256, parallel},
        {256, parallel, 16, 128, 128 * tile, false, false, 16, tree},
        {256, parallel, 16, 128, 128 * tile, true, true, 16, tree},
        // No grid.
        {0, parallel, 16, 128, 0, false, false, 0, parallel},
    };
    for(Row const& r : rows)
    {
        size_t      grid      = r.grid;
        reduction_t reduction = r.reduction;
        bool const  fell      = streamKWorkspaceFit(
            grid, reduction, r.tiles, r.iters, tile, r.workspace, r.streamKDP, r.dpOnly);
        EXPECT_EQ(grid, r.wantGrid) << r.grid << " " << r.workspace;
        EXPECT_EQ(reduction, r.wantReduction) << r.grid << " " << r.workspace;
        EXPECT_EQ(fell, grid != r.grid) << r.grid << " " << r.workspace;
    }
}

// uniformSummationOrderKernelObstacle(): the static obstacles of a Stream-K
// launch the kernel decides alone, which uniform summation order refuses.
TEST(TilewrightLaunchTest, KernelObstacleNamesWhatTheKernelDecides)
{
    auto const device = makeDevice(AMDGPU::Processor::gfx950, 256);
    auto       gemm   = ContractionProblemGemm::GEMM(
        false, false, 4096, 4096, 512, 4096, 512, 4096, 1.0, false, 1);
    gemm.setComputeInputTypeA(rocisa::DataType::BFloat16);
    gemm.setComputeInputTypeB(rocisa::DataType::BFloat16);
    gemm.setParams().setUniformSummationOrder(1);
    auto const kernelToken = [](ContractionSolution const& s) -> std::string {
        char const* token = s.uniformSummationOrderKernelObstacle();
        return token != nullptr ? token : "";
    };
    auto const with = [](Kernel kernel, auto&& edit) {
        auto solution                     = makeSolution(kernel);
        solution->sizeMapping.LocalSplitU = 1;
        edit(*solution);
        return solution;
    };
    auto const none   = [](ContractionSolution&) {};
    Kernel     atomic = hybrid(2);
    atomic.atomic     = 1;
    Kernel packedC0 = hybrid(2), packedC1 = hybrid(2), both = atomic;
    packedC0.packBatchDims = 1;
    packedC1.packBatchDims = 2;
    both.packBatchDims     = 1;
    EXPECT_EQ(kernelToken(*with(hybrid(2), none)), "");
    std::vector<std::pair<std::string, std::shared_ptr<ContractionSolution>>> const obstructed = {
        {"StreamKAtomic", with(atomic, none)},
        {"StreamKAtomic", with(both, none)},
        {"UseInitialStridesCD",
         with(hybrid(2), [](ContractionSolution& s) { s.problemType.useInitialStridesCD = true; })},
        {"PackedC0Index", with(packedC0, none)},
        {"PackedC1Index", with(packedC1, none)},
        {"WaveSplitK",
         with(hybrid(2),
              [](ContractionSolution& s) {
                  s.sizeMapping.workGroupSize = TensileLite::dim3(128, 1, 2);
              })},
    };
    for(auto const& [token, solution] : obstructed)
    {
        EXPECT_EQ(kernelToken(*solution), token);
        EXPECT_FALSE(solution->uniformSummationOrderSupported(gemm, *device)) << token;
    }
    // The MX envelope, from the problem type the solution holds.
    auto const mx = [&](int format, int block, int instK, int depthU) {
        return kernelToken(*with(hybrid(2), [&](ContractionSolution& s) {
            s.problemType.mxBlockA             = block;
            s.problemType.mxScaleFormat        = format;
            s.sizeMapping.matrixInstruction[2] = instK;
            s.sizeMapping.depthU               = depthU;
        }));
    };
    EXPECT_EQ(mx(0, 32, 128, 256), "MXScaleFormat");
    EXPECT_EQ(mx(3, 32, 128, 256), "MXScaleFormat");
    EXPECT_EQ(mx(1, 16, 128, 256), "MXBlockSize");
    EXPECT_EQ(mx(1, 32, 32, 256), "MXMatrixInstK");
    EXPECT_EQ(mx(1, 32, 128, 384), "MXScaleSwizzleGranule");
    EXPECT_EQ(mx(1, 32, 128, 96), "MXScaleSwizzleGranule");
    EXPECT_EQ(mx(1, 32, 128, 256), "");
    // Stream-K cuts these in whole scale epochs of two and four iterations.
    EXPECT_EQ(mx(1, 32, 128, 128), "");
    EXPECT_EQ(mx(1, 32, 128, 64), "");
    // In-memory swizzled scales (format 2) meet no MX clause; the
    // host-preswizzled ones (format 1) meet each.
    EXPECT_EQ(mx(2, 32, 128, 256), "");
    EXPECT_EQ(mx(2, 16, 32, 96), "");
    EXPECT_EQ(mx(1, 16, 32, 96), "MXBlockSize");
}

// Expected keys: lib/hipblaslt_launch.launch_key at rule version 3 of the same
// kernels. A Stream-K kernel hashes nine values; other kernels keep their keys.
TEST(TilewrightLaunchTest, LaunchKeysOfRuleVersionThree)
{
    auto const key = [](Kernel const& kernel, int version = 3) {
        return Launch("gfx950", {makeSolution(kernel)}, nullptr, version)
            .launch_key(tilewright::Config{});
    };
    auto perTile            = hybrid(2);
    perTile.perTile         = true;
    auto atomic             = hybrid(2);
    atomic.atomic           = 1;
    auto narrow             = hybrid(2);
    narrow.workspacePerElem = 2;
    auto obstacle           = hybrid(2);
    obstacle.initialStrides = true;
    auto byHand             = hybrid(2);
    byHand.handwritten      = true;
    Kernel const staticGrid{.occupancy = 2, .strategy = 2, .assign = 0};
    Kernel       queue{.occupancy = 2, .strategy = 2, .assign = 1};
    queue.perTile = true;
    EXPECT_EQ(key(hybrid(2)), 0x683f2c52bc20650cull);
    EXPECT_EQ(key(perTile), 0x8ee6d58e798a4f6dull);
    EXPECT_EQ(key(atomic), 0xb35a98b051c5b178ull);
    EXPECT_EQ(key(staticGrid), 0x87f665da871b8f8aull);
    EXPECT_EQ(key(queue), 0xedda592695024407ull);
    EXPECT_EQ(key(byHand), 0x6d97df395f16a58dull);
    EXPECT_EQ(key(narrow), 0xca4300c295f6348aull);
    EXPECT_EQ(key(obstacle), 0x8739f35bc70faf2dull);
    EXPECT_EQ(key({.occupancy = 2, .strategy = 1}), 0x862a415d97fba99dull);
    EXPECT_EQ(key({.occupancy = 2}), 0x40d69e0cf0f65c45ull);
    // Versions 1 and 2 keep their keys whatever the new inputs.
    for(int const version : {1, 2})
        for(Kernel const& k : {perTile, narrow, obstacle})
            EXPECT_EQ(key(k, version), 0x138281954b1a0008ull) << version;
}

// The worked examples of the launch rule's version 3 (a 256-CU gfx950, 256x256
// tiles of depth 64 and four bytes per partial element): plans under the
// call's uniform summation order and workspace, which planOf() also checks
// against resolvePersistentSettings().
TEST(TilewrightLaunchTest, RuleVersionThreePlansTheCallsSummationOrderAndWorkspace)
{
    Kernel const sk{.occupancy = 1, .strategy = 2, .assign = 0, .mt = 256, .perTile = true};
    Kernel       noPerTile  = sk;
    noPerTile.perTile       = false;
    Kernel obstacle         = sk;
    obstacle.initialStrides = true;
    Kernel queue            = sk;
    queue.assign            = 1;
    queue.occupancy         = 3;
    Kernel hybridSk         = sk;
    hybridSk.assign         = 2;
    Kernel const dp{.occupancy = 1, .strategy = 1, .assign = 0, .mt = 256};
    auto const   plan = [](Kernel const& k, tilewright::Problem const& p, auto const& c, int v) {
        return planOf(k, p, c, AMDGPU::Processor::gfx950, v);
    };

    // A: 300 tiles of 64 iterations.
    auto const a = bf16Problem(3840, 5120, 4096);
    EXPECT_EQ(plan(sk, a, context(0), 2), Plan(false, 200, "tree"));
    EXPECT_EQ(plan(sk, a, call(false, Unbounded), 3), Plan(false, 200, "tree"));
    EXPECT_EQ(plan(sk, a, call(false, 32 * MiB), 3), Plan(false, 300, "none"));
    EXPECT_EQ(plan(sk, a, call(true, 128 * MiB), 3), Plan(false, 300, "none"));
    // B: 16 tiles of 35 iterations.
    auto const b = bf16Problem(1024, 1024, 2240);
    EXPECT_EQ(plan(sk, b, context(0), 2), Plan(false, 64, "tree"));
    EXPECT_EQ(plan(sk, b, call(false, 8 * MiB), 3), Plan(false, 16, "none"));
    EXPECT_EQ(plan(sk, b, call(true, 128 * MiB), 3), Plan(false, 64, "tree"));
    EXPECT_EQ(plan(noPerTile, b, call(true, 128 * MiB), 3), Plan(false, 16, "none"));
    // D: a dynamic queue of 1000 tiles whose partials outgrow 128 MiB.
    auto const d = bf16Problem(6400, 10240, 4096);
    EXPECT_EQ(plan(queue, d, context(0), 2), Plan(true, 768, "none"));
    EXPECT_EQ(plan(queue, d, call(false, Unbounded), 3), Plan(true, 768, "none"));
    for(bool const uso : {false, true})
        EXPECT_EQ(plan(queue, d, call(uso, 128 * MiB), 3), Plan(true, 1000, "none")) << uso;
    // E: 16 tiles of 128 iterations, parallel.
    auto const e = bf16Problem(1024, 1024, 8192);
    EXPECT_EQ(plan(sk, e, context(0), 2), Plan(false, 256, "parallel"));
    EXPECT_EQ(plan(sk, e, call(false, 32 * MiB), 3), Plan(false, 16, "none"));
    EXPECT_EQ(plan(sk, e, call(true, 128 * MiB), 3), Plan(false, 256, "parallel"));
    EXPECT_EQ(plan(obstacle, e, call(true, 128 * MiB), 3), Plan(false, 256, "tree"));
    // F: one iteration per tile.
    auto const f = bf16Problem(3840, 5120, 64);
    EXPECT_EQ(plan(sk, f, context(0), 2), Plan(false, 200, "none"));
    EXPECT_EQ(plan(sk, f, call(false, 0), 3), Plan(false, 300, "none"));
    EXPECT_EQ(plan(sk, f, call(true, 128 * MiB), 3), Plan(false, 300, "none"));
    // G: Hybrid at half the CUs takes the dynamic sub-mode, which takes no snap.
    EXPECT_EQ(plan(hybridSk, d, call(false, 128 * MiB, 128), 3),
              plan(hybridSk, d, call(true, 128 * MiB, 128), 3));
    EXPECT_EQ(plan(hybridSk, d, call(true, 128 * MiB, 128), 3), Plan(true, 256, "none"));
    // H: DataParallel reads the workspace through Origami, never the order.
    EXPECT_EQ(plan(dp, a, context(0), 2), Plan(false, 200, "none"));
    for(bool const uso : {false, true})
    {
        EXPECT_EQ(plan(dp, a, call(uso, 32 * MiB), 3), Plan(false, 300, "none")) << uso;
        EXPECT_EQ(plan(dp, a, call(uso, Unbounded), 3), Plan(false, 200, "none")) << uso;
    }
}

// Versions 1 and 2 plan without the call's summation order and workspace, and
// version 3 at neither (off, unbounded) plans as version 2.
TEST(TilewrightLaunchTest, OlderRuleVersionsIgnoreTheSummationOrderAndWorkspace)
{
    std::vector<Kernel> kernels = {hybrid(1), hybrid(2)};
    Kernel              sk{.occupancy = 1, .strategy = 2, .assign = 0, .mt = 256, .perTile = true};
    kernels.push_back(sk);
    sk.assign = 1;
    kernels.push_back(sk);
    sk.assign = 0;
    sk.atomic = 1;
    kernels.push_back(sk);
    kernels.push_back({.occupancy = 2, .strategy = 1});
    std::mt19937_64 rng(3);
    auto const draw = [&](size_t hi) { return std::uniform_int_distribution<size_t>(1, hi)(rng); };
    for(int i = 0; i < 40; ++i)
    {
        auto const   p      = bf16Problem(64 * draw(160), 64 * draw(160), 64 * draw(256), draw(3));
        size_t const budget = std::array<size_t, 4>{0, 64, 128, 224}[i % 4];
        for(Kernel const& k : kernels)
        {
            Plan const v2 = planOf(k, p, context(budget), AMDGPU::Processor::gfx950, 2);
            EXPECT_EQ(planOf(k, p, call(false, Unbounded, budget), AMDGPU::Processor::gfx950, 3),
                      v2);
            for(int const version : {1, 2})
            {
                Plan const plain
                    = planOf(k, p, context(budget), AMDGPU::Processor::gfx950, version);
                for(bool const uso : {false, true})
                    for(size_t const workspace : {size_t{0}, MiB, 128 * MiB})
                        EXPECT_EQ(planOf(k,
                                         p,
                                         call(uso, workspace, budget),
                                         AMDGPU::Processor::gfx950,
                                         version),
                                  plain)
                            << version << " " << uso << " " << workspace;
            }
        }
    }
}

// The ranker's context is the call's: its problem's uniform summation order
// and workspace (TilewrightRanker::executionContext()), with which plan()
// gives what resolvePersistentSettings() launches on the scope's device.
TEST(TilewrightLaunchTest, PlansTheScopeProblemUnderItsSummationOrderAndWorkspace)
{
    auto const   device = makeDevice(AMDGPU::Processor::gfx950, 256);
    Kernel const sk{.occupancy = 1, .strategy = 2, .assign = 0, .mt = 256, .perTile = true};
    Launch const v3("gfx950", {makeSolution(sk)}, nullptr, 3);
    auto const   problem = bf16Problem(3840, 5120, 4096);
    auto const   bf16    = rocisa::DataType::BFloat16;
    size_t const m = 3840, n = 5120, k = 4096;
    auto         gemm = ContractionProblemGemm::GEMM_Strides(false,
                                                     false,
                                                     bf16,
                                                     bf16,
                                                     bf16,
                                                     bf16,
                                                     m,
                                                     n,
                                                     k,
                                                     1,
                                                     m,
                                                     m * k,
                                                     k,
                                                     k * n,
                                                     m,
                                                     m * n,
                                                     m,
                                                     m * n,
                                                     0.0);
    gemm.setComputeInputTypeA(bf16);
    gemm.setComputeInputTypeB(bf16);
    for(bool const uso : {false, true})
        for(size_t const workspace : {Unbounded, 128 * MiB, 32 * MiB, size_t{0}})
        {
            gemm.setParams().setUniformSummationOrder(uso ? 1 : 0);
            gemm.setWorkspaceSize(workspace);
            auto const ctx = TilewrightRanker::executionContext(gemm);
            EXPECT_EQ(ctx.uniform_summation_order, uso);
            EXPECT_EQ(ctx.workspace_bytes, workspace);
            TilewrightLaunchScope<ContractionProblemGemm> scope(gemm, *device);
            auto const plan     = v3.plan(tilewright::Config{}, problem, ctx);
            auto const resolved = v3.resolvedPlan(tilewright::Config{}, problem, ctx);
            EXPECT_EQ(plan.grid_wgs, resolved.grid_wgs) << uso << " " << workspace;
            EXPECT_EQ(plan.partial_reduction, resolved.partial_reduction)
                << uso << " " << workspace;
            EXPECT_EQ(plan.dynamic_queue, resolved.dynamic_queue) << uso << " " << workspace;
            bool const whole = uso || workspace < 64 * MiB;
            EXPECT_EQ(plan.grid_wgs, whole ? 300u : 200u) << uso << " " << workspace;
        }
}

// One thread plans with launches of versions 1 and 3 and with both summation
// orders: the launch problem it reuses is the one each asks for.
TEST(TilewrightLaunchTest, OneThreadPlansRuleVersionsAndSummationOrdersApart)
{
    auto const   device = makeDevice(AMDGPU::Processor::gfx950, 256);
    Kernel const sk{.occupancy = 1, .strategy = 2, .assign = 0, .mt = 256, .perTile = true};
    Launch const v1("gfx950", {makeSolution(sk)}, device, 1);
    Launch const v3("gfx950", {makeSolution(sk)}, device, 3);
    auto const   problem  = bf16Problem(3840, 5120, 4096);
    auto const   resolved = [&](Launch const& launch, tilewright::ExecutionContext const& c) {
        auto const p = launch.resolvedPlan(tilewright::Config{}, problem, c);
        return Plan(p.dynamic_queue, p.grid_wgs, reductionName(p.partial_reduction));
    };
    auto const on = call(true, 128 * MiB), off = call(false, Unbounded);
    for(int round = 0; round < 2; ++round)
    {
        EXPECT_EQ(resolved(v3, on), Plan(false, 300, "none"));
        EXPECT_EQ(resolved(v1, on), Plan(false, 200, "tree"));
        EXPECT_EQ(resolved(v3, off), Plan(false, 200, "tree"));
        EXPECT_EQ(resolved(v3, on), Plan(false, 300, "none"));
        EXPECT_EQ(resolved(v1, off), Plan(false, 200, "tree"));
    }
}

// TENSILE_STREAMK_DATA_PARALLEL, read once per process, changes the workspace
// fit, so a version 3 plan is resolvePersistentSettings()'s under it. The test
// runs itself in a child process with the setting.
TEST(TilewrightLaunchTest, StreamKDataParallelLeavesRuleVersionThreeToTheLaunch)
{
    char const* const child = "TILEWRIGHT_LAUNCH_TEST_STREAMK_DP";
    if(env(child).empty())
    {
        std::string const self    = std::filesystem::read_symlink("/proc/self/exe").string();
        std::string const command = std::string("TENSILE_STREAMK_DATA_PARALLEL=1 ") + child + "=1 '"
                                    + self
                                    + "' "
                                      "--gtest_filter=TilewrightLaunchTest."
                                      "StreamKDataParallelLeavesRuleVersionThreeToTheLaunch"
                                    + " > /dev/null 2>&1";
        EXPECT_EQ(std::system(command.c_str()), 0);
        return;
    }
    ASSERT_TRUE(Debug::Instance().useStreamKDataParrallel());
    // A dynamic queue's straddling grid writes no partials under StreamKDP, so
    // a workspace too small for them leaves it as it is.
    Kernel const queue{.occupancy = 3, .strategy = 2, .assign = 1, .mt = 256};
    auto const   d = bf16Problem(6400, 10240, 4096);
    for(bool const uso : {false, true})
        EXPECT_EQ(planOf(queue, d, call(uso, 128 * MiB), AMDGPU::Processor::gfx950, 3),
                  Plan(true, 768, "none"))
            << uso;
    EXPECT_EQ(planOf(queue, d, context(0), AMDGPU::Processor::gfx950, 2), Plan(true, 768, "none"));
}

namespace
{
    bool samePlan(tilewright::LaunchPlan const& a, tilewright::LaunchPlan const& b)
    {
        return a.dynamic_queue == b.dynamic_queue && a.grid_wgs == b.grid_wgs
               && a.partial_reduction == b.partial_reduction;
    }

    bool samePlans(std::vector<tilewright::LaunchPlan> const& a,
                   std::vector<tilewright::LaunchPlan> const& b)
    {
        return std::equal(a.begin(), a.end(), b.begin(), b.end(), samePlan);
    }

    // Sets `name` (unsets it for nullptr) and re-reads the debug settings for
    // the lifetime of the object.
    class ScopedDebugEnv
    {
    public:
        ScopedDebugEnv(char const* name, char const* value)
            : m_name(name)
        {
            if(char const* previous = std::getenv(name))
                m_previous = previous;
            set(value);
        }

        ~ScopedDebugEnv()
        {
            set(m_previous ? m_previous->c_str() : nullptr);
        }

    private:
        void set(char const* value)
        {
            if(value)
                setenv(m_name.c_str(), value, 1);
            else
                unsetenv(m_name.c_str());
            Debug::Instance().reloadDebugBitsForTest();
        }

        std::string                m_name;
        std::optional<std::string> m_previous;
    };

    std::vector<Kernel> persistentKernels()
    {
        return {hybrid(1),
                hybrid(4),
                {.occupancy = 2, .strategy = 2, .assign = 0},
                {.occupancy = 2, .strategy = 2, .assign = 1},
                {.occupancy = 1, .strategy = 2, .assign = 0, .atomic = 1},
                {.occupancy = 2, .strategy = 1, .assign = 0}};
    }

    std::vector<std::pair<tilewright::Problem, tilewright::ExecutionContext>> probes()
    {
        std::vector<std::pair<tilewright::Problem, tilewright::ExecutionContext>> out;
        for(auto const& problem : {bf16Problem(4096, 4096, 512),
                                   bf16Problem(128 * 20, 128, 64 * 96),
                                   bf16Problem(256, 256, 64 * 64),
                                   bf16Problem(128 * 4500, 128, 512),
                                   bf16Problem(1024, 1024, 512, 16),
                                   bf16Problem(128 * 1884, 128, 512)})
            for(size_t const budget : {0, 224, 240})
                for(auto const schedule : {tilewright::Schedule::Default,
                                           tilewright::Schedule::Dynamic,
                                           tilewright::Schedule::Auto})
                    out.emplace_back(problem, context(budget, schedule));
        return out;
    }

    // The plans of `kernels` at every probe on `device`, each checked against
    // resolvedPlan(), with followsRule() expected to be `rule`.
    std::vector<tilewright::LaunchPlan> checkedPlans(std::vector<Kernel> const&      kernels,
                                                     std::shared_ptr<hip::HipAMDGPU> device,
                                                     bool                            rule)
    {
        std::vector<tilewright::LaunchPlan> plans;
        for(auto const& kernel : kernels)
        {
            Launch const launch("gfx950", {makeSolution(kernel)}, device);
            EXPECT_EQ(launch.followsRule(0, *device), rule)
                << "strategy " << kernel.strategy << " assignment " << kernel.assign;
            for(auto const& [problem, ctx] : probes())
            {
                auto const plan = launch.plan(tilewright::Config{}, problem, ctx);
                EXPECT_TRUE(samePlan(plan, launch.resolvedPlan(tilewright::Config{}, problem, ctx)))
                    << "strategy " << kernel.strategy << " assignment " << kernel.assign << " m "
                    << problem.size.m << " cu_budget " << ctx.cu_budget << " schedule "
                    << static_cast<int>(ctx.schedule);
                plans.push_back(plan);
            }
        }
        return plans;
    }
}

TEST(TilewrightLaunchTest, PlansFromRecordsOnlyUnderDefaultLaunchSettings)
{
    auto const kernels  = persistentKernels();
    auto const defaults = checkedPlans(kernels, makeDevice(AMDGPU::Processor::gfx950, 256), true);

    auto const offDefault = [&](char const* what, auto&& change) {
        auto device = makeDevice(AMDGPU::Processor::gfx950, 256);
        change(*device);
        EXPECT_FALSE(samePlans(checkedPlans(kernels, device, false), defaults)) << what;
    };
    offDefault("fixed grid", [](hip::HipAMDGPU& d) { d.persistentFixedGrid = 64; });
    offDefault("CU-count grid", [](hip::HipAMDGPU& d) { d.persistentDynamicGrid = 0; });
    offDefault("min-resources grid", [](hip::HipAMDGPU& d) { d.persistentDynamicGrid = 1; });
    offDefault("CU cap", [](hip::HipAMDGPU& d) { d.persistentMaxCUs = 128; });

    auto multiplied                      = makeDevice(AMDGPU::Processor::gfx950, 256);
    multiplied->persistentGridMultiplier = 4;
    EXPECT_TRUE(samePlans(checkedPlans(kernels, multiplied, true), defaults));

    auto fewerCus              = makeDevice(AMDGPU::Processor::gfx950, 256);
    fewerCus->computeUnitCount = 240;
    EXPECT_FALSE(samePlans(checkedPlans(kernels, fewerCus, true), defaults));

    // reloadDebugBitsForTest() refreshes the one Debug that TensileLite reads,
    // so a forced Hybrid sub-mode reaches the launch: it is a debug setting of
    // defaultPersistentLaunchDebug(), and the plans come from
    // resolvePersistentSettings().
    for(auto const* value : {"0", "1"})
    {
        ScopedDebugEnv const env("TENSILE_STREAMK5_FORCE_MODE", value);
        EXPECT_FALSE(samePlans(
            checkedPlans(kernels, makeDevice(AMDGPU::Processor::gfx950, 256), false), defaults))
            << value;
    }
    {
        ScopedDebugEnv const env("TENSILE_STREAMK5_FORCE_MODE", "-1");
        EXPECT_TRUE(samePlans(
            checkedPlans(kernels, makeDevice(AMDGPU::Processor::gfx950, 256), true), defaults));
    }

    auto unmodelled                = makeDevice(AMDGPU::Processor::gfx950, 256);
    unmodelled->analyticalHardware = nullptr;
    EXPECT_FALSE(Launch("gfx950", {makeSolution(hybrid(2))}).followsRule(0, *unmodelled));
    EXPECT_FALSE(Launch("gfx950", {makeSolution(hybrid(2))}).followsRule(0, AMDGPU()));
    EXPECT_TRUE(Launch("gfx950", {makeSolution({.occupancy = 2})}).followsRule(0, *unmodelled));
}

// The plans take the CU budget as ContractionProblem::setSmCountTarget() keeps
// it, not rounded down to a multiple of 32, each the plan
// resolvePersistentSettings() gives: 31 CUs are fewer than the device has, not
// every CU.
TEST(TilewrightLaunchTest, PlansTakeTheCuBudgetAsItIs)
{
    auto const device  = makeDevice(AMDGPU::Processor::gfx950, 256);
    auto const problem = bf16Problem(4096, 4096, 512);
    for(auto const& kernel : persistentKernels())
    {
        Launch const launch("gfx950", {makeSolution(kernel)}, device);
        ASSERT_TRUE(launch.followsRule(0, *device));
        for(size_t const budget : {31, 100, 240})
        {
            auto const ctx = context(budget);
            EXPECT_TRUE(samePlan(launch.plan(tilewright::Config{}, problem, ctx),
                                 launch.resolvedPlan(tilewright::Config{}, problem, ctx)))
                << "strategy " << kernel.strategy << " assignment " << kernel.assign
                << " cu_budget " << budget;
        }
    }
    Launch const launch("gfx950", {makeSolution(hybrid(2))}, device);
    EXPECT_FALSE(samePlan(launch.plan(tilewright::Config{}, problem, context(31)),
                          launch.plan(tilewright::Config{}, problem, context(0))));
}

TEST(TilewrightLaunchTest, PlansFromRecordsOnlyKernelsTheRecordHolds)
{
    auto const device = makeDevice(AMDGPU::Processor::gfx950, 256);
    auto const plans  = [&](Kernel const& kernel, tilewright::Problem const& problem, bool rule) {
        Launch const launch("gfx950", {makeSolution(kernel)}, device);
        EXPECT_EQ(launch.followsRule(0, *device), rule);
        auto const plan = launch.plan(tilewright::Config{}, problem, context(0));
        EXPECT_TRUE(samePlan(plan, launch.resolvedPlan(tilewright::Config{}, problem, context(0))));
        return plan;
    };

    auto const batched = bf16Problem(1000, 128, 512, 16);
    for(Kernel kernel : {Kernel{.occupancy = 2}, hybrid(2)})
    {
        auto const unpacked  = plans(kernel, batched, true);
        kernel.packBatchDims = 1;
        EXPECT_FALSE(samePlan(plans(kernel, batched, false), unpacked)) << kernel.strategy;
    }
    plans({.occupancy = 2, .strategy = 1, .packBatchDims = 1}, batched, false);

    plans({.occupancy = 2, .strategy = 1, .clusterX = 2}, bf16Problem(4096, 4096, 512), false);
    plans({.occupancy = 2, .strategy = 1, .clusterX = 1}, bf16Problem(4096, 4096, 512), true);
    plans({.occupancy = 2, .strategy = 2, .clusterX = 2}, bf16Problem(4096, 4096, 512), false);

    auto handwritten               = makeSolution(hybrid(2));
    handwritten->customKernel.name = "tilewright-launch-test-kernel";
    EXPECT_FALSE(Launch("gfx950", {handwritten}, device).followsRule(0, *device));
    handwritten->customKernel.generated = true;
    EXPECT_TRUE(Launch("gfx950", {handwritten}, device).followsRule(0, *device));

    Launch const rejected("gfx950", {makeSolution({.strategy = 1, .assign = 2})}, device);
    EXPECT_FALSE(rejected.followsRule(0, *device));
    EXPECT_THROW(rejected.plan(tilewright::Config{}, bf16Problem(4096, 4096, 512), context(0)),
                 std::runtime_error);
}

TEST(TilewrightLaunchTest, ScopesOfOneThreadPlanForTheirOwnDevice)
{
    auto const fixed           = makeDevice(AMDGPU::Processor::gfx950, 256);
    fixed->persistentFixedGrid = 64;
    auto const usual           = makeDevice(AMDGPU::Processor::gfx950, 256);
    Launch     launch("gfx950", {makeSolution(hybrid(2))});

    auto gemm = ContractionProblemGemm::GEMM(
        false, false, 4096, 4096, 512, 4096, 512, 4096, 1.0, false, 1);
    gemm.setComputeInputTypeA(rocisa::DataType::BFloat16);
    gemm.setComputeInputTypeB(rocisa::DataType::BFloat16);
    auto const problem = bf16Problem(4096, 4096, 512);
    for(int round = 0; round < 2; ++round)
    {
        {
            TilewrightLaunchScope<ContractionProblemGemm> scope(gemm, *fixed);
            auto const plan = launch.plan(tilewright::Config{}, problem, context(0));
            EXPECT_EQ(plan.grid_wgs, 64u);
            EXPECT_TRUE(
                samePlan(plan, launch.resolvedPlan(tilewright::Config{}, problem, context(0))));
        }
        {
            TilewrightLaunchScope<ContractionProblemGemm> scope(gemm, *usual);
            auto const plan = launch.plan(tilewright::Config{}, problem, context(0));
            EXPECT_EQ(plan.grid_wgs, 512u);
            EXPECT_TRUE(
                samePlan(plan, launch.resolvedPlan(tilewright::Config{}, problem, context(0))));
        }
    }
}

// The plans of hipblaslt_launch.py's SPATIAL_CLUSTER_PLANS: kernels of the
// spatial-cluster layout planned under rule version 4, with no ranking scope,
// by resolveClusteredStreamKSettings(), whose plan is the physical workgroups
// of its logical clusters and the reduction of its schedule. Versions 1 to 3
// keep reporting the logical clusters it launches.
TEST(TilewrightLaunchTest, PlansSpatialClustersAsTheirSchedule)
{
    struct Case
    {
        size_t m, n, k, batch, tileM, tileN, depthU;
        int    occupancy;
        size_t x, y;
        bool   perTile;
        size_t cuCount, budget;
        bool   uso;
        size_t workspace;
        Plan   want;
    };
    std::vector<Case> const cases = {
        {1024,
         256,
         64 * 300,
         1,
         256,
         256,
         64,
         1,
         2,
         2,
         true,
         256,
         0,
         true,
         Unbounded,
         {false, 256, "parallel"}},
        {1000,
         256,
         4096,
         4,
         256,
         256,
         64,
         1,
         4,
         1,
         true,
         256,
         0,
         false,
         Unbounded,
         {false, 128, "parallel"}},
        {1024,
         1024,
         64 * 300,
         1,
         256,
         256,
         64,
         2,
         2,
         4,
         false,
         96,
         192,
         false,
         Unbounded,
         {false, 96, "parallel"}},
        {8192,
         8192,
         16384,
         4,
         256,
         256,
         64,
         2,
         2,
         1,
         true,
         96,
         0,
         false,
         128 * MiB,
         {false, 190, "tree"}},
        {8192,
         1024,
         64 * 300,
         4,
         256,
         256,
         64,
         1,
         8,
         1,
         true,
         256,
         192,
         false,
         Unbounded,
         {false, 192, "tree"}},
        {1000,
         4096,
         16384,
         4,
         256,
         128,
         256,
         2,
         4,
         4,
         true,
         256,
         192,
         false,
         Unbounded,
         {false, 336, "tree"}},
        {2048,
         256,
         4096,
         1,
         128,
         128,
         128,
         1,
         2,
         1,
         true,
         96,
         0,
         true,
         Unbounded,
         {false, 96, "tree"}},
        {1000,
         1000,
         16384,
         1,
         128,
         128,
         128,
         2,
         4,
         4,
         true,
         256,
         192,
         true,
         Unbounded,
         {false, 192, "tree"}},
        {8192,
         256,
         64 * 65536,
         1,
         256,
         256,
         64,
         1,
         8,
         1,
         false,
         256,
         64,
         true,
         Unbounded,
         {false, 32, "none"}},
        {16384,
         256,
         64 * 300,
         1,
         256,
         128,
         256,
         1,
         2,
         1,
         true,
         96,
         64,
         false,
         4 * MiB,
         {false, 64, "none"}},
        {2048,
         4096,
         16384,
         1,
         256,
         256,
         64,
         1,
         2,
         2,
         false,
         256,
         192,
         true,
         Unbounded,
         {false, 128, "none"}},
        {128 * 481,
         1000,
         0,
         1,
         128,
         128,
         128,
         2,
         2,
         2,
         false,
         96,
         0,
         true,
         Unbounded,
         {false, 192, "none"}},
        {256,
         256,
         4096,
         1,
         256,
         128,
         256,
         1,
         8,
         1,
         true,
         96,
         0,
         false,
         Unbounded,
         {false, 16, "none"}},
        {256,
         1000,
         64 * 65536,
         4,
         256,
         256,
         64,
         2,
         8,
         1,
         true,
         96,
         0,
         true,
         128 * MiB,
         {false, 128, "none"}},
    };
    for(auto const& c : cases)
    {
        auto const solution
            = makeClusterSolution(c.x, c.y, c.tileM, c.tileN, c.depthU, c.occupancy, c.perTile);
        ASSERT_TRUE(solution->usesStreamKSpatialCluster());
        auto const   device  = makeDevice(AMDGPU::Processor::gfx1250, c.cuCount);
        auto const   problem = bf16Problem(c.m, c.n, c.k, c.batch);
        auto const   ctx     = call(c.uso, c.workspace, c.budget);
        Launch const launch("gfx1250", {solution}, device, 4);
        auto const   plan = launch.plan(tilewright::Config{}, problem, ctx);
        EXPECT_EQ(Plan(plan.dynamic_queue, plan.grid_wgs, reductionName(plan.partial_reduction)),
                  c.want)
            << c.m << 'x' << c.n << 'x' << c.k << " batch " << c.batch << " cluster " << c.x << 'x'
            << c.y;
        Launch const older("gfx1250", {solution}, device, 3);
        EXPECT_EQ(older.plan(tilewright::Config{}, problem, ctx).grid_wgs * c.x * c.y,
                  plan.grid_wgs)
            << c.m << 'x' << c.n << 'x' << c.k;
    }
    // A cluster of more workgroups than the device's resident budget holds
    // has no launch.
    Launch const small(
        "gfx1250", {makeClusterSolution(8, 1)}, makeDevice(AMDGPU::Processor::gfx1250, 4), 4);
    EXPECT_THROW(
        small.plan(tilewright::Config{}, bf16Problem(8192, 8192, 4096), call(false, Unbounded)),
        std::runtime_error);
}

// launch_key() of hipblaslt_launch.py under rule version 4 for kernels of the
// spatial-cluster layout, which test_launch_keys_of_rule_version_four_are_those_tensilelite_pins
// pins: the version 3 values, then the layout, the cluster and the unit of
// K. Version 3 keeps its key, which the cluster does not change.
TEST(TilewrightLaunchTest, LaunchKeysOfRuleVersionFour)
{
    auto const key = [](std::shared_ptr<ContractionSolution> const& solution, int version) {
        return Launch("gfx1250", {solution}, nullptr, version).launch_key(tilewright::Config{});
    };
    auto const plain                    = makeClusterSolution(2, 1);
    auto const perTile                  = makeClusterSolution(2, 1, 256, 256, 64, 1, true);
    auto const square                   = makeClusterSolution(4, 4, 256, 256, 64, 1, true);
    auto const subtile                  = makeClusterSolution(2, 1);
    subtile->sizeMapping.useSubtileImpl = true;
    auto const scaled                   = makeClusterSolution(8, 1);
    scaled->problemType.mxScaleFormat   = 1;
    ASSERT_EQ(scaled->skKGranularity(), 4u);
    EXPECT_EQ(key(plain, 4), 0x0663633002d68789ull);
    EXPECT_EQ(key(perTile, 4), 0x4be1bafa8af4c8e8ull);
    EXPECT_EQ(key(square, 4), 0xe6ced38ab1e5622bull);
    EXPECT_EQ(key(subtile, 4), 0x57ce1e5ec6f562c8ull);
    EXPECT_EQ(key(scaled, 4), 0x7b65c0e4b328e7a6ull);
    EXPECT_EQ(key(plain, 3), 0xe95a5d101448f4e9ull);
    EXPECT_EQ(key(square, 3), 0xc2b2b3d456df0a88ull);
    EXPECT_EQ(key(subtile, 3), key(plain, 3));
}

namespace
{
    // The device DumpsPoolPlans plans for, and the rule version it plans
    // under: TILEWRIGHT_LAUNCH_PLANS_PROCESSOR (default gfx950) with
    // TILEWRIGHT_LAUNCH_PLANS_CU_COUNT CUs (default 256), under
    // TILEWRIGHT_LAUNCH_PLANS_RULE_VERSION (default the version the
    // processor's models train on).
    struct DumpDevice
    {
        AMDGPU::Processor processor = AMDGPU::Processor::gfx950;
        std::string       arch      = "gfx950";
        size_t            cuCount   = 256;
        int               version   = 0;
    };

    DumpDevice dumpDevice()
    {
        DumpDevice d;
        if(std::string const name = env("TILEWRIGHT_LAUNCH_PLANS_PROCESSOR"); !name.empty())
        {
            d.processor = AMDGPU::toProcessor(name);
            if(d.processor == AMDGPU::Processor::gfx000)
                throw std::invalid_argument("unknown processor " + name);
            d.arch = AMDGPU::toString(d.processor);
        }
        if(std::string const cus = env("TILEWRIGHT_LAUNCH_PLANS_CU_COUNT"); !cus.empty())
            d.cuCount = std::stoul(cus);
        std::string const version = env("TILEWRIGHT_LAUNCH_PLANS_RULE_VERSION");
        d.version = version.empty() ? Launch::ruleVersion(d.arch) : std::stoi(version);
        return d;
    }

    // The CU budgets of the contexts DumpsPoolPlans draws: every CU (0), and
    // a quarter, three eighths, half, three quarters and seven eighths of the
    // device's CUs.
    std::array<size_t, 6> dumpBudgets(size_t cuCount)
    {
        return {0, cuCount / 4, cuCount * 3 / 8, cuCount / 2, cuCount * 3 / 4, cuCount * 7 / 8};
    }
}

TEST(TilewrightLaunchTest, DumpsPlansOfTheDeviceAndRuleVersionItIsGiven)
{
    {
        ScopedDebugEnv const processor("TILEWRIGHT_LAUNCH_PLANS_PROCESSOR", nullptr);
        ScopedDebugEnv const cus("TILEWRIGHT_LAUNCH_PLANS_CU_COUNT", nullptr);
        ScopedDebugEnv const version("TILEWRIGHT_LAUNCH_PLANS_RULE_VERSION", nullptr);
        auto const           d = dumpDevice();
        EXPECT_EQ(d.processor, AMDGPU::Processor::gfx950);
        EXPECT_EQ(d.arch, "gfx950");
        EXPECT_EQ(d.cuCount, 256u);
        EXPECT_EQ(d.version, Launch::TrainedRuleVersion);
        EXPECT_EQ(dumpBudgets(d.cuCount), (std::array<size_t, 6>{0, 64, 96, 128, 192, 224}));
    }
    // A CU count no device of these tests has.
    ScopedDebugEnv const processor("TILEWRIGHT_LAUNCH_PLANS_PROCESSOR", "gfx1250");
    ScopedDebugEnv const cus("TILEWRIGHT_LAUNCH_PLANS_CU_COUNT", "300");
    {
        ScopedDebugEnv const version("TILEWRIGHT_LAUNCH_PLANS_RULE_VERSION", nullptr);
        auto const           d = dumpDevice();
        EXPECT_EQ(d.processor, AMDGPU::Processor::gfx1250);
        EXPECT_EQ(d.arch, "gfx1250");
        EXPECT_EQ(d.cuCount, 300u);
        EXPECT_EQ(d.version, 4);
        EXPECT_EQ(dumpBudgets(d.cuCount), (std::array<size_t, 6>{0, 75, 112, 150, 225, 262}));
        auto const device = makeDevice(d.processor, d.cuCount);
        EXPECT_EQ(device->analyticalHardware->arch, origami::hardware_t::architecture_t::gfx1250);
        Launch const launch(d.arch, {makeSolution(hybrid(1))}, device, d.version);
        EXPECT_EQ(launch.id(), "hipblaslt:gfx1250:4");
        // As many tiles as the device has CUs: one workgroup each, where
        // planOf's gfx950 device streams them.
        auto const problem = bf16Problem(128 * 300, 128, 64 * 300);
        auto const plan    = launch.plan(tilewright::Config{}, problem, context(0));
        EXPECT_EQ(Plan(plan.dynamic_queue, plan.grid_wgs, reductionName(plan.partial_reduction)),
                  Plan(false, 300, "none"));
        EXPECT_EQ(planOf(hybrid(1), problem, context(0)), Plan(false, 200, "tree"));
    }
    {
        ScopedDebugEnv const version("TILEWRIGHT_LAUNCH_PLANS_RULE_VERSION", "1");
        EXPECT_EQ(dumpDevice().version, 1);
    }
    ScopedDebugEnv const unknown("TILEWRIGHT_LAUNCH_PLANS_PROCESSOR", "gfx9999");
    EXPECT_THROW(dumpDevice(), std::invalid_argument);
}

// Pool file: "library <dir>/<stem>" lines (the logic file without .dat[.zlib];
// a solution index resolves to the first listed library that holds it), then
// "pool <solution index> ..." in pool order. The plans file starts with
// "# launch <launch model id> cu_count <CUs>"; each plan line names its
// context's uniform summation order and workspace ("-" for one that holds
// every partial tile), which only rule version 3 plans under.
TEST(TilewrightLaunchTest, DumpsPoolPlans)
{
    std::string const poolPath = env("TILEWRIGHT_LAUNCH_PLANS_POOL");
    std::string const outPath  = env("TILEWRIGHT_LAUNCH_PLANS_OUT");
    if(poolPath.empty() || outPath.empty())
        GTEST_SKIP() << "set TILEWRIGHT_LAUNCH_PLANS_POOL and TILEWRIGHT_LAUNCH_PLANS_OUT";
    int const  samples = env("TILEWRIGHT_LAUNCH_PLANS_SAMPLES").empty()
                             ? 64
                             : std::stoi(env("TILEWRIGHT_LAUNCH_PLANS_SAMPLES"));
    auto const seed    = env("TILEWRIGHT_LAUNCH_PLANS_SEED").empty()
                             ? 0ull
                             : std::stoull(env("TILEWRIGHT_LAUNCH_PLANS_SEED"));

    using Master = MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>;
    std::vector<std::shared_ptr<SolutionLibrary<ContractionProblemGemm>>> libraries;
    std::vector<int>                                                      poolIndices;
    std::ifstream                                                         in(poolPath);
    ASSERT_TRUE(in) << poolPath;
    for(std::string line; std::getline(in, line);)
    {
        std::istringstream words(line);
        std::string        word;
        words >> word;
        if(word == "library")
        {
            std::string path;
            words >> path;
            auto library = LoadLibraryFile<ContractionProblemGemm>(path + ".dat");
            ASSERT_TRUE(library) << path;
            libraries.push_back(library);
        }
        else if(word == "pool")
        {
            for(int index; words >> index;)
                poolIndices.push_back(index);
        }
    }
    ASSERT_FALSE(libraries.empty());
    ASSERT_FALSE(poolIndices.empty());

    std::vector<std::shared_ptr<ContractionSolution>> pool;
    for(int index : poolIndices)
    {
        std::shared_ptr<ContractionSolution> solution;
        for(auto const& library : libraries)
        {
            auto const* master = dynamic_cast<Master const*>(library.get());
            ASSERT_NE(master, nullptr);
            if(master->blobCache)
                solution = master->blobCache->get(index);
            else if(auto it = master->solutions.find(index); it != master->solutions.end())
                solution = it->second;
            if(solution)
                break;
        }
        ASSERT_TRUE(solution) << "no library holds solution " << index;
        pool.push_back(solution);
    }

    auto const&      type   = pool.front()->problemType;
    DumpDevice const dumped = dumpDevice();
    auto const       device = makeDevice(dumped.processor, dumped.cuCount);
    Launch const     launch(dumped.arch, pool, device, dumped.version);
    auto const       toTw    = datatypeToTilewrightDatatype;
    auto const       miDtype = type.f32XdlMathOp == rocisa::DataType::XFloat32
                                   ? tilewright::DataType::XFloat32
                                   : toTw(type.computeInputTypeA);
    auto const       budgets = dumpBudgets(dumped.cuCount);
    std::mt19937_64  rng(seed);
    auto const       logUniform = [&](double hi) {
        return static_cast<size_t>(
            std::exp(std::uniform_real_distribution<double>(0.0, std::log(hi))(rng)));
    };

    std::ofstream out(outPath);
    ASSERT_TRUE(out) << outPath;
    out << "# launch " << launch.id() << " cu_count " << dumped.cuCount << '\n';
    out << "# pos solution m n k batch mi_dtype cu_budget schedule uso workspace dynamic_queue "
           "grid_wgs partial_reduction mt_m mt_n depth_u occupancy\n";
    size_t const workspaces[] = {Unbounded, 128 * MiB, MiB, 0};
    for(size_t pos = 0; pos < pool.size(); ++pos)
    {
        auto const&        sm = pool[pos]->sizeMapping;
        tilewright::Config config{};
        config.index = pos;
        out << "key " << pos << ' ' << pool[pos]->index << ' ' << launch.launch_key(config) << '\n';
        for(int s = 0; s < samples; ++s)
        {
            size_t const              batch = std::uniform_int_distribution<int>(0, 9)(rng) == 0
                                                  ? size_t(1) << std::uniform_int_distribution<int>(1, 6)(rng)
                                                  : 1;
            tilewright::Problem const problem{
                .size        = {logUniform(65536.0), logUniform(65536.0), logUniform(131072.0)},
                .batch       = batch,
                .a_transpose = type.transA ? tilewright::Transpose::T : tilewright::Transpose::N,
                .b_transpose = type.transB ? tilewright::Transpose::T : tilewright::Transpose::N,
                .a_dtype     = toTw(type.aType),
                .b_dtype     = toTw(type.bType),
                .c_dtype     = toTw(type.cType),
                .d_dtype     = toTw(type.dType),
                .mi_dtype    = miDtype};
            size_t const budget = budgets[std::uniform_int_distribution<int>(0, 5)(rng)];
            auto const   schedule
                = static_cast<tilewright::Schedule>(std::uniform_int_distribution<int>(0, 2)(rng));
            bool const   uso       = std::uniform_int_distribution<int>(0, 1)(rng) == 1;
            size_t const workspace = workspaces[std::uniform_int_distribution<int>(0, 3)(rng)];
            auto const   ctx       = call(uso, workspace, budget, schedule);
            auto const   plan      = launch.plan(config, problem, ctx);
            out << pos << ' ' << pool[pos]->index << ' ' << problem.size.m << ' ' << problem.size.n
                << ' ' << problem.size.k << ' ' << problem.batch << ' '
                << static_cast<int>(problem.mi_dtype) << ' ' << ctx.cu_budget << ' '
                << static_cast<int>(ctx.schedule) << ' ' << ctx.uniform_summation_order << ' '
                << (workspace == Unbounded ? std::string("-") : std::to_string(workspace)) << ' '
                << plan.dynamic_queue << ' ' << plan.grid_wgs << ' '
                << reductionName(plan.partial_reduction) << ' ' << sm.macroTile.x << ' '
                << sm.macroTile.y << ' ' << sm.depthU << ' ' << sm.CUOccupancy << '\n';
        }
    }
    ASSERT_TRUE(out.good());
}

#endif
