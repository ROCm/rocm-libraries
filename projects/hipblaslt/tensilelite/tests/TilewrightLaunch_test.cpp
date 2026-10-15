// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// TilewrightLaunch: the launch plans TensileLite gives MLREC_v3 models. The
// expected plans are those of the training pipeline's launch mirror
// (training_pipeline/lib/hipblaslt_launch.py and its tests).
//
// TilewrightLaunchTest.DumpsPoolPlans writes the plans of a library's pool for
// training_pipeline/scripts/check_launch_plans.py to compare; it runs only
// when TILEWRIGHT_LAUNCH_PLANS_POOL and TILEWRIGHT_LAUNCH_PLANS_OUT are set.

#include <gtest/gtest.h>

#if defined(TENSILELITE_HAS_TILEWRIGHT)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
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

    std::shared_ptr<hip::HipAMDGPU> makeDevice(AMDGPU::Processor processor, size_t cuCount)
    {
        using arch_t    = origami::hardware_t::architecture_t;
        auto const arch = processor == AMDGPU::Processor::gfx942 ? arch_t::gfx942 : arch_t::gfx950;
        auto       device        = std::make_shared<hip::HipAMDGPU>();
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
    };

    std::shared_ptr<ContractionSolution> makeSolution(Kernel const& k)
    {
        auto  solution            = std::make_shared<ContractionSolution>();
        auto& sm                  = solution->sizeMapping;
        sm.tileProcessingStrategy = static_cast<TileProcessingStrategy>(k.strategy);
        sm.workAssignment         = static_cast<WorkAssignment>(k.assign);
        sm.streamKAtomic          = k.atomic;
        sm.macroTile              = TensileLite::dim3(128, 128, 1);
        sm.workGroupSize          = TensileLite::dim3(k.wgX, k.wgY, 1);
        sm.threadTile             = TensileLite::dim3(1, 1, 1);
        sm.clusterDim             = TensileLite::dim3(k.clusterX, 1, 1);
        sm.depthU                 = 64;
        sm.matrixInstruction      = {16, 16, 32, 1};
        sm.CUOccupancy            = k.occupancy;
        sm.workspaceSizePerElemC  = 4;
        sm.workGroupMapping       = 1;
        sm.globalSplitU           = 1;
        sm.packBatchDims          = k.packBatchDims;
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
                AMDGPU::Processor                   processor = AMDGPU::Processor::gfx950)
    {
        Launch launch("gfx950", {makeSolution(kernel)}, makeDevice(processor, 256));
        auto   plan = launch.plan(tilewright::Config{}, problem, ctx);
        return {plan.dynamic_queue, plan.grid_wgs, reductionName(plan.partial_reduction)};
    }

    Kernel hybrid(int occupancy)
    {
        return {.occupancy = occupancy, .strategy = 2, .assign = 2};
    }

    std::string env(char const* name)
    {
        char const* value = std::getenv(name);
        return value ? value : "";
    }
}

TEST(TilewrightLaunchTest, IdNamesTheArchAndRuleVersion)
{
    EXPECT_EQ(Launch("gfx950", {}).id(), "hipblaslt:gfx950:" + std::to_string(Launch::RuleVersion));
    EXPECT_NE(Launch("gfx942", {}).id(), Launch("gfx950", {}).id());
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

// Expected keys: lib/hipblaslt_launch.launch_key of the same kernels.
TEST(TilewrightLaunchTest, LaunchKeysMatchThePipeline)
{
    auto const key = [](Kernel const& kernel) {
        return Launch("gfx950", {makeSolution(kernel)}).launch_key(tilewright::Config{});
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
    tilewright::Config outside{};
    outside.index = 1;
    EXPECT_THROW(Launch("gfx950", {makeSolution(hybrid(2))}).launch_key(outside),
                 std::out_of_range);
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
    plans({.occupancy = 2, .strategy = 2, .clusterX = 2}, bf16Problem(4096, 4096, 512), true);

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

// Pool file: "library <dir>/<stem>" lines (the logic file without .dat[.zlib];
// a solution index resolves to the first listed library that holds it), then
// "pool <solution index> ..." in pool order.
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

    auto const&     type    = pool.front()->problemType;
    size_t const    cuCount = 256;
    auto const      device  = makeDevice(AMDGPU::Processor::gfx950, cuCount);
    Launch const    launch("gfx950", pool, device);
    auto const      toTw      = datatypeToTilewrightDatatype;
    auto const      miDtype   = type.f32XdlMathOp == rocisa::DataType::XFloat32
                                    ? tilewright::DataType::XFloat32
                                    : toTw(type.computeInputTypeA);
    size_t const    budgets[] = {0, 64, 96, 128, 192, 224};
    std::mt19937_64 rng(seed);
    auto const      logUniform = [&](double hi) {
        return static_cast<size_t>(
            std::exp(std::uniform_real_distribution<double>(0.0, std::log(hi))(rng)));
    };

    std::ofstream out(outPath);
    ASSERT_TRUE(out) << outPath;
    out << "# pos solution m n k batch mi_dtype cu_budget schedule dynamic_queue grid_wgs "
           "partial_reduction mt_m mt_n depth_u occupancy\n";
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
            auto const ctx = context(
                budgets[std::uniform_int_distribution<int>(0, 5)(rng)],
                static_cast<tilewright::Schedule>(std::uniform_int_distribution<int>(0, 2)(rng)));
            auto const plan = launch.plan(config, problem, ctx);
            out << pos << ' ' << pool[pos]->index << ' ' << problem.size.m << ' ' << problem.size.n
                << ' ' << problem.size.k << ' ' << problem.batch << ' '
                << static_cast<int>(problem.mi_dtype) << ' ' << ctx.cu_budget << ' '
                << static_cast<int>(ctx.schedule) << ' ' << plan.dynamic_queue << ' '
                << plan.grid_wgs << ' ' << reductionName(plan.partial_reduction) << ' '
                << sm.macroTile.x << ' ' << sm.macroTile.y << ' ' << sm.depthU << ' '
                << sm.CUOccupancy << '\n';
        }
    }
    ASSERT_TRUE(out.good());
}

#endif
