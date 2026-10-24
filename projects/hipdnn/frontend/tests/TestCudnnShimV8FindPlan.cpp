// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// The ported time_sorted_plan and cudnnFindPlan* against the shim and the mock
// hipDNN backend, which plays back the profiling times and records the
// profiling-control calls the plan timer makes.
#include <hipdnn_compatibility/cudnn/cudnn_frontend_find_plan.h>

#include <gtest/gtest.h>

#include "fake_backend/ShimV8PlanFixture.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using fe::CudnnFindSamplingTechnique;
using hipdnn_shim_test::BuilderFailure;
using hipdnn_shim_test::builderFailureOf;

fe::EngineConfigList heuristicSource(fe::OperationGraph& graph)
{
    fe::EngineHeuristics_v8 heuristics = fe::EngineHeuristicsBuilder_v8()
                                             .setOperationGraph(graph)
                                             .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                                             .build();
    return heuristics.getEngineConfig(heuristics.getEngineConfigCount());
}

class TestCudnnShimV8FindPlan : public hipdnn_shim_test::ShimV8PlanFixture
{
protected:
    void SetUp() override
    {
        ShimV8PlanFixture::SetUp();
        _graph = std::make_unique<fe::OperationGraph>(convGraph());
        _pack = std::make_unique<fe::VariantPack>(variantPack());
        ASSERT_EQ(_graph->get_status(), CUDNN_STATUS_SUCCESS) << _graph->get_error();
        ASSERT_EQ(_pack->get_status(), CUDNN_STATUS_SUCCESS) << _pack->get_error();
    }

    void TearDown() override
    {
        _pack.reset();
        _graph.reset();
        ShimV8PlanFixture::TearDown();
    }

    // Plans for engines 0..count-1, in index order.
    fe::executionPlans_t plans(std::int64_t count)
    {
        fe::executionPlans_t built;
        for(std::int64_t i = 0; i < count; ++i)
        {
            built.push_back(planFor(*_graph, i));
        }
        return built;
    }

    template <CudnnFindSamplingTechnique Technique>
    fe::executionPlans_t timeSorted(fe::executionPlans_t candidates, std::uint64_t maxPlans = 1000)
    {
        return fe::time_sorted_plan<Technique>(_handle, std::move(candidates), *_pack, maxPlans);
    }

    std::size_t timelineCount(const std::string& event) const
    {
        const std::vector<std::string> timeline = profilingTimeline();
        return static_cast<std::size_t>(std::count(timeline.begin(), timeline.end(), event));
    }

    static std::vector<float> timesOf(const fe::executionPlans_t& sorted)
    {
        std::vector<float> times;
        times.reserve(sorted.size());
        for(const fe::ExecutionPlan& plan : sorted)
        {
            times.push_back(plan.getExecutionTime());
        }
        return times;
    }

    static fe::EngineConfigGenerator heuristicGenerator()
    {
        const std::array<fe::GeneratorSource, 1> sources = {heuristicSource};
        return {static_cast<int>(sources.size()), sources.data()};
    }

    std::unique_ptr<fe::OperationGraph> _graph;
    std::unique_ptr<fe::VariantPack> _pack;
};

TEST_F(TestCudnnShimV8FindPlan, PlansSortedByTime)
{
    installProfilingTimes({3.0F, 1.0F, 2.0F});

    const fe::executionPlans_t sorted
        = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(plans(3));

    EXPECT_EQ(tagsOf(sorted),
              (std::vector<std::string>{"ConvFwd_eng1", "ConvFwd_eng2", "ConvFwd_eng0"}));
    EXPECT_EQ(timesOf(sorted), (std::vector<float>{1.0F, 2.0F, 3.0F}));
}

// The plan timer's profiling-control calls around a timed run (the contract
// with PlanTimer and deviceSynchronize in cudnn_frontend_shim.h).
TEST_F(TestCudnnShimV8FindPlan, TimerBracketsEachTimedRun)
{
    installProfilingTimes({2.0F});

    const fe::executionPlans_t sorted
        = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(plans(1));

    ASSERT_EQ(sorted.size(), 1U);
    EXPECT_EQ(profilingTimeline(),
              (std::vector<std::string>{// The timer, then a device synchronize.
                                        "create",
                                        "create",
                                        "sync",
                                        "destroy",
                                        // Warm-up, then a device synchronize.
                                        "execute",
                                        "create",
                                        "sync",
                                        "destroy",
                                        // One timed run.
                                        "reset",
                                        "handle",
                                        "start",
                                        "execute",
                                        "stop",
                                        "finalize",
                                        "elapsed",
                                        // The timer goes out of scope.
                                        "destroy"}));
    for(const auto& call : setAttributesOn(HIPDNN_BACKEND_PROFILING_CONTROL_EXT))
    {
        if(call.name == HIPDNN_ATTR_PROFILING_HANDLE_EXT)
        {
            EXPECT_EQ(call.value<void*>(), static_cast<void*>(_handle));
        }
        else
        {
            EXPECT_TRUE(call.value<bool>());
        }
    }
}

// One descriptor serves every measurement: RESET before each START.
TEST_F(TestCudnnShimV8FindPlan, TimerReusedAcrossRuns)
{
    const fe::executionPlans_t sorted
        = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(plans(2));

    EXPECT_EQ(sorted.size(), 2U);
    // The timer, the first synchronize and one synchronize per plan.
    EXPECT_EQ(timelineCount("create"), 4U);
    EXPECT_EQ(timelineCount("sync"), 3U);
    for(const char* event : {"reset", "handle", "start", "stop", "finalize", "elapsed"})
    {
        EXPECT_EQ(timelineCount(event), 6U) << event;
    }
    EXPECT_EQ(timelineCount("destroy"), timelineCount("create"));
}

// A failed create leaves the timer inert: no profiling call, every method
// NOT_INITIALIZED. time_sorted_plan ignores the statuses, as upstream ignores
// the CUDA event errors, so every plan whose warm-up ran is kept.
TEST_F(TestCudnnShimV8FindPlan, TimerInertWithoutDescriptor)
{
    failProfilingCreate();

    fe::detail::PlanTimer timer(_handle);
    EXPECT_EQ(timer.start(), CUDNN_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(timer.stop(), CUDNN_STATUS_NOT_INITIALIZED);
    float ms = 7.0F;
    EXPECT_EQ(timer.elapsedMs(&ms), CUDNN_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(ms, 7.0F);
    EXPECT_EQ(timer.elapsedMs(nullptr), CUDNN_STATUS_NOT_INITIALIZED);
    EXPECT_EQ(fe::detail::deviceSynchronize(), CUDNN_STATUS_NOT_INITIALIZED);
    EXPECT_TRUE(setAttributesOn(HIPDNN_BACKEND_PROFILING_CONTROL_EXT).empty());

    failExecuteFor(1, 1);
    const fe::executionPlans_t sorted
        = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(plans(3));

    EXPECT_EQ(tagsOf(sorted), (std::vector<std::string>{"ConvFwd_eng0", "ConvFwd_eng2"}));
    EXPECT_EQ(timesOf(sorted), (std::vector<float>{0.0F, 0.0F}));
    EXPECT_TRUE(setAttributesOn(HIPDNN_BACKEND_PROFILING_CONTROL_EXT).empty());
    EXPECT_EQ(timelineCount("elapsed"), 0U);
}

// A negative or non-finite span is EXECUTION_FAILED and leaves the output alone;
// a null output is BAD_PARAM before any backend read.
TEST_F(TestCudnnShimV8FindPlan, TimerRejectsInvalidElapsed)
{
    installProfilingTimes({-1.0F,
                           std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::infinity(),
                           2.5F});
    fe::detail::PlanTimer timer(_handle);

    for(int run = 0; run < 3; ++run)
    {
        SCOPED_TRACE(run);
        ASSERT_EQ(timer.start(), CUDNN_STATUS_SUCCESS);
        ASSERT_EQ(timer.stop(), CUDNN_STATUS_SUCCESS);
        float ms = 7.0F;
        EXPECT_EQ(timer.elapsedMs(&ms), CUDNN_STATUS_EXECUTION_FAILED);
        EXPECT_EQ(ms, 7.0F);
    }
    EXPECT_EQ(timer.elapsedMs(nullptr), CUDNN_STATUS_BAD_PARAM);
    EXPECT_EQ(timelineCount("elapsed"), 3U);

    float ms = 7.0F;
    EXPECT_EQ(timer.elapsedMs(&ms), CUDNN_STATUS_SUCCESS);
    EXPECT_EQ(ms, 2.5F);
}

// A plan whose span is invalid keeps the time of 0 it started with.
TEST_F(TestCudnnShimV8FindPlan, InvalidElapsedKeepsZeroTime)
{
    installProfilingTimes({1.0F, -1.0F});

    const fe::executionPlans_t sorted
        = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(plans(2));

    EXPECT_EQ(tagsOf(sorted), (std::vector<std::string>{"ConvFwd_eng1", "ConvFwd_eng0"}));
    EXPECT_EQ(timesOf(sorted), (std::vector<float>{0.0F, 1.0F}));
}

TEST_F(TestCudnnShimV8FindPlan, FailedWarmupSkipsPlan)
{
    failExecuteFor(1, 1);
    installProfilingTimes({2.0F, 1.0F});

    const fe::executionPlans_t sorted
        = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(plans(3));

    EXPECT_EQ(tagsOf(sorted), (std::vector<std::string>{"ConvFwd_eng2", "ConvFwd_eng0"}));
    EXPECT_EQ(timelineCount("elapsed"), 2U);
    // Engine 1 ran only its failed warm-up.
    EXPECT_EQ(executedEngineIds(), (std::vector<std::int64_t>{0, 0, 1, 2, 2}));
}

// Upstream keys a std::set on the time, so plans that tie collapse into one.
TEST_F(TestCudnnShimV8FindPlan, EqualTimesKeepEveryPlan)
{
    installProfilingTimes({1.5F, 1.5F, 0.5F});

    const fe::executionPlans_t sorted
        = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(plans(3));

    EXPECT_EQ(tagsOf(sorted),
              (std::vector<std::string>{"ConvFwd_eng2", "ConvFwd_eng0", "ConvFwd_eng1"}));
    EXPECT_EQ(timesOf(sorted), (std::vector<float>{0.5F, 1.5F, 1.5F}));
}

TEST_F(TestCudnnShimV8FindPlan, SamplingCountsIterations)
{
    {
        SCOPED_TRACE("once");
        installProfilingTimes({});
        const std::size_t before = timelineCount("elapsed");
        timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(plans(2));
        EXPECT_EQ(timelineCount("elapsed") - before, 2U);
    }
    {
        SCOPED_TRACE("median of three");
        installProfilingTimes({});
        const std::size_t before = timelineCount("elapsed");
        timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(plans(2));
        EXPECT_EQ(timelineCount("elapsed") - before, 6U);
    }
    {
        // Stops at the first run that is not 5% faster than the best so far.
        SCOPED_TRACE("till stable");
        installProfilingTimes({10.0F, 9.0F, 8.9F});
        const std::size_t before = timelineCount("elapsed");
        const fe::executionPlans_t sorted
            = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE>(plans(1));
        EXPECT_EQ(timelineCount("elapsed") - before, 3U);
        EXPECT_EQ(timesOf(sorted), std::vector<float>{8.9F});
    }
    {
        SCOPED_TRACE("till stable, capped");
        std::vector<float> falling(150);
        float time = 1000.0F;
        for(float& sample : falling)
        {
            sample = time;
            time *= 0.9F;
        }
        installProfilingTimes(falling);
        const std::size_t before = timelineCount("elapsed");
        const fe::executionPlans_t sorted
            = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE>(plans(1));
        EXPECT_EQ(timelineCount("elapsed") - before, 100U);
        EXPECT_EQ(timesOf(sorted), std::vector<float>{falling[99]});
    }
}

TEST_F(TestCudnnShimV8FindPlan, MedianOfThreeChosen)
{
    installProfilingTimes({5.0F, 1.0F, 3.0F, 2.0F, 9.0F, 4.0F, 7.0F, 7.0F, 1.0F});

    const fe::executionPlans_t sorted
        = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(plans(3));

    EXPECT_EQ(tagsOf(sorted),
              (std::vector<std::string>{"ConvFwd_eng0", "ConvFwd_eng1", "ConvFwd_eng2"}));
    EXPECT_EQ(timesOf(sorted), (std::vector<float>{3.0F, 4.0F, 7.0F}));
}

TEST_F(TestCudnnShimV8FindPlan, MaxPlansLimitsEvaluation)
{
    installEngines(4);
    fe::executionPlans_t candidates = plans(4);
    failExecuteFor(0, 1);
    installProfilingTimes({2.0F, 1.0F, 0.5F});

    const fe::executionPlans_t sorted
        = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(std::move(candidates), 2);

    // A failed warm-up does not count toward the limit; engine 3 never runs.
    EXPECT_EQ(tagsOf(sorted), (std::vector<std::string>{"ConvFwd_eng2", "ConvFwd_eng1"}));
    EXPECT_EQ(executedEngineIds(), (std::vector<std::int64_t>{0, 1, 1, 2, 2}));
}

TEST_F(TestCudnnShimV8FindPlan, FindPlanAndCacheKeepsFront)
{
    installProfilingTimes({3.0F, 1.0F, 2.0F});
    fe::EngineConfigGenerator generator = heuristicGenerator();
    fe::ExecutionPlanCache cache("find");

    const fe::ExecutionPlan front
        = generator.cudnnFindPlanAndCache<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(
            _handle, *_graph, *_pack, cache);

    EXPECT_EQ(front.getTag(), "ConvFwd_eng1");
    const fe::ExecutionPlan* cached = nullptr;
    ASSERT_TRUE(cache.get_plan_from_cache(*_graph, cached));
    EXPECT_EQ(cached->get_raw_desc(), front.get_raw_desc());
}

// Upstream reads front() of the empty list when no plan ran.
TEST_F(TestCudnnShimV8FindPlan, EmptyFindAndCacheThrows)
{
    for(std::int64_t engine = 0; engine < 3; ++engine)
    {
        failExecuteFor(engine, 0);
    }
    fe::EngineConfigGenerator generator = heuristicGenerator();
    fe::ExecutionPlanCache cache("find");

    const BuilderFailure failure = builderFailureOf([&] {
        [[maybe_unused]] const fe::ExecutionPlan plan
            = generator.cudnnFindPlanAndCache<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(
                _handle, *_graph, *_pack, cache);
    });

    ASSERT_TRUE(failure.thrown);
    EXPECT_EQ(failure.status, CUDNN_STATUS_NOT_SUPPORTED);
    EXPECT_NE(failure.what.find("no execution plan ran"), std::string::npos) << failure.what;
    const fe::ExecutionPlan* cached = nullptr;
    EXPECT_FALSE(cache.get_plan_from_cache(*_graph, cached));
    EXPECT_EQ(timelineCount("elapsed"), 0U);
}

} // namespace
