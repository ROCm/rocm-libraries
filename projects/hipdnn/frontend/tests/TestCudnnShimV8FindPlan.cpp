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
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

using fe::CudnnFindSamplingTechnique;
using hipdnn_shim_test::BuildFailure;
using hipdnn_shim_test::failureOf;

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
// NOT_INITIALIZED. Every plan whose warm-up ran is kept, unmeasured, in
// heuristic order.
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
    EXPECT_EQ(
        timesOf(sorted),
        (std::vector<float>{std::numeric_limits<float>::max(), std::numeric_limits<float>::max()}));
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

// A plan whose span is invalid sorts after every measured plan.
TEST_F(TestCudnnShimV8FindPlan, InvalidElapsedSortsLast)
{
    installProfilingTimes({-1.0F, 1.0F});

    const fe::executionPlans_t sorted
        = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(plans(2));

    EXPECT_EQ(tagsOf(sorted), (std::vector<std::string>{"ConvFwd_eng1", "ConvFwd_eng0"}));
    EXPECT_EQ(timesOf(sorted), (std::vector<float>{1.0F, std::numeric_limits<float>::max()}));
}

// A failed run ends the plan's sampling; it does not reuse an earlier run's time.
TEST_F(TestCudnnShimV8FindPlan, FailedLaterRunSortsLast)
{
    {
        SCOPED_TRACE("median of three");
        installProfilingTimes({1.0F, -1.0F, 5.0F, 5.0F, 5.0F});
        const std::size_t before = timelineCount("elapsed");
        const fe::executionPlans_t sorted
            = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE>(plans(2));
        EXPECT_EQ(tagsOf(sorted), (std::vector<std::string>{"ConvFwd_eng1", "ConvFwd_eng0"}));
        EXPECT_EQ(timesOf(sorted), (std::vector<float>{5.0F, std::numeric_limits<float>::max()}));
        EXPECT_EQ(timelineCount("elapsed") - before, 5U);
    }
    {
        SCOPED_TRACE("till stable");
        installProfilingTimes({-1.0F, 3.0F, 3.0F});
        const fe::executionPlans_t sorted
            = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_TILL_STABLE>(plans(2));
        EXPECT_EQ(tagsOf(sorted), (std::vector<std::string>{"ConvFwd_eng1", "ConvFwd_eng0"}));
        EXPECT_EQ(timesOf(sorted), (std::vector<float>{3.0F, std::numeric_limits<float>::max()}));
    }
}

// A rejected START or STOP leaves the plan unmeasured, like an invalid span.
TEST_F(TestCudnnShimV8FindPlan, RejectedTimerCallSortsLast)
{
    using ::testing::_;

    installProfilingTimes({2.0F, 1.0F, 3.0F});
    int stops = 0;
    ON_CALL(*_mockBackend, backendSetAttribute(_, HIPDNN_ATTR_PROFILING_STOP_EXT, _, _, _))
        .WillByDefault([&stops](hipdnnBackendDescriptor_t,
                                hipdnnBackendAttributeName_t,
                                hipdnnBackendAttributeType_t,
                                std::int64_t,
                                const void*) {
            return ++stops == 1 ? HIPDNN_STATUS_INTERNAL_ERROR : HIPDNN_STATUS_SUCCESS;
        });

    const fe::executionPlans_t sorted
        = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(plans(3));

    // Engine 0's STOP fails before its span is read, so engines 1 and 2 read 2 and 1.
    EXPECT_EQ(tagsOf(sorted),
              (std::vector<std::string>{"ConvFwd_eng2", "ConvFwd_eng1", "ConvFwd_eng0"}));
    EXPECT_EQ(timesOf(sorted), (std::vector<float>{1.0F, 2.0F, std::numeric_limits<float>::max()}));
}

TEST_F(TestCudnnShimV8FindPlan, RejectedStartSortsLast)
{
    using ::testing::_;

    installProfilingTimes({1.0F, 2.0F});
    int starts = 0;
    ON_CALL(*_mockBackend, backendSetAttribute(_, HIPDNN_ATTR_PROFILING_START_EXT, _, _, _))
        .WillByDefault([&starts](hipdnnBackendDescriptor_t,
                                 hipdnnBackendAttributeName_t,
                                 hipdnnBackendAttributeType_t,
                                 std::int64_t,
                                 const void*) {
            return ++starts == 1 ? HIPDNN_STATUS_INTERNAL_ERROR : HIPDNN_STATUS_SUCCESS;
        });

    const fe::executionPlans_t sorted
        = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(plans(2));

    // Engine 0's span is never read, so engine 1 reads the first time.
    EXPECT_EQ(tagsOf(sorted), (std::vector<std::string>{"ConvFwd_eng1", "ConvFwd_eng0"}));
    EXPECT_EQ(timesOf(sorted), (std::vector<float>{1.0F, std::numeric_limits<float>::max()}));
    EXPECT_EQ(timelineCount("elapsed"), 1U);
}

// The warm-up ran, but a timed run failed: its span measures nothing.
TEST_F(TestCudnnShimV8FindPlan, FailedTimedRunSortsLast)
{
    installProfilingTimes({0.5F, 1.0F});
    failExecuteFor(0, 2);

    const fe::executionPlans_t sorted
        = timeSorted<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(plans(2));

    EXPECT_EQ(tagsOf(sorted), (std::vector<std::string>{"ConvFwd_eng1", "ConvFwd_eng0"}));
    EXPECT_EQ(timesOf(sorted), (std::vector<float>{0.5F, std::numeric_limits<float>::max()}));
    // The timer is still stopped after the failed run.
    EXPECT_EQ(timelineCount("stop"), 2U);
    EXPECT_EQ(timelineCount("elapsed"), 1U);
}

TEST_F(TestCudnnShimV8FindPlan, FindPlanAndCacheSkipsFailedTimedRun)
{
    installProfilingTimes({0.01F, 1.0F, 2.0F});
    failExecuteFor(0, 2);
    fe::EngineConfigGenerator generator = heuristicGenerator();
    fe::ExecutionPlanCache cache("find");

    const fe::ExecutionPlan front
        = generator.cudnnFindPlanAndCache<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(
            _handle, *_graph, *_pack, cache);

    EXPECT_EQ(front.getTag(), "ConvFwd_eng1");
    EXPECT_EQ(front.getExecutionTime(), 0.01F);
    const fe::ExecutionPlan* cached = nullptr;
    ASSERT_TRUE(cache.get_plan_from_cache(*_graph, cached));
    EXPECT_EQ(cached->getTag(), "ConvFwd_eng1");
}

// hipDNN keeps a failed call's message until it is read; a failed timer call
// must not leave it for the next shim error on the thread.
TEST_F(TestCudnnShimV8FindPlan, TimerFailureDrainsHipdnnMessage)
{
    using ::testing::_;

    std::string pending;
    ON_CALL(*_mockBackend, getLastErrorString(_, _))
        .WillByDefault([&pending](char* message, std::size_t size) {
            const std::size_t length = std::min(pending.size(), size - 1);
            std::memcpy(message, pending.data(), length);
            message[length] = '\0';
            pending.clear();
        });
    const auto failSet = [this, &pending](hipdnnBackendAttributeName_t failing) {
        ON_CALL(*_mockBackend, backendSetAttribute(_, failing, _, _, _))
            .WillByDefault([&pending](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      std::int64_t,
                                      const void*) {
                pending = "stale profiling error";
                return HIPDNN_STATUS_BAD_PARAM;
            });
    };

    fe::detail::PlanTimer timer(_handle);
    {
        SCOPED_TRACE("handle");
        failSet(HIPDNN_ATTR_PROFILING_HANDLE_EXT);
        EXPECT_NE(timer.start(), CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(pending, "");
    }
    {
        SCOPED_TRACE("stop");
        failSet(HIPDNN_ATTR_PROFILING_STOP_EXT);
        EXPECT_NE(timer.stop(), CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(pending, "");
        ON_CALL(*_mockBackend, backendSetAttribute(_, HIPDNN_ATTR_PROFILING_STOP_EXT, _, _, _))
            .WillByDefault(::testing::Return(HIPDNN_STATUS_SUCCESS));
    }
    {
        SCOPED_TRACE("finalize");
        ON_CALL(*_mockBackend, backendFinalize(_))
            .WillByDefault([&pending](hipdnnBackendDescriptor_t) {
                pending = "stale profiling error";
                return HIPDNN_STATUS_BAD_PARAM;
            });
        EXPECT_NE(timer.stop(), CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(pending, "");
    }
    {
        SCOPED_TRACE("elapsed");
        ON_CALL(*_mockBackend,
                backendGetAttribute(_, HIPDNN_ATTR_PROFILING_ELAPSED_MS_EXT, _, _, _, _))
            .WillByDefault([&pending](hipdnnBackendDescriptor_t,
                                      hipdnnBackendAttributeName_t,
                                      hipdnnBackendAttributeType_t,
                                      std::int64_t,
                                      std::int64_t*,
                                      void*) {
                pending = "stale profiling error";
                return HIPDNN_STATUS_BAD_PARAM;
            });
        float ms = 7.0F;
        EXPECT_NE(timer.elapsedMs(&ms), CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(pending, "");
    }
    {
        SCOPED_TRACE("device sync");
        failSet(HIPDNN_ATTR_PROFILING_DEVICE_SYNC_EXT);
        EXPECT_NE(fe::detail::deviceSynchronize(), CUDNN_STATUS_SUCCESS);
        EXPECT_EQ(pending, "");
    }
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

// The predicate drops engine 1 before timing, so engines 0 and 2 read the
// first two times.
TEST_F(TestCudnnShimV8FindPlan, FindPlanAndCacheAppliesPredicate)
{
    installProfilingTimes({3.0F, 1.0F, 2.0F});
    fe::EngineConfigGenerator generator = heuristicGenerator();
    fe::ExecutionPlanCache cache("find");

    const fe::ExecutionPlan front
        = generator.cudnnFindPlanAndCache<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(
            _handle, *_graph, *_pack, cache, [](const fe::ExecutionPlan& plan) {
                return plan.getTag() == "ConvFwd_eng1";
            });

    EXPECT_EQ(front.getTag(), "ConvFwd_eng2");
    EXPECT_EQ(front.getExecutionTime(), 1.0F);
    EXPECT_EQ(timelineCount("elapsed"), 2U);
}

// With a saturation count of 2, the fastest plan is cached only once it has
// won twice.
TEST_F(TestCudnnShimV8FindPlan, FindPlanAndCacheWaitsForStableFront)
{
    installProfilingTimes({3.0F, 1.0F, 2.0F, 3.0F, 1.0F, 2.0F});
    fe::EngineConfigGenerator generator = heuristicGenerator();
    fe::ExecutionPlanCache cache("find");
    cache.set_saturation_count(2);
    const fe::ExecutionPlan* cached = nullptr;

    const fe::ExecutionPlan first
        = generator.cudnnFindPlanAndCache<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(
            _handle, *_graph, *_pack, cache);
    EXPECT_EQ(first.getTag(), "ConvFwd_eng1");
    EXPECT_FALSE(cache.get_plan_from_cache(*_graph, cached));

    const fe::ExecutionPlan second
        = generator.cudnnFindPlanAndCache<CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(
            _handle, *_graph, *_pack, cache);
    EXPECT_EQ(second.getTag(), "ConvFwd_eng1");
    ASSERT_TRUE(cache.get_plan_from_cache(*_graph, cached));
    EXPECT_EQ(cached->get_raw_desc(), second.get_raw_desc());
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

    const BuildFailure failure = failureOf([&] {
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
