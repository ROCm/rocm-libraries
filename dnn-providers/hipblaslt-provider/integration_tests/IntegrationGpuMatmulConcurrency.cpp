// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/ShapeUtilities.hpp>
#include <hipdnn_frontend.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>

#include "utils/GpuTestUtils.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;
using namespace test_gpu_utils;

namespace
{

constexpr int64_t DIM = 64;
constexpr int NUM_ROUNDS = 10;
constexpr int NUM_THREADS = 4;
constexpr int NUM_ITERATIONS = 40;
constexpr float TOLERANCE = 2e-4f;

// Per-thread host data: distinct inputs per thread so that a GEMM reading another
// thread's buffers or bias produces a value that differs from its own reference.
struct ThreadProblem
{
    std::vector<float> a;
    std::vector<float> b;
    std::vector<float> bias;
    std::vector<float> reference;
};

ThreadProblem makeProblem(int threadIndex, bool withBias)
{
    const auto n = static_cast<size_t>(DIM);
    ThreadProblem p;
    p.a = randomUniform(n * n, 1000U + static_cast<unsigned>(threadIndex));
    p.b = randomUniform(n * n, 2000U + static_cast<unsigned>(threadIndex));
    if(withBias)
    {
        // Constant per-thread scale times a per-column pattern.
        p.bias.resize(n);
        for(size_t j = 0; j < n; ++j)
        {
            p.bias[j]
                = static_cast<float>(threadIndex + 1) * (1.0f + static_cast<float>(j) / 64.0f);
        }
    }

    p.reference.assign(n * n, 0.0f);
    for(size_t i = 0; i < n; ++i)
    {
        for(size_t j = 0; j < n; ++j)
        {
            float acc = 0.0f;
            for(size_t k = 0; k < n; ++k)
            {
                acc += p.a[i * n + k] * p.b[k * n + j];
            }
            p.reference[i * n + j] = acc + (withBias ? p.bias[j] : 0.0f);
        }
    }
    return p;
}

// Records only the first mismatch so the failure message names where it happened.
class FailureRecorder
{
public:
    void record(const std::string& message)
    {
        if(_count.fetch_add(1) == 0)
        {
            const std::lock_guard<std::mutex> lock(_mutex);
            _first = message;
        }
    }

    bool failed() const
    {
        return _count.load() > 0;
    }

    int count() const
    {
        return _count.load();
    }

    std::string first() const
    {
        const std::lock_guard<std::mutex> lock(_mutex);
        return _first;
    }

private:
    std::atomic<int> _count{0};
    mutable std::mutex _mutex;
    std::string _first;
};

class IntegrationGpuMatmulConcurrency : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();

        ASSERT_EQ(hipInit(0), hipSuccess);

        const auto pluginPath = std::filesystem::weakly_canonical(
            hipdnn_data_sdk::utilities::getCurrentExecutableDirectory() / PLUGIN_PATH);
        const std::string pluginPathStr = pluginPath.string();
        const std::array<const char*, 1> paths = {pluginPathStr.c_str()};
        ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(
                      paths.size(), paths.data(), HIPDNN_PLUGIN_LOADING_ABSOLUTE),
                  HIPDNN_STATUS_SUCCESS);

        auto [handle, err] = createHipdnnHandle();
        ASSERT_TRUE(err.is_good()) << err.err_msg;
        _handle = std::move(handle);
    }

    // Builds a finalized fp32 matmul graph [1,64,64] x [1,64,64], optionally followed by a
    // pointwise ADD of a [1,1,64] bias, on the hipBLASLt engine. Returns the name of the stage
    // that declined, or an empty string when the graph is ready to execute.
    std::string buildGraph(Graph& graph, bool withBias) const
    {
        graph.set_name(withBias ? "MatmulBiasConcurrency" : "MatmulConcurrency");
        graph.set_io_data_type(DataType::FLOAT)
            .set_intermediate_data_type(DataType::FLOAT)
            .set_compute_data_type(DataType::FLOAT);

        const std::vector<int64_t> dims = {1, DIM, DIM};
        const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);

        const auto a = std::make_shared<TensorAttributes>(
            makeTensorAttributes("a", DataType::FLOAT, dims, strides).set_uid(UID_A));
        const auto b = std::make_shared<TensorAttributes>(
            makeTensorAttributes("b", DataType::FLOAT, dims, strides).set_uid(UID_B));

        auto c = graph.matmul(a, b, MatmulAttributes());
        if(withBias)
        {
            const std::vector<int64_t> biasDims = {1, 1, DIM};
            const auto bias = std::make_shared<TensorAttributes>(
                makeTensorAttributes("bias",
                                     DataType::FLOAT,
                                     biasDims,
                                     hipdnn_data_sdk::utilities::generateStrides(biasDims))
                    .set_uid(UID_BIAS));

            c->set_data_type(DataType::FLOAT);
            PointwiseAttributes biasAttrs;
            biasAttrs.set_mode(PointwiseMode::ADD);
            c = graph.pointwise(c, bias, biasAttrs);
        }
        c->set_output(true).set_data_type(DataType::FLOAT).set_uid(UID_C);

        auto result = graph.validate();
        if(result.is_bad())
        {
            return "validate: " + result.err_msg;
        }
        result = graph.build_operation_graph(*_handle);
        if(result.is_bad())
        {
            return "build_operation_graph: " + result.err_msg;
        }
        result = graph.create_execution_plan_ext(
            hipdnn_data_sdk::utilities::engineNameOrIdToId("HIPBLASLT_ENGINE"), {});
        if(result.is_bad())
        {
            return "create_execution_plan_ext: " + result.err_msg;
        }
        result = graph.check_support();
        if(result.is_bad())
        {
            return "check_support: " + result.err_msg;
        }
        result = graph.build_plans();
        if(result.is_bad())
        {
            return "build_plans: " + result.err_msg;
        }
        return {};
    }

    // Runs NUM_ROUNDS rounds of NUM_THREADS threads, each with its own handle and buffers,
    // executing the one shared finalized graph. The graph is the only shared object.
    static void runSharedGraph(const Graph& shared, bool withBias, FailureRecorder& failures)
    {
        std::vector<ThreadProblem> problems;
        problems.reserve(NUM_THREADS);
        for(int t = 0; t < NUM_THREADS; ++t)
        {
            problems.push_back(makeProblem(t, withBias));
        }

        int64_t workspaceSize = 0;
        ASSERT_TRUE(shared.get_workspace_size(workspaceSize).is_good());

        for(int round = 0; round < NUM_ROUNDS && !failures.failed(); ++round)
        {
            std::vector<std::thread> threads;
            threads.reserve(NUM_THREADS);
            for(int t = 0; t < NUM_THREADS; ++t)
            {
                threads.emplace_back(
                    [&shared, &problems, &failures, workspaceSize, withBias, round, t]() {
                        runThread(shared,
                                  problems[static_cast<size_t>(t)],
                                  workspaceSize,
                                  withBias,
                                  round,
                                  t,
                                  failures);
                    });
            }
            for(auto& thread : threads)
            {
                thread.join();
            }
        }
    }

    static void runThread(const Graph& shared,
                          const ThreadProblem& problem,
                          int64_t workspaceSize,
                          bool withBias,
                          int round,
                          int thread,
                          FailureRecorder& failures)
    {
        const std::string where
            = "round " + std::to_string(round) + " thread " + std::to_string(thread) + ": ";

        auto [handle, err] = createHipdnnHandle();
        if(err.is_bad())
        {
            failures.record(where + "createHipdnnHandle: " + err.err_msg);
            return;
        }

        const auto elements = static_cast<size_t>(DIM * DIM);
        const size_t bytes = elements * sizeof(float);
        const auto dA = deviceAlloc(bytes);
        const auto dB = deviceAlloc(bytes);
        const auto dC = deviceAlloc(bytes);
        const auto dBias
            = withBias ? deviceAlloc(static_cast<size_t>(DIM) * sizeof(float)) : nullptr;
        const auto dWorkspace
            = workspaceSize > 0 ? deviceAlloc(static_cast<size_t>(workspaceSize)) : nullptr;
        if(!dA || !dB || !dC || (withBias && !dBias) || (workspaceSize > 0 && !dWorkspace))
        {
            failures.record(where + "device allocation failed");
            return;
        }

        if(hipMemcpy(dA.get(), problem.a.data(), bytes, hipMemcpyHostToDevice) != hipSuccess
           || hipMemcpy(dB.get(), problem.b.data(), bytes, hipMemcpyHostToDevice) != hipSuccess
           || (withBias
               && hipMemcpy(dBias.get(),
                            problem.bias.data(),
                            problem.bias.size() * sizeof(float),
                            hipMemcpyHostToDevice)
                      != hipSuccess))
        {
            failures.record(where + "host to device copy failed");
            return;
        }

        std::unordered_map<int64_t, void*> pack
            = {{UID_A, dA.get()}, {UID_B, dB.get()}, {UID_C, dC.get()}};
        if(withBias)
        {
            pack[UID_BIAS] = dBias.get();
        }

        std::vector<float> result(elements);
        for(int iter = 0; iter < NUM_ITERATIONS; ++iter)
        {
            const std::string at = where + "iteration " + std::to_string(iter) + ": ";

            if(hipMemset(dC.get(), 0, bytes) != hipSuccess)
            {
                failures.record(at + "hipMemset failed");
                return;
            }
            const auto execErr = shared.execute(*handle, pack, dWorkspace.get());
            if(execErr.is_bad())
            {
                failures.record(at + "execute: " + execErr.err_msg);
                return;
            }
            if(hipDeviceSynchronize() != hipSuccess
               || hipMemcpy(result.data(), dC.get(), bytes, hipMemcpyDeviceToHost) != hipSuccess)
            {
                failures.record(at + "synchronize or device to host copy failed");
                return;
            }

            for(size_t i = 0; i < elements; ++i)
            {
                const float expected = problem.reference[i];
                const float diff = std::fabs(result[i] - expected);
                if(!(diff <= TOLERANCE + TOLERANCE * std::fabs(expected)))
                {
                    failures.record(at + "mismatch at index " + std::to_string(i) + ": got "
                                    + std::to_string(result[i]) + ", expected "
                                    + std::to_string(expected));
                    return;
                }
            }
        }
    }

    HipdnnHandlePtr _handle;
};

} // namespace

// One finalized graph, executed by several threads that each own a handle and buffers.
// Fails if concurrent executes share mutable hipBLASLt state (e.g. one matmul descriptor).
TEST_F(IntegrationGpuMatmulConcurrency, SharedGraphAcrossThreadsWithOwnHandles)
{
    Graph graph;
    const auto declined = buildGraph(graph, false);
    ASSERT_TRUE(declined.empty()) << "hipBLASLt engine declined the matmul graph at " << declined;

    FailureRecorder failures;
    runSharedGraph(graph, false, failures);

    EXPECT_EQ(failures.count(), 0) << "first failure: " << failures.first();
}

// Each thread binds a different bias buffer to the same finalized fused graph.
// Fails if the per-call bias pointer is stored in state shared between executes.
TEST_F(IntegrationGpuMatmulConcurrency, SharedFusedBiasGraphWithDistinctBiasBuffers)
{
    Graph graph;
    const auto declined = buildGraph(graph, true);
    ASSERT_TRUE(declined.empty()) << "hipBLASLt engine declined the matmul+bias graph at "
                                  << declined;

    FailureRecorder failures;
    runSharedGraph(graph, true, failures);

    EXPECT_EQ(failures.count(), 0) << "first failure: " << failures.first();
}

// Concurrent get_knobs_for_engine() calls on one finalized graph. The call takes no handle, so
// threads that each own a handle still go through the graph's own plugin handle, whose
// engine-details buffers every call stores and removes. Fails (crash in the plugin handle's
// map) if that bookkeeping is not synchronized, or if a call returns a different knob list than
// the single-threaded reference. The hipBLASLt engine publishes no knobs, so the reference list
// is asserted empty: an engine that gains knobs must revisit this test, whose per-knob field
// comparison then starts comparing content.
TEST_F(IntegrationGpuMatmulConcurrency, ConcurrentKnobQueriesOnSharedGraph)
{
    constexpr int64_t KNOB_DIM = 32;
    constexpr int NUM_KNOB_THREADS = 16;
    constexpr int NUM_KNOB_QUERIES = 3000;

    Graph graph;
    graph.set_name("MatmulKnobQueryConcurrency");
    graph.set_io_data_type(DataType::FLOAT)
        .set_intermediate_data_type(DataType::FLOAT)
        .set_compute_data_type(DataType::FLOAT);

    const std::vector<int64_t> dims = {1, KNOB_DIM, KNOB_DIM};
    const auto strides = hipdnn_data_sdk::utilities::generateStrides(dims);
    const auto a = std::make_shared<TensorAttributes>(
        makeTensorAttributes("a", DataType::FLOAT, dims, strides).set_uid(UID_A));
    const auto b = std::make_shared<TensorAttributes>(
        makeTensorAttributes("b", DataType::FLOAT, dims, strides).set_uid(UID_B));
    const auto c = graph.matmul(a, b, MatmulAttributes());
    c->set_output(true).set_data_type(DataType::FLOAT).set_uid(UID_C);

    auto result = graph.validate();
    ASSERT_TRUE(result.is_good()) << result.err_msg;
    result = graph.build_operation_graph(*_handle);
    ASSERT_TRUE(result.is_good()) << result.err_msg;

    // This test loads only the hipBLASLt plugin, so its engine is the one the ranking returns.
    const int64_t engineId = hipdnn_data_sdk::utilities::engineNameOrIdToId("HIPBLASLT_ENGINE");
    std::vector<int64_t> rankedEngineIds;
    result = graph.get_ranked_engine_ids(rankedEngineIds);
    ASSERT_TRUE(result.is_good()) << result.err_msg;
    ASSERT_FALSE(rankedEngineIds.empty());
    ASSERT_EQ(rankedEngineIds.front(), engineId);

    std::vector<Knob> referenceKnobs;
    result = graph.get_knobs_for_engine(engineId, referenceKnobs);
    ASSERT_TRUE(result.is_good()) << result.err_msg;
    ASSERT_TRUE(referenceKnobs.empty())
        << "HIPBLASLT_ENGINE now publishes knobs; check the per-knob comparison covers them";

    const Graph& sharedGraph = graph;
    FailureRecorder failures;
    std::vector<std::thread> threads;
    threads.reserve(NUM_KNOB_THREADS);
    for(int t = 0; t < NUM_KNOB_THREADS; ++t)
    {
        threads.emplace_back([&sharedGraph, &failures, &referenceKnobs, engineId, t]() {
            for(int iter = 0; iter < NUM_KNOB_QUERIES; ++iter)
            {
                std::vector<Knob> knobs;
                const auto err = sharedGraph.get_knobs_for_engine(engineId, knobs);
                const std::string at
                    = "thread " + std::to_string(t) + " query " + std::to_string(iter) + ": ";
                if(err.is_bad())
                {
                    failures.record(at + "get_knobs_for_engine: " + err.err_msg);
                    return;
                }
                if(knobs.size() != referenceKnobs.size())
                {
                    failures.record(at + "got " + std::to_string(knobs.size()) + " knobs, expected "
                                    + std::to_string(referenceKnobs.size()));
                    return;
                }
                for(size_t i = 0; i < knobs.size(); ++i)
                {
                    const auto& got = knobs[i];
                    const auto& expected = referenceKnobs[i];
                    const bool sameConstraint
                        = (got.constraint() == nullptr) == (expected.constraint() == nullptr)
                          && (got.constraint() == nullptr
                              || got.constraint()->toString() == expected.constraint()->toString());
                    if(got.knobId() != expected.knobId()
                       || got.description() != expected.description()
                       || got.isDeprecated() != expected.isDeprecated()
                       || got.valueType() != expected.valueType()
                       || got.defaultValue() != expected.defaultValue() || !sameConstraint)
                    {
                        failures.record(at + "knob " + std::to_string(i) + " is " + got.toString()
                                        + ", expected " + expected.toString());
                        return;
                    }
                }
            }
        });
    }
    for(auto& thread : threads)
    {
        thread.join();
    }

    EXPECT_EQ(failures.count(), 0) << "first failure: " << failures.first();
}
