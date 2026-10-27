// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Convolution fprop through the cuDNN v8 (v0.x builder) surface of the shim, run on a real
// provider and checked against the CPU reference. `--overhead` instead measures the shim's
// per-call cost against direct hipDNN backend calls on the same objects.

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <hipdnn_data_sdk/utilities/PlatformUtils.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_test_sdk/utilities/CpuFpReferenceConvolution.hpp>
#include <hipdnn_test_sdk/utilities/DynamicTolerances.hpp>

#include "V8SampleUtils.hpp"

namespace fe = v8_sample::fe;
namespace sdk = hipdnn_data_sdk::utilities;

namespace
{

constexpr int64_t X_UID = 'x';
constexpr int64_t W_UID = 'w';
constexpr int64_t Y_UID = 'y';

using v8_sample::ConvProblem;

fe::Operation
    makeFpropOp(const ConvProblem& problem, const TensorLayout& layout, cudnnDataType_t type)
{
    return v8_sample::makeConvOp(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR,
                                 problem,
                                 v8_sample::makeTensor(X_UID, problem.xDims(), layout, type),
                                 v8_sample::makeTensor(W_UID, problem.wDims(), layout, type),
                                 v8_sample::makeTensor(Y_UID, problem.yDims(), layout, type));
}

template <typename T>
bool validateConv(const ConvProblem& problem,
                  const TensorLayout& layout,
                  sdk::Tensor<T>& x,
                  sdk::Tensor<T>& w,
                  sdk::Tensor<T>& y)
{
    sdk::Tensor<T> yRef(problem.yDims(), layout);
    hipdnn_test_sdk::utilities::CpuFpReferenceConvolution::fprop(
        x, w, yRef, problem.stride, problem.dilation, problem.prePadding, problem.postPadding);

    const auto tolerance
        = hipdnn_test_sdk::utilities::conv::calculateConvFpropTolerance<T, T, float>(
            0.0, 1.0, 0.0, 1.0, problem.wDims());
    return v8_sample::matches("y", yRef, y, tolerance, tolerance);
}

// Overhead mode.

using Clock = std::chrono::steady_clock;

constexpr int WARMUP_CALLS = 100;
constexpr int ROUNDS = 200;
constexpr int BATCH = 16;
constexpr double EXECUTE_BUDGET_NS = 1000.0;
// VariantPackBuilder::build() allocates and crosses the backend lock several times per call.
constexpr double BUILD_AND_EXECUTE_BUDGET_NS = 5000.0;

double msSince(Clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

bool envEnabled(const char* name)
{
    std::string value = sdk::getEnv(name);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return !value.empty() && value != "0" && value != "off";
}

double percentile(std::vector<double> values, double quantile)
{
    std::sort(values.begin(), values.end());
    const auto index
        = static_cast<size_t>(std::lround(quantile * static_cast<double>(values.size() - 1)));
    return values[index];
}

struct OverheadResult
{
    double p10;
    double p50;
    double p90;
    double shim;
    double raw;
    double budget;

    bool pass() const
    {
        return p50 < budget;
    }
};

// Each arm returns false on a failed call. Every batch starts on an idle queue, and the arm
// order alternates per round so drift and ordering effects cancel.
template <typename ShimArm, typename RawArm>
OverheadResult measure(const char* metric, ShimArm& shimArm, RawArm& rawArm, double budget)
{
    bool ok = true;
    for(int i = 0; i < WARMUP_CALLS; ++i)
    {
        ok = shimArm() && ok;
    }
    for(int i = 0; i < WARMUP_CALLS; ++i)
    {
        ok = rawArm() && ok;
    }
    HIP_CHECK(hipDeviceSynchronize());

    auto timeBatch = [&ok](auto& arm) {
        HIP_CHECK(hipDeviceSynchronize());
        const Clock::time_point start = Clock::now();
        for(int i = 0; i < BATCH; ++i)
        {
            if(!arm())
            {
                ok = false;
            }
        }
        const Clock::time_point stop = Clock::now();
        return std::chrono::duration<double, std::nano>(stop - start).count() / BATCH;
    };

    std::vector<double> diffs;
    std::vector<double> shimTimes;
    std::vector<double> rawTimes;
    diffs.reserve(ROUNDS);
    shimTimes.reserve(ROUNDS);
    rawTimes.reserve(ROUNDS);
    for(int round = 0; round < ROUNDS; ++round)
    {
        double shimNs = 0.0;
        double rawNs = 0.0;
        if(round % 2 == 0)
        {
            shimNs = timeBatch(shimArm);
            rawNs = timeBatch(rawArm);
        }
        else
        {
            rawNs = timeBatch(rawArm);
            shimNs = timeBatch(shimArm);
        }
        diffs.push_back(shimNs - rawNs);
        shimTimes.push_back(shimNs);
        rawTimes.push_back(rawNs);
    }
    HIP_CHECK(hipDeviceSynchronize());

    if(!ok)
    {
        throw std::runtime_error(std::string(metric) + ": an execute call failed while timing");
    }
    return OverheadResult{percentile(diffs, 0.1),
                          percentile(diffs, 0.5),
                          percentile(diffs, 0.9),
                          percentile(shimTimes, 0.5),
                          percentile(rawTimes, 0.5),
                          budget};
}

void printOverhead(const char* metric, const OverheadResult& result)
{
    std::cout << std::fixed << std::setprecision(1) << "OVERHEAD " << metric
              << " p10=" << result.p10 << " p50=" << result.p50 << " p90=" << result.p90
              << " shim=" << result.shim << " raw=" << result.raw << ' '
              << (result.pass() ? "PASS" : "OVER_BUDGET") << '\n';
}

// Two threads build a plan from the same engine config at once; both plans must then run
// and validate. Execution stays on this thread: a handle is not for concurrent use.
bool concurrentPlanBuild(cudnnHandle_t handle,
                         const fe::ManagedOpaqueDescriptor& config,
                         const std::string& graphTag,
                         const ConvProblem& problem,
                         sdk::Tensor<float>& x,
                         sdk::Tensor<float>& w)
{
    std::array<std::optional<fe::ExecutionPlan>, 2> plans;
    std::array<std::string, 2> errors;
    std::atomic<int> arrived{0};
    auto build = [&](size_t slot) {
        try
        {
            arrived.fetch_add(1);
            while(arrived.load() < 2)
            {
                std::this_thread::yield();
            }
            plans[slot] = fe::ExecutionPlanBuilder()
                              .setHandle(handle)
                              .setEngineConfig(config, graphTag)
                              .build();
        }
        catch(const std::exception& e)
        {
            errors[slot] = e.what();
        }
        catch(...)
        {
            errors[slot] = "unknown exception";
        }
    };
    std::thread first(build, 0);
    std::thread second(build, 1);
    first.join();
    second.join();

    bool allPassed = true;
    for(size_t slot = 0; slot < plans.size(); ++slot)
    {
        if(!plans[slot])
        {
            std::cout << "Concurrent plan " << slot << " failed to build: " << errors[slot] << '\n';
            allPassed = false;
            continue;
        }
        sdk::Tensor<float> y(problem.yDims(), TensorLayout::NCHW);
        y.fillWithValue(0.0f);
        v8_sample::execute(
            handle,
            *plans[slot],
            {X_UID, W_UID, Y_UID},
            {x.memory().deviceData(), w.memory().deviceData(), y.memory().deviceData()});
        y.memory().markDeviceModified();
        allPassed = validateConv(problem, TensorLayout::NCHW, x, w, y) && allPassed;
    }
    return allPassed;
}

int runOverhead(cudnnHandle_t handle, const v8_sample::Options& options)
{
    if(envEnabled("HIPDNN_LOG_LEVEL") || envEnabled("CUDNN_LOGINFO_DBG")
       || envEnabled("CUDNN_FRONTEND_LOG_INFO"))
    {
        std::cout << "Skipping overhead: logging enabled\n";
        ++v8_sample::summary().skipped;
        return v8_sample::finish(options, true);
    }

    const ConvProblem problem = v8_sample::problemFrom(options.config, {1, 8, 8, 8, 8});
    const fe::Operation op = makeFpropOp(problem, TensorLayout::NCHW, CUDNN_DATA_FLOAT);

    Clock::time_point start = Clock::now();
    fe::OperationGraph graph = v8_sample::makeGraph(handle, {&op});
    const double m3 = msSince(start);

    start = Clock::now();
    const int64_t engineCount = graph.getEngineCount();
    const double m4 = msSince(start);
    std::cout << "Engines for the graph: " << engineCount << '\n';

    start = Clock::now();
    const fe::EngineConfigList configs = v8_sample::instantConfigs(graph);
    if(configs.empty())
    {
        v8_sample::skipVariant("overhead fp32 [NCHW]");
        return v8_sample::finish(options, true);
    }
    const fe::ExecutionPlan plan = fe::ExecutionPlanBuilder()
                                       .setHandle(handle)
                                       .setEngineConfig(configs[0], graph.getTag())
                                       .build();
    const double m5 = msSince(start);
    std::cout << "Plan: " << plan.getTag() << '\n';

    sdk::Tensor<float> x(problem.xDims(), TensorLayout::NCHW);
    sdk::Tensor<float> w(problem.wDims(), TensorLayout::NCHW);
    sdk::Tensor<float> y(problem.yDims(), TensorLayout::NCHW);
    x.fillWithRandomValues(0.0f, 1.0f);
    w.fillWithRandomValues(0.0f, 1.0f);
    y.fillWithValue(0.0f);
    const sdk::Workspace workspace(static_cast<size_t>(plan.getWorkspaceSize()));
    void* workspacePtr = workspace.get();
    std::array<void*, 3> pointers
        = {x.memory().deviceData(), w.memory().deviceData(), y.memory().deviceData()};
    const std::array<int64_t, 3> uids = {X_UID, W_UID, Y_UID};

    const fe::VariantPack pack = fe::VariantPackBuilder()
                                     .setWorkspacePointer(workspacePtr)
                                     .setDataPointers(3, pointers.data())
                                     .setUids(3, uids.data())
                                     .build();

    // The baseline runs the hipDNN plan and variant pack behind the shim's objects directly
    // (get_raw_desc() is the hipDNN descriptor). It calls the C symbols so the frontend's
    // backend lookup is measured.
    hipdnnBackendDescriptor_t hipPlan = plan.get_raw_desc();
    hipdnnBackendDescriptor_t hipPack = pack.get_raw_desc();

    auto m1Shim = [&]() {
        return cudnnBackendExecute(handle, plan.get_raw_desc(), pack.get_raw_desc())
               == CUDNN_STATUS_SUCCESS;
    };
    auto m1Raw
        = [&]() { return hipdnnBackendExecute(handle, hipPlan, hipPack) == HIPDNN_STATUS_SUCCESS; };
    auto m2Shim = [&]() {
        const fe::VariantPack perCall = fe::VariantPackBuilder()
                                            .setWorkspacePointer(workspacePtr)
                                            .setDataPointers(3, pointers.data())
                                            .setUids(3, uids.data())
                                            .build();
        return cudnnBackendExecute(handle, plan.get_raw_desc(), perCall.get_raw_desc())
               == CUDNN_STATUS_SUCCESS;
    };
    auto m2Raw = [&]() {
        hipdnnBackendDescriptor_t perCall = nullptr;
        bool ok = hipdnnBackendCreateDescriptor(HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR, &perCall)
                  == HIPDNN_STATUS_SUCCESS;
        ok = ok
             && hipdnnBackendSetAttribute(perCall,
                                          HIPDNN_ATTR_VARIANT_PACK_DATA_POINTERS,
                                          HIPDNN_TYPE_VOID_PTR,
                                          3,
                                          static_cast<const void*>(pointers.data()))
                    == HIPDNN_STATUS_SUCCESS;
        ok = ok
             && hipdnnBackendSetAttribute(
                    perCall, HIPDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, HIPDNN_TYPE_INT64, 3, uids.data())
                    == HIPDNN_STATUS_SUCCESS;
        ok = ok
             && hipdnnBackendSetAttribute(perCall,
                                          HIPDNN_ATTR_VARIANT_PACK_WORKSPACE,
                                          HIPDNN_TYPE_VOID_PTR,
                                          1,
                                          static_cast<const void*>(&workspacePtr))
                    == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendFinalize(perCall) == HIPDNN_STATUS_SUCCESS;
        ok = ok && hipdnnBackendExecute(handle, hipPlan, perCall) == HIPDNN_STATUS_SUCCESS;
        if(perCall != nullptr)
        {
            ok = hipdnnBackendDestroyDescriptor(perCall) == HIPDNN_STATUS_SUCCESS && ok;
        }
        return ok;
    };

    const OverheadResult m1 = measure("M1", m1Shim, m1Raw, EXECUTE_BUDGET_NS);
    const OverheadResult m2 = measure("M2", m2Shim, m2Raw, BUILD_AND_EXECUTE_BUDGET_NS);

    // PyTorch's two engine-config sources: instant heuristics, then the fallback list.
    const std::array<fe::GeneratorSource, 2> sources
        = {v8_sample::instantConfigs, [](fe::OperationGraph& opGraph) -> fe::EngineConfigList {
               fe::EngineFallbackList fallback
                   = fe::EngineFallbackListBuilder()
                         .setOperationGraph(opGraph)
                         .setOperation(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR)
                         .build();
               return fallback.getFallbackList();
           }};
    fe::EngineConfigGenerator generator(static_cast<int>(sources.size()), sources.data());
    const fe::executionPlans_t candidates = generator.cudnnGetPlan(handle, graph);
    std::set<std::string> distinct;
    for(const fe::ExecutionPlan& candidate : candidates)
    {
        distinct.insert(candidate.getTag());
    }
    // Each candidate may need more workspace than the plan the timing pack above was sized for.
    int64_t timingWorkspaceSize = 0;
    for(const fe::ExecutionPlan& candidate : candidates)
    {
        timingWorkspaceSize = std::max(timingWorkspaceSize, candidate.getWorkspaceSize());
    }
    const sdk::Workspace timingWorkspace(static_cast<size_t>(timingWorkspaceSize));
    const fe::VariantPack timingPack = fe::VariantPackBuilder()
                                           .setWorkspacePointer(timingWorkspace.get())
                                           .setDataPointers(3, pointers.data())
                                           .setUids(3, uids.data())
                                           .build();
    const fe::executionPlans_t timed
        = fe::time_sorted_plan<fe::CudnnFindSamplingTechnique::CUDNN_FIND_SAMPLE_ONCE>(
            handle, candidates, timingPack);

    const bool concurrent = concurrentPlanBuild(handle, configs[0], graph.getTag(), problem, x, w);

    printOverhead("M1", m1);
    printOverhead("M2", m2);
    std::cout << std::fixed << std::setprecision(3) << "REPORT M3=" << m3 << "ms M4=" << m4
              << "ms M5=" << m5 << "ms\n";
    std::cout << "BENCHMARK plans=" << candidates.size() << " distinct=" << distinct.size()
              << " timed=" << timed.size() << " workspace=" << timingWorkspaceSize << '\n';
    std::cout << "CONCURRENT " << (concurrent ? "PASS" : "FAIL") << '\n';

    v8_sample::ranVariant();
    // Wall-clock budgets depend on runner load, so they gate only when asked to.
    const bool withinBudget = m1.pass() && m2.pass();
    if(!withinBudget && !options.has("--enforce-budget"))
    {
        std::cout << "Overhead budget exceeded; reported only (pass --enforce-budget to gate)\n";
    }
    return v8_sample::finish(options,
                             concurrent && (withinBudget || !options.has("--enforce-budget")));
}

} // namespace

template <typename InputType, typename IntermediateType>
bool SampleRunner::operator()(const TensorLayout& layout)
{
    std::ostringstream variant;
    variant << hipdnn_frontend::getDataTypeEnumFromType<InputType>() << " [" << layout << "]";
    std::cout << "Running v8 convolution fprop " << variant.str()
              << (config.cpuValidation ? " (with CPU validation)" : "") << "...\n";

    const ConvProblem problem = v8_sample::problemFrom(config, {16, 16, 16, 16, 16});
    const fe::Operation op = makeFpropOp(problem, layout, v8_sample::cudnnTypeOf<InputType>());
    fe::OperationGraph graph = v8_sample::makeGraph(handle, {&op});

    std::optional<fe::ExecutionPlan> plan = v8_sample::firstPlan(handle, graph);
    if(!plan)
    {
        return v8_sample::skipVariant(variant.str());
    }
    std::cout << "Plan: " << plan->getTag() << '\n';

    sdk::Tensor<InputType> x(problem.xDims(), layout);
    sdk::Tensor<InputType> w(problem.wDims(), layout);
    sdk::Tensor<InputType> y(problem.yDims(), layout);
    x.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    w.fillWithRandomValues(static_cast<InputType>(0.0f), static_cast<InputType>(1.0f));
    y.fillWithValue(static_cast<InputType>(0.0f));

    v8_sample::execute(handle,
                       *plan,
                       {X_UID, W_UID, Y_UID},
                       {x.memory().deviceData(), w.memory().deviceData(), y.memory().deviceData()});
    y.memory().markDeviceModified();
    v8_sample::ranVariant();

    bool validationPassed = true;
    if(config.cpuValidation)
    {
        std::cout << "CPU reference validation:\n";
        validationPassed = validateConv(problem, layout, x, w, y);
    }

    std::cout << "v8 convolution fprop complete for " << variant.str() << ".\n\n";
    return validationPassed;
}

int main(int argc, char** argv)
{
    return v8_sample::guardedMain([argc, argv]() {
        RETURN_SUCCESS_IF_NO_DEVICE();

        const v8_sample::Options options
            = v8_sample::parseOptions(argc, argv, {"--overhead", "--enforce-budget"});

        const auto handle = v8_sample::createHandle();
        if(options.has("--overhead"))
        {
            return runOverhead(*handle, options);
        }
        const bool allPassed = run(SampleRunner{*handle, options.config});
        return v8_sample::finish(options, allPassed);
    });
}
