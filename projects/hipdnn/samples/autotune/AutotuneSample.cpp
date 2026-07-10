// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Standalone sample demonstrating the hipDNN autotune API end-to-end.
// Follows the pattern of KnobsUsage.cpp: a main() with demonstration
// scenarios in individual functions.
//
// Run with --help for usage information.
// Default mode uses compact tensors and the RUN_UNTIL_STABLE strategy. Use
// --strategy=average to benchmark with FIXED_AVERAGE instead, --iterations=N to
// set the iteration count, and --large for larger tensor dimensions.

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hipdnn_data_sdk/utilities/EngineNames.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_data_sdk/utilities/Workspace.hpp>
#include <hipdnn_frontend.hpp>

#include "../utils/Helpers.hpp"

using namespace hipdnn_frontend;
using namespace hipdnn_data_sdk;

// --- Helpers ---

/// Retrieves the engine name for a given engine ID. Returns a hex-formatted
/// fallback string when the engine ID is not in the registered engine map.
static std::string getEngineName(int64_t engineId)
{
    try
    {
        return std::string(utilities::getEngineNameFromId(engineId));
    }
    catch(const std::out_of_range&)
    {
        std::ostringstream oss;
        oss << "engine_0x" << std::hex << std::uppercase << engineId;
        return oss.str();
    }
}

// --- ConvGraph helper ---

/// All state needed to run autotune scenarios against a convolution graph.
struct ConvGraphState
{
    std::shared_ptr<graph::Graph> graph;
    std::shared_ptr<graph::Tensor_attributes> xAttr;
    std::shared_ptr<graph::Tensor_attributes> wAttr;
    std::shared_ptr<graph::Tensor_attributes> yAttr;
    std::unique_ptr<utilities::Tensor<float>> xTensor;
    std::unique_ptr<utilities::Tensor<float>> wTensor;
    std::unique_ptr<utilities::Tensor<float>> yTensor;
    std::unordered_map<int64_t, void*> variantPack;
};

/// Creates a convolution fprop graph, validates it, builds the operation graph,
/// and allocates device tensors. Does NOT call build() (that is the simple path).
/// For autotune, call add_*() + autotune() after this function.
static ConvGraphState buildConvGraph(hipdnnHandle_t handle, bool largeMode)
{
    // Tensor dimensions: compact by default for fast completion, larger with --large
    // for meaningful timing differentiation.
    const int64_t n = largeMode ? 64 : 1;
    const int64_t c = largeMode ? 32 : 4;
    const int64_t h = largeMode ? 32 : 4;
    const int64_t w = largeMode ? 32 : 4;

    const int64_t k = largeMode ? 32 : 4;
    // Filter channels must match input channels
    const int64_t r = 3;
    const int64_t s = 3;

    const int64_t padH = 1;
    const int64_t padW = 1;
    const int64_t strideH = 1;
    const int64_t strideW = 1;
    const int64_t dilH = 1;
    const int64_t dilW = 1;

    const auto inputType = hipdnn_frontend::DataType::FLOAT;
    const auto layout = utilities::TensorLayout::NCHW;

    auto graph = std::make_shared<graph::Graph>();
    graph->set_io_data_type(inputType).set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    auto xAttr = createTensor({n, c, h, w}, inputType, layout);
    auto wAttr = createTensor({k, c, r, s}, inputType, layout);

    graph::ConvFpropAttributes convAttributes;
    convAttributes.set_name("autotune_conv_fprop");
    convAttributes.set_padding({padH, padW});
    convAttributes.set_stride({strideH, strideW});
    convAttributes.set_dilation({dilH, dilW});

    auto yAttr = graph->conv_fprop(xAttr, wAttr, convAttributes);
    yAttr->set_output(true);

    HIPDNN_FE_CHECK(graph->validate());
    HIPDNN_FE_CHECK(graph->build_operation_graph(handle));

    // Allocate device tensors
    auto xTensor = std::make_unique<utilities::Tensor<float>>(xAttr->get_dim(), layout);
    auto wTensor = std::make_unique<utilities::Tensor<float>>(wAttr->get_dim(), layout);
    auto yTensor = std::make_unique<utilities::Tensor<float>>(yAttr->get_dim(), layout);

    xTensor->fillWithRandomValues(0.0f, 1.0f);
    wTensor->fillWithRandomValues(0.0f, 1.0f);
    yTensor->fillWithValue(0.0f);

    std::unordered_map<int64_t, void*> variantPack;
    variantPack[xAttr->get_uid()] = xTensor->memory().deviceData();
    variantPack[wAttr->get_uid()] = wTensor->memory().deviceData();
    variantPack[yAttr->get_uid()] = yTensor->memory().deviceData();

    ConvGraphState state;
    state.graph = std::move(graph);
    state.xAttr = std::move(xAttr);
    state.wAttr = std::move(wAttr);
    state.yAttr = std::move(yAttr);
    state.xTensor = std::move(xTensor);
    state.wTensor = std::move(wTensor);
    state.yTensor = std::move(yTensor);
    state.variantPack = std::move(variantPack);

    return state;
}

// --- Scenario 1: Quick Autotune ---

/// Simplest possible autotune flow:
/// add_all_engines() -> autotune() -> execute()
static void demonstrateStandardAutotune(hipdnnHandle_t handle,
                                        bool largeMode,
                                        AutotuneStrategy strategy,
                                        int iterations)
{
    std::cout << "\n=== Scenario 1: Quick Autotune ===\n";

    auto state = buildConvGraph(handle, largeMode);

    // Discover and add all available engines
    HIPDNN_FE_CHECK(state.graph->add_all_engines());

    // Allocate workspace for the largest engine
    int64_t maxWs = 0;
    HIPDNN_FE_CHECK(state.graph->get_estimated_max_workspace_size(maxWs));
    const utilities::Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::STANDARD;
    config.strategy = strategy;
    config.warmupIterations = 1;
    if(iterations > 0)
    {
        if(strategy == AutotuneStrategy::RUN_UNTIL_STABLE)
        {
            config.maxIterations = iterations;
            // windowSize must stay <= maxIterations and >= 2
            config.windowSize = std::max(2, std::min(config.windowSize, iterations));
        }
        else // FIXED_AVERAGE
        {
            config.timedIterations = iterations;
        }
    }

    // Autotune selects the fastest engine and sets it as active
    HIPDNN_FE_CHECK(
        state.graph->autotune(handle, state.variantPack, workspace.get(), maxWs, config));

    // After autotune(), get_workspace_size() returns the active plan's workspace
    int64_t ws = 0;
    HIPDNN_FE_CHECK(state.graph->get_workspace_size(ws));
    const utilities::Workspace execWorkspace(static_cast<size_t>(ws));

    // Execute with the autotuned engine
    HIPDNN_FE_CHECK(state.graph->execute(handle, state.variantPack, execWorkspace.get()));

    // Verify output
    state.yTensor->memory().markDeviceModified();
    const auto* yHostPtr = state.yTensor->memory().hostData();
    std::cout << "  Execution OK, output[0..4]: ";
    for(int i = 0; i < 5; ++i)
    {
        std::cout << yHostPtr[i] << " ";
    }
    std::cout << '\n';

    std::cout << "  Autotuned successfully.\n";
}

// --- Scenario 2: Exhaustive Autotune with Result Inspection ---

/// Uses EXHAUSTIVE mode and inspects the ranked results.
/// Prints a table showing all engines with timing and status.
static void demonstrateExhaustiveAutotune(hipdnnHandle_t handle,
                                          bool largeMode,
                                          AutotuneStrategy strategy,
                                          int iterations)
{
    std::cout << "\n=== Scenario 2: Exhaustive Autotune ===\n";

    auto state = buildConvGraph(handle, largeMode);

    // Discover and add all available engines
    HIPDNN_FE_CHECK(state.graph->add_all_engines());

    int64_t maxWs = 0;
    HIPDNN_FE_CHECK(state.graph->get_estimated_max_workspace_size(maxWs));
    const utilities::Workspace workspace(static_cast<size_t>(maxWs));

    // EXHAUSTIVE mode; strategy/iterations applied the same way as Scenario 1
    AutotuneConfig config;
    config.mode = TuneMode::EXHAUSTIVE;
    config.strategy = strategy;
    config.warmupIterations = 1;
    config.primingFailurePolicy = PrimingFailurePolicy::BENCHMARK_UNPRIMED;
    if(iterations > 0)
    {
        if(strategy == AutotuneStrategy::RUN_UNTIL_STABLE)
        {
            config.maxIterations = iterations;
            config.windowSize = std::max(2, std::min(config.windowSize, iterations));
        }
        else // FIXED_AVERAGE
        {
            config.timedIterations = iterations;
        }
    }

    // Use the overload that populates results
    std::vector<AutotuneResult> results;
    HIPDNN_FE_CHECK(state.graph->autotune(
        handle, state.variantPack, workspace.get(), maxWs, config, {}, &results));

    // Print ranked results table
    std::cout << "  " << std::left << std::setw(6) << "Rank" << std::setw(30) << "Engine"
              << std::setw(12) << "Min (ms)" << std::setw(12) << "Avg (ms)" << std::setw(14)
              << "Workspace" << std::setw(10) << "Converged" << std::setw(12) << "Exhaustive"
              << '\n';
    std::cout << "  " << std::string(96, '-') << '\n';

    for(const auto& result : results)
    {
        const std::string rankStr = result.rank >= 0 ? std::to_string(result.rank) : "FAIL";
        const std::string name
            = result.engineName.empty() ? getEngineName(result.engineId) : result.engineName;

        std::cout << "  " << std::left << std::setw(6) << rankStr << std::setw(30) << name;

        if(result.succeeded)
        {
            std::cout << std::setw(12) << std::fixed << std::setprecision(4) << result.minTimeMs
                      << std::setw(12) << std::fixed << std::setprecision(4) << result.avgTimeMs
                      << std::setw(14) << result.workspaceSize << std::setw(10)
                      << (result.converged ? "yes" : "no") << std::setw(12)
                      << (result.ranExhaustive ? "yes" : "no");
        }
        else
        {
            std::cout << std::setw(12) << "n/a" << std::setw(12) << "n/a" << std::setw(14)
                      << result.workspaceSize << std::setw(10) << "n/a" << std::setw(12) << "n/a";
            if(!result.errorMessage.empty())
            {
                std::cout << " [" << result.errorMessage << "]";
            }
        }
        std::cout << '\n';
    }

    std::cout << "  Total engines benchmarked: " << results.size() << '\n';
}

// --- Scenario 3: Filtered Autotune ---

/// Demonstrates engine discovery and workspace-constrained autotuning using
/// pre-filtering by estimated workspace size.
///
/// This scenario shows the recommended pattern for workspace-constrained
/// autotune when you want to avoid compiling engines that are too large:
///   1. Call get_engine_configs() to inspect EngineConfigInfo entries
///   2. Filter by estimatedWorkspaceSize before adding to the graph
///   3. Call add_engine_configs() with only the engines that fit
///   4. Run autotune() on the pre-filtered set
///
/// This avoids compiling engines whose estimated workspace exceeds the budget,
/// saving time compared to the general overload (which compiles all engines
/// first, then filters by actual compiled workspace size).
///
/// Alternative: general autotune overload with workspace limit parameter
// ---
/// Instead of pre-filtering, you can use the general autotune() overload that
/// takes a workspaceSize parameter. This compiles and benchmarks all candidates,
/// then filters out any plan whose actual (compiled) workspace exceeds the limit:
///
///   graph->add_all_engines();
///   int64_t maxWs = 0;
///   graph->get_estimated_max_workspace_size(maxWs);
///   Workspace workspace(maxWs);
///
///   AutotuneConfig config;
///   config.mode = TuneMode::STANDARD;
///
///   int64_t workspaceBudget = 256 * 1024 * 1024;  // 256 MB limit
///   std::vector<AutotuneResult> results;
///   // General overload: pass workspaceBudget as the 4th argument
///   graph->autotune(handle, variantPack, workspace.get(),
///                   workspaceBudget, config, {}, &results);
///
/// In the general overload, plans exceeding the workspace limit appear in
/// results with succeeded=false. If the fastest plan is too large, the
/// next-best plan that fits is selected automatically.
static void demonstrateFilteredAutotune(hipdnnHandle_t handle,
                                        bool largeMode,
                                        AutotuneStrategy strategy,
                                        int iterations)
{
    std::cout << "\n=== Scenario 3: Filtered Autotune (Workspace Constrained) ===\n";

    auto state = buildConvGraph(handle, largeMode);

    // Step 1: Discover available engines
    std::vector<EngineConfigInfo> configs;
    HIPDNN_FE_CHECK(state.graph->get_engine_configs(configs));

    std::cout << "  Discovered " << configs.size() << " engine(s):\n";
    for(const auto& cfg : configs)
    {
        const std::string name
            = cfg.engineName.empty() ? getEngineName(cfg.engineId) : cfg.engineName;
        std::cout << "    " << name << " (workspace=" << cfg.estimatedWorkspaceSize
                  << ", exhaustive=" << (cfg.supportsExhaustive ? "yes" : "no")
                  << ", knobs=" << cfg.knobs.size() << ")\n";
    }

    // Step 2: Filter engines by workspace size
    // Use a generous limit: allow any engine with workspace <= 256 MB
    const int64_t workspaceLimit = int64_t{256} * 1024 * 1024;
    std::vector<EngineConfigInfo> filteredConfigs;
    for(const auto& cfg : configs)
    {
        if(cfg.estimatedWorkspaceSize <= workspaceLimit)
        {
            filteredConfigs.push_back(cfg);
        }
    }

    std::cout << "  After filtering (workspace <= " << workspaceLimit
              << " bytes): " << filteredConfigs.size() << " engine(s)\n";

    if(filteredConfigs.empty())
    {
        std::cout << "  No engines within workspace limit, skipping autotune.\n";
        return;
    }

    // Step 3: Add only filtered engines
    HIPDNN_FE_CHECK(state.graph->add_engine_configs(filteredConfigs));

    // Allocate workspace up to the limit
    int64_t maxWs = 0;
    HIPDNN_FE_CHECK(state.graph->get_estimated_max_workspace_size(maxWs));
    const int64_t allocatedWs = std::min(maxWs, workspaceLimit);
    const utilities::Workspace workspace(static_cast<size_t>(allocatedWs));

    // Step 4: Autotune with pre-filtered engines
    AutotuneConfig config;
    config.mode = TuneMode::STANDARD;
    config.strategy = strategy;
    config.warmupIterations = 1;
    if(iterations > 0)
    {
        if(strategy == AutotuneStrategy::RUN_UNTIL_STABLE)
        {
            config.maxIterations = iterations;
            config.windowSize = std::max(2, std::min(config.windowSize, iterations));
        }
        else // FIXED_AVERAGE
        {
            config.timedIterations = iterations;
        }
    }

    HIPDNN_FE_CHECK(
        state.graph->autotune(handle, state.variantPack, workspace.get(), allocatedWs, config));

    std::cout << "  Autotuned with workspace constraint successfully.\n";
}

// --- Scenario 4: Save Results to Config File ---

/// Autotunes and saves results to a JSON config file that can be reused via
/// HIPDNN_HEUR_CONFIG_PATH environment variable.
static void demonstrateSaveToConfigFile(hipdnnHandle_t handle,
                                        bool largeMode,
                                        AutotuneStrategy strategy,
                                        int iterations)
{
    std::cout << "\n=== Scenario 4: Save Results to Config File ===\n";

    const std::filesystem::path configFile = "sample_autotune_results.json";

    auto state = buildConvGraph(handle, largeMode);

    // Discover and add all available engines
    HIPDNN_FE_CHECK(state.graph->add_all_engines());

    int64_t maxWs = 0;
    HIPDNN_FE_CHECK(state.graph->get_estimated_max_workspace_size(maxWs));
    const utilities::Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::STANDARD;
    config.strategy = strategy;
    config.warmupIterations = 1;
    if(iterations > 0)
    {
        if(strategy == AutotuneStrategy::RUN_UNTIL_STABLE)
        {
            config.maxIterations = iterations;
            config.windowSize = std::max(2, std::min(config.windowSize, iterations));
        }
        else // FIXED_AVERAGE
        {
            config.timedIterations = iterations;
        }
    }

    // Write results to a config file
    const AutotuneStorageConfig storageConfig{configFile, false};

    HIPDNN_FE_CHECK(state.graph->autotune(
        handle, state.variantPack, workspace.get(), maxWs, config, storageConfig));

    std::cout << "  Results saved to " << configFile << '\n';
    std::cout << "  To reuse: export HIPDNN_HEUR_CONFIG_PATH=" << configFile << '\n';

    // Clean up the demo file
    std::error_code removeEc;
    const bool removed = std::filesystem::remove(configFile, removeEc);
    if(removed && removeEc.value() == 0)
    {
        std::cout << "  (Demo file cleaned up)\n";
    }
}

// --- Scenario 5: Compiled-Plan Autotune ---

/// Demonstrates the compiled-plan autotune path (cuDNN-compatible workflow):
/// create_execution_plans() -> build_plans(ALL) -> autotune() -> execute()
static void demonstrateCompiledPlanAutotune(hipdnnHandle_t handle,
                                            bool largeMode,
                                            AutotuneStrategy strategy,
                                            int iterations)
{
    std::cout << "\n=== Scenario 5: Compiled-Plan Autotune ===\n";

    auto state = buildConvGraph(handle, largeMode);

    // Create execution plans using fallback heuristic mode
    HIPDNN_FE_CHECK(state.graph->create_execution_plans({HeuristicMode::FALLBACK}));

    // Build all plans so they are available for autotuning
    HIPDNN_FE_CHECK(state.graph->build_plans(BuildPlanPolicy::ALL));

    // Allocate workspace large enough for any candidate plan
    const int64_t maxWs = state.graph->get_autotune_workspace_size();
    const utilities::Workspace workspace(static_cast<size_t>(maxWs));

    AutotuneConfig config;
    config.mode = TuneMode::STANDARD;
    config.strategy = strategy;
    config.warmupIterations = 1;
    if(iterations > 0)
    {
        if(strategy == AutotuneStrategy::RUN_UNTIL_STABLE)
        {
            config.maxIterations = iterations;
            config.windowSize = std::max(2, std::min(config.windowSize, iterations));
        }
        else // FIXED_AVERAGE
        {
            config.timedIterations = iterations;
        }
    }

    // Autotune across all compiled plans
    std::vector<AutotuneResult> results;
    HIPDNN_FE_CHECK(
        state.graph->autotune(handle, state.variantPack, workspace.get(), config, {}, &results));

    // Print results table (ranked by autotune performance)
    std::cout << "  " << std::left << std::setw(6) << "Rank" << std::setw(30) << "Plan Name"
              << std::setw(14) << "Workspace" << std::setw(12) << "Time (ms)" << '\n';
    std::cout << "  " << std::string(62, '-') << '\n';

    for(size_t i = 0; i < results.size(); ++i)
    {
        const auto& result = results[i];
        const std::string name
            = result.engineName.empty() ? getEngineName(result.engineId) : result.engineName;

        std::cout << "  " << std::left << std::setw(6) << i << std::setw(30) << name
                  << std::setw(14) << result.workspaceSize;

        if(result.succeeded)
        {
            std::cout << std::setw(12) << std::fixed << std::setprecision(4) << result.minTimeMs;
        }
        else
        {
            std::cout << std::setw(12) << "FAIL";
        }
        std::cout << '\n';
    }

    // Execute with the autotuned winner
    int64_t ws = 0;
    HIPDNN_FE_CHECK(state.graph->get_workspace_size(ws));
    const utilities::Workspace execWorkspace(static_cast<size_t>(ws));

    HIPDNN_FE_CHECK(state.graph->execute(handle, state.variantPack, execWorkspace.get()));

    std::string winnerName;
    HIPDNN_FE_CHECK(state.graph->get_plan_name(winnerName));
    std::cout << "  Winner: " << winnerName << '\n';
    std::cout << "  Compiled-plan autotune completed successfully.\n";
}

// --- Scenario 6: Manual Benchmark Loop ---

/// Demonstrates the cuDNN-compatible manual benchmark loop:
/// create_execution_plans() -> build_plans(ALL) -> plan-indexed iteration
/// with HIP event timing and workspace filtering.
static void demonstrateManualBenchmarkLoop(hipdnnHandle_t handle, bool largeMode)
{
    std::cout << "\n=== Scenario 6: Manual Benchmark Loop ===\n";

    auto state = buildConvGraph(handle, largeMode);

    // Create execution plans and build all candidates
    HIPDNN_FE_CHECK(state.graph->create_execution_plans({HeuristicMode::FALLBACK}));
    HIPDNN_FE_CHECK(state.graph->build_plans(BuildPlanPolicy::ALL));

    const int64_t planCount = state.graph->get_execution_plan_count();
    std::cout << "  Found " << planCount << " compiled plan(s)\n";

    // Print plan info table before benchmarking
    std::cout << "  " << std::left << std::setw(8) << "Index" << std::setw(30) << "Plan Name"
              << std::setw(14) << "Workspace" << '\n';
    std::cout << "  " << std::string(52, '-') << '\n';

    for(int64_t i = 0; i < planCount; ++i)
    {
        std::string name;
        HIPDNN_FE_CHECK(state.graph->get_plan_name_at_index(i, name));
        const int64_t planWs = state.graph->get_workspace_size_plan_at_index(i);
        std::cout << "  " << std::left << std::setw(8) << i << std::setw(30) << name
                  << std::setw(14) << planWs << '\n';
    }

    // Workspace budget: 256 MB
    const int64_t workspaceBudget = int64_t{256} * 1024 * 1024;

    // Allocate workspace large enough for any candidate plan
    const int64_t maxWs = state.graph->get_autotune_workspace_size();
    const utilities::Workspace workspace(static_cast<size_t>(maxWs));

    // Manual benchmark loop with HIP event timing
    hipEvent_t startEvent;
    hipEvent_t stopEvent;
    HIP_CHECK(hipEventCreate(&startEvent));
    HIP_CHECK(hipEventCreate(&stopEvent));

    float bestTimeMs = std::numeric_limits<float>::max();
    int64_t bestIndex = -1;

    std::cout << "\n  Benchmarking plans:\n";

    for(int64_t i = 0; i < planCount; ++i)
    {
        const int64_t planWs = state.graph->get_workspace_size_plan_at_index(i);
        if(planWs > workspaceBudget)
        {
            std::string name;
            HIPDNN_FE_CHECK(state.graph->get_plan_name_at_index(i, name));
            std::cout << "    [" << i << "] " << name << " -- skipped (workspace " << planWs
                      << " exceeds budget)\n";
            continue;
        }

        HIP_CHECK(hipEventRecord(startEvent));
        auto result
            = state.graph->execute_plan_at_index(handle, state.variantPack, workspace.get(), i);
        HIP_CHECK(hipEventRecord(stopEvent));
        HIP_CHECK(hipEventSynchronize(stopEvent));

        if(!result.is_good())
        {
            std::string name;
            HIPDNN_FE_CHECK(state.graph->get_plan_name_at_index(i, name));
            std::cout << "    [" << i << "] " << name << " -- failed (" << result.get_message()
                      << ")\n";
            continue;
        }

        float elapsedMs = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsedMs, startEvent, stopEvent));

        std::string name;
        HIPDNN_FE_CHECK(state.graph->get_plan_name_at_index(i, name));
        std::cout << "    [" << i << "] " << name << " -- " << std::fixed << std::setprecision(4)
                  << elapsedMs << " ms\n";

        if(elapsedMs < bestTimeMs)
        {
            bestTimeMs = elapsedMs;
            bestIndex = i;
        }
    }

    HIP_CHECK(hipEventDestroy(startEvent));
    HIP_CHECK(hipEventDestroy(stopEvent));

    if(bestIndex < 0)
    {
        std::cout << "  No plan succeeded during manual benchmark loop.\n";
        return;
    }

    // Activate the best plan
    HIPDNN_FE_CHECK(state.graph->build_plan_at_index(bestIndex));

    int64_t ws = 0;
    HIPDNN_FE_CHECK(state.graph->get_workspace_size(ws));
    const utilities::Workspace execWorkspace(static_cast<size_t>(ws));

    HIPDNN_FE_CHECK(state.graph->execute(handle, state.variantPack, execWorkspace.get()));

    std::string winnerName;
    HIPDNN_FE_CHECK(state.graph->get_plan_name(winnerName));
    std::cout << "  Winner: " << winnerName << " (" << std::fixed << std::setprecision(4)
              << bestTimeMs << " ms)\n";
    std::cout << "  Manual benchmark loop completed successfully.\n";
}

// --- Argument Parsing ---

struct AutotuneSampleConfig
{
    int scenario = 0; // 0 = run all, 1-6 = specific scenario
    bool largeMode = false;
    AutotuneStrategy strategy = AutotuneStrategy::RUN_UNTIL_STABLE; // --strategy
    int iterations = 0; // -- iterations; 0 = strategy defaults, >0 = override
};

static AutotuneSampleConfig parseArgs(int argc, char** argv)
{
    AutotuneSampleConfig cfg;

    for(int i = 1; i < argc; ++i)
    {
        const std::string arg(argv[i]);

        if(arg == "--help" || arg == "-h")
        {
            std::cout
                << "Usage: " << argv[0] << " [OPTIONS]\n"
                << "  --scenario=N    Run specific scenario (1-6, default=all)\n"
                << "  --large         Use larger tensor dimensions\n"
                << "  --strategy=S    Benchmarking strategy: stable (RUN_UNTIL_STABLE,\n"
                << "                  default) or average (FIXED_AVERAGE)\n"
                << "  --iterations=N  Iteration count: max iterations for stable, timed\n"
                << "                  iterations for average (default: strategy default)\n"
                << "  --verify-cpu    Accepted but ignored (compatibility with test harness)\n"
                << "  --help, -h      Show this help\n"
                << "\n"
                << "Scenarios:\n"
                << "  Plan-spec path (hipDNN-native autotune):\n"
                << "    1  Quick autotune           add_all_engines -> autotune -> execute\n"
                << "    2  Exhaustive autotune       EXHAUSTIVE mode with ranked result table\n"
                << "    3  Filtered autotune         get_engine_configs -> workspace filter -> "
                   "add_engine_configs\n"
                << "    4  Save to config file       autotune with AutotuneStorageConfig for "
                   "persistence\n"
                << "\n"
                << "  Compiled-plan path (cuDNN-compatible):\n"
                << "    5  Compiled-plan autotune   create_execution_plans -> build_plans(ALL) -> "
                   "autotune\n"
                << "    6  Manual benchmark loop    Plan-indexed iteration with timing and "
                   "workspace filtering\n";
            exit(EXIT_SUCCESS);
        }

        if(arg.rfind("--scenario=", 0) == 0)
        {
            cfg.scenario = std::stoi(arg.substr(11));
            if(cfg.scenario < 1 || cfg.scenario > 6)
            {
                std::cerr << "Invalid scenario: " << cfg.scenario << " (must be 1-6)\n";
                exit(EXIT_FAILURE);
            }
        }
        else if(arg == "--large")
        {
            cfg.largeMode = true;
        }
        else if(arg.rfind("--strategy=", 0) == 0)
        {
            const std::string value = arg.substr(11);
            if(value == "stable")
            {
                cfg.strategy = AutotuneStrategy::RUN_UNTIL_STABLE;
            }
            else if(value == "average")
            {
                cfg.strategy = AutotuneStrategy::FIXED_AVERAGE;
            }
            else
            {
                std::cerr << "Invalid --strategy: " << value << " (use stable or average)\n";
                exit(EXIT_FAILURE);
            }
        }
        else if(arg.rfind("--iterations=", 0) == 0)
        {
            cfg.iterations = std::stoi(arg.substr(13));
            if(cfg.iterations < 1)
            {
                std::cerr << "--iterations count must be >= 1\n";
                exit(EXIT_FAILURE);
            }
        }
        else if(arg == "--verify-cpu" || arg == "-vc")
        {
            // Accepted but ignored: autotune has no CPU validation path.
            // This flag is passed by add_hipdnn_sample_test().
        }
        else
        {
            std::cerr << "Unknown argument: " << arg << " (use --help)\n";
            exit(EXIT_FAILURE);
        }
    }

    return cfg;
}

// --- Main ---

int main(int argc, char* argv[])
{
    try
    {
        const auto config = parseArgs(argc, argv);

        std::cout << "hipDNN Autotune Sample"
                  << (config.largeMode ? " (large mode)" : " (fast mode)") << '\n';

        initializeFrontendLogging();

        // Check GPU availability
        int deviceCount = 0;
        if(hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount == 0)
        {
            std::cout << "SKIPPED: No GPU devices available.\n";
            return 0;
        }

        hipdnnHandle_t handle = nullptr;
        HIPDNN_CHECK(hipdnnCreate(&handle));

        if(config.scenario == 0 || config.scenario == 1)
        {
            demonstrateStandardAutotune(
                handle, config.largeMode, config.strategy, config.iterations);
        }
        if(config.scenario == 0 || config.scenario == 2)
        {
            demonstrateExhaustiveAutotune(
                handle, config.largeMode, config.strategy, config.iterations);
        }
        if(config.scenario == 0 || config.scenario == 3)
        {
            demonstrateFilteredAutotune(
                handle, config.largeMode, config.strategy, config.iterations);
        }
        if(config.scenario == 0 || config.scenario == 4)
        {
            demonstrateSaveToConfigFile(
                handle, config.largeMode, config.strategy, config.iterations);
        }
        if(config.scenario == 0 || config.scenario == 5)
        {
            demonstrateCompiledPlanAutotune(
                handle, config.largeMode, config.strategy, config.iterations);
        }
        if(config.scenario == 0 || config.scenario == 6)
        {
            demonstrateManualBenchmarkLoop(handle, config.largeMode);
        }

        HIPDNN_CHECK(hipdnnDestroy(handle));

        std::cout << "\nDone.\n";
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
