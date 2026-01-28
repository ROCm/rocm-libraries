# MIOpen Plugin Knobs

This document describes the configuration knobs available for the MIOpen legacy plugin. These knobs allow users to fine-tune the behavior of the MIOpen engine.

## Overview

The hipDNN frontend knob API allows consumers to send configuration values to engines implemented in plugins. Knobs are passed through `Graph::create_execution_plan_ext()` and affect how the plugin builds and executes operations.

## Global Knobs

### `global.benchmarking`

*   **Type**: Integer (Boolean)
*   **Default**: 0 (Disabled)
*   **Description**: Enables or disables benchmarking mode for MIOpen operations.
    *   When set to `1` (Enabled): The plugin will trigger MIOpen's solver tuning mechanism. This involves running various solver implementations to find the most performant one for the given problem configuration. The results are updated in the performance database. This process can be time-consuming but ensures optimal performance for subsequent runs.
    *   When set to `0` (Disabled): The plugin uses the default or previously cached solver configuration. This is faster for startup but may not provide the absolute best performance if the optimal solver hasn't been found yet.

#### Supported Operations

The benchmarking knob is supported by the following MIOpen operations:

| Operation | Plan Builder | Notes |
|-----------|-------------|-------|
| Convolution Forward | `MiopenConvFwdPlan` | Triggers `miopenFindSolutions` with tuning enabled |
| Convolution Backward Data | `MiopenConvBwdPlan` | Triggers `miopenFindSolutions` with tuning enabled |
| Convolution Backward Weights | `MiopenConvWrwPlan` | Triggers `miopenFindSolutions` with tuning enabled |
| Convolution Forward + Bias + Activation | `MiopenConvFwdBiasActivPlan` | Triggers fusion plan tuning |
| BatchNorm Forward Training | `MiopenBatchnormFwdTrainingPlan` | Triggers `miopenFindSolutions` with tuning enabled |
| BatchNorm Forward Inference | `MiopenBatchnormFwdInferencePlan` | Triggers `miopenFindSolutions` with tuning enabled |
| BatchNorm Forward Inference (Variance) | `MiopenBatchnormFwdInferenceWithVariancePlan` | Triggers `miopenFindSolutions` with tuning enabled |
| BatchNorm Backward | `MiopenBatchnormBwdPlan` | Triggers `miopenFindSolutions` with tuning enabled |

#### Implementation Details

When benchmarking is enabled, the plugin uses a `ScopedTuningPolicy` RAII guard to temporarily enable MIOpen's solver search. This:
1. Sets the MIOpen find mode to enable exhaustive solver search
2. Updates the MIOpen performance database with the best-found solver
3. Restores the original find mode when the scope exits

#### Usage Example

```cpp
#include <hipdnn_frontend/Knob.hpp>
#include <hipdnn_plugin_sdk/GlobalKnobDefines.hpp>

// Build the graph
hipdnn_frontend::graph::Graph graph;
// ... configure graph nodes ...

// Validate and build operation graph
graph.validate();
graph.build_operation_graph(handle);

// Get available engines
std::vector<int64_t> engineIds;
graph.get_ranked_engine_ids(engineIds);

// Create execution plan with benchmarking enabled
std::vector<hipdnn_frontend::KnobSetting> settings;
settings.emplace_back(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME, static_cast<int64_t>(1));
graph.create_execution_plan_ext(engineIds[0], settings);

// Build and execute
graph.build_plans();
graph.execute(handle, variantPack, workspace);
```

#### Querying Available Knobs

You can query the knobs supported by an engine before creating an execution plan:

```cpp
// Get knob descriptors for an engine
std::vector<hipdnn_frontend::Knob> knobs;
graph.get_knobs_for_engine(engineId, knobs);

// Or get a lookup map by knob name
std::unordered_map<std::string, hipdnn_frontend::Knob> knobLookup;
graph.get_knob_lookup_for_engine(engineId, knobLookup);

// Check if benchmarking is supported
if (knobLookup.find(hipdnn_plugin_sdk::BENCHMARKING_KNOB_NAME) != knobLookup.end()) {
    // Engine supports benchmarking knob
}
```

## Performance Considerations

*   **First Run**: When benchmarking is enabled for the first time on a specific problem configuration, the solver search may take significant time (seconds to minutes depending on the operation complexity).
*   **Subsequent Runs**: After the optimal solver is cached in the performance database, subsequent runs will be fast regardless of the benchmarking setting.
*   **Database Location**: The MIOpen performance database is typically stored in `~/.config/miopen/` on Linux.
*   **Recommendation**: Enable benchmarking during model warm-up or offline tuning phases, then disable it for production inference.
