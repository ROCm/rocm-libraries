# RFC: Autotuning for hipDNN

**Status**: Draft
**Authors**: Chris Miserva

## Table of Contents

1. [Summary](#1-summary)
2. [Motivation](#2-motivation)
3. [Requirements](#3-requirements)
4. [Terminology](#4-terminology)
5. [Architecture](#5-architecture)
6. [API Design](#6-api-design)
   - [6.1 Core Types](#61-core-types)
   - [6.2 Engine Discovery and Plan Spec Collection](#62-engine-discovery-and-plan-spec-collection)
     - [6.2.1 Concepts: Graph, Engine Configs, and Execution Plans](#621-concepts-graph-engine-configs-and-execution-plans)
     - [6.2.2 Types](#622-types)
     - [6.2.3 API on Graph](#623-api-on-graph)
     - [6.2.4 Plan Spec Collection Semantics](#624-plan-spec-collection-semantics)
     - [6.2.5 What Happens Under the Hood](#625-what-happens-under-the-hood)
     - [6.2.6 Typical Flow](#626-typical-flow)
     - [6.2.7 Filtering](#627-filtering)
   - [6.3 Autotuning API on Graph](#63-autotuning-api-on-graph)
   - [6.4 Benchmarking Flows](#64-benchmarking-flows)
   - [6.5 Config File Output](#65-config-file-output)
7. [Porting Guide: cuDNN to hipDNN](#7-porting-guide-cudnn--hipdnn)
8. [Complete Example](#8-complete-example)
9. [Risks and Mitigations](#9-risks-and-mitigations)
10. [Test Plan / Strategy](#10-test-plan--strategy)
- [Appendix A: Porting Guide from cuDNN Autotuning to hipDNN](#appendix-a-porting-guide-from-cudnn-autotuning-to-hipdnn)
  - [A.1 Auto Workflow Comparison Examples](#a1-auto-workflow-comparison-examples)
  - [A.2 Step-by-Step API Mapping](#a2-step-by-step-api-mapping)
  - [A.3 Key Differences and Similarities](#a3-key-differences-and-similarities)
- [Appendix B: Use Cases](#appendix-b-use-cases)

---

## 1. Summary

This RFC defines the autotuning system for hipDNN. Autotuning benchmarks execution plans (engine configurations) for a given operation graph on the current hardware and selects the fastest one, replacing heuristic-only engine selection with empirically measured performance data.

The autotuning runtime lives in `hipdnn_frontend` and persists results to **engine override configuration files** in the same JSON format used by `HIPDNN_HEUR_CONFIG_PATH`. On subsequent runs, the `SelectionHeuristic::Config` policy loads these files and selects the autotuned engine without re-benchmarking.

Runtime caching is out of scope; a general caching facility will be addressed in a future RFC.

---

## 2. Motivation

Deep learning operations can be computed by many algorithms, each with different performance characteristics depending on input dimensions, data types, workspace memory, and GPU architecture. No single algorithm is universally fastest, and heuristic engine ordering may be inaccurate for unusual configurations, new hardware, or corner cases. Autotuning addresses this by empirically measuring each candidate.

---

## 3. Requirements

1. **Two tuning modes**: Auto-tune (simple wall-time) and Exhaustive-tune (internal priming via temporary plans + wall-time)
2. **Config file output**: Write ranked winners to engine override JSON files, enabling reuse via `HIPDNN_HEUR_CONFIG_PATH`
3. **Benchmarking strategies**: Single-shot, fixed-average, run-until-stable
4. **Separated phases**: Inspect/filter engines between discovery and plan spec collection
5. **Knob variant autotuning**: Benchmark the same engine with different knob configurations
6. **cuDNN API parity**: Match cuDNN's autotuning support, then extend
7. **Extensible ranking**: User-provided ranking function for custom criteria

---

## 4. Terminology

| Term | Meaning |
|------|---------|
| **Auto-tune mode** | Benchmark engines using standard execution (no special flags). |
| **Exhaustive-tune mode** | Benchmark with engine-internal cache priming (plan-spec path only; see § 6.3). `autotune()` builds temporary priming plans using each plan spec's knob settings plus `global.benchmarking=1`, executes them once to prime engine caches, discards them, then compiles the real plans before timed runs. |
| **Benchmarking knob** | The `global.benchmarking` knob: a per-engine flag that tells the engine to initialize its internal cache state (e.g., MIOpen's `find` phase). Not all engines support this. On the plan-spec path, it is managed exclusively by `autotune()` in EXHAUSTIVE mode; `add_engine_*()` strips it with a warning log (see § 6.2.4). On the compiled-plan path, users set it directly via `create_execution_plan_ext()` knob settings. |
| **Graph signature** | Deterministic hash of operation graph structure, tensor shapes, data types, and compute configuration. |
| **Device signature** | Deterministic key derived from GPU properties (architecture, CU count, memory). |
| **Engine config** | Lightweight descriptor of an available engine implementation for a graph (`EngineConfigInfo`): engine ID, name, available knobs, workspace size, and exhaustive benchmarking support. Returned by `get_engine_configs()`. |
| **Plan spec** | An engine ID paired with specific knob settings. The composite key `(engineId, knobSettings)` uniquely identifies an autotuning candidate. Stored by `add_engine_*()` calls. |
| **Engine variant** | User-facing `EngineVariant` struct pairing an engine ID with specific knob settings for explicit plan spec construction. |

---

## 5. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                    hipdnn_frontend (header-only)                     │
│                                                                      │
│  ┌─────────────────┐    ┌─────────────────────────────────────────┐  │
│  │ Autotuning      │    │ Config File Writer                      │  │
│  │ Runtime         │───▶│ (EngineOverrideConfig JSON format)      │  │
│  │                 │    │                                         │  │
│  │ • AUTO mode     │    │ • Write ranked winners                  │  │
│  │ • EXHAUSTIVE    │    │ • Append to existing file               │  │
│  │   mode          │    │ • Replace matching (op, tensors)        │  │
│  │ • Strategies    │    │ • Autotune metadata                     │  │
│  │                 │    └─────────────────────────────────────────┘  │
│  └─────────────────┘                                                 │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    hipdnn_backend                                    │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐    │
│  │ SelectionHeuristic::Config (policy)                          │    │
│  │                                                              │    │
│  │ • Loads HIPDNN_HEUR_CONFIG_PATH at policyFinalize()          │    │
│  │ • matchOperation(op, tensors) - reorders engine candidates   │    │
│  │ • Returns matched engineId for matching ops                  │    │
│  └──────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

> The config file is integrated as the `SelectionHeuristic::Config` policy, which loads `HIPDNN_HEUR_CONFIG_PATH` and reorders engine candidates during `create_execution_plans()`. Autotuning output uses the same JSON format consumed by this policy.

---

## 6. API Design

### 6.1 Core Types

```cpp
enum class TuneMode {
    AUTO,        // Simple wall-time comparison (no engine-internal cache priming)
    EXHAUSTIVE   // Build temporary priming plans internally, prime engine caches, then wall-time comparison
};

enum class AutotuneStrategy {
    SINGLE_SHOT,       // 1 timed run, take the result
    FIXED_AVERAGE,     // Average of N runs
    RUN_UNTIL_STABLE   // Run until timing variance stabilizes, up to a cap (default)
};

// User-provided ranking comparator (optional)
using AutotuneRankingFn = std::function<void(std::vector<AutotuneResult>&)>;

struct AutotuneConfig {
    TuneMode mode = TuneMode::AUTO;
    AutotuneStrategy strategy = AutotuneStrategy::RUN_UNTIL_STABLE;

    // Warmup
    int warmupIterations = 1;

    // FIXED_AVERAGE parameter
    int timedIterations = 10;

    // RUN_UNTIL_STABLE parameters
    // Validation: windowSize >= 2, maxIterations >= windowSize,
    // stabilityThreshold must be in range (0.0, 1.0) exclusive.
    // autotune() returns an error if any of these constraints are violated.
    int maxIterations = 100;
    int windowSize = 3;
    float stabilityThreshold = 0.05f;      // Coefficient of variation threshold (5%)

    // Engine filter: filters candidates (empty = all). Unmatched IDs silently ignored.
    std::vector<int64_t> engineIdFilter;

    // Custom ranking (nullptr = rank by minTimeMs)
    AutotuneRankingFn rankingFn = nullptr;

    // EXHAUSTIVE mode: when false (default), abort on priming failure.
    // When true: benchmark without priming; ranExhaustive=false, errorMessage notes reason.
    // No effect in AUTO mode.
    bool continueOnPrimingFailure = false;

};

struct AutotuneStorageConfig {
    // Config file output:
    std::filesystem::path filePath;
    bool deleteAllExistingFileContent = false;  // When true, clear file before writing
};

struct AutotuneResult {
    int rank;                               // 0-based (0 = fastest); -1 for failed engines
    int64_t engineId;
    std::vector<KnobSetting> knobSettings;  // Informational, records knobs explicitly set on the engine.
    std::string engineName;
    float minTimeMs;                        // Used for ranking
    float avgTimeMs;
    float stddevMs;                         // 0.0 for SINGLE_SHOT
    int iterationsRun;                      // Actual iterations executed
    bool converged;                         // true for SINGLE_SHOT/FIXED_AVERAGE; false only for
                                            // RUN_UNTIL_STABLE when maxIterations reached.
    int64_t workspaceSize;
    bool succeeded;
    std::string errorMessage;
    TuneMode modeUsed;
    bool ranExhaustive;                     // true if primed via temporary benchmarking plan;
                                            // false for AUTO mode or unsupported engines.
    AutotuneStrategy strategyUsed = AutotuneStrategy::RUN_UNTIL_STABLE;
                                            // Strategy used (for config file metadata)
    std::string deviceName;                 // e.g., "gfx942" (for config file metadata)
};
```

### 6.2 Engine Discovery and Plan Spec Collection

#### 6.2.1 Concepts: Graph, Engine Configs, and Execution Plans

The autotuning preparation pipeline has four stages:

1. **Operation graph**: The computation description (operations, tensor shapes, data types). Built once via `build_operation_graph(handle)`. Not executable at this stage.

2. **Engine configs**: Lightweight descriptors of available engine implementations for the graph (see `EngineConfigInfo` in § 6.2.2). No kernels are compiled at this stage.

3. **Plan specs**: Engine ID + knob settings, collected via `add_engine_*()` calls and stored on the graph without compiling any kernels. An intermediate stage between engine configs and compiled plans, representing the set of autotuning candidates.

4. **Execution plans**: Compiled, executable kernel plans. Compiled by `build_plans()` (existing non-autotune path) or by `autotune()` (which compiles plan specs, then benchmarks them).

The key design difference from cuDNN: users can inspect engine configs, create variants, and collect plan specs *before* paying the cost of kernel compilation. Compilation is deferred until `autotune()` (see also § A.3 point 3).

#### 6.2.2 Types

```cpp
// Read-only snapshot of an engine. Provided for inspection and filtering only. Do not
//modify fields directly. Use add_engine_*() to create plan specs from selected configs.
struct EngineConfigInfo {
    int64_t engineId;                  // Used by add_engine_configs() to create the plan spec
    std::string engineName;            // Informational, for filtering and logging
    std::vector<Knob> knobs;           // Informational, shows the engine's available knobs.
                                       // Ignored by add_engine_*() functions. Use add_engine_variants()
                                       // or add_engine() to set custom knobs on engines for autotune().
    bool supportsExhaustive;           // Informational. For filtering exhaustive-capable engines
    int64_t estimatedWorkspaceSize;    // Informational, pre-compile workspace estimate, for filtering.
};

struct EngineVariant {
    int64_t engineId;
    std::map<KnobType_t, KnobValueVariant> knobSettings;
};
// Note: EngineVariant::knobSettings uses std::map (user-friendly input);
// AutotuneResult/PlanSpec use std::vector<KnobSetting> (internal). Converted by add_engine_variants().

struct KnobSweepAxis {
    KnobType_t knobId;
    std::vector<KnobValueVariant> values;
};

struct EngineSweepSpec {
    int64_t engineId;
    std::vector<KnobSweepAxis> axes;            // Knobs to sweep (Cartesian product)
    std::map<KnobType_t, KnobValueVariant> fixedSettings;  // Knobs held constant for each combination of axes
};
```

#### 6.2.3 API on Graph

**Engine discovery**:

```cpp
Error get_engine_configs(std::vector<EngineConfigInfo>& configs,
                         const std::vector<HeuristicMode>& modes = {HeuristicMode::FALLBACK});
```

Queries the backend for engine implementations applicable to the current operation graph on the current hardware, ranked by the specified heuristic mode(s). Only engines that pass the backend's applicability check are returned.

**Plan spec collection** (for autotuning):

All `add_engine_*()` functions store plan specs on the graph without compiling kernels; compilation is deferred to `autotune()`. Multiple calls can be composed on the same graph (see § 6.2.4).

```cpp
// Add plan specs using each engine config's default knob settings.
Error add_engine_configs(const std::vector<EngineConfigInfo>& configs);

// Add plan specs by engine ID list with default knob settings.
// Convenience overload: loops calling add_engine(id) for each entry.
Error add_engines(const std::vector<int64_t>& engineIds);

// Add a single plan spec, optionally with explicit knob settings.
Error add_engine(int64_t engineId,
                 const std::vector<KnobSetting>& knobSettings = {});

// Add plan specs from explicit user-defined variants (engine ID + knob settings).
Error add_engine_variants(const std::vector<EngineVariant>& variants);

// Add plan specs from a Cartesian product sweep over knob axes for each engine.
Error add_engine_sweep(const std::vector<EngineSweepSpec>& specs);
```

**Mutual exclusion**: `add_engine_*()` returns an error if compiled plans already exist on the graph (i.e., `create_execution_plans()`, `create_execution_plan_ext()`, or `build()` was already called). The two paths are mutually exclusive per Graph instance (see § 6.3).

**Validation**: `add_engine_*()` validates knob names and ranges at call time but does NOT compile. See § 6.2.4 for complete validation rules.

**Convenience shortcut**:

```cpp
// Discover all engines and add plan specs with default knobs. Equivalent to:
//   get_engine_configs(configs, modes);
//   add_engine_configs(configs);
Error add_all_engines(const std::vector<HeuristicMode>& modes = {HeuristicMode::FALLBACK});
```

**Workspace query**:

```cpp
// Returns the estimated maximum workspace size across all added plan specs.
// Pre-compile estimate from engine config metadata; actual post-compile workspace
// may differ (compilation can change requirements). Not guaranteed >= actual.
// Call after add_engine_*(), before allocating workspace for autotune/execute.
Error get_estimated_max_workspace_size(int64_t& maxSize) const;
```

**Workspace sizing**:
1. **Plan-spec path**: `get_estimated_max_workspace_size()` returns a pre-compile estimate. Allocate this size and pass it to the general `autotune()` overload. All plan specs are compiled regardless of their estimated workspace size — `autotune()` does not pre-filter by estimate. After compilation, plans whose actual workspace exceeds the provided `workspaceSize` are skipped (not executed). Users who want to avoid compiling engines with large workspace estimates should filter `EngineConfigInfo` entries before calling `add_engine_configs()` (see example below).
2. **Compiled-plan path**: Use `get_autotune_workspace_size()` for the accurate post-compile maximum.
3. After autotuning, `execute()` only needs the winner's `workspaceSize`, which may be smaller.

**User-side pre-filtering** (plan-spec path, optional):
```cpp
auto configs = graph->get_engine_configs({HeuristicMode::A});
for (auto& c : configs) {
    if (c.estimatedWorkspaceSize <= myBudget)
        graph->add_engine_configs({c});
}
graph->autotune(handle, variantPack, workspace, myBudget, config);  // General overload
```
This avoids compiling engines whose _estimated_ workspace exceeds the budget.

#### 6.2.4 Plan Spec Collection Semantics

**Additive with deduplication**: All `add_engine_*()` calls append to the same internal plan spec list, deduplicated by `(engineId, knobSettings)`. Identical entries are silently skipped; the same engine with *different* knob settings produces distinct entries (variant autotuning). This deduplication key does not carry through to the config file match key, which uses `(operation, tensors)` only (see § 6.5).

**Zero specs added**: Empty input returns an error. Batch calls (`add_engine_variants()`, `add_engine_sweep()`) that skip all entries due to validation return success with a warning log per skipped entry.

**Empty knob settings = default**: `add_engine(id, {})` is equivalent to `add_engine(id)`.

Multiple `add_engine_*()` calls can be composed freely:

```cpp
// Add default-knob specs for engines you don't want to sweep
graph->add_engine_configs(baseline_configs);

// Add knob-swept variant specs for engines you do want to sweep
graph->add_engine_variants(miopen_variants);

// Dedup ensures no redundant specs even if engine IDs overlap.
// autotune() compiles and benchmarks all specs.
```

**Hardware support filtering**: `get_engine_configs()` only returns engines applicable to the current graph and hardware. Engines added via `add_engine()` or `add_engine_variants()` with IDs not from `get_engine_configs()` may fail compilation inside `autotune()`; batch operations skip them with a warning log.

**Engine ID validation** (checked at add time):
- `add_engine()`: Hard error for invalid engine IDs.
- `add_engine_configs()`: Does NOT validate engine IDs at add time; trusts that configs came from `get_engine_configs()`. Invalid IDs fail later during compilation inside `autotune()`.
- `add_engine_variants()` and `add_engine_sweep()`: Skip invalid engine IDs with a warning log (batch semantics).
- Nonexistent knob or out-of-range knob value: hard error (message identifies the first invalid entry).
- `global.benchmarking` knob specified: stripped with a warning log (see § 4 for rationale).

#### 6.2.5 What Happens Under the Hood

All `add_engine_*()` functions validate then store (see § 6.2.4). No compilation occurs. If the caller already knows valid engine IDs, `add_engine()`, `add_engine_variants()`, and `add_engine_sweep()` can be called without first calling `get_engine_configs()`. Per-function details:

**`add_engine_configs(configs)`**: For each `EngineConfigInfo`, stores a plan spec using the engine's default knob settings.

**`add_engine_variants(variants)`**: For each `EngineVariant`, stores a plan spec using the specified `engineId` and `knobSettings`.

**`add_engine_sweep(specs)`**: For each `EngineSweepSpec`, computes the Cartesian product of all `axes` values, merges each combination with `fixedSettings`, then stores each as `add_engine_variants` does. Returns an error if the product exceeds 10,000 plan specs per call; a warning is logged above 1,000.

**Cartesian product example:**

```cpp
EngineSweepSpec spec;
spec.engineId = 42;  // engine name: "MIOPEN_ENGINE_X"
spec.axes = {
    {"SPLIT_K",   {int64_t{1}, int64_t{2}, int64_t{4}}},
    {"TILE_SIZE", {int64_t{0}, int64_t{1}}}
};
spec.fixedSettings = {{"REDUCTION_MODE", int64_t{1}}};
```

The sweep produces 3 x 2 = 6 plan specs:

| Spec | `SPLIT_K` | `TILE_SIZE` | `REDUCTION_MODE` (fixed) |
|------|-----------|-------------|--------------------------|
| 1 | 1 | 0 | 1 |
| 2 | 1 | 1 | 1 |
| 3 | 2 | 0 | 1 |
| 4 | 2 | 1 | 1 |
| 5 | 4 | 0 | 1 |
| 6 | 4 | 1 | 1 |

Each row is a separate plan spec. All share the fixed `REDUCTION_MODE=1`, while `SPLIT_K` and `TILE_SIZE` vary across all combinations.

#### 6.2.6 Typical Flow

```
1. graph->build_operation_graph(handle)   // build the operation graph
2. graph->get_engine_configs(configs)     // discover engine configs (no compilation)
3. // user inspects, filters, reorders configs
4. graph->add_engine_*(...)               // validate inputs, store plan specs (no compilation)
5. auto estimatedWorkspaceSize = graph->get_estimated_max_workspace_size()
6. // user allocates workspace of at least estimatedWorkspaceSize bytes
7. graph->autotune(handle, ..., estimatedWorkspaceSize) // compile plan specs, then benchmark
```

`add_all_engines()` is a convenience shortcut for steps 2-4 using all engines with default knobs.

> **Note**: The existing non-autotune flow (`build_operation_graph()` → `create_execution_plan_ext()` → `check_support()` → `build_plans()` → `execute()`) remains unchanged. `autotune()` also accepts pre-compiled plans; see § 6.3.

#### 6.2.7 Filtering

```cpp
// Remove plan specs or compiled plans whose workspace exceeds the given limit.
Graph& deselect_workspace_greater_than(int64_t workspace);

// Remove plan specs or compiled plans whose engine name matches any entry in engine_names.
Graph& deselect_engines(const std::vector<std::string>& engine_names);

// Remove plan specs or compiled plans whose engine ID matches any entry in engine_ids.
Graph& deselect_engines(const std::vector<int64_t>& engine_ids);
```

All filtering methods return `Graph&` for method chaining. They operate on whichever collection is populated: `_planSpecs` (plan-spec path) or `_compiledPlans` (compiled-plan path).

> **Index invalidation**: `deselect_*` methods remove entries from the underlying collection, causing indices to shift. Any previously cached indices (e.g., from `get_execution_plan_count()` or `execute_plan_at_index()`) are invalidated after each call. Re-query `get_execution_plan_count()` after deselection. This differs from cuDNN, which marks plans as "barred" without removing them (see also § A.3 point 5).

### 6.3 Autotuning API on Graph

Three groups of `autotune()` overloads (each with UID-based and tensor-attribute-based variants, 6 overloads total):

**Compiled-plan overloads; hipDNN overloads with no workspace-size parameter (for compiled plans only):**
```cpp
Error autotune(hipdnnHandle_t handle,
               const std::unordered_map<int64_t, void*>& variantPack,
               void* workspace,
               const AutotuneConfig& config = {},
               const AutotuneStorageConfig& storageConfig = {},
               std::vector<AutotuneResult>* results = nullptr);
```
Compiled-plan path only. Returns an error if plan specs exist. No workspace-size filtering, assumes the caller has allocated workspace to the maximum reported by compiled plans (via `get_autotune_workspace_size()`).

**General overloads; hipDNN overloads with required workspace-size parameter:**
```cpp
Error autotune(hipdnnHandle_t handle,
               const std::unordered_map<int64_t, void*>& variantPack,
               void* workspace,
               int64_t workspaceSize,
               const AutotuneConfig& config = {},
               const AutotuneStorageConfig& storageConfig = {},
               std::vector<AutotuneResult>* results = nullptr);
```
Works with both paths. `workspaceSize` is the caller's allocated workspace buffer size. All plan specs are compiled regardless of estimated workspace size. Workspace sizes of compiled plans are typically the same or smaller than estimated, but can be larger. Plans whose actual (post-compile) workspace exceeds `workspaceSize` are skipped with a warning log if the pre-compile estimate indicated the plan would fit. Skipped plans are captured in `AutotuneResult` with both the estimated and actual workspace sizes. To avoid compiling engines with large estimated workspace sizes, pre-filter `EngineConfigInfo` before `add_engine_configs()` (see § 6.2.3).

**cuDNN-compatible overloads; forwarding overloads matching cuDNN's signature:**
```cpp
// UID-based
Error autotune(hipdnnHandle_t handle,
               std::unordered_map<int64_t, void*>& variantPack,
               void* workspace,
               void* user_impl = nullptr);

// Tensor-attribute-based
Error autotune(hipdnnHandle_t handle,
               std::unordered_map<std::shared_ptr<TensorAttributes>, void*>& tensorLookup,
               void* workspace,
               void* user_impl = nullptr);
```
Forward to the compiled-plan overloads with defaults (AUTO, RUN_UNTIL_STABLE, no file output, no results). Compiled-plan path only.

All overload groups benchmark execution plans and select the best. Subsequent `execute()` calls use the winning plan. Pass a non-null `results` pointer to get ranked results; omit or pass `nullptr` to discard them. If `storageConfig.filePath` is non-empty, results are also written to the config file.

**Two data paths**: `autotune()` accepts candidates from either of two mutually exclusive paths (see § 6.2.1) and auto-detects which to use:

1. **Plan-spec path** (`add_engine_*()` → `autotune()`): `autotune()` compiles stored plan specs, then benchmarks. Both AUTO and EXHAUSTIVE modes available. In EXHAUSTIVE mode, priming plans are built and executed first (see § 6.4).

2. **Compiled-plan path** (`create_execution_plans()` → `build_plans(ALL)` → `autotune()`): `autotune()` benchmarks pre-compiled plans directly. AUTO mode only (EXHAUSTIVE returns an error; use `create_execution_plan_ext()` knob settings for `global.benchmarking` instead for compiled plans path).

If both or neither path's artifacts are present, `autotune()` returns an error. The two paths are mutually exclusive per Graph instance (see § 6.2.3).

**`create_execution_plan_ext()` limitation**: Clears compiled plans on each call, so only one plan can exist at a time. Multi-plan autotuning on the compiled-plan path requires `create_execution_plans()` → `build_plans(ALL)`.

**Error conditions**: `autotune()` returns an error (with error log) when:

*Upfront checks (validated before benchmarking begins):*
- Invalid handle
- `variantPack` fails tensor UID validation (`INVALID_VALUE`): all non-virtual tensor UIDs required by the graph must be present. Missing UIDs are reported in the error message.
- Workspace pointer null but plans require workspace
- No autotuning candidates (no plan specs and no compiled plans)
- EXHAUSTIVE mode requested on compiled-plan path
- `RUN_UNTIL_STABLE` parameter validation fails (see `AutotuneConfig` constraints in § 6.1)

*Runtime errors (discovered during benchmarking):*
- All plans fail compilation or benchmarking (every result `succeeded = false`)
- HIP memory operations fail
- EXHAUSTIVE mode: priming failure when `continueOnPrimingFailure` is `false` (default)

Failed entries (`succeeded = false`) are always placed after successful entries, with `rank = -1`.

**Plan selection**: After `autotune()` completes, the winning plan is stored on the graph and subsequent `execute()` calls use it, same as after `build(handle)`.

**Ranking function contract**:
- Must only reorder the vector in-place; must not add, remove, or modify elements.
- Receives only succeeded entries; failed entries are appended afterward.
- `autotune()` re-assigns `rank` fields (0-based) after the callback returns.
- If the function throws, `autotune()` falls back to default ranking (by `minTimeMs`).

**`engineIdFilter` behavior**: On both paths, `engineIdFilter` silently ignores engine IDs that have no matching candidates. No warning is logged for unmatched filter entries.

**Tensor data validity**: Autotuning assumes idempotent execution or separate input/output buffers. Repeated execution may produce different results for non-idempotent operations (in-place ops or operations where the output is also an input).

**`HIPDNN_HEUR_CONFIG_PATH` interaction**: The config file influences engine ordering in the heuristic selection policy but does not affect `autotune()`, which benchmarks all candidates independently.

### 6.4 Benchmarking Flows

**AUTO mode**:
```
1. Plan-spec path: compile all plan specs into execution plans
   Compiled-plan path: use pre-compiled plans as-is
2. For each compiled plan:
     a. device synchronize (ensure GPU is idle before this plan's benchmarking)
     b. warmup iterations (discard timing)
     c. device synchronize
     d. [strategy-specific timed iteration loop]
     e. compute avgTimeMs and track minTimeMs
sort by minTimeMs ascending (or user rankingFn)
```

**EXHAUSTIVE mode** (plan-spec path only):
```
1. For each plan spec where the engine's supportsExhaustive is true:
     * build a temporary priming plan using the plan spec's knob settings plus global.benchmarking = 1
     * execute the priming plan once
     * discard the priming plan
     * On priming failure:
       - If continueOnPrimingFailure is false (default): abort, return error
       - If continueOnPrimingFailure is true: mark engine as unprimed, continue
2. Compile all plan specs into execution plans (engines are now primed)
3. For each compiled plan:
     a. device synchronize (ensure GPU is idle before this plan's benchmarking)
     b. warmup iterations (discard timing)
     c. device synchronize
     d. [strategy-specific timed iteration loop]
     e. compute avgTimeMs and track minTimeMs
sort by minTimeMs ascending (or user rankingFn)
```

> **Stream**: `autotune()` uses the stream set on the handle via `hipdnnSetStream()`.

> **Synchronization**: `hipDeviceSynchronize()` is called before and after warmup (before timed runs), preventing cross-plan interference. `hipEventSynchronize(stop)` is used between timed iterations. `hipStreamSynchronize()` is not used during autotuning.

Engines where `supportsExhaustive` is false skip priming and are compiled and benchmarked normally.

When `continueOnPrimingFailure` is `true` and priming fails, the engine is still benchmarked (unprimed). Its `AutotuneResult::ranExhaustive` is `false`, and `errorMessage` notes the priming failure even though `succeeded` may be `true`.

**Strategy implementations**:

- **SINGLE_SHOT**: One `hipEventRecord` pair around one execution. Fast, rough ranking.
- **FIXED_AVERAGE**: Per-iteration event timing for N executions. Reports min, avg, and stddev across all N timings.
- **RUN_UNTIL_STABLE**: Per-iteration event timing. Checks if the coefficient of variation of the last `windowSize` timings is below `stabilityThreshold`. Stops when stable or `maxIterations` reached.

**Warmup failure**: Plans that fail during warmup are marked `succeeded = false`; the autotuner proceeds to the next plan.

### 6.5 Config File Output

Reuses the `EngineOverrideConfig` JSON format with autotuning metadata added:

```json
{
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE_DETERMINISTIC",
      "tensors": [
        { "dim": [16, 64, 56, 56], "stride": [200704, 3136, 56, 1] },
        { "dim": [64, 64, 3, 3], "stride": [576, 9, 3, 1] }
      ],
      "knobs": [
        { "knob_id": "global.workspace_size_limit", "type": "int", "value": 16777216 },
        { "knob_id": "global.search_mode", "type": "int", "value": 2 }
      ],
      "autotune_metadata": {
        "rank": 0,
        "min_time_ms": 0.0412,
        "avg_time_ms": 0.0439,
        "stddev_ms": 0.0018,
        "workspace_size": 16777216,
        "mode": "exhaustive",
        "ran_exhaustive": true,
        "strategy": "run_until_stable",
        "iterations_run": 37,
        "converged": true,
        "device": "gfx942",
        "timestamp": "2026-04-21T10:30:00Z"
      }
    },
    {
      "op": "matmul",
      "engine_name": "ROCBLAS_ENGINE_DEFAULT",
      "tensors": [
        { "dim": [128, 512], "stride": [512, 1] },
        { "dim": [512, 256], "stride": [256, 1] }
      ],
      "autotune_metadata": {
        "rank": 0,
        "min_time_ms": 0.0234,
        "avg_time_ms": 0.0241,
        "stddev_ms": 0.0009,
        "workspace_size": 0,
        "mode": "auto",
        "ran_exhaustive": false,
        "strategy": "fixed_average",
        "iterations_run": 10,
        "device": "gfx942",
        "timestamp": "2026-04-21T10:32:00Z"
      }
    }
  ]
}
```

**Metadata fields** (`autotune_metadata`; informational only, ignored by `matchOperation()`):
- `rank`: 0-based ranking (0 = fastest)
- `min_time_ms`: minimum observed time (used for ranking)
- `avg_time_ms`: average time across timed iterations
- `stddev_ms`: standard deviation (0.0 for SINGLE_SHOT)
- `workspace_size`: workspace bytes required
- `mode`: `"auto"` or `"exhaustive"`
- `ran_exhaustive`: `true` if primed via temporary benchmarking plan, `false` otherwise
- `strategy`: `"single_shot"`, `"fixed_average"`, or `"run_until_stable"`
- `iterations_run`: actual timed iterations executed
- `converged`: present only for `"run_until_stable"`; `true` if variance stabilized, `false` if `maxIterations` reached
- `device`: device name (e.g., `"gfx942"`)
- `timestamp`: ISO 8601 timestamp

**Knob settings in config file entries**: Each entry may include a `knobs` array recording the knob settings active during the winning autotune run. These are informational metadata, not applied on load, not part of the match key. The user is responsible for configuring knobs to match the autotuned configuration if needed (via `add_engine()` with knob settings, or `create_execution_plan_ext()`). Default-knob entries omit the `knobs` key entirely. The `type` field (`"int"`, `"double"`, or `"string"`) enables correct deserialization since JSON does not distinguish `int64_t` from `double`.

The config file is a lightweight engine *selection* hint; plan serialization is the mechanism for full *restoration* of the winning plan with all configuration (knobs, workspace, compilation state).

**Write behavior**: `AutotuneFileWriter` writes the rank-0 winner (fastest successful result) for the graph's core operation. If the file already contains an entry for the same `(operation, tensor shape)`, it is **unconditionally replaced** — there is no comparison against the previous entry's timing. At most one entry exists per `(operation, tensor shape)` combination.

`AutotuneStorageConfig::deleteAllExistingFileContent` (default `false`) controls whether unrelated entries are preserved. When `false`, only the matching entry is replaced; other entries for different operations or tensor shapes are kept. When `true`, all existing content is deleted before writing.

**Concurrent access**: Config file append is not safe for concurrent writers. For concurrent scenarios, use separate output files per process and merge afterward.


**Core operation mapping**: For multi-operation graphs, the config file entry is keyed by the core operation:

1. Convolution, GEMM, SDPA (highest priority)
2. Normalization
3. Pointwise (lowest priority)

## 7. Porting Guide: cuDNN → hipDNN

For API mapping, key differences, and complete porting examples, see Appendix A. For concrete use cases, see Appendix B.

---

## 8. Complete Example

```cpp
#include <hipdnn_frontend.h>

int main() {
    hipdnnHandle_t handle;
    hipdnnCreate(&handle);

    auto graph = std::make_shared<hipdnn_frontend::Graph>();
    // ... configure graph ...
    graph->validate();
    graph->build_operation_graph(handle);

    // Discover engines
    std::vector<hipdnn_frontend::EngineConfigInfo> configs;
    graph->get_engine_configs(configs);

    // Optional user-side pre-filtering: avoid compiling engines with large workspace estimates.
    // Without this, autotune() compiles all plan specs and skips execution of those whose
    // actual post-compile workspace exceeds the provided workspaceSize.
    std::erase_if(configs, [](const auto& c) { return c.estimatedWorkspaceSize > (256 << 20); });

    // Step 1: Add default-knob plan specs per engine config (up to N specs).
    graph->add_engine_configs(configs);

    // Step 2: Add knob-swept plan specs for MIOpen engines (3 variants each).
    std::vector<hipdnn_frontend::EngineVariant> variants;
    for (const auto& config : configs) {
        if (config.engineName.find("MIOPEN") != std::string::npos) {
            for (int64_t ws : {int64_t{0}, int64_t{1 << 20}, int64_t{16 << 20}}) {
                variants.push_back({
                    .engineId = config.engineId,
                    .knobSettings = {{"global.workspace_size_limit", ws}}
                });
            }
        }
    }
    // Duplicates from Step 1 are skipped in add_engine_variants().
    graph->add_engine_variants(variants);

    int64_t maxWorkspaceSize;
    graph->get_estimated_max_workspace_size(maxWorkspaceSize);
    void* workspace;
    hipMalloc(&workspace, maxWorkspaceSize);

    std::unordered_map<int64_t, void*> variantPack;
    // ... populate ...

    // Autotune with convergence-based strategy, persist to config file (general overload)
    std::vector<hipdnn_frontend::AutotuneResult> results;
    auto err = graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
        {.mode = hipdnn_frontend::TuneMode::EXHAUSTIVE,
         .strategy = hipdnn_frontend::AutotuneStrategy::RUN_UNTIL_STABLE,
         .maxIterations = 50,
         .stabilityThreshold = 0.03f,
        {.filePath = "autotune_results.json"},
        &results);

    // Execute with autotuned winner
    graph->execute(handle, variantPack, workspace);

    hipFree(workspace);
    hipdnnDestroy(handle);
    return 0;
}
```

---

## 9. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| **Cartesian product growth** | Error at 10,000 plan specs per `add_engine_sweep()` call, warning at 1,000. Validate-then-store rejects invalid knobs early. |
| **Config file format changes** | Backward-compatible: `matchOperation()` ignores unknown fields. |
| **Config file conflicts on append** | Same (op, tensors) entries are replaced, preventing stale entries. |

---

## 10. Test Plan / Strategy

### 10.1 Approach

Most autotune logic (type validation, deduplication, statistics, ranking, serialization, config parsing, graph API guards) can be unit-tested without a GPU and should be tested alongside implementation.

End-to-end workflows (benchmark engines, pick a winner, `execute()` succeeds) require integration tests on GPU hardware with a real engine plugin.

A sample program demonstrates API usage but is not a source of test coverage; any workflow it shows must also be covered by unit or integration tests.

### 10.2 Unit Tests

Write unit tests alongside each stage of implementation. Key areas:

- **Data structures** (§ 6.1, § 6.2.2): Each new type (`PlanSpec`, `AutotuneConfig`, `AutotuneResult`, `EngineConfigInfo`, etc.):
  - Default construction, field validation, equality semantics
  - `PlanSpec` deduplication: knob order must not affect equality
- **Algorithms**:
  - `CartesianProduct` (§ 6.2.5): correctness across axis counts, combination limit enforcement
  - `BenchmarkStatistics` (§ 6.4): empty input, single value, uniform values, division-by-zero in CoV
- **Autotune logic** (§ 6.3, § 6.4):
  - `global.benchmarking` knob stripped from user-provided plan specs
  - Ranking: default comparator (`minTimeMs`), failed-engine ordering, custom `AutotuneRankingFn`
  - Engine ID filtering as subset operations
  - Post-compile workspace filtering: compiled plans exceeding the provided `workspaceSize` are skipped
  - `RUN_UNTIL_STABLE` convergence detection with known-stable and known-unstable sequences
  - Variant pack validation: null handle, missing tensor UIDs, extra UIDs accepted
- **Config file I/O** (§ 6.5):
  - Knob serialization for each value type (int, double, string)
  - Override entry construction: metadata fields match § 6.5 schema (conditional presence of `converged`; `ran_exhaustive` always present); edge case for tensors with empty strides
  - Write modes: new file, append, replace matching, delete all
  - Round-trip test (write -> load -> `matchOperation()` -> verify knob values)
  - Recovery from corrupt JSON
- **Config reader extensions**: `EngineOverrideConfig` knob parsing:
  - Each knob type, type aliases, wildcards with knobs
  - Missing/empty knobs fields (backward compatibility with pre-autotune config files)
- **Graph API guards** (§ 6.2.3, § 6.3):
  - Mutual exclusion: `add_engine_*()` after `create_execution_plans()` (and vice versa) returns error
  - Precondition: `add_engine_*()` returns error before `build_operation_graph()`
  - Plan-indexed access: out-of-bounds, negative index, uncompiled plan
  - Compiled plan storage: single-plan and multi-plan paths produce consistent vector model; plan spec compilation produces valid plans
  - `deselect_*` methods on both paths; edge cases: all removed, none removed, empty input, unknown names, unqueried workspace

### 10.3 Integration Tests

Integration tests need a **custom test plugin** (not the real MIOpen provider) providing:

- At least 3 engines with different knob configurations and workspace sizes
- At least one engine with a `global.benchmarking` knob for EXHAUSTIVE mode testing
- No-op execution (HIP event timing overhead is sufficient for validating the benchmarking flow)

Cover three autotune workflows end-to-end, verifying `execute()` succeeds after autotuning:

- **Plan-spec path**: `add_all_engines()` -> `autotune()` -> `execute()`. Also test: EXHAUSTIVE mode (`ranExhaustive` flag), config file persistence (write -> reload -> `execute()`), engine ID filtering, post-compile workspace filtering (plans whose actual workspace exceeds the provided buffer are skipped, not pre-filtered by estimate)
- **Compiled-plan path**: `create_execution_plans()` -> `build_plans(ALL)` -> `autotune()` -> `execute()`. Also test: EXHAUSTIVE rejection, multi-engine plan count, `build_plans(ALL)` vs `HEURISTICS_CHOICE`
- **Manual benchmark loop**: `create_execution_plans()` -> `build_plans(ALL)` -> loop `execute_plan_at_index()` -> `build_plan_at_index(best)` -> `execute()`. Verify index selection with multiple engines

---

## Appendix A: Porting Guide from cuDNN Autotuning to hipDNN

Covers how hipDNN autotuning differs from cuDNN and what must change when porting.

Reference cuDNN sample: [cudnn-frontend/samples/cpp/misc/autotuning.cpp](https://github.com/NVIDIA/cudnn-frontend/blob/f26b794e2d36f40cf14fd4af8919d2b097dc546f/samples/cpp/misc/autotuning.cpp#L4).

### A.1 Auto Workflow Comparison Examples

See A.3 for detailed descriptions of the differences.

#### API Equivalence

**Plan-spec path** (hipDNN-native, with engine inspection):

<table>
<tr><th>cuDNN</th><th>hipDNN (plan-spec path)</th></tr>
<tr>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
graph.create_execution_plans({HeurMode_t::A});
graph.check_support();
graph.build_plans(BuildPlanPolicy_t::ALL);

int64_t ws;
ws = graph.get_autotune_workspace_size();
void* workspace;
cudaMalloc(&workspace, ws);

std::unordered_map<int64_t, void*> variant_pack =
    {{a_uid, a_ptr}, {w_uid, w_ptr}, {y_uid, y_ptr}};

graph.autotune(handle, variant_pack, workspace);

graph.execute(handle, variant_pack, workspace);
cudaFree(workspace);
</pre></td>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
graph.add_all_engines();
//
//

int64_t ws;
graph.get_estimated_max_workspace_size(ws);
void* workspace;
hipMalloc(&workspace, ws);

std::unordered_map<int64_t, void*> variant_pack =
    {{a_uid, a_ptr}, {w_uid, w_ptr}, {y_uid, y_ptr}};

graph.autotune(handle, variant_pack, workspace, ws);

graph.execute(handle, variant_pack, workspace);
hipFree(workspace);
</pre></td>
</tr>
</table>

**Compiled-plan path** (cuDNN drop-in):

<table>
<tr><th>cuDNN</th><th>hipDNN (compiled-plan path)</th></tr>
<tr>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
graph.create_execution_plans({HeurMode_t::A});
graph.check_support();
graph.build_plans(BuildPlanPolicy_t::ALL);

int64_t ws;
ws = graph.get_autotune_workspace_size();
void* workspace;
cudaMalloc(&workspace, ws);

std::unordered_map<int64_t, void*> variant_pack =
    {{a_uid, a_ptr}, {w_uid, w_ptr}, {y_uid, y_ptr}};

graph.autotune(handle, variant_pack, workspace);

graph.execute(handle, variant_pack, workspace);
cudaFree(workspace);
</pre></td>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
graph.create_execution_plans({HeuristicMode::FALLBACK});
graph.check_support();
graph.build_plans(BuildPlanPolicy_t::ALL);

int64_t ws;
ws = graph.get_autotune_workspace_size();
void* workspace;
hipMalloc(&workspace, ws);

std::unordered_map<int64_t, void*> variant_pack =
    {{a_uid, a_ptr}, {w_uid, w_ptr}, {y_uid, y_ptr}};

graph.autotune(handle, variant_pack, workspace);

graph.execute(handle, variant_pack, workspace);
hipFree(workspace);
</pre></td>
</tr>
</table>

> The compiled-plan path uses `get_autotune_workspace_size()` (accurate post-compile) rather than `get_estimated_max_workspace_size()` (pre-compile estimate). EXHAUSTIVE mode is not available on this path.

#### Simple Autotune Workflow With hipDNN Extensions

<table>
<tr><th>cuDNN</th><th>hipDNN</th></tr>
<tr>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
graph.create_execution_plans({HeurMode_t::A});
graph.check_support();
graph.build_plans(BuildPlanPolicy_t::ALL);
void* workspace =
    allocate(graph.get_autotune_workspace_size());
graph.autotune(handle, variant_pack, workspace);
//
//
graph.execute(handle, variant_pack, workspace);
</pre></td>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
//
//
graph.add_all_engines();
int64_t ws;
graph.get_estimated_max_workspace_size(ws);
void* workspace = allocate(ws);
graph.autotune(handle, variant_pack, workspace, ws,
    {.mode = TuneMode::EXHAUSTIVE},
    {.filePath = "autotune_results.json"});
graph.execute(handle, variant_pack, workspace);
</pre></td>
</tr>
</table>

#### With Engine Filtering and hipDNN Autotune Extensions

<table>
<tr><th>cuDNN</th><th>hipDNN</th></tr>
<tr>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);

graph.create_execution_plans({HeurMode_t::A});
graph.check_support();
// Opaque filtering (no engine inspection):
graph.deselect_workspace_greater_than(256 << 20);
graph.deselect_engines({"engine_1", "engine_2"});
//
//
graph.build_plans(BuildPlanPolicy_t::ALL);

void* workspace =
    allocate(graph.get_autotune_workspace_size());
graph.autotune(handle, variant_pack, workspace);
//
//
//
graph.execute(handle, variant_pack, workspace);
</pre></td>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);

std::vector<EngineConfigInfo> configs;
graph.get_engine_configs(configs);
std::erase_if(configs, [](const auto& c) {
    return c.estimatedWorkspaceSize > (256 << 20)
        || c.engineName == "engine_1"
        || c.engineName == "engine_2";
});
graph.add_engine_configs(configs);

int64_t ws;
graph.get_estimated_max_workspace_size(ws);
void* workspace = allocate(ws);
graph.autotune(handle, variant_pack,
    workspace, ws,
    {.mode = TuneMode::EXHAUSTIVE},
    {.filePath = "autotune_results.json"});
graph.execute(handle, variant_pack, workspace);
</pre></td>
</tr>
</table>

#### cuDNN Manual Benchmarking Loop vs. hipDNN `autotune()`

The [cuDNN autotuning sample](https://github.com/NVIDIA/cudnn-frontend/blob/f26b794e2d36f40cf14fd4af8919d2b097dc546f/samples/cpp/misc/autotuning.cpp#L4) implements a manual benchmarking loop using `execute_plan_at_index()`. hipDNN's `autotune()` replaces this pattern (first example below). For direct porting, hipDNN also provides the same plan-indexed APIs for a manual loop (second example below).

<table>
<tr><th>cuDNN (manual benchmarking loop)</th><th>hipDNN (RFC)</th></tr>
<tr>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
graph.create_execution_plans({HeurMode_t::A});
graph.check_support();
graph.build_plans(BuildPlanPolicy_t::ALL);

auto plan_count =
    graph.get_execution_plan_count();

// Find max workspace across all plans
int64_t ws = 0;
for (int i = 0; i < plan_count; i++)
    ws = std::max(ws,
        graph.get_workspace_size_plan_at_index(i));
void* workspace;
cudaMalloc(&workspace, ws);

// Benchmark each plan
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
std::vector<float> times(plan_count, 10.0f);

for (int i = 0; i < plan_count; i++) {
    // Warmup
    auto err = graph.execute_plan_at_index(
        handle, variant_pack, workspace, i);
    if (err.is_bad()) continue;
    cudaDeviceSynchronize();

    // Timed iterations
    cudaEventRecord(start, stream);
    for (int iter = 0; iter < 10; iter++)
        graph.execute_plan_at_index(
            handle, variant_pack, workspace, i);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    times[i] = ms / 10.0f;
}

// Select winner
auto best = std::min_element(
    times.begin(), times.end());
auto idx = std::distance(times.begin(), best);

graph.build_plan_at_index(idx);
graph.execute(handle, variant_pack, workspace);
</pre></td>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
//
//
graph.add_all_engines();
//
//
//
//
//
//
//
int64_t maxWorkspaceSize;
graph.get_estimated_max_workspace_size(maxWorkspaceSize);
void* workspace;
hipMalloc(&workspace, maxWorkspaceSize);

// autotune() compiles + benchmarks all plan specs
std::vector<AutotuneResult> results;
graph.autotune(handle,
    variant_pack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::EXHAUSTIVE,
     .warmupIterations = 1,
     .timedIterations = 10},
    {}, &results);
printf("Winner: %s (%.4f ms)\n",
    results[0].engineName.c_str(),
    results[0].minTimeMs);
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//
graph.execute(handle, variant_pack, workspace);
</pre></td>
</tr>
</table>

#### cuDNN Manual Benchmarking Loop vs. hipDNN Manual Benchmarking Loop

For direct porting, hipDNN provides the same plan-indexed APIs as cuDNN. The loop structure is nearly identical, with `cuda*` calls replaced by `hip*` equivalents.

<table>
<tr><th>cuDNN (manual benchmarking loop)</th><th>hipDNN (manual benchmarking loop)</th></tr>
<tr>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
graph.create_execution_plans({HeurMode_t::A});
graph.check_support();
graph.build_plans(BuildPlanPolicy_t::ALL);

auto plan_count =
    graph.get_execution_plan_count();

// Find max workspace across all plans
int64_t ws = 0;
for (int i = 0; i < plan_count; i++)
    ws = std::max(ws,
        graph.get_workspace_size_plan_at_index(i));
void* workspace;
cudaMalloc(&workspace, ws);

// Benchmark each plan
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);
std::vector<float> times(plan_count, 10.0f);

for (int i = 0; i < plan_count; i++) {
    // Warmup
    auto err = graph.execute_plan_at_index(
        handle, variant_pack, workspace, i);
    if (err.is_bad()) continue;
    cudaDeviceSynchronize();

    // Timed iterations
    cudaEventRecord(start, stream);
    for (int iter = 0; iter < 10; iter++)
        graph.execute_plan_at_index(
            handle, variant_pack, workspace, i);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    times[i] = ms / 10.0f;
}

// Select winner
auto best = std::min_element(
    times.begin(), times.end());
auto idx = std::distance(times.begin(), best);

graph.build_plan_at_index(idx);
graph.execute(handle, variant_pack, workspace);
</pre></td>
<td><pre lang="cpp">
graph.validate();
graph.build_operation_graph(handle);
graph.create_execution_plans({HeuristicMode::FALLBACK});
//
graph.build_plans(BuildPlanPolicy::ALL);

auto plan_count =
    graph.get_execution_plan_count();

// Find max workspace across all plans
int64_t ws = 0;
for (int i = 0; i < plan_count; i++)
    ws = std::max(ws,
        graph.get_workspace_size_plan_at_index(i));
void* workspace;
hipMalloc(&workspace, ws);

// Benchmark each plan
hipEvent_t start, stop;
hipEventCreate(&start);
hipEventCreate(&stop);
std::vector<float> times(plan_count, 10.0f);

for (int i = 0; i < plan_count; i++) {
    // Warmup
    auto err = graph.execute_plan_at_index(
        handle, variant_pack, workspace, i);
    if (err.is_bad()) continue;
    hipDeviceSynchronize();

    // Timed iterations
    hipEventRecord(start, stream);
    for (int iter = 0; iter < 10; iter++)
        graph.execute_plan_at_index(
            handle, variant_pack, workspace, i);
    hipEventRecord(stop, stream);
    hipEventSynchronize(stop);

    float ms;
    hipEventElapsedTime(&ms, start, stop);
    times[i] = ms / 10.0f;
}

// Select winner
auto best = std::min_element(
    times.begin(), times.end());
auto idx = std::distance(times.begin(), best);

graph.build_plan_at_index(idx);
graph.execute(handle, variant_pack, workspace);
</pre></td>
</tr>
</table>

### A.2 Step-by-Step API Mapping

Status tags: **existing** = already in hipDNN, unchanged; **RFC** = new, proposed by this RFC.

| cuDNN | hipDNN | Source | Notes |
|-------|--------|--------|-------|
| `graph.validate()` | `graph.validate()` | **existing** | Identical |
| `graph.build_operation_graph(handle)` | `graph.build_operation_graph(handle)` | **existing** | Identical |
| `graph.create_execution_plans({HeurMode_t::A})` | `graph.get_engine_configs(configs)` (plan-spec) or `graph.create_execution_plans({HeuristicMode::FALLBACK})` (compiled-plan) | **RFC** / **existing** | Plan-spec: exposes engine list for inspection. Compiled-plan: as in cuDNN |
| `graph.check_support()` | _(inside `autotune()`)_ (plan-spec) or `graph.check_support()` (compiled-plan) | **RFC** / **existing** | Plan-spec: unsupported engines filtered at compilation. Compiled-plan: as in cuDNN |
| `graph.build_plans(BuildPlanPolicy_t::ALL)` | `graph.add_engine_configs(configs)` (plan-spec) or `graph.build_plans(BuildPlanPolicy_t::ALL)` (compiled-plan) | **RFC** / **existing** | Plan-spec: stores without compiling (`add_all_engines()` as shortcut). Compiled-plan: compiles all, `autotune()` benchmarks directly |
| `graph.deselect_workspace_greater_than(n)` | `graph.deselect_workspace_greater_than(n)` | **RFC** | Filters by workspace size. Get-filter-add recommended on plan-spec path |
| `graph.deselect_engines(barred)` | `graph.deselect_engines(barred)` | **RFC** | Filters by engine name or ID |
| `graph.get_autotune_workspace_size()` | `graph.get_estimated_max_workspace_size(maxSize)` (plan-spec) or `graph.get_autotune_workspace_size()` (compiled-plan) | **RFC** / **RFC** | Plan-spec: pre-compile estimate (not guaranteed accurate). Compiled-plan: queries pre-compiled plans |
| `graph.autotune(handle, variant_pack, workspace)` | `graph.autotune(handle, variant_pack, workspace, ...)` | **RFC** | Three tiers (see § 6.3) |
| `graph.execute(handle, variant_pack, workspace)` | `graph.execute(handle, variant_pack, workspace)` | **existing** | Identical |
| `graph.get_execution_plan_count()` | `graph.get_execution_plan_count()` | **RFC** | Number of compiled plans; cuDNN manual-loop compatibility |
| `graph.execute_plan_at_index(handle, variant_pack, workspace, i)` | `graph.execute_plan_at_index(handle, variantPack, workspace, index)` | **RFC** | Execute specific compiled plan by index |
| `graph.build_plan_at_index(idx)` | `graph.build_plan_at_index(index)` | **RFC** | Set active plan to given index |
| `graph.get_workspace_size_plan_at_index(i)` (two overloads) | `graph.get_workspace_size_plan_at_index(index, size)` (error-code) and `graph.get_workspace_size_plan_at_index(index)` (direct-return, -1 on invalid) | **RFC** | Workspace size for a compiled plan by index |
| _(no cuDNN equivalent)_ | `graph.get_plan_name_at_index(index, name)` | **RFC** | Plan name for a compiled plan by index |

> Both paths are mutually exclusive on the same Graph instance (see § 6.3).

### A.3 Key Differences and Similarities

**1. Tensor pointer mapping (identical)**

Both use `std::unordered_map<int64_t, void*>` directly (tensor UID -> device pointer) with no wrapper type:

```cpp
// cuDNN and hipDNN - same type, same usage:
std::unordered_map<int64_t, void*> variant_pack = {
    {x_uid, x_ptr}, {w_uid, w_ptr}, {y_uid, y_ptr}};
```

**2. `autotune()` return semantics**

Both select the winner and mutate graph state for `execute()`. cuDNN returns `error_t` with no result access; hipDNN returns `Error` with an optional `results` pointer:

```cpp
// cuDNN - no access to results:
graph.autotune(handle, variant_pack, workspace);

// hipDNN - results pointer omitted (compiled-plan overload):
auto err = graph.autotune(handle, variant_pack, workspace, config);

// hipDNN - with access to all results (compiled-plan overload):
std::vector<AutotuneResult> results;
auto err = graph.autotune(handle, variant_pack, workspace, config, storageConfig, &results);
for (const auto& r : results)
    printf("#%d %-30s min=%.4f ms\n", r.rank, r.engineName.c_str(), r.minTimeMs);
// Winner is results[0]; graph is configured so execute() uses that plan.
```

**3. Engine discovery is collapsed in cuDNN but separated in hipDNN**

cuDNN uses `create_execution_plans()` + `check_support()` + `build_plans(ALL)` as opaque steps. hipDNN separates discovery from compilation via the plan-spec path (see § 6.2.1), while also supporting a compiled-plan path that mirrors cuDNN directly. The two paths are mutually exclusive per Graph instance (see § 6.3).

```cpp
// cuDNN (3 calls, no inspection):
graph.create_execution_plans({HeurMode_t::A});
graph.check_support();
graph.build_plans(BuildPlanPolicy_t::ALL);

// hipDNN - compiled-plan path (cuDNN drop-in):
graph.create_execution_plans({HeuristicMode::FALLBACK});
graph.check_support();
graph.build_plans(BuildPlanPolicy_t::ALL);

// hipDNN - plan-spec path, simple (1 call, no inspection):
graph.add_all_engines();

// hipDNN - plan-spec path, with inspection (2 calls):
std::vector<EngineConfigInfo> configs;
graph.get_engine_configs(configs);
// ... inspect, filter, reorder ...
graph.add_engine_configs(configs);
```

**4. Benchmarking parameters are configurable**

cuDNN hardcodes a convergence-based algorithm. hipDNN's defaults match cuDNN's behavior but exposes all parameters via `AutotuneConfig` (see § 6.1):

```cpp
// cuDNN - hardcoded convergence-based parameters:
graph.autotune(handle, variant_pack, workspace);

// hipDNN - equivalent default behavior (cuDNN-compatible overload):
graph.autotune(handle, variant_pack, workspace);

// hipDNN - compiled-plan path with explicit configuration overrides (compiled-plan overload):
graph.autotune(handle, variant_pack, workspace,
    {.strategy = AutotuneStrategy::RUN_UNTIL_STABLE,
     .maxIterations = 100,
     .stabilityThreshold = 0.02f});

// hipDNN - plan-spec path with EXHAUSTIVE mode (general overload):
graph.autotune(handle, variant_pack, workspace, workspaceSize,
    {.mode = TuneMode::EXHAUSTIVE,
     .stabilityThreshold = 0.02f});
```

**5. Plan filtering: get-filter-add pattern and dedicated filter methods**

hipDNN provides both approaches:
- **Get-filter-add** (plan-spec path, recommended): `get_engine_configs()` -> filter -> `add_engine_configs()`. See "With Engine Filtering" in § A.1.
- **Dedicated methods** (see § 6.2.7): `deselect_workspace_greater_than()` and `deselect_engines()` for cuDNN API parity, required on the compiled-plan path.

> **Caveat (compiled-plan path)**: hipDNN's `deselect_*` methods remove entries, causing indices to shift (cuDNN marks them as barred without removing). Re-query `get_execution_plan_count()` after deselection.

**6. hipDNN extended autotune features**

| hipDNN Feature | cuDNN Equivalent |
|----------------|------------------|
| `AutotuneResult` vector with per-engine timing data | None (cuDNN autotune is opaque) |
| `TuneMode::EXHAUSTIVE` (benchmarking knob priming; plan-spec path only) | None (cuDNN has no equivalent knob) |
| `AutotuneStrategy` (SINGLE_SHOT, FIXED_AVERAGE, RUN_UNTIL_STABLE) | Hardcoded convergence strategy |
| `AutotuneConfig` (warmup, iterations, priming failure control) | Hardcoded defaults |
| Config file output (`AutotuneStorageConfig`) | None |
| Knob variant autotuning (`EngineVariant`, `EngineSweepSpec`, `add_engine_*()`) | Limited `create_execution_plan(id, knob_map)` |
| Custom ranking function (`AutotuneRankingFn`) | None |

---

## Appendix B: Use Cases

All examples assume this setup boilerplate:
```cpp
hipdnnHandle_t handle;
hipdnnCreate(&handle);

auto graph = std::make_shared<hipdnn_frontend::Graph>();
// ... configure graph operations, tensors, etc. ...
graph->validate();
graph->build_operation_graph(handle);
```

### Core Autotuning

**1. Quick autotune (AUTO or EXHAUSTIVE).** Use `EXHAUSTIVE` to prime engine-internal caches before timing (more accurate for lazy-compilation engines).

```cpp
graph->add_all_engines();
int64_t maxWorkspaceSize;
graph->get_estimated_max_workspace_size(maxWorkspaceSize);
void* workspace = allocate(maxWorkspaceSize);

graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::AUTO});       // or TuneMode::EXHAUSTIVE
graph->execute(handle, variantPack, workspace);
```

**2. Inspect autotune results.** Ranked results for logging or programmatic decisions.

```cpp
graph->add_all_engines();
int64_t maxWorkspaceSize;
graph->get_estimated_max_workspace_size(maxWorkspaceSize);
void* workspace = allocate(maxWorkspaceSize);

std::vector<AutotuneResult> results;
graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::EXHAUSTIVE},
    {}, &results);

for (const auto& r : results) {
    printf("#%d %-30s min=%.4f ms  avg=%.4f ms  ws=%lld %s\n",
           r.rank, r.engineName.c_str(), r.minTimeMs, r.avgTimeMs,
           r.workspaceSize, r.succeeded ? "OK" : r.errorMessage.c_str());
}
```

**3. AutotuneConfig options.** All assume `add_all_engines()` and workspace allocation (as in use case 1) are done.

```cpp
// Custom warmup/iteration counts
graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::EXHAUSTIVE, .warmupIterations = 10, .timedIterations = 50});

// Workspace-constrained: compile all, skip execution of plans whose actual workspace exceeds 64 MB
int64_t wsLimit = 64 << 20;
graph->autotune(handle, variantPack, workspace, wsLimit,
    {.mode = TuneMode::AUTO});

// Autotune specific engines only
graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::EXHAUSTIVE, .engineIdFilter = {42, 17}});

// Combined: mode, warmup, iterations, and output file
graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::EXHAUSTIVE, .warmupIterations = 5, .timedIterations = 20},
    {.filePath = "autotune_results.json"});
```

**4. Custom stream.** Set the stream on the handle before calling `autotune()`.

```cpp
hipStream_t myStream;
hipStreamCreate(&myStream);
hipdnnSetStream(handle, myStream);

graph->autotune(handle, variantPack, workspace, maxWorkspaceSize, {.mode = TuneMode::AUTO});
```

### Engine Discovery and Filtering

**5. Inspect engines before adding plan specs.**

```cpp
std::vector<EngineConfigInfo> configs;
graph->get_engine_configs(configs);

printf("Found %zu engines:\n", configs.size());
for (const auto& c : configs)
    printf("  [%lld] %s  exhaustive=%s  knobs=%zu\n",
           c.engineId, c.engineName.c_str(),
           c.supportsExhaustive ? "yes" : "no", c.knobs.size());

graph->add_engine_configs(configs);
```

**6. Exclude specific engines by name.**

```cpp
std::vector<EngineConfigInfo> configs;
graph->get_engine_configs(configs);

configs.erase(std::remove_if(configs.begin(), configs.end(),
    [](const auto& c) {
        return c.engineName.find("DETERMINISTIC") != std::string::npos;
    }),
    configs.end());

graph->add_engine_configs(configs);
```

**7. Benchmark only exhaustive-capable engines.**

```cpp
std::vector<EngineConfigInfo> configs;
graph->get_engine_configs(configs);

configs.erase(std::remove_if(configs.begin(), configs.end(),
    [](const auto& c) { return !c.supportsExhaustive; }),
    configs.end());

graph->add_engine_configs(configs);
int64_t maxWorkspaceSize;
graph->get_estimated_max_workspace_size(maxWorkspaceSize);
void* workspace = allocate(maxWorkspaceSize);
graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::EXHAUSTIVE});
```

**8. Add a single engine for targeted autotuning or debugging.**

```cpp
graph->add_engine(42, {{"global.workspace_size_limit", int64_t{16 << 20}}});
int64_t maxWorkspaceSize;
graph->get_estimated_max_workspace_size(maxWorkspaceSize);
void* workspace = allocate(maxWorkspaceSize);
graph->autotune(handle, variantPack, workspace, maxWorkspaceSize);
graph->execute(handle, variantPack, workspace);
```

### Benchmarking Strategies

**9. Single-shot benchmarking.** One timed iteration per plan; fast, rough ranking.

```cpp
graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::AUTO,
     .strategy = AutotuneStrategy::SINGLE_SHOT});
```

**10. Convergence-based benchmarking.** Run until timing stabilizes.

```cpp
graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::EXHAUSTIVE,
     .strategy = AutotuneStrategy::RUN_UNTIL_STABLE,
     .maxIterations = 100,
     .windowSize = 5,
     .stabilityThreshold = 0.02f});
```

### Knob Variant Autotuning

**11. Benchmark an engine with different knob settings.**

```cpp
std::vector<EngineConfigInfo> configs;
graph->get_engine_configs(configs);

std::vector<EngineVariant> variants;
for (const auto& config : configs) {
    if (config.engineName.find("MIOPEN") != std::string::npos) {
        for (int64_t ws : {int64_t{0}, int64_t{1 << 20}, int64_t{16 << 20}, int64_t{256 << 20}}) {
            variants.push_back({
                .engineId = config.engineId,
                .knobSettings = {{"global.workspace_size_limit", ws}}
            });
        }
    } else {
        variants.push_back({.engineId = config.engineId, .knobSettings = {}});
    }
}

graph->add_engine_variants(variants);
int64_t maxWorkspaceSize;
graph->get_estimated_max_workspace_size(maxWorkspaceSize);
void* workspace = allocate(maxWorkspaceSize);
std::vector<AutotuneResult> results;
graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::EXHAUSTIVE},
    {}, &results);

for (const auto& r : results)
    printf("#%d %-25s min=%.4f ms\n",
           r.rank, r.engineName.c_str(), r.minTimeMs);
```

**12. Automated knob sweep.** Framework generates the Cartesian product from specified axes.

```cpp
std::vector<EngineConfigInfo> configs;
graph->get_engine_configs(configs);

auto miopenId = configs[0].engineId;  // assuming first is MIOpen

EngineSweepSpec spec;
spec.engineId = miopenId;
spec.axes = {
    {"global.workspace_size_limit",
     {int64_t{0}, int64_t{1 << 20}, int64_t{16 << 20}, int64_t{256 << 20}}}
};
// Generates 4 variants with workspace_size_limit values: 0, 1M, 16M, 256M

graph->add_engine_sweep({spec});
int64_t maxWorkspaceSize;
graph->get_estimated_max_workspace_size(maxWorkspaceSize);
void* workspace = allocate(maxWorkspaceSize);
graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::EXHAUSTIVE});
```

### Custom Ranking

**13. Custom ranking.** Rank by workspace size (smallest first), breaking ties by speed.

```cpp
graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::EXHAUSTIVE,
     .rankingFn = [](std::vector<AutotuneResult>& results) {
         std::sort(results.begin(), results.end(), [](const auto& a, const auto& b) {
             if (a.workspaceSize != b.workspaceSize)
                 return a.workspaceSize < b.workspaceSize;
             return a.minTimeMs < b.minTimeMs;
         });
     }});
```

### Config File Output and Reuse

**14. Save results to config file.** Reusable via `HIPDNN_HEUR_CONFIG_PATH`.

```cpp
graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::EXHAUSTIVE},
    {.filePath = "autotune_results.json"});
```

**15. Overwrite config file.**

```cpp
graph->autotune(handle, variantPack, workspace, maxWorkspaceSize,
    {.mode = TuneMode::EXHAUSTIVE},
    {.filePath = "my_engine_overrides.json",
     .deleteAllExistingFileContent = true});
```

**16. Reuse autotuned results.**

```bash
# First run: autotune and save
./my_app  # writes autotune_results.json

# Subsequent runs: skip autotuning, use saved results
export HIPDNN_HEUR_CONFIG_PATH=autotune_results.json
./my_app  # SelectionHeuristic::Config policy picks the saved winner
```

**17. Build a library of autotuned configurations.** Accumulate results across runs into one config file.

```bash
# Run 1: autotune convolution graphs
./tune_conv_graphs --output my_overrides.json

# Run 2: autotune matmul graphs, results are appended by default
./tune_matmul_graphs --output my_overrides.json

# Production: use the combined file
export HIPDNN_HEUR_CONFIG_PATH=my_overrides.json
./my_training_app
```
