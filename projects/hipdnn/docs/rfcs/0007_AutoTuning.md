# RFC 0007: Auto-Tuning

**Contributors:** Sam Reeder
**Status:** Draft
**Created:** 2026-02-26

---

## Table of Contents

- [Executive Summary](#executive-summary)
- [Problem Statement](#problem-statement)
- [User Stories](#user-stories)
- [Design Goals](#design-goals)
- [cuDNN API Review](#cudnn-api-review)
- [Proposed Design](#proposed-design)
  - [Tuning Modes](#tuning-modes)
  - [API Surface](#api-surface)
  - [Benchmarking Flow](#benchmarking-flow)
  - [Result Representation](#result-representation)
  - [Config File Output](#config-file-output)
  - [Caching Architecture](#caching-architecture)
  - [Extensibility](#extensibility)
- [Phased Execution Plan](#phased-execution-plan)
- [cuDNN API Parity and Divergence](#cudnn-api-parity-and-divergence)
- [Risks](#risks)
- [Testing Plan](#testing-plan)
- [Future Considerations](#future-considerations)
- [Glossary](#glossary)

---

## Executive Summary

This RFC proposes an auto-tuning feature for hipDNN that benchmarks available engines for a given operation and returns ranked results by execution time. The feature is designed in three phases: (1) run-and-rank with immediate results, (2) config file persistence for result reuse, and (3) user-level and system-level caching. Two tuning modes are defined: a lightweight auto-tune mode that compares engines without the benchmarking knob, and an exhaustive-tune mode that enables the benchmarking knob for deeper sampling across engine configurations. The design builds on the existing engine override config system (RFC 0004), engine ID registry (RFC 0003), and the `global.benchmarking` knob infrastructure.

---

## Problem Statement

hipDNN currently supports manual engine selection via environment variables (`HIPDNN_DEFAULT_ENGINE`, `HIPDNN_ENGINE_OVERRIDE_FILE`) and programmatic preferred engine IDs. However:

1. **No automated comparison.** Users must manually try each engine and measure performance themselves to determine which is fastest for their workload.
2. **No persistence of tuning results.** Even if a user discovers the optimal engine, there is no built-in way to record and replay that selection.
3. **The benchmarking knob exists but has no coordinated workflow.** The `global.benchmarking` knob unlocks additional plan variants within engines, but there is no facility to systematically evaluate them and persist winners.
4. **cuDNN provides auto-tuning capabilities** (heuristic modes, `BuildPlanPolicy_t::ALL`, `time_sorted_plan`, `ExecutionPlanCache`). hipDNN needs API parity and should exceed cuDNN's capabilities where cuDNN falls short — particularly around persistent storage, multi-level caching, and fine-grained tuning control.

---

## User Stories

1. **As a framework developer**, I want to run a single API call that benchmarks all applicable engines for my operation graph and returns ranked results, so I can select the fastest engine without manual trial-and-error.

2. **As a framework developer**, I want to run an exhaustive tuning pass that explores engine configurations (knob permutations) via the benchmarking flag, so I can find the absolute best configuration for latency-sensitive workloads.

3. **As a deployment engineer**, I want to export tuning results to a config file compatible with `HIPDNN_ENGINE_OVERRIDE_FILE`, so I can apply tuned selections across inference runs without re-tuning.

4. **As a deployment engineer**, I want to provide an existing config file and have new tuning results appended to it, so I can incrementally build a tuning database across different operations.

5. **As a system administrator**, I want tuning results cached at the user level or system level, so all applications on a machine benefit from tuning performed once.

6. **As a plugin author**, I want to hook into the tuning framework with custom benchmarking logic or scoring criteria, so I can extend tuning without reimplementing the orchestration.

---

## Design Goals

1. **Phased delivery.** Each phase is independently useful and shippable.
2. **Two tuning modes.** Auto-tune (engine-vs-engine, no benchmarking knob) and exhaustive-tune (benchmarking knob enabled, broader plan space).
3. **cuDNN API parity.** Match cuDNN's `create_execution_plans` + `build_plans(ALL)` + benchmark loop pattern, while improving on its limitations.
4. **Config file interop.** Output is compatible with the existing engine override config JSON format.
5. **Generic and extensible.** Users can customize scoring, filtering, and storage without reimplementing the tuning loop.
6. **Deterministic benchmarking.** Warm-up, multi-iteration timing with event synchronization, and configurable iteration counts.

---

## cuDNN API Review

cuDNN frontend provides the following auto-tuning capabilities:

### Heuristic Modes
```cpp
enum class HeurMode_t { A, B, FALLBACK };
```
- **Mode A:** Fast heuristic, returns engines ordered by predicted performance.
- **Mode B:** Slower, more comprehensive heuristic (mutex-protected).
- **FALLBACK:** Fallback engine list when other modes unavailable.

### Build Policies
```cpp
enum class BuildPlanPolicy_t { HEURISTICS_CHOICE, ALL };
```
- `HEURISTICS_CHOICE` — build only the top-ranked plan.
- `ALL` — build every candidate plan for benchmarking.

### Benchmarking
```cpp
enum class CudnnFindSamplingTechnique {
    CUDNN_FIND_SAMPLE_ONCE,
    CUDNN_FIND_SAMPLE_MEDIAN_OF_THREE,
    CUDNN_FIND_SAMPLE_TILL_STABLE,  // 100 iters, 0.95 stability threshold
};
```
Workflow: warm-up run → CUDA event record → N iterations → event synchronize → elapsed time.

### Caching
```cpp
class ExecutionPlanCache_v2 {
    // Maps feature vectors (operation graph fingerprints) to best plans
    // Device-aware, thread-safe, saturation tracking
};
```

### Key Graph API
```cpp
graph.create_execution_plans({HeurMode_t::A});
graph.build_plans(BuildPlanPolicy_t::ALL);
int64_t count = graph.get_execution_plan_count();
graph.execute_plan_at_index(handle, variant_pack, workspace, i);
graph.get_plan_name_at_index(i, name);
```

### Limitations of cuDNN's Approach
1. **No persistent storage.** `ExecutionPlanCache` is in-memory only; no file export or system-level cache.
2. **No incremental config building.** Cannot append results to an existing config file.
3. **Rigid sampling techniques.** Three fixed strategies, no user-defined iteration counts or scoring functions.
4. **No separation between lightweight and exhaustive modes.** User must manually manage knob exploration.
5. **Cache is per-process.** No mechanism to share results across processes or users.

---

## Proposed Design

### Tuning Modes

Two modes are provided, distinguished by whether the benchmarking knob is active:

| Mode | Benchmarking Knob | Scope | Use Case |
|------|-------------------|-------|----------|
| **Auto-Tune** | Off (`global.benchmarking = 0`) | Compares default plan from each applicable engine | Quick engine selection for typical workloads |
| **Exhaustive-Tune** | On (`global.benchmarking = 1`) | Compares all plan variants across all applicable engines | Maximum performance extraction for latency-critical paths |

Both modes use the same benchmarking flow (warm-up → timed iterations → rank by execution time) but differ in the plan space they explore.

### API Surface

#### Core Tuning API (Frontend `Graph` extensions)

```cpp
/// Tuning mode selector
enum class TuneMode {
    AUTO,        // Engine-vs-engine, default plans only
    EXHAUSTIVE,  // All plan variants via benchmarking knob
};

/// Controls where/how tuning results are stored (Phase 2+)
enum class TuneStorage {
    NONE,        // Return results only, no persistence
    FILE,        // Write to config file
    USER_CACHE,  // Store in user-level cache (Phase 3)
    SYSTEM_CACHE // Store in system-level cache (Phase 3)
};

/// Tuning configuration
struct TuneConfig {
    TuneMode mode            = TuneMode::AUTO;
    int32_t  warmup_runs     = 1;
    int32_t  benchmark_runs  = 10;
    int64_t  max_workspace   = -1;  // -1 = no limit
    TuneStorage storage      = TuneStorage::NONE;
    std::string config_file_path;   // For TuneStorage::FILE
    bool append_to_file      = false;
};

/// Single tuning result entry
struct TuneResult {
    int64_t      engine_id;
    std::string  engine_name;
    float        execution_time_ms;   // Average over benchmark_runs
    int64_t      workspace_size;
    std::vector<KnobSetting> knob_settings;  // Knobs used (relevant for exhaustive)
};

/// Execute tuning and return ranked results
Error tune(
    hipdnnHandle_t handle,
    const TuneConfig& config,
    std::vector<TuneResult>& ranked_results
);
```

#### Config File API (Phase 2)

```cpp
/// Write tuning results to engine override config file
/// If append=true and file exists, merge results (new results take precedence
/// for matching operation+tensor patterns)
static Error write_tune_results_to_config(
    const std::string& filepath,
    const std::string& operation_name,
    const std::vector<std::shared_ptr<graph::TensorAttributes>>& tensors,
    const std::vector<TuneResult>& ranked_results,
    bool append = false
);

/// Load and apply a tuning config file (wraps existing EngineOverrideConfig)
static Error load_tune_config(const std::string& filepath);
```

#### Cache API (Phase 3)

```cpp
/// Query the tuning cache for a previously-tuned operation
Error query_tune_cache(
    TuneStorage cache_level,
    std::vector<TuneResult>& cached_results
) const;

/// Clear tuning cache entries
static Error clear_tune_cache(TuneStorage cache_level);
```

### Benchmarking Flow

Both tuning modes follow the same measurement protocol:

```
for each candidate plan:
    1. Allocate workspace (skip if exceeds max_workspace)
    2. Execute warmup_runs iterations (results discarded)
    3. Record HIP event (start)
    4. Execute benchmark_runs iterations
    5. Record HIP event (stop)
    6. Synchronize on stop event
    7. Compute average_time = elapsed / benchmark_runs
    8. Record TuneResult

Sort results by average_time ascending
Return ranked list
```

**Auto-Tune mode** iterates over `get_ranked_engine_ids()` and builds the default plan for each engine.

**Exhaustive-Tune mode** sets `global.benchmarking = 1` before plan creation, which causes engines to expose all available plan variants. Each variant is benchmarked independently.

HIP events (`hipEventCreate`, `hipEventRecord`, `hipEventSynchronize`, `hipEventElapsedTime`) are used for GPU-accurate timing, matching cuDNN's approach.

### Result Representation

`TuneResult` captures everything needed to reproduce the winning configuration:

- **engine_id / engine_name** — which engine won
- **execution_time_ms** — measured performance
- **workspace_size** — memory cost
- **knob_settings** — the exact knob configuration (important for exhaustive mode where knob permutations are explored)

Results are returned as a vector sorted by execution time (fastest first). The caller decides what to do with them — use the winner directly, inspect trade-offs, or persist.

### Config File Output

Phase 2 writes results to the existing engine override config JSON format:

```json
{
  "engine_overrides": [
    {
      "op": "conv_fprop",
      "engine_name": "MIOPEN_ENGINE",
      "tensors": [
        { "dim": [1, 3, 224, 224], "stride": [150528, 50176, 224, 1] },
        { "dim": [64, 3, 7, 7] }
      ]
    }
  ]
}
```

When `append_to_file = true`:
1. Load existing config file.
2. Match incoming results against existing rules by operation + tensor pattern.
3. If a match exists, update the engine_name with the new winner.
4. If no match, append a new rule.
5. Write the merged config back.

This allows incremental tuning — tune convolutions today, matmuls tomorrow, and accumulate into a single config file.

### Caching Architecture

Phase 3 introduces two cache levels:

| Level | Location | Scope | Use Case |
|-------|----------|-------|----------|
| **User Cache** | `$XDG_CACHE_HOME/hipdnn/tune/` or `~/.cache/hipdnn/tune/` | Per-user, all processes | Developer workstation, shared across runs |
| **System Cache** | `/etc/hipdnn/tune/` or `$HIPDNN_SYSTEM_CACHE_DIR` | All users on machine | Inference servers, cluster-wide deployment |

Cache keys are derived from:
- Operation graph fingerprint (operation type + tensor dimensions + strides + data types)
- GPU architecture (e.g., `gfx942`)
- hipDNN version
- Plugin versions

Cache invalidation occurs when any component of the key changes.

### Extensibility

The tuning framework is designed for user extension at several points:

1. **Custom scoring.** Users receive the full `TuneResult` vector and can re-rank by their own criteria (e.g., workspace-weighted score, power efficiency).
2. **Custom benchmarking.** `TuneConfig` controls iteration counts. Future extension point: user-provided benchmark callback.
3. **Custom filtering.** Users can pre-filter engines via `deselect_engines()` or workspace limits before tuning.
4. **Custom storage.** `TuneStorage::NONE` returns raw results; users can implement their own persistence.

---

## Phased Execution Plan

### Phase 1: Run-and-Rank

**Scope:** `tune()` API with `TuneMode::AUTO` and `TuneMode::EXHAUSTIVE`. Returns ranked `TuneResult` vector. No persistence.

**Deliverables:**
- `TuneMode` enum, `TuneConfig` struct, `TuneResult` struct
- `Graph::tune()` implementation with HIP event-based benchmarking
- Warm-up + multi-iteration measurement loop
- Unit tests for both tuning modes
- Sample application demonstrating tuning workflow

**Dependencies:** Existing `get_ranked_engine_ids()`, `create_execution_plan_ext()`, `global.benchmarking` knob.

### Phase 2: Config File Persistence

**Scope:** `TuneStorage::FILE` support. Write/append results to engine override config JSON.

**Deliverables:**
- `write_tune_results_to_config()` static method
- Merge logic for append mode (match by op + tensor pattern)
- `load_tune_config()` convenience wrapper
- Integration tests: tune → write → reload → verify engine selection
- Documentation for config file workflow

**Dependencies:** Phase 1, existing `EngineOverrideConfig` parser.

### Phase 3: Multi-Level Caching

**Scope:** User-level and system-level caches. Cache lookup integrated into engine selection path.

**Deliverables:**
- Cache key derivation (graph fingerprint + GPU arch + versions)
- Cache storage format (JSON or FlatBuffer)
- User cache at `~/.cache/hipdnn/tune/`
- System cache at `/etc/hipdnn/tune/` (configurable)
- Cache invalidation logic
- `query_tune_cache()` and `clear_tune_cache()` APIs
- Automatic cache consultation during `get_ranked_engine_ids()` (opt-in)
- Cache management CLI or utility

**Dependencies:** Phase 2, version fingerprinting.

---

## cuDNN API Parity and Divergence

| Capability | cuDNN | hipDNN (Proposed) | Notes |
|------------|-------|-------------------|-------|
| Heuristic engine ranking | `create_execution_plans({HeurMode_t::A})` | `get_ranked_engine_ids()` (existing) | Parity |
| Build all plans | `build_plans(BuildPlanPolicy_t::ALL)` | `tune()` with `TuneMode::EXHAUSTIVE` | Extended — hipDNN ties build-all to benchmarking knob |
| Timed benchmarking | `time_sorted_plan<TILL_STABLE>()` | `tune()` with configurable iterations | Extended — user controls warmup and iteration count |
| In-memory cache | `ExecutionPlanCache_v2` | Phase 3 caching | Extended — hipDNN adds file and system-level persistence |
| File persistence | Not supported | Phase 2 config file output | hipDNN-only |
| Workspace filtering | `deselect_workspace_greater_than()` | `TuneConfig::max_workspace` | Parity |
| Plan name query | `get_plan_name_at_index()` | `TuneResult::engine_name` | Parity |
| Sampling techniques | 3 fixed (`ONCE`, `MEDIAN_OF_THREE`, `TILL_STABLE`) | User-configurable `warmup_runs` + `benchmark_runs` | Extended |
| Auto vs exhaustive mode | Not distinguished | `TuneMode::AUTO` vs `TuneMode::EXHAUSTIVE` | hipDNN-only |
| System-level cache | Not supported | Phase 3 system cache | hipDNN-only |

---

## Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Exhaustive tuning is slow for operations with many plan variants | Users may not complete tuning | Document expected times; support workspace filtering and engine deselection to reduce search space |
| Cache key collisions (different operations hash to same fingerprint) | Wrong engine selected | Use comprehensive fingerprints including full tensor metadata; validate on cache hit |
| Config file format changes break existing override files | User configs stop working | Version the config format; maintain backward compatibility or migration tooling |
| Benchmarking results vary between runs due to GPU thermal state or contention | Inconsistent tuning outcomes | Multiple iterations with warm-up; document best practices (idle GPU, consistent clocks) |
| System cache permissions and security | Untrusted cache entries | System cache is read-only for non-root; validate cache entries against expected schema |

---

## Testing Plan

### Phase 1
- **Unit tests:** `TuneConfig` defaults, `TuneResult` sorting, benchmarking loop with mock engines.
- **Integration tests:** End-to-end `tune()` on a real convolution with MIOpen plugin; verify results are ordered by execution time; verify both AUTO and EXHAUSTIVE modes produce valid results.
- **Regression tests:** Existing engine selection tests continue to pass.

### Phase 2
- **Unit tests:** Config file writing (new file, append, merge on conflict).
- **Integration tests:** Tune → write config → set `HIPDNN_ENGINE_OVERRIDE_FILE` → verify selected engine matches tuning winner.
- **Edge cases:** Empty config file, malformed JSON, read-only filesystem.

### Phase 3
- **Unit tests:** Cache key generation, cache hit/miss, invalidation on version change.
- **Integration tests:** Tune → cache → new process → cache hit → verify engine selection.
- **Concurrency tests:** Multiple processes reading/writing cache simultaneously.

---

## Future Considerations

- **Distributed tuning.** Aggregate tuning results from a fleet of machines into a shared cache (e.g., Redis, shared filesystem).
- **Online tuning.** Continuously tune during production inference, gradually converging on optimal engines.
- **Cost-aware tuning.** Factor in workspace memory as a first-class dimension alongside execution time.
- **Profile-guided tuning.** Use actual workload traces to weight tuning toward frequently-executed operations.
- **User-defined benchmark callbacks.** Allow users to provide custom benchmarking functions for specialized measurement requirements.

---

## Glossary

| Term | Definition |
|------|------------|
| **Auto-Tune** | Tuning mode that compares default plans from each applicable engine without enabling the benchmarking knob |
| **Exhaustive-Tune** | Tuning mode that enables the `global.benchmarking` knob, expanding the plan space to include all engine configuration variants |
| **Engine** | A backend implementation (e.g., MIOpen, hipBLASLt) that can execute an operation graph |
| **Plan** | A fully-configured, ready-to-execute instance of an engine for a specific operation graph |
| **Knob** | A tunable parameter exposed by an engine that affects plan behavior or performance |
| **Engine Override Config** | JSON file specifying which engine to use for specific operation + tensor patterns |
| **Feature Vector** | A fingerprint derived from an operation graph's metadata, used as a cache key |
| **TuneResult** | A data structure capturing the measured performance of a single engine/plan configuration |

