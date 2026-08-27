# hipDNN Environment Configuration

This document describes the environment variables and runtime configuration options for hipDNN.

## Table of Contents

- [Environment Variables](#environment-variables)
  - [Plugin Discovery](#plugin-discovery)
  - [Heuristic Policy Selection](#heuristic-policy-selection)
  - [Benchmarking](#benchmarking)
  - [Caching](#caching)
  - [Autotune Ranking Cache](#autotune-ranking-cache-autotune-rankings)
  - [Logging Variables](#logging-variables)
  - [MIOpen Plugin Logging](#miopen-plugin-logging)
  - [Test Configuration](#test-configuration)
- [Logging Configuration APIs](#logging-configuration-apis)
  - [User Log Callbacks](#user-log-callbacks)
  - [Log Level APIs](#log-level-apis)
- [Error Handling](#error-handling)

---

## Environment Variables

### Plugin Discovery

hipDNN supports a plugin architecture for both execution engines and heuristic policies. Plugins are automatically discovered at runtime from configurable search paths.

#### HIPDNN_PLUGIN_DIR

Specifies the directory path where hipDNN will search for engine plugins (shared libraries that provide operation implementations).

| Value      | Description                                            |
|------------|--------------------------------------------------------|
| (unset)    | Uses default path: `hipdnn_plugins/engines/` (relative to current working directory) |
| `<path>`   | Search for engine plugins in the specified directory  |

When set, this variable completely overrides the default search path. The path can be absolute or relative to the current working directory.

**Example:**
```bash
export HIPDNN_PLUGIN_DIR=/opt/rocm/lib/hipdnn/plugins/engines
```

**Notes:**
- Engine plugins must implement the Engine Plugin API (see `EnginePluginApi.h`)
- Plugin libraries are typically named `libhipdnn_provider_*.so` (Linux) or `hipdnn_provider_*.dll` (Windows)
- Only plugins whose API version major matches `HIPDNN_ENGINE_API_VERSION_MAJOR` (declared in `hipdnn_plugin_sdk/engine_api_version.h`) will be loaded
- See the [Plugin Development Guide](PluginDevelopment.md) for details on creating engine plugins

#### HIPDNN_HEURISTIC_PLUGIN_DIR

Specifies the directory path where hipDNN will search for heuristic plugins (shared libraries that provide operation selection policies).

| Value      | Description                                            |
|------------|--------------------------------------------------------|
| (unset)    | Uses default path: `hipdnn_plugins/heuristics/` (relative to current working directory) |
| `<path>`   | Search for heuristic plugins in the specified directory |

When set, this variable completely overrides the default search path. The path can be absolute or relative to the current working directory.

**Example:**
```bash
export HIPDNN_HEURISTIC_PLUGIN_DIR=/opt/rocm/lib/hipdnn/plugins/heuristics
```

**Notes:**
- Heuristic plugins must implement the Heuristic Plugin API (see `HeuristicsPluginApi.h`)
- Heuristic plugins have independent API versioning from engine plugins
- Only plugins with API version matching the Heuristic API major version will be loaded
- Each heuristic plugin must provide a unique policy ID and policy name
- See the [Plugin Development Guide](PluginDevelopment.md) for details on creating heuristic plugins

### Heuristic Policy Selection

hipDNN's heuristic framework selects an engine for each graph by running a configurable list of selection policies (the *outer loop*). The following variables tune that loop and the behavior of two built-in policies.

#### HIPDNN_HEUR_POLICY_ORDER

Overrides the heuristic policy order for the outer loop. Read by every `EngineHeuristicDescriptor::finalize()` call.

| Value      | Description                                                |
|------------|------------------------------------------------------------|
| (unset)    | Use the descriptor's `HIPDNN_ATTR_ENGINEHEUR_POLICY_ORDER_EXT` attribute if set; otherwise fall back to the built-in default `[SelectionHeuristic::Config, SelectionHeuristic::StaticOrdering]`. |
| `<list>`   | Comma-separated tokens consulted in the order written. Each token is either a policy name (hashed via `policyNameToId`) or a raw decimal int64 policy ID. Whitespace around tokens is trimmed; empty tokens are skipped. |

This variable has the **highest priority** — it overrides both the descriptor attribute and the built-in default.

**Example:**
```bash
# By name
export HIPDNN_HEUR_POLICY_ORDER="SelectionHeuristic::Config,SelectionHeuristic::StaticOrdering"

# By raw ID (or mixed names + IDs)
export HIPDNN_HEUR_POLICY_ORDER="-1234567890123456789,SelectionHeuristic::StaticOrdering"
```

#### HIPDNN_HEUR_CONFIG_PATH

Path to a JSON rule file consumed by the `SelectionHeuristic::Config` built-in policy. The file maps convolution op + tensor-shape patterns to a preferred engine name; the policy walks conv-like nodes in the serialized graph and, on the first matching rule, reorders the candidate engines so the chosen one runs first. Re-read on every `Finalize` invocation — there is no process-wide cache.

| Value      | Description                                                |
|------------|------------------------------------------------------------|
| (unset)    | The `SelectionHeuristic::Config` policy declines, allowing subsequent policies to run. |
| `<path>`   | Absolute or working-directory-relative path to a JSON rule file. |

If the file is missing, unreadable, fails to parse, no rule matches, or the matched engine name is not among the current candidates, the policy declines (so the outer loop continues with the next policy).

**Example:**
```bash
export HIPDNN_HEUR_CONFIG_PATH=/etc/hipdnn/engine_overrides.json
```

#### HIPDNN_HEUR_FALLBACK_ENGINE_ORDER

Replaces the built-in ordering used by `SelectionHeuristic::StaticOrdering`. When set, **only** engines named here are eligible — anything else is dropped from the candidate list.

| Value      | Description                                                |
|------------|------------------------------------------------------------|
| (unset)    | Use the built-in static ordering (MIOpen-first, deterministic engines last). |
| `<list>`   | Comma-separated engine names or raw IDs, applied in the order written. Whitespace is trimmed and empty tokens are skipped. |

Each entry accepts any spelling `hipdnn_list_engines` prints an engine under: the name it declares, or — for an engine that declares none — its ID in `0x`-prefixed hexadecimal. A decimal ID is also accepted. A registered name is resolved as a name even if it reads as a number.

Entries that are not among the current candidates are silently skipped. If no listed engine matches any candidate, the policy declines so the outer loop can try the next policy.

**Example:**
```bash
# By name
export HIPDNN_HEUR_FALLBACK_ENGINE_ORDER="MIOPEN_ENGINE,HIPBLASLT_ENGINE"

# By ID, in the form printed for an engine that declares no name, mixed with a name
export HIPDNN_HEUR_FALLBACK_ENGINE_ORDER="0x1A2B3C4D5E6F7080,MIOPEN_ENGINE"
```

#### HIPDNN_DISABLE_EXACT_ENGINE_CACHE

> [!NOTE]
> This name is **provisional**. It may be renamed or superseded by a future shared
> cache-configuration mechanism.

Disables the exact-match autotune cache consulted by `SelectionHeuristic::Config`: a
machine-written record, keyed on the full serialized graph plus device properties, that
captures the engine order measured by a prior exhaustive-autotune run. When enabled (the
default), a cache hit for the current graph wins outright and pre-empts the fuzzy
`HIPDNN_HEUR_CONFIG_PATH` rules. This variable only toggles that lookup on or off — it does
not select or move the cache's on-disk location, which remains `HIPDNN_CACHE_DIR`.

| Value      | Description                                                |
|------------|------------------------------------------------------------|
| (unset)    | Exact-match cache enabled (default). |
| `1`, `true`, `on`, `yes`, `enable`, `enabled` | Disable the exact-match cache; the policy falls through directly to the fuzzy rules and static ordering. |
| Anything else (including `0`, `false`, `off`) | Cache remains **enabled** — presence alone does not disable it, and an operator scripting `HIPDNN_DISABLE_EXACT_ENGINE_CACHE=0` gets the behavior that spelling implies. |

Values are matched case-insensitively against the literal truthy set above after trimming
surrounding whitespace; an unrecognized value is silently treated as unset rather than
rejected. Read fresh on every `policyFinalize()` call, so a change takes effect without a
process restart.

**Example:**
```bash
# Disable the exact-match cache for this run; only fuzzy rules and static
# ordering are consulted.
export HIPDNN_DISABLE_EXACT_ENGINE_CACHE=1
```

**Notes:**
- Disabling this cache does not disable the fuzzy `HIPDNN_HEUR_CONFIG_PATH` mechanism —
  the two are independent policies within `SelectionHeuristic::Config`.
- A cache entry that names a candidate engine never actually benchmarked is rejected
  automatically regardless of this variable; this variable is the coarse-grained on/off
  switch, not a way to tune that per-entry applicability check.

### Benchmarking

#### HIPDNN_FORCE_BENCHMARKING

A process-wide override for the `global.benchmarking` knob, independent of any engine's own knob setting. Every provider implementing `global.benchmarking` -- today the generic kernel ingestor engine and the MIOpen provider -- consults it.

| Value      | Description                                            |
|------------|--------------------------------------------------------|
| (unset)    | No effect. Benchmarking is whatever the `global.benchmarking` knob says (the default) |
| `1`, `true`, `on`, `yes`, `enable`, `enabled` | Force benchmarking **on**, regardless of the knob |
| `0`, `false`, `off`, `no`, `disable`, `disabled` | Force benchmarking **off**, overriding the knob and autotune's EXHAUSTIVE priming |

Values are case-insensitive and tolerant of surrounding whitespace (`ON`, ` On `, `TRUE`, `Off` all resolve). Any value not in the table is ignored and treated as unset, never as on, and logs a warning naming the variable and the value. The empty string is indistinguishable from unset and is silently ignored.

The override needs no autotune call and no knob setting to take effect: setting it to `1` benchmarks a plain `hipdnnExecute()` with no other configuration.

**Example:**
```bash
# Force benchmarking on for every provider that implements the knob
export HIPDNN_FORCE_BENCHMARKING=1

# Force it off, even if a caller's knob or autotune run asked for it
export HIPDNN_FORCE_BENCHMARKING=0
```

**Notes:**
- With benchmarking on, the first `execute()` of a plan samples every knob-filtered candidate kernel before caching the winner for the plan's life, so it is slower than subsequent calls.
- The variable is process-wide, with no per-provider or per-engine granularity: a leaked value from one test or shell changes an unrelated run.
- `HIPDNN_FORCE_BENCHMARKING=0` also defeats `Graph::autotune()` in EXHAUSTIVE mode, which otherwise sets `global.benchmarking=1` on its priming plans.
- Sampling executes each candidate against the buffers you passed in, so benchmarking assumes idempotent execution or separate input and output buffers -- the same assumption `autotune()` documents. A graph whose output tensor is also one of its inputs is recomputed in place once per sample. The winner runs last, so the final contents are correct, but the buffer is written many times before that.

### Caching

#### HIPDNN_CACHE_DIR

The root directory hipDNN writes its on-disk caches to. Two features share it today: the kernel ingestor's winner cache, which persists the ranking a benchmarking run measured so a later process can reuse it instead of re-benchmarking, and the autotune exact-match ranking cache (see below).

| Value      | Description                                            |
|------------|--------------------------------------------------------|
| (unset)    | A per-user default is used: `~/.cache/hipdnn/` on Linux, `%USERPROFILE%\.hipdnn\cache\` on Windows (the default) |
| `<path>`   | Use the given directory as the cache root instead       |

A leading `~` (and, on Windows, a leading `%USERPROFILE%`) is expanded; every other character is left alone, so an embedded `~` is never substituted. If the relevant home variable is unset or empty, the value is used verbatim rather than being redirected somewhere unexpected. Relative paths resolve against the current working directory. The directory is created if it does not exist.

Each feature owns a child directory beneath the root, so caches never collide: the ingestor's winner cache lives under `ingestor-winners/`, further split per hipDNN version, per engine, and per GPU architecture; the autotune exact-match ranking cache lives under `autotune-rankings/`, further split per `data_sdk` library version (see below).

**Example:**
```bash
# Point every hipDNN cache at a scratch directory, e.g. for a CI job
export HIPDNN_CACHE_DIR=/tmp/hipdnn-cache

# Isolate one reproduction from a developer's normal cache
export HIPDNN_CACHE_DIR=$PWD/.hipdnn-cache
```

**Notes:**
- Caching is **on by default**. Setting `HIPDNN_CACHE_DIR` to an empty string does **not**
  disable caching -- an empty value is treated as unset and the per-user default applies,
  leaving the cache enabled. Use `HIPDNN_DISABLE_CACHE` (below) to turn it off.
- Cache entries are stamped with the hipDNN version that produced them, so an upgrade does not read measurements taken by a different build. Stale directories from older versions are not removed automatically; deleting the cache root is always safe.
- Two checkouts, or two CI jobs, that should not share measurements need different values here. Records are keyed by graph content and device, not by checkout.
- The winner cache is append-only and is never compacted, so a long-lived cache directory grows with the number of distinct graphs benchmarked.
- A lookup alone can create a file: `openLineStore()` opens (creating it if absent) the shard
  file and writes its version-stamp line before the caller learns whether the key it is querying
  has a record in it. A pure read miss therefore still leaves a version-stamped, otherwise-empty
  shard behind. This is a property of the shared `LineStore` layer underneath every disk cache --
  both the ingestor winner cache and the autotune ranking cache do it -- not something specific
  to one subtree.

#### HIPDNN_DISABLE_CACHE

Turns hipDNN's on-disk caching off entirely, regardless of `HIPDNN_CACHE_DIR`. This is the
dedicated kill switch: no directory is created, no cache file is opened, and every consumer
falls back to its in-memory-only behavior (e.g. the kernel ingestor re-benchmarks every
process instead of reusing a persisted ranking).

| Value      | Description                                            |
|------------|--------------------------------------------------------|
| (unset)    | No effect. Caching follows `HIPDNN_CACHE_DIR` as described above (the default) |
| `1`, `true`, `on`, `yes`, `enable`, `enabled` | Disable on-disk caching |
| Any other value | Ignored; treated as unset (fails open, not silently on) |

Values are case-insensitive and tolerant of surrounding whitespace, matching
`HIPDNN_FORCE_BENCHMARKING`'s parsing.

**Example:**
```bash
# Disable the on-disk cache for one run, e.g. to force a clean re-benchmark
export HIPDNN_DISABLE_CACHE=1
```

**Notes:**
- Takes precedence over `HIPDNN_CACHE_DIR`: both set is not an error, but the disable wins.
- Read once per `cacheRoot()` call, not cached across the process, so it can be toggled
  between runs without restarting anything long-lived.

### Autotune Ranking Cache (`autotune-rankings/`)

The second `HIPDNN_CACHE_DIR` consumer: the exact-match engine-ranking record consulted by the
`SelectionHeuristic::Config` built-in policy and gated by `HIPDNN_DISABLE_EXACT_ENGINE_CACHE`
(above). It persists the engine order an exhaustive autotune run measured, keyed on the (graph,
device) pair that produced it, so a later process facing the identical graph on the identical
device can reuse the ranking instead of re-benchmarking.

**On-disk layout:**
`$HIPDNN_CACHE_DIR/autotune-rankings/<data_sdk-version>/<combined-key-hex>.jsonl` -- one shard
file per key, named by the hex encoding of a 64-bit hash of the serialized graph's
cache-relevant content concatenated with a 64-bit hash of the serialized device properties. The
graph hash ignores fields the schema marks `(cache_ignore)` and resolves `(cache_uid)` tensor
references positionally, so two graphs that differ only there share a shard.

The directory component is the `data_sdk` library's version string
(`MAJOR.MINOR.PATCH.TWEAK`), where `TWEAK` is the short git commit hash the build was configured
from (or the literal `unknown` if that git command failed at configure time). A rebuild from a
different commit therefore reads and writes a different subdirectory: a field-deployed cache can
never be misread by a newer build, but a developer who rebuilds every few minutes will rarely see
a hit at all, since every rebuild starts that commit's subtree empty.

**What populates it.** `Graph::autotuneExhaustiveSweep()` (Python:
`autotune_exhaustive_sweep()`). A record is keyed on the graph and device *only* -- not on
knobs, engine filters, or the workspace budget -- and every later run of that graph consults
it however that run is configured. A ranking is written only when the sweep covered every
engine a later run could see: one candidate per engine with no knob variants, no
`engineIdFilter` or deselect filter, and a workspace at least as large as
`get_autotune_workspace_size()` reports. A sweep that does not meet this still runs and still
returns its results, but declines the write and reports `NOT_ATTEMPTED_PARTIAL_SWEEP`.

**A re-tune refreshes the record.** Writing a ranking identical to what a shard already holds
for that key is a no-op. Writing a ranking that measured the same
engines in a different order appends a new line -- lookups resolve multiple lines for one key
last-line-wins, so the newest measurement is what a later process sees. The one write that is
*not* accepted as a refresh: one whose sampled-engine set is a strict subset of what the shard
already has recorded (e.g. a run scoped with an engine filter) is declined outright, so a
narrower sweep can never regress a full-coverage record to one that then fails every later
lookup.

**A lookup never creates anything.** Reads open the shard without `O_CREAT`, so a key with no
shard is an ordinary miss leaving behind no file and no open descriptor; only the write path
creates the versioned subtree and its shards. The read path runs for every graph on the
default heuristic policy list over an unbounded key space, so a creating read would cost one
empty file and one process-lifetime descriptor per distinct graph ever looked up.

**Decline modes visible in `HIPDNN_LOG_LEVEL=info` logging**, each with its own distinguishable
`[BuiltInConfig] policyFinalize:` log fragment:

| Log fragment                            | Meaning                                                            |
|------------------------------------------|---------------------------------------------------------------------|
| `disabled`                              | `HIPDNN_DISABLE_EXACT_ENGINE_CACHE` is set; lookup skipped entirely |
| `unkeyable (no device properties set)`  | The descriptor never had device properties set                     |
| `unkeyable (empty graph or device view)`| The serialized graph or device buffer was empty                    |
| `exact-match cache miss`                | The (graph, device) key has no shard record                        |
| `exact-match cache unavailable` (warn)  | The shard could not be opened, locked, or read, or its version line did not match this build |
| `rejected -- unsampled`                 | A live candidate engine was never sampled by the stored ranking     |
| `declined -- malformed record` (warn)   | The stored order holds an engine id more than once, so it cannot be applied to the candidate set. Re-run the exhaustive sweep to replace it |
| `declined -- fewer than 2 candidates`   | Fewer than two ids remain after filtering the stored order to live candidates |
| `hit (exact)`                           | Every stored id is a live candidate; the stored order applies unchanged |
| `hit (partial)`                         | The stored order applies after dropping sampled-but-now-absent ids  |

Every one of these is a fall-through, not an error: on any decline the outer heuristic loop
simply continues to the fuzzy `HIPDNN_HEUR_CONFIG_PATH` rules and static ordering, exactly as if
the exact-match cache did not exist.

**Deleting `autotune-rankings/` is always safe.** Every read path above fails soft to "consult
the fuzzy rules instead" -- there is no path that treats a missing, empty, or unreadable shard as
an error. The next matching graph simply autotunes and repopulates it.

### Logging Variables

hipDNN provides the following environment variables to control logging behavior:
#### HIPDNN_LOG_LEVEL

Sets the minimum severity that will be emitted. Levels are inclusive: choosing a level enables messages at that level and all higher severities.

| Level  | Description                                                |
|--------|------------------------------------------------------------|
| `off`  | Disables all logging (default)                             |
| `info` | General informational messages                             |
| `warn` | Potential issues that do not interrupt execution           |
| `error`| Recoverable errors that may affect results or performance  |
| `fatal`| Unrecoverable errors; the operation will not continue      |

**Example:**
```bash
export HIPDNN_LOG_LEVEL=info
```

#### HIPDNN_LOG_FILE

Specifies the file path where logs will be **appended**. If not set, logs are written to `stderr`.

**Example:**
```bash
export HIPDNN_LOG_FILE=/path/to/hipdnn.log
```

#### HIPDNN_LOG_GRAPH_DIR

Controls graph structure logging. When set to a non-empty directory path, graphs are written as JSON files during finalization.

| Value      | Description                                            |
|------------|--------------------------------------------------------|
| (unset)    | Graph logging disabled (default)                       |
| `<path>`   | Write graph structures as JSON files to the given directory |

Graph JSON files are written to the directory specified by `HIPDNN_LOG_GRAPH_DIR`. If the directory does not exist, it is created automatically. Relative paths are resolved against the current working directory. Files are named `graph_<id>.json` after the graph's own ID, so finalizing a graph again, or replaying a serialized one, reuses a single file. A graph rebuilt from scratch is a new graph with a new ID and is written separately; point the variable at a per-process directory if many processes share one output location.

This variable is independent of `HIPDNN_LOG_LEVEL` and `HIPDNN_LOG_FILE`.

**Example:**
```bash
export HIPDNN_LOG_GRAPH_DIR=/tmp/hipdnn_graphs
```

### MIOpen Plugin Logging

> [!TIP]
> 💡 When using the MIOpen Provider Plugin, you can use MIOpen-specific environment variables to control the underlying library's logging behavior.

For more details about MIOpen logging, see the latest [MIOpen Debug and Logging documentation](https://rocm.docs.amd.com/projects/MIOpen/en/develop/how-to/debug-log.html). All MIOpen environment variables remain compatible with hipDNN's MIOpen Provider Plugin.

### Test Configuration

#### HIPDNN_GLOBAL_TEST_SEED

Controls the random number generator seed used across hipDNN tests. This allows for reproducible test runs or full randomization when needed.

| Value        | Description                                                |
|--------------|------------------------------------------------------------|
| (not set)    | Uses default seed value of `1` (default behavior)         |
| `<number>`   | Uses the specified numeric seed (e.g., `42`, `12345`)     |
| `RANDOM`     | Generates a random seed using `std::random_device`        |

> [!NOTE]
> The `RANDOM` value is case-insensitive (`random`, `Random`, `RANDOM` all work).

**Examples:**
```bash
# Use a specific seed for consistent results
export HIPDNN_GLOBAL_TEST_SEED=42

# Use default seed (1) for reproducible tests
unset HIPDNN_GLOBAL_TEST_SEED

# Use random seed for each test run
export HIPDNN_GLOBAL_TEST_SEED=RANDOM
```

**Best Practices:**
- Use the default seed (1) for CI/CD pipelines to ensure consistent test results
- Use a specific numeric seed when debugging to reproduce exact test conditions
- Use `RANDOM` during development to catch edge cases with different data patterns

---


## Logging Configuration APIs

### User Log Callbacks

One or more user callback functions can be registered to receive log messages from the hipDNN library. User callbacks do not replace console/file logging, rather they provide an additional parallel method to receive log messages.

Each callback is uniquely identified by the composite key `(callback, userHandle)`, which allows registering the same callback function multiple times with different user handles (e.g., for different logging destinations).

#### Callback Signature

User callbacks must conform to the following C signature:

```c
typedef void (*hipdnnUserLogCallback_t)(hipdnnUserLogCallbackHandle_t userHandle,
                                        hipdnnSeverity_t severity,
                                        const char* message);
```

- `userHandle` — The opaque user-provided pointer passed back on each invocation
- `severity` — The severity level of the log message
- `message` — The formatted log message (null-terminated string)

#### Registering a Callback

The frontend API function `setUserLogCallback()` registers, updates, or removes a callback:

```cpp
Error setUserLogCallback(hipdnnUserLogCallback_t callback,
                         hipdnnSeverity_t minLevel,
                         LogCallbackMode mode,
                         hipdnnUserLogCallbackHandle_t userHandle);
```

**Parameters:**
- `callback` — The callback function to invoke (must be non-null)
- `minLevel` — Minimum severity level for messages delivered to this callback. Use `HIPDNN_SEV_OFF` to remove the callback.
- `mode` — `LogCallbackMode::ASYNC` (default, non-blocking) or `LogCallbackMode::SYNC` (blocking)
- `userHandle` — Non-null opaque pointer passed to the callback and used as part of the unique ID

**Behavior:**
- If `(callback, userHandle)` is not yet registered: **adds** a new registration
- If `(callback, userHandle)` is already registered: **updates** the level and/or mode
- If `minLevel` is `HIPDNN_SEV_OFF`: **removes** the registration

> [!NOTE]
> * The messages delivered to the callback are also subject to the global log level set by the `HIPDNN_LOG_LEVEL` environment variable or the `setGlobalLogLevel()` API. A callback registered at `HIPDNN_SEV_INFO` will not receive info-level messages if the global level is set to `HIPDNN_SEV_WARN` or higher.
> * The hipDNN logger has an internal log message queue of 8192 messages. Once the message queue is full, the
> hipDNN library will block on subsequent logging calls until space is made available in the queue. The
> callbacks must ensure that they are consuming messages and returning promptly to avoid stalling hipDNN.

#### Callback Modes

| Mode | Description |
|------|-------------|
| `LogCallbackMode::ASYNC` | Callback is invoked on a background worker thread. hipDNN is not blocked while the callback runs. Recommended for production use. |
| `LogCallbackMode::SYNC` | Callback is invoked on the calling thread. hipDNN blocks until the callback returns. Recommended only for debugging or testing. |

#### Removing a Callback

To remove a callback, call `setUserLogCallback()` with `minLevel` set to `HIPDNN_SEV_OFF`:

```cpp
auto error = setUserLogCallback(myCallback, HIPDNN_SEV_OFF, LogCallbackMode::ASYNC, myHandle);
```

After this call returns:
- The callback will not be invoked again
- Any pending async log messages for this callback are abandoned
- The caller can safely destroy data referenced by `userHandle`

#### Example

```cpp
#include <hipdnn_frontend.hpp>

using namespace hipdnn_frontend;

struct MyLogContext {
    std::ofstream logFile;
};

void myLogCallback(hipdnnUserLogCallbackHandle_t userHandle,
                   hipdnnSeverity_t severity,
                   const char* message) {
    auto* ctx = static_cast<MyLogContext*>(userHandle);
    ctx->logFile << message << std::endl;
}

// Register an async callback at INFO level
MyLogContext ctx{std::ofstream("my_log.txt")};
auto error = setUserLogCallback(&myLogCallback, HIPDNN_SEV_INFO,
                                LogCallbackMode::ASYNC, &ctx);

// ... use hipDNN APIs ...

// Remove the callback when done
error = setUserLogCallback(&myLogCallback, HIPDNN_SEV_OFF,
                           LogCallbackMode::ASYNC, &ctx);
// ctx can now be safely destroyed
```

### Log Level APIs

The following frontend API functions can programatically read and override the log level set by the `HIPDNN_LOG_LEVEL` environment variable:
```
Error getGlobalLogLevel(hipdnnSeverity_t& level)
```
Returns the current log level in use by the hipDNN library, including `HIPDNN_SEV_OFF` if logging is not enabled.
```
Error setGlobalLogLevel(hipdnnSeverity_t level)
```
Sets hipDNN to the specified log level. Use `HIPDNN_SEV_OFF` to disable logging.

## Error Handling

hipDNN provides functions for retrieving error information:

### Getting Error Strings

```c
// Convert status code to string
const char* error_str = hipdnnGetErrorString(status);

// Get detailed error message for the current thread
char message[HIPDNN_ERROR_STRING_MAX_LENGTH];
hipdnnGetLastErrorString(message, sizeof(message));
```

### Best Practices

1. Check return status codes from all hipDNN API calls
2. Use `hipdnnGetLastErrorString` for detailed error context
3. Enable appropriate logging levels during development and debugging
4. Configure logging to files for production deployments
