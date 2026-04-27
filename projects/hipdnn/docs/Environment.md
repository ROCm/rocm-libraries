# hipDNN Environment Configuration

This document describes the environment variables and runtime configuration options for hipDNN.

## Table of Contents

- [Environment Variables](#environment-variables)
  - [Plugin Discovery](#plugin-discovery)
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
- Only plugins with API version matching the hipDNN backend major version will be loaded
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

Graph JSON files are written to the directory specified by `HIPDNN_LOG_GRAPH_DIR`. If the directory does not exist, it is created automatically. Relative paths are resolved against the current working directory. Files are named `graph_<hash>.json` where `<hash>` is derived from the graph content, ensuring identical graphs are not duplicated.

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
