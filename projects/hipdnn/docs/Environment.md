# hipDNN Environment Configuration

This document describes the environment variables and runtime configuration options for hipDNN.

## Table of Contents

- [Environment Variables](#environment-variables)
  - [Logging Variables](#logging-variables)
  - [MIOpen Plugin Logging](#miopen-plugin-logging)
  - [Test Configuration](#test-configuration)
- [Logging Configuration APIs](#logging-configuration-apis)
  - [Global Log Callback](#global-log-callback)
  - [Log Level APIs](#log-level-apis)
- [Error Handling](#error-handling)

---

## Environment Variables

### Logging Variables

hipDNN provides two environment variables to control logging behavior:
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

### Global Log Callback

A callback function can be registered to receive log messages from the hipDNN library. Once a callback function is registered, all logs will be output to the registered logging callback insteasd of the console or file specified by the `HIPDNN_LOG_FILE` environment variable. Setting the logging callback to `nullptr` will re-enable logging to the console or log file.

The logging callback is registered using the following frontend API function:
```
Error setGlobalLoggingCallback(hipdnnBackendLogOutputCallback_t callback, bool async = true);
```
This function registers the logging callback with hipDNN. If `async` is true then logs will be output using a separate thread so that the hipDNN library is not blocked while the callback function is running. Setting `callback` to `nullptr` will disable the logging callback.

Logs output using the callback function are filtered by the leve set by the `HIPDNN_LOG_LEVEL` environment variable described above, or programatically using the `getGlobalLogLevel()` API function described below.

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
