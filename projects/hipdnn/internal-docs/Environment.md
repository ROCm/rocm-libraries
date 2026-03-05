# hipDNN Environment Configuration

This document describes the environment variables and runtime configuration options for hipDNN.

## Table of Contents

- [Environment Variables](#environment-variables)
  - [Logging Configuration](#logging-configuration)
  - [MIOpen Plugin Logging](#miopen-plugin-logging)
  - [Test Configuration](#test-configuration)
- [Error Handling](#error-handling)

---

## Environment Variables

### Logging Configuration

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

### Frontend and Plugin Logging

The frontend and plugins can be configured to use the same logging destination as the backend, which is lazy-initialized automatically:

1. Initialize logging using the `initializeCallbackLogging` function
2. Pass `hipdnnLoggingCallback_ext` as the callback function (accessible via plugin API or backend header)
3. This ensures all components log to the same destination

### MIOpen Plugin Logging

> [!TIP]
> 💡 When using the MIOpen legacy plugin, you can use MIOpen-specific environment variables to control the underlying library's logging behavior.

For more details about MIOpen logging, see the latest [MIOpen Debug and Logging documentation](https://rocm.docs.amd.com/projects/MIOpen/en/develop/how-to/debug-log.html). All MIOpen environment variables remain compatible with hipDNN's MIOpen legacy plugin.

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
