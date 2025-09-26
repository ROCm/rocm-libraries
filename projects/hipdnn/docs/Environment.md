# hipDNN Environment Configuration

This document describes the environment variables and runtime configuration options for hipDNN.

## Table of Contents

- [Environment Variables](#environment-variables)
  - [Logging Configuration](#logging-configuration)
  - [MIOpen Plugin Logging](#miopen-plugin-logging)
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
