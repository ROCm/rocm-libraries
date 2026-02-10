# MIOpen Provider Plugin - Performance Characteristics

## Algorithm Selection Behavior

### Lazy Algorithm Selection for Convolution Operations

The MIOpen Provider Plugin performs **algorithm selection during the first execution** of convolution plans, rather than during plan construction. This is necessary because MIOpen's algorithm selection APIs require actual device memory buffers to benchmark candidate kernels, which are only available during plan execution.

#### Affected Operations
- Convolution Forward (Conv Fwd)
- Convolution Backward Data (Conv Bwd)
- Convolution Backward Weights (Conv Wrw)

#### Performance Impact

| Execution | Performance | Notes |
|-----------|-------------|-------|
| **First** | May be slower | Algorithm selection overhead (small unless benchmarking is enabled) |
| **Subsequent** | Normal performance | Uses cached algorithm selection |

The first execution performs algorithm selection. The overhead is typically small, but if benchmarking is enabled, kernel benchmarking adds significant overhead. All subsequent executions use the cached selection and execute at normal speed.

#### Monitoring

The plugin logs when algorithm selection occurs:

```
Convolution Fwd: Performing algorithm selection (first execution)
Convolution Fwd: Selected algorithm=1, time=2.45, workspace_size=8388608
```

#### Best Practices

**For latency-sensitive applications:**
- Perform a "warm-up" execution during initialization
- Separate first-execution timing from performance metrics

**For optimal performance:**
- Reuse plan instances across multiple executions
- Algorithm selection is thread-safe and occurs only once per plan instance
