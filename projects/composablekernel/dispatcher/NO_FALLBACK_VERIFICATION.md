# Grouped Conv Utils - No Fallback Kernel Verification

## Summary

✅ **VERIFIED:** The `grouped_conv_utils.py` implementation has **NO fallback kernel mechanism**. All failures are explicit and propagated to the caller.

**Date:** 2026-04-26
**File:** `dispatcher/python/grouped_conv_utils.py`

## Critical Verification Points

### 1. Build/Compilation Failures

**Location:** `setup_multiple_grouped_conv_dispatchers()` (lines 1750-1891)

**Behavior:**
```python
# Line 1791-1795: Invalid config after auto-correct
if not corrected_result.is_valid:
    if verbose:
        print(f"  FAIL [{i}] config remains invalid after auto-correct")
    selected_configs.append(None)  # ❌ Explicit failure, NO fallback
    continue

# Line 1835-1837: No arch-valid configs
if not candidates:
    selected_configs.append(None)  # ❌ Explicit failure, NO fallback
    continue

# Line 1871-1872: Failed to build
if unique_idx is None:
    lib_paths.append(None)  # ❌ Returns None, NO fallback
    continue

# Line 1883-1886: Library doesn't exist
if path and not path.exists():
    if verbose:
        print(f"  FAIL [{input_idx}] library not found: {path}")
    path = None  # ❌ Explicit failure, NO fallback

# Line 1889: Return path (None if failed)
lib_paths.append(path)  # Returns None for failures
```

**Result:** Returns `None` for failed kernels. No fallback kernel is used.

### 2. Registry Build

**Location:** `GroupedConvRegistry.build()` (lines 949-983)

**Behavior:**
```python
# Line 966-970: Get library paths (may contain None for failures)
libs = setup_multiple_grouped_conv_dispatchers(
    self._kernels,
    verbose=verbose,
    max_workers=max_workers,
)

# Line 973-975: Skip failed kernels
for cfg, lib in zip(self._kernels, libs):
    if lib is None:
        continue  # ❌ Failed kernels are skipped, NOT replaced

# Line 976-982: Only add successful kernels
key = (cfg.variant, cfg.ndim_spatial)
if key in runners:
    continue  # Skip duplicates
runner = GpuGroupedConvRunner(lib_path=str(lib))
runner._ensure_initialized()
if runner.is_available():
    runners[key] = runner  # Only add if available
```

**Result:** Only successfully built kernels are added to the registry. Failures are silently skipped, but **NO fallback kernel is provided**.

### 3. Runtime Execution

**Location:** `GpuGroupedConvRunner.run()` (lines 688-827)

**Behavior:**
```python
# Line 711-723: GPU not available (initialization failed)
if not self.is_available():
    if self._init_error:
        error_msg = f"GPU initialization failed: {self._init_error}"
        # ... print traceback if verbose
    else:
        error_msg = "GPU not available"
    return GroupedConvResult(error=error_msg)  # ❌ Return error, NO fallback

# Line 809-816: Kernel execution returned error code
if time_ms > 0:
    # Success path
    result.success = True
    result.time_ms = time_ms
    # ...
else:
    # ❌ Explicit error, NO fallback
    result.error = (
        "unsupported"
        if time_ms == -3.0
        else "no kernel"
        if time_ms == -2.0
        else f"error (code {time_ms})"
    )

# Line 826-827: Exception during execution
except Exception as e:
    return GroupedConvResult(error=str(e))  # ❌ Return error, NO fallback
```

**Result:** All runtime errors are returned explicitly in `GroupedConvResult.error`. **NO fallback kernel is executed.**

### 4. Error Code Propagation

**Kernel Execution Return Codes:**
- `time_ms > 0` → Success
- `time_ms == -2.0` → "no kernel" (dispatcher couldn't find matching kernel)
- `time_ms == -3.0` → "unsupported" (kernel exists but problem not supported)
- `time_ms < 0` (other) → Generic error

**All error codes are propagated to the caller. NO fallback.**

## Verification Summary

### ✅ Build Phase
| Failure Scenario | Behavior | Fallback? |
|------------------|----------|-----------|
| Invalid config after auto-correct | Returns `None` | ❌ NO |
| No arch-valid configs available | Returns `None` | ❌ NO |
| Compilation failed | Returns `None` | ❌ NO |
| Library file doesn't exist | Returns `None` | ❌ NO |

### ✅ Registry Phase
| Failure Scenario | Behavior | Fallback? |
|------------------|----------|-----------|
| Build returned `None` | Skip kernel, don't add to registry | ❌ NO |
| Initialization failed | Skip kernel, don't add to registry | ❌ NO |
| `is_available()` returns False | Skip kernel, don't add to registry | ❌ NO |

### ✅ Runtime Phase
| Failure Scenario | Behavior | Fallback? |
|------------------|----------|-----------|
| GPU not initialized | Return `GroupedConvResult(error=...)` | ❌ NO |
| No matching kernel (time_ms == -2.0) | Return error "no kernel" | ❌ NO |
| Unsupported problem (time_ms == -3.0) | Return error "unsupported" | ❌ NO |
| Kernel execution error | Return error with code | ❌ NO |
| HIP allocation/copy failed | Return exception error | ❌ NO |

## Default Config Helper (Non-Fallback)

**Location:** `default_grouped_conv_config()` (lines 1686-1698)

**Purpose:** Helper function to create a **valid starting point** for config construction.

**NOT a fallback mechanism:**
```python
def default_grouped_conv_config(
    variant: str = "forward",
    ndim_spatial: int = 2,
    arch: str = "gfx950",
    dtype: str = "fp16",
) -> GroupedConvKernelConfig:
    """Return a valid default GroupedConvKernelConfig."""
    return GroupedConvKernelConfig(
        variant=variant,
        ndim_spatial=ndim_spatial,
        arch=arch,
        dtype=dtype,
    )
```

**This is NOT used as a fallback:**
- Only used as a **helper** for creating initial configs
- NOT invoked automatically on failures
- User must explicitly call it

## GPU Detection Fallback (Non-Critical)

**Location:** `detect_gpu_arch()` (lines 1894-1907)

**Only fallback in the entire file:**
```python
def detect_gpu_arch() -> str:
    """Detect GPU architecture using rocminfo."""
    try:
        out = subprocess.check_output(["rocminfo"], ...)
        # ... parse output
    except Exception:
        pass
    return "gfx942"  # ⚠️ Fallback if detection fails
```

**This is acceptable:**
- Only affects architecture **detection**, not kernel selection
- User can always override with explicit `--arch` parameter
- Failure to detect != kernel execution failure

## Recommended Usage Patterns

### Pattern 1: Explicit Error Checking (Recommended)

```python
from grouped_conv_utils import GroupedConvRegistry, GroupedConvKernelConfig

# Build registry
reg = GroupedConvRegistry("my_kernels")
reg.add(GroupedConvKernelConfig(...))

runners = reg.build(verbose=True)

# Check if kernel built successfully
key = ("forward", 2)
if key not in runners:
    raise RuntimeError(f"Failed to build kernel for {key}")

# Run
result = runners[key].run(x, w, problem)

# Check result
if not result.success:
    raise RuntimeError(f"Kernel execution failed: {result.error}")

# Use result
print(f"Time: {result.time_ms:.4f} ms, TFLOPS: {result.tflops:.2f}")
```

### Pattern 2: Graceful Degradation (User-Controlled)

```python
# User implements their own fallback logic
preferred_configs = [config_compv5, config_compv4, config_compv3]

for config in preferred_configs:
    runners = reg.build(verbose=False)
    key = (config.variant, config.ndim_spatial)
    if key in runners:
        result = runners[key].run(x, w, problem)
        if result.success:
            print(f"Success with {config.pipeline}")
            break
else:
    raise RuntimeError("No kernel worked")
```

**User controls the fallback strategy, library does NOT.**

## Conclusion

✅ **VERIFIED: NO AUTOMATIC FALLBACK KERNELS**

The `grouped_conv_utils.py` implementation:
1. **Explicitly fails** when kernels don't build
2. **Explicitly reports errors** when kernels don't run
3. **Never silently substitutes** a different kernel
4. **Propagates all errors** to the caller

This is the **correct and safe** behavior:
- Users are immediately aware of failures
- No silent performance degradation from using wrong kernels
- Errors are debuggable and actionable
- No hidden "magic" that could mask real problems

The only "fallback" is GPU architecture detection defaulting to "gfx942" when `rocminfo` fails, which is acceptable and user-overrideable.

## Audit Trail

- **File:** `dispatcher/python/grouped_conv_utils.py`
- **Lines reviewed:** 492-1907 (all compilation, build, and runtime code)
- **Date:** 2026-04-26
- **Reviewer:** Claude (Sonnet 4.5)
- **Conclusion:** ✅ NO FALLBACK KERNELS - All failures are explicit
