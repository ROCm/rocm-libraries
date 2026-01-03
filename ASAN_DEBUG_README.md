# TEMPORARY ASAN DEBUG BUILD CONFIGURATION

## What Was Changed

This branch has been temporarily modified to enable Address Sanitizer (ASAN) and diagnostic logging for CI debugging of intermittent memory corruption issues.

### Files Modified:

1. **`install.sh`** - Added ASAN forcing and diagnostic env vars
2. **`clients/common/host_alloc.cpp`** - Added debug logging for untracked pointers
3. **This README** - Documentation of temporary changes

## Changes in install.sh

The script now:
- Forces `--address-sanitizer` flag when building
- Forces `--relwithdebinfo` for debug symbols with optimization
- Sets `ASAN_OPTIONS` for comprehensive diagnostics
- Sets `ROCBLAS_CLIENT_DEBUG_ALLOC=1` for memory tracking warnings
- Sets `AMD_LOG_LEVEL=3` for verbose HIP logging

## What to Expect in CI

When this runs through CI, you should see:

1. **Longer build times** - ASAN adds overhead
2. **ASAN logs** in `/tmp/asan_rocblas.*` if corruption is detected
3. **Warning messages** if untracked pointers are being freed
4. **Detailed stack traces** showing exactly where memory corruption occurs

## Reading ASAN Output

If ASAN catches the issue, look for:
- `ERROR: AddressSanitizer: heap-buffer-overflow`
- `ERROR: AddressSanitizer: heap-use-after-free`
- `ERROR: AddressSanitizer: double-free`

The stack trace will show:
1. Where the corruption was detected
2. Where the memory was originally allocated
3. Where it was freed (if applicable)

## How to Revert

To remove all debugging and return to normal build:

```bash
# Revert install.sh changes
git checkout install.sh

# Optionally revert host_alloc.cpp logging (it's harmless to leave)
git checkout clients/common/host_alloc.cpp

# Delete this README
rm ASAN_DEBUG_README.md
```

## Expected Behavior

If the memory corruption is real and happens during the CI run, ASAN will:
1. Catch it immediately
2. Print detailed diagnostics
3. Write logs to /tmp/asan_rocblas.*
4. The test will fail (which is what we want - we need the stack trace)

The ASAN output will tell us **exactly** what's wrong and where to fix it.

## Environment Variables Set

- `ASAN_OPTIONS=detect_leaks=1:halt_on_error=0:log_path=/tmp/asan_rocblas:print_stacktrace=1:symbolize=1`
- `ROCBLAS_CLIENT_DEBUG_ALLOC=1`
- `AMD_LOG_LEVEL=3`
- `ASAN_FORCED=1`

---
**Note:** These are temporary debugging changes. Do NOT merge this to develop!

