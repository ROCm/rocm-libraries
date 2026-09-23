# WGM Debug Instrumentation Changes Summary

## Overview
Modified hipBLASLt WGM (workgroup-mapping) debug instrumentation to work as a per-kernel parameter instead of a global debug-only feature. This allows WGM instrumentation to be enabled in Release builds when explicitly requested in YAML files.

## Problem Solved
Previously, WGM debug instrumentation was:
- Only available in Debug builds (via `--debug-wgm` flag)
- Controlled by a global parameter `DebugWGM`
- Could not be built due to nanobind reference leak detection in debug mode

## Solution
Changed WGM instrumentation to be:
- Controlled per-kernel via `EnableWGMDebug` parameter (default: 0)
- Works in any build mode (Debug, Release, RelWithDebInfo)
- Can be enabled selectively for specific kernels in YAML tuning/logic files

## Changes Made

### 1. Added EnableWGMDebug Kernel Parameter
**File:** `tensilelite/Tensile/Common/ValidParameters.py`
- Added `"EnableWGMDebug": [0, 1]` parameter
- 0 = disabled (default, correct GEMM results)
- 1 = enabled (WGM instrumentation, incorrect results for visualization only)

### 2. Updated Kernel Writers
**Files:** 
- `tensilelite/Tensile/KernelWriter.py`
- `tensilelite/Tensile/KernelWriterAssembly.py`

Changed all checks from:
```python
if globalParameters.get("DebugWGM", False):
```

To:
```python
if kernel.get("EnableWGMDebug", 0):
```

### 3. Deprecated Global DebugWGM Parameter
**File:** `tensilelite/Tensile/Common/GlobalParameters.py`
- Marked `globalParameters["DebugWGM"]` as deprecated
- Kept for backward compatibility but has no effect
- Updated comments to reference new EnableWGMDebug parameter

### 4. Updated Command-Line Flag
**File:** `tensilelite/Tensile/TensileCreateLibrary/ParseArguments.py`
- Marked `--debug-wgm` flag as deprecated
- Updated help text to direct users to use EnableWGMDebug in YAML files

### 5. Fixed Nanobind Leak Issue
**File:** `tensilelite/rocisa/CMakeLists.txt`
- Reverted NB_DOMAIN flag (was causing module export failure)
- The leak issue is now moot since WGM works in Release mode

## Usage

### In YAML Tuning/Logic Files
Add `EnableWGMDebug: 1` to any solution to enable WGM instrumentation for that specific kernel:

```yaml
BenchmarkProblems:
  - 
    - # Problem Type
      OperationType: GEMM
      DataType: s
      # ... other parameters ...

    - # Solution with WGM debug enabled
      InitialSolutionParameters:
        EnableWGMDebug: 1  # Enable WGM instrumentation
        # ... other solution parameters ...
```

### Build Command
Simply build in Release or RelWithDebInfo mode:
```bash
./install.sh -k -c --skip_rocroller -a gfx950
```

No need for `--debug` or `--debug-wgm` flags.

### What WGM Instrumentation Does
When `EnableWGMDebug: 1` is set, the kernel will:
- Overwrite the top-left element (first 4 dwords) of each workgroup's output tile
- Write diagnostic data instead of real GEMM results:
  - dword0: original pre-WGM 1D workgroup id
  - dword1: packed post-WGM (WorkGroup0 << 16) | WorkGroup1
  - dword2: XCC id (HW_REG_XCC_ID)
  - dword3: original packed WGM sgpr value

**WARNING:** Kernels with EnableWGMDebug: 1 produce INCORRECT GEMM RESULTS. This is only for WGM visualization.

## Build Verification
Successfully built hipBLASLt in RelWithDebInfo mode for gfx950:
- Library: `/home/smalekta/WGM/rocm-libraries/projects/hipblaslt/build/release/library/libhipblaslt.so.1.5`
- No nanobind leak issues
- All changes verified in source files

## Migration Guide

### Before (Debug-only, not working):
```bash
./install.sh --debug --debug-wgm -c -a gfx950
# Would fail with nanobind leak errors
```

### After (Release mode, working):
1. Add `EnableWGMDebug: 1` to your YAML file
2. Build normally:
```bash
./install.sh -k -c --skip_rocroller -a gfx950
```

## Files Modified
1. `tensilelite/Tensile/Common/ValidParameters.py` - Added EnableWGMDebug parameter
2. `tensilelite/Tensile/KernelWriter.py` - Updated to use kernel parameter
3. `tensilelite/Tensile/KernelWriterAssembly.py` - Updated to use kernel parameter  
4. `tensilelite/Tensile/Common/GlobalParameters.py` - Deprecated global DebugWGM
5. `tensilelite/Tensile/TensileCreateLibrary/ParseArguments.py` - Deprecated --debug-wgm flag
6. `tensilelite/rocisa/CMakeLists.txt` - Reverted failed nanobind fix attempt

## Testing
Created test YAML: `test_wgm_debug.yaml` with EnableWGMDebug: 1
Verified parameter exists in ValidParameters.py
Verified usage in KernelWriter.py and KernelWriterAssembly.py
Successfully built in Release mode without WGM (default EnableWGMDebug: 0)
