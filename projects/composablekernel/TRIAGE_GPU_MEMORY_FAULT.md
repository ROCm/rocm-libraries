# Triage Guide: GPU Memory Access Fault

## Problem Statement

GEMM Universal benchmark binary crashes with:
```
Memory access fault by GPU node-2 (Agent handle: 0x...) on address 0x7f...
```

Binary details:
- Built for: gfx950
- Running on: AMD Instinct MI355X (gfx950)
- Size: 8192x8192x8192 FP16 GEMM

## Step-by-Step Triage Process

### Step 1: Verify ROCm Runtime Setup

```bash
# 1.1 Check ROCm version
rocm-smi --showproductname
rocminfo | grep -E "Name:|Agent|GFX"

# 1.2 Check if GPU is accessible
rocm-smi

# 1.3 Verify hipcc works
hipcc --version

# 1.4 Check GPU memory availability
rocm-smi --showmeminfo
```

**Expected:** GPU visible, memory available, no errors

**If fails:** ROCm runtime issue - check installation

---

### Step 2: Test Simple HIP Program

Create and run a minimal HIP program:

```bash
cd /workspace/rocm-libraries/projects/composablekernel

cat > test_hip.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    std::cout << "Found " << deviceCount << " GPU(s)" << std::endl;

    if (deviceCount > 0) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        std::cout << "Device 0: " << prop.name << std::endl;
        std::cout << "GCN Arch: " << prop.gcnArchName << std::endl;

        // Test simple memory allocation
        float* d_ptr;
        size_t size = 1024 * 1024 * sizeof(float);  // 4MB
        hipError_t err = hipMalloc(&d_ptr, size);

        if (err != hipSuccess) {
            std::cerr << "hipMalloc failed: " << hipGetErrorString(err) << std::endl;
            return 1;
        }

        std::cout << "Successfully allocated 4MB on GPU" << std::endl;
        hipFree(d_ptr);
    }

    return 0;
}
EOF

# Compile and run
hipcc -o test_hip test_hip.cpp
./test_hip
```

**Expected:** "Successfully allocated 4MB on GPU"

**If fails:** GPU not accessible - driver/runtime issue

---

### Step 3: Test with Smaller Problem Size

Try a much smaller matrix to isolate memory issues:

```bash
cd build

# Test with 512x512x512 (much smaller)
./bin/benchmark_gemm_universal_fp16_ccr_compv3_cshuffle_intrawave_False_False_False_False_64x64x64_2x2x1_16x16x32 \
  -m=512 \
  -n=512 \
  -k=512 \
  -init=2 \
  -warmup=1 \
  -repeat=2 \
  -verify=0 \
  -json_output=false
```

**Expected:** Should run (uses ~2MB vs ~2GB for 8192³)

**If succeeds:** Memory size issue - GPU out of memory
**If fails:** Not a memory size issue - something else wrong

---

### Step 4: Check for Existing Working Examples

```bash
cd /workspace/rocm-libraries/projects/composablekernel

# Find any test binaries
find build* -name "test_*" -executable -type f | head -10

# Try running a simple test
./build/bin/test_gemm_tile_engine_fp16  # or any test that exists

# Or try an example
find build* -name "example_*" -executable | head -5
```

**Expected:** Some examples/tests should work

**If some work:** Compare what's different (flags, size, data type)
**If none work:** System-wide GPU issue

---

### Step 5: Build with Debug Info and Verbose Output

```bash
cd build

# Rebuild with debug symbols
cmake \
  -D CMAKE_BUILD_TYPE=Debug \
  -D GPU_TARGETS="gfx950" \
  -D GEMM_UNIVERSAL_DATATYPE="fp16" \
  -D GEMM_UNIVERSAL_LAYOUT="ccr" \
  -D GEMM_UNIVERSAL_CONFIG_FILE="default_ci_config.json" \
  -G Ninja \
  ..

# Rebuild the benchmark
ninja benchmark_gemm_universal_fp16_ccr_compv3_cshuffle_intrawave_False_False_False_False_64x64x64_2x2x1_16x16x32

# Run with verbose HIP output
HIP_VISIBLE_DEVICES=0 HSA_ENABLE_SDMA=0 AMD_LOG_LEVEL=4 \
  ./bin/benchmark_gemm_universal_fp16_ccr_compv3_cshuffle_intrawave_False_False_False_False_64x64x64_2x2x1_16x16x32 \
  -m=512 -n=512 -k=512 -init=0 -warmup=1 -repeat=1 -verify=0
```

**Expected:** More detailed error messages

---

### Step 6: Check GPU Memory Requirements

Calculate memory needed:

```python
# For 8192x8192x8192 FP16 GEMM
M = N = K = 8192
bytes_per_element = 2  # FP16

A_size = M * K * bytes_per_element  # Matrix A
B_size = K * N * bytes_per_element  # Matrix B
C_size = M * N * bytes_per_element  # Matrix C

total_GB = (A_size + B_size + C_size) / (1024**3)
print(f"Minimum GPU memory needed: {total_GB:.2f} GB")
# Expected: ~2 GB
```

Check available memory:

```bash
rocm-smi --showmeminfo vram | grep "Total Memory"
```

**Expected:** Should have >2GB free
**If not:** Reduce matrix size or free GPU memory

---

### Step 7: Try Different Kernel Configuration

Build a simpler kernel configuration:

```bash
cd build

# Look for what's available
ninja -t targets | grep "benchmark_gemm_universal_fp16_ccr" | head -20

# Try a different pipeline (mem instead of compv3)
ninja benchmark_gemm_universal_fp16_ccr_mem_default_intrawave_False_False_False_False_64x64x64_2x2x1_16x16x16

# Test it
./bin/benchmark_gemm_universal_fp16_ccr_mem_default_intrawave_False_False_False_False_64x64x64_2x2x1_16x16x16 \
  -m=512 -n=512 -k=512 -init=0 -warmup=1 -repeat=1 -verify=0
```

**Expected:** Some configurations may work

**If succeeds:** Issue with specific kernel config
**If fails:** System-wide issue

---

### Step 8: Check for Known Issues

```bash
# Check ROCm release notes for gfx950 issues
cat /opt/rocm/.info/version
cat /opt/rocm/share/doc/rocm/release_notes.txt 2>/dev/null | grep -i gfx950

# Check system logs
dmesg | tail -50 | grep -i "gpu\|amd\|hip"

# Check for HSA errors
cat /sys/kernel/debug/kfd/error
```

**Look for:** Known bugs, compatibility issues, driver errors

---

### Step 9: Test with Verification Enabled

Try CPU/GPU verification to isolate the issue:

```bash
cd build

# Test with CPU verification (smaller size)
./bin/benchmark_gemm_universal_fp16_ccr_compv3_cshuffle_intrawave_False_False_False_False_64x64x64_2x2x1_16x16x32 \
  -m=512 \
  -n=512 \
  -k=512 \
  -init=0 \
  -warmup=0 \
  -repeat=1 \
  -verify=1  # CPU verification

# Test with GPU verification
./bin/benchmark_gemm_universal_fp16_ccr_compv3_cshuffle_intrawave_False_False_False_False_64x64x64_2x2x1_16x16x32 \
  -m=512 \
  -n=512 \
  -k=512 \
  -init=0 \
  -warmup=0 \
  -repeat=1 \
  -verify=2  # GPU verification
```

**Expected:** Should identify where it fails (data init, GPU execution, verification)

---

### Step 10: Compare with Working Configuration

If you have access to a working system or older build:

```bash
# Option A: Use existing binaries from other build
ls -la build_pr5639/bin/benchmark* 2>/dev/null | head -5

# Option B: Check git history for working configs
git log --oneline --all -- tile_engine/ops/gemm/gemm_universal/ | head -20

# Option C: Test on different GPU if available
HIP_VISIBLE_DEVICES=1 ./bin/benchmark_...
```

---

## Decision Tree

```
GPU Memory Fault
       |
       ├─ Simple HIP test works?
       │     ├─ Yes → GPU accessible, continue
       │     └─ No → Fix ROCm runtime/driver
       │
       ├─ Small size (512³) works?
       │     ├─ Yes → Memory issue, reduce size or optimize
       │     └─ No → Continue
       │
       ├─ Different kernel config works?
       │     ├─ Yes → Issue with specific kernel, investigate config
       │     └─ No → Continue
       │
       ├─ Any CK test/example works?
       │     ├─ Yes → Issue specific to GEMM Universal
       │     └─ No → System-wide CK incompatibility
       │
       └─ Debug build with verbose output shows what?
             ├─ Specific error → Fix that error
             └─ Same generic fault → Contact AMD support
```

## Quick Diagnostic Commands

Run all diagnostics at once:

```bash
#!/bin/bash
echo "=== GPU Hardware ==="
rocm-smi --showproductname

echo -e "\n=== ROCm Version ==="
cat /opt/rocm/.info/version

echo -e "\n=== GPU Memory ==="
rocm-smi --showmeminfo vram | grep "Total"

echo -e "\n=== hipcc Version ==="
hipcc --version

echo -e "\n=== Test Small Size ==="
cd /workspace/rocm-libraries/projects/composablekernel/build
./bin/benchmark_gemm_universal_fp16_ccr_compv3_cshuffle_intrawave_False_False_False_False_64x64x64_2x2x1_16x16x32 \
  -m=512 -n=512 -k=512 -init=0 -warmup=1 -repeat=1 -verify=0 2>&1 | tail -10

echo -e "\n=== Recent dmesg ==="
dmesg | tail -20 | grep -i "gpu\|amd\|hip"
```

## Most Likely Causes (In Order)

1. **GPU Memory Exhaustion** (60% probability)
   - Fix: Use smaller problem size or different GPU

2. **ROCm/Driver Incompatibility with gfx950** (20%)
   - Fix: Update ROCm, check release notes, contact AMD

3. **Kernel Configuration Issue** (10%)
   - Fix: Try different tile sizes, pipelines, schedulers

4. **Build Configuration Problem** (5%)
   - Fix: Clean rebuild with verbose output

5. **Hardware Issue** (5%)
   - Fix: Test on different GPU, check hardware

## Contact Information

If all steps fail:
- AMD ROCm Support: https://github.com/ROCm/ROCm/issues
- ComposableKernel Issues: https://github.com/ROCm/composable_kernel/issues
- Include: GPU model, ROCm version, error messages, diagnostic output

---

**Start with Step 1-3. Most issues will be identified there.**
