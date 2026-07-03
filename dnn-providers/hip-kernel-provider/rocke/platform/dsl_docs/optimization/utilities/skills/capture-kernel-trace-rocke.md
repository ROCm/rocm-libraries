---
name: capture-kernel-trace-rocke
description: >
  Capture GPU kernel ATT (Advanced Thread Trace) via rocprofv3 for CK DSL kernels.
  Discovers kernel names from compiled HSACO, configures input.yaml with target
  kernel_include_regex, runs rocprofv3 with debug info enabled, and downloads
  the latest ui_output_agent_* directory for analysis.
  Usage: /capture-kernel-trace-rocke <kernel_script.py> [kernel_name_pattern]
tools: Bash,Read,Write,Edit,Grep,Glob
---

# Capture Kernel Trace (CK DSL)

⚠️ **IMPORTANT**: ATT (Advanced Thread Trace) requires `rocprof-trace-decoder` library to be installed.
If the decoder is not available on your system, use **PMC (Performance Counter) profiling** instead (see Alternative: PMC Profiling section below).

Capture rocprofv3 ATT traces from CK DSL kernels running on local GPU or remote Docker container,
then download the trace output for analysis.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<kernel_script>` | Yes | Python script that compiles/runs CK DSL kernel, e.g. `bench_conv.py` |
| `[kernel_pattern]` | No | Kernel name regex. If omitted, discover via `--stats` first |

If no kernel script is provided, ask the user.

## Connection Info

**Check MEMORY.md for the user's current remote access configuration.** If not found, ask the user for:
- SSH host and user
- Docker container name (if applicable)
- CK DSL install path on remote (e.g. `<repo>/dnn-providers/hip-kernel-provider/rocKE/Python`)

SSH command pattern (adjust per environment):
```bash
ssh $USER@$HOST \
  "docker exec -e PYTHONPATH=<rocke_root> \
   $CONTAINER bash -c '<CMD>'"
```

For local execution (no SSH/Docker):
```bash
PYTHONPATH=/path/to/rocke <CMD>
```

---

## Workflow

```
Step 1: Deploy kernel script to remote container (if remote)
Step 2: Discover kernel names (if pattern not provided)
Step 3: Configure input.yaml with kernel_include_regex
Step 4: Run rocprofv3 -i input.yaml to collect ATT trace
Step 5: Find and download latest ui_output_agent_* to local
```

---

## Step 1: Deploy Kernel Script

If running on a remote container, copy the kernel script:

```bash
# Copy local file to container via SSH + docker cp
scp $KERNEL_SCRIPT $USER@$HOST:/tmp/
ssh $USER@$HOST "docker cp /tmp/$KERNEL_SCRIPT $CONTAINER:/tmp/"
```

If the kernel script is already on the remote (e.g., in CK DSL examples), skip this step.

---

## Step 2: Kernel Discovery (if no pattern provided)

Run rocprofv3 in stats mode to list kernel names:

```bash
# Remote
ssh $USER@$HOST \
  "docker exec -e PYTHONPATH=<rocke_root> \
   $CONTAINER bash -c \
   'cd /tmp && rocprofv3 --stats --kernel-trace -f csv -o /tmp/discover -- python $KERNEL_SCRIPT 2>&1'"

# Local
rocprofv3 --stats --kernel-trace -f csv -o /tmp/discover -- python $KERNEL_SCRIPT 2>&1
```

Parse output to find kernel names:

```bash
cat /tmp/discover_kernel_stats.csv
```

**CK DSL Kernel Naming**:
- Compiled kernels typically have names like: `conv_implicit_gemm_v4r1_nhwc_kc_gemmm_gemmn_gemmk_<config>`
- Look for mangled LLVM function names in the CSV
- May contain config details like tile sizes in the name

Present the kernel list and let the user pick, or auto-select the CK DSL kernel
(typically the longest name with config details).

---

## Step 3: Configure input.yaml

Create the input.yaml with the target `kernel_include_regex`:

```yaml
jobs:
   -
       kernel_include_regex: <KERNEL_PATTERN>
       kernel_iteration_range: "[1, [2-4]]"
       output_file: out
       output_directory: /tmp/kernel_trace_output
       output_format: [csv]
       truncate_kernels: true
       sys_trace: true
       advanced_thread_trace: true
       att_target_cu: 1
       att_shader_engine_mask: "0xf"
       att_simd_select: "0xf"
       att_buffer_size: "0x6000000"
```

Key configuration:
- `kernel_include_regex`: Exact name or regex from Step 2
- `kernel_iteration_range`: `"[1, [2-4]]"` skips warmup (iteration 0), traces iterations 2-4
- `att_target_cu: 1`: Single CU for manageable output
- `att_buffer_size: "0x6000000"`: 96MB per SE (increase to `0xC000000` if truncated)

---

## Step 4: Run rocprofv3 with ATT

**NOTE**: CK DSL does not expose a `debug=` parameter in `compile_kernel()`. Source mapping in ATT traces
depends on LLVM backend flags which are not directly controllable from CK DSL user code.

**For ISA-level analysis** (which always works), you can extract and disassemble the HSACO after rocprof completes:
```python
# Extract ISA from compiled HSACO (no debug info required)
# See src/stage3_extract_isa/extract_isa.py for automated extraction
```

Run rocprofv3:

```bash
# Remote
ssh $USER@$HOST \
  "docker exec -e PYTHONPATH=<rocke_root> \
   $CONTAINER bash -c \
   'cd /tmp && rm -rf /tmp/kernel_trace_output && rocprofv3 -i /tmp/input_trace.yaml -- python $KERNEL_SCRIPT 2>&1'"

# Local
PYTHONPATH=/path/to/rocke \
  rocprofv3 -i /tmp/input_trace.yaml -- python $KERNEL_SCRIPT 2>&1
```

Timeout: allow 3-5 minutes for JIT compilation + trace collection.

---

## Step 5: Download Trace Output

### 5.1 Find the latest ui_output_agent_* directory

```bash
# Remote
ssh $USER@$HOST \
  "docker exec $CONTAINER bash -c \
   'ls -td /tmp/kernel_trace_output/ui_output_agent_* 2>/dev/null | head -5'"

# Local
ls -td /tmp/kernel_trace_output/ui_output_agent_* 2>/dev/null | head -5
```

The output directories are named `ui_output_agent_<PID>_dispatch_<N>`. Pick the latest.

### 5.2 Download to local (remote only)

```bash
# Create local destination
LOCAL_TRACE_DIR=./trace_data/$(date +%Y%m%d_%H%M%S)_$KERNEL_SHORT_NAME
mkdir -p $LOCAL_TRACE_DIR

# Copy from container to host, then to local
UI_OUTPUT_DIR=<latest ui_output_agent_* path>

ssh $USER@$HOST "docker cp $CONTAINER:$UI_OUTPUT_DIR /tmp/ui_trace_download"
scp -r $USER@$HOST:/tmp/ui_trace_download/* $LOCAL_TRACE_DIR/
```

Also download supporting files:

```bash
# Kernel trace CSV (timing, VGPR info)
ssh $USER@$HOST "docker cp $CONTAINER:/tmp/kernel_trace_output/out_kernel_trace.csv /tmp/"
scp $USER@$HOST:/tmp/out_kernel_trace.csv $LOCAL_TRACE_DIR/
```

### 5.3 Verify download

```bash
ls -la $LOCAL_TRACE_DIR/
# Should contain: code.json, occupancy.json, filenames.json, wstates*.json, se*_*.json

# Quick validation
python3 -c "
import json, sys
with open('$LOCAL_TRACE_DIR/code.json') as f:
    data = json.load(f)
n = len(data.get('code', []))
has_src = sum(1 for i in data.get('code', []) if i[3])
print(f'Instructions: {n}, with source mapping: {has_src} ({100*has_src//max(n,1)}%)')
"
```

---

## Output

After capture, report:

1. **Trace location**: Local path to the downloaded trace directory
2. **Kernel info**: Name, VGPR/AGPR counts, grid size, duration (from out_kernel_trace.csv)
3. **Source mapping**: Whether debug info is present (% of instructions with source annotations)
4. **Instruction count**: Total instructions in code.json
5. **Next step**: Suggest running `/kernel-trace-analysis` on the downloaded trace for bottleneck analysis

Example output:
```
Trace captured: ./trace_data/20260516_153000_conv_implicit_gemm/
  Kernel: conv_implicit_gemm_v4r1_nhwc_kc_gemmm_gemmn_gemmk_64x128x64
  Duration: 182.7 us
  arch_vgpr=104, accum_vgpr=128, SGPR=80
  Instructions: 2845, source-mapped: 2103 (74%)

Run /kernel-trace-analysis to analyze bottlenecks.
```

---

## Alternative: PMC Profiling

If ATT is blocked due to missing `rocprof-trace-decoder`, use PMC (Performance Monitor Counters) instead:

```yaml
# pmc_config.yaml
jobs:
   -
       kernel_include_regex: <KERNEL_NAME>
       output_file: pmc_pass1
       output_directory: pmc_output
       output_format: [csv]
       pmc: true
       counters:
          - MfmaUtil
          - VALUBusy
          - MemUnitBusy
          - MemUnitStalled
          - ALUStalledByLDS
          - LDSBankConflict
          - MeanOccupancyPerActiveCU
   -
       kernel_include_regex: <KERNEL_NAME>
       output_file: pmc_pass2
       output_directory: pmc_output
       output_format: [csv]
       pmc: true
       counters:
          - FetchSize
          - WriteSize
          - VFetchInsts
          - VWriteInsts
```

Run:
```bash
rocprofv3 -i pmc_config.yaml -- python kernel.py
```

PMC gives high-level bottleneck categories (MFMA utilization, memory stalls, LDS conflicts) without instruction-level detail.

---

## Error Handling

| Error | Fix |
|-------|-----|
| `rocprof-trace-decoder library path not found` | **Use PMC profiling instead** (see Alternative section above) |
| `INVALID_SHADER_DATA` | aqlprofile/decoder version mismatch, update both |
| Empty ui_output_agent_* | kernel_include_regex didn't match -- re-check kernel name from Step 2 |
| No source mapping in code.json | CK DSL doesn't expose debug flags; use ISA disassembly instead |
| Trace truncated (missing instructions) | Increase `att_buffer_size` to `0xC000000` (192MB) |
| SSH timeout | Increase timeout, check host connectivity |
| `kernel_iteration_range` mismatch | Test runs fewer iterations than expected -- use `"[0, [1-2]]"` |
| `ModuleNotFoundError: rocke` | Set PYTHONPATH to CK DSL root: `export PYTHONPATH=/path/to/composablekernel/python` |

---

## CK DSL-Specific Notes

### Debug Info in CK DSL

Unlike FlyDSL's `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` environment variable, CK DSL controls
debug info at compile time:

```python
from rocke.helpers import compile_kernel

# WITHOUT debug info (default)
artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")

# WITH debug info (for ATT source mapping)
artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950", debug=True)
```

The `debug=True` flag passes `-g` to LLVM, generating DWARF debug information in the HSACO.
This enables `code.json` to contain source file:line annotations for each ISA instruction.

### Kernel Naming Convention

CK DSL kernel names include configuration details:
- Format: `<base_name>_<layout>_<variant>_<tile_config>_<pipeline>_<scheduler>`
- Example: `conv_implicit_gemm_v4r1_nhwc_kc_gemmm_gemmn_gemmk_64x128x64_mem_intrawave`
- The name is set via `ImplicitGemmConvSpec.name` parameter

Use the full kernel name (or regex matching it) in `kernel_include_regex`.

### Running CK DSL Kernels

CK DSL kernels can be run via:

1. **run_manifest API** (recommended for benchmarking):
```python
from rocke.run_manifest import run_manifest
summary = run_manifest(manifest_path, hsaco_path, verify=False)
```

2. **Direct Runtime API** (for custom control):
```python
from rocke.runtime.hip_module import Runtime
rt = Runtime()
mod = rt.module_load_data(artifact.hsaco)
func = mod.get_function(artifact.kernel_name)
func.launch(grid=..., block=..., args=...)
```

For profiling, ensure the kernel is actually launched (not just compiled).

### Example Kernel Script

```python
#!/usr/bin/env python3
"""CK DSL Conv2D for profiling with rocprofv3."""
import sys
from pathlib import Path
sys.path.insert(0, '<repo>/dnn-providers/hip-kernel-provider/rocKE/Python')

from rocke.helpers import compile_kernel, make_conv_manifest, write_artifact
from rocke.instances.conv_implicit_gemm import (
    ConvProblem, ImplicitGemmConvSpec, build_implicit_gemm_conv
)
from rocke.run_manifest import run_manifest
import tempfile

# Problem definition
problem = ConvProblem(
    N=16, Hi=56, Wi=56, C=512, K=512, Y=3, X=3,
    sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
)

# Kernel config
spec = ImplicitGemmConvSpec(
    problem=problem,
    name="conv_profile",  # Kernel name
    tile_m=64, tile_n=128, tile_k=64,
    warp_m=2, warp_n=2,
    warp_tile_m=32, warp_tile_n=32, warp_tile_k=16,
    pipeline="mem", epilogue="cshuffle"
)

# Compile with debug info for ATT source mapping
print("Compiling kernel with debug info...")
kernel = build_implicit_gemm_conv(spec)
artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950", debug=True)
print(f"Kernel name: {artifact.kernel_name}")

# Run kernel
with tempfile.TemporaryDirectory() as tmpdir:
    manifest = make_conv_manifest(
        artifact=artifact, block_m=spec.tile_m, block_n=spec.tile_n, block_k=spec.tile_k,
        threads_per_block=spec.block_size,
        conv=[problem.N, problem.Hi, problem.Wi, problem.C, problem.K,
              problem.R, problem.S, problem.sH, problem.sW, problem.pH, problem.pW,
              problem.dH, problem.dW],
        groups=1, cpg=problem.C, kpg=problem.K,
        conv_layout="implicit_gemm", grid_order="NM",
        warmup_iters=2, timed_iters=5
    )

    paths = write_artifact(artifact, Path(tmpdir), manifest)
    summary = run_manifest(paths['manifest'], paths['hsaco'], verify=False)
    print(f"TFLOPS: {summary.tflops:.2f}")
```

Save this as `bench_conv_profile.py` and use it with rocprofv3.

---

## See Also

- `/kernel-trace-analysis` - Analyze captured ATT traces
- `src/stage3_extract_isa/extract_isa.py` - Extract ISA from CK DSL HSACO
- `.claude/OPTIMIZATION_RUNBOOK.md` Section 10 - Profiling methodology
