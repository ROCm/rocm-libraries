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

⚠️ **IMPORTANT**: ATT (Advanced Thread Trace) requires the `rocprof-trace-decoder` library.
It does **not** ship with ROCm and is not in the ROCm apt repository — it is distributed
separately from <https://github.com/ROCm/rocprof-trace-decoder> (pick the release asset matching
the distro; it installs a single `librocprof-trace-decoder.so` into `/opt/rocm/lib`). Override the
location with `ROCPROF_TRACE_DECODER_LIB` if it lives elsewhere. If you cannot install it, use
**PMC (Performance Counter) profiling** instead (see Alternative: PMC Profiling below).

Capture rocprofv3 ATT traces from CK DSL kernels running on local GPU or remote Docker container,
then download the trace output for analysis.

## Quick path: one command

`tools/stage2_capture/capture_att_trace.py` does the whole of Steps 2-5 — preflights the decoder,
discovers the kernel name, runs the capture, and reports each decoded dispatch with the numbers
that say whether it is usable:

```bash
python3 tools/stage2_capture/capture_att_trace.py --output-dir ./att_out \
  -- python3 -m rocke.run_manifest out/kernel.hsaco out/manifest.json
```

Each invocation writes a fresh `capture-<trace-id>` generation below `att_out`
and prints the exact decoded dispatch folder to open. Completed, truncated, and
nonempty unfinalized generations remain separate and cannot satisfy a later
capture that matched no dispatch. A generation is removed automatically only
when the current attempt published nothing and `rmdir()` proves it is empty.

Pass `--kernel-regex` to skip the discovery pass. The manual steps below remain the reference for
remote/Docker captures and for anything the wrapper does not cover.

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
- CK DSL install path on remote (e.g. `<repo>/dnn-providers/hip-kernel-provider/rocke/platform/python`)

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

**NOTE**: `compile_kernel()` has no `debug=` parameter, but source mapping *is* available — set
`ROCKE_DEBUG_LOC=1` on the process that builds the kernel. See
[Debug Info in CK DSL](#debug-info-in-ck-dsl) below for what it does and why it is opt-in. Without
it there is no DWARF and the `Source` column stays empty.

**For ISA-level analysis** (which works either way), you can extract and disassemble the HSACO after
rocprof completes:
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

## Step 6: View the trace in WaveScope

A decoded `ui_output_*_dispatch_*` folder is exactly what the **WaveScope** viewer reads. It
gives a per-wave timeline over the ISA listing, dependency brackets from memory ops to the
`s_waitcnt` that waits on them, an occupancy heatmap, and rule-based bottleneck detection —
the interactive equivalent of `tools/stage4_analyze/parse_kernel_trace.py`.

Install the extension from the WaveScope releases page:

```bash
cursor --install-extension wavescope-<version>.vsix --force   # or: code --install-extension ...
```

On a remote session install it on the **remote** side — the extension reads the trace from the
remote filesystem and streams it into the webview, so nothing is copied to the client.

Then run **WaveScope: Open Trace Folder...** and pick the dispatch folder. The Source tab appears
only when the kernel was built with `ROCKE_DEBUG_LOC=1` (see
[Debug Info in CK DSL](#debug-info-in-ck-dsl) below); everything else works either way.

### Closing the loop with an agent

WaveScope carries a two-way annotation protocol, which is the reason to prefer it over reading
`code.json` by hand:

- An agent analyzing the trace writes **`annotations.json`** into the dispatch folder — bottleneck
  findings, each anchored to instruction indices, so the viewer overlays numbered flags on the
  timeline and you can verify a claim by looking at it rather than trusting it. `n`/`p` walk the
  findings in severity order.
- You reply with **`notes.json`**, authored by pressing `m` and marking the thing the agent missed
  (a block, a dragged time window, a dependency bracket, every match of a search). Notes tagged
  `constraint` or `rejected` are hard limits the agent may not violate or re-propose; `question`
  notes must be answered in its next pass.

Two files, one writer each: the agent owns `annotations.json` and rewrites it wholesale each
round, so notes must not share it. Instruction and wave/time anchors always work; the
source-line anchor needs the trace to have been captured with `ROCKE_DEBUG_LOC=1`.

---

## Output

After capture, report:

1. **Trace location**: Local path to the downloaded trace directory
2. **Kernel info**: Name, VGPR/AGPR counts, grid size, duration (from out_kernel_trace.csv)
3. **Source mapping**: % of instructions with source annotations (high with `ROCKE_DEBUG_LOC=1`, 0%
   without it — see [Debug Info in CK DSL](#debug-info-in-ck-dsl) below)
4. **Instruction count**: Total instructions in code.json
5. **Next step**: Open the folder in WaveScope (Step 6), or run `/kernel-trace-analysis` for a
   text-only bottleneck report

Example output:
```
Trace captured: ./trace_data/20260516_153000_conv_implicit_gemm/
  Kernel: conv_implicit_gemm_v4r1_nhwc_kc_gemmm_gemmn_gemmk_64x128x64
  arch_vgpr=104, accum_vgpr=128, SGPR=80
  Instructions: 2845, source-mapped: 0 (0%)   # 0% => rebuilt needed with ROCKE_DEBUG_LOC=1

Open in WaveScope, or run /kernel-trace-analysis to analyze bottlenecks.
```

### Reading `code.json`: totals, not averages

`Latency` (col 7) and `Stall` (col 8) are **hit-weighted totals summed over every execution**,
not per-execution averages. Divide by `Hit` (col 6) for a per-execution figure. Reading them as
averages inflates per-instruction cost by the hit count and yields stall figures larger than the
kernel's whole wall-clock, which is the single easiest way to misread this file. `Latency` is
inclusive of `Stall`, so a class's actual compute is `(latency - stall) * hits`.

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
| `rocprof-trace-decoder library path not found` | Install it from <https://github.com/ROCm/rocprof-trace-decoder>, or set `ROCPROF_TRACE_DECODER_LIB`. Failing that, **use PMC profiling** (see Alternative section) |
| `INVALID_SHADER_DATA` | aqlprofile/decoder version mismatch, update both |
| Empty ui_output_agent_* | kernel_include_regex didn't match -- re-check kernel name from Step 2 |
| No source mapping in code.json | The kernel was built without `ROCKE_DEBUG_LOC=1`, so there is no DWARF. Rebuild with it set and re-capture, or analyze with ISA disassembly / WaveScope's Trace tab |
| Stall cycles exceed kernel wall-clock | Cols 7/8 are hit-weighted totals, not averages -- divide by `Hit` |
| Trace truncated (missing instructions) | Increase `att_buffer_size` to `0xC000000` (192MB) |
| SSH timeout | Increase timeout, check host connectivity |
| `kernel_iteration_range` mismatch | Test runs fewer iterations than expected -- use `"[0, [1-2]]"` |
| `ModuleNotFoundError: rocke` | Set PYTHONPATH to CK DSL root: `export PYTHONPATH=/path/to/composablekernel/python` |

---

## CK DSL-Specific Notes

### Debug Info in CK DSL

**Source mapping is opt-in, via `ROCKE_DEBUG_LOC=1`.** Set it on the process that *builds*
the kernel — it is read when `IRBuilder` constructs the kernel, not at compile time:

```bash
ROCKE_DEBUG_LOC=1 python your_bench.py
```

```python
from rocke.helpers import compile_kernel

# No debug= parameter; the env var (or IRBuilder(capture_loc=True)) is the switch.
artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")
```

With it set, `IRBuilder` records the authoring Python call stack on every `Op.loc`, the
lowering turns each stack into a `DICompileUnit` / `DISubprogram` / `DILocation` chain, and
comgr's normal compile carries the resulting DWARF into the `.hsaco` — no `-g` needed, because
the metadata is in the IR rather than requested from a source file. The `Source` column of
`code.json` then names the Python line that emitted each instruction, and
`tools/wavescope/emit_inline_frames.py` recovers the *call stack* above that line from the
same DWARF.

Two properties worth knowing:

- **Off by default, and byte-identical when off.** Capturing a frame per op costs real time on
  sweeps that build thousands of kernels, and the metadata changes the emitted `.ll` bytes,
  which the IR goldens and the byte-identity gate both pin.
- **Backend-independent.** The location rides the serialized `ck.dsl.ir/v1` artifact as `@loc`,
  and both the Python lowerer and the C++ engine emit the same metadata from it, so
  `ROCKE_BACKEND=cpp` (the default when `rocke_engine` is installed) produces the same DWARF —
  `ROCKE_BACKEND=both` asserts exactly that.

Without the variable set there is no DWARF and the `Source` column is empty; analyze at ISA
level instead. `llvm-objdump` and `tools/stage3_extract_isa/extract_isa.py` give you the
disassembly, and WaveScope's Trace tab correlates ISA against the wave timeline without
needing source.

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
sys.path.insert(0, '<repo>/dnn-providers/hip-kernel-provider/rocke/platform/python')

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

print("Compiling kernel...")
kernel = build_implicit_gemm_conv(spec)
artifact = compile_kernel(kernel, isa="amdgcn-amd-amdhsa--gfx950")
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

- `tools/stage2_capture/capture_att_trace.py` - One-command capture (the Quick path above)
- `/kernel-trace-analysis` - Analyze captured ATT traces
- `src/stage3_extract_isa/extract_isa.py` - Extract ISA from CK DSL HSACO
- `.claude/OPTIMIZATION_RUNBOOK.md` Section 10 - Profiling methodology
