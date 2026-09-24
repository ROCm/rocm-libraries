---
name: capture-kernel-trace-rocke
description: >
  Capture WaveScope evidence for rocKE kernels with ATT instruction traces,
  PMC counter bundles, or both. ATT locates source-level stalls; PMC supplies
  dispatch-wide bottleneck rules and repeatable before/after verdicts.
  Usage: /capture-kernel-trace-rocke <kernel_script.py> [kernel_name_pattern]
tools: Bash,Read,Write,Edit,Grep,Glob
---

# Capture WaveScope Evidence (rocKE)

ATT and PMC answer different questions and can be used independently or together:

- **ATT** shows where and why waves stall, down to ISA and Python source locations.
- **PMC** measures dispatch-wide LDS, L2, VALU and matrix activity, then compares
  repeated runs against a noise floor.
- **Combined** is the preferred optimization loop: ATT identifies what to change;
  PMC and the perf verdict establish whether the change helped.

ATT requires `rocprof-trace-decoder`, distributed separately from
<https://github.com/ROCm/rocprof-trace-decoder>. Install the release matching the
distro into `/opt/rocm/lib`, or set `ROCPROF_TRACE_DECODER_LIB`. PMC capture does
not require that decoder and remains available when ATT cannot run.

## Quick path: choose an evidence mode

Run these commands from `dnn-providers/hip-kernel-provider/rocke/platform/`.

**ATT only — instruction and source diagnosis:**

```bash
python3 dsl_docs/optimization/utilities/tools/wavescope/capture_wavescope_trace.py \
  --output-dir ./att-out --kernel-regex '<kernel-name>' -- python3 bench.py
```

**PMC only — counter diagnosis and repeatable verification:**

```bash
python3 dsl_docs/optimization/utilities/tools/wavescope/capture_wavescope_pmc.py \
  --output-dir ./pmc-out --arch '<arch>' --op '<operation>' \
  --shape '<shape-json>' --kernel-name '<kernel-name>' \
  --match-kernel '<kernel-name>' --repeats 3 --warmup 5 --per-dispatch \
  -- python3 bench.py
```

**Combined — recommended for optimization:** run ATT first and use its reported
dispatch folder as `TRACE_DIR`. Put the PMC bundle beside the ATT capture tree and
run PMC with `--trace-dir "$TRACE_DIR"`. The utility copies one successful
repeat's replay CSVs beside `code.json`, so WaveScope discovers and merges them
when it opens that one folder. Keeping the bundle outside `TRACE_DIR` prevents the
browser folder picker from recursively loading retained repeats a second time.
The ATT and PMC captures are separate executions and remain `UNBOUND`; matching
identifiers are evidence to check, not proof that the workload or binary was
identical.

Each ATT invocation writes a fresh `capture-<trace-id>` generation. A combined
capture keeps the complete PMC bundle and ATT generation under one common run
root, while publishing only WaveScope-readable CSV sidecars in the dispatch folder.
The manual ATT steps below remain the reference for remote/Docker captures and for
anything the wrapper does not cover.

## Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `<kernel_script>` | Yes | Python script that compiles/runs a rocKE kernel, e.g. `bench_conv.py` |
| `[kernel_pattern]` | No | Kernel name regex. If omitted, discover via `--stats` first |

If no kernel script is provided, ask the user.

## Connection Info

**Check MEMORY.md for the user's current remote access configuration.** If not found, ask the user for:
- SSH host and user
- Docker container name (if applicable)
- rocKE Python package path on remote (for example, `<repo>/dnn-providers/hip-kernel-provider/rocke/platform/python`)

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

```text
Shared setup: deploy the launcher and identify the target kernel.
ATT branch: capture/decode the dispatch, then inspect its ISA/source timeline.
PMC branch: capture repeatable counters and the before/after perf verdict.
Combined branch: use ATT to choose an edit, PMC to validate it, then repeat.
```

Steps 1-6 below describe the manual ATT branch. See
[PMC profiling and the combined pipeline](#pmc-profiling-and-the-combined-pipeline)
for the counter branch.

---

## Step 1: Deploy Kernel Script

If running on a remote container, copy the kernel script:

```bash
# Copy local file to container via SSH + docker cp
scp $KERNEL_SCRIPT $USER@$HOST:/tmp/
ssh $USER@$HOST "docker cp /tmp/$KERNEL_SCRIPT $CONTAINER:/tmp/"
```

If the kernel script is already on the remote (for example, in rocKE examples), skip this step.

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

**rocKE Kernel Naming**:
- Compiled kernels typically have names like: `conv_implicit_gemm_v4r1_nhwc_kc_gemmm_gemmn_gemmk_<config>`
- Look for mangled LLVM function names in the CSV
- Names may contain configuration details such as tile sizes

Present the kernel list and let the user pick, or auto-select the rocKE kernel
(typically the longest name with configuration details).

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
[Debug Info in rocKE](#debug-info-in-rocke) below for what it does and why it is opt-in. Without
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
[Debug Info in rocKE](#debug-info-in-rocke) below); everything else works either way.

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
   without it — see [Debug Info in rocKE](#debug-info-in-rocke) below)
4. **Instruction count**: Total instructions in code.json
5. **Next step**: Open the folder in WaveScope (Step 6), or run `/kernel-trace-analysis` for a
   text-only bottleneck report

For a PMC or combined run, also report:

1. **Bundle location** and per-sample profiler status
2. **Counter groups** and the recommended successful-sample CSV path
3. **Derived diagnostics** (`l2_hit_rate`, `lds_bank_conflict_rate`,
   `valu_utilization`, `matrix_share`) when their inputs are usable
4. **Verification verdict** (`no_baseline`, `improved`, `within_noise`, or
   `regressed`) and its noise floor
5. **Association**: `UNBOUND` unless ATT/PMC workload, GPU and binary identity were
   independently established

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
inclusive of `Stall`, so a class's actual compute is `latency - stall`.

---

## PMC Profiling and the Combined Pipeline

PMC is not merely an ATT fallback. It provides dispatch-wide evidence ATT does not:
repeat-to-repeat spread, before/after comparison, LDS conflict rate, L2 hit rate,
VALU utilization, and the matrix-work guard. Use PMC alone when instruction tracing
is unavailable or unnecessary; use ATT and PMC together when optimizing a kernel.

Do not hand-write a rocprofv3 counter YAML. Counter names, availability and hardware
block limits differ by architecture. The shared utility probes the installed
profiler, normalizes names, and groups ratio inputs into coherent replay passes:

```bash
export ARCH='<arch>'
export OP='<operation>'
export SHAPE_JSON='<shape-json>'
export KERNEL_NAME='<kernel-name>'
export TRACE_DIR='<reported-att-dispatch-folder>'
export RUN_DIR="$(dirname "$TRACE_DIR")"

python3 dsl_docs/optimization/utilities/tools/wavescope/capture_wavescope_pmc.py \
  --output-dir "$RUN_DIR/pmc_bundle" --trace-dir "$TRACE_DIR" \
  --store-history --cache /tmp/rocke-perf-history \
  --arch "$ARCH" --op "$OP" --shape "$SHAPE_JSON" \
  --kernel-name "$KERNEL_NAME" --match-kernel "$KERNEL_NAME" \
  --repeats 3 --warmup 5 --per-dispatch \
  -- python3 kernel.py
```

On the tested CDNA selection, all inputs for WaveScope's LDS, L2 and VALU PMC rules
land in `pmc_1`; `pmc_2` contains LDS instruction and wait counts. RDNA uses one
pass on the tested gfx1201 system, but several diagnostic inputs returned zero and
the corresponding rules were explicitly skipped. Always use the groups and sample
status recorded in `manifest.json` rather than assuming a fixed layout.

For the complete agent optimization loop:

1. **Capture ATT.** Run the ATT quick path with the baseline launcher, then set
   `TRACE_DIR` to the reported `ui_output_*_dispatch_*` folder.
2. **Establish a PMC baseline.** Run the PMC command above. It stores the complete
   bundle beside the ATT dispatch tree and publishes one successful repeat's
   replay CSVs beside `code.json`. The first stored run reports `no_baseline`.
3. **Locate the bottleneck.** Open `TRACE_DIR` in WaveScope. It automatically
   merges the top-level `*_counter_collection.csv` sidecars with ATT evidence;
   no manual CSV upload is needed.
4. **Make one kernel change.** Keep launch shape, GPU and binary inputs controlled.
5. **Verify the change.** Run PMC into a new output directory with the same identity
   and history cache. Accept `improved`; treat `within_noise` as no proven change;
   stop on `regressed` (the CLI exits 1).
6. **Re-capture ATT when needed.** To inspect the changed kernel, create a new ATT
   folder and publish that run's PMC sidecars into it. Do not infer per-instruction
   attribution from a dispatch-wide PMC counter.

The complete PMC bundle remains beside the ATT capture tree with its manifest,
JSON, hash inventory and all repeats. Only one successful repeat's replay CSVs are
copied to the dispatch-folder top level, where WaveScope discovers and merges them.
Existing sidecars are never overwritten.

PMC capture does not need WaveScope or the ATT decoder. Opening colocated PMC and
ATT artifacts does not bind their separate executions; verify workload, GPU, shape
and binary identity before correlating them.

## Error Handling

| Error | Fix |
|-------|-----|
| `rocprof-trace-decoder library path not found` | Install it or set `ROCPROF_TRACE_DECODER_LIB`; use the PMC-only mode when ATT is unavailable |
| `INVALID_SHADER_DATA` | aqlprofile/decoder version mismatch, update both |
| Empty ui_output_agent_* | kernel_include_regex didn't match -- re-check kernel name from Step 2 |
| No source mapping in code.json | The kernel was built without `ROCKE_DEBUG_LOC=1`, so there is no DWARF. Rebuild with it set and re-capture, or analyze with ISA disassembly / WaveScope's Trace tab |
| Stall cycles exceed kernel wall-clock | Cols 7/8 are hit-weighted totals, not averages -- divide by `Hit` |
| Trace truncated (missing instructions) | Increase `att_buffer_size` to `0xC000000` (192MB) |
| SSH timeout | Increase timeout, check host connectivity |
| `kernel_iteration_range` mismatch | Test runs fewer iterations than expected -- use `"[0, [1-2]]"` |
| `ModuleNotFoundError: rocke` | Set `PYTHONPATH` to the rocKE package root: `export PYTHONPATH=<repo>/dnn-providers/hip-kernel-provider/rocke/platform/python` |

---

## rocKE-Specific Notes

### Debug Info in rocKE

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

rocKE kernel names include configuration details:
- Format: `<base_name>_<layout>_<variant>_<tile_config>_<pipeline>_<scheduler>`
- Example: `conv_implicit_gemm_v4r1_nhwc_kc_gemmm_gemmn_gemmk_64x128x64_mem_intrawave`
- The name is set via `ImplicitGemmConvSpec.name` parameter

Use the full kernel name (or regex matching it) in `kernel_include_regex`.

### Running rocKE Kernels

rocKE kernels can be run via:

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
"""rocKE Conv2D for profiling with rocprofv3."""
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

- `tools/wavescope/capture_wavescope_trace.py` - Source-correlated ATT capture
- `tools/wavescope/capture_wavescope_pmc.py` - PMC capture, artifact export and verification
- `/kernel-trace-analysis` - Analyze captured ATT traces
- `src/stage3_extract_isa/extract_isa.py` - Extract ISA from a rocKE HSACO
- `.claude/OPTIMIZATION_RUNBOOK.md` Section 10 - Profiling methodology
