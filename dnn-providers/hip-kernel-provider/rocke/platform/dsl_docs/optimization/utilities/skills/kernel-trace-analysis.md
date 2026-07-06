---
name: kernel-trace-analysis
description: >
  Profile GPU kernels using rocprofv3 to collect ATT instruction-level traces, then
  analyze the trace data using hotspot_analyzer.py to identify top-K stall hotspots
  (VMEM-load, VMEM-wait, LDS/SMEM-wait, barrier, MFMA stalls) mapped back to source
  lines, and produce an actionable optimization plan.
  Usage: /kernel-trace-analysis <cmd>
  Can also analyze an existing dispatch dir directly: /kernel-trace-analysis --dir <path>
tools: Read,Edit,Bash,Grep,Glob,Agent,Write
note: All analysis is done programmatically via hotspot_analyzer.py + code.json. Do NOT use GUI tools.
---

# Kernel Trace Analysis

Profile and analyze GPU kernel ATT traces to identify stall hotspots and produce
an optimization plan.

## Arguments

| Argument | Description |
|----------|-------------|
| `<CMD>` | Command to profile. Example: `python bench_pa.py --batch 32` |
| `--dir <path>` | Skip collection; analyze existing `ui_output_agent_*_dispatch_*` directory |
| `--topk N` | Show top-N hotspots (default: 15) |

---

## Hotspot Analyzer Script

The hotspot analyzer is located at `scripts/hotspot_analyzer.py`.
It reads a `ui_output_agent_*_dispatch_*` directory and reports top-K stall hotspots.

---

## Workflow

⚠️ **PREREQUISITE CHECK**: ATT analysis requires `rocprof-trace-decoder` library. If not installed, use **Mode C: PMC Analysis** instead.

Before collecting ATT for an optimization experiment, run the
`/isa-inspection-rocke` checklist when the change is expected to alter matrix
opcodes, packed dtype handling, LDS handoffs, wait/barrier placement, or
occupancy. ISA inspection catches wrong-codegen and hidden conversion/packing
taxes before you spend time interpreting stall traces.

### Mode A: Analyze existing dispatch directory

If the user provides `--dir <path>` or already has a `ui_output_agent_*_dispatch_*` directory:

```bash
# Write hotspot_analyzer.py (see above), then:
python /tmp/hotspot_analyzer.py <dispatch_dir> --topk 15 --mode both
python /tmp/hotspot_analyzer.py <dispatch_dir> --topk 5 --mode src --detail --context 4
```

Skip to **Step 4: Interpret Results**.

---

### Mode B: Full collection workflow

#### Step 1: Kernel Discovery

```bash
touch /tmp/trace_ts
rocprofv3 --stats --kernel-trace -f csv -- <CMD> 2>&1
find . -maxdepth 3 -name "*stats*" -newer /tmp/trace_ts -type f 2>/dev/null
```

Parse the stats CSV and present a kernel table:

| Rank | Kernel Name | Calls | Total (us) | Avg (us) | % GPU Time |
|------|-------------|-------|------------|----------|------------|

Ask the user which kernel to trace if not obvious.

**Prefer `results.db`** if available — use sqlite3 for structured queries:
```bash
sqlite3 results.db "
SELECT ks.KernelName, COUNT(*) calls,
       ROUND(AVG(kd.end-kd.start)/1000.0,1) avg_us
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id=ks.id
GROUP BY ks.KernelName ORDER BY avg_us DESC LIMIT 20;"
```

#### Step 2: Configure input.yaml

```bash
cp ~/Documents/input.yaml /tmp/trace_input.yaml
```

Edit `/tmp/trace_input.yaml`:

```yaml
jobs:
   -
       kernel_include_regex: <KERNEL_NAME_PATTERN>
       kernel_iteration_range: "[1, [3-4]]"
       output_file: out
       output_directory: kernel_trace_output
       output_format: [csv]
       truncate_kernels: true
       sys_trace: true
       advanced_thread_trace: true
       att_target_cu: 1
       att_shader_engine_mask: "0xf"
       att_simd_select: "0xf"
       att_buffer_size: "0x6000000"
```

Key notes:
- `kernel_iteration_range`: `"[1, [3-4]]"` skips warmup, traces dispatches 3-4
- `att_buffer_size`: 96MB per SE; increase to `"0xC000000"` if truncated
- `att_target_cu: 1`: single CU keeps output manageable

#### Step 3: Collect ATT Trace

```bash
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 rocprofv3 -i /tmp/trace_input.yaml -- <CMD> 2>&1
find . -type d -name "ui_output_agent_*" -newer /tmp/trace_ts 2>/dev/null
```

If `rocprof-trace-decoder` library is missing:
```bash
wget -q https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.6/rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh
chmod +x rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh
./rocprof-trace-decoder-manylinux-2.28-0.1.6-Linux.sh --skip-license --prefix=/tmp/rtd-install
find /tmp/rtd-install -name '*.so*' -exec cp -a {} /opt/rocm/lib/ \;
ldconfig
```

**Output structure:**
```
ui_output_agent_<PID>_dispatch_<N>/
├── code.json          ← PRIMARY: per-instruction stall/cycle data
├── snapshots.json     ← source file path mapping (virtual → local filename)
├── source_0_*.py      ← embedded source files
├── filenames.json     ← wave file index
├── occupancy.json     ← occupancy timeline
└── se*_sm*_sl*_wv*.json  ← per-wave raw traces
```

---

## Step 4: Run hotspot_analyzer.py

Write the script (see above), then run:

```bash
# Full report
python /tmp/hotspot_analyzer.py <dispatch_dir> --topk 15 --mode both

# Source-level with code context (best for optimization)
python /tmp/hotspot_analyzer.py <dispatch_dir> --topk 5 --mode src --detail --context 4

# ASM-only for instruction-level detail
python /tmp/hotspot_analyzer.py <dispatch_dir> --mode asm --topk 20
```

---

## Step 5: Interpret Results

### code.json field reference

Each row in `code["code"]` is:
```
[asm, _, pc_index, source_loc, _, pc_addr, exec_count, total_cycles, stall_cycles, issue_cycles]
  0   1     2          3       4     5          6            7              8             9
```

- **col[8] `stall_cycles`**: cycles the instruction was blocked from issuing — **primary hotspot metric**
- **col[7] `total_cycles`**: total cycles charged to this instruction across all waves
- **col[3] `source_loc`**: `"/path/to/file.py:LINE"` — virtual path resolved via `snapshots.json`
- **col[6] `exec_count`**: number of wave-threads that executed this instruction

### snapshots.json: resolving source paths

`snapshots.json` encodes a nested dict tree mapping virtual paths to local filenames:
```json
{"/": {"FlyDSL": {"kernels": {"pa_decode_sw_fp8_ps.py": "source_0_pa_decode_sw_fp8_ps.py"}}}}
```
Flatten recursively: `/FlyDSL/kernels/pa_decode_sw_fp8_ps.py` → `source_0_pa_decode_sw_fp8_ps.py`

### Stall type classification

| Type | Instructions | Root Cause |
|------|-------------|------------|
| `VMEM-load` | `buffer_load_*`, `global_load_*` | Load itself stalled (VMEM queue full or back-pressure from no compute to hide behind) |
| `VMEM-wait` | `s_waitcnt vmcnt(N)` | Waiting for outstanding VMEM loads to complete |
| `LDS/SMEM-wait` | `s_waitcnt lgkmcnt(N)` | Waiting for LDS or SMEM ops |
| `barrier` | `s_barrier` | Cross-wave sync — slowest wave dominates |
| `MFMA/FMA` | `v_mfma_*` | MFMA dependency chain (RAW hazard) |
| `LDS` | `ds_read_*`, `ds_write_*` | LDS access latency |

### Common hotspot patterns

#### Pattern 1: V/K loads inside MFMA loop → very high stall rate (80–95%)

```python
# BAD: load and MFMA alternate — only 1 MFMA of hiding time
for k_step in range_constexpr(QKHELOOP * 2):
    if k_step % 2 == 0:
        v_data = buffer_ops.buffer_load(...)   # stall_rate ~92%
    acc = rocdl.mfma_f32_16x16x32_fp8_fp8(...)

# GOOD: batch all loads before the MFMA loop
for td in range_constexpr(TLOOP):
    v_prefetch[td] = [buffer_ops.buffer_load(...) for _ in range_constexpr(QKHELOOP)]

for td in range_constexpr(TLOOP):
    for k_step in range_constexpr(QKHELOOP * 2):
        acc = rocdl.mfma_f32_16x16x32_fp8_fp8(...)   # entire QK MFMA hides VMEM latency
    v_results[td] = v_prefetch[td]   # already in registers
```

#### Pattern 2: Sequential loads with no compute → VMEM queue saturation

```python
# BAD: all loads back-to-back, no compute interleaved
for td in range_constexpr(TLOOP):
    for qkhe in range_constexpr(QKHELOOP):
        k4 = buffer_ops.buffer_load(k_rsrc, ka_dw, ...)   # queue fills up

# GOOD: prefetch next tile's K loads during current tile's MFMA computation
```

#### Pattern 3: LDS prob reads immediately before PV MFMA → lgkmcnt stall

```python
# BAD: LDS reads and MFMA in same loop
for vhe in ...:
    for vt in ...:
        p_i64 = lds_read(...)    # issued here
        tmp = mfma(v_i64, p_i64, ...)   # immediately consumed → lgkmcnt stall

# GOOD: batch all LDS reads first, then all MFMAs
for vhe in ...:
    for vt in ...:
        p_i64s.append(lds_read(...))    # all LDS reads issued first

for vhe in ...:
    for vt in ...:
        tmp = mfma(v_i64s[...], p_i64s[...], ...)   # LDS data already ready
```

#### Pattern 4: Scale loads too close to usage

```python
# BAD: scale load and usage separated by only TLOOP MFMAs
for td in range_constexpr(TLOOP):
    k_scale = buffer_ops.buffer_load(ks_rsrc, ...)   # issued here
# ... small compute gap ...
    result = acc * k_scale   # used too soon → stall

# GOOD: issue scale loads at the very beginning of the block,
# before K loads, to maximise latency hiding distance
```

#### Pattern 5: Hotspot attributed to kernel entry line

When `@flyc.kernel` / kernel decorator line appears as the top hotspot with a mix of
VMEM-wait + barrier stall types — this is a **debug info aggregation artifact**.
MLIR/compiler-generated instructions (address arithmetic, cndmask, prologue setup) map
to the outermost scope line. Ignore this line; focus on lines with explicit user ops.

### Register pressure check (architecture-aware)

`hotspot_analyzer.py` auto-detects the GPU architecture from ISA instruction patterns
and prints occupancy using the correct formula. The key difference:

| Property | CDNA3 (gfx942) | CDNA4 (gfx950) |
|---|---|---|
| VGPR pools | 256 arch + 256 accum, **separate** | 256 arch + 256 accum, **combined 512 pool, flexibly split** |
| Occupancy formula | `256 / max(arch_alloc, accum_alloc)` | `512 / (arch_alloc + accum_alloc)` |
| Alloc granularity | 8 VGPRs | 8 VGPRs |
| LDS size | 64 KB | 160 KB |
| LDS alloc block | 256 bytes | 1280 bytes |
| VMCNT width | 6 bits (max 63 in-flight) | 6 bits (max 63 in-flight) |
| LGKMCNT width | 4 bits (max 15 in-flight) | 4 bits (max 15 in-flight) |

**Auto-detection**: gfx950-specific instructions (`v_mfma_scale_f32_*`, `v_mfma_f32_16x16x128_*`,
`v_mfma_f32_32x32x64_*`) indicate CDNA4. Absence indicates CDNA3.

```bash
sqlite3 results.db "
SELECT ks.KernelName, ki.arch_vgpr_count, ki.accum_vgpr_count, ki.lds_size
FROM rocpd_kernel_dispatch kd
JOIN rocpd_info_kernel_symbol ks ON kd.kernel_symbol_id=ks.id
JOIN rocpd_info_kernel ki ON kd.kernel_id=ki.id LIMIT 5;"
```

**Warning**: `maxnreg` forcing `accum_vgpr=0` doubles occupancy but causes MFMA spills through
arch_vgpr — measured 4.5x GPU slowdown. Do not use `maxnreg` for MFMA-heavy kernels.

### MFMA latency reference (cycles = pipeline depth)

| Instruction | Variant | Cycles | Notes |
|---|---|---|---|
| `v_mfma_f32_*_f16` / `_bf16` | 16x16x16 | 16 | |
| `v_mfma_f32_*_f16` / `_bf16` | 32x32x8 | 32 | |
| `v_mfma_f32_*_fp8_fp8` | 16x16x32 | 16 | CDNA3+CDNA4 |
| `v_mfma_f32_*_fp8_fp8` | 32x32x16 | 32 | CDNA3+CDNA4 |
| `v_mfma_f32_16x16x128_f8f6f4` | 16x16x128 | 16 or 32 | CDNA4 only; 32 if either A or B is FP8 |
| `v_mfma_f32_32x32x64_f8f6f4` | 32x32x64 | 32 or 64 | CDNA4 only; 64 if either A or B is FP8 |
| `v_mfma_scale_f32_16x16x128_f8f6f4` | 16x16x128 | 16 or 32 | CDNA4 only; with block exponent scaling |
| `v_mfma_scale_f32_32x32x64_f8f6f4` | 32x32x64 | 32 or 64 | CDNA4 only; with block exponent scaling |
| `v_mfma_f32_*_f32` | 16x16x4 | 32 | |
| `v_mfma_f32_*_f32` | 32x32x2 | 64 | |
| `v_mfma_f64_16x16x4_f64` | 16x16x4 | 64 | |

### MFMA dependency NOPs (CDNA4, from ISA reference Table 38)

These are the minimum independent instructions (or s_nop counts) required between
MFMA result production and consumption. The values vary by MFMA variant:

| Dependency pattern | Required waits | Comment |
|---|---|---|
| XDL write -> same XDL read SrcC (accumulate, exact same vDst) | 0-2 | Forwarding path; back-to-back accumulation OK |
| XDL write -> VALU/VM/LDS/FLAT read result (RAW) | 5, 8, 12, or 20 | No forwarding; must wait for MFMA commit to VGPR |
| XDL write -> MFMA read as SrcA or SrcB | 5, 8, 12, or 20 | No forwarding path |
| Non-DLops VALU write -> MFMA read | 2 | No 4/8 cycle forwarding path |
| VALU writes SGPR -> VMEM reads that SGPR | 5 | **HW does NOT check this** — user must add waits |
| V_CMPX* writes EXEC -> V_MFMA* | 4 | No EXEC forwarding with MFMA |

Wait counts for "5, 8, 12, or 20" depend on MFMA variant:
- 5 waits: 16x16 4-block variants (8 cycle MFMAs)
- 8 waits: 16x16x16 F16/BF16 etc. (16 cycle MFMAs)
- 12 waits: 32x32x8, 16x16x4 F32, etc. (32 cycle MFMAs)
- 20 waits: 32x32x4 F32, 32x32x2 F32 (64 cycle MFMAs)

---

## Step 6: Optimization Plan

After running `hotspot_analyzer.py --detail`, produce a prioritized plan:

```
## Stall Summary
- Total stalls: X cycles (Y% of kernel)
- Top stall type: VMEM-load (Z%)

## Hotspot Analysis

### #1 :LINE  stall=XK (N%)  VMEM-load  stall_rate=92%
Root cause: buffer_load inside QK MFMA loop — only 1 MFMA of hiding time.
Fix: Move all V loads before the QK MFMA loop.
Estimated gain: ~20% kernel cycle reduction.

### #2 :LINE  stall=XK (N%)  VMEM-load  stall_rate=80%
Root cause: K loads sequential with no compute interleaved.
Fix: Prefetch next tile's K during current tile's MFMA (double-buffer pattern).
See /prefetch-data-load skill.

### #3 ...

## Priority Order
1. [HIGH]  Fix V-load position (24% of all stalls, easy refactor)
2. [HIGH]  K-load cross-tile prefetch (8% of stalls, needs _process_block restructure)
3. [MED]   Move scale loads earlier (8% of stalls, trivial move)
4. [LOW]   Batch LDS reads before PV MFMA (4% of stalls, loop split)
```

---

## Error Handling

| Error | Fix |
|-------|-----|
| `rocprof-trace-decoder library path not found` | Install decoder .so (see Step 3) |
| Trace output empty | Check `kernel_include_regex` matches exactly |
| Trace truncated | Increase `att_buffer_size` to `"0xC000000"` |
| `kernel_iteration_range` mismatch | Adjust range; try `"[0, [1-2]]"` |
| `INVALID_SHADER_DATA` | aqlprofile/decoder version mismatch — update both |
| Source loc all `""` | Set `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1`; check `-g` flag in compile pipeline |
| Top hotspot is kernel decorator line | Debug info artifact — skip it, focus on op lines |
| `--att` flag error | `--att` is boolean, no value; use `-i input.yaml` for full config |
