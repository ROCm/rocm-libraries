---
name: profile-ck-kernel
description: Use when profiling CK Tile kernel performance, collecting ATT thread traces, analyzing PMC hardware counters, dumping annotated ISA assembly, or comparing optimization results. Triggers on requests for profiling, performance analysis, ATT, PMC, bottleneck analysis, bank conflict analysis, or A/B optimization comparison of ck_tile kernels.
---

# Profile CK Kernel

Automates full-stack profiling of CK Tile kernels: benchmark, ATT thread trace, PMC hardware counters, and annotated ISA assembly generation.

## Confirmation Gate

Before executing, **always confirm with the user**:
> "This will run a full profiling pass on `<binary>` (ATT trace + PMC counters + annotated assembly). This takes ~15-30 min and requires exclusive GPU access. Proceed?"

If the user declines, abort. Do not skip this gate.

## Pre-Checks

Run these before any profiling work:

```bash
# 1. GPU occupancy — abort if any GPU >10%
rocm-smi --showuse

# 2. GPU arch
rocminfo | grep -oE 'gfx[0-9a-z]+' | head -1

# 3. Binary existence
ls build/bin/<target_binary>
```

**Binary arguments resolution** (in priority order):
1. User explicitly provided → use directly
2. Check memory for previously recorded arguments for this kernel
3. Explore codebase: read the example's `main.cpp`, `script/benchmark_*.sh`, `script/smoke_test*.sh` to discover expected args
4. If discovered via exploration, **save to memory** for future use
5. Only ask user as last resort

## Parallel Execution Structure

```
                  +-- Agent A: GPU work (serial) --------+
Confirmed? --yes--+                                      +--> Merge --> Report
                  +-- Agent B: CPU work (parallel) ------+
```

Both agents inherit the parent session's model. Use `superpowers:dispatching-parallel-agents` to launch them.

**Agent A and B run concurrently.** Within Agent A, GPU tasks are serial (ATT then PMC) to avoid interference.

---

## Agent A: GPU Profiling (Serial)

### A1. Benchmark

```bash
# -v=0 skips correctness verification (profiling only)
./<binary> <args> -v=0
```

Capture kernel duration, TFLOPS/bandwidth from stdout. Include `rocm-smi --showuse` output in report.

### A2. ATT Trace + RCV Package

```bash
# CRITICAL: Do NOT use -o flag (breaks RCV directory structure)
rocprofv3 --att --att-activity 8 --att-target-cu 1 --att-buffer-size 0x10000000 \
  -- ./<binary> <args> -v=0

# Package for RCV
tar czf <name>_att.tar.gz ui_output_agent_*_dispatch_*/
```

Output: `<name>_att.tar.gz` — open directly in RCV desktop app.

### A3. PMC Counters

```bash
# Ensure ck-rocprof is set up
ck-rocprof status || ck-rocprof setup

# Collect PMC data (multiple passes, takes time)
ck-rocprof run <name> ./<binary> <args>

# Analyze key blocks
ck-rocprof analyze <name>       # Block 12: LDS (default)
ck-rocprof analyze <name> 2     # Block 2: Speed-of-Light
ck-rocprof analyze <name> 7     # Block 7: L2 Cache
ck-rocprof analyze <name> 11    # Block 11: Vector L1D
ck-rocprof analyze <name> 16    # Block 16: Instruction Mix
ck-rocprof analyze <name> 17    # Block 17: CU Metrics
```

Key metrics to extract:
- LDS bank conflicts/access (target: <0.01)
- Occupancy (waves/CU)
- L2 hit rate
- HBM bandwidth utilization (% peak)
- VGPR/SGPR usage
- Instruction mix (MFMA / VALU / VMEM / LDS / SALU ratios)

---

## Agent B: ISA Assembly Annotation (CPU Only)

### B1. Compile with --save-temps

```bash
cd /root/rocm-libraries/projects/composablekernel/build
cmake --build . --target <target> -j$(nproc) -- -Xarch_device --save-temps
```

Produces `.s` files in the build tree.

### B2. Identify Kernel .s File

Find the `.s` file matching the kernel function name (mangled as `_ZN7ck_tile...`). If the file contains multiple kernels, extract the relevant one.

### B3. Read C++ Pipeline Sources

Identify the full source chain using CK Tile's four-layer hierarchy:
- **Kernel**: `include/ck_tile/ops/<op>/kernel/<kernel>.hpp`
- **Pipeline**: `include/ck_tile/ops/<op>/pipeline/<pipeline>.hpp`
- **Block ops**: `include/ck_tile/ops/<op>/block/<block_op>.hpp`
- **Warp ops**: `include/ck_tile/ops/<op>/warp/<warp_op>.hpp`
- **Epilogue**: relevant epilogue files

### B4. Annotate ISA Line-by-Line

For every instruction in the kernel ISA, add comments explaining:
1. **Pipeline stage**: which C++ source function/stage (e.g., "Main Loop - Load A tile", "Epilogue - Store C")
2. **Algorithm meaning**: what the instruction does in algorithmic context (e.g., "MFMA accumulate: C[m,n] += A[m,k] * B[k,n]")
3. **Register context**: what each register holds (e.g., "v[0:3] = A tile fragment row 0")

Use `.loc` directives in the `.s` file to map assembly back to C++ source lines.

**Output format example:**
```asm
; ===== Pipeline Stage: Main Loop - Load A tile from global memory =====
; C++ source: block_gemm_areg_bsmem_creg.hpp:142 — load_tile(a_window)
; Loads 128 bytes (8x fp16) of A tile from HBM into VGPRs

s_waitcnt vmcnt(0)                                         ; wait for prior global loads
buffer_load_dwordx4 v[0:3], v8, s[0:3], 0 offen           ; A[m, k+0:k+3] → v[0:3]
buffer_load_dwordx4 v[4:7], v9, s[0:3], 0 offen           ; A[m, k+4:k+7] → v[4:7]

; ===== Pipeline Stage: Main Loop - MFMA Accumulation =====
; C++ source: warp_gemm_impl.hpp:87 — operator()(a_frag, b_frag, c_frag)
v_mfma_f32_32x32x8_f16 a[0:15], v[0:1], v[32:33], a[0:15] ; C[0:15] += A[0:1] * B[0:1]
```

### B5. Cross-Verify Annotations (Mandatory)

**Do not skip this step.** Verification checklist:
1. `.loc` directives match claimed C++ source files and line numbers
2. Register usage is consistent (same register = same logical variable throughout)
3. Pipeline stage boundaries match C++ control flow (loop structure, branch targets)
4. MFMA instruction shapes match the warp GEMM configuration in the Problem/Shape types
5. 100% of kernel ISA is annotated (no unannotated sections)

If any discrepancy: fix, re-verify, repeat until clean.

---

## Final Outputs

All outputs go to `build/profile_results/<name>/`:

```
build/profile_results/<name>/
  <name>_att.tar.gz              # RCV package (open in RCV desktop app)
  <name>_annotated.s             # Line-by-line annotated kernel ISA
  <name>_profile_report.md       # Comprehensive markdown report
```

### Report Structure (`<name>_profile_report.md`)

1. **Summary**: kernel name, GPU arch, binary, args, duration, TFLOPS/bandwidth
2. **GPU Occupancy**: `rocm-smi --showuse` output at profiling time
3. **ATT Trace**: RCV package location, how to open, what to look for
4. **PMC Analysis**: table of key metrics vs targets

   | Metric | Value | Target | Status |
   |--------|-------|--------|--------|
   | LDS Bank Conflicts/Access | 0.005 | <0.01 | OK |
   | Occupancy | 4 waves/CU | - | - |

5. **Bottleneck Analysis**: compute-bound vs memory-bound vs latency-bound determination
6. **Optimization Suggestions**: actionable items with references to CK Tile source files
7. **Assembly Summary**: annotated `.s` file location, instruction count breakdown (MFMA/VALU/VMEM/LDS/SALU)

---

## Comparison Mode

When user requests A/B comparison (e.g., "compare with baseline", "did optimization help?"):

1. Check if `build/profile_results/<name>_baseline/` exists
2. If no baseline: save current results as baseline, inform user
3. If baseline exists: run new profile, generate diff table:

   | Metric | Baseline | Current | Delta |
   |--------|----------|---------|-------|
   | Duration | 1.2 ms | 0.9 ms | -25% |
   | LDS Conflicts | 0.02 | 0.005 | -75% |

4. Include conclusion: what improved, what regressed, net assessment

---

## Key Rules (from CLAUDE.md)

- `rocprofv3 --att` must **NOT** use `-o` flag (breaks RCV output)
- Always use `-v=0` for profiling runs (skip correctness verification)
- Always check GPU occupancy before profiling (`rocm-smi --showuse`, abort if >10%)
- Report must include GPU occupancy at profiling time
- Use `run_in_background=true` for compilation and profiling commands (they take long)
- Use `-j$(nproc)` for compilation, `-G Ninja` for CMake
- Assembly annotation accuracy is **non-negotiable** — verify until clean
