# ILP Model System — Unified Instruction-Level Parallelism Methodology

## Overview

The HunTian ILP (Instruction-Level Parallelism) model system provides
per-architecture hardware models capturing instruction latency, execution
port topology, memory hierarchy, and pipeline parameters. Originally
developed for the HunTian SASS assembler (`sass-assembler`), the same
methodology is used to guide SWMMAC kernel optimization on RDNA4.

## Data Sources

```
Pascal GP106:   /data/rtl-sdr/ptx_gp106 (probe chain measurement)
Broadwell-EP:   /data/rtl-sdr/cpu_probe (objdump + perf stat)
Volta/Ampere:   denvdis SM80/SM120 data
Zen4/Zen5:      Intel SDM + AMD PPR (public)
RDNA4 (gfx1200): /data/rtl-sdr/swmmac (SWMMAC probe chain + DOE)
```

## Hardware Model Structure

Each hardware model captures:

| Parameter Group | Fields | GPU vs CPU |
|----------------|--------|------------|
| Pipeline | issue_width, retire_width, out_of_order, rob_size, phys_regs | GPU: warp_inst_buffer, scoreboard_regs |
| Execution | fma_units, fp_ports, int_ports, ld_ports, st_ports | GPU: sm_count, warps_per_sm |
| Memory | l1/l2/l3/dram latency + size | GPU: shared_memory, smem_banks |
| Special | has_tensor_cores, zero_overhead_switch | GPU: warp_size, max_active_warps |

## RDNA4 (gfx1200) Model

```
Architecture:       AMD RDNA4 (RX 9060 XT)
Compute Units:      32 CUs
SIMD Units:         64 SIMD32 (2 per CU)
Clock:              2780 MHz base, ~3150 MHz boost
Theoretical FP32:   25.6 TFLOPS

Pipeline:
  Issue Width:      1 per SIMD (in-order)
  Wavefront Size:   Wave32
  Max Waves/CU:     64
  Warp Buffer:      4 instructions per wave
  Scoreboard:       256 VGPR

Memory:
  Infinity Cache:   64 MB (~100 cycles)
  VRAM:             ~300 cycles
  LDS:              20 cycles (32 banks)
  VGPR:             256 registers per wave

FPU:
  FP32 FMA/ADD/MUL: 4 cycles (v_fmac_f32 / v_add_f32 / v_mul_f32)
  FP32 SQRT:        16 cycles
  INT32 ADD:        4 cycles
  INT32 MUL:        16 cycles
  SWMMAC:           26 cycles (hardware pipeline, 16-chain unroll)
```

## Key Instruction Latencies (RDNA4)

| Instruction | Mnemonic | Latency (cycles) | Throughput |
|-------------|----------|-----------------|------------|
| FP32 FMA | v_fmac_f32 | 4 | 1/cycle |
| FP32 ADD | v_add_f32 | 4 | 1/cycle |
| FP32 MUL | v_mul_f32 | 4 | 1/cycle |
| FP32 SQRT | v_sqrt_f32 | 16 | 1/16cp |
| INT32 ADD | v_add_u32 | 4 | 1/cycle |
| INT32 MUL | v_mul_lo_u32 | 16 | 1/16cp |
| Global Load | global_load_dword | ~300 | L2 miss penalty |
| Shared Load | ds_read_b32 | ~20 | 1/cycle (32 banks) |
| Global Store | global_store_dword | ~300 | write-combine |
| Shared Store | ds_write_b32 | ~20 | 1/cycle |
| Branch | s_branch | 4 | scalar pipe |
| Barrier | s_barrier | ~20 | workgroup sync |
| SWMMAC | v_swmmac_* | **26** | 1/cycle (pipelined) |

## How ILP Models Guide SWMMAC Optimization

The RDNA4 model directly informs the SWMMAC optimization tiers:

| Optimization | ILP Insight | Result |
|-------------|-------------|--------|
| 16-chain unroll | SWMMAC latency=26 → 16 chains fill pipeline | 0 bubbles |
| StaggeredPipeline | barrier=20c → wave-level atomicAdd avoids sync | +84% |
| L2-persistent | VRAM=300c → counter in L2 avoids cache misses | +194% |
| Wave-level claiming | scoreboard=256 VGPR → readfirstlane preserves EXEC | +6% |
| Dual-wave occupancy | VGPR budget → INT4 VGPR=19 fits dual-wave | occupancy ×2 |

## Multi-Architecture Coverage

| Architecture | Model | Data Source | Status |
|-------------|-------|------------|--------|
| Pascal GP106 (sm_61) | pascal_gp106_model() | Probe chain + cuobjdump | ✅ Verified |
| Volta GV100 (sm_70) | volta_gv100_model() | denvdis SM80 + cuobjdump | ✅ Verified |
| Ampere GA100 (sm_80) | ampere_ga100_model() | denvdis SM80 | ✅ Verified |
| Hopper GH100 (sm_90) | hopper_gh100_model() | DeepGEMM + denvdis SM120 | ✅ Verified |
| **RDNA4 (gfx1200)** | **rdna4_model()** | **SWMMAC probe chain** | **✅ Verified** |
| Broadwell-EP (x86_64) | broadwell_model() | perf stat + objdump | ✅ Verified |
| Intel Core2 (x86_64) | core2_model() | Intel SDM | ✅ Verified |
| ARM Cortex-A78 | cortex_a78_model() | ARM TRM | ✅ Verified |
| LoongArch 3A6000 | loongarch_model() | Loongson TRM | ✅ Verified |

## Source Code

Full implementation: `src/sass/ilp_model.h` in the
[HunTian SASS Assembler](https://github.com/clearnature/sass-assembler) repository.
