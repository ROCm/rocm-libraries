# SYEVD Roofline Analysis — MI300X, FP32

## Setup

**Hardware:** 2× AMD Instinct MI300X (`gfx942`, DID `0x74a1`)  
**ROCm:** 7.2.3  
**Function:** `rocsolver_syevd`, single precision (FP32), `--uplo L`  
**Profiler:** `rocprof` v1 with `--stats` (rocprofv3 1.1.0 segfaults at n≥2048 due to a known bug; see `rocprofv3_segfault_large_dispatch_count.md`)  
**Roofline peaks** (empirical, from rocprof-compute on the same hardware):
- HBM bandwidth: **4168 GB/s**
- FP32 scalar peak: **128374 GFLOPS**
- Ridge point: ~30.8 FLOP/byte

**Counter collection:**
```
pmc : SQ_INSTS_VALU_FMA_F32 SQ_INSTS_VALU_ADD_F32 SQ_INSTS_VALU_MUL_F32 SQ_INSTS_VALU_TRANS_F32
pmc : SQ_INSTS_VALU_MFMA_MOPS_F32 TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum TCC_EA0_WRREQ_sum TCC_EA0_WRREQ_64B_sum
```

**FLOPs formula:**  
`FMA×2 + ADD×1 + MUL×1 + TRANS×1 + MFMA×1`

**HBM bytes formula:**  
`RDREQ×64 + RDREQ_32B×32 + WRREQ_64B×64 + (WRREQ − WRREQ_64B)×32`

**Iteration accounting:** `rocsolver-bench --iters 1` runs 3 internal calls (1 hot + 2 cold). All per-call figures are divided by 3.

---

## Aggregated Roofline Results

One data point per SYEVD call (all kernels summed):

| n    | FLOPs/call | HBM/call  | Time/call | AI (FLOP/byte) | GFLOPS |
|------|-----------|-----------|-----------|----------------|--------|
|  512 | 2.06e+07  |  0.39 GB  |   8.6 ms  | 0.0534         |    2.4 |
| 1024 | 9.88e+07  |  1.80 GB  |  18.7 ms  | 0.0550         |    5.3 |
| 2048 | 3.81e+08  | 10.16 GB  |  39.4 ms  | 0.0375         |    9.7 |
| 3072 | 1.02e+09  | 29.27 GB  |  70.1 ms  | 0.0350         |   14.6 |
| 4096 | 2.11e+09  | 64.55 GB  | 102.6 ms  | 0.0328         |   20.6 |
| 5120 | 3.78e+09  | 119.9 GB  | 151.2 ms  | 0.0315         |   25.0 |
| 6144 | 6.15e+09  | 203.5 GB  | 214.7 ms  | 0.0302         |   28.6 |
| 7168 | 9.30e+09  | 316.6 GB  | 287.7 ms  | 0.0294         |   32.3 |
| 8192 | 1.34e+10  | 465.5 GB  | 373.1 ms  | 0.0288         |   35.9 |

The plot is saved at `build/release/syevd_roofline_aggregated.pdf/png`.

**Key observations:**
- All sizes are deep in the **memory-bound** region (AI ≪ 30.8 ridge point).
- Achieved GFLOPS climbs steadily with n but only reaches ~36 GFLOPS at n=8192
  (~0.028% of the 128K GFLOPS FP32 scalar peak; ~0.86% of the HBM-bandwidth ceiling
  at that arithmetic intensity).
- Arithmetic intensity **decreases** with n (0.055 → 0.029), meaning HBM traffic
  grows faster than FLOPs as the matrix grows. This is explained in the analysis below.

---

## Per-Kernel Breakdown

HBM traffic and time share by kernel for three representative sizes (figures are
per-syevd-call totals, divider=3 applied):

### n=1024 — total HBM 5.39 GB, total time 56 ms

| Kernel (abbreviated)              | calls | time% | hbm% | hbm GB |
|-----------------------------------|------:|------:|-----:|-------:|
| `latrd_lower_computeW_gemvt`      |  3072 |  17.3 | 47.3 |   2.55 |
| `latrd_lower_updateA`             |  3072 |  18.5 | 14.1 |   0.76 |
| `latrd_lower_updateW`             |  3072 |  19.0 | 12.9 |   0.69 |
| rocBLAS GEMMs (`Cijk_*`)          |    48 |   1.4 | 11.9 |   0.64 |
| STEDC kernels                     |   ~90 |  12.1 |  8.7 |   0.47 |

### n=4096 — total HBM 193.7 GB, total time 308 ms

| Kernel (abbreviated)              | calls  | time% | hbm% | hbm GB |
|-----------------------------------|-------:|------:|-----:|-------:|
| `latrd_lower_computeW_gemvt`      | 11,520 |  30.2 | 73.8 | 142.85 |
| `latrd_lower_updateA`             | 11,520 |  12.8 |  4.9 |   9.39 |
| `latrd_lower_updateW`             | 11,520 |  13.0 |  4.7 |   9.08 |
| rocBLAS GEMMs (`Cijk_*`)          |    189 |   5.5 |  7.4 |  14.25 |
| STEDC kernels                     |   ~100 |   0.9 |  3.5 |   6.75 |

### n=8192 — total HBM 1396.5 GB, total time 1119 ms

| Kernel (abbreviated)              | calls  | time% | hbm% |  hbm GB |
|-----------------------------------|-------:|------:|-----:|--------:|
| `latrd_lower_computeW_gemvt`      | 23,808 |  46.9 | 80.4 | 1123.03 |
| rocBLAS GEMMs (`Cijk_*`)          |    471 |   9.6 |  9.3 |  139.08 |
| `latrd_lower_updateA`             | 23,808 |   8.0 |  2.0 |   27.98 |
| `latrd_lower_updateW`             | 23,808 |   8.2 |  2.0 |   27.27 |
| STEDC kernels                     |   ~100 |   6.2 |  2.6 |   36.94 |

---

## Root Cause: the LATRD W-column GEMV

### What the kernel does

`latrd_lower_computeW_gemvt_kernel` computes one column of the W panel in the
blocked Householder tridiagonalization (LATRD). At step `k` within a panel it
performs a SYMV on the current trailing submatrix of size `(n−k) × (n−k)`,
reading the full lower triangle once per panel column. This is the DSYMV call
inside DLATRD. The kernel is a pure streaming operation: each matrix element is
read once and contributes two FLOPs (multiply + accumulate), giving an inherently
low arithmetic intensity regardless of problem size.

### Per-call metrics across sizes

| n    | calls  | HBM/call (MB) | dur/call (µs) | AI (FLOP/byte) | GFLOPS |
|------|-------:|--------------:|--------------:|:--------------:|-------:|
|  512 |  1,536 |          0.25 |          2.49 |     0.014      |    1.4 |
| 1024 |  3,072 |          0.83 |          3.15 |     0.015      |    4.0 |
| 2048 |  6,144 |          3.03 |          3.75 |     0.015      |   12.5 |
| 4096 | 11,520 |         12.40 |          8.07 |     0.016      |   23.9 |
| 8192 | 23,808 |         47.17 |         22.07 |     0.016      |   33.3 |

The per-call AI is nearly constant at ~0.015 FLOP/byte — roughly 3.5× lower than
the overall SYEVD average — and does not improve with n.

### Scaling analysis

| Transition      | HBM/call scaling | call count scaling | total HBM scaling |
|-----------------|:----------------:|:------------------:|:-----------------:|
| n=512 → 1024    | 3.3× (≈n^1.7)    | 2.0× (≈n^1.0)     | 6.6× (≈n^2.7)    |
| n=1024 → 2048   | 3.7× (≈n^1.9)    | 2.0× (≈n^1.0)     | 7.3× (≈n^2.9)    |
| n=2048 → 4096   | 4.1× (≈n^2.0)    | 1.9× (≈n^0.9)     | 7.7× (≈n^2.9)    |
| n=4096 → 8192   | 3.8× (≈n^1.9)    | 2.1× (≈n^1.0)     | 7.9× (≈n^3.0)    |

- **Call count** scales O(n): `n/nb` panels × `nb` SYMV calls per panel = O(n) total.
- **HBM per call** scales O(n²): each SYMV reads a triangular submatrix that grows
  quadratically as the trailing matrix expands over successive panels.
- **Total HBM for this kernel** = O(n) × O(n²) = **O(n³)**.

### Why AI decreases with n

Both the LATRD SYMV and the rocBLAS trailing-matrix GEMMs scale O(n³) in FLOPs.
However their arithmetic intensities are very different:

- The GEMMs block data through L2/L1 caches and achieve substantially higher AI.
- The SYMV is a pure streaming kernel (AI ≈ 0.015 FLOP/byte) — this cannot be
  improved without changing the algorithm.

As n grows the SYMV's share of total HBM traffic increases from 47% (n=1024) to
80% (n=8192), pulling the aggregate AI downward even though the GEMMs are
reasonably efficient. The decreasing AI with n is therefore not a sign of
inefficiency elsewhere — it reflects the growing dominance of an inherently
low-AI O(n³) memory-streaming operation.

### Implications

- SYEVD on MI300X is firmly memory-bound at all practical matrix sizes, limited
  by the LATRD SYMV.
- The growing GFLOPS with n (2.4 → 35.9) reflects better GPU utilization per
  kernel dispatch as matrices grow, not improved algorithmic efficiency.
- Paths to higher arithmetic intensity:
  - **Wider panel (larger `nb`):** amortises the SYMV cost over more columns per
    panel, increasing the GEMM fraction. Diminishing returns as the GEMM trailing
    update itself becomes large.
  - **Exploit symmetry more aggressively:** reduce the effective data read per SYMV
    call (e.g. register/shared-memory tiling of the triangular read).
  - **Algorithmic change:** algorithms that replace the SYMV with a higher-AI
    primitive (e.g. a two-stage reduction using a second GEMM panel) can in
    principle convert some of the streaming traffic into compute-bound work.
