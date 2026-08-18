# LATRD / SYTRD Benchmarking Guide

## Prerequisites

Build the library and clients targeting the GPU on the system:

```bash
./install.sh -cna gfx950 --cmake-arg="-DROCSOLVER_FIND_PACKAGE_LAPACK_CONFIG=OFF"
```

Replace `gfx950` with the appropriate architecture (e.g. `gfx942` for MI300X, `gfx1100` for
RX 7900). All benchmarks below are run from `build/release/`.

## rocsolver-bench output format

`rocsolver-bench --perf 1` prints a **single number: GPU time in microseconds**. Lower is
better. Without `--perf 1` it prints a table with both `cpu_time_us` and `gpu_time_us`
columns; use the `gpu_time_us` column for comparisons.

```bash
# Example output with --perf 1
$ LD_LIBRARY_PATH=library/src:$LD_LIBRARY_PATH ./clients/staging/rocsolver-bench \
    -f latrd -n 2048 -k 64 --perf 1 --iters 30 -r s --uplo L
1427
```

```bash
# Example output without --perf 1
$ LD_LIBRARY_PATH=library/src:$LD_LIBRARY_PATH ./clients/staging/rocsolver-bench \
    -f latrd -n 2048 -k 64 --iters 5 -r s --uplo L
cpu_time_us     gpu_time_us
22868           1427
```

**Important:** the test/bench binaries link against the system `/opt/rocm/lib/librocsolver.so`
by default. Always prepend `LD_LIBRARY_PATH` to use the locally built library:

```bash
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH ./clients/staging/rocsolver-bench ...
```

---

## Execution paths

LATRD and SYTRD have several execution paths, selected via environment variables at runtime.
All variables are boolean (set to any non-empty value to enable; unset to disable).

### Path selection

| Environment variable | Effect |
|----------------------|--------|
| *(none set)* | **Fused kernel, canonical cooperative groups sync** (`cooperative_groups::this_grid().sync()`). Default for n < `LATRD_COOP_SWITCH_SIZE` (8192). |
| `LATRD_MULTI_KERNEL=1` | **Multi-kernel path**: separate kernel launches per step. Disables the fused kernel entirely. |
| `LATRD_SW_GRID_SYNC=1` | **Fused kernel, software grid sync** (`SoftwareGridSync::sync()`): includes full L2 flush/invalidate fences at each sync point. |
| `LATRD_SW_RAW_SYNC=1` | **Fused kernel, software barrier** (`SoftwareGridSync::barrier()`): no L2 fences; cross-block data exchanged via sc1 (L2-bypassing) raw buffer stores/loads. |
| `COOP_LAUNCH=1` | Force the fused cooperative kernel even when the default heuristic would choose the multi-kernel path. Rarely needed directly; prefer `LATRD_COOP_SWITCH_SIZE`. |

Only one of `LATRD_MULTI_KERNEL`, `LATRD_SW_GRID_SYNC`, `LATRD_SW_RAW_SYNC` should be set at
a time. If none is set, the canonical cooperative groups path is used (the ROCm platform
grid sync).

### Tuning parameters

| Environment variable | Default | Effect |
|----------------------|---------|--------|
| `LATRD_COOP_SWITCH_SIZE=N` | 8192 | Use the fused kernel only when `n < N`. Set to 0 to force the multi-kernel path for all sizes without setting `LATRD_MULTI_KERNEL`. |
| `LATRD_COOP_GRID_X=N` | `n/2` (capped by occupancy) | Override the number of thread blocks for the fused cooperative kernel. The default (`n/2`) is generally too large; values in the range 64-256 tend to perform better in practice. |

### Diagnostic

| Environment variable | Effect |
|----------------------|--------|
| `PRINT_DEBUG=1` | Print HIP call trace and a `[latrd_fused]` line per launch showing `n`, `grid_x`, and which sync path is active. Useful to confirm which path is running. |

---

## Which path is running?

Use `PRINT_DEBUG=1` to confirm. The `[latrd_fused]` line shows:

```
[latrd_fused] n=2048 max_blocks_per_sm=8 max_total=2048 grid_x=128 sw_grid_sync=0 sw_raw_sync=1
```

- `grid_x` -- actual number of thread blocks launched
- `sw_grid_sync` / `sw_raw_sync` -- which software sync variant is active (both 0 = canonical coop groups)

If no `[latrd_fused]` line appears, the multi-kernel path is running.

---

## Benchmarking LATRD

LATRD reduces one panel of `n` rows by `k` columns. Benchmark a single panel directly with
`-f latrd -n <n> -k <nb>` (nb is typically 64).

```bash
cd build/release

# Multi-kernel path
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    LATRD_MULTI_KERNEL=1 \
    ./clients/staging/rocsolver-bench -f latrd -n 2048 -k 64 --perf 1 --iters 30 -r s --uplo L

# Canonical coop groups (default fused)
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    ./clients/staging/rocsolver-bench -f latrd -n 2048 -k 64 --perf 1 --iters 30 -r s --uplo L

# Fused + software sync (full L2 fences)
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    LATRD_SW_GRID_SYNC=1 LATRD_COOP_GRID_X=128 \
    ./clients/staging/rocsolver-bench -f latrd -n 2048 -k 64 --perf 1 --iters 30 -r s --uplo L

# Fused + software barrier (sc1 raw stores, no L2 fences)
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    LATRD_SW_RAW_SYNC=1 LATRD_COOP_GRID_X=128 \
    ./clients/staging/rocsolver-bench -f latrd -n 2048 -k 64 --perf 1 --iters 30 -r s --uplo L
```

### Sweeping block counts

The number of thread blocks (`LATRD_COOP_GRID_X`) is the main tuning knob for the fused
paths. Sweep it to find the optimum for a given matrix size and architecture:

```bash
for blocks in 8 16 32 64 128 256; do
    printf "COOP_GRID_X=%d\t" $blocks
    LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
        LATRD_SW_RAW_SYNC=1 LATRD_COOP_GRID_X=$blocks \
        ./clients/staging/rocsolver-bench -f latrd -n 2048 -k 64 --perf 1 --iters 30 -r s --uplo L
done
```

---

## Benchmarking SYTRD

SYTRD drives LATRD in a loop over panels, then calls SYTD2 for the trailing submatrix. The
LATRD path selection env vars apply inside the SYTRD loop. Use `-f sytrd` with only `-n`
(no `-k`; the block size `nb` is fixed at 64 internally).

```bash
cd build/release

# Multi-kernel path
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    LATRD_MULTI_KERNEL=1 \
    ./clients/staging/rocsolver-bench -f sytrd -n 2048 --perf 1 --iters 20 -r s --uplo L

# Canonical coop groups
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    ./clients/staging/rocsolver-bench -f sytrd -n 2048 --perf 1 --iters 20 -r s --uplo L

# Fused + software barrier
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    LATRD_SW_RAW_SYNC=1 LATRD_COOP_GRID_X=128 \
    ./clients/staging/rocsolver-bench -f sytrd -n 2048 --perf 1 --iters 20 -r s --uplo L
```

**Note on SYTRD vs LATRD benchmarking:** SYTRD calls LATRD once per panel (n/nb calls for an
nxn matrix). The fused kernel is launched once per panel with a fresh cooperative launch,
meaning kernel launch overhead and sync-buffer allocation (`hipMalloc`/`hipFree`) are paid
per panel, not once for the whole matrix. For isolating LATRD kernel performance, benchmark
`-f latrd` directly.

---

## Correctness testing

```bash
cd build/release

# LATRD -- all paths should pass all tests
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    ./clients/staging/rocsolver-test --gtest_filter='checkin*LATRD*float/*'

LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH LATRD_MULTI_KERNEL=1 \
    ./clients/staging/rocsolver-test --gtest_filter='checkin*LATRD*float/*'

LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH LATRD_SW_GRID_SYNC=1 \
    ./clients/staging/rocsolver-test --gtest_filter='checkin*LATRD*float/*'

LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH LATRD_SW_RAW_SYNC=1 \
    ./clients/staging/rocsolver-test --gtest_filter='checkin*LATRD*float/*'

# SYTRD
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    ./clients/staging/rocsolver-test --gtest_filter='checkin*SYTRD*float/*'
```

### Known failures in software sync paths

The two SYTRD float test cases below fail with both `LATRD_SW_GRID_SYNC=1` and
`LATRD_SW_RAW_SYNC=1`, but pass with the canonical cooperative groups path and the
multi-kernel path. This is a pre-existing bug in `SoftwareGridSync`, not in the fused
kernel algorithm.

| gtest name | Parameters |
|------------|-----------|
| `checkin_lapack/SYTRD.__float/10` | n=130, k=130, lower |
| `checkin_lapack/SYTRD.__float/12` | n=150, k=200, lower |

---

## Benchmarking SYEVD

SYEVD computes all eigenvalues (and optionally eigenvectors) of a real symmetric matrix using
a divide-and-conquer algorithm. It calls SYTRD internally for the tridiagonalization step, so
the LATRD path selection env vars (`LATRD_MULTI_KERNEL`, `LATRD_SW_GRID_SYNC`, etc.) apply
here too. Use `-f syevd` with only `-n` (no `-k`; the SYTRD block size is fixed internally).

**Important:** always pass `--evect V` to benchmark the full SYEVD path including eigenvector
computation. Without it, only eigenvalues are computed, which skips a significant portion of
the divide-and-conquer work and does not reflect production use.

```bash
cd build/release

# Default matrix type (random, diagonally dominant)
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    ./clients/staging/rocsolver-bench -f syevd --evect V --uplo L -n 2048 --perf 1 --iters 21 -r s

# Toeplitz matrix (symmetric tridiagonal: diagonal=2, off-diagonal=1)
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    SYEVD_TEST_TOEPLITZ=1 \
    ./clients/staging/rocsolver-bench -f syevd --evect V --uplo L -n 2048 --perf 1 --iters 21 -r s

# Wilkinson matrix (symmetric tridiagonal, near-repeated eigenvalues)
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    SYEVD_TEST_WILKINSON=1 \
    ./clients/staging/rocsolver-bench -f syevd --evect V --uplo L -n 2048 --perf 1 --iters 21 -r s
```

### Matrix types

The input matrix is selected via environment variable. Only one should be set at a time;
if none is set the default initializer is used.

| Environment variable | Matrix | Notes |
|---|---|---|
| *(none)* | **Default** -- random, diagonally dominant | Diagonal shifted by +400; off-diagonal scaled by -4. General-purpose baseline. |
| `SYEVD_TEST_TOEPLITZ=1` | **Toeplitz** -- symmetric tridiagonal, diagonal=2, off-diagonal=1 | Known eigenvalues: 2 + 2cos(k?/(n+1)). Well-conditioned, good for convergence rate checks. |
| `SYEVD_TEST_WILKINSON=1` | **Wilkinson** -- symmetric tridiagonal with near-repeated eigenvalues | Classic stress test for eigenvalue solvers; exposes convergence difficulty. |
| `SYEVD_TEST_CLEMENT=1` | **Clement** -- symmetric tridiagonal, diagonal=0, off-diagonal=sqrt(i(n-i)) | Known eigenvalues symmetric about 0; tests behaviour with zero diagonal. |
| `SYEVD_TEST_EIG7=1` | **Eig7** -- n-1 eigenvalues clustered near 0 (multiples of ?), one eigenvalue = 1 | Stress test for deflation in the divide-and-conquer step. |

All four `SYEVD_TEST_*` variables have aliases without the `SYEVD_` prefix
(`TEST_TOEPLITZ`, `TEST_WILKINSON`, etc.) that apply to all eigensolver benchmarks.

### Running the sweep script

```bash
cd build/release

# Default matrix, single precision (with eigenvectors)
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    ./bench_syevd.sh ./clients/staging/rocsolver-bench

# Toeplitz matrix, double precision (with eigenvectors)
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    SYEVD_TEST_TOEPLITZ=1 \
    ./bench_syevd.sh ./clients/staging/rocsolver-bench "syevd --evect V --uplo L" d

# With numerical verification
VERIFY=1 LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    ./bench_syevd.sh ./clients/staging/rocsolver-bench
```

The script signature is:
```
bench_syevd.sh <bench-binary> [func] [prec] [device]
```
Defaults: `func="syevd --evect V --uplo L"`, `prec=s`, `device=0`.

The sweep covers n=320-2048 in steps of 64, n=2176-4096 in steps of 128, and n=4352-8192
in steps of 256. Iteration counts taper from 31 (small) to 11 (large) to keep total
benchmark time reasonable.

---

## Sweep scripts

The sweep scripts live in the **rocsolver root** (alongside `install.sh`). They are not
copied into the build directory by CMake -- copy the ones you need into `build/release/`
manually (e.g. `cp -p bench_*.sh build/release/`) before running the commands below.

Each script runs a matrix-size sweep for one function and prints `n<TAB>gpu_time_us` per
line. Each has a `_multikernel` variant that forces the multi-kernel path (via
`export LATRD_MULTI_KERNEL=1` inside the script) so the two paths can be compared without
setting the env var by hand.

| Default (fused) | Multi-kernel variant | Function |
|---|---|---|
| `bench_latrd.sh` | `bench_latrd_multikernel.sh` | LATRD single panel |
| `bench_sytrd.sh` | `bench_sytrd_multikernel.sh` | SYTRD |
| `bench_syevd.sh` | `bench_syevd_multikernel.sh` | SYEVD |

Both variants take the same arguments (`<bench-binary> [func] [prec] [device]`, plus `-k` for
LATRD) and honor the `VERIFY=1` env var. Invoke the multi-kernel variants exactly like the
originals -- the path env var is exported inside the script:

```bash
cd build/release

# Default fused path
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    ./bench_sytrd.sh ./clients/staging/rocsolver-bench

# Multi-kernel path (same sweep, same args)
LD_LIBRARY_PATH=$(pwd)/library/src:$LD_LIBRARY_PATH \
    ./bench_sytrd_multikernel.sh ./clients/staging/rocsolver-bench
```

---

## Interpreting results

- **Multi-kernel vs fused:** The multi-kernel path serializes computation at kernel
  boundaries and has low per-launch overhead. The fused kernel keeps all blocks alive for
  the entire LATRD reduction, avoiding kernel launch overhead but paying for grid-wide
  synchronization at each step.

- **Block count (`LATRD_COOP_GRID_X`):** Too few blocks under-utilizes the GPU; too many
  increases contention in the sync spin loop and wastes wavefronts spinning. The optimal
  value is architecture- and size-dependent; sweep 8-256 and pick the minimum.

- **`sync()` vs `barrier()`:** `SoftwareGridSync::sync()` issues full L2 flush/invalidate
  fences (`__ATOMIC_RELEASE/"agent"` + `__ATOMIC_ACQUIRE/"agent"`) at every sync point.
  `barrier()` (used by `LATRD_SW_RAW_SYNC`) skips those fences -- coherency is maintained
  instead by using sc1 (L2-bypassing) raw buffer stores and loads for all cross-block data.
  On MI350X and MI300X this yields roughly a 2x reduction in synchronization overhead,
  which is most visible at larger matrix sizes where there are more sync points per launch.
