# hip_mlops_engine benchmark vs MIOpen

Compares the provider's `HIP_MLOPS_ENGINE` with `MIOpenDriver -t 1` on BatchNorm
forward training, LayerNorm forward and LayerNorm backward. It supports the
decide-by-measurement items for warp-shuffle `reduce2` in BatchNorm forward
training, vector width 8 and LayerNorm scheduling.

## How it measures

- **Provider:** the script writes a bundle tree for the shapes in
  [bench_matrix.json](bench_matrix.json), cloned from the `standard` bundle templates
  in the build. It runs `hipdnn_integration_tests --benchmark` with GPU-reference
  verification (`--verification-mode cpu` if a case fails against the GPU reference;
  it is much slower), loading the engine's test config
  (`config/HIP_MLOPS_ENGINE.toml`, override with `--test-config`) so its
  validator overrides apply. A case that fails verification, or that the engine declines, gets
  no time and is marked `provider-no-result`. Reported time is the harness's
  `robust` device time (autotune mean without the slow tail).
- **MIOpen:** `MIOpenDriver <layer> ... -t 1 -V 0 -i 100`. BatchNorm uses
  spatial mode with running and saved statistics (`-r 1 -s 1`). LayerNorm uses
  elementwise-affine mode (`-m 0`). The reported time is the average GPU kernel time.
- Both tools run on the device in `--device` (`HIP_VISIBLE_DEVICES`).

`ratio = provider / MIOpen`; above 1.0 means the provider is slower.

## Matrix and run time

[bench_matrix.json](bench_matrix.json) holds 20 BatchNorm shapes and 26 LayerNorm
entries (transformer rows, NHWC small-inner, conv-like `k=3` and shapes taken from
MIOpen's gtests), in fp32, fp16 and bf16 and in the listed layouts: 402 cases. A full
run takes about 10 minutes on one MI210 (gfx90a). The LayerNorm backward `k=3` case
on 32x64x56x56 was removed because it only passes against the CPU reference on gfx90a.

## Reading the results

- Both timers include overhead beyond kernel time, and the overhead differs between
  the harness and `MIOpenDriver`. For short kernels (below roughly 0.05 ms) and for
  multi-kernel BatchNorm the ratio can be off by 20% or more, so only trust BatchNorm
  differences clearly larger than that. LayerNorm ratios on large kernels match
  `rocprofv3 --kernel-trace` durations.
- To confirm a result, trace one case on each side, for example
  `rocprofv3 --kernel-trace -- MIOpenDriver layernorm ... -t 0 -i 20`, and compare
  the kernel names, launch sizes and durations.
- The 5D `NDHWC` LayerNorm forward ratios are very large (up to about 230x). Both
  sides run the same rows; the provider launches one 1024-thread block per row, while
  MIOpen schedules many small rows per workgroup in channels-last layouts.

## Prerequisites

- An AMD GPU that nothing else is using while the script runs. The timings are
  sensitive to other GPU load; pick the device with `--device <n>`.
- ROCm with `MIOpenDriver` (default `<rocm>/bin/MIOpenDriver`, where `<rocm>` is the
  `ROCM_PATH` environment variable or `/opt/rocm`). Override with `--rocm <dir>` or
  `--miopen-driver <path>`.
- A provider build for the GPU architecture under test, containing
  `bin/hipdnn_integration_tests`, `lib/hipdnn_plugins/engines/libhip_kernel_provider.so`
  and `lib/integration-test-bundles` (the template bundles the script clones). Use the
  `hip-kernel-provider` superbuild preset from the repository root, for example:

  ```bash
  cmake --preset hip-kernel-provider -B <build> -DGPU_TARGETS=<arch> \
    -DENABLE_ASM_SDPA_ENGINE=OFF -DALLOW_FETCH_DEPS=ON -DENABLE_CLANG_TIDY=OFF
  ninja -C <build> hip_kernel_provider hipdnn_integration_tests
  ```

  `<arch>` is the target GPU (`gfx942`, `gfx1201`, ...).
- Python 3 with only the standard library. Run the script from any directory.

## Usage

Build `hipdnn_integration_tests` and the provider with the `hip-kernel-provider`
superbuild preset (see the `hipdnn-superbuild` skill), then:

```bash
./bench_vs_miopen.py run --build-dir <build> --label baseline_gfx942 --out-dir bench_results
```

Useful options: `--ops bn_fwd ln_fwd ln_bwd`, `--filter <regex on case id>`,
`--device <n>`, `--skip-miopen` (re-time only the provider after a change),
`--matrix <json>`. Outputs: `<label>.csv`, `<label>.md`, `provider.log`,
`miopen.log`.

Before/after a change, rebuild the provider, run again with another label
(`--skip-miopen` is fine), then:

```bash
./bench_vs_miopen.py compare bench_results/baseline.csv bench_results/after.csv
```

The CSV records the provider time next to the MIOpen time of the same shape, so run
the baseline once with MIOpen included and reuse its columns.

## Output

`run` prints a geometric-mean ratio per op and writes, into `--out-dir`:

- `<label>.csv`: one row per case. Columns: `op` (`bn_fwd`, `ln_fwd`, `ln_bwd`), `case`
  (`<dims>_<dtype>_<layout>[_k<normalized_dim_count>]`), `dtype`, `layout`, `dims`,
  `normalized_dim_count`, `tag` (shape group), `status`, `provider_min_ms`,
  `provider_robust_ms`, `miopen_min_ms` (BatchNorm only), `miopen_avg_ms`,
  `ratio_vs_miopen` (`provider_robust_ms / miopen_avg_ms`).
- `<label>.md`: the same data as one table per op, with a geometric mean.
- `provider.log`, `miopen.log`: raw tool output; read these when a case has no result.
- `bundles/`: the generated test bundles.

`status` is `ok` (both timed), `provider-no-result` (the provider declined the case,
skipped it or failed verification; see `provider.log`) or `miopen-no-result` (the
driver produced no time; see `miopen.log`). Exclude rows whose status is not `ok`
before computing statistics. The full matrix is long-running; use `--filter` or
`--ops` for quick checks.

Example: the ten worst cases and the geometric mean per op, shape group and dtype.

```bash
python3 - <<'PY'
import csv, math, collections
with open("bench_results/baseline.csv") as f:
    rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
for r in sorted(rows, key=lambda r: -float(r["ratio_vs_miopen"]))[:10]:
    print(r["op"], r["case"], round(float(r["ratio_vs_miopen"]), 2))
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["op"], r["tag"], r["dtype"])].append(float(r["ratio_vs_miopen"]))
for k, v in sorted(groups.items()):
    print(k, len(v), round(math.exp(sum(map(math.log, v)) / len(v)), 2))
PY
```
