# FlyDSL vs hipBLASLt Benchmark Artifacts

MXFP8 GEMM (TN) benchmarking: FlyDSL (via aiter docker) vs hipBLASLt (via Tensile tuning).

## Structure

```
configs/              6 Tensile winner configs (flydsl_mxf8_tn_{30..35}_winner.yaml)
run_tuning.sh         Run Tensile tuning for all configs
extract_tuning_perf.py  Extract tuning results into a pivot CSV
flydsl_benchmark/     FlyDSL side
  doBenchmark.sh        Docker-based flydsl benchmark runner
  test_gemm_a8w8_blockscale.py  aiter test script (copied from container)
compare_perf.py       Compare flydsl vs hipblaslt latency into a summary CSV
```

## Commands

### 1. Run hipBLASLt Tensile tuning

```bash
./run_tuning.sh run1
# outputs: run1/<config>/log.txt for each config
```

### 2. Extract hipBLASLt tuning results

```bash
python extract_tuning_perf.py run1 -o run1/tuning_results.csv
# or across multiple runs:
python extract_tuning_perf.py run1 run2 -o tuning_comparison.csv
```

### 3. Run FlyDSL benchmark

```bash
cd flydsl_benchmark
bash doBenchmark.sh 2>&1 | tee perf_flydsl.txt
```

### 4. Compare FlyDSL vs hipBLASLt

```bash
# from tuning output directory:
python compare_perf.py --flydsl flydsl_benchmark/perf_flydsl.txt --hipblaslt run1 -o flydsl_vs_hipblaslt.csv

# or from pre-extracted CSV:
python compare_perf.py --flydsl flydsl_benchmark/perf_flydsl.txt --hipblaslt-csv run1/tuning_results.csv
```

Output: `M,N,K,flydsl_us,hipblaslt_us,ratio,note` (ratio > 1 = hipBLASLt slower).
