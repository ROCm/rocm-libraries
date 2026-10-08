# MXFP8 TN GEMM configs: pre-shuffle A + Block128 + M/N swap + BF16 split-K reduction

This folder holds the tuned TensileLite configs for six MXFP8 TN GEMM problem sizes on gfx1250, plus the steps to reproduce the wall time and device time from scratch.
You do not need to know Tensile to use this README. Follow sections 1-5 in order.

## 0. What the configs are

| Item | Setting in the yaml |
|---|---|
| Data type | MXFP8 (`DataType: 15`) in, BF16 out (`DestDataType: 7`), TN (`TransposeA: true`, `TransposeB: false`) |
| M/N swap | Problem is written as M = weight dim (2048 ... 65536), N = 512 |
| Pre-shuffle | `SwizzleTensorA: true` (after the swap, A is the pre-shuffled weight) |
| Block scaling | `MXBlockA: [128, 128]` (weight, 128x128 tile), `MXBlockB: [128]` (activation, 1x128) |
| Rotating buffer | `RotatingBufferSize: 384` (MiB) |
| Split-K | `_gsu1` files: `GlobalSplitU: [1]` (no split-K). `_gsuN` files: `GlobalSplitU > 1` with `NarrowGSUWorkspace: [true]` (partials stored as BF16, reduced in FP32). No atomic reduction. |

Files and results (b8-3, Helios-R, 1x gfx1250):

```
┌──────────────────────────────────┬──────────────────────────┬───────┬─────────┬────────┐
│ yaml                             │ problem_size(swap)_MxNxK │  GSU  │ wall_us │ dev_us │
├──────────────────────────────────┼──────────────────────────┼───────┼─────────┼────────┤
│ flydsl_mxf8_tn_33_swap_gsu1.yaml │ 7168x512x16384           │  w/o  │  28.38  │ 27.12  │
│ flydsl_mxf8_tn_35_swap_gsu1.yaml │ 65536x512x1536           │  w/o  │  24.34  │ 23.20  │
│ flydsl_mxf8_tn_31_swap_gsu1.yaml │ 6144x512x7168            │  w/o  │  15.45  │ 14.86  │
│ flydsl_mxf8_tn_32_swap_gsu1.yaml │ 7168x512x3072            │  w/o  │   9.00  │  8.33  │
│ flydsl_mxf8_tn_30_swap_gsuN.yaml │ 2048x512x7168            │MB-GSU4│  11.68  │  9.81  │
│ flydsl_mxf8_tn_34_swap_gsu1.yaml │ 8192x512x1536            │  w/o  │   7.08  │  5.93  │
└──────────────────────────────────┴──────────────────────────┴───────┴─────────┴────────┘
```

- `wall_us`: Tensile client timing with `KernelTime: false` (average over 1000 back-to-back launches after 2 warmups).
- `dev_us`: median kernel execution time from rocprofv3 (section 5). For split-K with MultipleBuffer (MB) this is GEMM + the `PostGSU` reduction kernel.
- The table was measured during the tuning sweep. Re-running a single yaml gives numbers within a few percent of it.
- MB = `MultipleBuffer`, MBSK = `MultipleBufferSingleKernel` (`GlobalSplitUAlgorithm`).

## 1. Get the code

```bash
git clone --branch users/briannwu/gfx1250-swizzle-b-tdm-mx128 --single-branch \
    https://github.com/ROCm/rocm-libraries.git
cd rocm-libraries/projects/hipblaslt/tensilelite
export TL=$PWD            # used by the commands below
```

The configs in this folder are at commit `a4a165a4213` (or later) of that branch:
`projects/hipblaslt/tensilelite/flydsl_swapmn_PS_B128_BF16R/`.

## 2. Python environment and build (once per clone)

```bash
# 2.1 venv. Use a versioned interpreter (python3.12), not a floating "python3" symlink.
/usr/bin/python3.12 -m venv ~/tensile-venv
source ~/tensile-venv/bin/activate
python -m pip install --upgrade pip invoke
python -m pip install -r $TL/requirements.txt

# 2.2 rocisa (Python extension used by the kernel generator)
cd $TL
invoke rocisa

# 2.3 tensilelite-client (C++ benchmark executable), built into $TL/build_tmp
invoke build-client
```

Check the build:

```bash
find $TL/build_tmp -name tensilelite-client -type f
# expected: $TL/build_tmp/tensilelite/client/tensilelite-client
```

Tensile finds the client in `build_tmp` automatically. If you built somewhere else, add `--prebuilt-client=<path>/tensilelite-client` to every `Tensile/bin/Tensile` command below.

## 3. Run one config (wall time)

```bash
source ~/tensile-venv/bin/activate
cd $TL
Tensile/bin/Tensile flydsl_swapmn_PS_B128_BF16R/flydsl_mxf8_tn_33_swap_gsu1.yaml out_33
```

What Tensile does: generates the kernel assembly from the yaml, compiles it, builds a small library, then runs `tensilelite-client` to benchmark it.
A full run takes about 30-60 s and ends with `clientExit=0 (PASS)`.

The timing is printed as a CSV line in the console. The `time-us` column is the wall time:

```
run,problem-progress,solution-progress,operation,problem-sizes,...,solution,validation,time-us,gflops,...
0,0/0,0/1,Contraction_l_Alik_Bljk_Cijk_Dijk,"(7168,512,1,16384)",...,Cijk_Alik_Bljk_...,NO_CHECK,28.38,...
```

Extract it from a saved log:

```bash
Tensile/bin/Tensile flydsl_swapmn_PS_B128_BF16R/flydsl_mxf8_tn_33_swap_gsu1.yaml out_33 2>&1 | tee out_33.log
python3 - out_33.log <<'PY'
import csv, re, sys
rows = [l for l in open(sys.argv[1]) if re.match(r"^\d+,\d+/\d+,\d+/\d+,Contraction", l)]
r = next(csv.reader([rows[-1]]))
print(f"problem {r[4]}  validation {r[9]}  wall time {r[10]} us")
PY
```

Notes:

- The output folder (`out_33`) must not already contain a previous run of a different yaml. Delete it or use a new name.
- `validation` shows `NO_CHECK` because validation is turned off in these yamls (`NumElementsToValidate: 0`). For the `_gsuN` config, BF16 partials are expected to fail the strict FP32 reference check, so leave it off.
- The `KernelTime` setting in the yaml switches the timer: `false` = host wall clock, `true` = hipEvent on the GPU stream. With 1000 launches per measurement they differ by under 3 % (see section 6).

## 4. Device time with rocprofv3

After step 3, the run folder contains a client config for the winning kernel. Re-run that kernel under rocprofv3:

```bash
CLIENT=$(find $TL/build_tmp -name tensilelite-client -type f | head -1)
INI=$(ls out_33/4_LibraryClient/source/ClientParameters_*.ini | head -1)

rocprofv3 --kernel-trace --stats -d rocprof_33 -o run --output-format csv \
    -- $CLIENT --config-file $INI --best-solution
```

- `--kernel-trace`: records GPU start/end timestamps of every kernel launch.
- `--stats`: writes per-kernel summary statistics.
- `--best-solution`: runs only the winning kernel.

The output folder has two files that matter:

| File | Contents |
|---|---|
| `run_kernel_trace.csv` | One row per kernel launch, with `Start_Timestamp` / `End_Timestamp` in ns |
| `run_kernel_stats.csv` | One row per kernel name: `Calls, TotalDurationNs, AverageNs, MinNs, MaxNs, ...` |

The `dev_us` in the table above is the **median** per-launch duration from `run_kernel_trace.csv` (about 1002 launches: 2 warmups + 1000 timed).
`AverageNs` in `run_kernel_stats.csv` is the mean, which is pulled up by a few slow outliers. On 8192x512x1536 one run gave median 6.13 us, mean 6.47 us, min 5.17 us, max 11.7 us.
Run-to-run variation of the median is a few percent.

```bash
python3 - rocprof_33/run_kernel_trace.csv <<'PY'
import csv, statistics, sys, collections
d = collections.defaultdict(list)
for r in csv.DictReader(open(sys.argv[1])):
    d[r["Kernel_Name"]].append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)
gemm = post = 0.0
for n, xs in d.items():
    med = statistics.median(xs)
    print(f"median {med:8.3f} us  mean {statistics.mean(xs):8.3f} us  x{len(xs):>5}  {n[:80]}")
    if n.startswith("Cijk_BB_PostGSU"): post += med
    elif n.startswith("Cijk_"): gemm += med
print(f"device time (median) = GEMM {gemm:.3f} + PostGSU {post:.3f} = {gemm + post:.3f} us")
PY
```

How to read the rows:

| Row | Meaning |
|---|---|
| `Cijk_Alik_Bljk_F8BS_MXAE8B128x128_MXBE8B128_STA_...` | The GEMM kernel |
| `Cijk_BB_PostGSU<N>_VW<k>` | Split-K reduction kernel. Only present for `_gsuN` with MultipleBuffer. Add it to the GEMM time. |
| `__amd_rocclr_copyBuffer` | Data copies by the runtime. Do not count. |

MBSK does the reduction inside the GEMM kernel, so it has only the `Cijk_...` row.

## 5. Run all six configs (wall + KernelTime + device time)

```bash
source ~/tensile-venv/bin/activate
cd $TL
D=flydsl_swapmn_PS_B128_BF16R
CLIENT=$(find $TL/build_tmp -name tensilelite-client -type f | head -1)

for y in $D/*.yaml; do
  n=$(basename $y .yaml)
  for kt in false true; do
    sed "s/KernelTime: false/KernelTime: $kt/" $y > /tmp/${n}_kt$kt.yaml
    rm -rf out_${n}_kt$kt
    Tensile/bin/Tensile /tmp/${n}_kt$kt.yaml out_${n}_kt$kt > out_${n}_kt$kt.log 2>&1
  done
  INI=$(ls out_${n}_ktfalse/4_LibraryClient/source/ClientParameters_*.ini | head -1)
  rm -rf rocprof_$n
  rocprofv3 --kernel-trace --stats -d rocprof_$n -o run --output-format csv \
      -- $CLIENT --config-file $INI --best-solution > rocprof_$n.log 2>&1
done

python3 - $D <<'PY'
import csv, glob, re, sys, os, statistics, collections
def wall(log):
    rows = [l for l in open(log) if re.match(r"^\d+,\d+/\d+,\d+/\d+,Contraction", l)]
    r = next(csv.reader([rows[-1]])); return r[4], r[10]
def dev(trace):
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(trace)):
        if r["Kernel_Name"].startswith("Cijk_"):
            d[r["Kernel_Name"]].append((int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3)
    return sum(statistics.median(xs) for xs in d.values())   # GEMM (+ PostGSU)
print(f"{'yaml':34} {'problem':22} {'wall_us':>9} {'kerneltime_us':>14} {'dev_us':>8}")
for y in sorted(glob.glob(f"{sys.argv[1]}/*.yaml")):
    n = os.path.basename(y)[:-5]
    prob, w = wall(f"out_{n}_ktfalse.log"); _, k = wall(f"out_{n}_kttrue.log")
    print(f"{n:34} {prob:22} {float(w):9.2f} {float(k):14.2f} {dev(f'rocprof_{n}/run_kernel_trace.csv'):8.2f}")
PY
```
