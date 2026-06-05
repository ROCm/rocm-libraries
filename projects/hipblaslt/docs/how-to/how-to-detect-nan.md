# Detecting NaN in hipBLASLt GEMM Output

hipBLASLt ships an opt-in NaN scanner (`HIPBLASLT_CHECK_NUMERICS`) that can
tell you **which matmul call first produced NaN** in your training run.
Set five environment variables, re-run your job, and read the answer from
a log file — no changes to your training code.

> **Requires** a hipBLASLt build that includes the NaN-scanner feature
> (hipBLASLt PR 7423). Stock ROCm releases prior to that PR will
> silently ignore `HIPBLASLT_CHECK_NUMERICS*` env vars and emit nothing.
> See [§1.1 — Verify the scanner is active](#11-verify-the-scanner-is-active)
> to confirm your build supports it before running a long job.

---

## 1. Set the environment variables

```bash
export HIPBLASLT_CHECK_NUMERICS=warn
export HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY=1000
export HIPBLASLT_CHECK_NUMERICS_STOP_ON_FIRST=1
export HIPBLASLT_LOG_MASK=160
export HIPBLASLT_LOG_FILE=/tmp/hipblaslt.log
```

Then run your training script normally. No code changes needed.

| Variable | What it does |
|---|---|
| `HIPBLASLT_CHECK_NUMERICS=warn` | Turns the scanner on. |
| `HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY=1000` | Check 1 out of every 1000 matmul calls. +0.04% overhead (1-in-1000 sampling). |
| `HIPBLASLT_CHECK_NUMERICS_STOP_ON_FIRST=1` | Stop scanning and auto-log when the first NaN is found. |
| `HIPBLASLT_LOG_MASK=160` | Write per-matmul details to the log file. `160 = bits 5+7` (per-call API bench line + solution-index/profile line). Log size scales with matmul call count (≈2.5 KB/call). Set a smaller mask to reduce size at the cost of less per-call detail; the workflow below requires bit 5 (the bench line — exactly one per matmul call). See [§5 — Troubleshooting](#5-troubleshooting) for the bit-5-only setting. |
| `HIPBLASLT_LOG_FILE=...` | Where to write the log. **For multi-GPU runs, give each rank a separate file** — every rank writes to this path, so a shared filename would interleave or truncate across ranks. Use whatever per-rank scheme your launcher provides, e.g. `/tmp/hipblaslt.${HOSTNAME}.${LOCAL_RANK:-0}.log`, `/tmp/hipblaslt.${SLURM_LOCALID}.log`, `/tmp/hipblaslt.$$.log` (per-PID), etc. |

> Note that `STOP_ON_FIRST` is a per-process flag — each rank stops
> scanning on its own first NaN independently — so multiple ranks may
> log a `CHECK_NUMERICS` line. If so, work from the rank with the
> earliest reported `call_id`.

`STOP_ON_FIRST=1` is critical: it makes the scanner **auto-emit a log
line as soon as NaN is detected** and then suppress further scans — no
GPU sync, no manual drain, no Python shim. Works with PyTorch out of the
box.

---

## 1.1 Verify the scanner is active

Before kicking off a long training run, do a ~10-second sanity check. A
hipBLASLt build that's missing PR 7423 will silently ignore the
`CHECK_NUMERICS*` env vars, so it's worth confirming once per environment
(new container, new ROCm install, new `LD_LIBRARY_PATH`, etc.).

```bash
LOG=/tmp/hipblaslt_smoke.log
rm -f "$LOG"
HIPBLASLT_CHECK_NUMERICS=warn \
HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY=1 \
HIPBLASLT_CHECK_NUMERICS_STOP_ON_FIRST=1 \
HIPBLASLT_LOG_MASK=160 \
HIPBLASLT_LOG_FILE="$LOG" \
python -c '
import torch
x = torch.full((128, 128), float("nan"), device="cuda")
# Need >=2 matmuls: the auto-drain log line fires when the NEXT host-peek
# sees a non-zero flag, so one matmul alone produces no CHECK_NUMERICS line.
for _ in range(5):
    (x @ x).cpu()
'
echo "--- bench lines (env vars reaching hipBLASLt): $(grep -c '^hipblaslt-bench' "$LOG")"
echo "--- CHECK_NUMERICS lines (scanner is active): $(grep -c CHECK_NUMERICS "$LOG")"
```

Expected output on a working build:

```
--- bench lines (env vars reaching hipBLASLt): 5
--- CHECK_NUMERICS lines (scanner is active): 1
```

And `grep CHECK_NUMERICS "$LOG"` will show one auto-drain line:

```
[hipBLASLt CHECK_NUMERICS] auto-drain on host peek:
  first NaN observed at sampled matmul call #1,
  effective window [1..2], mode=2, scan_every=1.
  (STOP_ON_FIRST: further scans suppressed after this call.)
```

Interpretation:

| bench lines | CHECK_NUMERICS lines | Diagnosis |
|---|---|---|
| `0` | `0` | `LD_LIBRARY_PATH` is not resolving to hipBLASLt, or `HIPBLASLT_LOG_FILE` is unwritable. Check `ldd $(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')/libtorch_hip.so \| grep hipblaslt`. |
| `>=1` | `0` | Env vars are flowing, but **your hipBLASLt build does not include PR 7423**. Rebuild or use a patched image. |
| `>=1` | `>=1` | Scanner is active. You're good. |

> **Why the loop?** `STOP_ON_FIRST=1` emits the `CHECK_NUMERICS` log line via
> a **host-peek of the device flag on the NEXT matmul call**. The scan kernel
> for call #1 sets the flag; the host-peek for call #2 sees it non-zero,
> CAS-elects a logger thread, and emits the line. With only one matmul there
> is no "next" call to trigger the emission, so the line never appears in
> the log even though the scan happened. In real training this is never
> an issue (there are thousands of matmuls).

---

## 2. Read the result

After your job crashes (or you stop it), look at the log file:

### Step 1 — Find the call_id

```bash
grep 'CHECK_NUMERICS' /tmp/hipblaslt.log
```

Example output:

```
[2026-05-21 12:53:54.694][hipBLASLt CHECK_NUMERICS] auto-drain on host peek:
  first NaN observed at sampled matmul call #132000
  (true first NaN somewhere in (131000..132000] due to scan_every=1000),
  effective window [1..132054], mode=2, scan_every=1000.
  To bisect, re-run with
    HIPBLASLT_CHECK_NUMERICS_SCAN_FROM=131001
    HIPBLASLT_CHECK_NUMERICS_SCAN_UNTIL=132000
    HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY=1.
  (STOP_ON_FIRST: further scans suppressed after this call.)
```

Two important features of this line:
- The reported `call_id` (`132000`) is the **sampled** call that observed the
  NaN. With `SCAN_EVERY=1000`, the **true** first buggy matmul lies somewhere
  in the previous 1000 calls — the line tells you exactly that window.
- The line includes a **copy-pasteable bisect hint** (`SCAN_FROM`/`SCAN_UNTIL`/
  `SCAN_EVERY=1`) you can use on a re-run to pinpoint the exact call_id
  (only useful for deterministic bugs — see §3 below).

### Step 2 — Look up the matmul

```bash
sed -n '132000p' /tmp/hipblaslt.log
```

This works because, with `HIPBLASLT_LOG_MASK=160`, **each matmul call emits
exactly one "bench" line at the top of the log file with no header lines
above it** — so log line N corresponds to matmul `call_id` N. If you change
the log mask (or prepend anything to the file) this 1-to-1 mapping breaks
and the `sed` lookup will silently return the wrong matmul; adjust the line
offset accordingly, or grep for the `call_id` directly instead.

This prints one line with the matmul's shape, data types, transpose modes,
and `solution_index`. Example:

```
-m 128 -n 128 -k 1280 --transA T --transB N
  --a_type R_32F --b_type R_32F --d_type R_32F --solution_index 482365 ...
```

### Step 3 — Get the kernel name

```bash
grep 'solution_index: 482365,' /tmp/hipblaslt.log | head -1
```

### All three steps in one line

```bash
CID=$(grep -oP 'matmul call #\K[0-9]+' /tmp/hipblaslt.log | head -1) && \
echo "Call ID: $CID" && \
sed -n "${CID}p" /tmp/hipblaslt.log
```

### Enumerate candidate kernels in the 1000-call window

Since `SCAN_EVERY=1000`, the actual buggy call is somewhere in calls
`(CID-999)..CID` inclusive — i.e. the half-open window `(CID-1000, CID]`
that the scanner reports. With `CID=132000`, that's lines `131001..132000`:

```bash
CID=132000
sed -n "$((CID - 999)),${CID}p" /tmp/hipblaslt.log \
  | grep -oP -- '--solution_index \K[0-9]+' \
  | sort | uniq -c | sort -rn
```

---

## 3. Narrowing down: deterministic vs non-deterministic bugs

Some NaN bugs fire at the same matmul call on every run (deterministic);
others fire at different calls each run (HBM-contention races, etc.).

**Deterministic bugs:** re-run with the printed `SCAN_FROM`/`SCAN_UNTIL`/
`SCAN_EVERY=1` bisect hint. The next log line will report the exact
call_id of the buggy matmul.

**Non-deterministic bugs:** the bisect hint won't help — the bug fires at
a different `call_id` each run. Just re-run from the start with
`SCAN_EVERY=1` (and keep `STOP_ON_FIRST=1`). The scanner short-circuits
after the first NaN, so the per-call overhead only applies until then.
The next log line will report the exact buggy `call_id`.

---

## 4. What the scanner covers

| Covered | Not covered |
|---|---|
| NaN in **output (D matrix)** of `hipblasLtMatmul` | NaN from other libraries (aiter, CK, MIOpen, rocBLAS) |
| dtypes: f32, f64, f16, bf16, fp8 (e4m3, e5m2) | NaN in matmul **inputs** (A, B, C) |
| Standard and grouped-GEMM paths | Device-side user-args paths |
| | Sub-byte packed types (fp6, fp4) |
| | Non-GEMM operations (attention, convolution, etc.) |

**Note on NaN-in-inputs:** the scanner only inspects the output `D`. If a
NaN enters a matmul via `A`, `B`, or `C` (e.g. produced by an earlier
non-GEMM op), the scanner will flag the **first matmul that propagates it
to `D`**, not the op that originally produced it. The reported `call_id`
is therefore the earliest GEMM-visible symptom, which may be one or more
ops downstream of the true source.

---

## 5. Troubleshooting

### "I set the env vars but never see a `CHECK_NUMERICS` line."

Three possibilities, in order of likelihood:

1. **No NaN actually fired** during the run. Confirm with the smoke test
   in [§1.1](#11-verify-the-scanner-is-active) — if that produces a
   `CHECK_NUMERICS` line, your setup is fine and the bug just didn't
   trigger this run.
2. **`LD_LIBRARY_PATH` is resolving to an unpatched hipBLASLt** that
   silently ignores the env vars. The §1.1 smoke test's "bench lines ≥ 1
   but CHECK_NUMERICS lines = 0" row diagnoses exactly this. Verify with:
   ```bash
   ldd $(python -c 'import torch, os; \
       print(os.path.join(os.path.dirname(torch.__file__), "lib/libtorch_hip.so"))') \
       | grep hipblaslt
   ```
   and confirm the resolved `libhipblaslt.so` came from a build that
   includes PR 7423.
3. **The log file is being clobbered** by another process writing to the
   same path, or the path is unwritable. Try a per-PID path
   (`HIPBLASLT_LOG_FILE=/tmp/hipblaslt.$$.log`) and re-run.

### "My job NaNs in production, but the scanner reports no NaN in the bisect window on re-run."

This is a non-deterministic bug — see [§3](#3-narrowing-down-deterministic-vs-non-deterministic-bugs).
The narrowed-window bisect only works for bugs that fire at the same
`call_id` every run. For non-deterministic bugs, run the full 1000-call
window enumeration (§2 last code block) across multiple trials and look
for `solution_index` values that recur.

### "The log file is too big."

`HIPBLASLT_LOG_MASK=160` writes ≈2.5 KB per matmul call (bench line + a
profile/solution-index line). To roughly halve the size, drop bit 7 and
use **bit 5 only**:

```bash
export HIPBLASLT_LOG_MASK=32
```

The workflow in this doc still works with `mask=32` because Step 2 (`sed
-n '${CID}p'`) only needs the bench line. Step 3 (the `solution_index:
${SOL},` kernel-name grep) reads the profile line that bit 7 emits and
will return no result with `mask=32`; you'll still know the
`solution_index` from the bench line, just not the human-readable kernel
name. If you need the kernel name, keep `mask=160` and rotate /
compress logs out-of-band instead (e.g. with `logrotate` or
`gzip --rsyncable`).

Note: raising `HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY` does **not** shrink
the log — every matmul still emits its `LOG_MASK` lines regardless of
whether the scanner sampled it. `SCAN_EVERY` only trades bisect-window
granularity for scanner overhead.

---

## 6. Quick-reference cheat sheet

**Setup** (env vars only — no code changes):

```bash
export HIPBLASLT_CHECK_NUMERICS=warn
export HIPBLASLT_CHECK_NUMERICS_SCAN_EVERY=1000
export HIPBLASLT_CHECK_NUMERICS_STOP_ON_FIRST=1
export HIPBLASLT_LOG_MASK=160
export HIPBLASLT_LOG_FILE=/tmp/hipblaslt.log   # use a per-rank path for multi-GPU
```

**After a NaN crash** (single copy-pasteable pipeline):

```bash
LOG=/tmp/hipblaslt.log   # match what you set HIPBLASLT_LOG_FILE to

# 1. Find the call_id that first observed NaN
CID=$(grep -oP 'matmul call #\K[0-9]+' "$LOG" | head -1)
echo "Call ID: $CID"

# 2. Print the matmul shape + dtype + solution_index for that call
sed -n "${CID}p" "$LOG"

# 3. Extract solution_index and find the kernel-name line
SOL=$(sed -n "${CID}p" "$LOG" | grep -oP -- '--solution_index \K[0-9]+')
grep "solution_index: ${SOL}," "$LOG" | head -1
```
