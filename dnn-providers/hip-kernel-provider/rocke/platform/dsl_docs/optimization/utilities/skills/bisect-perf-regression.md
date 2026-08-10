---
name: bisect-perf-regression
description: >
  Find the exact commit that caused a GPU kernel performance regression using
  binary search (git bisect). Given a good commit (fast), a bad commit (slow,
  defaults to HEAD), and a benchmark command, automatically checks out commits,
  runs the benchmark, extracts the metric, and narrows down to the offending
  commit. Reports the regression commit with its diff and suggested root cause.
  Usage: /bisect-perf-regression <good_commit> [bad_commit] -- <bench_cmd>
allowed-tools: Bash Read Grep Glob Agent
---

# Bisect Performance Regression

Find the exact git commit that introduced a kernel performance regression using
binary search.

## Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `<GOOD_COMMIT>` | Yes | — | Commit hash or tag where performance was acceptable |
| `<BAD_COMMIT>` | No | `HEAD` | Commit hash where performance has regressed |
| `<BENCH_CMD>` | Yes | — | Benchmark command that prints a performance metric |

The arguments are parsed from the user's input. Typical invocations:

```
/bisect-perf-regression abc1234 def5678 -- python bench_pa.py --batch 32
/bisect-perf-regression v0.2.0 -- pytest tests/test_perf.py -k test_decode
/bisect-perf-regression abc1234 -- ./run_bench.sh
```

If any required argument is missing, ask the user before proceeding.

## Prerequisites

- Must be inside a git repository
- Working tree should be clean (no uncommitted changes) — the skill will
  `git stash` if needed and restore at the end
- The benchmark command must be runnable at every commit in the range
  (dependencies must be compatible)
- If a build step is needed between checkouts (e.g., `pip install -e .`),
  the user must include it in the bench command or specify it separately

## Algorithm

```
Binary search over commits between GOOD and BAD:

1. Establish baseline: run bench at GOOD, run bench at BAD
2. Verify regression exists: bad_metric must be significantly worse than good_metric
3. Bisect: pick midpoint commit, run bench, classify as good or bad
4. Repeat until a single commit is identified
5. Report the offending commit with diff and analysis
```

---

## Step 0: Validate Environment

Before starting, verify the environment is ready:

```bash
# Must be in a git repo
git rev-parse --is-inside-work-tree

# Check that both commits exist
git cat-file -t <GOOD_COMMIT>
git cat-file -t <BAD_COMMIT>

# Check working tree is clean
git status --porcelain
```

If working tree is dirty:
1. Show the user what's uncommitted
2. Ask: "Stash uncommitted changes before bisecting? They will be restored afterward."
3. If approved: `git stash push -m "bisect-perf-regression: auto-stash"`
4. Set a flag to `git stash pop` at the end

Save the current branch/commit to restore later:

```bash
ORIGINAL_REF=$(git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD)
```

---

## Step 1: Enumerate Commits

List all commits in the bisect range:

```bash
git rev-list --reverse <GOOD_COMMIT>..<BAD_COMMIT>
```

Report to user:

```
Bisect range: <GOOD_COMMIT_SHORT>..<BAD_COMMIT_SHORT>
Total commits in range: N
Estimated bisect steps: ceil(log2(N))
```

If > 100 commits, warn the user about the time cost.
If 0 commits, the range is invalid — ask the user to check the commits.

---

## Step 2: Establish Baselines

Run the benchmark at both endpoints to confirm the regression exists
and to calibrate the metric.

### 2.1 Determine the performance metric

Ask the user how to extract the metric if not obvious. Common patterns:

| Benchmark Output | Extraction Method |
|------------------|-------------------|
| `Latency: 1.23 ms` | `grep -oP 'Latency:\s*\K[\d.]+'` |
| `Throughput: 456 GB/s` | `grep -oP 'Throughput:\s*\K[\d.]+'` |
| `kernel_time_us: 789` | `grep -oP 'kernel_time_us:\s*\K[\d.]+'` |
| JSON output `{"time": 1.23}` | `python3 -c "import json,sys; print(json.load(sys.stdin)['time'])"` |
| pytest duration | `grep -oP '\d+\.\d+s'` |

If the user doesn't specify, attempt auto-detection:
1. Run the bench command once at the current commit
2. Show the output and ask: "Which number is the performance metric?
   Should lower be better (latency) or higher be better (throughput)?"

The user must confirm:
- **Metric extraction command** (grep/awk/python one-liner)
- **Polarity**: `lower_is_better` (latency, time) or `higher_is_better` (throughput, bandwidth)

### 2.2 Run baselines

```bash
# Baseline: GOOD commit
git checkout <GOOD_COMMIT> --quiet
# Optional build step if user specified
<BUILD_CMD>
# Run benchmark (multiple times for stability)
for i in 1 2 3; do <BENCH_CMD> 2>&1 | <METRIC_EXTRACTION>; done
```

Take the **median** of 3 runs as the baseline value.

```bash
# Baseline: BAD commit
git checkout <BAD_COMMIT> --quiet
<BUILD_CMD>
for i in 1 2 3; do <BENCH_CMD> 2>&1 | <METRIC_EXTRACTION>; done
```

Report baselines:

```
Baseline results:
  GOOD (<GOOD_SHORT>): <metric> = <value> <unit>
  BAD  (<BAD_SHORT>):  <metric> = <value> <unit>
  Regression: <percentage>% <worse_direction>
```

### 2.3 Validate regression

Calculate regression percentage:

```python
if lower_is_better:
    regression_pct = (bad_value - good_value) / good_value * 100
else:
    regression_pct = (good_value - bad_value) / good_value * 100
```

- If regression < 5%: warn that the difference may be noise. Ask user to
  confirm the threshold or increase run count.
- If regression < 0%: the "bad" commit is actually faster — the commits may
  be swapped. Ask the user.

Set the **threshold** for classifying a commit as "bad":

```python
# A commit is "bad" if its metric is within 30% of the regression toward the bad value
# This accounts for noise and gradual changes
threshold = good_value + (bad_value - good_value) * 0.3  # for lower_is_better
```

Let the user override this threshold if needed.

---

## Step 3: Binary Search (Bisect)

### 3.1 Bisect loop

```python
commits = [list of commits from git rev-list]
lo = 0                    # index of last known good
hi = len(commits) - 1     # index of first known bad

step = 0
while lo + 1 < hi:
    step += 1
    mid = (lo + hi) // 2
    commit = commits[mid]

    # Checkout and benchmark
    git checkout <commit> --quiet
    <BUILD_CMD>
    results = [run_bench() for _ in range(3)]
    metric = median(results)

    # Classify
    if is_bad(metric, threshold):
        hi = mid
        verdict = "BAD"
    else:
        lo = mid
        verdict = "GOOD"

    print(f"Step {step}: {commit[:8]} = {metric} -> {verdict}  (remaining: {hi-lo-1})")

# The first bad commit is commits[hi]
regression_commit = commits[hi]
last_good_commit = commits[lo]
```

### 3.2 Step-by-step reporting

After each bisect step, report progress:

```
Step 1/7: testing abc1234... metric=1.45ms -> GOOD (6 commits remaining)
Step 2/7: testing def5678... metric=2.31ms -> BAD  (3 commits remaining)
Step 3/7: testing 789abcd... metric=1.52ms -> GOOD (1 commit remaining)
...
```

### 3.3 Handle edge cases

**Build failure at a commit**:
- If the build or benchmark fails (non-zero exit code), skip this commit
- Expand the search: try the adjacent commit in the same direction
- If 3 consecutive commits fail, ask the user for guidance

**Flaky results (close to threshold)**:
- If the metric is within 10% of the threshold, run 5 iterations instead of 3
- If still ambiguous, report it and ask the user to classify manually

**Merge commits**:
- By default, follow first-parent only: `git rev-list --first-parent`
- If the regression commit is a merge, offer to re-bisect within the merged branch

---

## Step 4: Report the Regression Commit

Once the bisect is complete:

```bash
# Show the offending commit
git log -1 --format='%H%n%an <%ae>%n%ai%n%s%n%n%b' <REGRESSION_COMMIT>

# Show the diff
git diff <LAST_GOOD>..<REGRESSION_COMMIT> --stat
git diff <LAST_GOOD>..<REGRESSION_COMMIT>
```

### 4.1 Generate the report

```
============================================================
PERFORMANCE REGRESSION BISECT RESULT
============================================================

Regression introduced by:
  Commit:  <full_hash>
  Author:  <author>
  Date:    <date>
  Message: <commit message>

Performance impact:
  Before (<LAST_GOOD_SHORT>): <metric> = <good_value>
  After  (<REGRESSION_SHORT>): <metric> = <bad_value>
  Regression: <pct>% <direction>

Files changed:
  <file_list with +/- line counts>

Bisect log:
  Step 1: <commit> = <value> -> GOOD
  Step 2: <commit> = <value> -> BAD
  ...

============================================================
```

### 4.2 Analyze the diff for root cause

Read the diff and look for common regression patterns:

| Pattern | Example | Likely Cause |
|---------|---------|--------------|
| Changed loop bounds | `range(N)` -> `range(N*2)` | More iterations, doubled work |
| Added synchronization | Added `s_barrier`, `tl.debug_barrier()` | Extra sync stalls |
| Changed tile sizes | `BLOCK_SIZE=64` -> `BLOCK_SIZE=32` | Worse occupancy or more iterations |
| Added memory ops | New `tl.load` / `gl.load` inside loop | More memory traffic |
| Changed dtype | `fp16` -> `fp32` | 2x memory bandwidth, 2x register pressure |
| Removed prefetch | Deleted double-buffer logic | Load latency exposed |
| Changed `waves_per_eu` | `waves_per_eu=2` -> `waves_per_eu=1` | Reduced occupancy |
| Added masking | New `tl.where` / boundary checks | Extra ALU + potential branch divergence |
| Refactored layout | Changed `BlockedLayout` params | Possible bank conflicts or non-coalesced access |
| Added `num_stages` change | `num_stages=1` -> `num_stages=2` | Triton pipelining change |

Provide a short root cause hypothesis based on the diff.

---

## Step 5: Cleanup

Restore the original state:

```bash
# Return to original branch/commit
git checkout <ORIGINAL_REF> --quiet

# Restore stashed changes if any
git stash pop  # only if we stashed in Step 0
```

Verify the working tree is back to its original state:

```bash
git status
git log -1 --oneline
```

---

## Complete Execution Script

Here is the full procedure as pseudocode for reference:

```python
# === INPUTS ===
good_commit = "<GOOD_COMMIT>"
bad_commit  = "<BAD_COMMIT>"   # default: HEAD
bench_cmd   = "<BENCH_CMD>"
build_cmd   = "<BUILD_CMD>"    # optional, default: ""
metric_cmd  = "<METRIC_EXTRACTION>"
lower_is_better = True         # or False for throughput
num_runs    = 3                # runs per commit

# === SAVE STATE ===
original_ref = run("git symbolic-ref --short HEAD 2>/dev/null || git rev-parse HEAD")
stashed = False
if run("git status --porcelain").strip():
    run("git stash push -m 'bisect-perf-regression: auto-stash'")
    stashed = True

# === ENUMERATE ===
commits = run(f"git rev-list --reverse {good_commit}..{bad_commit}").splitlines()
total = len(commits)
steps = ceil(log2(total))
print(f"Bisecting {total} commits (~{steps} steps)")

# === BASELINES ===
def bench(commit):
    run(f"git checkout {commit} --quiet")
    if build_cmd:
        run(build_cmd)
    values = []
    for _ in range(num_runs):
        output = run(f"{bench_cmd} 2>&1")
        val = float(run(f"echo '{output}' | {metric_cmd}"))
        values.append(val)
    return median(values)

good_val = bench(good_commit)
bad_val  = bench(bad_commit)
regression_pct = abs(bad_val - good_val) / good_val * 100

if lower_is_better:
    threshold = good_val + (bad_val - good_val) * 0.3
    is_bad = lambda v: v > threshold
else:
    threshold = good_val - (good_val - bad_val) * 0.3
    is_bad = lambda v: v < threshold

# === BISECT ===
lo, hi = -1, total  # -1 = good_commit, total = bad_commit (virtual indices)
# Map: -1 -> good_commit, 0..total-1 -> commits[], total -> bad_commit
def get_commit(idx):
    if idx == -1: return good_commit
    if idx == total: return bad_commit
    return commits[idx]

lo, hi = -1, total
step = 0
log_entries = []

while hi - lo > 1:
    step += 1
    mid = (lo + hi) // 2
    commit = get_commit(mid)
    try:
        val = bench(commit)
        bad = is_bad(val)
    except Exception as e:
        # Build/bench failure — skip this commit
        print(f"Step {step}: {commit[:8]} SKIP (error: {e})")
        # Try shifting mid toward hi
        mid += 1
        if mid >= hi:
            mid = (lo + hi) // 2 - 1
        if mid <= lo:
            print("Cannot find testable commit in range")
            break
        commit = get_commit(mid)
        val = bench(commit)
        bad = is_bad(val)

    if bad:
        hi = mid
        verdict = "BAD"
    else:
        lo = mid
        verdict = "GOOD"

    remaining = hi - lo - 1
    log_entries.append(f"Step {step}: {commit[:8]} = {val} -> {verdict} ({remaining} left)")
    print(log_entries[-1])

first_bad = get_commit(hi)
last_good = get_commit(lo)

# === REPORT ===
print(f"\nRegression commit: {first_bad}")
run(f"git log -1 {first_bad}")
run(f"git diff {last_good}..{first_bad}")

# === CLEANUP ===
run(f"git checkout {original_ref} --quiet")
if stashed:
    run("git stash pop")
```

---

## Configuration Options

The user may customize these via conversation:

| Option | Default | Description |
|--------|---------|-------------|
| `num_runs` | 3 | Benchmark iterations per commit (more = less noise) |
| `threshold_pct` | 30% | % of regression gap to classify bad (lower = stricter) |
| `build_cmd` | none | Command to run after checkout (e.g., `pip install -e .`) |
| `first_parent` | true | Follow first-parent only (skip merge internals) |
| `timeout` | 600s | Max time per benchmark run |
| `skip_on_fail` | true | Skip commits where build/bench fails |
| `warmup_runs` | 1 | Discard first N runs before measuring |

---

## Error Handling

- **Benchmark command not found**: Check PATH, suggest activating venv or conda
- **Build failure**: Show error, try adjacent commit, report if > 3 failures
- **Metric extraction fails**: Show raw output, ask user to fix the extraction
- **Git checkout conflict**: `git checkout -f` only if user approves
- **Interrupted (Ctrl+C)**: Restore original branch before exiting
- **Regression not found**: If all commits are "good", the regression may be
  environmental (driver, library, hardware thermal). Suggest running the bad
  commit again to confirm.

## Example Sessions

### Example 1: Paged Attention Latency Regression

```
User: /bisect-perf-regression a1b2c3d -- python bench_pa.py --batch 32

Claude: The bad commit defaults to HEAD. Let me enumerate the range...

Bisect range: a1b2c3d..f4e5d6c (HEAD)
Total commits: 23
Estimated steps: 5

Running baseline at GOOD (a1b2c3d)... 142.3 us
Running baseline at BAD  (f4e5d6c)... 198.7 us
Regression: 39.6% slower (latency, lower is better)
Threshold: 159.2 us

Step 1/5: 7890abc = 143.1 us -> GOOD (11 left)
Step 2/5: 1234def = 197.2 us -> BAD  (5 left)
Step 3/5: 5678abc = 145.8 us -> GOOD (2 left)
Step 4/5: 9012def = 196.5 us -> BAD  (1 left)
Step 5/5: 3456abc = 195.9 us -> BAD  (0 left)

============================================================
REGRESSION COMMIT FOUND
============================================================
Commit:  3456abcdef123456
Author:  developer@example.com
Date:    2026-02-28
Message: refactor: split KV loop into separate key and value phases

Changed files:
  pa_decode_gluon.py | 45 +++++++++++++++++++++------------------------

Root cause: The refactor moved value cache loads AFTER the QK MFMA
computation, breaking the previous load-compute overlap. Before this
commit, value loads were issued concurrently with QK MFMA; after, they
are sequential, adding ~55us of exposed load latency.
```

### Example 2: With Build Step

```
User: /bisect-perf-regression v0.1.0 v0.2.0 -- python -m pytest tests/bench.py -k "test_throughput" --tb=no

Claude: Does this project need a build step after checkout?

User: Yes, run pip install -e . after each checkout

Claude: Got it. Using build_cmd="pip install -e ."
...
```
