# run_tests.py — Parallel rocBLAS Test Runner

Run tests with concurrent execution, persistent state, and live progress display.
Suitable for running long tests on simulation environments.

All dependencies are Python 3.8+ stdlib only.

## Job Groups

| Group        | Jobs    | Description                                                   |
|--------------|---------|---------------------------------------------------------------|
| AUXILIARY    |  13     | Single-name utility/sanity tests                              |
| L1_BLAS      |  42     | 14 functions × 3 variants (plain / batched / strided_batched) |
| L1_BLAS_EX   |  18     | 6 functions × 3 variants                                      |
| L2_BLAS      |  72     | 24 functions × 3 variants                                     |
| **Total**    | **145** |                                                               |

## Quick Reference

```bash
# Run everything (resumes automatically if state file exists)
python3 run_tests.py

# Run with 10 parallel jobs, custom executable, custom output dir
python3 run_tests.py -j 10 -e /opt/rocm/bin/rocblas-test -o /tmp/results

# Re-run from scratch (ignore previous state)
python3 run_tests.py --reset

# Run only the L2_BLAS group
python3 run_tests.py --group L2_BLAS

# Run two specific jobs
python3 run_tests.py --job L1_BLAS.dot --job L2_BLAS.gemv_batched

# Skip tests matching a GTest pattern across all jobs (repeatable)
python3 run_tests.py --exclude-pattern "*f32_c_LNN*"
python3 run_tests.py --exclude-pattern "*f32_c_LNN*" --exclude-pattern "*f32_c_LNU*"

# List all valid job IDs
python3 run_tests.py --list-jobs

# Plain output — no ANSI, safe for tee / CI logs
python3 run_tests.py --no-color 2>&1 | tee run.log
```

## CLI Options

| Flag                    | Default                      | Description                                                                       |
|-------------------------|------------------------------|-----------------------------------------------------------------------------------|
| `-e`, `--executable`    | `/opt/rocm/bin/rocblas-test` | Path to rocblas-test binary                                                       |
| `-o`, `--output-dir`    | `<cwd>/tests_output`         | Directory for log files and `run_state.json`                                      |
| `-j`, `--max-parallel`  | `8`                          | Maximum concurrent test jobs                                                      |
| `--group GROUP`         | _(all)_                      | Limit to one group; repeatable                                                    |
| `--job JOB_ID`          | _(all)_                      | Run a specific job by ID; repeatable                                              |
| `--list-jobs`           | —                            | Print all job IDs and exit                                                        |
| `--reset`               | —                            | Delete state file and start fresh                                                 |
| `--skip-failed`         | —                            | On partial resume, also skip previously failed tests (re-run only untested tests) |
| `--exclude-pattern PAT` | _(none)_                     | GTest pattern appended to every job's negative filter; repeatable                 |
| `--count-interval SEC`  | `30`                         | How often (seconds) to re-count in-progress test results                          |
| `--no-color`            | —                            | Plain output, no ANSI escape codes                                                |

## Resume Behaviour

Interrupted runs (SSH drop, OOM kill, Ctrl+C) are resumed automatically on
the next invocation:

- Jobs whose recorded PID is **still alive** are waited on (reattached via
  `waitpid`) — no duplicate execution.
- Jobs whose recorded PID is **dead** are reset to `not_started` and re-run;
  the old log is archived (see Output Files below).
- **Partial resume**: when a job is re-run (whether it was interrupted
  mid-run or completed with failures), only tests that did **not pass** in
  the previous log are re-executed.  Passed test names are appended to the
  GTest filter negative section so they are skipped.
- With `--skip-failed`: previously **failed** tests are also excluded — only
  tests that were never attempted at all are re-run.  `--skip-failed` never
  skips a whole job; it only filters individual tests within the partial resume.
- With `--exclude-pattern`: the supplied GTest pattern is appended to every
  job's negative filter on **every** invocation, regardless of resume state.
  Intended for known-broken tests (e.g. simulator crashes) that should be
  skipped without modifying job definitions.  Patterns are **not** persisted in
  `run_state.json` and must be supplied each time the script is invoked.
  Multiple patterns can be combined with repeated flags.
- Whole jobs that previously passed are always skipped (they are never
  re-queued).  Use `--reset` to re-run them.
- Use `--reset` to force a completely clean start.

## Output Files

All output goes to `--output-dir` (default: `<cwd>/tests_output`):

| File                           | Description                                                  |
|--------------------------------|--------------------------------------------------------------|
| `run_state.json`               | Persistent job state (written atomically)                    |
| `{GROUP}.{variant}.txt`        | stdout+stderr log for each job                               |
| `{GROUP}.{variant}.txt.prev`   | Archived log from first interrupted run                      |
| `{GROUP}.{variant}.txt.prev.N` | Archived logs from subsequent interrupted runs (N = 1, 2, …) |

## Display Modes

**TTY / interactive** (`sys.stdout.isatty()` = True):
- ANSI dashboard refreshed every 0.5 s showing group progress and currently
  running jobs with elapsed time.

**Non-TTY / pipe** (or `--no-color`):
- Plain timestamped lines: `[MM:SS] STARTED L2_BLAS.gemv_batched`

## Exit Codes

| Code  | Meaning                       |
|-------|-------------------------------|
| `0`   | All selected jobs passed      |
| `1`   | One or more jobs failed       |
| `130` | Interrupted by SIGINT/SIGTERM |

## Caveats

- **PID reuse**: `os.kill(pid, 0)` may hit a recycled PID on resume. Use
  `--reset` if you suspect this has happened.
- **Concurrent instances**: two instances targeting the same `--output-dir`
  will corrupt `run_state.json`. Don't do this.
- **Executable mismatch**: if the binary path differs from what's recorded in
  the state file, a warning is printed. Use `--reset` if you changed the
  binary.
