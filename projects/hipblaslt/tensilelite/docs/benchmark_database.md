# Benchmark database

## What it is and why

The client runs every solution on every problem size. A kernel that crashes, faults the GPU or hangs used to kill the client and lose the progress of the whole run.

With `--benchmark-db <path>` (or `benchmark-db=<path>` in `ClientParameters.ini`) the client records each (init config, solution) pair in a SQLite file. Running the client again with the same database resumes: pairs that already have a result are not launched again, and a kernel that killed the client is not launched again either. The option is off by default (empty path).

The contract:

- The `UNKNOWN` marker of a pair is committed durably before the pair is launched, so a crash or a node reset leaves it `UNKNOWN`.
- `PASS` or `FAILURE` is committed after the pair is validated.
- A pair left `UNKNOWN` is faulty and is never launched again.
- A faulty kernel is launched once: with `indirect-failures` on (the default), the remaining pairs of its solution are marked `INDIRECTLY_FAILED`, and with `indirect-verification-failures` on (the default), after a verification failure, except right after a session that died, `INDIRECT_VERIFICATION_FAILURE`; marked pairs are not launched.

| Value | Status | Meaning |
|-------|--------|---------|
| 0 | `NOT_BENCHMARKED` | Not started, or its problem did not finish. |
| 1 | `UNKNOWN` | Started and never finished: the client died while running it. |
| 2 | `PASS` | Finished with validation `PASSED` or `NO_CHECK`, or with no verdict. A pair whose solution the client declines to run (for example a rejected check or a skipped granularity) reports no verdict and is recorded as `PASS` with an empty validation. |
| 3 | `FAILURE` | Finished with a verdict other than `PASSED` or `NO_CHECK`. |
| 4 | `INDIRECTLY_FAILED` | Never launched: its solution left another pair `UNKNOWN` in a session that died. |
| 5 | `INDIRECT_VERIFICATION_FAILURE` | Never launched: its solution failed verification (verdict `FAILED`) on another pair, except right after a session that died. |
| 6 | `NOT_SELECTED` | Never launched: the solution iterator did not select it for its problem, for example by performance prediction. It is neither a failure nor a pass. |

### Schema at a glance

`schemaSql` in `client/src/BenchmarkDatabase.cpp` is the source of truth for the columns.

| Table | Purpose |
|-------|---------|
| `meta` | CSV options: `performance_metric`, `csv_export_extra_cols`, `csv_merge_same_problems`. |
| `solutions` | Complete solution spec, with `spec_format` `msgpack` or `yaml`. |
| `client_configs` | The client configuration as text, a single row. |
| `init_configs` | One row per problem. |
| `problem_values` | Per-problem values reported by the client. |
| `sessions` | One row per client launch. `end_time` stays NULL if the client died. |
| `runs` | One row per pair: status, session, times, validation, `caused_by_problem_id`. |
| `run_values` | Per-pair values as the text the results CSV uses. |
| `iterations` | Time and enqueue count of each sync. |

Invariants:

- `PRAGMA user_version` is 1. A file with tables and another or no version is rejected with a mismatch and left unmodified.
- `runs.status` has `CHECK` 0 to 6, and `caused_by_problem_id` is non-NULL exactly for statuses 4 and 5.
- `run_values` and `iterations` have a composite foreign key to `runs(problem_id, solution_id)`.
- `runs(solution_id, status)` is indexed.
- Solution specs are the deterministic msgpack encoding of the solution, or YAML when the build has no msgpack writer. If the serialization traits change (the set or order of written fields), opening an old database fails with `solution N spec differs` and the run needs a new database.

## Options

| Option | Default | Meaning |
|--------|---------|---------|
| `benchmark-db` | empty (off) | Path of the SQLite file. It is created and populated when it does not exist. |
| `indirect-failures` | `true` | Mark the remaining pairs of a solution that crashed, faulted or hung as `INDIRECTLY_FAILED`. |
| `indirect-verification-failures` | `true` | Mark the remaining pairs of a solution that failed verification as `INDIRECT_VERIFICATION_FAILURE`. |

The two indirect options need `benchmark-db`. Turning one off resets the pairs previously marked with its status to `NOT_BENCHMARKED` when the next session starts, so they are retried.

`benchmark-db`, `indirect-failures`, `indirect-verification-failures`, `help` and `config-file` are not part of the client configuration stored in the database, so changing them does not make a database mismatch. Every other option is part of it, including output-only ones such as `results-file`, `log-file`, `log-level`, `print-*`, `dump-tensors` and `library-update-*`, and must be unchanged on resume.

`init-seed=0` seeds the data with the current time, so a rerun draws different data; `sessions.init_seed` records the seed each session used.

### Restrictions

The client rejects these combinations with a diagnostic and exit status 1, before the database file is created:

| Combination | Message |
|-------------|---------|
| `num-benchmarks` greater than 1 | `benchmark-db cannot be combined with num-benchmarks greater than 1` |
| `fused-a2a` other than disabled | `benchmark-db cannot be combined with fused-a2a` |
| `selection-only` | `benchmark-db cannot be combined with selection-only` |
| More than one `icache-flush-args` value | `benchmark-db supports a single icache-flush-args value` |
| `best-solution` | `benchmark-db needs the all-solutions run, not best-solution` |
| A client without a solution writer | `benchmark-db needs a client built with the msgpack or YAML solution writer` |
| An option value that the config-file syntax cannot hold | `benchmark-db cannot record this configuration: ...` |

## Resume and rerun

On start the client opens the database, populating a new one or checking an existing one, and begins a session. Then, for each problem and solution:

1. A pair that is not `NOT_BENCHMARKED` is not launched. The client reports validation `SKIPPED_BY_DB` for it.
2. For a pending pair it commits `UNKNOWN`, runs the pair, then commits `PASS` or `FAILURE` together with the reported values and iterations.
3. After a verdict `FAILED`, the remaining pairs of the solution become `INDIRECT_VERIFICATION_FAILURE`.
4. A problem with no pending pair is skipped with `Skipping problem <n>: no pending runs in the benchmark database`.

With `indirect-failures` on, at the start of a session after one that died (its `end_time` is NULL), the pairs of every solution that has an `UNKNOWN` pair become `INDIRECTLY_FAILED`, with the first such problem as the cause. A complete database makes a rerun launch nothing.

### Pairs the iterator does not select

With `prediction-threshold` of 1.0 or less and a GPU with a performance model, the solution iterator ranks the solutions valid for a problem and runs only the best ones. When a problem finishes normally, every pair of that problem still `NOT_BENCHMARKED` becomes `NOT_SELECTED` in one transaction, with the session id recorded. Nothing else is set: no validation, times or cause.

- A problem that does not finish (a crash, an early exit, a lost device) marks nothing, so its remaining pairs stay `NOT_BENCHMARKED`. The resumed run computes the selection again, which is deterministic, and runs the selected pairs it has not finished.
- `NOT_SELECTED` is not pending. A resumed run neither launches nor changes such a pair, even when its selection differs, and a problem with only `NOT_SELECTED` pairs left is skipped.
- Solutions that are not valid for a problem (the iterator's check fails) are never ranked, so their pairs end `NOT_SELECTED`.
- When the prediction is unavailable (the GPU has no performance model) or `prediction-threshold` is above 1.0, every solution runs, in index order, and no pair becomes `NOT_SELECTED`, whatever the threshold. With 1.0 every valid solution is selected and the solutions run in the order of their prediction.
- `indirect-failures` and `indirect-verification-failures` only mark and reset `NOT_BENCHMARKED` pairs and their own statuses, so they never change a `NOT_SELECTED` pair.
- The legacy results CSV has an empty cell for a solution that was not visited, and so does the cell of a `NOT_SELECTED` pair.

Independently of `indirect-failures`, a verification failure in the first session after one that died is recorded but does not propagate to the other pairs of its solution, because it may be a false failure after a GPU fault.

The client compares an existing database with the run: solutions (count, name, spec format, spec), the client configuration and the init configs. A difference stops it with exit status 1 before a session starts:

```
benchmark database mismatch: client configuration differs at '<option>'
benchmark database mismatch: solution <index> is named '<name in database>' in the database but '<name in this run>' in this run
benchmark database mismatch: solution <index> spec differs
benchmark database mismatch: solution count differs: database has <n>, run has <m>
```

The client configuration contains absolute paths, so a configuration that names other paths is a mismatch. To start over, delete the `.db`, `-wal` and `-shm` files.

## Crash handling

- The session is a scope: `BenchmarkSession` begins it and the client ends it with `end()` once all problems are done, which sets `sessions.end_time`. An exception unwinding through the scope leaves the session open; a normal exit from the scope, such as the `exit-on-error` early return, ends it.
- `PairRecorder` commits `UNKNOWN` when it is created, before the first launch of the pair, and writes the verdict in `finish()`.
- A crash, hang, GPU fault or kill leaves the pair `UNKNOWN` and the session open. With `indirect-failures` on, the next session marks the remaining pairs of that solution `INDIRECTLY_FAILED`.
- A lost device (a sticky error after a failed pair, reported as `Device unusable (<error>), aborting.`) does the same: the pair stays `UNKNOWN`, the session stays open (the client unwinds with an exception) and the client exits with the clamped error count (at least 1).
- With `exit-on-error`, an early exit after a failure records the failed pair, ends the session and exits with the clamped error count (at least 1).
- A database failure raised during the session (in `beginSession`, `beginRun`, `endRun`, or `end()` on the normal path: an inconsistency or a failing SQLite call) prints `benchmark database error: ...` and exits with 1. The session stays open, except that a failing `beginSession` leaves no session. If `end()` fails in the destructor on the `exit-on-error` early return, the client prints `failed to end the benchmark session: ...`, leaves the session open and keeps the clamped error count. Errors raised while opening the database, such as a mismatch, exit with 1 before any session exists.
- On Linux the client in `benchmark-db` mode asks the kernel to SIGKILL it when its parent dies, so GPU work does not outlive the process that launched it. A client whose parent exits while the client is arranging the death signal exits with 1 and `benchmark-db client started after its parent exited` (a parent that died before the client started is not detected); if the request itself fails it exits with 1 and `benchmark-db cannot arrange to die with its parent: ...`.

When the client completes all problems, its exit status is the number of validation failures, capped at 255, as without a database.

## Durability

Each commit uses `synchronous=FULL` (one fsync). WAL is required: on a filesystem without it the client fails with `cannot enable write-ahead logging`. The `-wal` and `-shm` files must be copied together with the `.db` and never deleted alone.

Useful queries (`sqlite3 -readonly <path>`):

```sql
SELECT status, COUNT(*) FROM runs GROUP BY status;

SELECT c.problem_index, s.library_index, r.session_id
FROM runs r JOIN init_configs c ON c.id = r.problem_id JOIN solutions s ON s.id = r.solution_id
WHERE r.status = 1 ORDER BY c.problem_index, s.library_index;

SELECT c.problem_index, s.library_index, r.status, cc.problem_index AS caused_by
FROM runs r JOIN init_configs c ON c.id = r.problem_id JOIN solutions s ON s.id = r.solution_id
LEFT JOIN init_configs cc ON cc.id = r.caused_by_problem_id
WHERE r.status IN (4, 5) ORDER BY s.library_index, c.problem_index;
```

## Results and CSV

`tensilelite.BenchmarkDatabase` reads the database from Python. `formatSummary(summarize("run.db"))` formats the state of a run:

```
benchmark database: 18 pairs: 14 PASS, 0 FAILURE, 1 UNKNOWN, 0 NOT_BENCHMARKED, 3 INDIRECTLY_FAILED, 0 INDIRECT_VERIFICATION_FAILURE, 0 NOT_SELECTED
pairs left UNKNOWN (the client died while running them):
  problem 0, solution 4 <solution name>
pairs not launched because their solution crashed, faulted or hung on another problem:
  solution 4 <solution name>: 3 pairs, caused by problem 0
sessions that died: 1
```

Sections list the `UNKNOWN` pairs, the `FAILED` pairs, the `INDIRECTLY_FAILED` and `INDIRECT_VERIFICATION_FAILURE` pairs per solution and causing problem, the tainted failures (failed right after a session that died) and the sessions that died. `NOT_SELECTED` pairs are only counted, at the end of the first line: they are neither failures nor passes. A section with no pairs is omitted.

```python
from contextlib import closing
from tensilelite.BenchmarkDatabase import RunStatus, getPairs, openDatabase

with closing(openDatabase("run.db")) as connection:
    for pair in getPairs(connection, [RunStatus.UNKNOWN]):
        print(pair.problemIndex, pair.libraryIndex, pair.sessionId)
```

Main functions: `openDatabase`, `getPairs`, `getSessions`, `countPairsByStatus`, `summarize`, `formatSummary`, `writeResultsCsv`.

`writeResultsCsv` writes the results CSV from the database. The CSV has the layout and values of the client's own results CSV, and is byte-identical to it for a run that finished in a single session with every pair passed. It writes a temporary file and replaces the CSV atomically, from a single read snapshot. A pair that is not `PASS` has the cell `-1`, except a `NOT_SELECTED` pair, whose cell is empty as in the client's CSV because the client never visited it. A stored value that cannot be read raises `BenchmarkDatabaseError`. With `csv-export-extra-cols`, the best-of columns (winner, best time, bandwidth) consider only `PASS` pairs here, whereas the client also folds in a pair that failed verification after its timing was reported, so those columns can differ for runs with verification failures.

The client adds a CSV column the first time it visits a solution, writes the header after the first problem and every row with the columns known after its problem. `writeResultsCsv` reproduces that: a `NOT_SELECTED` pair is not visited and creates no column, so with prediction a solution first selected by a later problem has a cell without a header, and the rows before it are shorter. With `csv-merge-same-problems` every row has all columns, the late ones empty where unvisited. Solutions with the same name share one column. The one inexact part is the order of the columns that the same problem adds for the first time: the reader uses the library index order, which is the client's order without prediction, while a prediction run visits the solutions in the order of its ranking, which the database does not store.

### CSV parity test

The gtest `BenchmarkCsvParity*` (`tests/BenchmarkCsvParity_test.cpp`) and `Tests/unit/test_benchmark_csv_parity.py` share the fixtures in `tests/data/benchmark_db`: scenarios `default`, `merge`, `extra_cols`, `cu_efficiency`, `prediction` (pairs the iterator did not select), `late_columns` and `late_columns_merge` (solutions first visited by a later problem), each with a golden `<name>.csv` and a dump `<name>.db.txt`. The gtest checks that the client's CSV and database for a scripted run equal the fixtures, and the Python test checks that `writeResultsCsv` writes the golden CSV from the dump, both byte for byte. The one deliberate deviation concerns pairs that were never reached: the gtest replays them as `SKIPPED_BY_DB` (cell `-1`), while a real client leaves the cell empty and the Python reader writes `-1`. A `NOT_SELECTED` pair is not reported in the gtest. To change a fixture, copy the text the gtest prints on a mismatch over the file and check the Python test.

## GPU warm-up probe

`tensilelite-gpu-probe [--device-idx N]` is a standalone tool built with the client and installed in the tests component under `libexec/hipblaslt/tensilelite` when `HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS` is enabled. It is independent of the client libraries. After a GPU memory fault the first process that does real GPU work can compute wrong results. The probe does that work on the chosen device (default 0): it allocates two 128 MiB buffers, runs a copy kernel, reads the result back and verifies it, so the next client starts on a healthy device.

| Exit status | Meaning |
|-------------|---------|
| 0 | Verified; prints `gpu probe passed on device <N> (128 MiB copy verified)`. |
| 1 | HIP error or mismatch; prints `gpu probe failed on device <N>: <reason>` on stderr. |
| 2 | Usage error; prints `usage: <program> [--device-idx N]` on stderr. |

The gtests `GpuProbeArgs*` and `GpuProbeExitCode*` check its argument parsing and exit codes without a GPU (the exit-code tests need only the HIP runtime library).

## Build requirements

SQLite is required. CMake looks for the ROCm sysdeps config package in `${ROCM_PATH}/lib/rocm_sysdeps` first, then the system `libsqlite3`; configure fails if neither is found. With `msgpack-cxx` or `msgpackc-cxx` the host stores specs as msgpack, otherwise as YAML; the configure output has a status line for it. A client built against a host with neither writer rejects `--benchmark-db`.

## Known limitations

- A hung pair blocks the run until the client is killed externally; the pair stays `UNKNOWN`.
- With `indirect-failures` off, a faulty kernel can cost one crash per problem.
- The client configuration contains absolute paths, and each rerun repeats the client setup.
- `INVALID` (an exception while launching) is a direct failure and does not propagate.
- Independently of `indirect-failures`, verification propagation is skipped in the first session after one that died.
- The crash marking runs at the start of a session, before the selection of the unfinished problems is known. A crashed solution's `NOT_BENCHMARKED` pairs of those problems become `INDIRECTLY_FAILED` even if the prediction would not have selected them, and they are not retried.
- The schema version was not bumped. A version 1 database that an earlier build of this unreleased schema created (status `CHECK` 0 to 5) must be deleted, otherwise the first `NOT_SELECTED` marking fails with a `CHECK` error.
- The death-signal protection needs Linux.
- The combinations in the Restrictions table are unsupported.
