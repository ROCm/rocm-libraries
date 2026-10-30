# "Per-solution bookkeeping & reporting" investigation (gfx950 per-config rerun 2)

`full_runs_gfx950/per_config_rerun_2/AGGREGATE_analysis.txt` reports **Per-solution bookkeeping &
reporting** at 68,333.60 ms (2.7% of wall). This note explains where that time goes and what can
be done about it. Timing claims come from measurements; code reading was only used to decide what
to measure. Inferences are marked as such.

The section [Correctness-only runs](#correctness-only-runs-what-can-be-turned-off) covers which of
this work, and of the per-launch timing work, can be switched off when Tensile is run only to check
correctness (the common tests), rather than to benchmark.

| Timer | Count | Total (ms) | Mean (µs) |
| --- | --- | --- | --- |
| `post_solution_log` | 2,355,016 | 28,760.07 | 12.2 |
| `pre_solution_listener.BenchmarkTimer` | 2,355,016 | 11,859.71 | 5.0 |
| `run_current_solution_check` | 2,355,016 | 8,420.00 | 3.6 |
| `post_solution_reporting` | 2,355,016 | 5,582.67 | 2.4 |
| `pre_solution_listener.ProgressListener` | 2,355,016 | 3,134.22 | 1.3 |
| `pre_solution_listener.AllSolutionsIterator` | 2,355,016 | 2,647.18 | 1.1 |
| `post_problem` | 28,641 | 1,945.83 | 67.9 |
| `run_iteration_teardown` | 2,106,835 | 1,677.91 | 0.8 |
| `pre_solution_listener.MetaResultReporter` | 2,355,016 | 1,159.39 | 0.5 |
| 12 smaller timers | | 3,146.62 | |

## Summary

There is no single slow function. The section is a fixed host-side cost of about 29 µs per
solution, paid 2,355,016 times. Weighted by solution count, the per-config cost is 24.1 µs (p10),
25.6 µs (p50) and 36.6 µs (p90). The 13 configs with at least 50k solutions hold 85.8% of the
solutions and 80.7% of the time. Four things make up that per-solution cost (shares measured on
dtl unless stated):

1. **Preemption by spinning OpenMP workers, about 15% of the section.** In the production logs,
   9.9 s of these timers is in individual calls longer than 1 ms, including timers that only do
   CPU work. After each OpenMP parallel region (e.g. the CPU reference), the client's 24 idle
   libomp workers keep spinning (`KMP_BLOCKTIME` defaults to 200 ms) on the same 25 cores as the
   main thread: 11–24 workers were measured busy during the solution loop. In the dtl replay,
   `KMP_BLOCKTIME=0` cut the >1 ms share from 1.8–4.1 s to 0.08–0.16 s.
   It also cut `cpu_reference_gemm` from 5.8–14.8 s to 2.4–2.7 s, which is outside this section.
2. **The `report()` fan-out, about 38%.** Each solution makes about 19 `report()` calls. Each is
   forwarded to 3 reporters, which format numbers with a fresh `std::ostringstream` (about 150 ns
   each), up to three times per value.
3. **One stdout CSV row per solution, about 28%.** `LogReporter` copies the 30-column row into a
   new `unordered_map`, formats a row of about 1.1 KB (mostly the solution name), and flushes
   stdout with `std::endl`. Over the aggregate, that is 2,335,808 rows and 2.64 GB of text that
   nothing parses: Tensile reads only the `results-file` CSV.
4. **Predicate debug dumps for rejected solutions, 16% in dtl and about 6% of the aggregate
   (estimate).** The default `ClientLogLevel` is 3 (Debug). For every solution that fails its
   predicates (443,391 in the aggregate, 18.8%), `checkSolution` calls `debugEval` on the whole
   predicate tree, which costs 8.9 µs per rejected solution.

A prototype that fixes 2 and 3 produced output identical to the shipped client and cut the dtl
bookkeeping section by 33%. Adding 4 (Terse log level) brought the cut to 58%, and fixing 1 removes
the tail on top of that (see [Options](#options)).

## Method

- **Binary.** The measured client, `/tmp/tl_client_msgpack/tensilelite/client/tensilelite-client`
  (Release, `-O3 -DNDEBUG -std=c++20 -x hip --offload-arch=gfx950 -fopenmp=libomp`).
- **Rebuilds.** Copies of `tensilelite/client` were rebuilt outside the repo with the exact
  compile and link lines from that build's `compile_commands.json` and `link.txt`, linked against
  the same `libtensilelite-host.so`. There are three copies:
  - `ref`: unmodified; its timings match the shipped binary within run-to-run noise.
  - `instr`: `ref` plus finer `ScopedTimer`s (`bk_*`) inside the timers above, and per-reporter
    timers in `MetaResultReporter`.
  - `opt`: the prototype described under [Options](#options).
- **Workload.** `common__gemm__gfx950__dtl` (443,808 solutions; the largest config in this section,
  at 10.9 s). It was first run through pytest exactly as `full_runs_gfx950/profiling/run_rerun.sh`
  does (cached artifact, `TimingInstrumentation=True`, `--prebuilt-client`). That run gave 9.8 s
  for the section, against 10.9 s in production. Its 26 client invocations were then replayed
  directly, in the same order with the same `ClientParameters.ini`. The replay matches the pytest
  run (e.g. `post_solution_log` 2,780 ms vs 2,700 ms).
- **Variants.** Each variant was run 2–3 times. They use the same binary and inputs, and change only
  `KMP_BLOCKTIME` (environment) or `log-level` (ini edit).
- **Microbenchmarks.** The formatting primitives were compiled with the same compiler and `-O3`.
  Timings are medians of 9 repetitions.
- **Production logs.** All 184 `per_config_rerun_2` logs were scanned for per-call distributions
  and for stdout content (CSV rows, validation outcomes, predicate dumps).
- **Environment.** gfx950 host, 25 idle cores via `taskset`, `OMP_NUM_THREADS=25`. The host is
  shared: every GPU was 69–97% busy with other users' work. All conclusions are drawn from
  host-only timers. Whole-client wall times moved by ±20% between repeats, because of GPU-bound
  timers such as `gpu_input_reset`, and are not used.
- Scratch scripts, summaries and logs are in `/tmp/bookkeep`: `build.sh`, `direct.sh`, `summ.sh`,
  `cmp.sh`, `tail.sh`, `rej.sh`, `client_src/` (instr), `client_opt/` (opt) and `micro/fmt.cpp`.

## Where the time goes (one solution, dtl, instrumented build)

Means per call in µs. Each nested timer adds about 42 ns (calibrated `timing_overhead`), so the
instrumented parents read higher than in the `ref` build; both are shown.

| Timer / sub-step | `ref` mean | `instr` mean | Notes |
| --- | --- | --- | --- |
| `post_solution_log` | 6.62 | 6.12 | |
| ├─ `readCurrentRow` (copy 30 columns into a new map) | | 2.40 | done for every row, printed or not |
| ├─ `writeCurrentRow`: format the row | | 1.67 | `escape()` copies each value again |
| ├─ `writeCurrentRow`: `std::endl` flush | | 1.14 | one `write()` of about 1.1 KB per solution |
| └─ `pop` (restore the row from the stack) | | 0.39 | |
| `pre_solution_listener.BenchmarkTimer` | 4.91 | 6.62 | |
| ├─ `projectedPerformance` | | 0.27 | |
| └─ 9 `report()` calls | | 6.28 | about 0.5 µs per report after removing timer overhead |
| `run_current_solution_check` (average) | 5.05 | 5.18 | |
| ├─ predicate evaluation (all solutions) | | 0.60 | |
| ├─ hardware predicate (all solutions) | | 0.25 | |
| └─ reject path (173,006 rejected solutions) | | 10.44 | per rejected solution |
|     ├─ `debugEval` of problem and task predicates | | 8.86 | |
|     └─ `log()` of the dump | | 0.57 | |
| `post_solution_reporting` (4 double `report()` calls) | 1.73 | 2.50 | |
| `pre_solution_listener.ProgressListener` | 1.20 | 1.62 | `solution->name()` 0.21; 2 reports |
| `pre_solution_listener.AllSolutionsIterator` | 1.12 | 1.64 | 3 reports, one built with `concatenate` |
| `pre_solution_listener.MetaResultReporter` | 0.48 | 0.36 | `CSVStackFile::push` copies the row map |

### The `report()` fan-out

Per solution, the listeners make 19.2 `report()` calls: 10.0 double, 4.0 int, 3.1 string and 2.1
uint. `MetaResultReporter` forwards each call to 3 reporters; `LibraryUpdateReporter` is null
here. Including timer overhead, the per-call cost by reporter is:

| Reporter | Mean per call (µs) | What it does with a double |
| --- | --- | --- |
| `ResultFileReporter` | 0.273 | `ostringstream << value` to get a string, compares the key against about 8 keys, then `setValueForKey(key, value)`, which formats the double again with a second `ostringstream` |
| `LogReporter` | 0.168 | `setValueForKey(key, double)`: `ostringstream << setprecision(6)` |
| `PerformanceReporter` | 0.023 | key comparisons only |

Microbenchmark of the primitives, using the client's compiler and `-O3`:

| Operation | ns |
| --- | --- |
| `ostringstream` + `setprecision(6)` + double → string | 153.6 |
| `ostringstream` + double → string | 148.7 |
| `ostringstream` + int64 → string | 50.5 |
| `std::to_chars(double, general, 6)` → string (same text) | 36.9 |
| `std::to_chars(int64)` → string | 4.7 |
| copy 30–33 columns into a new `unordered_map<string,string>` (the `readCurrentRow` pattern) | 1,016 |

The fan-out comes to 3.7 s of the dtl replay's 9.8 s (about 8.4 µs per solution) after removing
the instrumentation's own overhead. That overhead was 34M nested timers at 42 ns, or 1.4 s.

### The stdout CSV row

In the dtl baseline log, the client wrote 1.19 M non-TIMING lines (520 MB), counted by line type:

| Line type | Lines | Bytes |
| --- | --- | --- |
| CSV row per solution (30 columns; about 1,000 of 1,125 bytes is the solution name) | 443,808 | 499.5 MB |
| predicate debug dump lines (`[!!] ...`) | 382,108 | 20.1 MB |
| blank, `####` and `++++` separators | 365,548 | 0.9 MB |

Across all 184 production logs: 2,335,808 CSV rows (2.64 GB) and 923,357 dump lines (48.8 MB).
Tensile does not read any of it: `ClientWriter.py` starts the client with `subprocess.Popen` and no
`stdout=` argument, and `BenchmarkProblems.py` reads results only from `results-file`. In
`run_rerun.sh`, the client's stdout and stderr were a regular file (`/proc/<pid>/fd/1` → the log),
not a pipe.

### Predicate debug dumps for rejected solutions

`SolutionIterator::checkSolution` evaluates the problem and task predicates. If either fails and
`logAtLevel(Verbose)` is true, it calls `debugEval` on both trees. `debugEval` re-evaluates every
term (the `And` nodes have 35–39 terms) and formats a row per term into an `ostringstream`; the
result is then logged. `ClientLogLevel` defaults to 3 (Debug) in `Common/GlobalParameters.py`, and
no common test YAML overrides it, so every rejected solution pays for this.

| | dtl | aggregate (184 logs) |
| --- | --- | --- |
| solutions rejected by predicates (`DID_NOT_SATISFY_ASSERTS`) | 173,006 (39.0%) | 443,391 (18.8%) |
| `debugEval` + log per rejected solution | 9.4 µs | |
| `run_current_solution_check`, Debug → Terse (dtl, `KMP_BLOCKTIME=0`) | 2,109 → 473 ms | |

The Python enum has no `Normal`, so Tensile can only select Error, Terse, Verbose or Debug. At
Terse, rows of PASSED solutions (row level Normal) are not printed. Rows of rejected solutions
(row level Terse) are still printed, but without the dump.

## The >1 ms tail: spinning OpenMP workers

Share of each timer spent in calls longer than 1 ms, across all 184 production logs:

| Timer | Total (ms) | in calls > 1 ms |
| --- | --- | --- |
| `post_solution_log` | 28,760.1 | 4,364.0 (15.2%) |
| `pre_solution_listener.BenchmarkTimer` | 11,859.7 | 1,022.7 (8.6%) |
| `run_current_solution_check` | 8,420.0 | 1,110.1 (13.2%) |
| `post_solution_reporting` | 5,582.7 | 1,175.2 (21.1%) |
| `pre_solution_listener.ProgressListener` | 3,134.2 | 321.2 (10.2%) |
| `pre_solution_listener.AllSolutionsIterator` | 2,647.2 | 312.6 (11.8%) |
| `post_problem` | 1,945.8 | 970.9 (49.9%) |
| `run_iteration_teardown` | 1,677.9 | 305.7 (18.2%) |
| `pre_solution_listener.MetaResultReporter` | 1,159.4 | 353.7 (30.5%) |
| **all nine** | 65,187.0 | 9,936.0 (15.2%) |

Medians are far below the means. For example, `post_solution_log` in `spmm_i8hs` has a median of
14.9 µs, a p99 of 5.0 ms and a maximum of 42.7 ms. Pure-CPU timers such as
`pre_solution_listener.ProgressListener` show the same spikes, so the cause is not I/O.

The client links libomp, and its process has 27 threads: main, 24 OpenMP workers and 2 runtime
threads. All of them share the 25 cores of the `taskset` mask. Per-thread CPU time was sampled
from `/proc/<pid>/task/*/stat` during the solution loop:

| `KMP_BLOCKTIME` | Windows sampled | Workers above 25% CPU | Whole process, cores busy |
| --- | --- | --- | --- |
| default (200 ms) | 6 | 11–24 | 12–20 |
| 0 | 3 | 0 | 1.2 (main 1.07) |

Effect on the dtl replay, in ms (each row is one run):

| Variant | Section total | ≤ 1 ms calls | > 1 ms calls | `cpu_reference_gemm` |
| --- | --- | --- | --- | --- |
| default | 10,246 | 8,466 | 1,780 | 5,819 |
| default | 14,419 | 10,770 | 3,649 | 14,770 |
| default | 14,523 | 10,407 | 4,116 | 14,142 |
| `KMP_BLOCKTIME=0` | 9,362 | 9,240 | 122 | 2,420 |
| `KMP_BLOCKTIME=0` | 9,813 | 9,730 | 83 | 2,737 |
| `KMP_BLOCKTIME=0` | 10,585 | 10,423 | 162 | 2,670 |

With `KMP_BLOCKTIME=0`, the tail disappears and run-to-run variance drops. The ≤1 ms part does not
shrink; it is the real per-solution work covered by the other findings.

Inference, not measured: with spinning workers, 26–27 threads compete for 25 cores, and the main
thread is regularly descheduled for a full timeslice. That would explain both the millisecond
spikes and their spread across every timer on the main thread. The same mechanism probably inflates
other main-thread sections. The `cpu_reference_gemm` drop (2–5×) is large and belongs to the
Reference GEMM section, so it is listed under open items.

## Options

All measurements below come from the dtl replay with `KMP_BLOCKTIME=0`, so that the tail does not
hide the differences. Totals are for the whole section, in ms.

| Variant | Runs | Section total | `post_solution_log` | `BenchmarkTimer` pre | `run_current_solution_check` | `post_solution_reporting` | Progress + AllSolutions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ref` | 3 | 9,362 / 9,813 / 10,585 | 2,890 | 2,136 | 2,109 | 757 | 1,010 |
| `opt` (B) | 3 | 6,507 / 6,588 / 6,708 | 1,367 | 838 | 2,225 | 335 | 793 |
| `ref` + Terse (C) | 2 | 7,285 / 7,724 | 2,090 | 2,170 | 473 | 755 | 1,006 |
| `opt` + Terse (B + C) | 3 | 4,099 / 4,151 / 4,115 | 738 | 822 | 421 | 327 | 788 |

Per-timer columns are means over the runs. The section drops from 9.9 s to 6.6 s with B (−33%),
and to 4.1 s with B + C (−58%), or 9.3 µs per solution.

### A. Stop idle OpenMP workers from spinning

Set `KMP_BLOCKTIME=0` for the client, either in the environment `ClientWriter.py` passes to the
client, or with `kmp_set_blocktime(0)` or a smaller worker pool in `main.cpp`. `OMP_WAIT_POLICY=passive`
should have a similar effect, but was not measured. Measured effect: the >1 ms tail falls from
1.8–4.1 s to 0.08–0.16 s in dtl. In the aggregate, that tail is 9.9 s of this section. It costs
nothing in the measured runs: `cpu_reference_gemm`, the main OpenMP user, got faster, not slower.

### B. Cheaper formatting and row handling (prototype `opt`, output-identical)

The changes, all in the client:

1. `CSVStackFile::setValueForKey(double)` and the integral `setValueForKey<T>` use `std::to_chars`
   instead of `std::ostringstream`. For doubles, `std::chars_format::general` with precision 6
   produces the same text as `<< setprecision(6)`.
2. `ResultFileReporter::reportValue<T>` builds its value string with `std::to_chars` (once) instead
   of an `ostringstream`.
3. `LogReporter::postSolution` looks up the 3–7 keys it uses (`findCurrentValue`) instead of
   copying all 30 columns with `readCurrentRow`. The header is still written on the first row.
4. `CSVStackFile::writeRow` ends rows with `'\n'` instead of `std::endl`. `LogReporter::postProblem`
   flushes once per problem.

Correctness check: `ref` and `opt` were run on three dtl invocations (3,508, 320 and 60,940 stdout
lines). The results CSVs were byte-identical, and stdout differed only in the time-based
`srand seed` line. dtl runs no timed iterations, so its rows are deterministic.

Measured: −33% on the section (9.9 → 6.6 s). `post_solution_log` fell by 53%, `BenchmarkTimer`
pre by 61% and `post_solution_reporting` by 56%. The remaining Progress/AllSolutions cost is mostly
string reports and `concatenate`, which the prototype does not touch.

Trade-off of item 4: if the client crashes, up to one problem's worth of stdout rows can be lost.
The client already prints a `##########################` / `++++` pair per problem, so the flush
point matches an existing boundary.

Robustness note: in two consecutive runs, about 1.5 minutes in total, 37 of 52 `opt` invocations
crashed with SIGSEGV. The crashed invocations never printed `loading config file`, which is printed
with `std::endl` during argument parsing, before any reporter code runs. That points either to
crashes before the prototype's code executed, or to log writes failing. The root filesystem was at
100% use (27 GB free) before that batch and at 96% after it. The crash did not reproduce in 4
isolated reruns of the same invocation, or in the other 208 `opt` invocations (26 before, 182
after). It was not seen in any `ref` run. Root cause unknown.

### C. Don't dump predicate trees by default

Either lower the default `ClientLogLevel`, or gate the `debugEval` call in `checkSolution` behind
its own option and keep the per-solution rows. Measured with Terse: `run_current_solution_check`
fell from 2,109 to 473 ms in dtl (−9.4 µs per rejected solution). Scaled to the aggregate's
443,391 rejected solutions, that is about 4.2 s (estimate). The dump remains useful when debugging
why a kernel is rejected, so it should stay available behind an option. Terse also removes the
stdout rows of passing solutions, which costs log readability in CI when a test fails.

### D. Larger changes (not measured)

- Skip the stdout row entirely unless asked for. It is 2.64 GB of text in the aggregate that no
  tool reads; `readCurrentRow`/`writeCurrentRow` account for about 6 µs per solution, before B.
- Report the 9 granularity and memory values in `BenchmarkTimer::preSolution` only when a consumer
  needs them (CSV extra columns or the log). They are 9 of the 19 `report()` calls.
- Replace `MetaResultReporter`'s per-call virtual fan-out over `std::string` keys with typed keys.
  Each call also copies a `shared_ptr` per reporter (`for(auto r : m_reporters)`).

### Estimated effect on the aggregate

These figures are estimates: they scale the dtl measurements to the aggregate's counts. Other
configs have a different mix, for example 18.8% rejected solutions overall against 39% in dtl.

| Option | Basis | Estimated saving |
| --- | --- | --- |
| A | measured >1 ms share in the production logs | about 9.9 s |
| B | 7.5 µs per solution (dtl) × 2,355,016 | about 17.6 s |
| C | 9.4 µs per rejected solution × 443,391 | about 4.2 s |
| A + B + C | | about 32 s of 68.3 s |

## Correctness-only runs: what can be turned off

`Tensile input.yaml out` serves two purposes: benchmarking, and the correctness common tests. A
correctness run needs only three things:

- whether each solution produced a correct result;
- where it differed;
- the client's exit code (`main` returns the number of failed validations, clamped to 255; the test
  fails on a non-zero code);

Everything else the client does per solution is reporting or timing. That includes the
`results-file` CSV: it has to exist, but its values carry no information in a correctness run (see
[The results CSV in correctness runs](#the-results-csv-in-correctness-runs)).

### The results CSV in correctness runs

The client writes one row per problem size. Each row holds the sizes and one GFlops cell per
solution: `-nan` for a passed solution when nothing is timed, and `-1` for a failed or rejected
one. Python then uses it like this:

1. `BenchmarkProblems` copies it to `2_BenchmarkData/` (`shutil.copy`, which fails if the file is
   missing). Pass/fail does not come from the CSV: it is the client's exit code
   (`benchmarkTestFails`), followed by `sys.exit(1)` under `ExitOnFails`. No common-test harness
   file reads the CSV, `2_BenchmarkData` or `3_LibraryLogic`.
2. LibraryLogic reads it (`addFromCSV`) only for YAMLs with a `LibraryLogic:` section. In the
   production logs, 26 of the 184 gfx950 configs did:
   - **10 `gemm/gfx950/*mxfp8*` correctness configs, `SyncsPerBenchmark: 0`, `LibraryType:
     Prediction`, followed by LibraryClient.** Prediction skips winner, range and exact logic, and
     `removeInvalidSolutions()` drops only solutions with GFlops exactly 0, which a correctness run
     never writes. Test: in a saved `subtile_mxfp8_correctness_regression` run, every `-nan` was
     replaced with `123.5`, and LibraryLogic and LibraryClient were re-run (with
     `ForceRedoBenchmarkProblems=False`, so the benchmark step was skipped). LibraryLogic then
     found exact winners for all 5 sizes instead of `{}`. Yet the written logic YAML was
     byte-identical, and LibraryClient passed.
   - **8 `ductile/gfx950/*` configs, `SyncsPerBenchmark: 0`.** Ductile's GA reads the GFlops cells
     as its fitness scores (`_read_and_validate_scores_csv`), and every one is NaN. Every Ductile
     log shows `# ExactWinners: {}`. The GA and its LibraryLogic step run without any performance
     signal; only the plumbing and the per-candidate correctness are tested. No LibraryClient
     follows.
   - **`groupedgemm/grouped_gemm` and the 7 `custom/*` configs.** These benchmark: `grouped_gemm`
     sets `SyncsPerBenchmark: 1`, and the custom configs leave it at the default of 1. They get
     real winners.
3. In the remaining 158 configs, the CSV is written, copied, and never read.

So in correctness mode the CSV needs only to exist, with the right shape (sizes plus one column per
solution), for the `shutil.copy` and `addFromCSV` calls. `BenchmarkProblems._writeSyntheticResultsCSV`
already writes such a file for `--cpu-only`. Two caveats:

- That synthetic file uses a fixed positive GFlops. Non-Prediction LibraryLogic would then pick
  winners where today it picks none. This was not tested for the Ductile configs.
- The per-solution `-1` for failures is not consumed in correctness runs, since failures already
  fail the test through the exit code.

### YAMLs with `LibraryLogic:` / `LibraryClient:` sections

**What the sections do.** `Tensile.py` runs these steps after BenchmarkProblems:

- `LibraryLogic:` reads `2_BenchmarkData/*.csv` and the solutions YAML, picks winners per size,
  and writes the logic YAML (the format hipBLASLt ships) to `3_LibraryLogic/`. It costs about
  0.2 s per config.
- `LibraryClient:` reads `3_LibraryLogic/`, so it depends on `LibraryLogic:`. It builds a real
  library from the logic with `TensileCreateLibrary` (1.9–3.3 s), then runs the client against that
  library: solution selection through the library, with validation on (0.26–0.43 s).

With an empty `LibraryClient:` section and no exact logic, `ClientWriter` falls back to
`ProblemSizesMockDummy`, which is a single size, `[128, 128, 1, 512]`. So the library client never
runs the YAML's own benchmark sizes.

**Who has them.** 30 common YAMLs and all 12 `custom/` YAMLs:

| Group | Added by | `SyncsPerBenchmark` | `LibraryClient` | What the section achieves |
| --- | --- | --- | --- | --- |
| 10 `gemm/gfx950/*mxfp8*` correctness/repro configs | #7781 (multi-DU support), all at once; the first common YAMLs to use `Prediction` | 0 | yes, empty section | `Prediction` logic lists every solution; the CSV values do not change it (tested above). The library client validates only the dummy 128×128×1×512 size, about 24 s for all 10. No reason is given in the commit message or the YAMLs. |
| 14 `ductile/*` (8 on gfx950) | #8831 (Ductile backend) | 0 | commented out | Mirrors a tuning flow, whose product is a logic file. With NaN scores, the logs show `Winners set()`, `Exact Logic: {}` and a warning, and nothing reads the file. Nothing in Ductile reads `config["LibraryLogic"]`. |
| `gemm/fp8nfp16mix_hfp8ns`, `gemm/xfp32` (+ `LibraryClient`), `groupedgemm/{grouped_gemm, gfx11/grouped_gemm_gfx11, grouped_gemm_ck_gfx942}`, `sparse/gfx94x/libray_logic` | older | 1 (opted out in the unit test: "LibraryLogic requires benchmark data to build the library table") | `gemm/xfp32` only | Exercise winner selection with real timing data. Only `grouped_gemm` runs on gfx950. |
| 12 `custom/*` (see below) | #9304 / #11155 | unset, so the default 1 | no | `GridBased` logic with real winners, because they benchmark. Nothing reads it. |

The opt-out reason "LibraryLogic requires benchmark data" holds for `GridBased` logic: with NaN
data it finds no winners. It does not hold for `Prediction` logic, whose output was byte-identical
with and without real values.

### The `custom/` folder

`Tests/custom/` is a sibling of `common/`. It holds 12 YAMLs: `custom_aiter_bf16`,
`custom_aiter_f4`, `custom_ck`, `custom_rocblas_gemv_bf16_tn_m1/_m2/_m4`, `custom_rr`, `custom_rr_f6`,
`custom_rr_mx`, `custom_tensile_sk`, `custom_triton_f4` and `custom_wave_bf16`. They come from the
"Gemm-From-Anywhere" custom-kernel framework (#9304, reverted in #11104, re-landed in #11155;
later additions such as #12965). Each feeds a prebuilt external kernel (aiter, CK, rocBLAS GEMV,
rr, Triton, wave, or Tensile StreamK) through the `CustomKernel` fork parameter, with
`ValidateMetadata: True`. The PR's test plan states their purpose as "build and dispatch each demo
kernel end-to-end through the real `CustomKernels:` fork-parameter path". Producing logic files is
not mentioned.

They run as part of the common tests, even though they are outside `common/`:

- `config_helpers.findConfigs()` walks all of `Tests/` and parametrizes every YAML into
  `common/test_config*.py`.
- `conftest.pytest_collection_modifyitems` marks each test with the top directory of its *test
  file*. So a `custom/` YAML gets the `common` marker, plus `custom` from `configMarks`.
- With `-m "common and not skip-gfx94X"` and `--gpu-targets gfx950`, `--collect-only` lists 552
  tests, including all 12 custom YAMLs.

Results on gfx950:

- **CI-like combined run (`full_runs_gfx950/run.sh`):** 7 PASSED (`aiter_bf16`, `aiter_f4`, the
  three `rocblas_gemv`, `triton_f4`, `wave_bf16`); 3 SKIPPED (`ck`, `rr`, `tensile_sk`); 2 FAILED
  (`rr_f6`, and `rr_mx`, which failed to assemble). That run was on an older commit; the current
  status of the two failures was not rechecked.
- **Per-config reruns:** the 7 passing configs took 5–7 s each.

None of the 12 sets `SyncsPerBenchmark`, so all of them time their kernels. All 12 have a
`GridBased` `LibraryLogic:` section and no `LibraryClient:`.

### Existing parameters already express "no benchmarking"

524 of the 540 common YAMLs set `SyncsPerBenchmark: 0`, so no timed enqueues run. Only 16 set it
to 1. Of those 16, 5 ran on gfx950, with 288 solutions out of 2,355,016 in total: the
`rotate_mode0/1`, `sk_sgemm_debug_modes`, `grouped_gemm` and `benchmark_runs/fp32_tn` configs.
For today's YAMLs, "`SyncsPerBenchmark == 0`" therefore matches "nothing is timed". It is not the
client's full rule, though, and "nothing is timed" is not the same as "this is a test". See
[Deciding when a run is a test](#deciding-when-a-run-is-a-test).

With `SyncsPerBenchmark: 0`, the following parameters change nothing, measured or read in code:

- `EnqueuesPerSync`, `MaxEnqueuesPerSync`, `MinFlopsPerSync`, `SleepPercent` and `KernelTime`: only
  read inside the timed-enqueue loop, or by `BenchmarkTimer` when it times that loop.
- `HardwareMonitor`: samples only around timed enqueues, and `Tensile.py` already forces it off on
  gfx950 and gfx1250.
- `SkipSlowSolutionRatio`: 0.0 in all 19 YAMLs that set it.

### What each piece of output is for

| Output | Needed for correctness? | Controlled by |
| --- | --- | --- |
| exit code = number of failed validations | yes | always |
| `results-file` CSV (one GFlops column per solution) | must exist with the right shape; values are NaN/−1 and do not change any output (see above) | always |
| FAILED row on stdout (row level Error) | yes | printed at every log level |
| mismatch lines (`[i] elem=… idx=…: ref!=gpu`), `Check failed in output tensor` | yes | printed with `std::cout` at every level, up to `ValidationMaxToPrint` |
| Synchronizer residue messages | yes | `LogLevel::Error` |
| CSV row for every PASSED solution | no | `ClientLogLevel` ≥ Verbose |
| CSV row and predicate `debugEval` dump for every rejected solution | no | `ClientLogLevel` ≥ Terse (row) and ≥ Verbose (dump) |
| granularity, tiles-per-CU and memory-byte columns (`BenchmarkTimer::preSolution`) | no | only shown in the stdout row, or with `CSVExportExtraCols` |
| time-us, gflops, gbps (`BenchmarkTimer::postSolution`) | no; always NaN when nothing is timed | always computed |
| warmup start/stop `hipEvent`s (2 per launched kernel) | no; only `BenchmarkTimer` reads them | always created |
| warmups beyond the first launch | no, unless the YAML deliberately relaunches (see below) | `NumWarmups` |

### Measurements

**Setup.** All runs replay the dtl client invocations (443,808 solutions) or the
`gemm/gfx950/xfp32` ones (94,668 solutions, `NumWarmups: 5`), with `KMP_BLOCKTIME=0`, 2 runs per
variant. The per-solution host cost is measured as `pre_solution` + `run_current_solution_check` +
`post_solution`. These are the enclosing timers, so work moved out of an inner timer is still
counted.

**Output equivalence.** This was checked for every variant against `ref`, on 3 dtl and 2 xfp32
invocations, at Debug and Error, both with clean runs and with injected failures. Failures were
injected by an `LD_PRELOAD` shim that flips one byte of every 7th device-to-host `hipMemcpy` of at
least 4 KiB (`/tmp/bookkeep/micro/corrupt.cpp`). In all cases:

- exit codes were identical (0, 20, 216 and 255);
- results CSVs were byte-identical;
- the FAILED rows (61, 216, 2,128 and 3,934 per invocation) and the mismatch locations were
  identical, after normalising the run-specific random data values (`srand seed` is time-based).

| Variant (dtl) | How | Per-solution host cost (ms) | dtl stdout |
| --- | --- | --- | --- |
| today (Debug) | – | 8,941 / 8,780 | 1,191,874 lines, 520 MB |
| `ClientLogLevel: 0` | existing parameter | 6,517 / 6,279 (−28%) | 10,350 lines, 0.31 MB |
| `ClientLogLevel: 0` + correctness mode (prototype `corr`) | client change | 3,121 / 3,167 (−65%) | 10,324 lines, 0.30 MB |
| `ClientLogLevel: 0` + `corr` + no timing events (prototype `corr2`) | client change | 3,166 / 3,103 | same |

With `ClientLogLevel: 0`, a clean invocation that printed 60,940 lines prints 87. One with 3,934
failures prints 14,259: exactly the FAILED rows and the mismatch lines.

`ClientLogLevel` is already settable from the test command line. `--tensile-options=…,
--global-parameters,ClientLogLevel=0` (the same mechanism as `TimingInstrumentation=True`) gave
`log-level=Error` in every `ClientParameters.ini` of `streamk/sk_dynamic_work_stealing`, and the
test passed.

Prototype `corr`, in `/tmp/bookkeep/client_corr`, applies when `num-syncs-per-benchmark == 0` (or
`num-enqueues-per-sync == 0`) and `csv-export-extra-cols` is false.

**Correction:** the `num-enqueues-per-sync == 0` half of that condition is wrong. With
`num-syncs-per-benchmark=1`, `num-enqueues-per-sync=0` and the default `min-flops-per-sync=1`, the
client still times every solution: 288/288 timed loops, all with real `time-us`; see
[Deciding when a run is a test](#deciding-when-a-run-is-a-test). No current YAML has that
combination (all six `EnqueuesPerSync: 0` files also set `SyncsPerBenchmark: 0`), so the
measurements above are unaffected. A real implementation must use the full rule below.

When the prototype's condition holds:

- `BenchmarkTimer::preSolution` skips `projectedPerformance` and its 9 reports.
- `BenchmarkTimer::postSolution` skips the performance calculation and reports the same NaN values
  directly. The original's 0/0 gives a negative quiet NaN, which prints as `-nan`; the prototype
  reproduces it.
- `LogReporter::postSolution` checks the row level before copying the row. The copy is only used for
  printed rows. One side effect: the CSV header line is printed just before the first printed row,
  instead of on the first solution.

Prototype `corr2` (`/tmp/bookkeep/client_corr2`) additionally creates the warmup `TimingEvents`
with null events, and skips `BenchmarkTimer`'s `hipEventSynchronize` after warmup 0. That wait is
taken over by the validators' blocking readback, so removing it only moves attribution. Measured
on dtl: `warmup_runs` (host side of the launches) fell from 1,981–2,046 to 1,291–1,295 ms,
`warmup_events_setup` from 72 to 30 ms, and `run_iteration_teardown` from 122–129 to 46–47 ms. That
is about 0.85 s, or 3.1 µs per launched solution. On xfp32 it saved about 2.4 µs per launched
solution.

### Warmups that nothing consumes

`numWarmupRuns()` is the maximum over listeners: the validators ask for 1 launch, and
`BenchmarkTimer` asks for `NumWarmups`. So with `SyncsPerBenchmark: 0`, every warmup beyond the
first only matters if the YAML relaunches deliberately. On gfx950, ten configs set
`SyncsPerBenchmark: 0` with `NumWarmups > 1`:

| Config | `NumWarmups` | Solutions | Assessment |
| --- | --- | --- | --- |
| `gemm/gfx950/xfp32` | 5 | 94,668 | no stated purpose; also sets `EnqueuesPerSync: 10`, which is unused with 0 syncs |
| 7 `ductile/gfx950/*` | 5–10 | 37–130 each | negligible |
| `streamk/sk_dynamic_work_stealing`, `sk_hybrid_work_stealing` | 2 | 32, 64 | deliberate: "Back-to-back enqueues reuse the same Synchronizer workspace, exercising the per-queue auto-reset across launches." |

`cluster_entry_handoff` (288 warmups) and the gfx12 configs are not run on gfx950.

xfp32 with `num-warmups=0` instead of 5 (one launch per solution; each row is one run, times in s):

| Variant | `main_total` | `warmup_runs` (host) | `gpu_input_reset` |
| --- | --- | --- | --- |
| as configured (5 warmups) | 76.4 | 2.91 | 46.4 |
| as configured (5 warmups) | 65.9 | 2.29 | 39.9 |
| `num-warmups=0` | 53.1 | 0.58 | 30.1 |
| `num-warmups=0` | 59.6 | 0.60 | 33.5 |
| `num-warmups=0` + `ClientLogLevel: 0` | 59.3 | 0.57 | 30.6 |
| `num-warmups=0` + `ClientLogLevel: 0` | 54.5 | 0.59 | 32.0 |

About 15 s (20%) of xfp32's client time goes. Most of the extra warmups' GPU time showed up in the
*next* solution's `gpu_input_reset`, whose blocking copy waits for the queued launches. That is the
same attribution effect described in `SynchronizerValidator.md`. Because the GPU is shared, the
GPU-bound timers vary by ±15%, but all four `num-warmups=0` runs are below both baseline runs. The
warmup count cannot be inferred away globally, because some configs relaunch on purpose. It is a
per-YAML fix.

### Options for correctness runs

| Option | Change type | Measured | Aggregate estimate (scaled from dtl/xfp32 counts) |
| --- | --- | --- | --- |
| 1. pass `ClientLogLevel=0` for the common tests (test harness / CI command line) | none: existing parameter | −28% per-solution host cost on dtl; stdout 520 MB → 0.31 MB | ≈ 9 s (2.1 µs per solution for the rows, plus 9.4 µs per rejected solution) and 2.6 GB of stdout |
| 2. drop `NumWarmups: 5` / `EnqueuesPerSync: 10` from `gemm/gfx950/xfp32.yaml` | YAML | xfp32 client time −20% (≈ 15 s) | ≈ 15 s; no other gfx950 config is affected |
| 3. infer "nothing is timed" in the client (prototype `corr`; the real rule is in [Deciding when a run is a test](#deciding-when-a-run-is-a-test)) | client code | a further −51% on top of option 1 (6.4 → 3.1 s on dtl) | ≈ 17 s (7.3 µs per solution) |
| 4. no timing events in correctness mode (prototype `corr2`) | client code | 3.1 µs per launched solution (dtl) | ≈ 5–6.5 s |
| 5. `KMP_BLOCKTIME=0` (option A above) | environment | removes the >1 ms tail | ≈ 9.9 s, plus the `cpu_reference_gemm` effect |

Option 1 could also be inferred rather than passed. `ClientWriter` could write `log-level=Error`
when `SyncsPerBenchmark == 0` and the YAML does not set `ClientLogLevel`. That needs a way to tell
"explicitly set" from "default" for that parameter; not implemented.

Remaining per-solution cost in correctness mode (dtl, `corr`): about 7 µs per solution. It is
`AllSolutionsIterator`/`ProgressListener` progress strings and solution name (1.0 s), predicate
evaluation (0.47 s), and `ResultFileReporter` building the results CSV. The CSV must exist, and its
formatting could use option B. Because its values carry no information when nothing is timed,
the client could also write it once per problem with constant cells, instead of fanning every
value out to `ResultFileReporter`. This is not prototyped.

Not affected: data initialisation, CPU reference, input reset, the kernel launch, and both
validators. These are the correctness work itself and are covered in the other docs.

### Planned follow-ups (recorded, not implemented)

**1. Extend the `SyncsPerBenchmark` guard to `custom/`.**

`Tests/unit/test_common_yaml_syncs_per_benchmark.py` today globs only `common/`
(`_COMMON_DIR.rglob`). It requires `GlobalParameters.SyncsPerBenchmark == 0`, as an int, for every
file not in `_SYNCSPERBENCHMARK_OPT_OUTS`. Each opt-out must give a reason, and stale opt-outs fail
the test. `custom/` YAMLs run as common tests (see above), so the test should also walk
`Tests/custom/`.

All 12 custom YAMLs would fail it today, since the key is unset. Each needs either
`SyncsPerBenchmark: 0` or an opt-out with a reason. Given their stated purpose (build and dispatch
end to end), `SyncsPerBenchmark: 0` looks right. Their only consumer of timing data is the
`LibraryLogic:` step, which nothing reads (see item 2).

**2. Remove `LibraryLogic:` / `LibraryClient:` from common YAMLs, and pin it with a test.**

Add a unit test, in the same style as the `SyncsPerBenchmark` guard: a dataclass whitelist with a
mandatory reason, plus a stale-entry check. It fails if any YAML under `common/` has a top-level
`LibraryLogic` or `LibraryClient` key, unless the file is whitelisted. Applying it to `custom/` as
well would match item 1, but that scope is still to be decided.

Proposed whitelist: the 6 files whose purpose is the logic/library path with real benchmark data,
which are already the `SyncsPerBenchmark` opt-outs with the LibraryLogic reason.

| File | Sections |
| --- | --- |
| `common/gemm/fp8nfp16mix_hfp8ns.yaml` | `LibraryLogic` |
| `common/gemm/xfp32.yaml` | `LibraryLogic` + `LibraryClient` |
| `common/groupedgemm/grouped_gemm.yaml` | `LibraryLogic` |
| `common/groupedgemm/gfx11/grouped_gemm_gfx11.yaml` | `LibraryLogic` |
| `common/groupedgemm/grouped_gemm_ck_gfx942.yaml` | `LibraryLogic` |
| `common/sparse/gfx94x/libray_logic.yaml` | `LibraryLogic` |

To remove:

- the 14 `ductile/*` sections: the logic is written from NaN data and never read;
- the 10 `gemm/gfx950/*mxfp8*` sections: the library client validates only the dummy
  128×128×1×512 size, about 24 s per gfx950 run;
- the 12 `custom/*` sections, if the scope is extended.

Decision needed before removal: the 10 mxfp8 configs are the only common coverage of `Prediction`
logic, library build from that logic, and `Prediction` selection in the client. Either accept
losing that coverage, or whitelist one designated config (for example
`subtile_mxfp8_tail_loop_smoke`), ideally with explicit `LibraryClient` sizes so that it validates
more than the dummy size.

**3. Copy the results CSV only if it exists.**

After the client runs, `BenchmarkProblems.main` does `shutil.copy(resultsFileName,
newResultsFileName)` unconditionally. The granularity CSV is already copied only `if
os.path.isfile(...)`. Apply the same guard to the results CSV, so that a run that did not write one
does not fail.

The remaining readers fail hard on a missing file, so they must only run when the CSV is written:

- `LibraryLogic.addFromCSV` calls `printExit`;
- Ductile's `_read_and_validate_scores_csv` checks `os.path.isfile` and calls `printExit`.

**4. Don't write the results CSV for tests.**

Have Tensile pass "no results file" to the client, and have the client skip `ResultFileReporter`
entirely in that case. Today `results-file` defaults to `results.csv`, and
`ResultFileReporter::Default` always constructs the reporter. An empty name would only produce a
silently failing `ofstream` while still doing all the per-solution formatting, so the client needs
an explicit "no reporter" path. The CSV must still be written whenever something reads it: a
`LibraryLogic:` section, or the Ductile backend. With item 2, that leaves only the whitelisted
files and Ductile's own benchmark runs.

### Deciding when a run is a test

**What the client actually checks.** Whether timed runs happen depends on several inputs, combined
across all listeners:

| Main-loop query (`main.cpp`) | Combined as | Who returns non-zero |
| --- | --- | --- |
| `needMoreBenchmarkRuns()` (outer loop) | any listener | `BenchmarkTimer` (`num-benchmarks`), `ReferenceValidator` (once, if enabled), `ProgressListener` (once) |
| `needMoreRunsInSolution()` (per-solution loop) | any listener | `ReferenceValidator` (until validated); `BenchmarkTimer` while `enqueuesDone < num-enqueues-per-sync × num-syncs-per-benchmark` and the solution is not skipped |
| `numWarmupRuns()` | max | `ReferenceValidator` 1 (until validated); `BenchmarkTimer` `num-warmups` |
| `numSyncs()` | max | only `BenchmarkTimer`: `num-syncs-per-benchmark` (the profiler returns 0 and is compiled out) |
| `numEnqueuesPerSync()` | max | only `BenchmarkTimer`: 0 if the solution is skipped (restore-from-log map, or skip-slow); otherwise `min(max(num-enqueues-per-sync, ceil(min-flops-per-sync / flops)), (size_t)max-enqueues-per-sync)` |

The timed loop is `if(enq) for(i < syncs)`. So nothing is timed for a solution exactly when one of
these holds:

- `num-syncs-per-benchmark == 0`;
- `max-enqueues-per-sync == 0`;
- `num-enqueues-per-sync == 0` **and** `min-flops-per-sync == 0`;
- the solution is skipped (`prob-sol-map` restore, or `skip-slow-solution-ratio` > 0 after a slow
  warmup).

A negative `max-enqueues-per-sync` (the default −1) casts to a huge `size_t`, so it means no cap.

Measured on one dtl invocation (288 run solutions; `ref` build):

| syncs | enqueues | min-flops | timed loops | rows with real `time-us` |
| --- | --- | --- | --- | --- |
| 0 | 1 | 1 | 0 | 0 (all NaN) |
| 1 | 0 | 1 | 288 | 288 |
| 1 | 0 | 0 | 0 | 0 (all NaN) |
| 1 | 1 | 1 | 288 | 288 |

`SyncsPerBenchmark` is therefore the only parameter that turns timing off on its own. `EnqueuesPerSync:
0` does not, unless `MinFlopsPerSync` is also 0 (no YAML sets it; its default is 1).
`MaxEnqueuesPerSync` is set in 4 YAMLs (`agntab_coverage_*`), always to 1.

**Three different questions.** "Is this a test?" mixes three separate decisions:

1. **Is anything timed?** This decides the timing-only work: granularity and performance values,
   timing events, the `validate_gpu_sync` wait, and warmups beyond what the validators need. Use
   the client's own rule above, evaluated in the client from its own arguments (prototype `corr`
   did this, with the wrong enqueue term). Inferring this is safe: when nothing is timed, the
   skipped values are NaN by construction.
2. **Is the results CSV consumed?** This decides whether to write and copy the CSV. It is a
   property of the YAML and backend, not of timing: write it if the config has `LibraryLogic:` or
   uses the Ductile backend. Python knows both before it launches the client.
3. **Is this run part of the test suite?** This decides verbosity (`ClientLogLevel`) and any
   test-only shortcuts. It cannot be inferred from timing parameters: 16 common YAMLs time on
   purpose and are still tests (`benchmark_runs/*`, `rotate_mode*`, `icache_flush`,
   `*_debug_modes`, the LibraryLogic opt-outs), and a user benchmarking with `SyncsPerBenchmark:
   0` is not running a test. The reliable signal is the harness saying so: `tensile_args` in
   `Tests/conftest.py` builds the argument list for every Tensile invocation in the common tests,
   so it can append a dedicated global parameter, which `ClientWriter` turns into client options.
   One constraint: the conftest `--global-parameters` option forwards a single value, which `tox`
   already uses for `CheckASMCodeSize=True`. So the flag should be appended by the fixture itself,
   or the option made to accept several values; `--tensile-options=…,--global-parameters,X=Y`
   also works, as tested for `ClientLogLevel=0`.

Recommendation (not implemented): use rule 1 inside the client for timing-only work, rule 2 in
Python for the CSV, and an explicit harness-set parameter for rule 3. Do not key everything on
`SyncsPerBenchmark`.

## Not measured / open items

- **`cpu_reference_gemm` and `KMP_BLOCKTIME`.** In the dtl replay, `cpu_reference_gemm` took
  5.8–14.8 s with the default blocktime and 2.4–2.7 s with `KMP_BLOCKTIME=0`, over 3 runs each. The
  Reference GEMM section is 1,080 s (42%) of the aggregate, so this deserves its own investigation.
  Mechanism not established.
- **CI thread counts.** Not checked: whether CI sets `OMP_NUM_THREADS` for the client. If it does
  not, libomp sizes the pool from the affinity mask, not from the pod's CPU quota, while 4 xdist
  workers run clients at the same time. Spin contention would then be worse than measured here.
  Inference.
- **`post_problem` (1.9 s, 68 µs mean, 50% in calls > 1 ms).** `ResultFileReporter` adds a column
  per solution name, so each problem's results row grows with the solution count. Not broken down.
- **Other configs.** Only dtl was replayed. The narrow per-config spread (24–37 µs per solution,
  p10–p90 weighted by solutions) suggests a similar mix elsewhere. Configs with timed iterations
  (non-NaN `time-us`) exercise a few extra branches in `ResultFileReporter`.
- **The `opt` SIGSEGV window** described under option B.
- **Whole-run effect.** Not measured; GPU sharing made whole-client wall times too noisy.
