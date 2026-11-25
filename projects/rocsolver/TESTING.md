# rocSOLVER Testing Strategy

Status: Draft\
Owning team: `@ROCm/solvers-reviewers` (see [Owners and Review Cadence](#owners-and-review-cadence))\
Technical POC: *TBD*\
Last Updated: November 10, 2026

This document describes how rocSOLVER is tested today, which signals gate a
merge, and how to run each part of the suite yourself. It follows the ROCm-wide
TESTING.md template and is written as a description of the current state rather
than an aspirational one.

## Component Overview

rocSOLVER is a GPU implementation of LAPACK, built on top of rocBLAS. It
optionally uses rocSPARSE for the sparse re-factorization routines.

The public interface is C99 and is kept source- and binary-compatible with
previous releases; all public symbols are prefixed `rocsolver`/`ROCSOLVER` or
`rocblas`/`ROCBLAS` (see [CONTRIBUTING.md](CONTRIBUTING.md)).

What makes the test suite large is that it is combinatorial. The benchmark
dispatch table in `clients/common/misc/rocsolver_dispatcher.hpp` lists 359
callable entry points covering 182 base routines, and each routine is
instantiated across:

- matrix data layout -- single matrix, `batched`, `strided_batched`
- precision -- `float`, `double`, `rocblas_float_complex`, `rocblas_double_complex`
- index type -- `rocblas_int`, `int64_t`
- where applicable, blocked and unblocked variants (for example `getf2` vs `getrf`)
- where a routine offers a choice of algorithm, each option -- `syevd`/`heevd`
  and `sygvd`/`hegvd` instantiate both the 1-stage and 2-stage tridiagonalization
  (`hetrd_alg_mode`); `gesvd`, `bdsqr`, `syev`/`heev` and `steqr` each
  instantiate a GPU-only class alongside a parallel `*_HYBRID` class built from
  the same `alg_mode`-templated base (for example `GESVD` vs `GESVD_HYBRID` in
  `clients/gtest/lapack/gesvd_gtest.cpp`)

That expansion is driven by the `INSTANTIATE` / `FOREACH_*` / `APPLY_STAMP`
macro pipeline in `clients/common/misc/client_util.hpp`.

Some functionality is behind build-time feature flags, which are mirrored into
the test and benchmark clients so that tests assert
`rocblas_status_not_implemented` when a feature is off:
`ROCSOLVER_ENABLE_CHOLQR`, `ROCSOLVER_ENABLE_EIGENSOLVERS_64`,
`ROCSOLVER_ENABLE_HESSENBERG`, `ROCSOLVER_ENABLE_EIG_2STAGE`.

## Development Workflow

Build the library and clients:

```bash
./install.sh -c                     # release build + tests, benchmarks, samples
./install.sh -d -c                  # ... also build/install dependencies first
```

`-c` implies `-DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON
-DBUILD_CLIENTS_SAMPLES=ON` and `-DBUILD_TESTING=ON
-DBUILD_CLIENTS_EXTRA_TESTS=ON` (`install.sh:593-594`). `-d` builds and installs
reference LAPACK 3.9.1, fmt 7.1.3 and googletest 1.11.0.

Binaries land in `build/release/clients/staging/` (or `build/debug/...` with
`-g`). Other flags that matter while testing:

| Flag | Effect |
| --- | --- |
| `-g` / `-k` | Debug / RelWithDebInfo build. Debug defaults `ARMOR_LEVEL` to 1. |
| `-n` | Skip small-size optimized kernels -- much faster to compile. |
| `--no-sparse` | Build without rocSPARSE; the `refact` tests then call `GTEST_SKIP()` rather than asserting `not_implemented`, so a clean run means skipped coverage, not validated behavior. |
| `--address-sanitizer` | `-DBUILD_ADDRESS_SANITIZER=ON`, see [Sanitizer Coverage](#sanitizer-coverage). |
| `--codecoverage` | `-DBUILD_CODE_COVERAGE=ON`; refused in Release mode. |
| `--clients-only` | Build only the clients, against an already-installed library. |
| `--cmake-arg` | Pass an extra argument through to CMake, e.g. `-DWERROR=ON`. |

On Windows, `rmake.py` takes the equivalent `-c` / `--clients-only` options.

### Finding a LAPACK the clients can link against

The clients require LAPACK, and by default rocSOLVER looks for it in CMake
*config* mode only -- `ROCSOLVER_FIND_PACKAGE_LAPACK_CONFIG` defaults to `ON`
(`CMakeLists.txt:62`), which selects `find_package(LAPACK 3.7 REQUIRED CONFIG)`
at `clients/CMakeLists.txt:44-47`.

Most distribution LAPACK packages -- including Ubuntu's `liblapack-dev` -- ship
the library and headers but no `LAPACKConfig.cmake`. On such a system a plain
`./install.sh -c` fails at configure time even though a perfectly usable LAPACK
is installed:

```
CMake Error at clients/CMakeLists.txt:45 (find_package):
  Could not find a package configuration file provided by "LAPACK" (requested
  version 3.7) with any of the following names:
    LAPACKConfig.cmake
    lapack-config.cmake
```

There are two ways out. Prefer the first if you already have a distro LAPACK:

```bash
# Use CMake's bundled FindLAPACK module instead of config mode
./install.sh -c --cmake-arg="-DROCSOLVER_FIND_PACKAGE_LAPACK_CONFIG=OFF"

# Or build reference LAPACK 3.9.1 from source into /usr/local
./install.sh -d -c
```

Module mode picks up an OpenBLAS-provided `liblapack.so` without complaint.

Then run the tests that correspond to what you changed:

| If you change... | Run... | GPU required |
| --- | --- | --- |
| A single routine in `library/src/lapack/` or `library/src/auxiliary/` | `./rocsolver-test --gtest_filter='checkin*<ROUTINE>*'` | Yes |
| Shared infrastructure under `library/src/include/` | `./rocsolver-test --gtest_filter='checkin*'` (~45 min) | Yes |
| Sparse re-factorization | `./rocsolver-test --gtest_filter='checkin*REFACT*:checkin*SPLITLU*'` | Yes |
| Algorithm internals you expect to be exercised only at large sizes | `./rocsolver-test --gtest_filter='daily*<ROUTINE>*'` | Yes |
| A whole tier, the way CI selects it | `ctest -L '^quick$'` (or `standard`, `comprehensive`) -- needs `shared/`, see below | Yes |
| `rocsolver-bench` arguments or output | `CTEST_OUTPUT_ON_FAILURE=1 ctest -R '^test-rocsolver-bench'` | Yes |
| Public headers | `./test-header`, `./test-rocsolver-dlopen` | No / Yes |
| Anything, before pushing | `clang-format -i -style=file <files>` or `scripts/reformat-files` | No |

The `ctest` rows above must be run **from the build root** (e.g. `build/release`),
not from `clients/staging` where the `./rocsolver-test` rows run -- CTest's
registrations live at the build root, and running from `staging` finds zero
tests with no error. Use `ctest --test-dir build/release ...` if you are
elsewhere. They also assume the super-repo's `shared/` is in your checkout; if
it is not, `ctest -L` silently matches nothing and the numbering of the
remaining tests shifts -- run `ctest -N` first to see what is actually
registered, and read
[A sparse checkout needs the super-repo's `shared/`](#a-sparse-checkout-needs-the-super-repos-shared).

`-L` matches labels as a regular expression, so an unanchored `ctest -L quick`
also selects `rocsolver-test_ffm-quick_suite` (and `full` selects `ffm-full`).
Anchor the pattern -- `ctest -L '^quick$'` -- to run only the requested tier.

### Startup cost: the test binary is not hung

`rocsolver-test` pays a fixed startup cost before it prints anything beyond its
version banner, and that cost is paid by *every* invocation -- including
`--gtest_list_tests`. `main()` in `clients/gtest/rocsolver_gtest_main.cpp` runs
`query_device_property()` and `set_device(0)` before `InitGoogleTest`, then
`rocblas_initialize()` before `RUN_ALL_TESTS()`, which is where test listing
actually happens.

On an idle, exclusively allocated node this is a few seconds. On a loaded or
oversubscribed machine -- for example while a parallel build is running -- it can
take minutes with no output after the banner, which looks exactly like a hang.
It isn't. If you need a fast answer about which tests exist, read the
`INSTANTIATE_TEST_SUITE_P` lines in the relevant `clients/gtest/**/*_gtest.cpp`
rather than waiting on `--gtest_list_tests`.

### A sparse checkout needs the super-repo's `shared/`

**The CTest categories do not work from a checkout of `projects/rocsolver/`
alone.** They are generated by `shared/ctest/TestCategories.cmake`
(`clients/gtest/CMakeLists.txt:273`), which also needs Python3. When `shared/`
is missing the step is skipped *silently*: the build looks fine but registers 3
tests instead of 9, and `ctest -L` matches nothing. Configure prints
`-- rocSOLVER: YAML-based test categorization` when the categories are active.

```bash
git sparse-checkout set projects/rocsolver shared
# then reconfigure from a clean CMakeCache.txt
ctest -N                 # confirm 9 tests, not 3
```

Note the categories are not just labels on `rocsolver-test` -- each becomes its
own entry, `rocsolver-test_<category>_suite`, with the tier's gtest filter and
timeout baked in. Their absence also renumbers the remaining tests, so a stale
`ctest -R` or test number may not mean what you expect.

One unrelated trap worth knowing, since it also surfaces as a `ctest` failure:
`clients/extras/CMakeLists.txt:62` bakes the *absolute* interpreter path into
the `test-rocsolver-bench` registration (`COMMAND "${Python3_EXECUTABLE}" ...`).
A build tree carried across container images or node rebuilds can therefore fail
with `Could not find executable /usr/bin/pythonX.Y` even though nothing is wrong
with the code. Reconfiguring from a clean `CMakeCache.txt` fixes it.

Formatting is not checked by the GitHub `pre-commit` job for rocSOLVER paths
(see [Static Analysis](#static-analysis)), so install the hooks locally:

```bash
scripts/install-hooks --get-clang-format      # scripts/uninstall-hooks to remove
```

## Testing Strategy and Layers

### Unit Testing Strategy

rocSOLVER's unit-scoped tests cover library mechanics and test infrastructure
rather than numerics. They are a small fraction of the suite.

| Test | Location | What it covers |
| --- | --- | --- |
| Infrastructure micro-tests | `clients/gtest/unit/` | The clients' own helpers: `gemm`, `gerand`, `herand`, `syrand`, `hbrand` (5 files). |
| Device memory model | `clients/gtest/memory_model_gtest.cpp` | The rocBLAS device-memory model: size queries, user-owned workspaces. |
| Logging | `clients/gtest/logging_gtest.cpp` | Trace/bench/profile logging output, driven through `ROCSOLVER_LOG_*` env vars. |
| ASAN helpers | `clients/gtest/asan_helpers_gtest.cpp` | The `ROCSOLVER_ASAN_VALUE` block-size substitution used under ASAN. |
| Managed memory (HMM) | `clients/gtest/managed_malloc_gtest.cpp` | Unified/managed allocations. Requires an XNACK-capable device and `HSA_XNACK=1`. |

In addition, every routine carries argument-validation coverage. By convention a
case with `m = n = 0` also runs that routine's `testing_<routine>_bad_arg<>()`
path, which checks the handling of a null handle, null pointers, invalid enum
values and invalid sizes. Negative sizes test the invalid-size return.

### Integration Testing Strategy

The bulk of the suite: each routine is run end to end on the GPU and compared
against a CPU reference. This is where nearly all rocSOLVER test weight sits --
94 gtest drivers over 103 routine test bodies.

**File layout.** Each routine follows a three-file pattern:

| File | Role |
| --- | --- |
| `clients/common/<group>/testing_<routine>.hpp` | The templated test body: `<r>_checkBadArgs()`, `testing_<r>_bad_arg<>()`, `<r>_initData<>()`, `<r>_getError()`, `<r>_getPerfData()`, `testing_<r>(Arguments&)`. Shared by the test and benchmark clients. |
| `clients/common/<group>/testing_<routine>.cpp` | Explicit template instantiation only, via `INSTANTIATE(...)` with the `FOREACH_*` lists. |
| `clients/gtest/<group>/<routine>_gtest.cpp` | The googletest wrapper: size ranges, an `Arguments` setup function, `TEST_P` cases per precision/layout, and `INSTANTIATE_TEST_SUITE_P`. |

Groups are `auxiliary/` (36 gtest files), `lapack/` (42), `refact/` (7) and
`unit/` (5).

**Parameterization.** Tests are googletest value-parameterized. Each
`*_gtest.cpp` declares its own size ranges -- `matrix_size_range` for
`checkin_lapack`, `large_matrix_size_range` for `daily_lapack`, and
`large_batch_size_range` for `weekly_lapack` -- converts a tuple to an
`Arguments` object, and defines `TEST_P` cases named `__float`, `__double`,
`__float_complex`, `__double_complex` plus `batched__*` and
`strided_batched__*` variants. Every fixture's `TearDown()` asserts
`hipGetLastError() == hipSuccess`, so a test cannot pass while leaving an
error pending on the device.

**Correctness checking.** The reference is Netlib LAPACK running on the CPU,
located by CMake (`find_package(LAPACK 3.7)`) and wrapped by
`clients/common/misc/lapack_host_reference.hpp` (`cpu_getrf`, `cpu_lange`,
`cpu_gemm`, `cpu_axpy`, `cpu_gecon`, ...). Helpers that have no LAPACK equivalent
live in `clients/common/misc/cpu_blas.hpp`. Not every routine is compared
entry-by-entry against the reference; some are validated implicitly, by the
residual of the problem they solve.

The error metric is relative and norm-based --
`norm_error()` in `clients/common/misc/norm.hpp` promotes both matrices to
double precision, computes `||gold||` with `cpu_lange`, forms the difference
with `cpu_axpy`, and returns `||gold - computed|| / ||gold||`. Norm types `'O'`,
`'I'` and `'F'` are available; `'F'` (Frobenius) is the usual choice. The
variants `norm_error_upperTr` and `norm_error_lowerTr` zero the opposite
triangle first.

Tolerances are expressed as a multiple of machine epsilon
(`clients/common/misc/rocsolver_test.hpp:50`):

```cpp
#define ROCSOLVER_TEST_CHECK(T, max_error, tol) EXPECT_LE((max_error), (tol)*get_epsilon<T>())
#define ROCSOLVER_TEST_CHECK2(err2, err1)       EXPECT_LE((err2), (err1))
```

By convention `tol` is a problem dimension, so the assertion reads "error at most
n epsilons" -- for example
`ROCSOLVER_TEST_CHECK(T, max_error, min(m, n))`. Both macros expand to nothing
when compiled into `rocsolver-bench` (`ROCSOLVER_CLIENTS_BENCH`), which is why
the same test body serves both clients.

Behaviour is controlled by flags on the `Arguments` object
(`clients/common/misc/rocsolver_arguments.hpp`):

| Flag | Default in tests | Effect |
| --- | --- | --- |
| `unit_check` | 1 | Performs the `EXPECT_LE` tolerance assertion. |
| `norm_check` | 0 | Computes and prints the error without asserting. Set by `rocsolver-bench -v`. |
| `hash_check` | 0 | Prints deterministic hashes of inputs/outputs, for reproducibility comparisons. |
| `singular` | 0 | Feeds degenerate/singular matrices. |
| `perf` | 0 | GPU timing only; skips the CPU reference entirely. |

`Arguments` validates that every argument supplied on the command line was
actually consumed (`validate_consumed()`), so a misspelled benchmark option is
an error rather than a silent no-op.

**Test data.** The sparse re-factorization tests read CSR matrices from
`clients/sparsedata/` -- 24 directories of the form `mat_<n>_<nnz>/` and
`posmat_<n>_<nnz>/`. They are copied into the staging directory at build time and
installed to `<datadir>/rocsolver/test`. At run time `get_sparse_data_dir()`
(`clients/common/misc/rocsolver_test.cpp:38`) searches, in order:
`$ROCSOLVER_TEST_DATA`, `<exe>/../share/rocsolver/test`, `<exe>/sparsedata`, then
the current directory. The MATLAB/Octave helpers used to generate and convert
this data are in `scripts/spdata/`.

### Build-Time Validation

Some correctness is enforced before anything runs:

- **`-DWERROR=ON`** -- used by the Jenkins precheckin and static jobs, so new
  compiler warnings fail the build there.
- **`ARMOR_LEVEL`** (`cmake/armor-config.cmake`) -- defaults to 1 in Debug and 0
  otherwise. Level >= 1 adds `_FORTIFY_SOURCE=1` (unless ASAN is on),
  `_GLIBCXX_ASSERTIONS` and `ROCSOLVER_VERIFY_ASSUMPTIONS`; level >= 2 adds
  `_GLIBCXX_DEBUG`. Applied to `rocsolver`, `rocsolver-test` and
  `rocsolver-bench`.
- **`test-header`** -- compiles the public headers standalone to catch C99 and
  self-containment regressions.
- **`test-rocsolver-dlopen`** -- confirms the shared library loads and resolves
  its symbols via `dlopen`.

The last two are built only with `-DBUILD_CLIENTS_EXTRA_TESTS=ON`, which
`install.sh -c` sets.

### Performance and Benchmarking Testing

`rocsolver-bench` runs any routine in the dispatch table standalone and reports
timing. Key options:

| Option | Meaning |
| --- | --- |
| `-f`, `--function` | Routine to run (default `potf2`). |
| `-r`, `--precision` | `s`, `d`, `c`, `z` (default `s`). |
| `--batch_count`, `-i`/`--iters` | Batch size; timed iterations (default 10). |
| `--perf 1` | GPU timing only -- skips the CPU reference, so the run is much faster. |
| `-v`, `--verify 1` | Compare against the CPU reference and print the error. |
| `--mem_query 1` | Report the required device workspace instead of running. |
| `--profile N`, `--profile_kernels` | Per-kernel profile logging to depth N. |
| `--alg_mode` | 0 = GPU-only, 1 = hybrid (CPU/GPU) where supported. |
| `--hetrd_alg_mode` | 0 auto / 1 one-stage / 2 two-stage. |
| `--hash 1` | Print deterministic hashes instead of errors. |

```bash
./rocsolver-bench -f geqrf_strided_batched -r d -m 30 -n 30 --lda 30 \
                  --strideA 900 --strideP 30 --batch_count 100 --perf 1
```

Output is a two-row `Arguments:` / `Results:` table. **`gpu_time_us` is in
microseconds and lower is better** -- it is a time, not a rate.

For sweeps, `scripts/perf/perfoptim-suite` drives a set of routines across
precisions and small/medium/large size classes and writes one CSV per
routine-precision under `rocsolver_customer01_benchmarks/`:

```bash
cd scripts/perf
./perfoptim-suite                 # all routines, s and d, all size classes
./perfoptim-suite geqrf s c small large
```

Performance is measured on demand by a developer. There is no automated
performance comparison on pull requests or on a nightly cadence, and no
performance baseline is stored in the repository; a change's performance impact
is established by running the benchmark before and after and reporting the
numbers in the pull request.

The benchmark client's own argument handling and output format are themselves
tested, by `clients/extras/test_rocsolver_bench.py` (a Python `unittest` suite
registered with CTest as `test-rocsolver-bench`).

## Pre-submit / CI Gates

rocSOLVER is validated by two independent systems. This is the most common
source of confusion, so it is worth stating plainly: the GitHub checks and the
internal Jenkins jobs run different filters on different hardware.

### Validation Gates and Ownership

| Validation | Blocking on GitHub? | Where it runs | Owner |
| --- | --- | --- | --- |
| Build (Linux, Windows) | Yes | TheRock Multi-Arch CI | TheRock / Math CI |
| `checkin` test subset | Yes | TheRock Multi-Arch CI | rocSOLVER |
| ASAN build and test | Yes | TheRock Multi-Arch CI ASAN | TheRock / rocSOLVER |
| PR description policy | Yes | Libraries PR Bot | ROCm repo policy |
| `pre-commit` | Yes (but a no-op for rocSOLVER paths) | GitHub Actions | ROCm repo policy |
| Precheckin (`checkin*`) | No -- internal | Jenkins | rocSOLVER |
| Extended (`daily*`) | No -- internal, nightly | Jenkins | rocSOLVER |
| Static build + format check | No -- internal | Jenkins | rocSOLVER |
| Static analysis | No -- internal, weekly | Jenkins | rocSOLVER |
| Code coverage | No -- reported, not blocking | Codecov | rocSOLVER |
| Performance | No -- not automated | local only | rocSOLVER |

**1. TheRock Multi-Arch CI (GitHub).** There is no rocSOLVER-specific workflow
or path filter. rocSOLVER is picked up by the generic "Configure Changed
Projects" job in `.github/workflows/therock-multi-arch-ci.yml` (at the
super-repo root), which maps changed paths to projects via
`.github/repos-config.json`. The aggregated gating check is
**"Multi-Arch CI Summary"**.

The test commands live in TheRock, not here. TheRock selects which tests to run
by **CTest label** -- `quick`, `standard`, `comprehensive` or `full` -- against
the `CTestTestfile.cmake` installed to `bin/rocsolver/`. You can run exactly
what CI runs:

```bash
ctest -L '^standard$'    # the checkin tier; also quick, comprehensive, full
```

A pull request touching `projects/rocsolver/**` runs **`standard`** -- that is,
`checkin_*`. Broader sweeps trade depth for breadth: scheduled runs, manual
dispatch, and changes that do not map to a single project run `quick` instead,
which is also the default when no tier is requested.

rocSOLVER's side of the contract is `clients/gtest/CMakeLists.txt`, which
generates one labelled `rocsolver-test_<category>_suite` entry per category in
`clients/gtest/test_categories.yaml`. It needs the super-repo's `shared/` -- see
[A sparse checkout needs the super-repo's `shared/`](#a-sparse-checkout-needs-the-super-repos-shared).

The installed file holds those six suites only, so `test-rocsolver-dlopen` and
`test-rocsolver-bench` are not run this way.

A companion workflow, `therock-multi-arch-ci-asan.yml`, runs on every pull
request as well; see [Sanitizer Coverage](#sanitizer-coverage).

**2. Jenkins (internal).** Defined by `.jenkins/*.groovy` in this project:

| Job | Build | Filter | Timeout |
| --- | --- | --- | --- |
| `precheckin.groovy` | `./install.sh -c --cmake-arg -DWERROR=ON` | `checkin*` | 45 min |
| `extended.groovy` | same | `daily*` | 420 min |
| `static.groovy` | `VERBOSE=1 ./install.sh -c --cmake-arg -DWERROR=ON --static` | `checkin*` | -- |
| `debug.groovy` | `./install.sh -c -g` | build only | -- |
| `staticanalysis.groovy` | -- | format check + static analysis, weekly (`0 1 * * 0`) | -- |

The shared test step (`.jenkins/common.groovy:75`) is:

```bash
./rocsolver-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}
if [ -f ./test-rocsolver-dlopen ]; then ./test-rocsolver-dlopen --gtest_color=yes; fi
HSA_XNACK=1 ./rocsolver-test --gtest_filter=*MANAGED_MALLOC* || true   # gfx90a only
CTEST_OUTPUT_ON_FAILURE=1 ctest -R '^test-rocsolver-bench'
```

These jobs honour pull request labels: `noOptimizations` adds `-n`,
`ci:no-sparse` adds `--no-sparse`, and `gfxall` builds every architecture.

**3. Libraries PR Bot.** A deterministic, rule-based policy checker -- no LLM
involved -- documented in the super-repo's
[docs/LIBRARIES_PR_BOT_FAQ.md](../../docs/LIBRARIES_PR_BOT_FAQ.md). For
rocSOLVER the relevant rules are:

- **PR description (blocking).** The body must be at least 30 characters and
  contain a tracking reference (`JIRA ID : X`, `Closes #123`, a bare `#123`, or
  an issue URL). Failing this adds a `Not ready to Review` label.
- **Unit test (warning only).** A pull request touching code is expected to
  touch a test file. rocSOLVER's tests satisfy this via the `*_gtest.*` and
  `testing_*` filename patterns, even though they live in `clients/gtest/`
  rather than `test/gtest/`.
- **Forbidden files (warning only)** -- keys, certificates, binaries.

**4. Other GitHub automation.** `.github/labeler.yml` applies the
`project: rocsolver` label to any change under `projects/rocsolver/**`.
`.github/skip-ci-config.toml` means documentation-only changes (`*.md`, `*.rst`,
`projects/*/docs/*`) do not trigger a build.

### PR Test Classification

Test suites are named by cadence tier. For a parameterized LAPACK test the
prefix is the first argument to `INSTANTIATE_TEST_SUITE_P`, so a full test name
looks like `checkin_lapack/SYGVDX.batched__float/30`. A plain `TEST_F`/`TEST`
fixture (argument-validation and infrastructure tests) carries the same prefix
directly in the class name instead, e.g. `checkin_misc_LOGGING` in
`logging_gtest.cpp`. Either way, the category patterns below match the
resulting gtest name -- a fixture with no `checkin_`/`daily_`/`weekly_` prefix
compiles and runs, but never in CI.

| Prefix | Instantiations | Meaning |
| --- | --- | --- |
| `checkin_lapack` | 197 | Small sizes. Fast enough to gate a pull request. |
| `daily_lapack` | 197 | Larger sizes, from `large_matrix_size_range`. Nightly. |
| `weekly_lapack` | 15 | Very large matrices and batch counts. |
| `known_bug` | 3 | Reproducers for accepted, open defects. Excluded from most runs. |

`clients/gtest/test_categories.yaml` maps those prefixes onto the CTest
categories that CI selects:

| Category | Patterns | Labels | Timeout |
| --- | --- | --- | --- |
| `quick` | 17 named `checkin*` patterns, excluding `*LARFB*` and `*known_bug*` | `quick`, `pre-commit`, `smoke` | 300 s |
| `standard` | `checkin_*` | `standard`, `pr`, `precheckin` | 2700 s |
| `comprehensive` | `daily_*` | `comprehensive`, `nightly`, `extended` | 25200 s |
| `full` | `checkin_*`, `daily_*`, `weekly_*` | `full`, `weekly`, `all` | 28800 s |
| `ffm-quick` / `ffm-full` | 4 patterns -- `STEDC` float, `GETF2` float-complex, `POTF2` double-complex, `GEQR2` double | `ffm-quick` / `ffm-full` | 7200 s |

All categories exclude `*DISABLED*`. The `quick` list is a hand-maintained
mirror of `quick_tests` in TheRock's `test_rocsolver.py`; the file carries an
explicit instruction to keep the two in sync until the legacy path is removed.

This table is a snapshot -- `test_categories.yaml` is the contract, and its
patterns change to satisfy CI needs. `quick` and the `ffm-*` tiers are
hand-curated allowlists, so a new test is not picked up automatically; if you
want it selected by a particular tier, check that file directly and update it
if needed.

Each row above becomes its own CTest entry named
`rocsolver-test_<category>_suite`, so you can run a tier without retyping its
filter -- provided the super-repo's `shared/` is in your checkout
(see [that section](#a-sparse-checkout-needs-the-super-repos-shared); without it
these entries do not exist):

```bash
ctest -L '^quick$'                    # by label -- anchored, see note above
ctest -R rocsolver-test_standard_suite
```

A third selection mechanism, `rtest.py` driven by `rtest.xml`, predates the CTest
categories and is still present:

```bash
./rtest.py -t psdb                    # checkin*-*known_bug*
./rtest.py -t osdb                    # daily*-*known_bug*
./rtest.py -t hmm                     # *MANAGED_MALLOC*-*known_bug*
./rtest.py --emulation smoke          # also: regression, extended
```

### Flaky Tests and Known Bugs

The `known_bug` suite prefix is the quarantine mechanism. A test that reproduces
an accepted, open defect is instantiated under that prefix instead of
`checkin_lapack`, which keeps the reproducer in the tree and runnable while
excluding it from gating runs. There are three today, in
`clients/gtest/lapack/cholqr_gtest.cpp` and
`clients/gtest/managed_malloc_gtest.cpp`.

`*known_bug*` is excluded by `rtest.xml` and by the `quick` and `ffm-*`
categories. The `standard` and `comprehensive` categories exclude only
`*DISABLED*`; the `known_bug` prefix does not match their `checkin_*` / `daily_*`
patterns, so those suites are not selected there either.

There is no retry-on-failure mechanism, no automatic quarantine, and no
flakiness dashboard. A test that fails intermittently is handled by hand:
investigate, and either fix it or move it to `known_bug` with a tracking issue.

### Coverage

Coverage instrumentation exists and is llvm-cov based, but is run manually:

```bash
./install.sh -c -g --codecoverage
cd build/debug
make coverage GTEST_FILTER='checkin*'     # also: coverage_cleanup, coverage_analysis, coverage_output
```

This produces `coverage-report/coverage.info` in lcov format, via
`LLVM_PROFILE_FILE` -> `llvm-profdata merge -sparse` -> `llvm-cov export`
(`CMakeLists.txt:327-367`). It requires `llvm-profdata` and `llvm-cov`.

A coverage build runs in CI, and Codecov posts a `project/rocSOLVER` check on
every pull request against the 80% target declared in the super-repo's
`codecov.yml`. **Currently, this check is informational and does not gate a
merge** -- it is not in the required-checks list, so a PR can land below
target. Gating may be enforced in the future.

Two constraints apply ROCm-wide and would shape any number that is produced:

- CI cannot yet merge coverage results across multiple GPU runs, so a figure
  gathered from a single runner understates architecture-specific paths.
- Host-side coverage tooling does not accurately capture device code. Device
  coverage via `--offload-new-driver` is being prototyped under RFC0013.

### Nightly and Extended Validation

| Cadence | What runs | Where |
| --- | --- | --- |
| Per pull request | `checkin`-tier tests, ASAN build/test | GitHub |
| Per pull request | `checkin*` on the Jenkins node set | Jenkins |
| Nightly | `daily*` (`extended.groovy`, 7 h budget) | Jenkins |
| Nightly | `therock-multi-arch-ci-nightly.yml`, ASAN nightly | GitHub |
| Weekly (Sun 01:00) | Format check and static analysis | Jenkins |
| Ad hoc | `weekly_lapack` tier, performance sweeps | Manual |

### Supported Configurations

| Where | Architectures | Tested or build-only |
| --- | --- | --- |
| GitHub, Linux | `gfx94X`, `gfx950` | Built and tested |
| GitHub, Linux | `gfx125X` | Build only -- no test hardware available |
| GitHub, Windows | `gfx110X` | Built and tested |
| Jenkins | gfx900, gfx906, gfx908, gfx90a | Built and tested |

The managed-memory tests additionally require a device supporting HMM and
`HSA_XNACK=1`; Jenkins runs them on gfx90a only, and tolerates their failure.

## Sanitizer Coverage

**AddressSanitizer** is supported and gated. `-DBUILD_ADDRESS_SANITIZER=ON`
(`install.sh --address-sanitizer`) switches the GPU target list to xnack-enabled
targets -- `gfx90a:xnack+`, `gfx908:xnack+`, `gfx942:xnack+`, `gfx950:xnack+`
(`CMakeLists.txt:181-213`) -- and suppresses `_FORTIFY_SOURCE`
(`cmake/armor-config.cmake:22`).

ASAN materially changes the library's resource budget, so rocSOLVER adapts to it
rather than just tolerating it. `library/src/include/asan_helpers.hpp` defines
`rocsolver_enable_asan` and `ROCSOLVER_ASAN_VALUE(asan_val, normal_val)`, which
substitutes smaller block sizes and thread counts under instrumentation (for
example `IAMAX_THDS`, `ROCSOLVER_TRSM_MAX_THREADS`, `GETF2_SSKER_MAX_M`). That
substitution is itself tested by `clients/gtest/asan_helpers_gtest.cpp`. Both
clients print a kernel warning at startup when built with ASAN
(`print_asan_kernel_warning()`).

ASAN runs on GitHub via `therock-multi-arch-ci-asan.yml`, which triggers on every
pull request; TheRock selects a faster host-only ASAN configuration for pull
request events and the full device-side instrumentation for manual dispatch.
`therock-multi-arch-ci-asan-nightly.yml` covers the scheduled run. The Jenkins
ASAN job remains a commented-out TODO (`.jenkins/common.groovy:36`).

**ThreadSanitizer, UBSan and MemorySanitizer are not configured** for rocSOLVER
in either CI system or in the build.

## Static Analysis

| Tool | Status |
| --- | --- |
| `clang-format` | Configured (`.clang-format`). Enforced by the Jenkins `static` and `staticanalysis` jobs (`formatCheck = true`), and by the optional local git hooks. **Not** enforced by GitHub CI. |
| Jenkins static analysis | Weekly (`staticanalysis.groovy`, Sundays 01:00). The analyzer itself lives in the external rocJenkins shared library. |
| `pre-commit` | The GitHub `pre-commit` check runs on every pull request, but rocSOLVER paths are listed in the top-level `exclude:` block of the super-repo `.pre-commit-config.yaml`. No hook -- including `clang-format` and the `bandit` security hook -- inspects rocSOLVER files, and the check reports success. The opt-in procedure is in the super-repo `CONTRIBUTING.md`. |
| `clang-tidy` | No `.clang-tidy` in this project, and rocSOLVER is not in the opted-in library list of `.github/workflows/clang-tidy.yml`. |
| CodeQL | Scheduled weekly, and configured for Python and GitHub Actions only -- it does not analyze rocSOLVER C++. |

## For New Contributors

Build with `./install.sh -c`, then run
`./build/release/clients/staging/rocsolver-test --gtest_filter='checkin*'`
before opening a pull request. The full suite is large; filter aggressively
while developing.

Useful environment variables:

| Variable | Purpose |
| --- | --- |
| `ROCSOLVER_TEST_DATA` | Directory holding the sparse CSR test matrices. |
| `ROCSOLVER_TEST_DEBUG` | Keep temporary log files when a logging test fails. |
| `HSA_XNACK=1` | Required for the managed-memory (HMM) tests. |
| `ROCSOLVER_LOG_PATH`, `ROCSOLVER_LOG_TRACE_PATH`, `ROCSOLVER_LOG_BENCH_PATH`, `ROCSOLVER_LOG_PROFILE_PATH` | Logging destinations; see `docs/howto/logging.rst`. |
| `LLVM_PROFILE_FILE`, `GTEST_LISTENER` | Used by the coverage targets. |

### Choosing the Right Test Type

| You are... | Do this |
| --- | --- |
| Adding a routine | Add `clients/common/<group>/testing_<r>.hpp` and `.cpp`, add `clients/gtest/<group>/<r>_gtest.cpp`, and add the entry to `clients/common/misc/rocsolver_dispatcher.hpp` so it is benchmarkable. Instantiate `checkin_lapack` and `daily_lapack` suites. Two separate, hand-maintained CMake registrations are required: the `testing_<r>.cpp` source goes in `clients/CMakeLists.txt`'s instantiation lists, and the `<r>_gtest.cpp` source goes in `clients/gtest/CMakeLists.txt` -- missing the second leaves the new tests out of `rocsolver-test` and CI even though the code compiles. |
| Adding a standalone test not tied to one routine | Write a plain `TEST_F`/`TEST` fixture, e.g. `logging_gtest.cpp`, and register it in `clients/gtest/CMakeLists.txt`'s `others_test_source` list. Name the fixture `checkin_misc_<Name>` (per `checkin_misc_LOGGING`) to be selected by any CI gate. Then check `test_categories.yaml` for whether your target tier is an automatic prefix match or a hand-curated allowlist needing an explicit addition. |
| Fixing a numerical bug | Add the failing dimensions to that routine's `matrix_size_range` so the reproducer runs in the `checkin` tier. |
| Fixing an argument-handling bug | Extend the routine's `<r>_checkBadArgs()`. |
| Changing algorithm selection | Exercise the relevant `--alg_mode` / `--hetrd_alg_mode` path and confirm both paths are covered by a test case. |
| Making a performance change | Run `rocsolver-bench --perf 1` before and after, and put the numbers in the pull request. |
| Adding a routine behind a feature flag | Confirm the test asserts `rocblas_status_not_implemented` when the flag is off. |
| Hitting an accepted open defect | Move the reproducer to a `known_bug` suite and link the tracking issue. |

New test cases should be small enough for the `checkin` tier unless they
genuinely need size -- the `checkin` budget is 45 minutes for the whole suite.

## Owners and Review Cadence

The enforced ownership is defined in the super-repo's root `.github/CODEOWNERS`:

```
/projects/rocsolver/                      @ROCm/solvers-reviewers
/projects/rocsolver/**/*.md               @ROCm/solvers-reviewers @ROCm/solvers-docs-reviewers
/projects/rocsolver/**/*.rst              @ROCm/solvers-reviewers @ROCm/solvers-docs-reviewers
/projects/rocsolver/docs/                 @ROCm/solvers-reviewers @ROCm/solvers-docs-reviewers
/projects/rocsolver/library/include/      @ROCm/solvers-reviewers @ROCm/solvers-docs-reviewers
```

Changes to this file therefore require both the solvers and the documentation
reviewer groups.

This project also carries its own `.github/CODEOWNERS` listing individual
maintainers. GitHub only honours a `CODEOWNERS` file at the repository root,
`.github/` or `docs/`, so the nested file is informational; the root file is what
is enforced.

Technical POC: *TBD*.

Review cadence for this document: *TBD*. It should be revisited whenever the CI
topology changes -- in particular as the migration from Jenkins to TheRock
progresses.

## Further Reading

- [CONTRIBUTING.md](CONTRIBUTING.md) -- pull request process, style, test basics
- [docs/howto/clients.rst](docs/howto/clients.rst) -- testing and benchmarking clients
- [docs/installation/build.rst](docs/installation/build.rst) -- build options
- [docs/reference/env_variables.rst](docs/reference/env_variables.rst) -- environment variables
- [docs/howto/logging.rst](docs/howto/logging.rst) -- logging
- [scripts/perf/README.md](scripts/perf/README.md) -- performance scripts
- [Libraries PR Bot FAQ](../../docs/LIBRARIES_PR_BOT_FAQ.md) -- super-repo pull request policy
- [rocBLAS TESTING.md](../rocblas/TESTING.md) -- sibling component, same template
