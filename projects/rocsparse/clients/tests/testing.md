# rocSPARSE Testing Strategy

**Status:** Draft
**Owner:** @doctorcolinsmith
**Technical Lead:** @ntrost57
**Last Updated:** 2026-08-06

This document describes how rocSPARSE is tested today, which signals actually gate a merge, and
where the gaps are. It follows the ROCm-wide TESTING.md template and is written as a description of
the current state rather than an aspirational one. A gap that is written down is one that can be
argued about and closed.

---

## Component Overview

rocSPARSE is the ROCm library of Basic Linear Algebra Subroutines (BLAS) for **sparse** computation,
implemented in HIP and optimized for AMD GPUs. It provides sparse level 1/2/3 routines, sparse
format conversions, preconditioner building blocks, and a set of generic API entry points, and it is
the AMD backend for the portable [hipSPARSE](https://github.com/ROCm/rocm-libraries/tree/develop/projects/hipsparse)
marshalling library.

**Where it sits in the ROCm stack:** Core math SDK. rocSPARSE sits directly on top of the HIP
runtime and toolchain. It optionally depends on **rocBLAS** (`BUILD_WITH_ROCBLAS`, default ON) and
**rocPRIM**, and optionally uses rocTX for tracing (`BUILD_WITH_ROCTX`).

**Key architectural constraint that shapes testing:** almost every rocSPARSE routine dispatches a
GPU kernel and its result is only meaningful on real AMD hardware. The hardware-independent surface
(argument validation, descriptor/handle management, format-conversion bookkeeping, workspace-size
queries) is small relative to the device code, so confidence comes overwhelmingly from
device-executed integration tests that compare against a host reference rather than from pure host
unit tests.

---

## Development Workflow

What a developer does between making a change and getting it merged.

**1. Build the library with test and benchmark clients:**

```bash
cd projects/rocsparse
# -c builds samples + tests + benchmarks; -d fetches build dependencies (incl. googletest)
./install.sh -dc -a gfx942
# equivalently, configure directly:
CXX=/opt/rocm/bin/amdclang++ cmake -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=ON ../..
```

Out-of-source builds land in `build/<release|debug|release-debug>/clients/staging/`.

The functional suites that read `.csr` inputs need the SuiteSparse test matrices, but you don't need a separate step to get them: the build above downloads and converts them automatically into `build/<release|debug|release-debug>/clients/matrices`.

> **Note:** If you already have the matrices downloaded to a folder, pass `--matrices-dir <path_o_matrix_folder>` to the install script to reuse them and avoid re-downloading on a rebuild:
>
> ```bash
> ./install.sh -c -a gfx942 --matrices-dir <my_matrix_folder>
> ```
>
> To populate such a folder once (e.g. a shared location outside the build tree), use `--matrices-dir-install`:
>
> ```bash
> ./install.sh --matrices-dir-install <my_matrix_folder>
> ```

**2. Run the tests that match what you touched:**

| You changed | Run this | Needs a GPU |
|---|---|---|
| Any routine | `./clients/staging/rocsparse-test --gtest_filter='*quick*-*known_bug*'` | Yes |
| A specific routine | `./clients/staging/rocsparse-test --gtest_filter='*csrmv*'` | Yes |
| Something PR-scoped | `./clients/staging/rocsparse-test --gtest_filter='*quick*:*pre_checkin*-*known_bug*'` | Yes |
| Argument/format validation only | the same, filtered to `*bad_arg*` | No (bad-arg cases are host-side) |

On the single-GPU container, run through the serializer pinned to device 0:

```bash
HIP_VISIBLE_DEVICES=0 gpu-run ./clients/staging/rocsparse-test --gtest_filter='*quick*-*known_bug*'
```

> rocSPARSE auto-skips large configs with "Insufficient memory" on memory-constrained cards. That is
> the suite's memory guard, not a failure.

**3. Add the right kind of test** — see [Choosing the Right Test Type](#choosing-the-right-test-type).

**4. Open the PR** targeting `develop`. The pre-checkin GTest run (`*quick*:*pre_checkin*`, excluding
`*known_bug*`) is the merge gate; another rocSPARSE team member reviews and approves.

---

## Testing Strategy and Layers

### Unit Testing Strategy

**Purpose:** validate individual library components in isolation — host-pure utilities, internal
building blocks (primitives, algorithm-selection logic, info structs), and hardware-independent logic
such as argument validation, error-code propagation, and type/format bookkeeping.

> **Work in progress:** the dedicated unit-test layer described below is being landed through a
> series of in-flight PRs (`clients/unittests/`) and is **not yet on `develop`**. The details here
> (binaries, sources, CTest wiring) reflect that parallel work and will be finalized in this document
> once those PRs merge.

rocSPARSE has a dedicated unit-test layer under `clients/unittests/`, being introduced in phases and
kept deliberately separate from the YAML/Arguments-driven integration suite (`rocsparse-test`). It
does **not** use the `gentest` data pipeline, and it compiles selected library translation units
directly into the test binary — because `librocsparse` is built with hidden symbol visibility, the
internal (non-exported) symbols under test are not reachable by linking `roc::rocsparse`, so the
`.cpp` files are compiled in without any library source changes. Because it reaches into private
headers/sources under `library/src`, this layer only builds **in-tree** (it is skipped under
`BUILD_CLIENTS_ONLY`). It is split into two binaries:

* **`rocsparse-unit-test`** — fast, **GPU-independent** (CPU-only) unit tests with a minimal gtest
  main. Exercises host-pure components (e.g. enum-trait, index-type and data-type utilities, the
  csrmv-adaptive analysis logic). Registered with CTest (`ctest -R rocsparse-unit-test`) so CI can
  run it as an early, no-GPU gate.
* **`rocsparse-unit-test-device`** — **GPU** unit tests that link `hip::device` and launch individual
  library kernels/primitives in isolation (e.g. internal scan/find/sort/RLE primitives, host and
  numeric level-1/2/3, conversion, preconditioner, reordering, and generic paths, plus internal
  algorithm-selection and info-struct logic). It must run on a GPU through the serializer and is
  CTest-labeled `gpu`, so it is scheduled on GPU nodes only and is **not** part of the fast CPU gate.

In addition, the integration binary `rocsparse-test` carries the hardware-independent cases that are
naturally expressed through its YAML pipeline rather than as standalone units:

* **Bad-argument tests** — every routine has a `testing_<routine>_bad_arg` implementation
  (`clients/testings/testing_<routine>.cpp`) exercised through YAML `function: <routine>_bad_arg`
  entries. These validate null-pointer handling, invalid sizes, unsupported types, and status-code
  propagation, and do not require a kernel launch.
* **Auxiliary API tests** — `clients/tests/test_auxiliary.cpp` (`TEST(auxiliary_pre_checkin, ...)`)
  covers handle create/destroy and descriptor management directly.

**Framework:** GoogleTest (pinned to `v1.15.2` in `deps/external-gtest.cmake`; auto-downloaded when
absent via `deps/CMakeLists.txt` `BUILD_GTEST=ON`).

**Location / structure:** unit tests live in `clients/unittests/` (`unit_test_*.cpp` +
`rocsparse_unit_test_main.cpp`), with the internal library `.cpp` files under test compiled directly
into the two binaries. This is distinct from the integration suite's three-layer per-routine pattern
(`clients/tests/test_<routine>.yaml` → `test_<routine>.cpp` → `clients/testings/testing_<routine>.cpp`,
expanded by `clients/common/rocsparse_gentest.py` into `rocsparse_test.data`), which drives the
`_bad_arg` and numerical cases.

**How to run:** `ctest -R rocsparse-unit-test` runs the CPU-only unit binary (no GPU);
`HIP_VISIBLE_DEVICES=0 gpu-run ./clients/staging/rocsparse-unit-test-device` runs the device unit
binary on the GPU. The integration binary's host-side subset runs via
`./clients/staging/rocsparse-test --gtest_filter='*bad_arg*'` (a device/driver may be initialized,
but no compute kernel is launched).

**What is NOT covered by the CPU-only unit binary:** numerical correctness of routines, kernel
selection, and device-side conversions — these require a GPU and are exercised by
`rocsparse-unit-test-device` (unit level) and by the integration suite (end-to-end).

**Coverage expectation:** the long-term ROCm-wide goal is >95% line coverage of hardware-independent
paths; this is not mandated initially and will be pursued in phases. A useful property of the unit
layer is that library sources compiled into the two binaries are instrumented under
`BUILD_CODE_COVERAGE` (`-fprofile-instr-generate -fcoverage-mapping`), so device kernels and internal
primitives exercised by `rocsparse-unit-test-device` begin to emit LLVM profile data that the
integration suite's host-side instrumentation cannot capture. See [Coverage](#coverage).

---

### Integration Testing Strategy

**Purpose:** validate behavior that requires a real GPU — numerical correctness of every sparse
routine against a host reference, format conversions, and kernel selection across data types, index
types, matrix formats, and problem sizes.

Integration tests are the bulk of rocSPARSE testing: roughly **150+ routines**, each with a
parameterized/typed GoogleTest suite driven by YAML. Cases are typed via
`clients/tests/rocsparse_test_config.hpp` (real/complex, mixed index types) and parameterized with
`TEST_P` + `INSTANTIATE_TEST_CATEGORIES`, so a new data type or index combination inherits the
standard case matrix rather than requiring hand-written variants.

**Runtime tiers (YAML `category:` field → GTest suite prefix):**

| Tier | Meaning | Typical use |
|---|---|---|
| `quick` | small, fast cases | local smoke, PR |
| `pre_checkin` | PR / checkin scope | PR merge gate |
| `nightly` | larger shapes and broader type coverage | nightly |
| `stress` | largest / longest cases | weekly / release |
| `known_bug` | cases exposing a tracked defect | excluded from all gating runs |

**Per-routine time budgets (rough guidelines):** each tier has a target maximum wall-clock time for
a *single routine's* cases in that tier. The budget applies to the total time of one routine's cases
at one tier — e.g. `./rocsparse-test --gtest_filter=*quick/csrmv*` should finish in under 1000 ms.

| Tier | Target max time per routine | Example filter |
|---|---|---|
| `quick` | < 1000 ms (1 s) | `--gtest_filter=*quick/csrmv*` |
| `pre_checkin` | < 10000 ms (10 s) | `--gtest_filter=*pre_checkin/csrmv*` |
| `nightly` | < 100000 ms (100 s) | `--gtest_filter=*nightly/csrmv*` |

> These budgets are rough guidelines, not hard limits: actual run time depends on the hardware. They
> exist to keep any one routine from dominating a tier's total run time; a routine that consistently
> and substantially exceeds its budget is a signal to trim redundant cases (see the test-size guidance
> below).

**Test data / matrices:** functional suites read `.csr` matrices converted from **24 SuiteSparse**
matrices (e.g. `nos1`–`nos7`, `amazon0312`, `webbase-1M`) downloaded and MD5-verified by
`cmake/ClientMatrices.cmake` (mirror overridable via `ROCSPARSE_TEST_MIRROR`). The runtime matrix
directory is set with `--matrices-dir` or `ROCSPARSE_CLIENTS_MATRICES_DIR`.

**What requires GPU hardware:** all numerical-correctness and conversion cases. **What runs without
a compute kernel:** `*bad_arg*` and auxiliary API cases.

**Managed-memory (HMM) coverage:** on gfx90a/gfx942 the legacy Jenkins lane executes the entire
pre-checkin filter twice — once with `HSA_XNACK=0` and once with `HSA_XNACK=1` — to cover managed
memory. `rtest.xml` also defines a dedicated `hmm` set (`--gtest_filter=*csrmv_managed*`, run with
`HSA_XNACK=1`).

**What runs on PRs:** the CTest `standard` category (`quick` + `pre_checkin`, excluding `*known_bug*`)
on changed projects, via TheRock CI (default `test_type: standard`).
**What runs nightly:** the `comprehensive` category (adds the `nightly` tier) via the TheRock nightly
workflows; the legacy Jenkins `extended.groovy` mirrors this with `*nightly*` (timeout ~600 min).
**What runs at release / on demand:** `full` category (adds the `stress` tier), and the emulation
`smoke`/`regression`/`extended` YAML subsets driven through `rtest.py`.

**Test-size / coverage guidance:** prefer a small set of representative sizes over exhaustive
numerical variants — the typed/parameterized suites already sweep type and index combinations, so
adding many near-duplicate sizes increases run time without adding meaningful coverage. Large configs
are intentionally guarded and skipped with "Insufficient memory" on small cards.

---

### Performance and Benchmarking Testing

**Purpose:** detect throughput/bandwidth regressions per routine on a given architecture. Absolute
numbers are not comparable across GFX targets.

| Item | Detail |
|---|---|
| Stack layer | Core SDK (sparse BLAS primitive) |
| Metrics measured | Time, GFLOP/s, and memory bandwidth per routine |
| How benchmarks are run | `rocsparse-bench` from `clients/staging/`, e.g. `./rocsparse-bench -f csrmv --bench-x -M 10 20 30 40` (command-line expansion sweeps option lists) |
| Baseline — stored per architecture | Not stored/aggregated in this repo (manual comparison); perf regression is tracked outside this repo. Results must not be aggregated across GFX |
| Where results are stored | JSON by default (`a.json`); optional YAML export; optional `rocsparse_bench_memstat.json` with memstat |
| Regression threshold | In-repo: `scripts/rocsparse-bench-regression.py` compares JSON runs with a `--tol` (default 2%); not wired into public CI. Perf regression is tracked outside this repo |
| Gating approach | Manual in-repo; perf regression tracked outside this repo |
| GPU profiling | Not integrated into the benchmark flow |

**Gating:**

| Gating Level | Status | Notes |
|---|---|---|
| Public CI (TheRock) automated gate | No | No perf gate in this repo's PR or nightly CI |
| Regression tracked outside this repo | Yes | Perf regression is tracked outside this repo |
| Manual review (in-repo) | Yes | `rocsparse-bench-regression.py` compares runs on request |
| Release qualification | Partial | Performance reviewed before release; not an automated sign-off in this repo |

**Known gaps:** no per-architecture baseline or automated regression threshold in this repo /
TheRock CI. Perf regression is tracked outside this repo; the gap here is that this coverage is not
visible or reproducible from this repository.

---

## Why We Test This Way

rocSPARSE owns its kernels, but nearly all meaningful behavior is only observable on real AMD
hardware, so the strategy is integration-dominant: a large YAML-driven GoogleTest suite runs each
routine on-device and validates against a host reference. The hardware-independent surface (argument
validation, handle/descriptor lifecycle, auxiliary API) is exercised by the `*bad_arg*` and auxiliary
cases inside that integration binary. Alongside it, a dedicated unit-test layer
(`clients/unittests/`) is being introduced to test individual components in isolation — a fast
CPU-only `rocsparse-unit-test` for host-pure logic and a `rocsparse-unit-test-device` that launches
single kernels/primitives on the GPU — reaching internal code that the end-to-end suite exercises
only indirectly.

The tier system (`quick` → `stress`) is a sampling strategy over a large combinatorial space (routine
× data type × index type × matrix format × size); exhaustive coverage is not achievable, so the tiers
trade breadth for time-to-signal. Typed/parameterized suites keep the matrix maintainable: adding a
type or index combination extends coverage without new hand-written cases.

---

## Pre-submit / CI Gates

The presubmit gate is the monorepo **TheRock CI** GitHub Actions workflow
(`.github/workflows/therock-ci*.yml`), which runs on every pull request and push to `develop`
(`.github/scripts/therock_configure_ci.py`). It builds and tests only the projects whose files
changed and, by default, runs the CTest **`standard`** category — i.e. `quick` + `pre_checkin`,
excluding `*known_bug*` (see `clients/tests/test_categories.yaml`). The scope can be widened per PR
with labels (`test:rocsparse`, `test_type:comprehensive` / `test_type:full`); doc-only changes
(`*.md`, `docs/*`) skip CI. rocSPARSE and hipSPARSE share TheRock's `sparse` component
(`projects_to_test: [rocsparse, hipsparse]` in `.github/scripts/therock_matrix.py`), so a PR touching
either one builds and tests both. Broader tiers run in the TheRock nightly workflows
(`therock-ci-nightly.yml`, `therock-multi-arch-ci-nightly.yml`) and dedicated ASAN workflows
(`therock-multi-arch-ci-asan*.yml`). Repo-wide quality workflows (`pre-commit`, `clang-tidy`,
`codeql`) apply to all components.

Internal AMD **Jenkins** pipelines (`.jenkins/`) also exist but are legacy: they run across a range
of GPUs (gfx908 / gfx90a / gfx942 / gfx950 / gfx1201) on cron/nightly triggers and mirror the same GTest filters —
`precheckin.groovy` runs `*quick*:*pre_checkin*`, `extended.groovy` runs `*nightly*`, `asan.groovy`
runs `*quick*:*pre_checkin*` under AddressSanitizer, and `codecov.groovy` uploads coverage. All
Jenkins test commands exclude `*known_bug*`.

### Validation Gates and Ownership

| Validation Area | Required Before Merge | Owner | Notes |
|---|---|---|---|
| Build (Linux; changed projects) | Yes | CI / DevOps | TheRock CI; Jenkins `precheckin`/`static`/`debug` (legacy) |
| Unit / bad-arg tests | Yes | Component team | In the CTest `standard` category (`quick`+`pre_checkin`) |
| Integration tests | Yes | Component team | `standard` category, excluding `*known_bug*` |
| Formatting | Yes | CI / DevOps | Repo-wide `pre-commit` |
| Static analysis | Yes | CI / DevOps | `clang-format`, `clang-tidy`, `codeql`, Jenkins `staticanalysis.groovy` |
| ASAN | Separate lane | CI / DevOps | Dedicated TheRock ASAN workflows (multi-arch / nightly) + Jenkins `asan.groovy`; not a confirmed per-PR blocking gate |
| Code coverage | No | Component team / CI | `codecov.groovy`, informational (Codecov flag `rocSPARSE`) |
| Shared validation infra | N/A | TheRock team | Shared build/validation infrastructure |
| Release qualification | N/A | Component team + QA + TPM | Readiness and known-gap review |

**Build command pattern (Jenkins pre-checkin):**

```bash
./install.sh --matrices-dir-install ${JENKINS_HOME_DIR}/rocsparse_matrices \
  && ./install.sh -c -a $gfx_arch --matrices-dir ${JENKINS_HOME_DIR}/rocsparse_matrices
```

### PR Test Classification

| Status | Applies to |
|---|---|
| Trusted gate | TheRock CI `standard` category (`quick`+`pre_checkin`, excluding `*known_bug*`) on changed projects |
| Informational | Coverage upload (Codecov); nightly `*nightly*` / comprehensive results; ASAN lane |
| Unstable / flaky | `known_bug`-tagged cases (excluded from gating) |

**Flaky / known-bug policy:** rocSPARSE has no `known_bugs.yaml`. A case that exposes a tracked
defect is tagged `category: known_bug` (either directly in the routine's YAML or reassigned by
`rocsparse_gentest.py` from a `Known bugs` note in `rocsparse_common.yaml`), and every gating run
excludes `*known_bug*`. This keeps known failures out of blocking runs but does not by itself carry an
owner or ticket per case — a tracked quarantine list with expiry is a gap (see
[Known Gaps](#known-gaps-summary)). A known-bug tag is not an accepted permanent state.

---

## Coverage

**Tool:** LLVM source-based coverage (`-fprofile-instr-generate -fcoverage-mapping`), enabled with
`-DBUILD_CODE_COVERAGE=ON` (via `install.sh --codecoverage`, which requires a Debug `-g` or
RelWithDebInfo `-k` build). `lcov`/`gcovr` are explicitly not used because they do not work with LLVM
coverage.

**How to build and run:**

```bash
./install.sh -kc --codecoverage -a gfx942
cd build/release-debug
make coverage_cleanup coverage GTEST_FILTER='*quick*:*pre_checkin*-*known_bug*'
# targets: coverage_analysis -> coverage (llvm-profdata merge -> llvm-cov export lcov -> filter -> genhtml)
```

The report is produced under `coverage-report/`; exclusions are applied by
`scripts/filter_lcov_exclusions.py`. Jenkins `codecov.groovy` uploads `coverage.info` to Codecov with
flag `rocSPARSE`.

**Code coverage vs. test coverage** are distinct:
* *Code coverage* = fraction of lines executed (e.g. 700 of 1,000 → 70%).
* *Test coverage* = fraction of intended functionality exercised (types, index types, formats, sizes,
  platforms). Historically, host-side instrumentation did not capture device kernels, so rocSPARSE's
  measured code coverage understated work done on-device. The new `rocsparse-unit-test-device` binary
  (`clients/unittests/`) is instrumented under `BUILD_CODE_COVERAGE` and compiles library device
  sources directly in, so it begins to capture device-path coverage; broad device coverage is still
  being phased in.

**Wavefront-size gap:** the reported coverage percentage is currently computed on either 32-wavefront
hardware or 64-wavefront hardware, but not both — a report is not generated for each and then combined
into a single report. This is a major gap because many rocSPARSE routines dispatch to different kernels
at runtime based on the wavefront size (32 vs. 64), so in any given coverage run the alternate branch
(the `if` or the `else`) is never hit. This is part of why measured coverage sits below the 80% target.

**Scope:** measured on Linux; Windows coverage is not tracked separately.

---

## Nightly Validation

Beyond PR validation (the `standard` category), the TheRock nightly workflows run the
`comprehensive` category and additional GPU families; the legacy Jenkins `extended.groovy` mirrors
this. Nightly adds:

* The `nightly` tier — larger shapes and broader type/format coverage (Jenkins `extended.groovy`
  timeout ~600 min).
* Additional GPU architectures beyond the PR runners, plus a dedicated ASAN lane
  (`therock-multi-arch-ci-asan-nightly.yml`).
* Emulation subsets (`smoke`, `regression`, `extended`) via `rtest.py --emulation`, reading
  `rocsparse_smoke.yaml`, `rocsparse_regression.yaml`, and `rocsparse_extended.yaml`.

---

## Supported Configurations

Default GPU targets come from `DEFAULT_GPU_TARGETS` in the root `CMakeLists.txt`.

| Configuration | Linux | Windows | Notes |
|---|---|---|---|
| gfx908 / gfx90a / gfx942 / gfx950 | Full — PR / Nightly / Release | Not tested | Core CDNA targets |
| gfx1201 | Full — PR / Nightly / Release | Not tested | RDNA |
| gfx1151 (Strix Halo) | Not tested | Full — PR / Nightly / Release | `f64_r` / `f64_c` cases excluded (`exclude_gpu_gfx1151`) |
| ASAN (gfx908 / gfx90a / gfx942 `xnack+`) | Partial — Dedicated lane / Nightly | Not tested | AddressSanitizer build, xnack+ only |
| gfx900 / gfx906 | Not tested | Not tested | Older CDNA/GCN |
| gfx1030 / gfx110x / gfx1200 | Not tested | Not tested | RDNA |

**Explicitly not tested / guaranteed:** non-listed gfx targets; any
configuration marked "Not tested" above (including gfx900 / gfx906 and most RDNA parts). Note that
Linux and Windows validation cover disjoint sets of targets — the CDNA parts and gfx1201 are validated
on Linux only, while gfx1151 is validated on Windows only. Configurations are guarded at runtime by the
"Insufficient memory" skip on small cards.

---

## ASAN / TSAN / Sanitizer Coverage

**AddressSanitizer:** `-DBUILD_ADDRESS_SANITIZER=ON` (`install.sh --address-sanitizer`) builds with
`-fsanitize=address -shared-libasan`, links with `lld`, defines `ROCSPARSE_WITH_ASAN`, and depends on
`hip-runtime-amd-asan`. Under ASAN, GPU targets are restricted to `gfx908:xnack+`, `gfx90a:xnack+`,
`gfx942:xnack+`, and the Fortran clients are forced off. It catches host and (xnack+) device memory
errors: out-of-bounds and use-after-free.

**Runtime (Jenkins `asan.groovy`):** `ASAN_OPTIONS=detect_leaks=1`,
`LSAN_OPTIONS=suppressions=suppr.txt` (suppresses leaks in `libhsa-runtime64`, `libamd_comgr`,
`libamdhip64`, `libhsakmt`), `ASAN_SYMBOLIZER_PATH=/opt/rocm/llvm/bin/llvm-symbolizer`. The ASAN lane
runs `*quick*:*pre_checkin*`.

**What is explicitly not covered:** TSAN, UBSAN, and MSAN are not wired into rocSPARSE; non-xnack
device configurations are not ASAN-covered.

---

## Choosing the Right Test Type

| Scenario | What to add |
|---|---|
| New host-pure utility or internal building block (no GPU) | A unit test in `clients/unittests/` built into `rocsparse-unit-test`; compile the internal `.cpp` under test directly into the binary |
| New internal kernel/primitive to test in isolation | A unit test built into `rocsparse-unit-test-device` (runs on GPU via the serializer) |
| New argument/format validation or status-code path | A `testing_<routine>_bad_arg` case (host-side, no kernel) |
| New or changed on-device routine behavior | A YAML case in `test_<routine>.yaml` at the appropriate tier, validated against the host reference |
| New data type or index type | Extend the typed config (`rocsparse_test_config*`); the parameterized suite picks up the combinations |
| Bug fix | A regression case that fails before the fix; if it exposes a still-open defect, tag `category: known_bug` with a tracking issue |
| Performance-sensitive change | Run `rocsparse-bench` before/after and compare with `scripts/rocsparse-bench-regression.py`; include the delta in the PR |
| New GPU architecture | Validate on that target and update Supported Configurations |

---

## Known Gaps Summary

| Gap | Regression risk | Impact | Mitigation today |
|---|---|---|---|
| No public/TheRock perf regression gate (PR or nightly) in this repo | Medium | Medium | Perf regression is tracked outside this repo; in-repo `rocsparse-bench-regression.py` for manual checks |
| No per-architecture perf baseline reproducible from this repo | Medium | Medium | Perf regression is tracked outside this repo; in-repo JSON logs only |
| Device-code coverage not captured (host-side instrumentation only) | High | High | Codecov reports host paths; device coverage untracked |
| No tracked quarantine list (owner + ticket + expiry) for `known_bug` cases | Low | Low | `*known_bug*` excluded from gating; linkage is ad-hoc |
| Dedicated unit-test layer still being phased in | Low | Medium | `rocsparse-unit-test` (CPU) + `rocsparse-unit-test-device` (GPU) under `clients/unittests/`; `*bad_arg*` / auxiliary cases also cover host paths inside `rocsparse-test` |
| No TSAN/UBSAN/MSAN | High | High | ASAN only |

---

## Owners and Review Cadence

**Review this document when:**
* A new test tier, CTest category, or CI lane is added.
* A regression escapes to a downstream consumer (e.g. hipSPARSE) — direct evidence of a gap here.
* Before a major release, alongside the known-gap review.

The measure of whether this document is working: the Known Gaps table shrinks over time.
