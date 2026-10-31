# hipBLAS Testing

## Component Overview

hipBLAS is a BLAS marshalling library. It presents one API (C, Fortran, and ILP64 forms, plus a small set of rocSOLVER-backed routines) and forwards each call to a worker BLAS library. The AMD build (`library/src/amd_detail/`) calls rocBLAS and, when `BUILD_WITH_SOLVER=ON`, rocSOLVER. The CUDA build (`USE_CUDA=ON`, `library/src/nvidia_detail/`) calls cuBLAS.

Two facts shape testing:

**The client suite tests the shim and the worker together.** A passing `hipblas-test` case means hipBLAS accepted the arguments, selected the C or Fortran and 32-bit or ILP64 entry point, set pointer mode, and returned a result that matches a host reference BLAS. Kernel selection and GEMM performance live in rocBLAS or cuBLAS. A hipBLAS failure can be a marshalling bug or a backend bug; the client cannot tell them apart by itself.

**Most validation requires a GPU.** `hipblas-test` exits at startup when `query_device_property()` finds no device. Numerical checks, bad-argument checks, and auxiliary set/get calls all go through the hipBLAS handle and device memory. There is no separate host-only test binary.

Major dependencies: HIP, the selected backend (rocBLAS, and rocSOLVER when solvers are enabled, or cuBLAS), an ILP64 host reference (AOCL BLIS plus libFLAME when `LINK_BLIS=ON` and those libraries are found, otherwise OpenBLAS64 / TheRock `therock-host-blas64` plus LAPACK), Google Test, and `shared/ctest` when the monorepo checkout contains it. OpenMP is used for the host reference when the toolchain provides it.

## Development Workflow

What a developer typically does between a change and merge:

**1. Build** with clients enabled. From `projects/hipblas`:

```bash
python3 rmake.py -c          # library + tests, benchmarks, and samples
python3 rmake.py -dc         # also build client dependencies the first time
```

`rmake.py -n` sets `BUILD_WITH_SOLVER=OFF` (no rocSOLVER functions; ignored for the CUDA backend). `rmake.py --cuda` sets `USE_CUDA=ON`. Staged binaries land under `build/release/clients/staging/`.

**2. Run tests matched to the change.**

| You changed | Run this | Needs a GPU |
| --- | --- | --- |
| One routine or its harness | `hipblas-test --gtest_filter=*quick*<function>*` | Yes |
| Bad-argument / status paths | `hipblas-test --gtest_filter=*pre_checkin*<function>*` | Yes |
| Smoke matrix | `hipblas-test --yaml hipblas_smoke.yaml` | Yes |
| CTest label (`shared/ctest` present) | `ctest -L quick` or `ctest -L standard` from `build/release` | Yes |
| Jenkins-style pre-checkin / extended set | `python3 rtest.py -t psdb` or `-t osdb` from `projects/hipblas` | Yes |
| Solver routine (`getrf`, `getrs`, `getri`, `geqrf`, `gels`) | Rebuild with `BUILD_WITH_SOLVER=ON`; filter `*getrf*` (and the matching name) | Yes |
| CUDA backend behavior | Rebuild with `--cuda`; `backend_flags: NVIDIA` cases are the ones that instantiate | Yes (NVIDIA) |
| Performance-sensitive marshalling or a backend swap | `hipblas-bench` on representative sizes; compare to a baseline manually | Yes |
| Format / hooks only | `pre-commit run --files <changed files>` from the monorepo root | No |

**3. Add the right validation** — see [Coverage Expectations by Change Type](#coverage-expectations-by-change-type) and [Choosing the Right Test Type](#choosing-the-right-test-type).

**4. Open a PR** targeting `develop`, following the repository [CONTRIBUTING.md](../../CONTRIBUTING.md). hipBLAS-specific notes are in [`docs/contribute.rst`](docs/contribute.rst).

**5. Watch CI.** TheRock runs `test/therock/test_hipblas.py` (see [Pre-submit / CI Gates](#pre-submit--ci-gates)). Math CI / Jenkins jobs are not defined in this tree.

---

The remainder of this document has two layers:

1. **Testing mechanics** — how `hipblas-test`, YAML, CTest, and `rtest.py` are wired (below).
2. **Testing strategy** — what we validate, how CI gates merge, known gaps, and what contributors should run ([strategy sections](#testing-strategy-and-layers)).

Client CLI and the smoke YAML are also described in [`docs/how-to/using-hipblas-clients.rst`](docs/how-to/using-hipblas-clients.rst). Build flags are in [`docs/install/Linux_Install_Guide.rst`](docs/install/Linux_Install_Guide.rst) and [`docs/install/Windows_Install_Guide.rst`](docs/install/Windows_Install_Guide.rst).

## Testing mechanics

How the hipBLAS clients are built and executed.

## Client executables

After building with clients enabled, binaries are staged under `projects/hipblas/build/release/clients/staging/`:

| Client | Role |
|--------|------|
| `hipblas-test` | Correctness and API validation via Google Test; primary regression suite |
| `hipblas-bench` | Performance measurement and optional correctness checks; CLI and YAML driven |
| `hipblas-example-*` | Sample programs under `clients/samples/` (`BUILD_CLIENTS_SAMPLES`). Usage examples, outside the gtest suite |

`hipblas-test` and `hipblas-bench` share `clients/common/` and the `testing_*.hpp` harnesses. Both compare GPU results with an ILP64 host reference (AOCL or OpenBLAS64, plus LAPACK for solver routines). Limit host thread count with `OMP_NUM_THREADS` so the reference library does not oversubscribe the machine.

A static hipBLAS build loads rocBLAS code objects at runtime. If they are not in the standard ROCm location or next to the executable, set `ROCBLAS_TENSILE_LIBPATH` to the `rocblas/library` directory (`clients/hipblas_clients_readme.txt`).

## Architecture overview

```
YAML test definitions (*_gtest.yaml, hipblas_smoke.yaml)
        │
        ▼
hipblas_gentest.py  ──►  hipblas_gtest.data  (generated argument records)
        │
        ▼
hipblas-test  ◄──  *_gtest.cpp  (Google Test registration, type dispatch)
        │
        ▼
testing_*.hpp  (per-routine harness: setup, API call, verify)
        │
        ├── hipBLAS API (C / Fortran / ILP64, host and device pointer mode)
        │     └── rocBLAS or cuBLAS, plus rocSOLVER when solvers are enabled
        └── Host reference (CBLAS / LAPACK, ILP64)
```

**Data-driven core.** Parameter combinations live in YAML. At build time `clients/common/hipblas_gentest.py` reads `clients/gtest/hipblas_gtest.yaml` (which includes every suite and `clients/include/hipblas_common.yaml`) and writes `hipblas_gtest.data` next to the executable. C++ supplies typed harnesses; YAML supplies sizes, precisions, transposes, batching, API form, backend, and category.

**Runtime YAML.** `hipblas-test --yaml <file>` and `hipblas-bench --yaml <file>` shell out to the staged `hipblas_gentest.py`, using `hipblas_template.yaml` as the schema, and load the resulting temporary data file. `hipblas_smoke.yaml` is this path: it is copied to staging and is not folded into `hipblas_gtest.data`. `--data <file>` loads an already-generated data file. Only one of `--yaml` and `--data` may be set. With neither, the executable loads `hipblas_gtest.data` beside itself. Parsing that file can take a couple of minutes before the first gtest line appears.

**Shared infrastructure.** `clients/include/` holds the harness headers, `Arguments`, memory helpers, initialization, and `unit_check_general` / `norm_check_general`. `clients/common/` is linked into both `hipblas-test` and `hipblas-bench`. The bench binary is compiled with `HIPBLAS_BENCH`; the test binary with `GOOGLE_TEST`. The same `testing_*.hpp` functions serve both: `arg.timing` turns on the bench timing loop, `arg.unit_check` / `arg.norm_check` turn on comparison.

## Directory layout

| Path | Purpose |
|------|---------|
| `clients/gtest/` | Google Test entry points (`*_gtest.cpp`), suite YAML (`*_gtest.yaml`), `hipblas_gtest.yaml`, `test_categories.yaml`, `hipblas_test.cpp` |
| `clients/gtest/blas1/`, `blas2/`, `blas3/`, `blas_ex/`, `solver/`, `auxil/` | One `*_gtest.cpp` / `*_gtest.yaml` pair per family |
| `clients/include/` | Harness headers in the same level split, plus `hipblas_test.hpp`, `type_dispatch.hpp`, `testing_macros.hpp`, `hipblas_common.yaml`, `hipblas_smoke.yaml`, `hipblas_template.yaml` |
| `clients/common/` | `Arguments`, `hipblas_gentest.py`, argument parsing, CBLAS interface, norms, host allocation |
| `clients/benchmarks/` | `hipblas-bench` |
| `clients/samples/` | `hipblas-example-*` programs |
| `rtest.py` / `rtest.xml` | Source-tree orchestration (`python3 rtest.py -t psdb` from `projects/hipblas`) |
| `clients/gtest/test_categories.yaml` | CTest category definitions (labels, timeouts, gtest filters) |
| `../../shared/ctest/` | Shared monorepo CTest framework. Used when this directory is present at configure time |
| `scripts/performance/` | `hipblas-bench` sweep drivers and multiplot YAML/CSV under `scripts/performance/multiplot/` |
| `test/therock/test_hipblas.py` | TheRock PR/nightly invocation of the installed `hipblas-test` |

Harness headers follow the BLAS level and the calling convention: non-batched, `_batched`, and `_strided_batched`. C versus Fortran and 32-bit versus ILP64 are selected inside those templates from `Arguments::api` (`C`, `C_64`, `FORTRAN`, `FORTRAN_64`). `DAPI_CHECK` / `DAPI_EXPECT` in `testing_macros.hpp` call the `_64` entry point when `arg.api` has the 64-bit bit set. The harness picks the Fortran wrapper when `arg.api` is `FORTRAN` or `FORTRAN_64`.

Solver harnesses (`clients/include/solver/`) and their gtest sources are compiled only when `BUILD_WITH_SOLVER=ON` (the default). `rmake.py -n` turns that off.

## Test patterns

### Harness function (`testing_*.hpp`)

Each routine has a templated function taking `const Arguments&`. Typical flow, as in `clients/include/blas1/testing_axpy.hpp`:

1. Reject or quick-return invalid sizes (`HIPBLAS_STATUS_SUCCESS` on a zero-size call, `HIPBLAS_STATUS_INVALID_VALUE` where the API requires it).
2. Allocate and initialize host and device data (`hipblas_init_vector` / `hipblas_init_matrix`, `host_vector`, `device_vector`, and the batched / strided helpers).
3. Call hipBLAS in device pointer mode and again in host pointer mode (`CHECK_HIPBLAS_ERROR`, `DAPI_CHECK`).
4. Run the host reference (`ref_*` in `clients/common/cblas_interface.cpp`, or LAPACK helpers for solver routines).
5. When `arg.unit_check` is set (the YAML default is `1`), compare with `unit_check_general`. When `arg.norm_check` is set (default `0`), record `norm_check_general`.
6. When `arg.timing` is set, time `arg.iters` iterations after `arg.cold_iters` warmup and print gflops.

Bad-argument tests are separate `testing_*_bad_arg` templates. They walk host and device pointer mode and use `DAPI_EXPECT` for `HIPBLAS_STATUS_NOT_INITIALIZED`, `HIPBLAS_STATUS_INVALID_VALUE`, and `HIPBLAS_STATUS_INVALID_ENUM`. Extra null-pointer checks run only when `arg.bad_arg_all` is true. Several NVIDIA entries set `bad_arg_all: false` because cuBLAS does not reject the same pointers.

`CHECK_DEVICE_ALLOCATION` turns `hipErrorOutOfMemory` into `GTEST_SKIP()`. A `std::bad_alloc` from the host side is also a skip.

### Google Test glue (`*_gtest.cpp`)

Each family follows the same pattern (see `clients/gtest/blas1/axpy_gtest.cpp`):

- **`HipBLAS_Test<>` (CRTP)** in `clients/include/hipblas_test.hpp` — parameterized on `Arguments`. `PrintToStringParamName` prefixes the YAML `category` onto `name_suffix()`, so `--gtest_filter` can select `*quick*axpy*` or `*pre_checkin*gemm*`.
- **Filter template** — partial specializations on supported type combinations derive from `hipblas_test_valid`. Anything else derives from `hipblas_test_invalid` and is dropped at instantiation.
- **`type_dispatch.hpp`** — `hipblas_simple_dispatch`, `hipblas_blas1_dispatch`, and the level-specific dispatchers map `Arguments` datatypes onto those templates.
- **`type_filter()` / `function_filter()`** — `type_filter` calls the dispatch helper. `function_filter` keeps the suite on its function name and the matching `*_bad_arg` name.
- **`TEST_P` + `INSTANTIATE_TEST_CATEGORIES`** — one instantiation. The macro registers a single suite prefix `_` and relies on the category string inside the parameter name. Per-category `INSTANTIATE_TEST_SUITE_P` calls are commented out in `hipblas_test.hpp` for that reason.
- **`CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES`** inside `TEST_P` — runs the harness once on the current device, with the per-test alarm from `HIPBLAS_TEST_TIMEOUT`.

There is no threads / streams / multi-GPU dispatch macro. `Arguments` has no `threads`, `streams`, `devices`, or `HMM` fields. Every suite's `TEST_P` uses `CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES`. Device containers accept an HMM flag in their constructors, and current harnesses leave that flag at its default (ordinary `hipMalloc`).

**YAML global filters (`os_flags`, `backend_flags`, `gpu_arch`).** These are applied when the gtest suite instantiates, in `hipblas_client_global_filters()` (`clients/gtest/hipblas_test.cpp`). `hipblas_gentest.py` stores them on `Arguments`; it does not drop cases.

For the fields to take effect, `type_filter()` must call `hipblas_client_global_filters()` before accepting the type combination. The standard `*_gtest.cpp` functor does that, then checks the FILTER template. A `type_filter()` that returns true unconditionally keeps every case, including ones whose `backend_flags` or `gpu_arch` do not match this build.

| Field | Meaning |
|-------|---------|
| `os_flags` | Bitmask: `LINUX` (1), `WINDOWS` (2), `ALL_OS` (3). Default `ALL_OS`. A case is kept when its mask overlaps the host OS. |
| `backend_flags` | Bitmask: `AMD` (1), `NVIDIA` (2), `ALL_BACKEND` (3). Default `ALL_BACKEND`. The AMD or NVIDIA bit is fixed at compile time (`USE_CUDA` defines `__HIP_PLATFORM_NVCC__`). Half-precision and many batched Level-1 cases set `backend_flags: AMD`. Bad-argument cases often have a separate NVIDIA entry. |
| `gpu_arch` | Up to four characters matched against the GPU name after the `gfx` prefix. `?` matches one character. Empty (the default) allows every architecture. A non-empty value is an allowlist: `90a` matches `gfx90a`; `942` matches `gfx942` and does not match `gfx90a` or `gfx1201`. |
| `api` | `C`, `C_64`, `FORTRAN`, `FORTRAN_64`. On a CUDA build with cuBLAS older than 12, `C_64` and `FORTRAN_64` are filtered out. |

`hipblas_test.cpp` lists the category strings `quick`, `pre_checkin`, `nightly`, `multi_gpu`, `HMM`, and `known_bug`. The helper that consults that list has no callers. Live suite YAML uses `quick` and `pre_checkin`.

### YAML categories

Each test entry sets `category`. The string is copied into the gtest parameter name.

| Category | Where it appears | Typical use |
|----------|------------------|-------------|
| `quick` | Active suite YAML | Small numerical cases: a few sizes, transposes, precisions, and API forms |
| `pre_checkin` | Active suite YAML | Bad-argument cases, and some extra numerical coverage |
| `nightly` | Default in `hipblas_common.yaml` when an entry omits `category` | Bench template default. Suite files override it |
| `known_bug` | Assigned by `hipblas_gentest.py` | Quarantine. Normal filters exclude `*known_bug*` |
| `stress` | Commented-out Level-1 entries (large ILP64 sizes, often `os_flags: [LINUX]` and `gpu_arch: '90a'`) | Not generated into `hipblas_gtest.data` while those lines stay comments |

`hipblas_gentest.py` rewrites a case to `category: known_bug` when the YAML document has a `Known bugs` list and every specified field matches (`function` is a glob). `clients/include/hipblas_smoke.yaml` contains a commented `#include: known_bugs.yaml`. No `known_bugs.yaml` is checked in, so nothing is quarantined that way today. Filters still exclude `*known_bug*` so a future list takes effect without changing the runners.

Suite files `include: hipblas_common.yaml`. `clients/gtest/hipblas_gtest.yaml` includes every suite, including the solver suites. Solver cases are only instantiated when the matching `*_gtest.cpp` was compiled (`BUILD_WITH_SOLVER=ON`).

`hipblas_smoke.yaml` is a separate, smaller matrix (one modest size per Level-1, Level-2, Level-3, extension, and auxiliary function). It is invoked with `--yaml`, which regenerates data at runtime.

### Verification macros

| Macro / helper | Use |
|----------------|-----|
| `CHECK_HIP_ERROR` | HIP API success |
| `CHECK_HIPBLAS_ERROR` | hipBLAS success (`EXPECT_HIPBLAS_STATUS` ..., `HIPBLAS_STATUS_SUCCESS`) |
| `EXPECT_HIPBLAS_STATUS` / `DAPI_EXPECT` | Expected status, including the ILP64 entry point |
| `DAPI_CHECK` | Success check on the 32-bit or `_64` entry point |
| `CHECK_DEVICE_ALLOCATION` | Skip the test on `hipErrorOutOfMemory` |
| `unit_check_general` / `unit_check_mixed` | Element-wise comparison with the host reference when `arg.unit_check` is set |
| `norm_check_general` | Relative error when `arg.norm_check` is set |
| `CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES` | Per-test alarm; SIGSEGV and C++ exceptions become gtest failures |

## Running tests

### Direct `hipblas-test`

From a `projects/hipblas` build:

```bash
./build/release/clients/staging/hipblas-test --gtest_filter=*quick*axpy*
./build/release/clients/staging/hipblas-test --gtest_filter=*pre_checkin*gemm*-*known_bug*
./build/release/clients/staging/hipblas-test --yaml hipblas_smoke.yaml
```

Filter syntax: `--gtest_filter=POSITIVE[-NEGATIVE]`. The category word (`quick`, `pre_checkin`) is part of the parameter name, so `*quick*` selects the numerical tier and `*pre_checkin*` selects the pre-checkin tier.

Pass lines are hidden unless `GTEST_LISTENER=PASS_LINE_IN_LOG`.

### CTest (`shared/ctest`)

hipBLAS registers labeled CTest suites at configure time when both of these exist:

- `clients/gtest/test_categories.yaml`
- `${ROCM_LIBRARIES_ROOT}/shared/ctest/TestCategories.cmake`

A sparse checkout without `shared/ctest` skips categorization (CMake prints `hipBLAS: shared/ctest not found`). There is no `HIPBLAS_ENABLE_CTEST` option; presence of the shared module is the switch. The call to `apply_test_category_labels()` does not pass `USE_RTEST_DRIVER`, so each CTest entry runs `hipblas-test` with a gtest filter. It does not call `rtest.py`.

```
clients/gtest/test_categories.yaml
        │
        ▼
apply_test_category_labels()     ← shared/ctest/TestCategories.cmake
        │
        ▼
parse_test_categories.py
        │
        ▼
build/release/clients/.../test_categories.cmake
        │
        ▼
ctest -L <label>
```

**Category suites** (`clients/gtest/test_categories.yaml`):

| CTest label | Also labeled | gtest patterns | Timeout |
|-------------|--------------|----------------|---------|
| `quick` | `smoke` | `*pre_checkin*` excluding `*known_bug*` | 900 s |
| `standard` | `pr`, `precheckin` | `*quick*` and `*pre_checkin*` excluding `*known_bug*` | 3600 s |
| `comprehensive` | `nightly`, `extended` | same patterns as `standard` | 3600 s |
| `full` | `stress`, `weekly` | same patterns as `standard` | 7200 s |
| `ffm-quick` | `ffm-quick` | `*blas2*pre_checkin*` excluding `*batched*` and `*known_bug*` | 7200 s |
| `ffm-full` | `ffm-full` | `*blas1*pre_checkin*` excluding `*strided*` and `*known_bug*` | 7200 s |

The CTest label `quick` selects the YAML category `pre_checkin`. The YAML category `quick` (the numerical cases) is selected by the CTest labels `standard`, `comprehensive`, and `full`. Those three labels currently share one filter. `test_categories.yaml` notes that `full` is standing in until TheRock grows a real stress run. The commented Level-1 `category: stress` blocks are not part of any of these filters.

Generated test names follow the shared framework: `hipblas-test_<category>_suite` (for example `hipblas-test_quick_suite`).

```bash
cd projects/hipblas/build/release
ctest -N -L quick
ctest -L standard -V
```

**Install-tree CTest.** When categories are enabled, CMake installs `CTestTestfile.cmake` to `${CMAKE_INSTALL_BINDIR}/hipblas` (typically `bin/hipblas/`) with paths relative to the installed `hipblas-test`. Also installed beside the binary: `hipblas_gtest.data`, `hipblas_common.yaml`, `hipblas_template.yaml`, `hipblas_smoke.yaml`, and `hipblas_gentest.py`.

```bash
cd /opt/rocm/bin/hipblas
ctest -L standard -N
ctest -L standard
```

`exclude_gpu` from `shared/ctest` is unused. Per-case GPU, OS, and backend scoping is the YAML `gpu_arch` / `os_flags` / `backend_flags` path described above.

Full framework documentation: [`shared/ctest/README.md`](../../shared/ctest/README.md).

### Orchestrated runs (`rtest.py`)

`rtest.py` and `rtest.xml` live at the hipBLAS project root. Run them from `projects/hipblas` (the script opens `./rtest.xml` and, unless that file sits in a directory named `staging`, points at `build/release/clients/staging`):

```bash
cd projects/hipblas
python3 rtest.py -t psdb
python3 rtest.py -t osdb
python3 rtest.py -t psdb -g          # debug build tree
python3 rtest.py -t psdb --install_dir build
```

| `-t` set | What `rtest.xml` runs |
|----------|------------------------|
| `psdb` | `hipblas-test --gtest_filter=*-*known_bug*` |
| `osdb` | the same filter |

Both sets run every instantiated test except `known_bug`. The XML comment records that the suite is small enough to run in full for pre-checkin and for the extended set. There are no `smoke` or `ctest_*` sets in this file. A `<run>` may set `vram_min` or `time_max`; the current file sets neither.

On Windows, `CREATE_TEST_APP_LOCAL_DEPLOY` copies `rtest.py` and `rtest.xml` into the staging directory next to `hipblas-test`. The Linux client install does not install `rtest.py`.

### Benchmarking

```bash
./build/release/clients/staging/hipblas-bench -f gemm -r f32_r \
  --transposeA N --transposeB N -m 4096 -n 4096 -k 4096 \
  --alpha 1 --lda 4096 --ldb 4096 --beta 0 --ldc 4096
./build/release/clients/staging/hipblas-bench -v 1 -f gemm -r f32_r -m 128 -n 128 -k 128
./build/release/clients/staging/hipblas-bench --yaml <file>.yaml
```

`-v 1` enables the correctness check inside the bench harness. On the AMD backend, `ROCBLAS_LAYER=2` prints the underlying rocBLAS invocation so a command line can be copied and edited; turn logging off before collecting timings.

Sweep YAML and plotting scripts live under `scripts/performance/multiplot/`. `scripts/performance/multiplot/README` describes `benchmark_plot.sh` for Level-1, Level-2, and Level-3 sweeps and for plots against a theoretical memory-bandwidth ceiling on a few GFX names.

## Useful environment variables

| Variable | Effect |
|----------|--------|
| `HIPBLAS_TEST_TIMEOUT` | Per-test alarm in seconds. Default 600. `0` disables the alarm. Unix only |
| `HIPBLAS_TEST_NO_SIGACTION` | Any non-empty value skips installation of the signal handlers |
| `HIPBLAS_TEST_CONTINUE_ON_FATAL_SIGNAL` | After a caught fatal signal, fail the test and continue. Unset, the process aborts |
| `HIPBLAS_CLIENT_RAM_GB_LIMIT` | Cap host allocation. Cases that would exceed it skip |
| `HIPBLAS_CLIENT_NO_ALLOC_CHECK` | Skip the host-memory pre-check |
| `HIPBLAS_CLIENT_ALLOC_AVAILABLE` | Use `MemAvailable` when too many cases skip on the default memory reading |
| `HIPBLAS_CLIENT_ALLOC_FILL_HEX_BYTE` | Fill pattern for host allocations |
| `GTEST_LISTENER=PASS_LINE_IN_LOG` | Print each passing test. The default listener hides pass lines |
| `OMP_NUM_THREADS` | Host reference thread count |
| `ROCBLAS_LAYER` | On the AMD backend, rocBLAS logging. `ROCBLAS_LAYER=2` prints a bench-style command |
| `ROCBLAS_TENSILE_LIBPATH` | Directory of rocBLAS Tensile code objects when a static hipBLAS build cannot find them |
| `HSA_XNACK` | Set to `1` by `test/therock/test_hipblas.py` when `ARTIFACT_GROUP` contains `asan` |
| `TEST_TYPE` | Read by the TheRock runner: `quick` selects the smoke YAML; any other value selects `*pre_checkin*` |
| `SHARD_INDEX`, `TOTAL_SHARDS` | TheRock gtest sharding (`GTEST_SHARD_INDEX` is `SHARD_INDEX - 1`) |

## CI integration

- **TheRock** invokes the installed binary through [`test/therock/test_hipblas.py`](../../test/therock/test_hipblas.py). `TEST_TYPE=quick` runs `hipblas-test --yaml $THEROCK_BIN_DIR/hipblas_smoke.yaml` with `-*known_bug*`. The default `TEST_TYPE=full` runs `--gtest_filter=*pre_checkin*-*known_bug*`. A comment in that file (`TODO(#2616)`) records that the broader suite stays off until machine issues are resolved. The script sets `HSA_XNACK=1` for ASAN artifacts and honors gtest sharding.
- **CTest labels** in `test_categories.yaml` line up with Jenkins job names (smoke, precheckin, extended, weekly) by label text. `comprehensive` and `full` use the same gtest patterns as `standard`.
- **`rtest.py -t psdb` and `-t osdb`** run every non-`known_bug` test. Math CI job definitions that call `rtest.py` are not in this repository.
- **Monorepo checkout:** include `shared/ctest` (or use a full clone) when you want `ctest -L` and the install-tree `CTestTestfile.cmake`.

## Adding tests

1. Add `clients/include/<level>/testing_<fn>.hpp` (and `_batched` / `_strided_batched` variants) with the numerical function and, where the API has status checks worth locking, `testing_<fn>_bad_arg`.
2. Add `clients/gtest/<level>/<fn>_gtest.cpp`. Derive from `HipBLAS_Test<>`, call `hipblas_client_global_filters()` inside `type_filter_functor`, dispatch with the matching `hipblas_*_dispatch` helper, and finish with `INSTANTIATE_TEST_CATEGORIES`. Wrap the body of `TEST_P` in `CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES`.
3. Add `clients/gtest/<level>/<fn>_gtest.yaml`. `include: hipblas_common.yaml`. Give numerical cases `category: quick` and bad-argument cases `category: pre_checkin`. Set `backend_flags` when a precision or a status check exists on only one backend. Set `api` to the forms you intend to lock (`C`, `FORTRAN`, `C_64`, `FORTRAN_64`).
4. `include:` the YAML from `clients/gtest/hipblas_gtest.yaml`.
5. Add the `.cpp` to `hipblas_test_source` and the YAML to the matching `HIPBLAS_*_YAML_DATA` list in `clients/gtest/CMakeLists.txt`, so `hipblas_gtest.data` rebuilds. Solver files go in the `BUILD_WITH_SOLVER` source and YAML lists.

There is no `clients/gtest/README.md` in this project. The steps above are the registration path.

---

## Testing Strategy and Layers

### Unit Testing Strategy

**Purpose.** Lock argument validation, status codes, and small auxiliary behavior (handle modes, set/get matrix and vector).

**Framework and location.**

| Item | Detail |
| --- | --- |
| Framework | Google Test, inside the `hipblas-test` binary |
| Location | `testing_*_bad_arg` in `clients/include/`, registered from `clients/gtest/*_gtest.cpp`, with cases in `*_gtest.yaml` (`function: <name>_bad_arg`, usually `category: pre_checkin`) |
| How to run | `./hipblas-test --gtest_filter=*pre_checkin*axpy_bad_arg*` |
| Naming | Parameter names start with the YAML category, then the function and its precision and sizes |

These cases still create a hipBLAS handle and, for pointer checks, device allocations. `hipblas-test` requires a GPU even for bad-argument tests.

**What the bad-argument templates cover.** Null handle, null pointers (when `bad_arg_all` is set and pointer mode is host), invalid enums, and the quick-return success path for a zero length. They do not cover numerical results.

**Coverage expectation.** There is no enforced host-side coverage floor. New status-code behavior needs a `testing_*_bad_arg` case, or an auxiliary gtest, that fails without the change. Gaps belong in [Known Risks and Gaps](#known-risks-and-gaps).

### Integration Testing Strategy

**Purpose.** Show that each public routine, on a real device, matches the host reference across precision, transpose, batching, pointer mode, and the C / Fortran / ILP64 entry points. This is the primary signal for hipBLAS, because the library's job is to forward those calls correctly.

**What is covered.** `hipblas-test` driven by `*_gtest.yaml` through `hipblas_gentest.py` into `hipblas_gtest.data`. Levels 1–3, extensions (`blas_ex`), grouped-batched GEMM, auxiliary set/get and mode APIs, and, when `BUILD_WITH_SOLVER=ON`, `getrf` / `getrf_npvt`, `getrs`, `getri`, `geqrf`, and `gels` (including batched and strided-batched forms where the YAML lists them). Results are compared when `unit_check` is on, which is the default in `hipblas_common.yaml`.

**Tiers.** The YAML category and the CTest label are different knobs. Read both columns.

| Knob | What it selects | Duration (order of magnitude) |
| --- | --- | --- |
| YAML `quick` | Small numerical cases in `*_gtest.yaml` | Minutes for one family; longer for `*quick*` across the binary |
| YAML `pre_checkin` | Bad-argument cases and extra numerical cases | Same order as `quick` for a single family |
| `--yaml hipblas_smoke.yaml` | One modest problem per function, all levels | Minutes |
| CTest `quick` / TheRock default `TEST_TYPE` | `*pre_checkin*` | CTest timeout 15 minutes. TheRock runs this filter as its default "full" job |
| CTest `standard` | `*quick*` and `*pre_checkin*` | CTest timeout 1 hour |
| CTest `comprehensive`, CTest `full` | Same gtest patterns as `standard` | Timeouts 1 hour and 2 hours |
| `rtest.py -t psdb` or `-t osdb` | Every instantiated test except `*known_bug*` | The full generated suite |
| `known_bug` | Quarantine category | Excluded by the filters above |

**What requires GPU hardware.** The whole client suite. Startup fails when no device is visible.

**What runs on PRs.** TheRock's default invocation is the `*pre_checkin*` filter (see `test/therock/test_hipblas.py`). `TEST_TYPE=quick` switches that job to `hipblas_smoke.yaml`. The YAML `quick` numerical tier is what `ctest -L standard` and `rtest.py -t psdb` run locally. HMM and multi-GPU are not parameters of this suite.

### Performance and Benchmarking Testing

**Purpose.** Measure marshalling-plus-backend throughput for a routine, and give a place to compare backends or releases by hand.

| Item | Detail |
| --- | --- |
| Tool | `hipblas-bench` (CLI and `--yaml`). Sweeps also via `scripts/performance/multiplot/` |
| Metrics | Gflops and microseconds from the harness timing loop (`arg.cold_iters`, `arg.iters`) |
| Correctness alongside timing | `-v 1` or `unit_check` / `norm_check` in the YAML |
| Baseline | Per machine. Not stored as a merge gate in this repository |
| Regression threshold | No automated threshold in hipBLAS CI |
| Gating | Manual, from bench output attached to a performance-sensitive PR |

hipBLAS performance is mostly the backend's performance. A regression that shows up only in `hipblas-bench` still needs a rocBLAS or cuBLAS investigation when the marshalling change is not the suspect.

### Build-Time Validation

hipBLAS does not generate kernels. A successful library build checks that the selected backend (`amd_detail` or `nvidia_detail`) compiles and links against rocBLAS or cuBLAS, and against rocSOLVER when solvers are on. Client builds additionally require an ILP64 CBLAS and LAPACK.

`BUILD_CODE_COVERAGE=ON` adds a `coverage` target. `make coverage GTEST_FILTER=<filter>` runs `hipblas-test` under LLVM profile instrumentation and writes an HTML report (`CMakeLists.txt` at the project root). That target is a local tool. It is not a documented merge gate.

Invalid YAML fails the `hipblas-test-data` custom command, because `hipblas_gentest.py` exits on a missing key or a bad type. That is the build-time check for new suite entries.

## Pre-submit / CI Gates

hipBLAS is exercised by TheRock (via `test/therock/test_hipblas.py`) and by Math CI (Jenkins). Job definitions for Jenkins are not in this repository. Exact required GitHub checks depend on branch protection.

### Validation Gates and Ownership

| Validation area | Required before merge | Owner | Notes |
| --- | --- | --- | --- |
| Library build (Linux, AMD backend) | Yes | CI / component team | rocBLAS, and rocSOLVER when solvers are on |
| `hipblas-test` on GPU | Yes, in the TheRock lane | Component team | Default lane is `*pre_checkin*`. Smoke YAML when `TEST_TYPE=quick` |
| CTest categories | When `shared/ctest` is in the checkout | Component team | `ctest -L standard` is the local match for numerical plus bad-argument coverage |
| Formatting / pre-commit | Yes | CI | Repo-root [`.pre-commit-config.yaml`](../../.pre-commit-config.yaml) |
| Code coverage floor | No | — | `BUILD_CODE_COVERAGE` is optional |
| Address sanitizer | Lane-dependent | Component team | `BUILD_ADDRESS_SANITIZER`. TheRock sets `HSA_XNACK=1` when the artifact group name contains `asan` |
| CUDA backend (`USE_CUDA`) | Separate packaging (`hipblas-alt`) | Component team | `backend_flags: NVIDIA` cases. ILP64 entry points require cuBLAS 12 or newer |
| Performance benchmarks | No | Component team | `hipblas-bench` is manual |
| Solver routines | When the PR touches them | Component team | Need `BUILD_WITH_SOLVER=ON` and rocSOLVER. Several solver YAML entries set `backend_flags: AMD` |

### PR Test Classification

| Status | Applies to |
| --- | --- |
| **Trusted gate** | AMD library build; TheRock `*pre_checkin*` (or smoke YAML when `TEST_TYPE=quick`); formatting |
| **Local / broader gate** | `ctest -L standard` and `rtest.py -t psdb` (YAML `quick` and `pre_checkin` together). Run these before pushing a numerical change, because the default TheRock filter does not include YAML `quick` |
| **Unstable** | Quarantine by giving the case `category: known_bug`, or by a `Known bugs` list that `hipblas_gentest.py` rewrites to that category, with a tracking ticket and a plan to remove it |

### TheRock `TEST_TYPE` and CTest labels

These are the in-tree controls. This repository does not define GitHub labels that widen the hipBLAS lane.

| Goal | Control |
| --- | --- |
| Smoke matrix on TheRock | `TEST_TYPE=quick` in the environment of `test/therock/test_hipblas.py` |
| Bad-argument tier on TheRock | Default `TEST_TYPE` (the script treats every value other than `quick` as `*pre_checkin*`) |
| Numerical `quick` plus `pre_checkin` locally | `ctest -L standard` or `python3 rtest.py -t psdb` |
| Install-tree check | `ctest -L <label>` from `bin/hipblas/` |

Widening the default TheRock job to the numerical tier means changing `test/therock/test_hipblas.py` (the `#2616` TODO) or `test_categories.yaml`. A PR that only adds YAML `category: quick` cases does not make that job run them.

### Flaky Test Policy

A flaky test is not a permanent gate. Quarantine it as `category: known_bug` (directly, or through a `Known bugs` entry the generator matches) only with a tracking ticket and a plan to delete the quarantine. Prefer a fix or a narrower `backend_flags` / `gpu_arch` allowlist over a blanket exclusion.

### Known Bugs and Expected Failures

`hipblas_gentest.py` supports a `Known bugs` list at the top of a YAML document. A case whose listed fields all match is rewritten to `category: known_bug` unless it is already `known_bug` or `disabled`. `function` matches as a glob. `known_bug_platforms` can record a platform set instead of rewriting the category; that string is stored on the test and the current runners key off the category name.

No suite currently ships a `Known bugs` list, and `known_bugs.yaml` is not in the tree. The exclude patterns are already in `rtest.xml`, `test_categories.yaml`, and `test/therock/test_hipblas.py`, so adding the file and including it from the suite YAML is enough to make those runners skip the new quarantines.

## Sanitizer Coverage

| Sanitizer | Build flag / notes | CI |
| --- | --- | --- |
| ASAN | `BUILD_ADDRESS_SANITIZER` in the top-level `CMakeLists.txt` (`-fsanitize=address -shared-libasan`). `hipblas-test` adds `-mllvm -asan-globals=0` so gtest string literals in the test TUs do not emit misaligned `asan_globals` metadata. `libhipblas` keeps global instrumentation | TheRock ASAN artifacts set `HSA_XNACK=1` from `test_hipblas.py`. Not documented here as a universal PR gate |
| TSAN / UBSAN | No hipBLAS CI lane in this tree | — |

GPU-side sanitizer behavior is limited by the ROCm stack. Host marshalling and the client benefit most from an ASAN build of `libhipblas`.

## Static Analysis

`clang-format` is the documented formatter (see [`docs/conceptual/library-source-code-organization.rst`](docs/conceptual/library-source-code-organization.rst)). The monorepo pre-commit config runs the hooks that apply to touched files. There is no documented clang-tidy merge gate for hipBLAS in this file. Run whatever extra static check your change needs locally, and treat a CI job as required only when that job is attached to the PR.

## Why We Test This Way

hipBLAS is a thin, multi-backend API. One data-driven binary can instance the same harness for every precision, transpose, batch form, pointer mode, and language binding, and can drop cases that do not exist on the backend being built. The host reference is what makes a marshalling bug visible: a swapped leading dimension or a missed Fortran entry point disagrees with CBLAS even when the backend itself is correct. CTest (`shared/ctest`) and the install-tree `CTestTestfile.cmake` are how monorepo automation and packaged builds select a tier. The TheRock script is intentionally narrower than `rtest.py -t psdb`; that split is recorded above so a green PR is not read as "every YAML `quick` case ran."

## Key Quality Concerns

1. **Marshalling correctness** — Wrong sizes, strides, pointer mode, or entry point silently produce wrong BLAS results downstream. Validated by `unit_check_general` against the host reference in `hipblas-test`.
2. **API compatibility across backends** — The same call must be meaningful on rocBLAS and on cuBLAS, with explicit `backend_flags` where the backends differ (half precision, batched Level-1, bad-argument strictness, solvers). Validated by building both backends and by the filter in `hipblas_client_global_filters()`.
3. **Fortran and ILP64** — `api: [C, FORTRAN, C_64, FORTRAN_64]` on numerical cases, `DAPI_*` in the harness, and `BUILD_FORTRAN_CLIENTS` (default ON except on Windows). cuBLAS older than 12 does not instantiate the `_64` APIs.
4. **Solver surface** — `getrf`, `getrs`, `getri`, `geqrf`, `gels` depend on rocSOLVER and on `BUILD_WITH_SOLVER`. Validated only in AMD client builds that include those sources.
5. **Packaging** — Installed `hipblas-test`, `hipblas_gtest.data`, the smoke/common/template YAML, `hipblas_gentest.py`, and `bin/hipblas/CTestTestfile.cmake` must stay consistent. A static build must still find rocBLAS code objects (`ROCBLAS_TENSILE_LIBPATH`).

## Release Validation

Before release sign-off, expect at least:

- `rtest.py -t psdb` (all non-`known_bug` cases) on the release GPU matrix, not only the TheRock `*pre_checkin*` filter
- A CUDA-backend run of the cases with `backend_flags: NVIDIA` or `ALL_BACKEND`, on a cuBLAS new enough for the ILP64 entry points you claim
- A `BUILD_WITH_SOLVER=ON` run covering `*getrf*:*getrs*:*getri*:*geqrf*:*gels*`
- Review of any `known_bug` cases that were added since the last release
- Bench spot-checks with `hipblas-bench` on representative sizes, interpreted together with rocBLAS or cuBLAS performance data
- QA outside this repository for the packaged ROCm layout

## Dependencies and Validation Handoffs

| Dependency | Owning team | How hipBLAS clients use it | Known gap |
| --- | --- | --- | --- |
| rocBLAS | rocBLAS | AMD backend for every numerical case | Kernel bugs look like hipBLAS failures. See [`projects/rocblas/TESTING.md`](../rocblas/TESTING.md) |
| rocSOLVER | rocSOLVER | Solver routines when `BUILD_WITH_SOLVER=ON` | Client sources and several YAML entries are AMD-only. `rmake.py -n` builds a library without them |
| cuBLAS | NVIDIA | CUDA backend (`USE_CUDA`, package name `hipblas-alt`) | ILP64 filtered out when cuBLAS major version is below 12. Bad-argument strictness differs (`bad_arg_all: false`) |
| Host reference (AOCL ILP64 or OpenBLAS64) plus LAPACK | External / TheRock | Linked into `hipblas-test` and `hipblas-bench` | `LINK_BLIS=OFF` can hit 32-bit overflow in the host reference on large sizes. The install guides say to exclude `*stress*`; those YAML entries are commented out today. Set `OMP_NUM_THREADS` |
| hipBLAS-common | hipBLAS-common | Public headers and types | Version floor is in the top-level `CMakeLists.txt` |
| TheRock / `shared/ctest` | TheRock | `test_hipblas.py`, install-tree CTest | Default TheRock filter is `*pre_checkin*`. `shared/ctest` is absent from a hipBLAS-only sparse checkout |
| Math CI (Jenkins) | DevOps | Historical precheckin / extended / weekly names, mirrored as CTest labels | Job definitions are not in this tree. `rtest.py` is the in-repo driver those jobs have used |

## Supported Configurations

| Configuration | Validation level | Frequency | Notes |
| --- | --- | --- | --- |
| Linux + AMD GPU, rocBLAS backend | Full client suite via `rtest.py` or `ctest -L standard` | Local and, for `*pre_checkin*` / smoke, TheRock | Primary development path. `BUILD_FORTRAN_CLIENTS` defaults ON |
| Linux + NVIDIA GPU, `USE_CUDA=ON` | Cases whose `backend_flags` include NVIDIA | When a CUDA build is produced (`hipblas-alt`) | `LINK_BLIS` defaults OFF. Fortran clients still follow `BUILD_FORTRAN_CLIENTS` |
| Windows | Client build. `BUILD_FORTRAN_CLIENTS` defaults OFF. `CREATE_TEST_APP_LOCAL_DEPLOY` copies runtime DLLs and `rtest.*` into staging | Release / manual | Install prefix defaults under the build tree's `package/` directory |
| Specific GFX | Cases with an empty `gpu_arch`, plus any allowlisted pattern | Every run | A non-empty `gpu_arch` drops the case on other architectures at gtest instantiation |

State unsupported combinations in the PR (solver routines on CUDA, ILP64 on old cuBLAS, half precision on NVIDIA) by `backend_flags` and by which binary you ran.

## Coverage Expectations by Change Type

| Change type | Expected validation |
| --- | --- |
| New or changed BLAS routine | `testing_*.hpp`, `*_gtest.cpp`, `*_gtest.yaml`, registration in `hipblas_gtest.yaml` and `clients/gtest/CMakeLists.txt`. Run `*quick*<function>*` and `*pre_checkin*<function>*` |
| Bug fix in marshalling or status codes | A gtest case that fails without the fix (numerical case or `testing_*_bad_arg`) |
| Fortran or ILP64 entry point | YAML `api` values covering the forms you changed, with `BUILD_FORTRAN_CLIENTS=ON` |
| Backend-specific behavior | `backend_flags: AMD` or `NVIDIA`, and a build of that backend |
| Solver routine | `BUILD_WITH_SOLVER=ON`, solver YAML, rocSOLVER available |
| CTest tier or TheRock filter | `test_categories.yaml` and/or `test/therock/test_hipblas.py`. Check `ctest -N -L <label>` |
| Packaging / install layout | Install, then `ctest -N` from `bin/hipblas/`, and confirm `hipblas_gtest.data` and `hipblas_smoke.yaml` sit next to `hipblas-test` |
| Performance-only | `hipblas-bench` numbers in the PR. No client change is required when the backend is the thing that changed |

## Known Risks and Gaps

| Gap | Regression risk | What exists today |
| --- | --- | --- |
| TheRock default runs `*pre_checkin*` and leaves YAML `quick` unrun (`TODO(#2616)` in `test/therock/test_hipblas.py`) | High for numerical marshalling bugs that bad-argument tests do not catch | Local `ctest -L standard` and `rtest.py -t psdb` |
| CTest `comprehensive` and `full` use the same patterns as `standard` | Medium. The label names suggest a wider suite | Timeouts differ. Stress YAML is commented out |
| CTest label `quick` selects YAML `pre_checkin` | Medium for someone who assumes the label matches the YAML category | This document. `ctest -L standard` is the tier that includes both |
| `ffm-quick` is Level-2 `pre_checkin` without `*batched*`; `ffm-full` is Level-1 `pre_checkin` without `*strided*` | Medium. The names do not describe a superset relationship | The patterns in `test_categories.yaml` |
| Commented `category: stress` ILP64 sizes | Medium on large-index regressions | `api: C_64` / `FORTRAN_64` on the small `quick` cases. Host overflow warning in the install guides if `LINK_BLIS` is off |
| No checked-in `known_bugs.yaml` | Low until a quarantine is needed | Generator support and `-*known_bug*` filters are already in place |
| `HMM` and `multi_gpu` category names | Low. Nothing in the live YAML sets them, and `Arguments` has no fields that would fan a case out across devices or managed memory | Single-device `CATCH_SIGNALS_AND_EXCEPTIONS_AS_FAILURES` |
| No automated performance threshold | High for performance work that is actually in hipBLAS | Manual `hipblas-bench`. Backend regressions belong to rocBLAS CI |
| Jenkins job list is not in this repo | Medium | `rtest.xml` is the in-repo definition of `psdb` and `osdb` |
| Sparse checkout without `shared/ctest` | Low | Clients still build. `ctest -L` and install-tree CTest are skipped |

## Improvement Roadmap

Near term:

1. Extend `test/therock/test_hipblas.py` so the default lane includes the YAML `quick` numerical tier (`TODO(#2616)`), and keep `TEST_TYPE=quick` as the smoke YAML.
2. Make CTest `comprehensive` and `full` select something wider than `standard`, or rename the labels so they describe the shared filter.
3. Decide whether the commented Level-1 stress / large-ILP64 entries should be regenerated under a category the runners actually select.

Medium term:

1. Add a `known_bugs.yaml` only together with tickets, and include it from the suites that need it.
2. Keep `test_categories.yaml`, `rtest.xml`, and `test/therock/test_hipblas.py` in one place when timeouts or filters change, so the three descriptions cannot drift.
3. Record the required GitHub checks next to this file once branch protection for hipBLAS is stable.
4. If managed-memory or multi-GPU coverage becomes a product requirement, add the `Arguments` fields and a `TEST_P` wrapper first, then the YAML.

## Owners and Review Cadence

Update this document when the client tiers, CTest labels, TheRock `TEST_TYPE` behavior, or quarantine policy change. Review [Known Risks and Gaps](#known-risks-and-gaps) when a post-merge numerical escape turns out to have been outside the TheRock filter.

## For New Contributors

When changing hipBLAS:

1. Read [Key Quality Concerns](#key-quality-concerns) for the area you are touching.
2. Extend the YAML and the harness using [Adding tests](#adding-tests).
3. Run `hipblas-test --gtest_filter=*quick*<your-function>*` and the matching `*pre_checkin*` filter before pushing. For a numerical change, also run `ctest -L standard` or `python3 rtest.py -t psdb` locally, because the default TheRock job does not include YAML `quick`.
4. Update this document if you change tiers, CTest patterns, or what `test/therock/test_hipblas.py` runs.

### Choosing the Right Test Type

- **Bug fix** — a case that fails before the fix, in the YAML tier that matches the bug (`quick` for a wrong result, `pre_checkin` for a status code).
- **GPU numerical behavior of a hipBLAS routine** — rows in `*_gtest.yaml` plus the `testing_*.hpp` harness, compared with `unit_check_general`.
- **Invalid arguments / status codes** — `testing_*_bad_arg` and a `*_bad_arg` YAML entry. Set `bad_arg_all` per backend.
- **Fortran or ILP64** — `api:` values on an existing numerical case, with Fortran clients enabled.
- **One backend only** — `backend_flags: AMD` or `NVIDIA`.
- **Handle mode, atomics mode, set/get matrix and vector** — the auxiliary suites under `clients/gtest/auxil/`.
- **Performance** — `hipblas-bench` on sizes that represent the path you care about. Put the numbers in the PR.
- **CTest or TheRock tier** — `test_categories.yaml`, `test/therock/test_hipblas.py`, and `ctest -N -L <label>`.

## How This Document Is Used

Living description of how hipBLAS clients are built, filtered, and gated. Prefer an accurate gap (TheRock does not run YAML `quick` today) over a tier name that sounds broader than the filter. Update it in the same PR that changes the filter.

## Further reading

- [Using hipBLAS clients](docs/how-to/using-hipblas-clients.rst)
- [Library source layout](docs/conceptual/library-source-code-organization.rst)
- [Linux install — building clients and the host reference](docs/install/Linux_Install_Guide.rst)
- [Windows install](docs/install/Windows_Install_Guide.rst)
- [Client staging readme](clients/hipblas_clients_readme.txt)
- [rocBLAS testing — numerical backend on AMD](../rocblas/TESTING.md)
- [shared/ctest README](../../shared/ctest/README.md)
- [TheRock hipBLAS test script](../../test/therock/test_hipblas.py)
- [Contributing to the ROCm libraries](../../CONTRIBUTING.md)
