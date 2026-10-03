# Validate the JIT implementation

The JIT tests check the Jit stages, the comgr code-object builder, the source
bundle reader and, through the mock backend, the internal entry points that run
JIT solutions with the GEMM APIs. The JIT headers are not installed. The tests include them from
`library/src/amd_detail`. They build the gfx950 source bundles committed in
[`data`](data/README.md), so they need neither Python nor a generator.

## Build and run from a checkout

From the repository root, with `project_build` set as in the
[JIT build instructions](../../../JIT.md#build):

```bash
cmake -S projects/hipblaslt -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_JIT_TESTING=ON -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_HOST=ON -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950
cmake --build "$project_build" --parallel
ctest --test-dir "$project_build/clients/tests/jit" -L jit-cpu --output-on-failure
ctest --test-dir "$project_build/clients/tests/jit" -L jit-gpu --output-on-failure
```

`-L jit-cpu` runs the tests that need no GPU. `-L jit-gpu` runs the tests that
need device 0, which must be a gfx950. Each test writes under
`clients/tests/jit/scratch` in the build directory, which CTest empties before
the tests run.

`HIPBLASLT_JIT_TESTING=ON` links the mock backend that
`hipblaslt-jit-mock-backend-test` and `hipblaslt-jit-api-test` replay bundles
through. The CTest tests are:

- `jit-cpu`: `jit-source-bundle`, `jit-component` and `jit-code-object`. A
  build with `HIPBLASLT_ENABLE_JIT=OFF` has `jit-source-bundle` and
  `jit-disabled`.
- `jit-gpu`: `jit-code-object-gpu`, `jit-library`, `jit-library-concurrency`
  and `jit-bundle-freshness`, which read library entries; TensileLite queries
  the current device when it reads one.
  With `HIPBLASLT_JIT_TESTING=ON` in a build for gfx950 it also has
  `jit-mock-backend`, `jit-mock-backend-library`, `jit-bundle-failures`,
  `jit-helper-failures` and `jit-api-splitk`, `jit-api-streamk`, `jit-api-amax`
  and `jit-api-alpha-zero`.

A build with `HIPBLASLT_ENABLE_YAML=ON` has no `jit-library`,
`jit-library-concurrency` or `jit-bundle-freshness` and none of the tests that
replay bundles, because the library entries are MsgPack.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-source-bundle` | The source bundle reader: relative paths, symbolic links that escape the bundle, size limits and library formats |
| `jit-component` | Jit over fake stages, without a GPU: missing components rejected, the generator's units reaching the builder, count limiting, excluded kernels, the stage of each failure, publish and load ordering, scratch lifetime, and concurrent generation |
| `jit-code-object` | comgr assembly, HIP helper compilation and linking for gfx950, build options, concurrent builds, and the status and log of each kind of failed build, without a GPU; with `--bundle`, the same for the committed split-K bundle |
| `jit-code-object-gpu` | The same code objects loaded and launched on the GPU, with their results checked |
| `jit-bundle-freshness` | Each committed bundle's layout and code-object versions against this tree, its library entry read by the host library, and its build; a manifest with another layout version must be reported stale |
| `jit-mock-backend` | The in-process mock backend replaying the `splitk` source bundle through Jit and the comgr builder: C/C++ numerics, owned scalar values, copied algorithms outliving their owners, name lookups, 65 streams, insufficient workspace, forged tokens, the wrong device, NOT_SUPPORTED for a non-GEMM request or another ProblemType, generation, build and record faults, rejected mock options, and bundle lifetime |
| `jit-mock-backend-library` | `getLibraryAlgos` publishes the mock solution into a fresh JIT solution library and returns a reserved index, which `getAlgosFromIndex` and `hipblasLtMatmul` run with checked numerics. A query for two solutions generates only for the shortfall and skips the published kernel. A second process then runs that index before any lookup, and `getLibraryAlgos` finds it there with a backend that aborts the process if it generates |
| `jit-library` | The JIT solution library: cache-key fields and compiler-environment filtering; rejected group- or other-writable, linked and non-directory roots; the stock TensileLite loader reading a published library; exact-size matching with the solution predicates still applied; deduplication, hash collisions, order, count and excluded kernels; mismatched and tampered keys ignored and left untouched; index allocation up to `INT32_MAX` and exhaustion; a publisher killed after each publication step; readers reloading after another instance publishes; and a fused GEMM and all-to-all problem rejected by lookup, publication and the ProblemType key without touching the library, even beside a plain solution of the same sizes |
| `jit-library-concurrency` | Eight processes publish shared and private entries into one library while another process looks them up: shared entries get one index, private ones unique indices with no gaps, and every reader snapshot loads |
| `jit-api-splitk`, `jit-api-streamk`, `jit-api-amax` | Public execution, copied algorithms, workspace rules, repeated calls and state retained after failed preparation, on the replayed bundle of that name |
| `jit-api-alpha-zero` | Alpha=0 with nonzero descriptor K and null A/B still computes beta*C and output-amax through both public APIs |
| `jit-helper-failures` | A missing helper source or renamed helper symbols are detected before output/workspace writes; an earlier C++ launch remains usable |
| `jit-bundle-failures` | Damaged source bundles are rejected through the public API: a foreign target, an escaping symbolic link, missing sources or main assembly, an undefined main kernel, invalid assembly or helper source (the message names the comgr log), corrupt or truncated library entries, missing helper source or symbols, and unsupported problems |
| `jit-disabled` | The JIT headers are absent from the public include tree, `hipblaslt-ext.hpp` compiles without them, and the extension API links against the disabled library |

The `GemmPointerCheck` tests in `hipblaslt-test` check that `Gemm::setProblem`
rejects a null A or B when alpha is nonzero, also with K=0, in builds with and
without JIT.

## Mock backend and Jit component tests

`hipblaslt-jit-mock-backend-test` takes one argument, a source bundle
directory; CTest passes `data/gfx950/splitk`. It creates the mock backend with
`jit::mock::createBackend` from `hipblaslt-jit-mock.hpp`, so generation runs no
generator, and checks an FP16 problem with M=256, N=128, K=512, the record
fault and rejected mock options. With `--library` after the bundle it runs the
`jit-mock-backend-library` checks instead, and starts its second process
itself. That mode refuses to run unless `HIPBLASLT_JIT_LIBRARY_PATH` is set, so
that it never publishes into the default library.
`hipblaslt-jit-component-test` takes one argument, a fresh directory that it
uses as the scratch parent; it needs no GPU. Both are built only with
`HIPBLASLT_ENABLE_JIT=ON`.

`hipblaslt-jit-api-test --replay BUNDLE` runs the public execution checks on a
solution the mock backend replays from `BUNDLE`; `--m`, `--n`, `--k`,
`--amax`, `--alpha-zero` and `--workspace-fallback` shape the problem, and
`--second-replay` names the bundle of its second solution.
`test_bundle_failures.py` and `test_helper_failures.py` take that binary, a
valid split-K source bundle and a fresh output directory; they damage copies of
the bundle and replay them.

## JIT solution library tests

`hipblaslt-jit-library-test` compiles the JIT solution library directly. It
needs a GPU, because TensileLite queries the current device when it reads a
library entry. It takes a split-K source bundle, `data/gfx950/splitk` in CTest,
whose library entry it publishes under several kernel names, and a scratch
directory for the libraries it creates; it ignores
`HIPBLASLT_JIT_LIBRARY_PATH`. Adding `--writers N --per-writer M` runs the
multi-process check instead: N writer processes each publish M entries shared
by all writers and M of their own, while one reader process looks them up.

## Code-object tests

`hipblaslt-jit-code-object-test` compiles the comgr code-object builder
directly. `--out` names a fresh results directory, and either `--target`
selects a compile-only run for that target ID or `--gpu` also loads and runs
the results on device 0, which must match the target. `--ffm` runs the GPU part
on the simulator that `HSA_MODEL_TOPOLOGY` and `HSA_MODEL_LIB` select. Simulator
runs are manual; CTest does not run `--ffm`. `--bundle` adds the checks for a
TensileLite source bundle. `--only` selects tests by name.
