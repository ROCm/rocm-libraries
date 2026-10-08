# Validate the JIT implementation

The JIT tests check the comgr code-object builder, the source bundle reader,
the JIT solution library, the TensileLite loader and, through the replay
backend, the internal entry points that run a JIT solution with the GEMM APIs.
The JIT headers are not installed. The tests include them from
`library/src/amd_detail`. They build the gfx90a, gfx942 and gfx950 kernel
assembly committed in [`data`](data/README.md) and publish it into a JIT
solution library, so they need neither Python nor a generator. Replay tests
stage a temporary source directory from the same descriptions. That directory
is the input `readTensileSourceBundle` already accepts.

## Build and run from a checkout

From the repository root, with `project_build` set as in the
[JIT build instructions](../../../JIT.md#build):

```bash
cmake -S projects/hipblaslt -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_HOST=ON -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950
cmake --build "$project_build" --parallel
ctest --test-dir "$project_build/clients/tests/jit" -L jit-cpu --output-on-failure
ctest --test-dir "$project_build/clients/tests/jit" -L jit-gpu --output-on-failure
```

`-L jit-cpu` runs the tests that need no GPU. `-L jit-gpu` runs the tests that
need device 0. `GPU_TARGETS` is that device's architecture, `gfx90a`, `gfx942`
or `gfx950`; the command above uses `gfx950`. The publish, load, replay,
library and heuristic tests use the committed assembly of that architecture,
the same way on each of the three. `jit-publish`, `jit-builder` and
`jit-source-bundle` empty a directory under `clients/tests/jit/scratch` in the
build tree. `jit-loader` reads the library `jit-publish` wrote. The end-to-end,
heuristic and library tests create a temporary directory. The end-to-end and
heuristic tests point `HIPBLASLT_JIT_LIBRARY_PATH` at that directory.

The CTest tests are:

- `jit-cpu`: `jit-source-bundle` and `jit-builder`. A build with
  `HIPBLASLT_ENABLE_JIT=OFF` has `jit-source-bundle` and `jit-disabled`.
- `jit-gpu`: `jit-publish`, `jit-loader`, `jit-library`,
  `jit-library-concurrency`, `jit-end-to-end`, `jit-end-to-end-library`,
  `jit-heuristic-off`, `jit-heuristic-fallback` and `jit-heuristic-forced`,
  when `GPU_TARGETS` include an architecture with committed assembly. CTest
  runs `jit-publish` before `jit-loader`. The library tests load no code, but
  TensileLite queries the current device when it reads a library entry, and
  they publish the `plain` entry of that architecture. Replay and the heuristic
  queries use its `plain-pair`. A build with `HIPBLASLT_ENABLE_JIT=OFF` has
  `jit-heuristic-ignored` instead, which sets `HIPBLASLT_JIT=2` and requires
  that the queries still do not return JIT algorithms. A build with
  `HIPBLASLT_ENABLE_YAML=ON` has none of the MsgPack tests.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-source-bundle` | The source bundle reader: assembly, HIP sources and headers sorted by name, relative paths, symbolic links that escape the bundle, size limits, and a library entry that is missing, empty or not named `library/TensileLibrary.dat` |
| `jit-builder` | The comgr builder, without a GPU, building the hand-written HIP kernel `builder_test_kernel.hip` for each committed assembly target, then each assembly file linked with that kernel into one code object; each code object defines its kernels and has the builder's code-object version, and a kernel name it does not define fails the build |
| `jit-publish` | Builds the device's committed assembly and publishes it into the JIT solution library: `plain` for K=512, and `plain-pair`'s two solutions for K=1024 and K=256. Indices are at least `2^30` |
| `jit-loader` | Loads that library, finds each published solution and the pre-generated kernel, resolves the kernel from its code object, and finds nothing for a transposed A. An entry whose solutions are not 0 to N-1 is rejected. Launches no kernel |
| `jit-end-to-end` | `plain-pair` replayed and built with comgr. `getJitAlgo` publishes its first solution for K=512 and its second for K=256 as JIT library indices, and each runs through `hipblasLtMatmul` and `hipblaslt_ext::Gemm` with D checked against a host reference. The first lookup is a miss and generation runs. A second lookup in the same process is a hit and generation does not run. A problem neither solution solves is not supported |
| `jit-end-to-end-library` | `getLibraryAlgos` publishes the first `plain-pair` solution for K=512 and the second for K=256 into a fresh JIT solution library, each as its own entry, and returns two reserved indices, which `getAlgosFromIndex` and `hipblasLtMatmul` run with checked numerics. Each publish is a miss and generation runs. A later query of the same problem in the same process is a hit and generation does not run |
| `jit-library` | The JIT solution library: cache-key fields and compiler-environment filtering; rejected group- or other-writable, linked and non-directory roots; the stock TensileLite loader reading a published library; exact-size matching with the solution predicates still applied; deduplication, identity reuse, order, count and excluded kernels; mismatched and tampered keys ignored and left untouched; index allocation up to `INT32_MAX` and exhaustion; a library whose publish stopped before the master, the mapping, or the allocator update; a lock released when its process exits; readers reloading after another instance publishes; and a fused GEMM and all-to-all problem rejected by lookup, publication and the ProblemType key without touching the library, even beside a plain solution of the same sizes |
| `jit-library-concurrency` | Eight processes publish shared and private entries into one library while another process looks them up: shared entries get one index, private ones unique indices with no gaps, and every reader snapshot loads |
| `jit-disabled` | The JIT headers are absent from the public include tree, `hipblaslt-ext.hpp` compiles without them, and the extension API links against the disabled library |
| `jit-heuristic-off` | `HIPBLASLT_JIT=0`. Both heuristic queries for the plain-pair FP16 GEMM return no JIT algorithm |
| `jit-heuristic-fallback` | `HIPBLASLT_JIT=1`. With no device library, every returned algorithm is JIT and the first result for K=512 matches the host. With a device library, an Equality size returns Equality algorithms, then JIT, then the others, with no repeated kernel, and an untuned size starts with JIT. Without such a library the ordering check prints `SKIP heuristic-provider-order: the build has no device library with an Equality size` |
| `jit-heuristic-forced` | `HIPBLASLT_JIT=2`. Both queries return only JIT library indices. K=512 selects the solution ending in `_K512_WGM8` and K=256 the one ending in `_WGM1`. A transposed A returns no algorithm. The first K=512 result matches the host. The first query is a miss and generation runs. The C++ query of the same problem is a hit and generation does not run |
| `jit-heuristic-ignored` | Built only with `HIPBLASLT_ENABLE_JIT=OFF`, with `HIPBLASLT_JIT=2`. The queries do not return JIT algorithms |

## Test parameters

The tests are Catch2 cases. CMake compiles in the data directory, the library
directory and the HIP kernel path, and CTest selects a case by name. The only
environment variables a test sets are the user-facing ones: `HIPBLASLT_JIT` and
`HIPBLASLT_JIT_LIBRARY_PATH`, pointed at a scratch directory the test creates.
Run `jit-publish` before `jit-loader`; the loader reads the library the
publisher wrote. Problems are FP16 with M=256, N=128 and K=512, 1024 or 256.

`hipblaslt-jit-end-to-end-test` has two cases. Each stages `plain-pair` from
the data directory into a scratch directory and creates the replay backend with
`jit::replay::createBackend` from `hipblaslt-jit-replay.hpp`, so generation
runs no generator. The cases solve FP16 problems with M=256, N=128 and K=512
or 256. `getJitAlgo` and `getLibraryAlgos` publish a JIT library index. A
second lookup of the same problem in that process returns the index and does
not generate. The test reads a generation counter to tell the miss from the hit.

`hipblaslt-jit-heuristic-test` has one case per mode: `off`, `fallback`,
`forced`, and, when JIT is not built, `ignored`. The case sets `HIPBLASLT_JIT`
to 0, 1 or 2 and points `HIPBLASLT_JIT_LIBRARY_PATH` at its scratch directory.
Heuristic queries use the backend compiled into the library. A JIT result is a
solution library index from 2^30. The first query of a problem is a miss and
generation runs. The C++ query of that problem is a hit and generation does
not run.

## JIT solution library tests

`hipblaslt-jit-library-test` compiles the JIT solution library directly. It
publishes the `plain` entry of the device's architecture under several kernel
names with stand-in code objects, into a scratch directory the test creates.
It does not read `HIPBLASLT_JIT_LIBRARY_PATH`. The concurrency case runs eight
writer processes, each publishing four entries shared by all writers and four
of its own, while one reader process looks them up.
