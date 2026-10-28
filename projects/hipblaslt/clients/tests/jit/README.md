# Validate the JIT implementation

The JIT tests check the comgr code-object builder, the source bundle reader,
the support check, the JIT solution library and, through the replay backend,
the internal entry points that run a JIT solution with the GEMM APIs. The JIT
headers are not installed. The tests include them from
`library/src/amd_detail`. They build the gfx90a, gfx942 and gfx950 kernel
assembly committed in [`data`](data/README.md) and publish it into a JIT
solution library, so they need neither Python nor a generator. The replay
backend reads the same committed bundles in place.

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
- `jit-gpu`: `jit-publish`, `jit-loader`, `jit-launch`, `jit-library`,
  `jit-library-concurrency`, `jit-end-to-end`, `jit-end-to-end-library`,
  `jit-heuristic-off`, `jit-heuristic-fallback` and `jit-heuristic-forced`,
  when `GPU_TARGETS` include an architecture with committed assembly. CTest
  runs `jit-publish` before `jit-loader`. The library tests build and publish
  the `plain` assembly of that architecture and load no code, but TensileLite
  queries the current device when it reads a solution. Replay and the
  heuristic queries replay its `plain` bundle as `plain-pair`. A build with
  `HIPBLASLT_ENABLE_JIT=OFF` has
  `jit-heuristic-ignored` instead, which sets `HIPBLASLT_JIT=2` and requires
  that the queries still do not return JIT algorithms. A build with
  `HIPBLASLT_ENABLE_YAML=ON` has none of the MsgPack tests.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-source-bundle` | The source bundle reader: assembly, HIP sources and headers sorted by name, relative paths, symbolic links that escape the bundle, size limits, and a `library/` entry, which it rejects. The reader leaves the target, code-object version, solutions and kernel names at their defaults |
| `jit-disabled` | Built only with `HIPBLASLT_ENABLE_JIT=OFF`. The JIT headers are absent from the public include tree, `hipblaslt-ext.hpp` compiles without them, and the extension API links against the disabled library |
| `jit-builder` | Target-ID parsing, and the comgr builder without a GPU: the hand-written HIP kernel `builder_test_kernel.hip` built for each committed assembly target as an executable and as a relocatable, then each assembly file alone, from its relocatable, whose metadata note holds `custom.config`, and linked with that kernel from its source and from its relocatable. Each build returns one artifact of the kind asked for, and each executable defines its kernels and has the bundle's code-object version. A kernel name the executable does not define, a bundle without a target, a relocatable build with helpers, and a relocatable build of a relocatable fail the build |
| `jit-publish` | Reads the device's committed bundle with `readTensileSourceBundle`, which takes its solution from `custom.config`, builds it, and publishes, at each K, the solutions `supportingSolutions` keeps into the JIT solution library: `plain` for K=512, both `plain-pair` solutions for K=1024 and its any-K solution for K=256. Each solution gets its own index, from `jitIndexBase` (2^30) and below `jitIndexEnd` (0x60000000), and publishing again keeps them. The support check alone keeps solutions in order and drops those whose problem predicate rejects K or a transposed A, whose hardware predicate rejects another processor, or whose required workspace exceeds the limit. In a library of its own, with the allocator set to `jitIndexEnd - 1`, publishing at a new K takes that last index and the next new K fails as exhausted, leaving the allocator at `jitIndexEnd`; republishing keeps the last index, `jitIndexEnd` does not resolve, and an allocator above `jitIndexEnd` fails publication. `isJitIndex` holds from `jitIndexBase` to `jitIndexEnd - 1` only |
| `jit-loader` | Loads that library, finds the solutions published for each K in index order, resolves the kernel from its code object, and finds nothing for a transposed A. Both `plain-pair` solutions published for K=1024 map to one entry file that holds them. Launches no kernel |
| `jit-launch` | Compiles the device's committed `plain` assembly, publishes that executable, retrieves it with `findTopSolutions` and `getSolutionByIndex`, launches the kernel through the TensileLite solution adapter on known FP16 A, B and C, and compares D with a host reference. The library is under `<root>/v1/<git revision>`, and once that directory is renamed to another revision a new library instance finds nothing. Without a GPU, it also stores the HIP helper relocatable under its kernel name: a second acquire does not compile and returns the same bytes, and the gfx950 `plain` assembly linked with it defines both kernels |
| `jit-end-to-end` | `plain` replayed as `plain-pair` and built with comgr. `getJitAlgo` publishes its first solution for K=512 and its second for K=256 and returns a heuristic result whose algorithm is that JIT library index, and each runs through `hipblasLtMatmul` and `hipblaslt_ext::Gemm` with D checked against a host reference. The first lookup is a miss and generation runs. A second lookup in the same process is a hit and generation does not run. A problem neither solution solves is not supported |
| `jit-end-to-end-library` | `getLibraryAlgos` publishes the first `plain-pair` solution for K=512 and the second for K=256 into a fresh JIT solution library, each as its own entry, and returns two reserved indices, which `getAlgosFromIndex` and `hipblasLtMatmul` run with checked numerics. Each publish is a miss and generation runs. A later query of the same problem in the same process is a hit and generation does not run |
| `jit-library` | The JIT solution library, with executables built from the committed `plain` assembly: cache-key fields and compiler-environment filtering; rejected group- or other-writable, linked and non-directory roots; the stock TensileLite loader reading a published library; exact-size matching with the solution predicates still applied; deduplication, identity reuse, order, count and excluded kernels; mismatched and tampered keys ignored and left untouched; index allocation up to `jitIndexEnd - 1` and exhaustion, which leaves the allocator at `jitIndexEnd`; allocators outside the JIT range rejected; entries and mappings edited to name 7, `jitIndexEnd` or `INT32_MAX` returning nothing; prebuilt indices counted as JIT indices only in [`jitIndexBase`, `jitIndexEnd`), so a prebuilt library with indices from 0x60000000 up leaves the JIT library enabled; a library whose publish stopped before the master, the mapping, or the allocator update; a lock released when its process exits; readers reloading after another instance publishes; and a fused GEMM and all-to-all problem rejected by lookup, publication and the ProblemType key without touching the library, even beside a plain solution of the same sizes |
| `jit-library-concurrency` | Eight processes publish shared and private entries into one library while another process looks them up: shared entries get one index, private ones unique indices with no gaps, and every reader snapshot loads |
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

`hipblaslt-jit-end-to-end-test` has two cases. Each creates the replay backend
for the device's `plain` bundle in the data directory with
`jit::replay::createBackend` from `hipblaslt-jit-replay.hpp`, so generation
runs no generator. The cases solve FP16 problems with M=256, N=128 and K=512
or 256. `getJitAlgo` and `getLibraryAlgos` publish a JIT library index. A
second lookup of the same problem in that process returns the index and does
not generate. The test reads a generation counter to tell the miss from the hit.

`hipblaslt-jit-heuristic-test` has one case per mode: `off`, `fallback`,
`forced`, and, when JIT is not built, `ignored`. The case sets `HIPBLASLT_JIT`
to 0, 1 or 2 and points `HIPBLASLT_JIT_LIBRARY_PATH` at its scratch directory.
Heuristic queries use the backend compiled into the library. A JIT result is a
solution library index in [2^30, 0x60000000). The first query of a problem is a
miss and generation runs. The C++ query of that problem is a hit and
generation does not run.

## JIT solution library tests

`hipblaslt-jit-library-test` compiles the JIT solution library directly. It
builds the `plain` assembly of the device's architecture, and copies of it
with the kernel renamed, and publishes those executables with their solutions
into a scratch directory the test creates.
It does not read `HIPBLASLT_JIT_LIBRARY_PATH`. The concurrency case runs eight
writer processes, each publishing four entries shared by all writers and four
of its own, while one reader process looks them up.
