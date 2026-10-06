# Validate the JIT implementation

The JIT tests check the comgr code-object builder, the source bundle reader,
the TensileLite loader and, through the replay backend, the internal entry
points that run a JIT solution with the GEMM APIs. The JIT headers are not
installed. The tests include them from `library/src/amd_detail`. They build
the gfx90a, gfx942 and gfx950 kernel assembly committed in
[`data`](data/README.md), as the bundles that the `jit-bundles` test writes
with their library entries and manifests, so they need neither Python nor a
generator.

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
need device 0, which must be a gfx90a, gfx942 or gfx950; they run the bundles of
its architecture. Each test empties its own directory
under `clients/tests/jit/scratch` in the build directory before it runs.

The CTest tests are:

- `jit-cpu`: `jit-bundles`, `jit-source-bundle` and `jit-builder`. A build
  with `HIPBLASLT_ENABLE_JIT=OFF` has `jit-source-bundle` and `jit-disabled`.
  CTest runs `jit-bundles` before each test that reads a bundle.
- `jit-gpu`: `jit-loader`, `jit-library`, `jit-library-concurrency`,
  `jit-end-to-end`, `jit-end-to-end-library`, `jit-heuristic-off`,
  `jit-heuristic-fallback` and `jit-heuristic-forced`, when `GPU_TARGETS`
  include an architecture with committed bundles. A build with
  `HIPBLASLT_JIT_ENABLE_HIPKITTENS=ON` and gfx950 also has `jit-hipkittens`. The library tests load no
  code, but TensileLite queries the current device when it reads a library
  entry. A build with `HIPBLASLT_ENABLE_JIT=OFF` has `jit-heuristic-ignored`
  instead, which sets `HIPBLASLT_JIT=2` and requires that the queries still do
  not return JIT algorithms. A build with `HIPBLASLT_ENABLE_YAML=ON` has none
  of them, because the library entries are MsgPack.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-bundles` | Writes each bundle that `bundle_writer.cpp` describes under `clients/tests/jit/scratch/bundles/<architecture>` in the build directory: the committed assembly, and a library entry and manifest built from it and from the description of each solution. `plain` has one solution; `plain-pair` has two that run the plain kernel, the first for K a multiple of 512 and the second for any K |
| `jit-source-bundle` | The source bundle reader: assembly, HIP sources and headers sorted by name, relative paths, symbolic links that escape the bundle, size limits, and a library entry that is missing, empty or not named `library/TensileLibrary.dat` |
| `jit-builder` | The comgr builder, without a GPU, building the hand-written HIP kernel `builder_test_kernel.hip` for each written bundle's target, then each bundle's assembly linked with that kernel into one code object; each code object defines its kernels and has the builder's code-object version, and a kernel name it does not define fails the build |
| `jit-loader` | `plain` and `plain-pair` built with comgr for device 0 and loaded through the TensileLite loader. `plain` selects its solution for the FP16 GEMM it was generated for; `plain-pair` selects its first solution for K=512 and its second for K=256; neither selects anything for a transposed A. The loader rejects an entry with solutions 0 and 2, a solution whose kernel was not built, and a built kernel no solution names. Launches no kernel |
| `jit-end-to-end` | `plain-pair` replayed, built with comgr and loaded. `getJitAlgo` returns its first solution for K=512 and its second for K=256, and each runs through `hipblasLtMatmul` and `hipblaslt_ext::Gemm` with D checked against a host reference; a problem neither solution solves is not supported |
| `jit-end-to-end-library` | `getLibraryAlgos` publishes the first `plain-pair` solution for K=512 and the second for K=256 into a fresh JIT solution library, each as its own entry, and returns two reserved indices, which `getAlgosFromIndex` and `hipblasLtMatmul` run with checked numerics. Later queries return the same indices from the library without generating. A second process runs the indices before any query, then finds them the same way |
| `jit-library` | The JIT solution library: cache-key fields and compiler-environment filtering; rejected group- or other-writable, linked and non-directory roots; the stock TensileLite loader reading a published library; exact-size matching with the solution predicates still applied; deduplication, hash collisions, order, count and excluded kernels; mismatched and tampered keys ignored and left untouched; index allocation up to `INT32_MAX` and exhaustion; a publisher killed after each publication step; readers reloading after another instance publishes; and a fused GEMM and all-to-all problem rejected by lookup, publication and the ProblemType key without touching the library, even beside a plain solution of the same sizes |
| `jit-library-concurrency` | Eight processes publish shared and private entries into one library while another process looks them up: shared entries get one index, private ones unique indices with no gaps, and every reader snapshot loads |
| `jit-disabled` | The JIT headers are absent from the public include tree, `hipblaslt-ext.hpp` compiles without them, and the extension API links against the disabled library |
| `jit-heuristic-off` | `HIPBLASLT_JIT=0`. Both heuristic queries for the plain-pair FP16 GEMM return no JIT algorithm |
| `jit-heuristic-fallback` | `HIPBLASLT_JIT=1`. With no device library, every returned algorithm is JIT and the first result for K=512 matches the host. With a device library, an Equality size returns Equality algorithms, then JIT, then the others, with no repeated kernel, and an untuned size starts with JIT. Without such a library the ordering check prints `SKIP heuristic-provider-order: the build has no device library with an Equality size` |
| `jit-heuristic-forced` | `HIPBLASLT_JIT=2`. Both queries return only JIT algorithms. K=512 selects the solution ending in `_K512_WGM8` and K=256 the one ending in `_WGM1`. A transposed A returns no algorithm. The first K=512 result matches the host |
| `jit-heuristic-ignored` | Built only with `HIPBLASLT_ENABLE_JIT=OFF`, with `HIPBLASLT_JIT=2`. The queries do not return JIT algorithms |
| `jit-hipkittens` | Built only with `HIPBLASLT_JIT_ENABLE_HIPKITTENS` and gfx950. The compiled-in BF16 and FP16 variants, a capturing stream that returns no solution without compiling, a BF16 TN 256x256x128 GEMM whose D is 128, and M=128 rejected |

## Test arguments

`hipblaslt-jit-bundle-writer` takes the `data` directory and an output
directory. `hipblaslt-jit-builder-test` takes the output directory, the HIP
kernel and a scratch directory. `hipblaslt-jit-end-to-end-test` takes the output directory
and replays the `plain-pair` of device 0's architecture. It creates the replay
backend with `jit::replay::createBackend` from `hipblaslt-jit-replay.hpp`, so
generation runs no generator, and solves FP16 problems with M=256, N=128 and
K=512 or 256. `hipblaslt-jit-loader-test` takes the output directory and a
scratch directory, and checks the same problems with the bundles of the same
architecture. `hipblaslt-jit-source-bundle-test` takes a scratch directory.
`hipblaslt-jit-heuristic-test` takes `off`, `fallback`, `forced` or `ignored`,
and for the first three the bundle directory. CTest sets `HIPBLASLT_JIT`. The
test sets `HIPBLASLT_JIT_TEST_REPLAY` to that directory's `plain-pair` before
either heuristic query. `getIndexFromAlgo` is -1 for a JIT algorithm.

With `--library` after the output directory, `hipblaslt-jit-end-to-end-test`
runs the `jit-end-to-end-library` checks instead and starts its second process
itself. That mode empties `HIPBLASLT_JIT_LIBRARY_PATH` first and refuses to run
unless it is set, so that it never publishes into the default library.

## JIT solution library tests

`hipblaslt-jit-library-test` compiles the JIT solution library directly. It
takes the written gfx950 `plain` bundle, whose library entry it publishes
under several kernel names with stand-in code objects, and a scratch directory
for the libraries it creates; it ignores `HIPBLASLT_JIT_LIBRARY_PATH`. Adding
`--writers N --per-writer M` runs the multi-process check instead: N writer
processes each publish M entries shared by all writers and M of their own,
while one reader process looks them up.
