# Validate the JIT implementation

The JIT tests check the comgr code-object builder, the source bundle reader,
the support check and the JIT solution library.
The JIT headers are not installed. The tests include them from
`library/src/amd_detail`. They build the gfx942 and gfx950 kernel assembly
committed in [`data`](data/README.md) and publish it into a JIT solution
library, so they need neither Python nor a generator.

## Build and run from a checkout

Build as in the [JIT build instructions](../../../JIT.md#build), then, from
the repository root:

```bash
ctest --test-dir "$project_build/clients/tests/jit" -L jit-cpu --output-on-failure
ctest --test-dir "$project_build/clients/tests/jit" -L jit-gpu --output-on-failure
```

`-L jit-cpu` runs the tests that need no GPU. `-L jit-gpu` runs the tests that
need device 0. `GPU_TARGETS` is that device's architecture, `gfx942` or
`gfx950`. The publish, load and launch tests use the committed assembly of
that architecture, the same way on each. `jit-source-bundle`, `jit-builder`,
`jit-publish`, `jit-launch` and `jit-launch-cpu` empty their directories under
`clients/tests/jit/scratch` in the build tree. `jit-loader` reads the library
`jit-publish` wrote.

The CTest tests are:

- `jit-cpu`: `jit-source-bundle`, `jit-builder` and `jit-launch-cpu`. A build
  with `HIPBLASLT_ENABLE_JIT=OFF` has only `jit-source-bundle`.
- `jit-gpu`: `jit-publish`, `jit-loader` and `jit-launch`. CTest runs
  `jit-publish` before `jit-loader`.

`jit-launch-cpu` and the `jit-gpu` tests exist when `GPU_TARGETS` include an
architecture with committed assembly. A build with `HIPBLASLT_ENABLE_YAML=ON`
has none of them, because the library entry is MsgPack.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-source-bundle` | The source bundle reader: assembly, HIP sources and headers sorted by name, relative paths, symbolic links that escape the bundle, size limits, and a `library/` entry, which it rejects. The reader leaves the target, code-object version, solutions and kernel names at their defaults |
| `jit-builder` | Target-ID parsing, and the comgr builder without a GPU: the hand-written HIP kernel `builder_test_kernel.hip` built for each committed assembly target as an executable and as a relocatable, then each assembly file alone, from its relocatable, whose metadata note holds `custom.config`, and linked with that kernel from its source and from its relocatable. Each build returns one artifact of the kind asked for, and each executable defines its kernels and has the bundle's code-object version. A kernel name the executable does not define, a bundle without a target, a relocatable build with helpers, and a relocatable build of a relocatable fail the build |
| `jit-publish` | Reads the device's committed bundle with `readTensileSourceBundle`, which takes its solution from `custom.config`, builds it, and publishes, at each K, the solutions `supportingSolutions` keeps into the JIT solution library: `plain` for K=512, both `plain-pair` solutions for K=1024 and its any-K solution for K=256. Each solution gets its own index in the JIT range, from `JitKernelIndexBase` (2^30) and below `LocalKernelIndexBase` (0x60000000), and publishing again keeps them. The support check alone keeps solutions in order and drops those whose problem predicate rejects K or a transposed A, whose hardware predicate rejects another processor, or whose required workspace exceeds the limit. In a library of its own, with the allocator set to `LocalKernelIndexBase - 1`, publishing at a new K takes that last index and the next new K fails as exhausted, leaving the allocator at `LocalKernelIndexBase`; republishing keeps the last index, `LocalKernelIndexBase` does not resolve, and an allocator above it fails publication |
| `jit-loader` | Loads that library, finds the solutions published for each K in index order, and finds nothing for a transposed A. Both `plain-pair` solutions published for K=1024 map to one entry file that holds them. Launches no kernel |
| `jit-launch` | The `[jit-gpu]` cases of `hipblaslt-jit-launch-test`. Compiles the device's committed `plain` assembly, publishes that executable, retrieves it with `findTopSolutions` and `getSolutionByIndex`, launches the kernel through the TensileLite solution adapter on known FP16 A, B and C, and compares D with a host reference. The library is under `<root>/v1/<git revision>`, and once that directory is renamed to another revision a new library instance finds nothing |
| `jit-launch-cpu` | The `[jit-cpu]` cases of the same executable, without a GPU. It stores the HIP helper relocatable under its kernel name: a second acquire does not compile and returns the same bytes, a helper that does not define its kernel name is not stored, and another library instance reads the stored helper. A second acquire of a held file lock fails with a timeout after at least its 200 ms limit, and succeeds once the lock is released |

## Test parameters

The tests are Catch2 cases. CMake compiles in the data directory, the library
directory and the HIP kernel path, and CTest selects cases by name or tag.
Run `jit-publish` before `jit-loader`; the loader reads the library the
publisher wrote. Problems are FP16 with M=256, N=128 and K=512, 1024 or 256.
