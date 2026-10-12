# Validate the JIT implementation

The JIT tests check the comgr code-object builder, the source bundle reader,
the support check and the JIT solution library. The JIT headers are not
installed. The tests include them from `library/src/amd_detail`. They build the
gfx90a, gfx942 and gfx950 kernel assembly committed in [`data`](data/README.md)
and publish it into a JIT solution library, so they need neither Python nor a
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
need device 0, which must be a gfx90a, gfx942 or gfx950; they publish and load
the committed assembly of its architecture. Each test empties its own directory
under `clients/tests/jit/scratch` in the build directory before it runs, except
`jit-loader`, which reads the library `jit-publish` wrote.

The CTest tests are:

- `jit-cpu`: `jit-source-bundle` and `jit-builder`. A build with
  `HIPBLASLT_ENABLE_JIT=OFF` has `jit-source-bundle` and `jit-disabled`.
- `jit-gpu`: `jit-publish`, `jit-loader` and `jit-launch`, when `GPU_TARGETS`
  include an architecture with committed assembly. CTest runs `jit-publish`
  before `jit-loader`. A build with `HIPBLASLT_ENABLE_YAML=ON` does not have
  them, because the library entry is MsgPack.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-source-bundle` | The source bundle reader: assembly, HIP sources and headers sorted by name, relative paths, symbolic links that escape the bundle, size limits, and a `library/` entry, which it rejects. The reader leaves the target, code-object version, solutions and kernel names at their defaults |
| `jit-disabled` | Built only with `HIPBLASLT_ENABLE_JIT=OFF`. The JIT headers are absent from the public include tree, `hipblaslt-ext.hpp` compiles without them, and the extension API links against the disabled library |
| `jit-builder` | Target-ID parsing, and the comgr builder without a GPU: the hand-written HIP kernel `builder_test_kernel.hip` built for each committed assembly target as an executable and as a relocatable, then each assembly file alone, from its relocatable, whose metadata note holds `custom.config`, and linked with that kernel from its source and from its relocatable. Each build returns one artifact of the kind asked for, and each executable defines its kernels and has the bundle's code-object version. A kernel name the executable does not define, a bundle without a target, a relocatable build with helpers, and a relocatable build of a relocatable fail the build |
| `jit-publish` | Reads the device's committed bundle with `readTensileSourceBundle`, which takes its solution from `custom.config`, builds it, and publishes, at each K, the solutions `supportingSolutions` keeps into the JIT solution library: `plain` for K=512, both `plain-pair` solutions for K=1024 and its any-K solution for K=256. Each solution gets its own index, at least `2^30`, and publishing again keeps them. The support check alone keeps solutions in order and drops those whose problem predicate rejects K or a transposed A, whose hardware predicate rejects another processor, or whose required workspace exceeds the limit |
| `jit-loader` | Loads that library, finds the solutions published for each K in index order, resolves the kernel from its code object, and finds nothing for a transposed A. Both `plain-pair` solutions published for K=1024 map to one entry file that holds them. Launches no kernel |
| `jit-launch` | Compiles the device's committed `plain` assembly, publishes that executable, retrieves it with `findTopSolutions` and `getSolutionByIndex`, launches the kernel through the TensileLite solution adapter on known FP16 A, B and C, and compares D with a host reference. The library is under `<root>/v1/<git revision>`, and once that directory is renamed to another revision a new library instance finds nothing. Without a GPU, it also stores the HIP helper relocatable under its kernel name: a second acquire does not compile and returns the same bytes, and the gfx950 `plain` assembly linked with it defines both kernels |

## Test arguments

The tests are Catch2 executables. CMake compiles in the data directory, the
library directory and the HIP kernel path, so a developer runs the staged
binary with no arguments. Run `jit-publish` before `jit-loader`; the loader
reads the library the publisher wrote. Problems are FP16 with M=256, N=128
and K=512, 1024 or 256.
