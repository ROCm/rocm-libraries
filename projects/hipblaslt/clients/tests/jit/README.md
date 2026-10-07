# Validate the JIT implementation

The JIT tests check the comgr code-object builder, the source bundle reader,
the JIT solution library and the TensileLite loader. The JIT headers are not
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
  `HIPBLASLT_ENABLE_JIT=OFF` has `jit-source-bundle`.
- `jit-gpu`: `jit-publish` and `jit-loader`, when `GPU_TARGETS` include an
  architecture with committed assembly. CTest runs `jit-publish` before
  `jit-loader`. A build with `HIPBLASLT_ENABLE_YAML=ON` does not have them,
  because the library entry is MsgPack.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-source-bundle` | The source bundle reader: assembly, HIP sources and headers sorted by name, relative paths, symbolic links that escape the bundle, size limits, and a library entry that is missing, empty or not named `library/TensileLibrary.dat` |
| `jit-builder` | The comgr builder, without a GPU, building the hand-written HIP kernel `builder_test_kernel.hip` for each committed assembly target, then each assembly file linked with that kernel into one code object; each code object defines its kernels and has the builder's code-object version, and a kernel name it does not define fails the build |
| `jit-publish` | Builds the device's committed assembly and publishes it into the JIT solution library: `plain` for K=512, and `plain-pair`'s two solutions for K=1024 and K=256. Indices are at least `2^30` |
| `jit-loader` | Loads that library, finds each published solution and the pre-generated kernel, resolves the kernel from its code object, and finds nothing for a transposed A. An entry whose solutions are not 0 to N-1 is rejected. Launches no kernel |

## Test arguments

The tests are Catch2 executables. CMake compiles in the data directory, the
library directory and the HIP kernel path, so a developer runs the staged
binary with no arguments. Run `jit-publish` before `jit-loader`; the loader
reads the library the publisher wrote. Problems are FP16 with M=256, N=128
and K=512, 1024 or 256.
