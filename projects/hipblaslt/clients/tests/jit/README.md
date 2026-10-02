# Validate the JIT implementation

The JIT tests check the Jit stages and the comgr code-object builder. The JIT
headers are not installed. The tests include them from `library/src/amd_detail`.

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
load and launch code on device 0, which must be a gfx950. Each test writes under
`clients/tests/jit/scratch` in the build directory, which CTest empties before
the tests run. The CTest tests are:

- `jit-cpu`: `jit-component` and `jit-code-object`.
- `jit-gpu`: `jit-code-object-gpu`.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-component` | Jit over fake stages, without a GPU: missing components rejected, the generator's units reaching the builder, count limiting, excluded kernels, the stage of each failure, publish and load ordering, scratch lifetime, and concurrent generation |
| `jit-code-object` | comgr assembly, HIP helper compilation and linking for gfx950, build options, concurrent builds, and the status and log of each kind of failed build, without a GPU |
| `jit-code-object-gpu` | The same code objects loaded and launched on the GPU, with their results checked |

## Jit component test

`hipblaslt-jit-component-test` takes one argument, a fresh directory that it
uses as the scratch parent; it needs no GPU. It is built only with
`HIPBLASLT_ENABLE_JIT=ON`.

## Code-object tests

`hipblaslt-jit-code-object-test` compiles the comgr code-object builder
directly. `--out` names a fresh results directory, and either `--target`
selects a compile-only run for that target ID or `--gpu` also loads and runs
the results on device 0, which must match the target. `--ffm` runs the GPU part
on the simulator that `HSA_MODEL_TOPOLOGY` and `HSA_MODEL_LIB` select. Simulator
runs are manual; CTest does not run `--ffm`. `--bundle` adds the checks for a
TensileLite source bundle. `--only` selects tests by name.
