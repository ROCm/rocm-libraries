# Validate the JIT implementation

The JIT tests check the Jit stages. The JIT headers are not installed. The
tests include them from `library/src/amd_detail`.

## Build and run from a checkout

From the repository root, with `project_build` set as in the
[JIT build instructions](../../../JIT.md#build):

```bash
cmake -S projects/hipblaslt -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_HOST=ON -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950
cmake --build "$project_build" --parallel
ctest --test-dir "$project_build/clients/tests/jit" -L jit-cpu --output-on-failure
```

`-L jit-cpu` runs the tests that need no GPU. Each test writes under
`clients/tests/jit/scratch` in the build directory, which CTest empties before
the tests run. The CTest tests are:

- `jit-cpu`: `jit-component`.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-component` | Jit over fake stages, without a GPU: missing components rejected, the generator's units reaching the builder, count limiting, excluded kernels, the stage of each failure, publish and load ordering, scratch lifetime, and concurrent generation |

## Jit component test

`hipblaslt-jit-component-test` takes one argument, a fresh directory that it
uses as the scratch parent; it needs no GPU. It is built only with
`HIPBLASLT_ENABLE_JIT=ON`.
