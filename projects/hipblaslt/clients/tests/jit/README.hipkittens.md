# Validate the HipKittens JIT backend

These tests use the [HipKittens backend](../../../JIT_HIPKITTENS.md). The
[JIT test guide](README.md) covers the other JIT tests and how to build and run
them.

## Build and run from a checkout

`hipblaslt-jit-hipkittens-test` exists only in a build configured with
`-DHIPBLASLT_JIT_ENABLE_HIPKITTENS=ON` and gfx950 among `GPU_TARGETS`; the
`jit` preset sets the option. Every mode needs a gfx950 device, so the CTest
tests that run it have the `jit-gpu` label. `jit-hipkittens-install` is
registered only when install rules are generated (not with
`CMAKE_SKIP_INSTALL_RULES=ON`), and `jit-hipkittens-bench` only with
`HIPBLASLT_ENABLE_CLIENT=ON`. From the repository root:

```bash
cmake -S projects/hipblaslt -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_JIT_ENABLE_HIPKITTENS=ON \
  -DHIPBLASLT_BUILD_TESTING=ON -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950
cmake --build "$project_build" --parallel
ctest --test-dir "$project_build/clients/tests/jit" -R '^jit-hipkittens-' --output-on-failure
```

The binary takes a mode:

| Mode | Runs |
| --- | --- |
| `host <fresh-scratch>` | The `jit-hipkittens-backend` checks; needs a device but runs no kernel |
| `gpu` | The `jit-hipkittens-gemm` checks on gfx950; requires `HIPBLASLT_JIT_LIBRARY_PATH` |
| `library` | Prints the headers it uses, publishes one solution, runs it, and prints `INDEX <n>`; requires `HIPBLASLT_JIT_LIBRARY_PATH` |

`test_hipkittens_bench.py <test> <hipblaslt-bench> <fresh-output>` runs the
`jit-hipkittens-bench` test, and
`test_hipkittens_install.py <build> <test> <fresh-output>` the
`jit-hipkittens-install` test.

## What each test checks

| CTest test | Behavior under test |
| --- | --- |
| `jit-hipkittens-backend` | The backend without running a kernel: header discovery through the default location, `Options::headers` and `HIPBLASLT_JIT_HIPKITTENS_PATH`, and one "not available" failure for an empty directory, a missing, edited or linked-out file, and another commit's manifest; one solution for its domain, including beta 1 and -0.5, alpha 1.5, two batches or three with gaps between them, padded or odd leading dimensions of A, B, C and D, and FP16 from the FP16 variant, and `NotSupported` for each excluded transpose, type (FP32 D, also with FP16 A and B), alpha (0, on the device, or a vector), pointer-array batch, size, epilogue, a tensor over 4 GiB, alone or across its batches, and an A whose leading dimension times its columns reaches 4 GiB, and FP16 NN or K = 192 too; `TargetMismatch` for gfx942; the entry loaded by the Tensile loader with buffer limit checks spanning whole matrices, a K > 0 predicate, and no stride, alpha-1, beta-0 or batch predicate; and each comgr-built kernel's 84-byte arguments, 160,000-byte LDS, VGPRs (237 for BF16, 238 for FP16) and no spills |
| `jit-hipkittens-gemm` | On gfx950, `getJitAlgo` through `hipblasLtMatmul` and `Gemm` for seven shapes from 256×256×128 to 8192³ with beta 0 and C filled with NaN, for alpha 1.5 and -0.25 and beta 1, -0.5 and 2, with C separate or C = D, up to 4096³, for two to four batches, packed or with gaps between them, for padded or odd leading dimensions, also with batches and C = D, and for FP16 up to 4096³, with alpha, beta, batches with gaps, padded leading dimensions and C = D, compared with a CPU reference, with canaries around D, in its gaps and in its padding, and repeated runs identical; the shapes the kernel computes wrongly rejected before launch; base offsets of 2 and 16 bytes; and `getLibraryAlgos` publishing an index that also serves and runs ldA ≠ K, refuses an A that spans 4 GiB, and a second process runs with JIT off |
| `jit-hipkittens-bench` | A published HipKittens index run by `hipblaslt-bench --algo_method index --verify --alpha 2 --beta 1 --batch_count 3` through `hipblasLtMatmul` with JIT off |
| `jit-hipkittens-install` | `cmake --install --component runtime` installs the headers, their manifest and the license; a HipKittens index published and run against the installed library finds the installed headers, and after the installation is moved it runs again with the same index |
