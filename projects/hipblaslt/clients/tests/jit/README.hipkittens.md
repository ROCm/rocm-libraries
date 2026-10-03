# Validate the HipKittens JIT backend

These tests use the [HipKittens backend](../../../JIT_HIPKITTENS.md). The
[JIT test guide](README.md) covers the tests that need no generator, and the
[TensileLite JIT test guide](README.tensilelite.md) the shared driver,
`.github/scripts/test_hipblaslt_jit.py`, which also runs the cases below.

## Build and run from a checkout

`hipblaslt-jit-hipkittens-test` exists only in a build configured with
`-DHIPBLASLT_JIT_ENABLE_HIPKITTENS=ON` and gfx950 among `GPU_TARGETS`; the
`jit` preset sets the option. The driver's `hipkittens-*` cases print SKIP
when the binary is absent or the device is not gfx950. From the repository
root, with the environment of the TensileLite JIT test guide:

```bash
cmake --build "$project_build" --parallel 8 --target hipblaslt-jit-hipkittens-test hipblaslt-bench
"$project_python" .github/scripts/test_hipblaslt_jit.py \
  --build "$project_build" --architecture gfx950 --output "$(mktemp -d)/hipkittens" \
  --case hipkittens-backend --case hipkittens-gemm --case hipkittens-bench \
  --case hipkittens-install --case hipkittens-heuristic
```

The binary takes a mode:

| Mode | Runs |
| --- | --- |
| `host <fresh-scratch>` | The `hipkittens-backend` checks; needs a device but runs no kernel |
| `gpu` | The `hipkittens-gemm` checks on gfx950; requires `HIPBLASLT_JIT_LIBRARY_PATH` |
| `library` | Prints the headers it uses, publishes one solution, runs it, and prints `INDEX <n>`; requires `HIPBLASLT_JIT_LIBRARY_PATH` |
| `heuristic` | Runs M=1024 N=512 K=768 through `hipblasLtMatmul` without an algorithm, then the first result of `GemmInstance::algoGetHeuristic`, which must be the HipKittens kernel, and checks both against the CPU reference; needs `HIPBLASLT_JIT` and `HIPBLASLT_JIT_BACKENDS` naming `hipkittens` first |

`test_hipkittens_bench.py <test> <hipblaslt-bench> <fresh-output>` runs the
`hipkittens-bench` case,
`test_hipkittens_install.py <build> <test> <fresh-output>` the
`hipkittens-install` case, and
`test_hipkittens_heuristic.py <test> <hipblaslt-bench> <fresh-output>` the
`hipkittens-heuristic` case.

## What each driver case checks

| Driver case | Behavior under test |
| --- | --- |
| `hipkittens-backend` | The backend without running a kernel: header discovery through the default location, `Options::headers` and `HIPBLASLT_JIT_HIPKITTENS_PATH`, and one "not available" failure for an empty directory, a missing, edited or linked-out file, and another commit's manifest; one solution for its domain, including beta 1 and -0.5, alpha 1.5, two batches or three with gaps between them, and padded or odd leading dimensions of A, B, C and D, and `NotSupported` for each excluded transpose, type, alpha (0, on the device, or a vector), pointer-array batch, size, epilogue, a tensor over 4 GiB, alone or across its batches, and an A whose leading dimension times its columns reaches 4 GiB; `TargetMismatch` for gfx942; the entry loaded by the Tensile loader with buffer limit checks spanning whole matrices, a K > 0 predicate, and no stride, alpha-1, beta-0 or batch predicate; and the comgr-built kernel's 84-byte arguments, 160,000-byte LDS, 237 VGPRs and no spills |
| `hipkittens-gemm` | On gfx950, `getJitAlgo` through `hipblasLtMatmul` and `Gemm` for seven shapes from 256×256×128 to 8192³ with beta 0 and C filled with NaN, for alpha 1.5 and -0.25 and beta 1, -0.5 and 2, with C separate or C = D, up to 4096³, for two to four batches, packed or with gaps between them, and for padded or odd leading dimensions, also with batches and C = D, compared with a CPU reference, with canaries around D, in its gaps and in its padding, and repeated runs identical; the shapes the kernel computes wrongly rejected before launch; base offsets of 2 and 16 bytes; and `getLibraryAlgos` publishing an index that also serves and runs ldA ≠ K, refuses an A that spans 4 GiB, and a second process runs with JIT off |
| `hipkittens-bench` | A published HipKittens index run by `hipblaslt-bench --algo_method index --verify --alpha 2 --beta 1 --batch_count 3` through `hipblasLtMatmul` with JIT off |
| `hipkittens-heuristic` | In mode 2, `hipblaslt-bench --api_method mix --verify` for M=1024 N=512 K=768 with beta 0: with `HIPBLASLT_JIT_BACKENDS=tensilelite,hipkittens` and four requested, TensileLite solutions and then the HipKittens kernel, one key directory for each backend; the same order from `--api_method c --beta 1`, through `hipblasLtMatmulAlgoGetHeuristic`; unset, with the headers missing and as many requested as TensileLite returned, the same TensileLite solutions and no mention of HipKittens; listed, with the headers missing and one more requested, those TensileLite solutions first, no HipKittens kernel, and one configure warning naming HipKittens. Then the test's `heuristic` mode with `hipkittens,tensilelite`, after which only HipKittens has published |
| `hipkittens-install` | `cmake --install --component runtime` installs the headers, their manifest and the license; a HipKittens index published and run against the installed library finds the installed headers, and after the installation is moved it runs again with the same index |
