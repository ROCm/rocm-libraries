# HipKittens JIT backend

This is a source guide for contributors and integration developers, maintained
under the existing hipBLASLt code/documentation reviewer rules. The
[JIT guide](JIT.md) describes the parts of just-in-time (JIT) generation that
do not depend on a backend, and the
[HipKittens JIT test guide](clients/tests/jit/README.hipkittens.md) covers this
backend's tests.

The HipKittens backend serves [HipKittens](https://github.com/HazyResearch/HipKittens)
kernels: it returns the HIP source of a hipBLASLt-owned kernel template, with
the HipKittens headers it includes and its library entry, and the comgr builder
compiles it at run time. It is opt-in and built only in developer builds.

**Status:** `hipblaslt-jit-hipkittens.hpp` is in `library/src/amd_detail/`; it
is not installed, and only the JIT tests use it. Builds with the backend export
`jit::hipkittens::createBackend` and `jit::hipkittens::detail::resources` from
it for the test binaries; they are not a supported API. Tests create the
backend with `createBackend` and use it through `getJitAlgo` and
`getLibraryAlgos`. Heuristic queries use it only when `HIPBLASLT_JIT_BACKENDS`
names it.

## Heuristic queries

In a build with the backend, HipKittens is an opt-in backend with the
identifier `hipkittens`: with `HIPBLASLT_JIT_BACKENDS` unset or empty it is not
configured, and heuristic queries behave as in a build without it. Listed, it
serves the JIT step in its place in the list, as the
[JIT guide](JIT.md#heuristic-integration) describes, and is configured once
per process from the headers that `HIPBLASLT_JIT_HIPKITTENS_PATH` or the
default locations provide. For example, with `HIPBLASLT_JIT=2` and
`HIPBLASLT_JIT_BACKENDS=hipkittens`, a heuristic query for a problem in the
kernel's domain returns the HipKittens kernel, and `hipblasLtMatmul` without an
algorithm runs it. A problem outside the kernel's domain, or another device,
skips it without a report; when no listed backend serves the problem, the
query reports that no enabled JIT backend supports it. When its headers are
missing, each problem reports one configure failure, an error when the query
gets no JIT result:

```text
hipblaslt error: JIT configure failed for GEMM M=1024 N=512 K=768 ... EPILOGUE_DEFAULT: JIT backend HipKittens not available: headers not found at /opt/rocm/lib/hipblaslt/hipkittens/be1c91841b81; set HIPBLASLT_JIT_HIPKITTENS_PATH
```

`hipblasLtMatmulAlgoGetHeuristic` builds its problem with beta 1;
`GemmInstance::algoGetHeuristic` and `hipblasLtMatmul` without an algorithm
use the problem's own alpha and beta. The kernel serves any beta, so all three
return it.

## Build

`cmake/hipblaslt-jit-hipkittens.cmake` adds this backend to a build with
`HIPBLASLT_ENABLE_JIT=ON`. Its option `HIPBLASLT_JIT_ENABLE_HIPKITTENS` is `OFF`
by default; the `jit` preset and the gfx950 job of the JIT CI workflow turn it
on, and a build with it off installs nothing for HipKittens. When no
`GPU_TARGETS` entry has a HipKittens kernel, configuration prints a warning and
builds without the backend. The build runs `make_entries.py` with TensileLite
Python and the build's rocisa extension.

## Kernel

`library/src/amd_detail/hipkittens/` holds two variants of one template,
`gemm_tn_256x256x64_gfx950.hip`: the kernel of the HipKittens 256x256x64 BF16
GEMM, written over its element type, behind a wrapper that takes
`A, B, C, D, m, n, k, alpha, beta` and the leading dimensions and batch strides
of A, B, C and D (84 bytes of kernel arguments) and passes B and A to the
kernel in that order:
the kernel's row-major `C = a·bᵀ` is then hipBLASLt's column-major
`D = Aᵀ·B`. Its epilogue computes `alpha·Aᵀ·B + beta·C` in FP32 before
rounding to the element type; with beta 0 it does not read C, and C may be D.
It uses 160,000 bytes of LDS and launches 512 threads per 256x256 output tile
of each batch, with the batch in the grid's z dimension.

| Variant | Kernel | A, B, C and D | VGPRs |
| --- | --- | --- | --- |
| `gemm_bf16_tn_256x256x64_gfx950` | `HK_gemm_bf16_TN_MT256x256x64_W2x4_gfx950_abi5` | BF16 | 237 |
| `gemm_f16_tn_256x256x64_gfx950` | `HK_gemm_f16_TN_MT256x256x64_W2x4_gfx950_abi5`, built with `-DHK_FP16` | FP16 | 238 |

HipKittens' `mma_ABt` accumulates only BF16 in FP32, so the template repeats
its loop over HipKittens' base MFMA operation, which accumulates FP16 in FP32
too. Each variant serves gfx950 problems with:

- `opA = T` and `opB = N`, its type for A, B, C and D, and FP32 compute;
- alpha and beta on the host, alpha not 0 (hipBLASLt turns alpha 0 into
  K = 0), and no bias, activation or alpha vector;
- strided batches with any batch strides (not pointer arrays), M and N
  multiples of 256, and K a multiple of 128 (the kernel computes wrong results
  when K is an odd multiple of 64);
- any leading dimensions, each tensor, all its batches included, under
  4 GiB, and each matrix's leading dimension times its column count under
  4 GiB.

Other problems get `NotSupported`, and other devices `TargetMismatch`.

## Entries

At build time `make_entries.py` runs TensileLite to write each variant's
one-solution library entry, a custom kernel with the variant's ProblemType and
size predicates, and compiles it into the library. The kernel addresses each
matrix through one buffer descriptor, so the entry's buffer limit checks span
the whole matrix, not one macro tile. For each request the backend adds a
predicate requiring K > 0 and checks the request against the entry before
returning the solution. The entry depends on nothing else in the request,
because JIT library rows match only sizes. The comgr builder compiles the
source with the variant's flags (`-std=c++20 -DKITTENS_CDNA4 -w`, and
`-DHK_FP16` for FP16) after its own.

## Cache key

The backend identifier is `hipkittens`, and its version is a hash of the header
manifest and of each variant's name, source, entry and flags. Its solutions are
published in their own key directory, and `getAlgosFromIndex` resolves them
with JIT off, as for any backend.

## Headers

Configuration downloads HipKittens at commit `be1c918`, pinned by the archive's
SHA-256; for an offline build, set `FETCHCONTENT_SOURCE_DIR_HIPKITTENS` to an
unpacked archive of that commit. The build stages the headers the kernel
includes (`include/kittens.cuh`, `include/pyutils/util.cuh` and
`include/cdna4/`, 70 files) and a `manifest.json` of their sizes and SHA-256
hashes next to the built library, in `hipblaslt/hipkittens/be1c91841b81/`. The
runtime component installs them in
`<libdir>/hipblaslt/hipkittens/be1c91841b81/`. The backend uses the first of
these directories that exists:

1. `jit::hipkittens::Options::headers`;
2. `HIPBLASLT_JIT_HIPKITTENS_PATH`;
3. `hipblaslt/hipkittens/be1c91841b81` next to the loaded `libhipblaslt`, so a
   moved installation still finds its headers;
4. the configured installation directory.

Its `manifest.json` must equal the manifest compiled into the library, and each
listed file must be a regular file inside the directory with the listed size
and hash. The ROCm HIP headers must exist at
`<ROCm path>/include/hip/hip_runtime.h`. Otherwise `createBackend` returns
`HIPBLAS_STATUS_INVALID_VALUE` and its diagnostic names the cause, for example:

```text
JIT backend HipKittens not available: headers not found at /opt/rocm/lib/hipblaslt/hipkittens/be1c91841b81; set HIPBLASLT_JIT_HIPKITTENS_PATH
```

## License

HipKittens is MIT-licensed, Copyright (c) 2024 HazyResearch, and
[NOTICES.txt](NOTICES.txt) lists it. The runtime component installs its license
as `share/doc/hipblaslt/third-party/hipkittens/LICENSE`, and the kernel source
names the HipKittens commit and file that its kernel comes from.
