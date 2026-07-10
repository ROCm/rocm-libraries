# hipBLASLt Python Interface — Design

Date: 2026-07-10
Status: Design (pending implementation plan)

## Summary

A low-level Python package that binds the hipBLASLt C API, letting developers
drive GEMMs from Python for debugging, numerical-correctness investigation, and
exploratory benchmarking. It is deliberately *lower level* than PyTorch or CuPy:
it exposes the actual hipBLASLt control surface (matmul descriptor attributes,
matrix layouts, heuristic algorithm enumeration, workspace, epilogues, scales)
rather than a high-level tensor abstraction. "GEMM on numpy arrays" is shorthand
— numpy is the host / reference side; real GEMM inputs live in explicit device
memory.

The package is a nanobind C++ extension shipped within the hipBLASLt
distribution, reusing the build machinery already proven by `tensilelite/rocisa`.

## Primary users and goals

The bullseye user is a **hipBLASLt developer**, not an end-user ML practitioner.
Concrete workflows:

- **Confirm bugs** — run a GEMM and inspect the actual output values.
- **Numerical-correctness investigation** — compare the GPU result directly
  against a CPU reference computed with numpy/scipy.
- **Exploratory benchmarking** — quickly compare algorithms / problem sizes.

Because the tool exists to debug hipBLASLt *itself*, **full low-level control is
the essential requirement** (enumerate heuristic algorithms, pin a specific
algo, set workspace / epilogue / scales explicitly). A convenience auto-selecting
`matmul()` is a secondary nice-to-have and must be a thin shim over the
low-level layer, never a parallel code path — a convenience API that hides algo
selection would hide the very thing being debugged.

## Motivation: why not just use `hipblaslt-bench` / `hipblaslt-test`?

The existing C++ clients are statically-configured, compiled batch runners.
`hipblaslt-bench` is driven by CLI flags or a YAML "sequence" of layers;
`hipblaslt-test` is a gtest binary driven by YAML that is code-generated into
`hipblaslt_gtest.data`. Both require a rebuild to change what is tested in code,
and both emit results to stdout / logs rather than into a programmable
environment. The Python layer offers, over and above them:

1. **Direct numerical diff against a trusted reference in-process.** Compute
   `A @ B` with numpy/scipy right next to the GPU result and diff them
   element-wise, with full control over the metric (max abs error, ULP,
   relative error, which elements diverge). This is the single biggest win.
2. **Inspect outputs as arrays, not log lines.** Pull `D` back to host as a
   numpy array and slice / plot / histogram the error to find *which* rows and
   columns are wrong.
3. **Trivially construct pathological inputs.** Build the exact adversarial
   matrix that triggers a bug (specific NaN placement, denormals, rank-deficient
   input) with one line of numpy, instead of a fixed `--initialization` menu.
4. **Programmatic algorithm enumeration and bisection.** `heuristic()` returns
   the ranked algo list as Python objects; loop over every candidate, run each,
   and find *which* algo index produces wrong numbers — in a `for` loop, not by
   editing YAML and rebuilding.
5. **No rebuild to change the experiment.** A REPL or pytest changes the
   problem, dtype, epilogue, or algo instantly; the bug-hunting inner loop
   collapses from minutes to seconds.
6. **A bug repro is a short `.py` script**, self-contained and portable to a
   ticket, rather than a YAML edit plus a build recipe.
7. **Composable with the scientific-Python stack** — matplotlib, pandas,
   pytest parametrization, jupyter.
8. **Header-derived coverage harness** (see Testing) — a meta-test asserting
   every enum knob is exercised, which has no equivalent in the YAML-driven
   gtest suite.

### Non-goals (so the doc does not oversell)

- **Not** a replacement for `hipblaslt-bench`'s hardened timing methodology
  (cold iterations, rotating buffers, graph mode). For authoritative
  performance numbers, bench remains canonical; the Python layer's benchmarking
  value is *exploratory* (quick relative comparisons).
- **Not** a replacement for `hipblaslt-test` as the CI correctness gate.

The Python layer complements the C++ clients; it does not retire them.

## Ecosystem interop and agent workflows

Because the design adopts `ml_dtypes` (shared dtype semantics) and DLPack
(zero-copy tensor exchange), two capabilities emerge as first-class:

**Cross-framework comparison.** The layer becomes a neutral bench where
hipBLASLt sits next to any other implementation and they are diffed on identical
bits:

- hipBLASLt vs. PyTorch-ROCm / CuPy (inputs handed to both via DLPack).
- hipBLASLt vs. a CPU reference (numpy/scipy) — the correctness baseline.
- hipBLASLt vs. hipBLASLt across algos, dtypes, ROCm versions, or GPU archs.

Because inputs are exact shared bytes (not each library's own random
initialization), differences are attributable to the *math*, not to divergent
input generation — a guarantee the current CLI clients cannot make across tools.

**Programmability and agent-amenability.** Everything returns structured data
(arrays, lists of algo results, error dicts) rather than stdout to be scraped.
An automated agent can drive a bug hunt end-to-end: sweep sizes / dtypes /
algos, compute the numpy reference, flag divergences, emit a minimal `.py`
reproducer, and even write coverage tests for a newly-added enum — with no YAML
editing or rebuilds in the loop.

**Caveat (must stay honest).** Cross-framework diffs are only meaningful when
compute precision and accumulation order are matched, or the tolerance accounts
for them. Two libraries computing "the same" GEMM in different internal
precision will differ legitimately. The layer *enables* the comparison; the user
still interprets it.

## Architecture and package layout

A new self-contained package under `projects/hipblaslt/python/`. The Python
import name is **`hipblaslt`**, matching the library it wraps. It is a nanobind
C++ extension built with scikit-build-core, using `tensilelite/rocisa` as the
structural template but as its own independent package.

```
projects/hipblaslt/python/          (new)
├── pyproject.toml                  scikit-build-core + nanobind, mirrors rocisa's
├── CMakeLists.txt                  nanobind_add_module; links libhipblaslt + libamdhip64
├── src/
│   ├── module.cpp                  NB_MODULE entry
│   ├── device_array.cpp/.hpp       DeviceArray: hipMalloc/memcpy, DLPack, numpy view
│   ├── enums.cpp                   bind hipDataType, epilogue, all *_ATTR enums
│   ├── descriptors.cpp             Handle, MatmulDesc, MatrixLayout, Preference
│   └── matmul.cpp                  heuristic() + matmul() low-level call
├── hipblaslt/
│   ├── __init__.py                 re-exports compiled _core; thin convenience matmul()
│   └── _coverage.py                header-enum extraction for the coverage meta-test
└── tests/
    ├── conftest.py
    ├── test_*.py                   correctness + surface tests
    └── test_api_coverage.py        meta-test: every header enum value is exercised
```

Three layers, bottom to top:

1. **`_core`** (compiled) — a 1:1 mirror of the hipBLASLt C API. Opaque objects
   wrap the handles; no policy, no hidden auto-selection. This is the
   "full control" surface.
2. **`DeviceArray`** (compiled) — explicit host/device memory object, the
   data-plane counterpart to the control-plane in `_core`.
3. **`hipblaslt/__init__.py`** (pure Python) — thin re-exports plus the
   ancillary convenience `matmul(a, b)` shim, built strictly on top of layer 1.

Layers 1 and 2 are the product; layer 3 is a convenience skin that adds no
capability the lower layers lack, so nothing being debugged is ever hidden.

### Build integration

The Python package is built **opt-in**: it is not part of the default
`invoke build` and does not affect existing host/device/client builds unless
explicitly requested. A new flag on `invoke build` (e.g. `--python` /
`-p`, exact name settled in the implementation plan) enables configuring and
building the extension. Rationale: the package is a developer tool, not part of
the shipped library, and keeping it off the default path means it adds zero cost
or dependency risk to normal builds and CI until someone asks for it. Detailed
wiring (the flag plumbing through `tasks.py` / cmake, CI jobs, and wheel
strategy per Python × ROCm combination) is deferred to the implementation plan.

## Core objects and the low-level control surface

`_core` mirrors the C flow object-for-object:

- **`Handle`** — wraps `hipblasLtHandle_t`; RAII (context manager + explicit
  close).
- **`MatmulDesc`** — wraps `hipblasLtMatmulDesc_t`, constructed with compute type
  + scale type. A generic `.set_attribute(attr, value)` / `.get_attribute(attr)`
  pair covers *every* `HIPBLASLT_MATMUL_DESC_*` (transpose, epilogue, bias
  pointer, scale pointers, ...).
- **`MatrixLayout`** — wraps `hipblasLtMatrixLayout_t` (dtype, rows, cols, ld) +
  `.set_attribute()` for batch count / stride and the rest.
- **`Preference`** — wraps `hipblasLtMatmulPreference_t` (max workspace size,
  etc.).
- **`Algo`** — opaque wrapper around `hipblasLtMatmulAlgo_t`, returned by the
  heuristic query and passed back to matmul; exposes its index / identifiers for
  logging ("algo #7 is the bad one").

Two entry points:

- **`heuristic(handle, desc, a_layout, b_layout, c_layout, d_layout,
  preference, max_results) -> list[HeuristicResult]`** — wraps
  `hipblasLtMatmulAlgoGetHeuristic`, which returns candidate algorithms for the
  given problem in order of increasing estimated compute time. Each result
  carries an `Algo`, its required `workspace_size`, a `state`, and `waves_count`.
  This is the enumeration surface: inspect the list and pin one.
- **`matmul(handle, desc, alpha, A, a_layout, B, b_layout, beta, C, c_layout,
  D, d_layout, algo, workspace, stream=None)`** — near-verbatim
  `hipblasLtMatmul`. `alpha`/`beta` are host scalars; `A/B/C/D` are
  `DeviceArray`s; `algo` is one the caller picked (never auto-chosen at this
  layer); `workspace` is a `DeviceArray` sized from the heuristic result.

**Design stance: generic attributes over named methods.** Descriptor and layout
attributes go through generic `set_attribute` / `get_attribute` rather than
dozens of named methods. This keeps the binding thin, makes new upstream enum
values work with zero binding changes (the value flows straight through), and
lets the coverage meta-test enumerate knobs uniformly. The tradeoff is less
Python-native ergonomics (pass `enum + value` rather than `.set_epilogue(...)`),
which is acceptable because the audience thinks in these attributes anyway.

## DeviceArray and dtype handling

**`DeviceArray`** — torch-Tensor-like data plane:

- Owns a `hipMalloc`'d buffer, freed on destruction (RAII + explicit `.free()` /
  context manager). Carries `shape`, `dtype`, `strides`/`ld`, device id.
- Explicit movement, never implicit: `DeviceArray.from_numpy(arr, dtype=...)`
  does H2D; `.to_numpy()` does D2H. matmul takes `DeviceArray`s only, so a
  benchmark never accidentally measures a transfer.
- `.copy_from_host(arr)` / `.copy_to_host(out)` to reuse an allocation across
  benchmark iterations.
- DLPack `__dlpack__()` / `from_dlpack()` escape hatch: pass a torch/CuPy tensor
  in, or hand a `DeviceArray` out, zero-copy — without those frameworks being
  dependencies.

### fp8 element-type surface

"fp8" in this repo is not one type but **five** element types, split along two
axes — E4M3 vs. E5M2 (exponent/mantissa split), and **FNUZ vs. OCP** (AMD's
finite-only encoding vs. the OCP standard) — plus an extended E5M3 variant
(`hipblaslt_float8.h`, `hipblaslt-types.h`):

| Element type | Wrapper / enum | Notes |
|---|---|---|
| E4M3 FNUZ | `hipblaslt_f8_fnuz` | gfx94x / MI300 flavor |
| E5M2 FNUZ | `hipblaslt_bf8_fnuz` | gfx94x / MI300 flavor |
| E4M3 OCP | OCP E4M3 | OCP standard, gfx95x / MI350 |
| E5M2 OCP | OCP E5M2 | OCP standard |
| E5M3 EXT | `HIP_R_8F_E5M3_EXT = 34` | extended variant |

**Which fp8 types are usable is a runtime property of the GPU arch** (FNUZ on
MI300, OCP on MI350), not a property of the binding. The binding therefore
exposes **all five** uniformly; when the current arch cannot run a combination,
the library returns `HIPBLAS_STATUS_NOT_SUPPORTED`, which the status-checking
layer (see Error handling) surfaces as a clean Python exception. The binding
does not pre-filter by arch — it lets the device report support, which is the
honest behavior for a correctness tool.

### Dtype host representation and ground-truth encoding

- **Host representation: `ml_dtypes`.** `ml_dtypes` is an approved dependency
  (de-facto standard; JAX/TF) and provides numpy-compatible scalar types for
  most of the narrow surface: `float8_e4m3fn`, `float8_e5m2`,
  `float8_e4m3fnuz`, `float8_e5m2fnuz`, `float8_e8m0fnu` (the MX scale type),
  `float6_e2m3fn`, `float6_e3m2fn`, and `float4_e2m1fn`. This lets a host
  reference array be a *real* narrow-type array (inspectable as numpy) rather
  than an f32 stand-in. Caveats: (1) fp6/fp4/e8m0 are recent `ml_dtypes`
  additions, so the implementation plan must pin a minimum version and degrade
  gracefully if a type is absent; (2) `ml_dtypes` gives scalar *semantics*, not
  device *packing* — it stores sub-byte types one-value-per-byte, whereas the
  GPU wants them bit-packed.
- **Ground-truth encoding: hipBLASLt's own C++ converters.** The authoritative
  encode/decode of *device* bytes reuses hipBLASLt's own conversion routines
  (from `hipblaslt_float8.h` etc.) via `pack_*` / `unpack_*` helpers, so the
  Python-produced device bytes match what the library uses internally
  bit-for-bit — critical for correctness work. The sub-byte packing/layout for
  fp4 (4-bit) and fp6 (6-bit) is owned by these helpers so the user never
  hand-packs nibbles, even when `ml_dtypes` supplies the scalar type.
- **Cross-check test.** A test asserts `ml_dtypes` encoding == hipBLASLt encoding
  bit-for-bit. Any divergence (a rounding-mode or FNUZ-edge difference) is itself
  a finding — exactly the class of bug this tool exists to catch.

The only element type with likely no `ml_dtypes` equivalent is **E5M3 EXT**,
which remains a pack/unpack-helper-only ("Tier 2") type. `xfloat32` is likewise
handled via helpers.

**Reference-precision stance.** Host reference math is computed at *widened*
precision (e.g. f32) and compared against the narrow GPU result with an
appropriate tolerance, because a correctness investigation wants "is the GPU
result within expected error of the true math," not "does it match another
low-precision computation."

## Block scaling and MX types

MX (microscaling) is **not a data type — it is a block-scaling scheme**: an MX
tensor is a narrow element type (fp8 / fp6 / fp4) plus a separate tensor of
per-block scale factors. hipBLASLt models this through the
`hipblasLtMatmulMatrixScale_t` enum, set via the `A_SCALE_MODE` / `B_SCALE_MODE`
descriptor attributes (`hipblaslt.h`). The relevant modes:

| Scale mode | Meaning | Header status |
|---|---|---|
| `SCALAR_32F` (0) | one f32 scale for the whole tensor (non-MX fp8 default) | supported |
| `VEC32_UE8M0` (2) | UE8M0 scale per 32-element block — OCP **MXFP** | supported |
| `BLK32_UE8M0_32_8_EXT` (1001) | UE8M0 per-32-block, pre-swizzled for the kernel | supported |
| `OUTER_VEC_32F` (3) | per-row/col f32 vectors (A: M elems, B: N elems) | supported |
| `VEC16_UE4M3` (1), `VEC128_32F` (4), `BLK128x128_32F` (5), `VEC16_UE8M0_EXT` (1002), `VEC32_UE4M3_EXT` (1003), `VEC16/32_UE5M3_EXT` (1004/5) | other block sizes / scale encodings | "Not supported yet" |

Design implications:

- **Control plane is already covered by the generic-attribute design.** Selecting
  an MX mode is just `desc.set_attribute(A_SCALE_MODE, VEC32_UE8M0)` — no new
  binding code, and the coverage harness enumerates these scale-mode values
  automatically.
- **Data plane needs scale-tensor support** the base `DeviceArray` design did not
  yet call out: a `DeviceArray` must be able to hold a **block-scale tensor**
  (e.g. UE8M0 bytes, one per 32-element block) alongside the element tensor.
  Helpers must (a) build the block-scale tensor from a reference (per-block max →
  UE8M0 exponent), (b) apply the block scales when computing the numpy reference
  so the CPU comparison matches MX math, and (c) own the **pre-swizzle** layout
  for mode 1001 so the user never hand-arranges it.
- **Coverage must distinguish "enum exists" from "library supports it."** Many
  scale modes are "Not supported yet," and support is additionally
  arch-dependent. The coverage harness enumerates all scale modes but marks the
  unsupported ones `xfail` / `skip` via a runtime probe, so the meta-test stays
  green while still tracking the full surface and flips to a real test
  automatically when support lands upstream.

## Error handling and correctness safeguards

Because the tool's purpose is trusting the numbers, the binding must never be a
source of silent wrongness:

- **Every hipBLASLt/HIP status is checked and raised** as a Python exception
  (`HipblasLtError` carrying the status enum name + call site). No swallowed
  error codes.
- **Shape/dtype/ld validation at the Python boundary** before calling C;
  mismatches raise a clear Python error rather than reaching the C API as a
  corrupt descriptor. Validation is only at the boundary — internal library
  behavior is trusted.
- **Explicit workspace sizing** — passing a workspace smaller than the chosen
  algo's `workspaceSize` is a clear Python-side error, not undefined behavior.
- **Owned DeviceArray lifetime** — the binding keeps references to input arrays
  for the duration of a call so Python GC cannot free a buffer mid-kernel; no
  dangling device pointers reach matmul.

## Testing strategy

- **Tier 1: surface coverage.** `test_api_coverage.py` parses the public enums
  from the headers (via `_coverage.py`) — `HIPBLASLT_MATMUL_DESC_*`,
  `EPILOGUE_*`, `hipblasLtMatmulMatrixScale_t` (scale modes), layout/preference
  attributes, supported `hipDataType` / compute types — to auto-generate the
  denominator, and asserts each value is referenced by the suite. CI-gated,
  target ~100%; goes red when upstream adds an enum. Values the library reports
  as unsupported (via runtime probe — e.g. "Not supported yet" scale modes, or
  arch-gated fp8 encodings) are enumerated but marked `xfail` / `skip` so the
  meta-test stays green while still tracking the full surface.
- **Tier 2: numerical correctness.** For each compute-path knob, a pytest
  comparing the GPU result against a numpy/scipy reference on a representative
  small problem, with dtype-appropriate tolerance. Parametrized over the
  dtype × transpose × epilogue × scale-mode matrix (representatively sampled,
  not exhaustive). MX cases build a block-scale tensor and apply it in the numpy
  reference so the comparison reflects MX math.
- **Encoding cross-check.** A test asserts `ml_dtypes` narrow-type encoding ==
  hipBLASLt converter encoding, bit-for-bit, across the fp8/fp6/fp4/e8m0 types
  both support. Divergence is a reportable finding, not a test-infra bug.
- **Known bugs as `xfail`.** Mirror the existing `known_bugs.yaml` concept with
  `pytest.mark.xfail(reason=...)` so known-bad algo/dtype combinations are
  documented, not hidden.
- **Hardware reality.** These tests require a GPU: they skip cleanly with a
  clear message when no device is present, and are excluded from pure-host CI
  that cannot allocate device memory.

Explicitly **not** used: Python line-coverage (coverage.py) of the thin binding
— for a pass-through binding it is near-meaningless (100% line coverage of the
wrapper says nothing about whether hipBLASLt behaved correctly). C-level gcov of
libhipblaslt is meaningful but heavyweight and blind to GPU kernels; out of
scope for v1.

## Open questions / deferred to implementation plan

- Exact name of the opt-in `invoke build` flag (e.g. `--python` / `-p`) and its
  plumbing through `tasks.py` / cmake; CI jobs; wheel strategy per
  Python × ROCm combination.
- Exact `HeuristicResult` field surface and `Algo` identifier representation.
- Minimum `ml_dtypes` version pin (fp6/fp4/e8m0 are recent additions) and the
  graceful-degradation path when an installed version lacks a narrow type.
- Representation of the block-scale tensor on `DeviceArray` and the exact
  pre-swizzle layout for `BLK32_UE8M0_32_8_EXT` (mode 1001).
