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

A new self-contained package under `projects/hipblaslt/python/` (proposed name
`pyhipblaslt`; name is a bikeshed to settle later). It is a nanobind C++
extension built with scikit-build-core, using `tensilelite/rocisa` as the
structural template but as its own independent package. Detailed build wiring
(integration with `invoke build`, CI, wheel strategy) is deferred to the
implementation plan.

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
├── pyhipblaslt/
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
3. **`pyhipblaslt/__init__.py`** (pure Python) — thin re-exports plus the
   ancillary convenience `matmul(a, b)` shim, built strictly on top of layer 1.

Layers 1 and 2 are the product; layer 3 is a convenience skin that adds no
capability the lower layers lack, so nothing being debugged is ever hidden.

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

**Dtype handling — two tiers:**

- **Tier 1 — numpy-native / `ml_dtypes`** (f32, f64, f16, i32, i8, plus bf16 and
  some fp8 via `ml_dtypes`): `from_numpy` maps the numpy/`ml_dtypes` dtype →
  `hipDataType` and copies bytes. `ml_dtypes` is an approved dependency — it is
  the de-facto standard (JAX/TF) and gives real narrow-type host representations
  rather than f32 stand-ins, shrinking Tier 2.
- **Tier 2 — types numpy/`ml_dtypes` cannot represent** (fp6, fp4, e8m0 scales,
  xfloat32): the `DeviceArray` holds packed bytes tagged with a `pyhipblaslt`
  dtype enum; the library treats them as opaque device bytes. Explicit
  conversion helpers (`pack_*` / `unpack_*`) do the bit-level encode/decode in
  C++ **by reusing hipBLASLt's own conversion routines** (from
  `hipblaslt_float8.h` etc.), so the Python encode matches what the library uses
  internally bit-for-bit — critical for correctness work. Sub-byte types
  (fp4/fp6) carry packing/layout subtleties that the helpers own so the user
  never hand-packs nibbles.

**Reference-precision stance.** Host reference math is computed at *widened*
precision (e.g. f32) and compared against the narrow GPU result with an
appropriate tolerance, because a correctness investigation wants "is the GPU
result within expected error of the true math," not "does it match another
low-precision computation."

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
  `EPILOGUE_*`, layout/preference attributes, supported `hipDataType` / compute
  types — to auto-generate the denominator, and asserts each value is referenced
  by the suite. CI-gated, target ~100%; goes red when upstream adds an enum.
- **Tier 2: numerical correctness.** For each compute-path knob, a pytest
  comparing the GPU result against a numpy/scipy reference on a representative
  small problem, with dtype-appropriate tolerance. Parametrized over the
  dtype × transpose × epilogue matrix (representatively sampled, not
  exhaustive).
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

- Final package name (`pyhipblaslt` vs. alternatives).
- Build integration specifics: wiring into `invoke build`, CI jobs, wheel
  strategy per Python × ROCm combination. (The user has opinions here to capture
  in the plan.)
- Exact `HeuristicResult` field surface and `Algo` identifier representation.
- Which specific fp8 variants come from `ml_dtypes` vs. Tier-2 helpers.
