# rocKE tests

One by-layer, language-agnostic test tree for the rocKE engine. A layer dir
holds that layer's Python and C++/cross-language tests together. All paths are
derived relative to each file, so this tree is copy-able verbatim.

## Entry point

```
python tests/run_all.py            # relative-path guard + byte-identity gate + pytest (+ctest if built)
python tests/run_all.py --only gemm
python tools/check_byte_identity.py   # build engine fresh + byte-identity gate (llvm20/llvm22)
```

`conftest.py` puts `rocke/platform/python` on `sys.path` (so `import rocke` works);
`pytest.ini` uses `--import-mode=importlib` so same-named test modules coexist
across layers without `__init__.py`.

## Before merging: source backends and installed CI

Run both lanes below when changing backend selection, rejection tests, test
dependencies, or packaging. A source-tree pytest pass does not validate the
installed test artifact. `run_all.py` builds native code and can expose a binding
that the CI test environment does not have.

### Source backend matrix

Use a fresh virtualenv outside the platform tree. Install the test requirements
from the CI job's TheRock revision; keep Torch out of the minimal lane. Build a
fresh native extension with `ROCKE_BUILD_PYBIND=ON` for the binding-present rows.
Run each row in a separate process, with `PYTHONPATH` containing the platform
Python root and, only where indicated, the matching `cpp/bindings` build directory.

| Lane | `ROCKE_BACKEND` | `ROCKE_CPP_STRICT` | `rocke_engine` importable? | Rejection observed by the caller |
|---|---|---|---|---|
| Default installed fallback | unset | unset | No | Python exception after the C++ import fails |
| Explicit Python | `python` | unset | Either | Python exception |
| Permissive C++ | `cpp` | `0` | Yes | Python exception after native rejection |
| Strict C++ | `cpp` | `1` | Yes | Native exception; fallback is an error |
| Differential | `both` | `1` | Yes | Python runs first; its rejection prevents a native comparison |

For rejection-contract changes, run these affected modules under every row,
then the complete platform pytest suite in the default missing-binding lane:

```text
python -m pytest tests/test_rocke.py tests/core/test_gfx1250_scaled_wmma.py tests/core/test_storage.py -q -rs
python -m pytest tests -v -rs --timeout=60
```

Verify imports before each lane: print `rocke.__file__`, and use
`importlib.util.find_spec("rocke_engine")` to check binding absence; in the native
lanes import `rocke_engine` and print its `__file__`. Clear inherited backend,
fixture, and runtime overrides before configuring a lane. In particular, a
developer's `ROCKE_CPP_STRICT=1` must not leak into the default fallback run.
Use a unique temporary directory for each validation run (`TMPDIR` on POSIX,
`TEMP`/`TMP` on Windows). Some tests discover native artifacts under the system
temporary directory; an old `rocke_online` or `rocke_verify` build can otherwise
change which tests run and which library they load.

Native rejection tests must account for strict mode as well as the requested
backend. `resolve_backend() == "cpp"` does not establish which engine produced
the exception. Preserve the exact exception type and diagnostic for each mode;
accepting a tuple of unrelated exceptions would hide a broken dispatch contract.
The missing-binding regression tests deliberately block the import even when a
native extension is available.

Changes to emitted IR still require the byte-identity and golden checks, and
kernel changes require the relevant GPU numerical tests. This matrix adds
fallback coverage; it does not replace those checks.

### Installed artifact replay

Start from the failing job's **Print test reproduction command**. It pins the
artifact run, GPU family, test script, requirements files, and tier. TheRock's
`build_tools/github_actions/reproduce_test_failure.py` downloads and tests that
artifact; replaying it reproduces the old result, not an unbuilt local fix.
Validate a fix by rebuilding and packaging the changed revision with the same
settings, then testing its relocated artifact.

Run the component test script's setup as well as its CTest command. The
hipkernelprovider script installs the artifact's `rocke` and `rocke_library`
wheels with `--no-deps --reinstall` before running CTest. Those wheels are part
of the tested configuration: library imports in child processes can depend on
the installed `rocke_library` wheel even when the parent pytest process finds
the staged library through `conftest.py`. Building only the standalone CMake
install does not reproduce this setup. Use wheels built from the same revision
as the artifact, never an editable install or a wheel from an older build.

For example, [job 110907294612](https://github.com/ROCm/rocm-libraries/actions/runs/37019571757/job/110907294612)
used this selection after artifact setup:

```text
ctest -L ^standard$ -LE ex_gpu --output-on-failure --parallel 1 --timeout 7200 --test-dir build/bin/hip_kernel_provider -V --tests-information 1,,1
```

Before execution, use `ctest --test-dir build/bin/hip_kernel_provider -N -V`
with the same selection flags to check the selected names, working directory,
commands, and environment. `ROCKE_ENGINE_test_categories_external.yaml` in the
provider and TheRock's categorization script determine tier membership. A
passing command that selected zero tests is not validation. In particular,
check that `rocke_pytest` is selected when it is part of the intended coverage.

That entry runs from `build/bin/hip_kernel_provider`, with `PYTHONPATH=.`:

```text
python -m pytest ./tests --ignore ./tests/library -v -rs --timeout=60
```

Use a clean runtime environment with no editable source installs, no binding
build directory on `PYTHONPATH`, and no inherited native fixture paths. Verify
that `rocke.__file__` is under the relocated artifact and `rocke_engine` is absent.
Start from an unexecuted install when relocating it; omit `__pycache__` and
pytest caches so cached code objects cannot retain paths to the original tree.
Do not add a source directory to repair an installed import failure: install the
required module or data through CMake and the artifact manifest. Include imported
runner helpers even when their developer entry point is not executed by CI.

Record source SHA, artifact run, Python/dependency versions, LLVM/ROCm versions,
backend variables, selected CTest names, and pytest failures/skips. Compare the
same environment before and after the fix. Use CI-matched LLVM tools for object
tests; an older local compiler failure is separate evidence. Review skipped
tests explicitly: Torch-free execution, missing native fixtures, and unavailable
GPUs each leave different coverage gaps. Host pytest is not a full GPU job replay.

## Layout / coverage matrix

This table is an **inventory** of what lives where. It does *not* imply every
entry runs in the default runner -- see [Execution categories](#execution-categories)
for what is actually exercised vs. manual/diagnostic/demo.

| Layer | Python (pytest-collected) | C++ / cross-language |
|---|---|---|
| `core/` | `test_ir_serialize.py`, `dsl_optimization/` (constant-fold, unroll, barrier) | `ir_serialize_roundtrip.cpp` (**CTest**); `smoke.cpp` (optional build-only target, **not** registered); `ir_lower_cli.cpp` (manual CLI tool) |
| `helpers/` | (covered today via `test_rocke.py` TestHelpers; dedicated split is a follow-up) | - |
| `instances/` | `test_rocke_multiarch.py`, `test_gfx1250_*`, `test_moe_*`, `test_wmma_schedule.py`, `test_rocke_gfx950_smoke.py` | `tiled_attention_2d_reentrancy.cpp` (**CTest**); `parity/` 65 `*_emit.py`/`*_emit.c` pairs + `run_parity.py` (driven by the byte-identity gate, **not** pytest); `differential/` drivers `run_diff.py`, `fuzz_diff.py`, `ir_artifact_diff.py`, `numeric.py`; `jit_demo.cpp`, `gemm_jit_demo.cpp` (manual demos, **not** built/registered) |
| `runtime/` | (covered via `test_rocke.py`; dedicated split is a follow-up) | - |
| `dispatch/` | `dispatch_tests/{gemm,attention,conv,moe,norm}` | - |
| `analysis/` | (covered via `test_rocke.py`) | - |
| (root) | `test_rocke.py` (multi-layer monolith), `test_rocke_ci_static.py` | - |

> Notes: `instances/rocke_ir_parity_harness.py` is a helper imported by the
> gate, not a `test_`-prefixed pytest module. The pybind **`rocke_engine` module
> has no default-pytest coverage**: the C-engine vs Python-engine equivalence is
> owned by the byte-identity gate (which builds `rocke_core` and byte-compares the
> 65 `*_emit.c` outputs to `*_emit.py`), and the binding itself by the
> consistency proof documented in `cpp/bindings/README.md`.

## Execution categories

Four distinct things run here; don't conflate them:

1. **Default runner** (`python tests/run_all.py`): relative-path guard -> byte-identity
   gate (`tools/check_byte_identity.py`) -> `pytest` (the `test_*.py` modules
   above) -> `ctest` when any registered test executable is built in `--build-root`
   for the selected `--config`. CTest supplies the executable paths, including
   configuration directories and platform suffixes. The entire registered suite
   runs, so a partial build exposes missing tests as failures. No configured test
   build or no built registered tests is reported explicitly as a skipped stage.
2. **Diagnostics** (opt-in, not in the gate): `run_diff.py --ir` (IR-canonical
   diff), `fuzz_diff.py`, `ir_artifact_diff.py`.
3. **GPU / manual numeric lanes** (need a HIP device; skipped/not-collected
   otherwise): `differential/numeric.py`, `instances/test_rocke_numeric.py`.
4. **Manual C++ demos / tools** (not built or registered by CMake/CTest --
   compile by hand against `librocke_core.a`): `core/ir_lower_cli.cpp`,
   `instances/jit_demo.cpp`, `instances/gemm_jit_demo.cpp`, plus the build-only
   `core/smoke.cpp`.

## Multi-arch coverage (don't be blindsided by gfx950)

Byte-identity is a property of a single `(spec, arch)`: for the same spec and
arch the Python and C++ engines must produce the same output, **including both
rejecting** an unsupported `(spec, arch)` (the harness counts "both reject" as
parity-faithful, SKIP). So arch coverage is just a matter of which `(spec, arch)`
pairs the emitters enumerate - there is no global arch override.

- Of the 65 `*_emit` families, 16 are arch-prefixed: 12 cover gfx942/RDNA
  directly (`gfx942_*` x3, `gfx1151_*` x6, `gfx1201_*` x3) and 4 are `gfx950_*`;
  each such emitter config returns `(spec, arch)`. The remaining ~49 common
  families default to gfx950.
- The common families default to `gfx950`. To cover a common family on another
  arch, add a `(spec, arch="gfx942")` (etc.) config to that family's
  `*_emit.py` and `*_emit.c` - the normal gate (`run_diff` /
  `check_byte_identity.py`) then exercises it at that arch with no extra
  machinery. A config that is invalid on the chosen arch (e.g. a wave32 WMMA
  spec on CDNA gfx942) is rejected by **both** engines and counted SKIP.

(Earlier revisions had a `ROCKE_PARITY_ARCH` env that re-targeted common configs
onto another arch. It was removed: it is not needed for the parity contract and
it produced false mismatches when it forced an arch different from the one a
config pinned. The conv WMMA "finding" it surfaced was that artifact, not a real
divergence - the Python builder correctly rejects wave32 WMMA on gfx942.)

## Dedup / audit decisions

- REMOVED (duplicate of the differential gate): `run_gemm_parity.sh` and
  `run_ir_serialize_parity.sh` - the `gemm` and `ir_serialize` families are
  covered by `run_diff.py` (`--mode ll` / `--mode ir`) and
  `tools/check_byte_identity.py`. The micro-kernel harness was ported to
  `parity/run_parity.py` (cross-platform).
- NOT duplicates (kept): the 65 `*_emit.py` / `*_emit.c` pairs are the two
  oracles of the differential gate; `core/test_ir_serialize.py` (Python) and
  `core/ir_serialize_roundtrip.cpp` (C++) each validate their own engine.
- OVERLAP (consolidation is a tracked follow-up, not done here because it needs
  GPU validation): `instances/test_rocke_numeric.py` and
  `instances/differential/numeric.py` are both Python-engine on-GPU numeric
  lanes. Canonical lane: `differential/numeric.py` (parametrized L6). Fold the
  unique cases from `test_rocke_numeric.py` (rdna core parity, wmma_gemm) in,
  then drop the wrapper, validated on a GPU node.
- MISSING in C++ (tracked follow-up): no C-engine on-GPU numeric lane - L6 runs
  the Python engine only. Add one (compile C-emitted `.ll` -> HSACO -> launch ->
  compare) or extend `numeric.py` via the `rocke_engine` binding.
- EXCLUDED from rocKE: `test_gen_instances.py` (imports `ck4inductor`, a separate
  package) and `test_rocke_examples.py` (drives the external `example/ck_tile/dsl`
  tree, not part of rocKE) stay in `composablekernel/python/test`.

### Native storage parity in the standard runner

`run_all.py --build-root <build>` builds all configured targets before pytest,
then obtains the `rocke_storage` executable path from CTest. Both pytest passes
receive that path, so storage IR/HIP parity and serialization tests run automatically.
`--config` selects the native test configuration (default `Release`). A build or
fixture-discovery failure stops the runner instead of silently skipping coverage.

An explicit `ROCKE_STORAGE_TEST` overrides discovery and must name an existing
executable; it does not skip the configured build. With no configured build or
explicit override, the runner reports native storage parity as skipped; direct
pytest invocations can use the same override.
