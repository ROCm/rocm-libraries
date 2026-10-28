# rocKE tests

One by-layer, language-agnostic test tree for the rocKE engine. A layer dir
holds that layer's Python and C++/cross-language tests together. All paths are
derived relative to each file, so this tree is copy-able verbatim.

## Entry point

Install test dependencies from the platform directory with
`python -m pip install -r requirements.txt` or `python -m pip install -e '.[dev]'`.
Both include `ml_dtypes>=0.6.0`, which supplies independent FP4/FP6/FP8 reference
types. Host tests that need this optional runtime dependency use
`pytest.importorskip`; a minimal authoring installation can run the remaining
tests. Required gfx1250 numerical validation needs `ml_dtypes` installed and
`ROCKE_REQUIRE_GFX1250=1`; missing dependencies or the requested GPU must fail
that validation rather than count as successful coverage.

```
python tests/run_all.py              # complete developer pre-merge gate; see stages below
python tests/run_all.py --only gemm   # scopes byte identity only, not pytest/CTest
ROCKE_LLVM_FLAVOR=llvm20 python tools/check_byte_identity.py   # llvm20
ROCKE_LLVM_FLAVOR=llvm22 python tools/check_byte_identity.py   # llvm22
ROCKE_LLVM_FLAVOR=llvm23 python tools/check_byte_identity.py   # llvm23
```

`conftest.py` puts `rocke/platform/python` on `sys.path` (so `import rocke` works);
`pytest.ini` uses `--import-mode=importlib` so same-named test modules coexist
across layers without `__init__.py`.

On Linux, `test_compiler_detection_native.py` compiles its wrapper and fake COMGR
libraries from stdin and links the production `librocke_core.a`. Installed tests
carry that archive beside the test module, including in the test artifact. Source
runs build it in pytest's temporary directory, or use a matching fresh archive
specified by `ROCKE_TEST_ENGINE_ARCHIVE`. These fixtures require C/C++ compilers;
source builds also require CMake. They do not need a GPU or private source headers.

See [the pre-merge testing strategy](../../TESTING.md#51-before-merging-source-backends-and-installed-ci)
for the required source backend matrix and installed CI replay procedure.

## Layout / coverage matrix

This table is an **inventory** of what lives where. It does *not* imply every
entry runs in the default runner -- see [Execution categories](#execution-categories)
for what is actually exercised vs. manual/diagnostic/demo.

| Layer | Python (pytest-collected) | C++ / cross-language |
|---|---|---|
| `core/` | `test_ir_serialize.py`, `dsl_optimization/` (constant-fold, unroll, barrier) | `ir_serialize_roundtrip.cpp` (**CTest**); `smoke.cpp` (optional build-only target, **not** registered); `ir_lower_cli.cpp` (manual CLI tool) |
| `helpers/` | (covered today via `test_rocke.py` TestHelpers; dedicated split is a follow-up) | - |
| `instances/` | `test_rocke_multiarch.py`, `test_gfx1250_*`, `test_moe_*`, `test_wmma_schedule.py`, `test_rocke_gfx950_smoke.py` | `tiled_attention_2d_reentrancy.cpp` (**CTest**); `parity/` 47 `*_emit.py`/`*_emit.c` pairs + `run_parity.py` (driven by the byte-identity gate, **not** pytest); `differential/` drivers `run_diff.py`, `fuzz_diff.py`, `ir_artifact_diff.py`, `numeric.py`; `jit_demo.cpp`, `gemm_jit_demo.cpp` (manual demos, **not** built/registered) |
| `runtime/` | (covered via `test_rocke.py`; dedicated split is a follow-up) | - |
| `dispatch/` | `dispatch_tests/{gemm,attention,conv,moe,norm}` | - |
| `analysis/` | (covered via `test_rocke.py`) | - |
| `tools/` | `test_arch_domain_artifact.py` (source-only structure/generator units), `test_run_all.py`, tool-sidecar units | `rocke_arch_domain_reproducibility` (**CTest**, artifact reproduction through the host toolchain) |
| (root) | `test_rocke.py` (multi-layer monolith), `test_rocke_ci_static.py` | - |

> Notes: `instances/rocke_ir_parity_harness.py` is a helper imported by the
> gate, not a `test_`-prefixed pytest module. The pybind **`rocke_engine` module
> has no default-pytest coverage**: the C-engine vs Python-engine equivalence is
> owned by the byte-identity gate (which builds `rocke_core` and byte-compares the
> 81 matched `*_emit.c`/`*_emit.py` pairs), and the binding itself by the
> consistency proof documented in `cpp/bindings/README.md`.

## Execution categories

Four distinct things run here; don't conflate them:

1. **Default developer pre-merge runner** (`python tests/run_all.py`):
   relative-path guard -> byte-identity gate (`tools/check_byte_identity.py`) ->
   emitted-IR validity (`tools/check_ir_validity.py`) -> arch-domain warning gate
   (`tools/check_arch_domain.py`) -> platform pytest -> `ROCKE_BACKEND=both`
   differential pytest when `rocke_engine` is importable -> `ctest` when any
   registered native test executable is built in `--build-root` for the selected
   `--config`. CTest supplies executable
   paths, including configuration directories and platform suffixes. The entire
   registered suite runs, including the build-tree
   `rocke_arch_domain_reproducibility` toolchain gate, so a partial build exposes
   missing tests as failures. This is the complete pre-merge gate, not a fast
   smoke; the `standard` label controls installed CI selection and does not filter
   this local CTest phase. If `rocke_engine` is unavailable, the differential
   pytest stage reports a skip and GREEN does not imply that it ran. No configured
   test build or no built registered native tests is reported explicitly as a
   skipped stage.
2. **Diagnostics** (opt-in, not in the gate): `run_diff.py --ir` (IR-canonical
   diff), `fuzz_diff.py`, `ir_artifact_diff.py`.
3. **GPU / manual numeric lanes** (need a HIP device; skipped/not-collected
   otherwise): `differential/numeric.py`, `instances/test_rocke_numeric.py`.
4. **Manual C++ demos / tools** (not built or registered by CMake/CTest --
   compile by hand against `librocke_core.a`): `core/ir_lower_cli.cpp`,
   `instances/jit_demo.cpp`, `instances/gemm_jit_demo.cpp`, plus the build-only
   `core/smoke.cpp`.

### Arch-domain reproducibility CTest

`rocke_arch_domain_reproducibility` owns the full
`tools/gen_arch_domain.py --check` sweep. It is deliberately not collected by
pytest: the sweep starts external compiler/linker processes, has its own timeout
and resource contract, and is independent of `ROCKE_BACKEND`.

The same logical gate has two registrations: the build-tree entry invokes the
source tool and committed source column, while the installed entry invokes a
packaged test-only copy of the tool and the column in the relocated `rocke`
package. This matches the library-side CI model: CMake stages every input, CTest
owns execution, and no test reconstructs or reaches back into the source tree.

Reproduction uses the committed column only to schedule work, never as its
verdict. Declaration recognition is batched through `opt` and recursively split
on failure. Each compiler/architecture pair receives an intrinsic-free target
preflight. Calls for expected-positive cells are combined per architecture; a
failed module is recursively split until ordinary single-cell probes identify
every changed result. Expected-negative cells remain individual, and every
positive call still passes through compile, link, and target assembly. Special
integer and `immarg` operands come from explicit `ProbeSpec` witnesses rather
than speculative value sweeps. Target preflights, positive batches, and
negative singletons share a quota-aware worker pool capped at 32 by default;
`--jobs` is the explicit override for constrained or dedicated hosts. An
expected crash or timeout whose normalized evidence still matches is not rerun
serially; changed evidence, changed verdicts, unclassified results, and otherwise
transient-looking results are.

The migration preserves the former pytest item's result mapping:

| Condition | CTest result |
|---|---|
| Committed measurements reproduce cell-for-cell (compiler-build provenance may differ) | Pass |
| No clang | Skip |
| Clang flavor differs from rocKE's resolved flavor | Skip |
| No committed column for the host flavor | Skip |
| Clang exists but `opt` is absent | Fail |
| Clang and `opt` report different flavors | Fail |
| Flavor auto-detection fails, including unavailable COMGR evidence | Fail |
| Measurements drift or a probe is unclassified | Fail |

Run it from a configured platform build with:

```bash
ctest --test-dir <build> -R '^rocke_arch_domain_reproducibility$' -V
```

#### When kernel authors need this gate

Ordinary kernel work should run the complete `tests/run_all.py` pre-merge gate;
when its native build is ready, that runner includes this CTest. Run the focused
command above during iteration when changing intrinsic declarations,
architecture-domain data or generation, LLVM discovery/classification, or this
test's CMake/install wiring. Inspect a named skip rather than counting it as
measurement validation.

The runner has two independent selectors, neither of which scopes pytest or
CTest:

| Change or intent | Focused route |
|---|---|
| One parity emitter family | `python tests/run_all.py --only <family>`; scopes byte identity only |
| One representative emitted-IR corpus case | `python tests/run_all.py --only-ir <family/arch/variant>`; scopes IR validity and the arch-domain warning gate only |
| One Python behavior | invoke its pytest module or node ID directly |
| Arch-domain artifact reproduction | use the focused `ctest -R` command above |
| Full pre-merge validation | run `python tests/run_all.py` with no selectors |

#### CI enrollment and known limitations

This move exposes the sweep as one independently schedulable CTest in both local
builds and installed provider CI.

- The installed test package carries only the generator and its test-time Python
  dependencies; they remain test assets rather than production `rocke` APIs.
- The installed gate reads the committed column through package resources, so it
  verifies the exact relocated data consumed by CI. The build-tree gate resolves
  the same resource from the source package.
- Standard-tier categorization keeps the external compiler sweep out of quick and
  FFM installed-CI lanes. The local developer runner invokes unfiltered CTest and
  therefore includes it. Registration itself is pinned by the source runner
  contract test; only a selected, non-skipped CI result supplies reproduction
  evidence.
- Exit code 77 is an orchestration adapter preserving the old pytest skip
  behavior, not yet a typed semantic outcome from the generator.
- Flavor auto-detection can fail before a skip result when COMGR evidence is
  unavailable; this preserves existing behavior for the migration.
- Probe batching, worker tuning, caching, and timeout/retry policy remain future
  efficiency work now that CI supplies execution evidence for the standalone gate.

## Multi-arch coverage (don't be blindsided by gfx950)

Byte-identity is a property of a single `(spec, arch)`: for the same spec and
arch the Python and C++ engines must produce the same output, **including both
rejecting** an unsupported `(spec, arch)` (the harness counts "both reject" as
parity-faithful, SKIP). So arch coverage is just a matter of which `(spec, arch)`
pairs the emitters enumerate - there is no global arch override.

- Of the 81 matched `*_emit` families, 20 are arch-prefixed; each such emitter
  config returns `(spec, arch)`. The remaining 61 common families default to
  gfx950. These counts are an inventory snapshot; the emitter configs, not the
  totals, are authoritative.
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
- NOT duplicates (kept): the 81 matched `*_emit.py` / `*_emit.c` pairs are the two
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

### Native parity binaries in the default developer runner

`run_all.py --build-root <build>` builds all configured targets before pytest,
then obtains each native parity executable from CTest, as listed in
`NATIVE_PARITY_TESTS`: `rocke_storage` (`ROCKE_STORAGE_TEST`),
`rocke_ir_types` (`ROCKE_IR_TYPES_TEST`), `rocke_nontemporal_hip`
(`ROCKE_NONTEMPORAL_HIP_TEST`) and `rocke_unsigned_int_ops`
(`ROCKE_UNSIGNED_INT_OPS_CLI`). Both pytest passes receive those paths, so the
storage IR/HIP parity, serialization, C++ builder type-rule parity, nontemporal
HIP-source parity and unsigned integer op C API parity tests run automatically.
The installed `rocke_pytest` ctest sets `ROCKE_UNSIGNED_INT_OPS_CLI` itself.
`--config` selects the native test configuration (default `Release`). A build
or fixture-discovery failure stops the runner instead of silently skipping
coverage.

An explicit `ROCKE_STORAGE_TEST` / `ROCKE_IR_TYPES_TEST` /
`ROCKE_NONTEMPORAL_HIP_TEST` / `ROCKE_UNSIGNED_INT_OPS_CLI` overrides discovery
for that binary and must name an existing executable; it does not skip the
configured build. With no configured build or explicit override, the runner
reports that parity as skipped; direct pytest invocations can use the same
overrides.

### Optimization-barrier coverage

The standalone `rocke_optimization_barrier` CTest checks native scalar admission,
rejection, serialization round trips, and LLVM lowering. It is installed with the
other native tests and does not require a pytest executable-discovery fixture.

`core/test_optimization_barrier.py` checks the Python builder and serialized LLVM
output. The dedicated Python/C parity emitters cover LLVM, canonical IR, and
verification; committed golden entries anchor the existing configurations.
`instances/test_optimization_barrier_numeric.py` compiles through COMGR and
checks scalar payloads, lane predicates, and intermediate rounding on a GPU.
Native HIP-source lowering is outside this feature's supported path.

## Installed pinned-reference suites

These suites are registered by platform CMake for the installed provider, with
library tests staged under `tests/library/tests/`. They use NumPy and ROCm in an
environment without Torch; they do not invoke the offline CPU oracles.

| CTest entry | Scope | Registration |
|---|---|---|
| `rocke_reference_common_pytest` | Shared numerical budgets, artifact integrity, worker isolation, and that each installed bundle records the enrolled cohort | Host suite; skips the cohort check for a bundle that is not installed |
| `rocke_sdpa_reference_unit_pytest` | SDPA numerical contract and qualification guards | Host suite; independent of bundles |
| `rocke_conv_reference_unit_pytest` | Convolution numerical contract and qualification guards | Host suite; independent of bundles |
| `rocke_sdpa_gpu_<arch>_pytest` | The SDPA cases enrolled for `<arch>` and three failure checks | For each published SDPA architecture, when its bundle is installed |
| `rocke_conv_gpu_<arch>_pytest` | The forward convolution cases enrolled for `<arch>` and three failure checks | For each published convolution architecture, when its bundle is installed |

`ROCKE_INSTALL_TEST_GPU_REFERENCES` controls installation of all published
operation/architecture pairs and is on by default in provider builds.
`platform/cmake/PublishedGpuReferences.cmake`, relative to the rocKE root, lists
the published pairs. From the rocKE root,
`python library/tests/run_<operation>_reference.py cases --arch <arch>` lists an
architecture's enrolled cases and their count without a GPU. CMake uses the
standard archives and committed locks; verify replacement candidates with the
offline CLI using explicit `--bundle` and `--lock` paths. Each operation and
architecture retains a separate archive, lock, and installed payload.

The optional `conv_reference/check_torch_reference.py` and
`sdpa_reference/check_torch_reference.py` are explicit offline oracle audits.
They are excluded from installation and normal pytest discovery.
For adding cases, replacing bundles, or enrolling targets, follow the
[authoritative reference workflow](../../library/tests/reference_common/docs/gpu-reference-workflow.md).
See [the installed reference procedure](../../TESTING.md#running-installed-reference-tests-without-torch)
and the [convolution guide](../../library/tests/conv_reference/docs/conv-test-reference.md) for commands,
publication requirements, and coverage limits.

The convolution gradient host contract joins `rocke_library_conv_pytest`.
`rocke_library_gpu_pytest` also runs the dgrad and wgrad cases in
`conv_reference/gradient_cases.py` on each of gfx942/gfx950 against live NumPy
oracles, with three executions per case.
These gradients are not pinned-reference replay. Their buffers, auxiliary
tables, and workspaces have explicit initialization and integrity checks.
See the [coverage inventory](../../docs/conv-coverage-roadmap.md) for explicit
versus production-selected configurations.
