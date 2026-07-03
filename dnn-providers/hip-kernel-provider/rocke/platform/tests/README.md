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

`conftest.py` puts `rocKE/Python` on `sys.path` (so `import rocke` works);
`pytest.ini` uses `--import-mode=importlib` so same-named test modules coexist
across layers without `__init__.py`.

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
> consistency proof documented in `Cpp/bindings/README.md`.

## Execution categories

Four distinct things run here; don't conflate them:

1. **Default runner** (`python tests/run_all.py`): relative-path guard -> byte-identity
   gate (`tools/check_byte_identity.py`) -> `pytest` (the `test_*.py` modules
   above) -> `ctest` **iff** the registered binaries (`rocke_ir_serialize_roundtrip`,
   `rocke_tiled_attention_2d_reentrancy`) are present in `--build-root`. This is the
   only set that is gated.
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
