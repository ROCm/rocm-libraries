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

`conftest.py` puts `rocKE/Python` on `sys.path` (so `import ck_dsl` works);
`pytest.ini` uses `--import-mode=importlib` so same-named test modules coexist
across layers without `__init__.py`.

## Layout / coverage matrix

| Layer | Python | C++ / cross-language |
|---|---|---|
| `core/` | `test_ir_serialize.py`, `dsl_optimization/` (constant-fold, unroll, barrier), `test_ck_dsl_c_interface.py` (ckc_engine binding) | `smoke.cpp`, `ir_serialize_roundtrip.cpp`, `ir_lower_cli.cpp` |
| `helpers/` | (covered today via `test_ck_dsl.py` TestHelpers; dedicated split is a follow-up) | - |
| `instances/` | `test_ck_dsl_multiarch.py`, `test_gfx1250_*`, `test_moe_*`, `test_wmma_schedule.py`, `test_ck_dsl_gfx950_smoke.py`, `ck_dsl_ir_parity_harness.py` | `parity/` (65 `*_emit.py`/`*_emit.c` pairs + `run_parity.py`), `differential/` (`run_diff.py`, `fuzz_diff.py`, `ir_artifact_diff.py`, `numeric.py`), `tiled_attention_2d_reentrancy.cpp`, `jit_demo.cpp`, `gemm_jit_demo.cpp` |
| `runtime/` | (covered via `test_ck_dsl.py`; dedicated split is a follow-up) | - |
| `dispatch/` | `dispatch_tests/{gemm,attention,conv,moe,norm}` | - |
| `analysis/` | (covered via `test_ck_dsl.py`) | - |
| (root) | `test_ck_dsl.py` (multi-layer monolith), `test_ck_dsl_ci_static.py` | - |

## Multi-arch coverage (don't be blindsided by gfx950)

Byte-identity is a property of a single `(spec, arch)`: for the same spec and
arch the Python and C++ engines must produce the same output, **including both
rejecting** an unsupported `(spec, arch)` (the harness counts "both reject" as
parity-faithful, SKIP). So arch coverage is just a matter of which `(spec, arch)`
pairs the emitters enumerate - there is no global arch override.

- The 16 arch-prefixed families (`gfx942_*`, `gfx1151_*`, `gfx1201_*`) cover those
  archs directly; each emitter config returns `(spec, arch)`.
- The common families default to `gfx950`. To cover a common family on another
  arch, add a `(spec, arch="gfx942")` (etc.) config to that family's
  `*_emit.py` and `*_emit.c` - the normal gate (`run_diff` /
  `check_byte_identity.py`) then exercises it at that arch with no extra
  machinery. A config that is invalid on the chosen arch (e.g. a wave32 WMMA
  spec on CDNA gfx942) is rejected by **both** engines and counted SKIP.

(Earlier revisions had a `CKC_PARITY_ARCH` env that re-targeted common configs
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
  GPU validation): `instances/test_ck_dsl_numeric.py` and
  `instances/differential/numeric.py` are both Python-engine on-GPU numeric
  lanes. Canonical lane: `differential/numeric.py` (parametrized L6). Fold the
  unique cases from `test_ck_dsl_numeric.py` (rdna core parity, wmma_gemm) in,
  then drop the wrapper, validated on a GPU node.
- MISSING in C++ (tracked follow-up): no C-engine on-GPU numeric lane - L6 runs
  the Python engine only. Add one (compile C-emitted `.ll` -> HSACO -> launch ->
  compare) or extend `numeric.py` via the `ckc_engine` binding.
- EXCLUDED from rocKE: `test_gen_instances.py` (imports `ck4inductor`, a separate
  package) and `test_ck_dsl_examples.py` (drives the external `example/ck_tile/dsl`
  tree, not part of rocKE) stay in `composablekernel/python/test`.
