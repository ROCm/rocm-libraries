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

The default L3 byte-identity gate (`run_diff.py` / `check_byte_identity.py`)
compiles each `*_emit.c` at its hardcoded arch: the 16 arch-prefixed families
(`gfx942_*`, `gfx1151_*`, `gfx1201_*`) cover those archs, while the ~48 common
families run at `gfx950`.

To also exercise the common families on other supported archs there is a
**forced-arch IR-artifact sweep**: `_emit_common.run_emit` honors
`CKC_PARITY_ARCH`, so the Python engine builds+lowers a common family at any
arch, and the C engine lowers the same serialized IR at that arch via
`ir_lower_cli <arch>` - a genuine Python@arch vs C@arch byte-identity check with
no per-emitter edits.

```bash
# common families at the CDNA peer gfx942 (built engine + ir_lower_cli auto-built)
python tests/run_all.py --no-gate --no-pytest --arch-sweep gfx942
# or directly:
python tests/instances/differential/ir_artifact_diff.py --cli <ir_lower_cli> --arch gfx942
```

Result: gfx942 is GREEN - 491 common-family configs byte-identical, 0 drift, 0
error (arch-inapplicable configs SKIP). Arch-prefixed families are excluded from
the sweep (already run at their own arch).

**Which archs are meaningful to force?** The ~48 common families are CDNA specs
(wave64 + MFMA), so the forced sweep is meaningful only for **CDNA** targets
(`gfx942`, baseline `gfx950`). Forcing them onto an **RDNA** arch (`gfx1151`,
`gfx1201`, wave32 + WMMA) is garbage-in - the spec is semantically invalid there
and the two engines diverge for uninteresting reasons, so do NOT force common
families onto RDNA. RDNA byte-identity is covered by the 16 arch-prefixed
families (`gfx1151_*`, `gfx1201_*`, `gfx942_*`) that already run in the default
gate and are GREEN at both llvm flavors. A config the C engine declines as
not-applicable-to-this-arch (e.g. a WMMA op forced onto CDNA) is classified SKIP.

Finding surfaced by this (pre-existing, not a migration regression, tracked as a
follow-up): `conv_implicit_gemm` WMMA configs forced onto gfx942 are correctly
rejected by the C engine ("WMMA not available on gfx942") but the Python engine
emits anyway - a Python validator-permissiveness divergence to fix at the source.

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
