# rocKE Layout Restructure: Final Structure (SDPA carve)

**Branch:** `users/bharriso/rocke-layout-restructure` (off latest `develop`,
includes the merged `rocke-client` skeleton #8928).
**Scope (locked):** wrap existing `rocKE/` under a `platform/` parent (untouched
internally), and extract only the SDPA/MHA vertical into a new `library/`.
Everything non-attention stays in platform. C++ engine and byte-identity parity
stay where they are (dormant until JIT: status, not a move).

Paths under `dnn-providers/hip-kernel-provider/` (**BASE**), fixed location for now.

> **Update (AICK-1737):** we decided to release rocKE as kernel bundles under
> `hip-kernel-provider`, so the `library/api/` C++ client has been **removed**.
> This doc drops the `api/` layer; the rest of the carve is unaffected.

---

## 1. Top-level final tree

```
BASE/
├── CMakeLists.txt                      # add_subdirectory(rocke)  (was rocke-client + rocKE)
└── rocke/                              # project parent
    ├── CMakeLists.txt                  # add_subdirectory(platform); library python via rocke-pyenv/rocke-wheels
    ├── BUILDING.md                     # build/dev guide (editable installs, PYTHONPATH, rocke.assets)
    ├── platform/                       # former rocKE/, reparented verbatim (only attention-unwiring edits)
    │   ├── CMakeLists.txt              # former rocKE/CMakeLists.txt
    │   ├── pyproject.toml  requirements.txt
    │   ├── AGENTS.md  BUILD.md  README.md
    │   ├── Cpp/                        # C++ engine + bindings (dormant until JIT; unchanged)
    │   ├── Python/
    │   │   └── rocke/                  # the SDK package (import name `rocke`, installable)
    │   │       ├── __init__.py         # EDIT: drop attention re-exports only
    │   │       ├── __main__.py
    │   │       ├── core/               # ir, lower_llvm/hip/cktile, passes, ir_serialize, verify, arch, isa, backend
    │   │       ├── helpers/            # all helpers stay (generic toolkit, incl. attention-infra primitives)
    │   │       ├── runtime/            # comgr, hip_module, launcher
    │   │       ├── analysis/  benchmark/
    │   │       ├── instances/          # all kernels EXCEPT attention  (EDIT __init__s: drop attention)
    │   │       ├── dispatch/           # all families EXCEPT attention  (EDIT __init__s: drop attention)
    │   │       ├── heuristics/         # stays (1 lazy attention import repointed to library)
    │   │       ├── examples/           # all examples EXCEPT gfx*/attention
    │   │       └── sweep.py sweep_bench.py run_manifest.py torch_backend.py
    │   ├── tests/                      # C++ tests + platform pytest (minus attention python tests)
    │   ├── tools/  cmake/  dsl_docs/
    └── library/                        # the SDPA/MHA product: build-time-only Python (relies on platform `rocke.*`)
        ├── pyproject.toml              # build-time metadata (library python is NOT installed into the product)
        ├── kernels/                    # was platform instances attention (the SDPA kernel defs)
        ├── builders/                   # was platform examples/gfx*/attention (build-time drivers)
        ├── dispatch/                   # was dispatch/families/attention (1-of-N SDPA select)
        ├── tests/                      # the python attention tests
        └── benchmarks/                 # per-arch attention benchmark drivers + shape data
# removed: BASE/rocKE/ and BASE/rocke-client/  (their content now under rocke/)
# removed (AICK-1737): BASE/rocke/library/api/  and  BASE/rocke/library/CMakeLists.txt
```

## 2. platform: what it is
The entire former `rocKE/`, reparented as `rocke/platform/`, internally unchanged
except the attention-unwiring edits (§4). It remains:
- the installable **SDK** (`rocke` package at `platform/Python/rocke`, via
  `platform/pyproject.toml`), usable standalone by others to build kernels;
- the home of `core`, all `helpers` (the generic toolkit, including
  attention-infra primitives `mfma_attention`, `mfma_attention_bwd`, `qk_scale`,
  `attention`, `rotary`; reusable building blocks), `runtime`, `analysis`,
  `benchmark`, and all non-attention `instances`/`dispatch`/`examples`,
  `heuristics`, the C++ engine (`Cpp/`), tools, docs, tests.

## 3. library: the SDPA/MHA carve (relies on platform helpers)
Build-time Python (NOT installed into the product). Self-contained SDPA kernels
that import the platform SDK as `rocke.*`.

| `library/` dir | Source (from platform) | Contents |
|---|---|---|
| `kernels/` | `instances/common/` attention + arch attention | `attention_unified`, `attention_arch`, `_fmha_common`, `_fmha_warp_body`, `fmha_{appendkv,arch,bwd,fwd_fp8,head_grouping,mfma,paged_prefill,splitkv_decode,varlen}`, `sage_attention`, `sparse_attention`; `gfx1151/wmma_fmha_fwd`; `gfx1250/{_wmma_attention_common,attention_tiled_2d,attention_tiled_3d,wmma_attention_fwd}`; `gfx942/{attention_tiled_2d,attention_tiled_3d}`; `gfx950/{attention_tiled_2d,attention_tiled_2d_fastkv_regp,attention_tiled_3d}`; plus the inline arch-router builders `build_unified_attention_2d_tiled` / `build_unified_attention_3d_tiled` lifted out of `instances/__init__.py` |
| `builders/` | `examples/gfx{942,950,1151,1250}/attention/`, `examples/common/fmha_fwd_verify_hip.py`, plus the attention halves split out of mixed `examples/common` drivers (`parity_fmha_extended` from `parity_extended_kernels`, `hip_lowering_attention_parity` from `hip_lowering_parity`), the `dsl_probe`/`gen_sweep`/`stage1_benchmark` attention slices (`dsl_probe_attention_demos`, `gen_sdpa_sweep_data`, `benchmark_rocke_unified_attention`), and `gfx950/qwen3_30b_a3b/` decode drivers | the SDPA verify/parity/bench drivers (build-time) |
| `dispatch/` | `dispatch/families/attention.py` | `ATTENTION_REGISTRY`, `AttentionRequest`, `dispatch_attention`: the 1-of-N SDPA select |
| `tests/` | `tests/instances/test_gfx1250_attention.py` (+ any other python attention tests) | python attention tests (C++ `tiled_attention_2d_reentrancy.cpp` stays in platform; it tests the C++ engine) |

**Naming:** `kernels` (was *instances*) and `builders` (was *examples*) are
product nouns; platform keeps its DSL terms `instances`/`examples`.

## 4. The only platform edits (attention-unwiring)
Removing attention from the `instances` package requires editing its re-export
sites (all mechanical "drop/relocate attention"):
- `instances/__init__.py`: strip the attention re-export block (`UnifiedAttention*`, `build_unified_attention_*`) and **move** the inline `build_unified_attention_2d_tiled`/`_3d_tiled` arch-routers to `library/kernels/`.
- `instances/gfx942/__init__.py`, `instances/gfx950/__init__.py`, `instances/gfx1250/__init__.py`: drop attention re-exports.
- `dispatch/__init__.py`, `dispatch/families/__init__.py`: drop `dispatch_attention` / `ATTENTION_REGISTRY`.
- `heuristics/gen_sweep_data.py`: drop the attention sweep-generation path (moved to `library/builders/common/gen_sdpa_sweep_data.py`); the platform copy only *names* that tool in its help text, so it holds no `import` of the library and platform stays fully standalone.
- `examples/run_all.py` REGISTRY + `examples/_goldens/fmha_fwd_hip_mha.json`: remove the `fmha_fwd_hip_mha` (`family="attention"`) entry; its driver `examples/common/fmha_fwd_verify_hip.py` moves to `library/builders/`. Audit `_goldens/` for any other attention entries.
- `rocke/__init__.py`: drop any attention re-export it carries.
- `helpers/__init__.py`: untouched (all helpers stay).
Non-attention platform code is otherwise unchanged.

## 5. Imports
- Platform package name `rocke` is preserved, so non-attention code needs no
  import changes.
- **Library** python is build-time; its source root is **`rocke/library/`**
  (NOT `rocke/`, which would put `platform/` on `sys.path` and shadow the stdlib
  `platform` module). So `kernels`, `builders`, `dispatch` are top-level packages
  under that root (not a `library.*` package). Caveat: these are generic
  top-level names, acceptable for a build-time-only, controlled-PYTHONPATH
  context; wrap them in one named package later if needed.
- Modules import the SDK as `rocke.*` (e.g. `from rocke.helpers.mfma_attention
  import ...`, `from rocke.core.ir import ...`, `from rocke.dispatch.core import ...`).
- **Codemod for the moved files:** relative imports that pointed at platform
  (`from ..core`, `from ...helpers`, dispatch's `from ..core` to `rocke.dispatch.core`,
  etc.) become absolute `from rocke.<...>`; refs to the moved kernels become `from
  kernels.<...>` (e.g. `dispatch/attention` becomes `from kernels.common.attention_unified`);
  relative imports within a moved subpackage stay relative.
- **`rocke.examples.*.attention*` and `rocke.examples.common.fmha_fwd_verify_hip`
  become `builders.<...>`** (statements, string literals, `_goldens` JSON `module`,
  REGISTRY rows, `-m` docs), full-occurrence discipline, scoped to attention only.
- Build-time PYTHONPATH (conftest + a `cmake -E env` helper): `platform/Python`
  (for `rocke`) + `rocke/library/` (for `kernels`/`builders`/`dispatch`).
- **Editable-install migration:** per-script `sys.path`/`parents[N]`/`Path(__file__)`
  hacks are replaced by editable installs plus a `rocke.assets` accessor module
  (`platform_root`/`dsl_docs_dir`/`shape_utils_dir`, env overrides
  `ROCKE_PLATFORM_ROOT`/`ROCKE_DSL_DOCS`); the only residual path handling lives
  in the test `conftest.py` bootstraps and `assets.py` itself.
- One-way rule: `library` imports `platform` only; platform never imports
  `library` (verified: zero `kernels`/`builders`/`dispatch` imports anywhere
  under `platform/`, static and at runtime).

## 6. CMake
- `BASE/CMakeLists.txt`: `add_subdirectory(rocke)` replaces the `rocke-client` +
  `rocKE` adds; keep the `ROCKE_INSTALL_*` staging vars set before it and
  `add_subdirectory(src)` after (CTest GLOBAL-staging order preserved).
- `rocke/CMakeLists.txt`: `add_subdirectory(platform)`. The library python
  (`kernels`/`builders`/`dispatch`) is not a CMake subdirectory; it is
  editable-installed and wheeled by the `rocke-pyenv` / `rocke-wheels` blocks in
  this file, which reference `library/pyproject.toml` by path.
- `rocke/platform/CMakeLists.txt` = former `rocKE/CMakeLists.txt`, unchanged
  (still installs the `rocke` SDK package, now without attention, builds C++
  tests, stages CTest).
- ~~`rocke/library/CMakeLists.txt` + `api/CMakeLists.txt`~~: removed (AICK-1737)
  along with the `library/api/` client. Library is now Python-only.

## 7. Tests
- **platform/tests/**: C++ tests (incl. `tiled_attention_2d_reentrancy.cpp`),
  non-attention python tests; re-rooted `conftest.py` (platform root).
- **library/tests/**: the python attention tests (`test_gfx1250_attention.py`
  + any others importing the moved kernels/builders) with a conftest inserting
  `platform/Python` + `rocke/`.

## 8. Verification
- **Platform standalone:** clean venv, `pip install rocke/platform`, then
  `python -c "import rocke, rocke.core, rocke.helpers, rocke.instances, rocke.dispatch; from rocke import lower_kernel_to_llvm"` succeeds without library present. (Proves attention removal is clean and platform stays self-contained.)
- **Platform has no attention *kernels*:** `instances`/`dispatch` carry no
  attention modules and `dispatch.dispatch_attention` is no longer importable,
  but platform retains the attention-infra *helpers* (`helpers/attention`,
  `mfma_attention`, `qk_scale`, ...) and `instances/gfx1250/qwen3_kv_cache` by design.
- **Library on platform:** with both roots on the path (`platform/Python` +
  `rocke/library/`),
  `python -c "import kernels.common.attention_unified, dispatch, builders.gfx1250.attention..."` and `dispatch.attention.dispatch_attention` resolve.
- **Functional wiring (not just imports):** after the move, actually *run* the
  SDPA path end-to-end: invoke `dispatch.attention.dispatch_attention` to select
  and build an attention kernel, and run one `builders` driver, to prove the
  moved kernels+dispatch+builders are wired and produce output, not merely import.
- **Zero-leftover:** `search 'rocke\.examples.*attention'` and stale
  `rocke.instances.common.attention*` references resolve to none.
- **pytest parity:** identical pass/skip/fail for the moved attention tests
  (now under library) and the remaining platform tests, pre vs post.
- **CMake both modes** (provider opt-in + default OFF) configure/build/install;
  installed CTest lists the platform engine tests staged into the provider bucket.
- **Move integrity:** `git diff -M` shows renames; `pre-commit` clean.

## 9. Notes / follow-ups (out of scope here)
- **AOT** pre-built kernel list (Python builders to platform lowering to
  artifacts) is a follow-up, now tracked under AICK-1704 (migration into
  `hip-kernel-provider`, off POC / PR #9533).
- Optional later: rename `platform/Python` to `platform/python`, `Cpp` to `cpp`.
```
