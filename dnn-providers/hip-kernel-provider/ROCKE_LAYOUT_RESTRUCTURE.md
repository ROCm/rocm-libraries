# rocKE Layout Restructure — Final Structure (SDPA carve)

**Branch:** `users/bharriso/rocke-layout-restructure` (off latest `develop`,
includes the merged `rocke-client` skeleton #8928).
**Scope (locked):** wrap existing `rocKE/` under a `platform/` parent **untouched
internally**, and extract **only the SDPA/MHA vertical** into a new `library/`.
Everything non-attention stays in platform. C++ engine + byte-identity parity
stay where they are (dormant until JIT — status, not a move).

Paths under `dnn-providers/hip-kernel-provider/` (**BASE**) — fixed location for now.

---

## 1. Top-level final tree

```
BASE/
├── CMakeLists.txt                      # EDIT: add_subdirectory(rocke)  (was rocke-client + rocKE)
└── rocke/                              # NEW project parent
    ├── CMakeLists.txt                  # NEW: add_subdirectory(platform); add_subdirectory(library)
    ├── BUILDING.md                     # NEW: build/dev guide (editable installs, PYTHONPATH, rocke.assets)
    ├── platform/                       # = former rocKE/, reparented verbatim (only attention-unwiring edits)
    │   ├── CMakeLists.txt              # = former rocKE/CMakeLists.txt
    │   ├── pyproject.toml  requirements.txt
    │   ├── AGENTS.md  BUILD.md  README.md
    │   ├── Cpp/                        # C++ engine + bindings (dormant until JIT; unchanged)
    │   ├── Python/
    │   │   └── rocke/                  # the SDK package (import name `rocke`, installable)
    │   │       ├── __init__.py         # EDIT: drop attention re-exports only
    │   │       ├── __main__.py
    │   │       ├── core/               # ir, lower_llvm/hip/cktile, passes, ir_serialize, verify, arch, isa, backend
    │   │       ├── helpers/            # ALL helpers stay (generic toolkit, incl. attention-infra primitives)
    │   │       ├── runtime/            # comgr, hip_module, launcher
    │   │       ├── analysis/  benchmark/
    │   │       ├── instances/          # ALL kernels EXCEPT attention  (EDIT __init__s: drop attention)
    │   │       ├── dispatch/           # ALL families EXCEPT attention  (EDIT __init__s: drop attention)
    │   │       ├── heuristics/         # stays (1 lazy attention import repointed to library)
    │   │       ├── examples/           # ALL examples EXCEPT gfx*/attention
    │   │       └── sweep.py sweep_bench.py run_manifest.py torch_backend.py
    │   ├── tests/                      # C++ tests + platform pytest (minus attention python tests)
    │   ├── tools/  cmake/  dsl_docs/
    └── library/                        # NEW — the SDPA/MHA product (relies on platform `rocke.*`)
        ├── CMakeLists.txt              # NEW: add_subdirectory(api); (build-time python, NOT installed)
        ├── pyproject.toml              # NEW: build-time metadata (library python is NOT installed)
        ├── kernels/                    # was platform instances attention (the SDPA kernel defs)
        ├── builders/                   # was platform examples/gfx*/attention (build-time drivers)
        ├── dispatch/                   # was dispatch/families/attention (1-of-N SDPA select; C++ replaces later)
        ├── api/                        # was rocke-client (C-api / hipDNN provider plugin)
        └── tests/                      # the python attention tests
# removed: BASE/rocKE/ and BASE/rocke-client/  (their content now under rocke/)
```

## 2. platform — what it is
The entire former `rocKE/`, reparented as `rocke/platform/`, **internally
unchanged** except the attention-unwiring edits (§4). It remains:
- the installable **SDK** (`rocke` package at `platform/Python/rocke`, via
  `platform/pyproject.toml`), usable standalone by others to build kernels;
- the home of `core`, **all** `helpers` (the generic toolkit, including
  attention-infra primitives `mfma_attention`, `mfma_attention_bwd`, `qk_scale`,
  `attention`, `rotary` — they are reusable building blocks), `runtime`,
  `analysis`, `benchmark`, and **all non-attention** `instances`/`dispatch`/
  `examples`, `heuristics`, the C++ engine (`Cpp/`), tools, docs, tests.

## 3. library — the SDPA/MHA carve (relies on platform helpers)
Build-time Python (NOT installed) + the C-api. Self-contained SDPA kernels that
import the platform SDK as `rocke.*`.

| `library/` dir | Source (from platform) | Contents |
|---|---|---|
| `kernels/` | `instances/common/` attention + arch attention | `attention_unified`, `attention_arch`, `_fmha_common`, `_fmha_warp_body`, `fmha_{appendkv,arch,bwd,fwd_fp8,head_grouping,mfma,paged_prefill,splitkv_decode,varlen}`, `sage_attention`, `sparse_attention`; `gfx1151/wmma_fmha_fwd`; `gfx1250/{_wmma_attention_common,attention_tiled_2d,attention_tiled_3d,wmma_attention_fwd}`; `gfx942/{attention_tiled_2d,attention_tiled_3d}`; `gfx950/{attention_tiled_2d,attention_tiled_2d_fastkv_regp,attention_tiled_3d}`; **plus** the inline arch-router builders `build_unified_attention_2d_tiled` / `build_unified_attention_3d_tiled` lifted out of `instances/__init__.py` |
| `builders/` | `examples/gfx{942,950,1151,1250}/attention/`, `examples/common/fmha_fwd_verify_hip.py`, plus the attention halves split out of mixed `examples/common` drivers (`parity_fmha_extended` ← `parity_extended_kernels`, `hip_lowering_attention_parity` ← `hip_lowering_parity`), the `dsl_probe`/`gen_sweep`/`stage1_benchmark` attention slices (`dsl_probe_attention_demos`, `gen_sdpa_sweep_data`, `benchmark_rocke_unified_attention`), and `gfx950/qwen3_30b_a3b/` decode drivers | the SDPA verify/parity/bench drivers (build-time) |
| `dispatch/` | `dispatch/families/attention.py` | `ATTENTION_REGISTRY`, `AttentionRequest`, `dispatch_attention` — kept functional now (1-of-N SDPA select); C++ dispatch in `api` supersedes it later |
| `api/` | `rocke-client/` | C-api / hipDNN provider plugin; **library is the provider for now**. Preserve engine identity: name `"rocke-client"`, `ROCKE_ENGINE_ID` |
| `tests/` | `tests/instances/test_gfx1250_attention.py` (+ any other python attention tests) | python attention tests (C++ `tiled_attention_2d_reentrancy.cpp` stays in platform — it tests the C++ engine) |

**Naming:** `kernels` (was *instances*) + `builders` (was *examples*) —
product nouns; platform keeps its DSL terms `instances`/`examples`.

## 4. The only platform edits (attention-unwiring)
Removing attention from the `instances` package requires editing its re-export
sites (all mechanical "drop/relocate attention"):
- `instances/__init__.py` — strip the attention re-export block (`UnifiedAttention*`, `build_unified_attention_*`) and **move** the inline `build_unified_attention_2d_tiled`/`_3d_tiled` arch-routers to `library/kernels/`.
- `instances/gfx942/__init__.py`, `instances/gfx950/__init__.py`, `instances/gfx1250/__init__.py` — drop attention re-exports.
- `dispatch/__init__.py`, `dispatch/families/__init__.py` — drop `dispatch_attention` / `ATTENTION_REGISTRY`.
- `heuristics/gen_sweep_data.py` — drop the attention sweep-generation path (moved to `library/builders/common/gen_sdpa_sweep_data.py`); the platform copy only *names* that tool in its help text, so it holds **no** `import` of the library — platform stays fully standalone.
- `examples/run_all.py` REGISTRY + `examples/_goldens/fmha_fwd_hip_mha.json` — remove the `fmha_fwd_hip_mha` (`family="attention"`) entry; its driver `examples/common/fmha_fwd_verify_hip.py` moves to `library/builders/`. Audit `_goldens/` for any other attention entries.
- `rocke/__init__.py` — drop any attention re-export it carries.
- `helpers/__init__.py` — **untouched** (all helpers stay).
Non-attention platform code is otherwise unchanged.

## 5. Imports
- **Platform package name `rocke` is preserved** → non-attention code needs **no
  import changes**.
- **Library** python is build-time; its source root is **`rocke/library/`**
  (NOT `rocke/` — that would put `platform/` on `sys.path` and **shadow the
  stdlib `platform` module**). So `kernels`, `builders`, `dispatch` are
  **top-level** packages under that root (not a `library.*` package). Caveat:
  these are generic top-level names — acceptable for a build-time-only,
  controlled-PYTHONPATH context; wrap them in one named package later if needed.
- Modules import the SDK as `rocke.*` (e.g. `from rocke.helpers.mfma_attention
  import …`, `from rocke.core.ir import …`, `from rocke.dispatch.core import …`).
- **Codemod for the moved files:** relative imports that pointed at platform
  (`from ..core`, `from ...helpers`, dispatch's `from ..core` → `rocke.dispatch.core`,
  etc.) → absolute `from rocke.<...>`; refs to the moved kernels → `from
  kernels.<...>` (e.g. `dispatch/attention` → `from kernels.common.attention_unified`);
  relative imports *within* a moved subpackage stay relative.
- **`rocke.examples.*.attention*` and `rocke.examples.common.fmha_fwd_verify_hip`
  → `builders.<...>`** (statements + string literals + `_goldens` JSON `module` +
  REGISTRY rows + `-m` docs), full-occurrence discipline, scoped to attention only.
- Build-time PYTHONPATH (conftest + a `cmake -E env` helper): `platform/Python`
  (for `rocke`) + `rocke/library/` (for `kernels`/`builders`/`dispatch`).
- **Editable-install migration:** per-script `sys.path`/`parents[N]`/`Path(__file__)`
  hacks are replaced by editable installs plus a `rocke.assets` accessor module
  (`platform_root`/`dsl_docs_dir`/`shape_utils_dir`, env overrides
  `ROCKE_PLATFORM_ROOT`/`ROCKE_DSL_DOCS`); the only residual path handling lives
  in the test `conftest.py` bootstraps and `assets.py` itself.
- One-way rule: `library → platform` only; platform never imports `library`
  (verified: zero `kernels`/`builders`/`dispatch` imports anywhere under
  `platform/`, static and at runtime).

## 6. CMake
- `BASE/CMakeLists.txt`: `add_subdirectory(rocke)` replaces the `rocke-client` +
  `rocKE` adds; keep the `ROCKE_INSTALL_*` staging vars set **before** it and
  `add_subdirectory(src)` **after** (CTest GLOBAL-staging order preserved).
- `rocke/CMakeLists.txt` (NEW): `add_subdirectory(platform)`; `add_subdirectory(library)`.
- `rocke/platform/CMakeLists.txt` = former `rocKE/CMakeLists.txt`, unchanged
  (still installs the `rocke` SDK package — now without attention — builds C++
  tests, stages CTest).
- `rocke/library/CMakeLists.txt` (NEW): `add_subdirectory(api)`; **no Python
  install** (library python is build-time only). `api/CMakeLists.txt` = former
  `rocke-client/CMakeLists.txt` with the parent-relative `../../cmake/` →
  `../../../../cmake/` depth fix (+2); **preserve** the provider-plugin
  registration, install to `bin/rocke-client`, CTest targets
  (`rocke_client_tests`, `rocke_client_integration_tests`), and the engine
  name/`ROCKE_ENGINE_ID`.

## 7. Tests
- **platform/tests/** — C++ tests (incl. `tiled_attention_2d_reentrancy.cpp`),
  non-attention python tests; re-rooted `conftest.py` (platform root).
- **library/tests/** — the python attention tests (`test_gfx1250_attention.py`
  + any others importing the moved kernels/builders) with a conftest inserting
  `platform/Python` + `rocke/`.

## 8. Verification
- **Platform standalone:** clean venv, `pip install rocke/platform`, then
  `python -c "import rocke, rocke.core, rocke.helpers, rocke.instances, rocke.dispatch; from rocke import lower_kernel_to_llvm"` succeeds **without library present**. (Proves attention removal is clean and platform stays self-contained.)
- **Platform has no attention *kernels*:** `instances`/`dispatch` carry no
  attention modules and `dispatch.dispatch_attention` is no longer importable —
  but platform **retains** the attention-infra *helpers* (`helpers/attention`,
  `mfma_attention`, `qk_scale`, …) and `instances/gfx1250/qwen3_kv_cache` by design.
- **Library on platform:** with both roots on the path (`platform/Python` +
  `rocke/library/`),
  `python -c "import kernels.common.attention_unified, dispatch, builders.gfx1250.attention…"` and `dispatch.attention.dispatch_attention` resolve.
- **Functional wiring (not just imports):** after the move, actually *run* the
  SDPA path end-to-end — invoke `dispatch.attention.dispatch_attention` to select + build an
  attention kernel, and run one `builders` driver — to prove the moved
  kernels+dispatch+builders are wired and produce output, not merely import.
- **Zero-leftover:** `search 'rocke\.examples.*attention'` and stale
  `rocke.instances.common.attention*` references → none.
- **pytest parity:** identical pass/skip/fail for the moved attention tests
  (now under library) and the remaining platform tests, pre vs post.
- **api unchanged contract:** plugin name `"rocke-client"` + `ROCKE_ENGINE_ID`
  asserted by `TestPluginPublic`/`TestRockeClientLoad` still pass; installed
  `bin/rocke-client` present; provider CTest staged.
- **CMake both modes** (provider opt-in + default OFF) configure/build/install;
  installed CTest lists the same `rocke_client_*` + engine tests.
- **Move integrity:** `git diff -M` shows renames; `pre-commit` clean.

## 9. Notes / follow-ups (out of scope here)
- **AOT** pre-built kernel list (Python builders → platform lowering → artifacts
  the C-api ships) is a **follow-up**.
- **Provider split**: `library/api` later lifts to `dnn-providers/rocke-provider/`
  (keep engine identity); trivial move, no pre-split needed now.
- Optional later: rename `platform/Python`→`platform/python`, `Cpp`→`cpp`.
```
