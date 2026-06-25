# tilewright — ML recommender for GEMM kernel selection

tilewright is a standalone, **framework-agnostic** ML model that, given a GEMM problem
and a list of candidate kernel configs, ranks the configs by predicted performance
and picks the fastest. It is meant to **augment or replace the analytical "base
selection model"** a framework already uses (for hipBLASLt that base model is
`origami`; future backends target Triton, CK, …).

The engine core depends on **no** GEMM-framework headers. Each framework gets a thin
backend adapter that converts its own problem/config/hardware types into tilewright's
neutral types and maps the ranking back.

> Terminology: say **"base selection model"** for the analytical baseline in general
> (tilewright is backend-neutral). Only say **"origami"** for hipBLASLt-specific things
> — and origami types/headers must appear **only** under `backends/hipblaslt/`.

---

## Layout

```
include/tilewright/            public engine API (model.hpp) + neutral types (types.hpp)
src/tilewright/model.cpp       the whole engine: binary loader, routing, scoring
backends/hipblaslt/        the ONLY place allowed to #include origami/* headers
  include/.../adapter.hpp    origami<->tilewright type glue + rank_configs entry points
  weights/<arch>/<arch>/     deployed weights: *.tilewright.bin (Git LFS) + tilewright_index
                             (mirrors the Tensile Logic/asm_full/<arch>/<arch>/ layout)
python/                    nanobind bindings + python tests (singleton API only)
tests/                     Catch2 tests (test_model.cpp synthetic, test_real_kernels.cpp)
```

The hipBLASLt consumer lives outside this dir, in
`projects/hipblaslt/tensilelite/include/Tensile/{PredictionLibrary.hpp,
Serialization/PredictionLibrary.hpp}` — it includes `adapter.hpp` and links
`roc::tilewright`. The caller owns the on/off gate (`TENSILE_USE_TILEWRIGHT`).

---

## Model architecture

Per **arch** (e.g. gfx950) and per **(dtype, layout) library** there is one trained
model file. A model is a **grid of cells + a split tree**:

1. **Routing.** A problem maps to a coarse 96-cell base label
   `Mtier|Ntier|Ktier|Btier` (tiers: M/N Tiny≤32 Small≤128 Mid≤512 Large; K TinyK≤32
   MidK≤512 LargeK; B Bnone=1 / Bany). A **split tree** then refines that base label
   down a chain of `#AXIS<=v` / `#AXIS>v` thresholds to a **leaf cell**. Each leaf
   has its own trained weights. *(If the split table is missing, every Large/LargeK
   problem fails to reach its leaf and the recommender collapses — see history.)*
2. **Per-cell two-tower scorer.** For the routed cell:
   - **query tower** (problem features, `q_dim`) → query embedding (computed once).
   - **item tower** (per-config features, `i_dim`) → item embedding.
   - `embed_score = dot(q_emb, i_emb) / temperature`.
   - **interaction MLP** (problem×config features, `x_dim`) → scalar `inter_score`.
   - `score = embed_score + inter_score`; higher = better.
   Features are whitened per cell with stored mean/std (std<1e-6 → 1).
3. **Candidate filtering (pre-scoring).** Each config passes: **LDS-capacity gate**
   → **kernel-feasibility filter** (mirrors the base model's analytical feasibility:
   small-batch tile fit, Dot2, NTA/NTB cache-hint rules) → optional **smart_K
   signature whitelist** (the (mt,mi,cache-hint) tuples seen in training for that
   cell). Two-pass: if the whitelist rejects everything, re-scan without it.

`rank_configs` returns best-first; `stable_sort` makes element 0 the first-max
(matches numpy/torch argmax).

---

## API surfaces (two, both live)

| API | entry points | used by |
|---|---|---|
| **handle / per-library** | `load_model_by_index(stem, hint_dir)` → handle; `rank_configs(handle, …)` | TensileLite `PredictionLibrary` (the Tensile path) |
| **singleton** | `load_weights`/`rank_configs(Problem,…)`/`route(Problem)` | rocRoller `solution_selection.cpp`, python bindings, standalone tests |

The handle API loads **one model per Tensile library** (keyed by the logic-file
stem via `tilewright_index`); the singleton holds one process-wide model. Both share the
same scoring core (`rank_configs_impl`).

### `rank_configs` contract
Returns one `Result{config_index, score, scored}` per input config: **scored
survivors first, descending score**; filtered-out configs follow with `scored=false`
(the hipBLASLt adapter maps those to NaN latency, and `PredictionLibrary` skips them
— "all" never returns infeasible kernels).

### `min_scored` (ranking depth)
Default 0 = score only the smart_K whitelist (fast, highest-confidence; this is the
common top-1 path). `min_scored > |whitelist|` also scores the remaining
LDS+feasible configs as a **tier-2 block appended strictly after** the whitelist —
so top-1..top-|whitelist| (and hence the default pick) are byte-identical, while a
caller asking for N (e.g. `requested_solution`) gets a deeper ranked list. The
extra item-tower cost is paid only when the deeper list is requested. The hipBLASLt
caller passes `numSolutions` as `min_scored` (negative/`-1` ⇒ "all feasible").

---

## Weights: format, loading, deployment

- **File:** `<logic_stem>.tilewright.bin`, magic `MLREC_v1` (little-endian). Header:
  weight-dtype, feature-hash, arch, `q/i/x` dims, `n_cells`, `n_splits`; then the
  split rules, then per-cell {dims, temperature, q/i/x mean+std, smart_K signatures,
  MLP weights}. Weights may be **fp32 / bf16 / int8 / int4**; biases + scalers are
  always fp32. Stored via **Git LFS** (`*.bin filter=lfs`).
- **Source layout:** weights live under `backends/hipblaslt/weights/<arch>/<arch>/`,
  mirroring the Tensile logic tree (`.../Logic/asm_full/<arch>/<arch>/`).
- **`tilewright_index`** (text file co-located with the per-arch `.bin`):
  `<logic_stem>\t<weights_filename>` per line. `load_model_by_index(stem)` finds the
  index, matches the stem, loads the bin.
- **Discovery = same as kernels.** The handle API (`load_model_by_index(stem,
  hint_dir)`) looks **only** in `hint_dir` — the directory the Tensile logic file
  (`.dat`) for this library was loaded from. The weights ship co-located with the
  kernel library, so they are found exactly the way the `.co`/`.dat` are; **no env
  var, no separate search tree.** (No match → the library uses the base analytical
  path.) The hipBLASLt build co-locates them: it copies `tilewright_index` +
  the `*.tilewright.bin` from `weights/<arch>/<arch>/` into the flat per-arch
  Tensile library dir (`Tensile/library/<arch>/`), so the existing library
  install/packaging ships them alongside the kernels. The stem is matched after
  stripping all extensions, so double-extension logic files (`*.dat.zlib`) resolve
  to the bare index key.
- **Lazy:** the model loads at `.dat` deserialize time, which Tensile does lazily on
  first use of that (dtype,layout). A mixed workload loads only the models for the
  libraries actually used; a library with no `tilewright_index` entry falls back to the
  base model (per-library, no error).
- **Deploy contract:** the training pipeline's deploy stage (out of this repo) emits
  one `*.tilewright.bin` per Tensile logic library + the `tilewright_index` into
  `weights/<arch>/<arch>/`, materializes the LFS objects, and the build co-locates
  them in the per-arch kernel library dir.

---

## Compute precision
Scoring runs in **scalar fp32** on ROCm/AMD build hosts. On x86 (CI/dev) it uses
AVX512: **bf16 by default** (`TILEWRIGHT_COMPUTE_BF16`, opt out `=0`), int8 opt-in
(`TILEWRIGHT_COMPUTE_INT8=1`, needs AVX512-VNNI), else AVX512 fp32. All AVX512 paths
compile away on ROCm.

## Env knobs
`TENSILE_USE_TILEWRIGHT` (caller gate) ·
`TILEWRIGHT_COMPUTE_BF16` / `TILEWRIGHT_COMPUTE_INT8` (x86 precision) ·
`TILEWRIGHT_FORCE_CELL` (force a cell index, debug) · `TILEWRIGHT_DIAG` (load banner) ·
`TILEWRIGHT_PICK_LOG` (per-call top-1 line — used by the deploy parity check).

## Build & test

As part of hipBLASLt: registered as the `tilewright` monorepo component (root
`CMakeLists.txt` / `CMakePresets.json`); `add_subdirectory` builds `roc::tilewright`
(static engine) + `roc::tilewright-hipblaslt` (header-only backend). No manual steps.

**Standalone build + run the test suite** (AI agents: do this to verify a change;
users: same commands):

```bash
cd shared/tilewright
cmake -S . -B build/ -DCMAKE_BUILD_TYPE=Release -DTILEWRIGHT_BUILD_TESTING=ON
cmake --build build/ --parallel
cd build/ && ctest --output-on-failure        # or: ./tests/tilewright-tests
```

- `test_model.cpp` writes its own synthetic model → needs no deployed weights.
- `test_real_kernels.cpp` loads a deployed model; if the Git LFS `.bin` is only a
  133-byte pointer it **skips gracefully**. To exercise it, materialize the matching
  arch weights first (`git lfs pull`, or fetch the LFS object) so
  `weights/<arch>/<arch>/*.tilewright.bin` is real. Use the **current committed**
  model — a stale/mismatched `.bin` (different `i_dim`) yields garbage/non-deterministic
  results.
- Python bindings + their tests: add `-DTILEWRIGHT_ENABLE_PYTHON=ON` (then `ctest`
  runs the pytest suite too), or `pip install -e python/`.

CMake options: `TILEWRIGHT_BUILD_TESTING` (OFF), `TILEWRIGHT_ENABLE_PYTHON` (OFF),
`TILEWRIGHT_ENABLE_HIPBLASLT_BACKEND` (OFF), `TILEWRIGHT_ENABLE_AVX512` (OFF — opt-in;
the scorer is portable scalar fp32 by default, AVX512 paths are `__AVX512F__`-guarded
and must only be enabled for x86 hosts known to support it), `TILEWRIGHT_ENABLE_FETCH`
(ON), `TILEWRIGHT_ENABLE_INSTALL` (ON).
