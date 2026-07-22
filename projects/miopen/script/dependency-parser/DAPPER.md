# Dapper — selective gtest filtering for MIOpen

Dapper narrows the set of MIOpen gtest fixtures that run to those a change could
actually affect, computed from the git diff and the build's dependency graph. It is
**strictly subtractive**: the set it runs is always a subset of the user/category
filter (or a minimal default) — it never adds fixtures beyond what was requested.

The expensive part (deciding *what could be affected*) is done at build time on a
GPU-less machine; only the actual gtest execution needs a GPU. This makes Dapper a
natural fit for TheRock's split CI (a builder that defines artifacts + a separate GPU
runner that installs and executes them).

- [Modes](#modes)
- [Core concepts](#core-concepts)
- [Full test cycle (TheRock)](#full-test-cycle-therock)
- [Native MIOpen-CI cycle](#native-miopen-ci-cycle)
- [Attribution bridges](#attribution-bridges)
- [Configuration](#configuration)
- [Files](#files)
- [Known limitations](#known-limitations)

## Modes

Selected with the CMake cache variable `MIOPEN_DAPPER_MODE`:

| Mode | Meaning | Default in |
|------|---------|------------|
| `off` | Dapper disabled. The single gtest and its shard tests still build/run; no impact analysis. | — |
| `validate` | Native shard run uses the full category; Dapper computes the union only to *validate* coverage (`dapper_diff`). | native / MIOpen-CI |
| `union` | **Active** — the reduced subtractive union filter actually runs. | TheRock |

## Core concepts

> [!IMPORTANT]
> **Dapper analyzes _committed_ changes only.** The impact set comes from
> `git diff <base>..HEAD`, so **uncommitted working-tree edits are invisible to dapper** —
> including changes staged with `git apply` or a patch. **Commit your changes before running
> a dapper test**, otherwise dapper will not attribute (or select) the tests you just
> modified. (In native `validate` mode the full category still runs, so a broken test would
> fail there anyway; but dapper's *selection/compliance* only sees committed changes.)

- **user / category filter** — the fixtures the caller asked for. In TheRock these
  come from a *category* in `test_categories.yaml` (e.g. `standard`), selected at
  test launch.
- **dapper_filter** — the impact set: fixtures reachable from the files changed
  between the merge-base and HEAD, via the dependency graph. Computed on the builder.
- **union_filter** — `dapper_filter` ∩ category positives, plus the category's
  negatives. Always ⊆ the category (subtractive). Computed on the builder and burned
  into the install `CTestTestfile.cmake` (TheRock) / computed by `dapper_diff` (native).
- **fallback_mode** — how the effective filter is chosen when the impact set is
  unusable, decided on the builder and applied when the union is computed:
  - `union` — attributed changes exist → run the intersection.
  - `entire_category` — a change is compiled in but unattributable (common `.cpp`
    body a bridge could not resolve, a runtime-compiled kernel, or an
    undeterminable diff) → run the whole category. Safe: never skips.
  - `minimal` — nothing test-relevant changed → a smoke default.

## Full test cycle (TheRock)

Principle: **everything needed to decide what to run is computed on the GPU-less
builder; only gtest execution happens on the GPU runner.**

### Builder (no GPU)

1. **CMake configure** — `test/gtest/CMakeLists.txt` detects TheRock
   (`THEROCK_SUBPROJECT_TARGET`), defaults `MIOPEN_DAPPER_MODE=union`, and
   `apply_test_category_labels(...)` generates the normal install `CTestTestfile.cmake`
   (each `miopen_gtest_<category>_suite` invokes the binary directly with the full
   category `--gtest_filter`). In `union` mode it then calls
   `dapper_therock_generate_json()` (dapper.cmake) to set up the build-time steps below.
2. **Compile** — build `miopen_gtest`, producing the final `build.ninja`,
   `compile_commands.json`, object files, and the `.ninja_deps` log.
3. **Impact analysis + finalize** — the `dapper_therock_json` target
   (`DEPENDS miopen_gtest` and the configure-generated CTestTestfile) runs, in one
   command:
   - `main.py shas --base-ref <ref> --source-dir <miopen source>` — git merge-base +
     HEAD (git runs in the source worktree; the build dir is not a git repo).
   - `extract_gtest_fixtures.py` — `compile_commands.json` → per-source fixtures
     (keyed `bin/test_<stem>`).
   - `main.py parse build.ninja --bridges=<...>` — `ninja -t deps` per object, plus the
     selected attribution bridge(s) → `file → {tests}` mapping.
   - `main.py select ... --output miopen_dapper_tests.json` — git diff → changed files →
     affected fixtures → **`dapper_filter` + `fallback_mode`**.
   - `main.py finalize-ctest --ctest-in <install CTestTestfile> --ctest-out <finalized>
     --yaml test_categories.yaml --dapper-json miopen_dapper_tests.json` — for each
     Dapper-enabled category (`enable_dapper`), compute the union (honoring
     `fallback_mode`) and **burn it into `<category>_suite`'s `--gtest_filter`**, add a
     `<category>_unfiltered_suite` that retains the full filter, and record
     `category_<NAME>_filter` (original) + `category_<NAME>_union` (effective) in the JSON.

   All CPU-only: git, `ninja -t deps`, `nm`, C-preprocessing. All dapper computation
   happens here, single-process, atomic writes.
4. **Install** — `bin/miopen_gtest` and, under `bin/<PROJECT>/`: the **finalized**
   `CTestTestfile.cmake` (union burned in) and `miopen_dapper_tests.json` (reference /
   downloadable record). **No** python is installed to the runner.
5. **Package** — `miopen_test` (`bin/miopen_gtest*`) and `miopen_run`
   (`bin/<PROJECT>/**`, via the artifact catch-all). No TheRock-repo change is needed.

### Runner (GPU)

6. **Install artifacts** — fetched and flattened to `./build/`, so `bin/miopen_gtest`
   and `bin/<PROJECT>/*` sit next to each other.
7. **Dispatch** — `test_runner.py` runs
   `ctest -L ^<category>$ [-L ^ex_gpu_<arch>$] --test-dir ./build/bin/<PROJECT>`.
8. **Run** — `ctest` invokes the selected suite directly:
   `../miopen_gtest --gtest_filter=<union>` (the union was burned in at build time). No
   dapper code runs at ctest time; this is exactly develop's direct-binary invocation,
   only the filter value differs. Running `<category>_unfiltered_suite` runs the full
   category. **This is the only step that uses the GPU.**

`fallback_mode=entire_category` (unattributable change, or a missing/unreadable JSON at
finalize) makes the burned-in filter the full category — it never silently skips.

**Mental model:** builder = "diff → intersect with each category → burn the reduced
filter into the CTestTestfile"; runner = "just run ctest."

## Native MIOpen-CI cycle

One machine, GPU present, `MIOPEN_DAPPER_MODE=validate` (default). `dapper_init()`
wires the impact targets into `check`; `cmake` → build → `ctest`/`check`:

1. Shard tests run the **full category** on the GPU (`miopen_gtest_shardN`).
2. `dapper_tests_generate` (`select`) then `miopen_gtest_sharded_dapper` (`dapper_diff`)
   run afterward to *validate* that the shard run covered the impact set. The union is
   computed but not used to reduce the run.

Flipping native CI to active `union` is a follow-up (today exercised via the
`diff_check` target).

## Attribution bridges

The base mapping is the include graph (`ninja -t deps`). It attributes changed
**headers** to every test that includes them, but a change confined to a common
`.cpp` **body** (nothing `#include`s a `.cpp`) is not attributed. Bridges are additive
passes that close that gap; select one with `MIOPEN_DAPPER_BRIDGES` (comma list).

| Bridge | Module | How | Notes |
|--------|--------|-----|-------|
| `symbol` | `src/symbol_graph.py` | `nm` provider→consumer symbol graph: attribute a source to the tests that reference the out-of-line symbols it defines. | Precise (mirrors the linker); also handles library `.cpp`. |

Bridges only *add* edges to the mapping; the include graph is never modified.
`symbol` runs by default; set `MIOPEN_DAPPER_BRIDGES` to empty to disable all
bridges. A future runtime-kernel bridge plugs into the same registry. When
multiple bridges can coexist, a superseding bridge drops the ones it makes
redundant (see `BRIDGE_SUPERSEDES` in `main.py`).

## Configuration

| CMake cache var | Default | Purpose |
|-----------------|---------|---------|
| `MIOPEN_DAPPER_MODE` | `union` (TheRock) / `validate` (native) | `off` \| `validate` \| `union` |
| `MIOPEN_DAPPER_BASE_REF` | `origin/develop` | Ref to compute the impact diff against |
| `MIOPEN_DAPPER_BRIDGES` | `symbol` | Additive attribution bridges: `symbol` (set to empty to disable) |

Per category, `test_categories.yaml` sets `enable_dapper: "True"` to opt in. A category's
suite gets its union burned in only when `MIOPEN_DAPPER_MODE=union` **and** that category
has `enable_dapper` truthy; otherwise it runs the full category unchanged.

## Files

Tooling (`script/dependency-parser/`, all builder-side):
- `main.py` — CLI: `shas`, `parse` (with `--bridges`), `select`, `finalize-ctest`,
  `audit`, `optimize`. `finalize-ctest` burns the per-category union into the install
  CTestTestfile and records the filters in the JSON (TheRock).
- `src/enhanced_ninja_parser.py` — build.ninja + `ninja -t deps` → mapping; single-gtest
  synthetic `bin/test_<stem>` keys; `compiled_sources`.
- `src/extract_gtest_fixtures.py` — compile_commands → per-source fixtures.
- `src/selective_test_filter.py` — git diff → affected fixtures → `dapper_filter` +
  `fallback_mode`.
- `src/symbol_graph.py` (`symbol` bridge).
- `src/miopen_gtest_runner.py`, `src/dapper_diff.py` — native validate-mode analysis.
- `src/dapper_union.py` — single source of truth for the pure union math (pattern
  splitting/overlap + subtractive intersection + `fallback_mode` resolution). Used by
  `miopen_gtest_runner.py` (native) and by `main.py finalize-ctest` (TheRock). Not shipped
  to the runner.

Shared (`<rocm-libraries>/shared/ctest/`): unchanged from develop — dapper adds nothing
here. `parse_test_categories.py` / `TestCategories.cmake` generate the normal (direct
binary) install CTestTestfile; dapper rewrites it afterward on the builder.

Build/runtime artifacts installed to `bin/<PROJECT>/` on the runner (union mode):
`CTestTestfile.cmake` (union filters burned in, plus `<category>_unfiltered_suite` entries)
and `miopen_dapper_tests.json` (`dapper_filter`, `fallback_mode`, and per-category
`category_<NAME>_filter` / `category_<NAME>_union` — the downloadable record). No python
ships to the runner; `ctest` invokes the binary directly.

## Known limitations

- **Runtime-compiled GPU kernels** (HIPRTC/COMGR) have no build-time edge (include or
  symbol) to the fixtures that exercise them, so a kernel-only change is not attributed
  and falls back to `entire_category`. A future data-derived (coverage/trace) map is the
  intended fix.
- **Native `union`** is not yet a drop-in for `check`; it runs via `diff_check` today.
- **No compliance report on TheRock (future work).** The `dapper_diff` coverage check
  (COMPLIANT / FAIL / NOT VIABLE) runs only in native `validate` mode, which has the full
  shard run to compare against. TheRock `union` mode just runs the burned-in reduced set —
  nothing verifies that the reduction covered what a full run would have. Producing an
  equivalent compliance/coverage report for the TheRock path is left as future work.
- **Windows is unsupported (dapper is forced off).** `dapper_init()` sets
  `MIOPEN_DAPPER_MODE=off` on Windows hosts, so the build falls back to the normal
  full-category test flow. The build-time tooling is not yet Windows-ready; future work to
  enable it must address:
  - `symbol_graph.py` shells out to `nm`; Windows toolchains provide `llvm-nm` instead
    (make the tool configurable / add a fallback).
  - `extract_gtest_fixtures.py` parses `compile_commands.json` with `shlex` in POSIX mode
    and invokes the compiler's preprocessor via subprocess; both need Windows-aware handling.
  - Audit for other Unix-only assumptions (e.g. the `resource` module, path separators).
