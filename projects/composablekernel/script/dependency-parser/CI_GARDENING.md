# Smart Build — CI Gardening Guide

This document is for people who maintain, debug, and operate the CK smart-build
CI pipeline. It covers the decision flow, what every artifact means, how to
diagnose a misbehaving run, and how to safely override or disable the system.

> **Terminology:** see the **Glossary** in [README.md](README.md) for depmap,
> selection, false negative / blind spot, reachability, and related terms.

---

## 1. What the system does (one paragraph)

For each PR build, the smart-build system asks: *which test executables could
possibly be affected by the changed files?* It answers by mapping each changed
file through a pre-built header-dependency graph (`enhanced_dependency_mapping.json`)
to a list of test executables, then building and running only those. A PR that
touches only a single GEMM header might build 40 tests instead of all 2 000.
When the system can't safely answer the question — because CMakeLists.txt changed,
or because a safety check fails — it falls back to a full `ninja check`.

---

## 2. Decision tree

The selection pipeline runs on **every** build (an "as-if" computation, advisory);
the safety check only decides the final build mode at the end.

```
smart_build_ci.sh
│
├─ ci_safety_check.sh → remember FULL_REQUIRED (0/1); do NOT exit yet
│    (full when this PR changes CMakeLists / *.cmake / dependency-parser /
│     codegen inputs, or FORCE_CI / stale cache)
│
├─ compile_commands.json / build.ninja missing? ──YES──► full build, exit 1
│
│  ── as-if pipeline (runs every build, advisory) ──
├─ cmake-parse → enhanced_dependency_mapping.json   (missing → full, exit 1)
├─ reachability guardrail → reachability_result.json (non-fatal)
├─ select → tests_to_run.json                        (missing → full, exit 1)
├─ validate (--junit) → smoke_result.json + smoke_result.xml (non-fatal)
│    so the selective path + JUnit are exercised + published even on full builds
│
│  ── decide actual build mode ──
├─ FULL_REQUIRED? ──YES──► full build, exit 1 (as-if above was advisory)
├─ 0 tests selected? ──YES──► exit 0 (build_mode.env = none; no build/test)
└─ else ──► build_mode.env = selective; build_targets.txt = selected targets
                              build_mode.env = selective
```

---

## 3. Artifact reference

Every artifact is archived by Jenkins (`allowEmptyArchive: true`). Open the
build's **Artifacts** tab and look for these files.

| File | What it means | First thing to check |
|------|--------------|----------------------|
| `build_mode.env` | `SMART_BUILD_MODE=selective\|full\|none` | Which path the run took |
| `smart_build.log` | Build-phase log (selection + compile) from `smart_build.sh` | Build-stage failures |
| `smart_test.log` | Test-phase log (ctest) from `smart_test.sh` | Test-stage failures |
| `smart_build_ci.log` | Selection-phase log — **only produced when `smart_build_ci.sh` is run standalone**; in CI the output is folded into `smart_build.log` | Standalone runs only |
| `tests_to_run.json` | The selected executables and ctest regex | Which tests were chosen |
| `build_targets.txt` | Space-separated ninja targets, **or** the sentinel word `full`/`none` (see below) | What was actually built |
| `reachability_result.json` | Guardrail verdict — which ctest tests are unreachable | Ongoing FN monitoring |
| `smoke_result.json` | Whether every selected target is a real ninja target | Selection drift detection |
| `smoke_result.xml` | JUnit version of `smoke_result.json` | Published to Jenkins test results |

> **Always-on as-if:** `smart_build_ci.sh` computes the selection + selection-validity
> smoke (`smoke_result.json/.xml`) and reachability on **every** build, full or
> selective — so JUnit is published and the selective path is exercised on every
> run. JUnit is published in the **Smart Build** stage (the smoke is produced there),
> so it lands even if the Smart Test stage later fails.
>
> **As-if vs real (mode tag):** the smoke carries the build mode so advisory
> computations aren't mistaken for what ran. `smoke_result.json` has
> `"mode": full|selective|none` + `"advisory": true|false` (advisory on full/none),
> and the JUnit suite/classname is tagged (`smart-build.selection.<mode>`) so the
> Jenkins test-results trend keeps as-if (full/none) runs distinct from real
> selective ones.

> **Two stages:** the pipeline runs `smart_build.sh` (Smart Build stage) then
> `smart_test.sh` (Smart Test stage). A **build** failure and a **test** failure
> now surface as distinct Jenkins stage failures with separate timing — check the
> stage name first, then its log (`smart_build.log` vs `smart_test.log`). The two
> stages share the `build/` directory; `smart_test.sh` reads `build_mode.env`
> (and `tests_to_run.json` in selective mode) to know what to run.

> **`build_targets.txt` sentinels:** in `full` mode the file contains the single
> word `full`; in `none` mode it contains `none`. Only in `selective` mode does it
> hold a real space-separated target list. `smart_build.sh` checks for `none` and
> skips the build phase. Do not `ninja $(cat build_targets.txt)` without first
> checking `build_mode.env`.

> **Why `smart_build_ci.log` is usually absent in CI:** the Smart Build stage runs
> `smart_build.sh`, which sets `_SMART_BUILD_NESTED=1` and calls `smart_build_ci.sh`
> as a child. The child detects the nesting and skips its own log, so its
> selection-phase output appears in `smart_build.log` instead. The separate file
> appears only when you run `smart_build_ci.sh` by hand.

### `build_mode.env` values

| Value | Meaning |
|-------|---------|
| `selective` | Smart build ran, N tests selected and built |
| `full` | Safety check forced a full `ninja check` |
| `none` | Smart build ran, 0 tests selected (change touches no CK code) |

If the file is **absent**, `smart_build_ci.sh` crashed before it could write it
(e.g., CMake not configured). Check `smart_build_ci.log`.

> **Always-run class (non-compiled tests):** some ctest tests have no compiled
> `bin/` target — python script tests and `try_compile` tests. No source file maps
> to them, so the selector can never pick them. `smart_test.sh` runs this class on
> every `selective` and `none` build (in addition to the selected chunks), sourced
> automatically from `reachability_result.json` `non_compiled`. `full` covers them
> via the whole suite. The set is auto-derived: a newly added python test is picked
> up on the next build with no script edit. (If the reachability guardrail was
> skipped — e.g. `ctest -N` empty, see §4.6 — the class is empty and `smart_test.sh`
> logs that it found none.)

> **Scope: smart-build's test universe is the default `ctest -N` suite.** A test is
> covered exactly when it's registered with ctest via `add_test(...)` in a
> `CMakeLists.txt` — compiled tests by file→target mapping, non-compiled ones via the
> always-run class above. Adding a test the standard way (`add_test`) also edits a
> `CMakeLists.txt`, which the safety check (§7) treats as a full build, so it runs in
> the PR that introduces it.
>
> Some CK test suites run **outside** `smart_test.sh`, in their own flag-gated
> Jenkins stages/targets, and are governed by those flags rather than by selection:
> `ninja check-rocm-ck` (`RUN_ROCM_CK_TESTS`), `ninja check-builder`
> (`RUN_BUILDER_TESTS`), inductor codegen via `script/run_inductor_tests.sh`
> (`RUN_INDUCTOR_TESTS`), and the downstream (pytorch/aiter/fa), tile-engine, and
> FMHA stages. The always-run class covers only the non-compiled tests in the default
> `ctest -N` suite; these separate suites are unaffected by it and by selection.
>
> Because `rocm_ck`/`builder` are registered with ctest but built by their own
> targets, **full-mode `smart_test` excludes them** (`ctest -LE ROCM_CK_|BUILDER_SMOKE`,
> override with `CTEST_FULL_EXCLUDE_LABELS`) so they don't fail as "Not Run". They run
> instead as **dedicated per-arch stages** — `rocm_ck Tests (<arch>)` (gated by
> `RUN_ROCM_CK_TESTS`) and `Builder Tests (<arch>)` (gated by `RUN_BUILDER_TESTS`,
> excluding gfx10/gfx11/gfx1250) — in both the smart and run-all paths, each running
> its own `ninja check-rocm-ck` / `check-builder` (build + labeled ctest).

### `reachability_result.json` fields

```json
{
  "n_ctest": 651,          // total ctest-registered tests
  "n_reachable": 614,      // tests the filter can possibly select
  "n_false_negatives": 0,  // compiled tests the filter can NEVER select (alarm if >0)
  "false_negatives": [],
  "n_non_compiled": 37,    // python/try_compile tests — always-run class (run in selective/none), not FNs
  "non_compiled": [...],
  "allowlisted": [],       // tests suppressed via --allowlist (see §5)
  "classified": true,      // true = build.ninja was provided for classification
  "verdict": "pass"        // "fail" if any compiled test is unreachable
}
```

`verdict: "fail"` means there's a real dependency-extraction gap for those tests.
The build proceeds anyway (guardrail is non-fatal) but the tests are logged.
Track `false_negatives` across builds — a new entry means a newly-added test
wasn't reachable. File a ticket and add it to the allowlist (see §5) until fixed.

---

## 4. Common failure scenarios

### 4.1 "Full build mode" every run

**Symptom:** `build_mode.env` always contains `full`. PRs take 4 hours.

**Causes and fixes:**

| Cause | How to identify | Fix |
|-------|----------------|-----|
| PR touches CMakeLists.txt or *.cmake | Check `smart_build_ci.log`: "CI safety check failed" | Expected — those changes need full build |
| PR touches `script/dependency-parser/` | Same as above | Expected — any change to the tooling itself triggers full build |
| `DISABLE_SMART_BUILD=true` in env | Check job config | Remove override when done |
| `FORCE_CI=true` (nightly run) | Check job trigger | Expected for scheduled builds |

To see exactly which files triggered the safety check:
```bash
# In the build directory:
bash ../script/dependency-parser/ci_safety_check.sh 2>&1
```

### 4.2 Smart build selected 0 tests for a code change

**Symptom:** `build_mode.env = none`, but the PR clearly touches CK source.

**Step 1:** Check what files the PR changed:
```bash
git diff --name-only origin/develop HEAD -- projects/composablekernel
```

**Step 2:** Check if those files are in the depmap:
```bash
jq '.file_to_executables | keys | length' enhanced_dependency_mapping.json   # total files tracked
jq '."file_to_executables"."include/ck/my_header.hpp"' enhanced_dependency_mapping.json
```

**Likely causes:**

| Cause | Evidence | Fix |
|-------|----------|-----|
| File path mismatch (depmap uses different key) | `jq` returns `null` | Usually a `../`-containing path; check depmap keys with `jq '.file_to_executables | keys | map(select(contains("my_header")))' dep.json` |
| File genuinely has no test dependents | `jq` returns `[]` | Not a bug — the file isn't included by any test TU |
| depmap is stale / from different commit | Check depmap timestamp vs compile_commands.json | Regenerate: delete depmap and re-run cmake-parse |

**Step 3:** If you suspect a stale depmap, regenerate it:
```bash
rm enhanced_dependency_mapping.json
python3 ../script/dependency-parser/main.py cmake-parse \
    compile_commands.json build.ninja \
    --workspace-root .. --parallel 32 \
    --output enhanced_dependency_mapping.json
```

### 4.3 smoke_result.json verdict: fail

**Symptom:** `smoke_result.json` contains `"verdict": "fail"` and lists
`invalid_targets`.

This means the selector emitted an executable name that ninja doesn't know.
The selected target won't build. Causes:

| Cause | Fix |
|-------|-----|
| Depmap built with different cmake config than current build | Regenerate depmap |
| Target was renamed in CMakeLists.txt but depmap not refreshed | Regenerate depmap |
| Path normalization mismatch (`bin/test_x` vs `test_x`) | Check `ninja -t targets all \| grep test_x` |

The build will still attempt to proceed (smoke gate is non-fatal in production
mode) but the missing targets will cause `ninja` to fail with "unknown target".

### 4.4 reachability_result.json shows new false_negatives

**Symptom:** `reachability_result.json` → `"verdict": "fail"`, `false_negatives`
contains test names that weren't there before.

This means a compiled test registered with ctest is not reachable from any file
in the depmap. If a developer adds a new test and its source files aren't being
tracked, the filter can never select it.

**Triage:**
```bash
# Is there actually a bin/ target for the test?
ninja -t targets all | grep <test_name>

# Is the test's source in compile_commands.json?
jq '.[] | select(.file | contains("<test_name>"))' compile_commands.json

# Does the depmap have any entry for the test's source files?
jq '.file_to_executables | to_entries | map(select(.value | contains(["<test_name>"])))' dep.json
```

**Short-term fix:** Add to the allowlist file (see §5) to suppress the alarm
while the extraction gap is fixed.

**Long-term fix:** The gap is usually that `clang -MM` failed silently for that
TU (often a missing include, HIP-specific macro, or generated header). Run the
failing TU's compile command manually with `-MM` to see the error.

### 4.5 Build fails with "unknown target"

Ninja says it doesn't know a target from `build_targets.txt`.

```bash
# What's in build_targets.txt?
cat build_targets.txt

# Is that target known to ninja?
ninja -t targets all | grep <target>
```

If the target is absent, see §4.3 (smoke_result mismatch). If ninja fails with
an error unrelated to target lookup (e.g., compile error), the smart build ran
correctly — the failure is in the actual code, not the selection.

### 4.6 `ctest -N` returns no tests (reachability guardrail skipped)

The guardrail emits:
```
⚠ ctest -N returned no tests (not yet configured or wrong CWD?) - skipping reachability guardrail
```

This means CTest isn't configured in the build directory. The guardrail is a
monitoring tool (non-fatal); the build still proceeds. But if this happens
persistently it means the build environment doesn't have CTest set up, so the
reachability signal is dark. Check that `cmake -GNinja` was run in the build dir.

---

## 5. Reachability allowlist

If a compiled test legitimately can't be tracked (e.g., a test that uses
`add_test(COMMAND bash ...)` wrapping a compiled binary, or a test in a build
component not present in this configuration), you can suppress it:

```bash
# Create/edit the allowlist (one test name per line, # for comments)
cat >> script/dependency-parser/reachability_allowlist.txt << 'EOF'
# test_foo: uses a bash wrapper; tracked separately
test_foo
EOF
```

The guardrail invocation in `smart_build_ci.sh` does **not** pass `--allowlist` by
default — you must add the flag to wire the allowlist in:
```bash
python3 "${SCRIPT_DIR}/filter_oracle.py" reachability \
    --depmap enhanced_dependency_mapping.json \
    --ctest ctest_list.txt \
    --ninja build.ninja \
    --allowlist "${SCRIPT_DIR}/reachability_allowlist.txt" \
    --output reachability_result.json
```

### Codegen blind spot (build-time generated tests)

Some tests are generated at **build** time from a script/template
(`example/ck_tile/01_fmha/generate.py`, `test/ck_tile/{layernorm2d,rmsnorm2d}/generate.py`,
`cmake/*.in`). Two consequences for selection:

- Their sources don't exist when the depmap is built (pre-build), so they look
  unreachable. `script/dependency-parser/codegen_blindspots.json` inventories each
  generator → the tests its outputs feed. `smart_build_ci.sh` passes it via
  `filter_oracle.py reachability --codegen-inventory …`, which marks those tests
  as a known codegen class — reported under `codegen_allowlisted` in
  `reachability_result.json` rather than as false negatives.
- A change to a generator **input** maps to no test via `#include` analysis, so
  `ci_safety_check.sh` treats `**/generate.py` and `cmake/*.in` as build-infra and
  forces a full build (reason: "codegen input changed"). This backstop holds even
  if the inventory lags a newly added generator.

To inspect which ctest tests the inventory currently covers:
```bash
python3 filter_oracle.py codegen-allowlist \
    --inventory codegen_blindspots.json --ctest ctest_list.txt
```

> Root-cause note (two-part — `DEPENDS` alone is not enough). The codegen
> `add_custom_command`s are **mixed**: some already carry `DEPENDS <script>` (e.g.
> `01_fmha`, `gemm_streamk`), some don't (e.g. `layernorm2d`/`rmsnorm2d`). But the
> blind spot persists *even where `DEPENDS` is present* (streamk has it and was still
> a coverage FN), because the depmap never reads ninja's `CUSTOM_COMMAND` edges — it
> uses the `#include` graph (`clang -MM` on `compile_commands.json`) + exe↔object
> (`NinjaTargetParser`), so `generate.py`/templates are never keys in
> `file_to_executables`. The proper fix is therefore: (1) complete `DEPENDS` on every
> codegen command (also fixes incremental rebuilds), **and** (2) teach the depmap to
> parse `build <out>: CUSTOM_COMMAND | <inputs>` and chain generator-input → output →
> object → exe. With both, this inventory + the `generate*.py` backstop can be retired
> (the `cmake/*.in` `configure_file` case is configure-time, so it stays). See the
> Test Filtering design page (D11).

---

## 6. Emergency overrides

### Disable smart build for a single run

Set `DISABLE_SMART_BUILD=true` in the Jenkins job parameters before triggering.
`ci_safety_check.sh` checks this variable and exits 1 (→ full build).

### Force full build permanently for a branch

Set `FORCE_CI=true` in the branch's Jenkins configuration. This is what nightly
builds do.

### Run smart_build_ci.sh locally (reproducing a CI failure)

```bash
# In the build directory (must have compile_commands.json + build.ninja):
export BUILD_DIR=$(pwd)
export WORKSPACE_ROOT=$(cd .. && pwd)
export BASE_BRANCH=develop
export PARALLEL=32

bash ../script/dependency-parser/smart_build_ci.sh
# Artifacts written to $BUILD_DIR:
#   smart_build_ci.log, tests_to_run.json, build_targets.txt,
#   build_mode.env, reachability_result.json
```

> **`BASE_BRANCH` vs `CHANGE_TARGET`:** `smart_build_ci.sh`'s `select` step uses
> `BASE_BRANCH`. `ci_safety_check.sh` resolves its base as
> `${CHANGE_TARGET:-${BASE_BRANCH:-develop}}`, so in Jenkins `CHANGE_TARGET`
> (set for PR builds) takes precedence there. For consistent local behavior, set
> `BASE_BRANCH` and leave `CHANGE_TARGET` unset.

### Inspect the depmap interactively

```bash
# Files with the most dependents (potential blast-radius headers):
jq '.file_to_executables | to_entries | sort_by(.value | length) | reverse | .[0:10] | .[] | {file: .key, n_deps: (.value | length)}' enhanced_dependency_mapping.json

# Which executables does a specific file pull in?
jq '.file_to_executables."include/ck/tensor_descriptor.hpp"' enhanced_dependency_mapping.json

# What files does a specific executable depend on?
jq '.executable_to_files."bin/test_gemm"' enhanced_dependency_mapping.json | length

# How many files does each executable depend on (top 10 most connected):
jq '.executable_to_files | to_entries | sort_by(.value | length) | reverse | .[0:10] | .[] | {exe: .key, n_files: (.value | length)}' enhanced_dependency_mapping.json
```

---

## 7. Safety check logic (ci_safety_check.sh)

The safety check forces a full build when any of these holds. All path patterns
are scoped to `projects/composablekernel/` and matched against
`git diff origin/${BASE_BRANCH}...HEAD` (three-dot: only changes unique to the
branch, so merged-in develop commits don't trigger a false positive).

| Trigger | Rationale |
|---------|-----------|
| `**/CMakeLists.txt` | Build graph may have changed; depmap is potentially stale |
| `**/*.cmake`, `**/*.cmake.in` | CMake module/config changes |
| `script/dependency-parser/**` | The tooling itself changed; can't trust its own output |
| `script/cmake/**` | CMake helper scripts |
| `**/generate*.py`, `cmake/*.in` | Codegen inputs (D11): generated sources aren't tracked pre-build, and a generator-input change maps to no test via `#include`. Glob covers `generate.py` + siblings (`generate_test_files.py`, `generate_instances.py`, …) |
| `setup.py`, `pyproject.toml` | Python build config |
| dependency cache (`cmake_dependency_mapping.json`) older than 7 days | Stale cache; force a fresh full build |
| `FORCE_CI=true` env var | Nightly/scheduled builds always run everything |
| `DISABLE_SMART_BUILD=true` env var | Manual override |

A meaningful fraction of PRs hit the CMakeLists.txt trigger and fall back to full
build — this is expected, since structural changes to the build graph need a full
rebuild. (Exact rate depends on the PR mix; measure it from archived
`build_mode.env` artifacts rather than assuming a fixed percentage.)

The script exits 0 (selective OK) or 1 (full required). Its output is captured in
the run log (`smart_build.log`, or `smart_build_ci.log` for standalone runs).

> **Note on `build_mode.env`:** `ci_safety_check.sh` writes it with an `export `
> prefix (`export SMART_BUILD_MODE=full`), but `smart_build_ci.sh` overwrites it
> afterwards without the prefix. The final archived artifact has no `export `
> prefix.

---

## 8. Key invariants

Things that must always be true for the system to be safe. If any of these is
violated, disable smart build until fixed.

1. **depmap is built from the same commit as the PR diff base.** If the depmap
   was built from an older `origin/develop` than the PR's merge base, it may
   miss new test TUs added between the two commits.

2. **compile_commands.json is fresh.** If CMake was reconfigured between depmap
   generation and ninja invocation (e.g., a `.cmake` file changed), the depmap
   is stale. The safety check catches `.cmake` changes, but be aware of this if
   running manually.

3. **The build environment matches the depmap environment.** `clang -MM` is run
   with the exact flags from `compile_commands.json`. If the ROCm version or
   include paths change between depmap generation and the build, extracted deps
   could differ from actual deps.

4. **`ninja -t targets all` is the oracle for target existence**, not `ninja -n`.
   CK uses `GLOB CONFIGURE_DEPENDS` in CMakeLists.txt which regenerates
   `build.ninja` on every ninja call, making `ninja -n` exit 0 for any target
   (real or bogus). Do not use `ninja -n` for selection validation.

5. **Selection coverage equals the configure of the build dir the depmap comes
   from.** Two distinct cases:

   - **Same-flow gated components** — configured into the *main* build dir that
     the smart-build depmap + `ctest -N` are generated from (`ck.groovy` enables
     them right before the smart-build configure): `rocm_ck`
     (`CK_ENABLE_ROCM_CK`, when `RUN_ROCM_CK_TESTS`) and the experimental builder
     (`CK_EXPERIMENTAL_BUILDER`, when `RUN_BUILDER_TESTS`). These **are** covered
     when enabled; the only risk is the depmap configure diverging from the build
     configure (keep them in lockstep).

   - **Separate-stage components** — built and tested in their *own* build dir by a
     dedicated Jenkins stage: `codegen` / `composable_kernel_host`
     (`CK_USE_CODEGEN`, gfx9; built by `build_client_examples_and_codegen_tests` in
     `codegen/build`; tests `codegen_test_*`) and `dispatcher`. **Their own stage
     covers them**, so smart-build correctly leaves them to it — they sit outside
     smart-build's scope by design. (Nuance: `codegen` embeds CK headers via
     `add_embed_library` + hiprtc, so part of its header dependence is
     runtime/embedded rather than compile-`#include`.)

   Rule of thumb: a component built **in the smart-build flow** belongs in that
   flow's depmap configure to be selectable; a component built **in its own
   stage** is covered there, independently.

6. **Toolchain assumption: clang + Ninja generator on Linux.** CK builds with
   amdclang/hipcc, and the tooling builds on that:
   - depmap extraction uses `clang -MM` / `clang-scan-deps -format make`;
   - the build-graph layer (`ninja -t targets all` oracle, `ninja -t deps` ground
     truth, `NinjaTargetParser`) relies on the **Ninja generator** (`-G Ninja`);
   - paths are normalized to git's forward-slash, case-sensitive keys.

   **Porting to MSVC** would add: an MSVC dep backend using
   `cl /sourceDependencies <out>.json` (or `/showIncludes`) alongside the make
   backend; the **Ninja generator** with `deps = msvc` (which keeps `ninja -t deps`
   and the whole build-graph layer intact — use `-G Ninja` for an MSVC build too);
   and Windows path normalization (backslashes / drive letters / case) folded to
   git's keys. The HIP resource-dir injection applies to amdclang. This applies
   only as a future MSVC port — recorded here as a portability note.

---

## 9. Cross-node (CPU build / GPU test) migration

The pipeline is split into `smart_build.sh` and `smart_test.sh` so the test phase
can eventually run on a different (GPU) node than the build phase (CPU). Today
both run on one node; this section is the design for when they don't.

**The seam** is the `build/` directory plus the selection artifacts. For the test
phase to run elsewhere, the test node needs:
- the built test executables (`build/**/bin/*`, `build/test/**`),
- the linked instance libraries the tests load (`build/lib/**`),
- `CTestTestfile.cmake` (the whole tree — ctest reads it recursively),
- `build_mode.env` and (selective mode) `tests_to_run.json`.

`smart_test.sh` already reads `build_mode.env`/`tests_to_run.json` and assumes the
`build/` dir is present, so the script side needs no change — only the *transport*
of `build/` between nodes.

**Options for carrying `build/` (pick per infra):**

1. **Shared workspace (simplest, preferred if available).** If the CPU and GPU
   nodes mount the same NFS/Lustre workspace, no transport is needed — the GPU
   node just runs `smart_test.sh` in the same `build/` dir. Watch for stale
   `RPATH`/absolute paths baked at build time; keep the path identical on both.

2. **`stash`/`unstash` the build tree.** Easiest in Jenkins but the CK `build/`
   is multi-GB (thousands of instance objects), so stash/unstash is slow and
   can dominate the time saved. If used, stash only what §9's seam list needs,
   not the `.o` files.

3. **sccache + minimal stash.** Use sccache so recompiles are cheap, and stash
   only the test binaries + libs + `CTestTestfile.cmake` + the two JSON/env
   artifacts. Smallest transport, most plumbing.

**Caveats:** test exes built on the CPU node must target the GPU arch (build with
the right `ARCH_NAME`/`-DGPU_TARGETS`); the GPU node needs the matching ROCm
runtime; and `ctest` working directory / `RPATH` must resolve the same on both
nodes. None of this changes the scripts — it's Jenkinsfile + infra work, tracked
as a follow-up.

---

## 10. Nightly filter-coverage oracle

The nightly (develop / `RUN_ALL_UNIT_TESTS=true`) does a clean full `ninja check`,
which leaves the compiler's real `#include` graph in `.ninja_deps`. A non-fatal
step then measures how well the smart-build depmap *would have* covered that real
graph — for free, whole-repo, on the actual configuration. Each arch stage archives
`coverage_result_<arch>.json` (tagged via `coverage --label`), and a post step
merges them into `coverage_aggregate.json` (`coverage-aggregate`: union of false
negatives, worst-case coverages across arches):

```
main.py cmake-parse … --workspace-root $WS --output pre_depmap.json   # the prediction
main.py parse build.ninja --workspace-root $WS                        # -> enhanced_dependency_mapping.json (ground truth, via ninja -t deps)
filter_oracle.py coverage --pre pre_depmap.json --post enhanced_dependency_mapping.json --ctest ctest_list.txt --output coverage_result.json
```

`coverage_result.json` fields (three framings — see below for which to cite):
- `coverage` — **edge-level**: covered `file→test` edges / total. Header-weighted
  (a heavily-included header is one edge per dependent test), so it's the optimistic
  bound — not the headline run-accuracy number.
- `file_coverage` — **file-level**: of source files with tests, the fraction that
  resolve to *all* their tests. The decision-relevant view (a PR changes files).
- `test_coverage` — **test-level**: of tests, the fraction with *every* source dep
  captured. Pessimistic (one missing edge fails the whole test); maps most literally
  to the "≥99% running needed tests" criterion.
- `codegen_credited` (with `--codegen-inventory`) — the file/test numbers with the
  codegen-class tests excluded (their generated sources are the §7 backstop's job,
  not the depmap's).
- `scope` — `source` (default) or `all` (with `--include-nonsource`).
- `false_negatives` — `{file: [tests]}` the real build proves but the depmap lacks;
  `tests_with_fn` — the affected test names. The lists to drive to zero.
- `n_edges_*`, `n_files_*`, `n_tests_*`, `verdict`.

This is the cheap path to the run-accuracy signal: the expensive build already
happened nightly; the diff costs `cmake-parse` (minutes) + one `ninja -t deps`
(~2s). It validates the **compile/`#include`** channel only — runtime/behavioral
deps (data files, dlopen) are not covered.

**Keys and scope (D14).** The oracle canonicalizes keys to the project root on
both sides, so `--pre` and `--post` may use different `--workspace-root`s
(`cmake-parse` keys follow its workspace-root; `ninja -t deps` keys are always
project-root). By default coverage counts **PR-editable source only** — `build/`
outputs, vendored deps (gtest under `build/_deps`) and system headers are excluded
because the pre-build depmap never tracks them (so they'd otherwise read as a flood
of spurious FNs — e.g. ~2.6% with them vs ~99.97% edge-level without). Pass
`--include-nonsource` for the raw diff. A real gfx942 build measured **file 99.81% /
test 97.72%** raw (edge-level 99.97%), and **100% / 100% crediting the codegen
backstop** — the residual FNs are all the `gemm_streamk` build-time codegen cluster.

To reproduce on any full build dir: run the three commands above in it.

---

## 11. Measuring the exit criteria

The Code Red targets for this work item are **≥99% run accuracy** (run the tests a
change needs) and **≥95% skip accuracy** (skip the tests it doesn't). The procedure
to (re)produce both numbers:

> **Measurement cadence (by design).** Two signals run on **every** build, cheaply
> and with no full compile: the **reachability guardrail** (`reachability_result.json`)
> and the **selection-validity smoke/JUnit** (`smoke_result.xml`, always-on as-if).
> The **coverage oracle runs only on the full / run-all path** (`runAllUnitTests` =
> develop branch or `RUN_ALL_UNIT_TESTS`) because it needs the real post-build
> `#include` graph — a full compile. Running it per-PR would require a full build,
> defeating selective testing, so the run-accuracy number is harvested free from the
> nightly. So: **PR builds → reachability + smoke; nightly/develop → coverage.** On a
> multi-arch run-all build each arch writes `coverage_result_<arch>.json` and a post
> step merges them into `coverage_aggregate.json` (union of FNs, worst-case
> coverages).

### Run accuracy (≥99%) — coverage oracle

The number comes from `filter_oracle.py coverage` (§10), normally harvested free
from the nightly full build. To produce it on demand from any completed build dir:

```
DP=script/dependency-parser
python3 $DP/main.py cmake-parse compile_commands.json build.ninja --workspace-root . --output pre_depmap.json
python3 $DP/main.py parse build.ninja --workspace-root .          # -> enhanced_dependency_mapping.json
ctest -N > ctest_list.txt
python3 $DP/filter_oracle.py coverage --pre pre_depmap.json --post enhanced_dependency_mapping.json \
    --ctest ctest_list.txt --codegen-inventory $DP/codegen_blindspots.json --output coverage_result.json
```

Cite the **test-level** number (`test_coverage`) against the ≥99% criterion, and the
**file-level** (`file_coverage`) as the practical "did each changed file get all its
tests" view; the edge-level `coverage` is header-weighted and overstates. `false_negatives`
/ `tests_with_fn` list what to drive to zero. With `--codegen-inventory`, the
`codegen_credited` block reports the numbers with the codegen-class excluded (those
are the §7 backstop's responsibility). Keys are canonicalized, so `--pre`/`--post`
workspace-roots need not match. A GPU isn't required — the build only needs to
**compile** (`ninja tests`), not run.
Latest (gfx942): **file 99.81% / test 97.72%** raw → **100% / 100%** crediting the
codegen backstop; all residual FNs are the `gemm_streamk` codegen cluster (§7).

### Skip accuracy (≥95%) — PR-corpus audit

```
python3 script/dependency-parser/analyze_pr_selection.py <PR> [<PR> ...] \
    --depmap enhanced_dependency_mapping.json --ctest ctest_list.txt \
    --repo ROCm/rocm-libraries --summary summary.json
```

Per PR it prints `sel` (tests selected) / `code` (CK code files changed) /
`not_in_map` (files the depmap didn't see → potential under-selection). A healthy
result is a small `sel` relative to the full suite with `not_in_map=0`; broad
core-header changes legitimately fan out. The tool reads the depmap's
`repo.workspace_root` to match paths and **warns loudly if every code file is
unmapped** (a depmap-root mismatch). It reports selection behavior + blind-spot
flags, not a ground-truth %; pair it with `validate_pr.sh` for a per-PR certificate.

Record the dated numbers on the Confluence design page ("How we measure"); keep
this runbook to the *procedure*, not point-in-time results.

---

## 12. Contacts and escalation

- Smart-build tooling is in `projects/composablekernel/script/dependency-parser/`
- Unit tests: `uv run pytest tests/` (requires `uv sync` once)
- CI integration: `projects/composablekernel/vars/ck.groovy` (search `Smart Build`)
- For a suspect false negative (test not selected but should have been): check
  `reachability_result.json` → `false_negatives`, then run the TU's compile
  command with `-MM` to see what deps `clang` actually extracts.
