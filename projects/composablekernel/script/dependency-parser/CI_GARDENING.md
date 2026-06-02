# Smart Build — CI Gardening Guide

This document is for people who maintain, debug, and operate the CK smart-build
CI pipeline. It covers the decision flow, what every artifact means, how to
diagnose a misbehaving run, and how to safely override or disable the system.

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

```
smart_build_ci.sh
│
├─ ci_safety_check.sh exits 1?  ──YES──► full build  (build_mode.env = full)
│    (CMakeLists, *.cmake, or
│     dependency-parser files
│     changed in this PR)
│
├─ compile_commands.json missing? ──YES──► exit 1 (CMake not configured)
├─ build.ninja missing?           ──YES──► exit 1
│
├─ cmake-parse (main.py cmake-parse) produces enhanced_dependency_mapping.json
│
├─ Step 2b: reachability guardrail (non-fatal observability)
│    ctest -N → filter_oracle.py reachability
│    → reachability_result.json
│    verdict FAIL = some compiled tests are unreachable from any file in the
│                   depmap (filter can never select them → guaranteed FN if
│                   those tests' sources change). Does NOT abort the build.
│
├─ main.py select → tests_to_run.json
│
├─ 0 tests selected? ──YES──► exit 0  (build_mode.env = none)
│                              (no build, no test run)
│
└─ build targets extracted → build_targets.txt
                              build_mode.env = selective
```

---

## 3. Artifact reference

Every artifact is archived by Jenkins (`allowEmptyArchive: true`). Open the
build's **Artifacts** tab and look for these files.

| File | What it means | First thing to check |
|------|--------------|----------------------|
| `build_mode.env` | `SMART_BUILD_MODE=selective\|full\|none` | Which path the run took |
| `smart_build_ci.log` | Full stdout+stderr of `smart_build_ci.sh` | Start here for any CI failure |
| `smart_build.log` | Full stdout+stderr of `smart_build_and_test.sh` | Build/test failures |
| `tests_to_run.json` | The selected executables and ctest regex | Which tests were chosen |
| `build_targets.txt` | Space-separated ninja targets | What was actually built |
| `reachability_result.json` | Guardrail verdict — which ctest tests are unreachable | Ongoing FN monitoring |
| `smoke_result.json` | Whether every selected target is a real ninja target | Selection drift detection |
| `smoke_result.xml` | JUnit version of `smoke_result.json` | Published to Jenkins test results |

### `build_mode.env` values

| Value | Meaning |
|-------|---------|
| `selective` | Smart build ran, N tests selected and built |
| `full` | Safety check forced a full `ninja check` |
| `none` | Smart build ran, 0 tests selected (change touches no CK code) |

If the file is **absent**, `smart_build_ci.sh` crashed before it could write it
(e.g., CMake not configured). Check `smart_build_ci.log`.

### `reachability_result.json` fields

```json
{
  "n_ctest": 651,          // total ctest-registered tests
  "n_reachable": 614,      // tests the filter can possibly select
  "n_false_negatives": 0,  // compiled tests the filter can NEVER select (alarm if >0)
  "false_negatives": [],
  "n_non_compiled": 37,    // python/try_compile tests — always-run class, not FNs
  "non_compiled": [...],
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
jq 'keys | length' enhanced_dependency_mapping.json   # total files tracked
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

Pass it to the guardrail invocation in `smart_build_ci.sh`:
```bash
python3 "${SCRIPT_DIR}/filter_oracle.py" reachability \
    --depmap enhanced_dependency_mapping.json \
    --ctest ctest_list.txt \
    --ninja build.ninja \
    --allowlist "${SCRIPT_DIR}/reachability_allowlist.txt" \
    --output reachability_result.json
```

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

The safety check forces a full build when the PR touches any of:

| Pattern | Rationale |
|---------|-----------|
| `CMakeLists.txt`, `*.cmake` | Build graph may have changed; depmap is potentially stale |
| `script/dependency-parser/**` | The tooling itself changed; can't trust its own output |
| `FORCE_CI=true` env var | Nightly/scheduled builds always run everything |
| `DISABLE_SMART_BUILD=true` env var | Manual override |

**~47% of corpus PRs trigger full build** due to CMakeLists.txt touches. This is
expected — structural changes to the build graph need the full rebuild.

The script exits 0 (selective OK) or 1 (full required). Its output is captured
in `smart_build_ci.log`.

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

---

## 9. Contacts and escalation

- Smart-build tooling is in `projects/composablekernel/script/dependency-parser/`
- Unit tests: `uv run pytest tests/` (requires `uv sync` once)
- CI integration: `projects/composablekernel/vars/ck.groovy` (search `Smart Build`)
- For a suspect false negative (test not selected but should have been): check
  `reachability_result.json` → `false_negatives`, then run the TU's compile
  command with `-MM` to see what deps `clang` actually extracts.
