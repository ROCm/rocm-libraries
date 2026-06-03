---
name: therock-integrate
description: "Set up and launch a TheRock multi-arch integration CI run for a rocm-libraries feature. Creates a TheRock integration branch, bumps the rocm-libraries submodule to a branch or commit, enables the hipDNN/provider feature flags needed to exercise the feature, pushes the branch, and kicks off a multi-arch CI run on the GPU architectures (ASICs) and projects the user wants. Use when the user wants to integration-test a rocm-libraries branch or commit inside TheRock and run it through multi-arch CI. Discovers the available feature flags, GPU families, and test targets from the current rocm-libraries and TheRock trees instead of relying on a fixed list; infers the right choices when it can and asks the user when it cannot."
argument-hint: "[branch=<rocm-libraries-branch>] [sha=<commit>] [therock-branch=<name>] [base=<therock-base>] [flags=<NAME=ON,...>] [families=<gfx...,...>] [tests=<test:...,...>] [windows] [therock-root=<path>]"
allowed-tools: Bash, Read, Grep, Glob, Edit, Write
---

# TheRock Integrate

Use this skill when the user wants to integration-test a rocm-libraries feature inside TheRock and put it through multi-arch CI. The end goal is a launched multi-arch CI run, so do not stop after configuring the branch. The skill:

1. Creates an integration branch in TheRock.
2. Bumps the `rocm-libraries` submodule to the requested branch or commit.
3. Enables the feature flags needed to exercise the feature.
4. Pushes the branch and **kicks off a multi-arch CI run on the GPU architectures (ASICs) and projects the user wants.**

The set of feature flags, providers, GPU families, and test targets is **not** baked into this skill. Component CMake options, the TheRock flag registry, the GPU-family matrix, and the CI test-job list all change over time. Always discover the current state of the trees and either infer the correct choices or ask the user. Never assume a flag, family key, test label, or workflow name exists or is spelled a certain way without checking.

## Inputs

Infer these from the user request; ask for anything genuinely ambiguous, and always confirm the ASICs and projects before launching CI.

- **rocm-libraries ref**: a branch name or explicit commit SHA to pin TheRock's submodule to.
  - Default branch: the branch currently checked out in the rocm-libraries working tree (`git rev-parse --abbrev-ref HEAD`).
  - The ref must be reachable on `origin` (`https://github.com/ROCm/rocm-libraries`) so TheRock — and CI — can fetch it. If the branch is only local, tell the user to push it first.
- **TheRock integration branch name**: the branch this skill creates and pushes. If not given, propose one and confirm. Match existing precedent — `users/<user>/<topic>` or `integrate/<topic>`. CI dispatch requires this branch on the `ROCm/TheRock` origin (not a fork).
- **TheRock base branch**: the branch the integration branch is cut from. Default to TheRock's remote default branch (`git -C <therock> symbolic-ref --short refs/remotes/origin/HEAD`, commonly `origin/main`).
- **Feature flags override** (optional): an explicit `flags=NAME=ON,...` list. Honor it but still add any prerequisite flags from the interdependency discovery and tell the user what was added.
- **ASICs / GPU families**: the GPU families to build and test in CI (the "projects on ASICs" the user wants). Required before launching — confirm with the user. Resolve names against the live family matrix (see step 9); never invent family keys.
- **Projects / test scope**: which projects to exercise. This maps to the CI `test:<key>` labels. Derive from the feature(s) under test, discover the valid labels (step 9), and confirm. Empty means the full default test set for the chosen families.
- **Windows**: include Windows families in the run only if the user asks; default to Linux-only.
- **TheRock root**: the TheRock checkout that carries `rocm-libraries` as a submodule. Auto-locate; ask only if it cannot be found.

## Locate the repositories

1. Find the rocm-libraries working tree (the repo holding `projects/hipdnn` and `dnn-providers/`):
   ```bash
   git rev-parse --show-toplevel
   ```
2. Find the TheRock root. TheRock embeds rocm-libraries as a submodule. Confirm a candidate via its `.gitmodules`:
   ```bash
   grep -A2 'submodule "rocm-libraries"' <therock-root>/.gitmodules
   ```
   The entry's `path` is the submodule location and `branch` is the ref `--remote` bumps follow. If no candidate is known, ask the user rather than guessing.

## Workflow

1. **Resolve refs, roots, and inputs.** Determine the rocm-libraries ref, the TheRock root and base branch, and the requested ASICs and project/test scope. Activate the project Python environment the active workspace expects before running TheRock Python tooling (never call bare `python`).

2. **Verify the ref is fetchable.** For a branch: `git -C <rocm-libraries> ls-remote --exit-code origin <branch>`. For a SHA: confirm it exists on a pushed branch. If it is not on `origin`, stop and have the user push it — a submodule gitlink pointing at an unpushed commit will not resolve for CI.

3. **Create the integration branch in TheRock.**
   ```bash
   git -C <therock> fetch origin
   git -C <therock> switch -c <therock-branch> <base>
   ```

4. **Bump the rocm-libraries submodule (pristine first, then patches).** TheRock applies a local patch stream and marks the submodule path `skip-worktree` after patching, so commit the gitlink *before* patches are applied:
   ```bash
   git -C <therock>/<submodule-path> fetch origin
   git -C <therock>/<submodule-path> checkout <branch-or-sha>
   git -C <therock> add <submodule-path>
   git -C <therock> commit -m "Bump rocm-libraries to <ref> for integration testing"
   python ./build_tools/fetch_sources.py     # run from <therock>; re-applies patches + validates
   ```
   If `fetch_sources.py` fails applying patches, the target ref conflicts with `patches/<patch-tag>/rocm-libraries/`. Follow the conflict-resolution flow in TheRock's `docs/development/git_chores.md`; surface the conflict to the user rather than forcing past it.

5. **Discover the feature flags and their interdependencies.** Do not use a memorized list. Build the picture fresh:
   - **Component options (rocm-libraries side).** Enumerate the build-time feature gates, then filter out generic toggles (tests, sanitizers, coverage, clang-format/tidy, doc/header generation):
     ```bash
     grep -rnE 'option\(|cmake_dependent_option\(|set\([A-Z0-9_]+ .*CACHE BOOL' \
       <rocm-libraries>/projects/hipdnn <rocm-libraries>/dnn-providers \
       --include=CMakeLists.txt --include='*.cmake'
     ```
     Record each feature flag's exact name, owning component, default, and the compile definition it sets.
   - **Interdependencies (the part that bites).** A provider feature may require a hipDNN feature to be ON. Find these by searching each provider for references to another component's flag and for guard/skip patterns:
     ```bash
     grep -rnE 'HIPDNN_[A-Z0-9_]+|message\((WARNING|FATAL_ERROR)|if\(NOT |target_sources' \
       <rocm-libraries>/dnn-providers --include=CMakeLists.txt --include='*.cmake'
     ```
     A provider that warns-and-skips or fatals when another component's flag is OFF, or that conditionally adds sources behind it, depends on that flag. Treat this as a worked example to rediscover, not a fixed rule: hip-kernel-provider's ASM SDPA engine tests reference hipDNN's `HIPDNN_ENABLE_SDPA` and are skipped unless hipDNN was built with it ON, so enabling that provider's SDPA path requires enabling hipDNN's SDPA flag. Watch for flags whose defaults disagree across components (e.g. a hipDNN feature defaulting OFF while a provider engine that needs it defaults ON) — that combination silently drops coverage.
   - **TheRock surfacing.** Read TheRock's flag registry and integration file to learn how a rocm-libraries flag reaches a subbuild:
     - `FLAGS.cmake` — each `therock_declare_flag(NAME ... CMAKE_VARS VAR=VALUE SUB_PROJECTS <target>)` defines a `THEROCK_FLAG_<NAME>` cache var and forwards `VAR=VALUE` into the named subproject(s) when ON. Note which flags already exist and their `SUB_PROJECTS`.
     - `ml-libs/CMakeLists.txt` — each component is declared via `therock_cmake_subproject_declare(<target> ...)` gated by a `THEROCK_ENABLE_<COMPONENT>` feature. This is the subproject name to use as `SUB_PROJECTS` and tells you whether a component is wired into TheRock at all. A component present in rocm-libraries but absent here is not built by TheRock; flag that to the user.

6. **Decide which flags to enable.**
   - If the user gave an explicit `flags` list, start from it.
   - Otherwise infer the feature(s) under test from what the branch changed:
     ```bash
     git -C <rocm-libraries> diff --name-only <base>...<branch>
     ```
     Map the changed component(s) to their feature flag(s) from step 5.
   - Always add transitive prerequisites discovered in step 5 (e.g. a provider feature pulling in the hipDNN flag it needs).
   - When the mapping is ambiguous, when a changed area has no obvious feature flag, or when several feature sets fit, ask the user which features to enable and confirm the resolved flag list before changing anything.

7. **Apply the flags in TheRock (hybrid mechanism).** For each flag to enable:
   - **Already declared in `FLAGS.cmake`** → enable it through the branch override file, the lowest-friction mechanism. Create or update `BRANCH_CONFIG.json` at the TheRock root:
     ```json
     { "flags": { "<FLAG_NAME>": true } }
     ```
     Values may be JSON booleans (mapped to `ON`/`OFF`) or the strings `"ON"`/`"OFF"`. `BRANCH_CONFIG.json` is gitignored on the default branch, so stage it with `git -C <therock> add -f BRANCH_CONFIG.json` to commit it on the integration branch.
   - **Not yet declared in `FLAGS.cmake`** → add a `therock_declare_flag` block modeled on the existing ones, then enable it via `BRANCH_CONFIG.json` as above. Set `CMAKE_VARS <VAR>=ON` to the component option name from step 5 and `SUB_PROJECTS` to the subproject target from `ml-libs/CMakeLists.txt`:
     ```cmake
     therock_declare_flag(
       NAME <FLAG_NAME>
       DEFAULT_VALUE OFF
       DESCRIPTION "<what it enables>"
       CMAKE_VARS
         <COMPONENT_OPTION>=ON
       SUB_PROJECTS
         <subproject-target>
     )
     ```
   - If the feature needs more than a flag (e.g. a per-arch artifact split via `BUILD_TOPOLOGY.toml` or a `ml-libs/CMakeLists.txt` change), make those edits too — discover them from what the feature requires, and tell the user.
   - Commit the TheRock-side changes (`FLAGS.cmake`, `BRANCH_CONFIG.json`, topology, etc.) with a message naming the flags and the reason (integration testing of `<ref>`).

8. **(Optional) Validate the configuration locally before burning CI.** Re-sync, configure, and confirm the flags report ON. Keep full output in a log and show only a short tail on failure, per the active workspace's build-output policy:
   ```bash
   python ./build_tools/fetch_sources.py
   cmake -B <build-dir> -GNinja . -DTHEROCK_AMDGPU_FAMILIES=<family> > <log> 2>&1
   ```
   Inspect the "Build flags" report TheRock prints at configure time (or grep the log) to verify each intended `THEROCK_FLAG_<NAME>` is ON. Do not run a full `cmake --build`.

9. **Discover the CI parameters (no hardcoding).** Before launching, resolve the ASICs and test scope against the live CI definitions:
   - **Confirm the entrypoint workflow exists.** The multi-arch entrypoint is normally `multi_arch_ci.yml`; verify it is present and read its `workflow_dispatch` inputs (the family and test-label input names can change):
     ```bash
     ls <therock>/.github/workflows/ | grep -i 'multi_arch_ci'
     gh workflow list --repo ROCm/TheRock
     ```
     Read the chosen workflow's `on.workflow_dispatch.inputs` to get the exact input names for Linux/Windows families and test labels.
   - **Valid GPU family keys.** Read the family matrix and use its top-level keys (lowercase, suffix-less), not the build-string `family:` values:
     ```bash
     grep -nE '^[[:space:]]*"?gfx[0-9a-z]+"?[[:space:]]*:' \
       <therock>/build_tools/github_actions/amdgpu_family_matrix.py
     ```
     A family with an empty `test-runs-on`/test-runner field is build-only (its tests are skipped — no hardware). Map the user's requested ASICs to valid keys; reject and re-ask on any unknown name (a typo fails the whole run). Special values: `all` (every family), `none`/`""` (skip that platform).
   - **Valid test labels.** The project/test scope is expressed as `test:<key>` labels; discover the available job keys:
     ```bash
     grep -nE '"job_name"|^[[:space:]]*"[a-z0-9_]+":' \
       <therock>/build_tools/github_actions/fetch_test_configurations.py
     ```
     Map the projects the user wants (e.g. hipDNN + providers) to the matching `test:<key>` labels. If unsure which labels cover the feature, ask the user. Leaving the test labels empty runs the full default suite for the chosen families.
   - **(Optional) dry-run the matrix** to validate family/label names without spending CI minutes, using the workspace venv python and the env-var contract of `build_tools/github_actions/configure_multi_arch_ci.py` (family vars, test-label vars). Confirm names resolve before dispatching.

10. **Push the integration branch and launch multi-arch CI.** Confirm the ASIC list and project/test scope with the user, then:
    ```bash
    git -C <therock> push -u origin <therock-branch>
    gh workflow run <multi-arch-workflow> \
      --repo ROCm/TheRock \
      --ref <therock-branch> \
      -f <linux-families-input>="<gfx-keys>" \
      -f <windows-families-input>="<gfx-keys-or-empty>" \
      -f <linux-test-labels-input>="<test:...,...>"
    ```
    Use the exact input names discovered in step 9. Then locate and report the run so the user can follow it:
    ```bash
    gh run list --repo ROCm/TheRock --workflow <multi-arch-workflow> --branch <therock-branch> --limit 5
    ```
    Report the run URL. Offer to `gh run watch --repo ROCm/TheRock <run-id>` if the user wants to follow it live; do not block on it otherwise.

## Report

Summarize:

- The rocm-libraries ref pinned (branch or SHA) and how it was sourced.
- The TheRock integration branch created, its base, and that it was pushed to origin.
- The flags enabled, each tagged with the mechanism used (`BRANCH_CONFIG.json` override vs. new `therock_declare_flag` in `FLAGS.cmake`), and any prerequisite flags added because of an interdependency. Note any topology/subproject edits made.
- The multi-arch CI run launched: the ASICs/families, the project/test scope, the workflow used, and the run URL.
- Any points where the user was asked to confirm features, ASICs, or test scope.
- Follow-ups: unpushed refs, patch conflicts, build-only families (no test HW), or components present in rocm-libraries but not wired into TheRock.

## Notes

- The headline deliverable is a launched multi-arch CI run on the ASICs and projects the user wants. Configuring the branch and flags is the setup; do not stop before CI is kicked off (unless the user only asked for the setup).
- Never hardcode flags, providers, GPU family keys, test labels, or the workflow name. Rediscover them every run (steps 5 and 9); all of these drift.
- The interdependency search is a heuristic. When a provider feature and a hipDNN feature share a concept (SDPA being the canonical case), verify the dependency direction in the CMake/source and enable the prerequisite too.
- Build-side component selection is branch state (flags, `BRANCH_CONFIG.json`, topology), not a CI input. Test-side selection is the `test:<key>` labels. A run still compiles the full dependency chain because ml-libs depends on it.
- CI dispatch requires the branch on the `ROCm/TheRock` origin (a fork cannot dispatch the upstream workflow) and Actions permission on that repo. No open PR is required for `workflow_dispatch`.
- An unknown family key fails the whole run; validate against the live matrix first. Families with no test runner build but do not test.
- Bumping a submodule against a ref that conflicts with TheRock's `patches/` requires manual resolution; do not force past `fetch_sources.py` patch failures.
- `BRANCH_CONFIG.json` only enables flags that exist in `FLAGS.cmake`; a brand-new flag must be declared there first.
- Components in `dnn-providers/` that are not declared as subprojects in TheRock's `ml-libs/CMakeLists.txt` are not part of TheRock's build. Wiring a new provider into TheRock is a larger change than this skill performs — surface it to the user.
