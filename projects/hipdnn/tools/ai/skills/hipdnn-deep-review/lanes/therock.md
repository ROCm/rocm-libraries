# Lane: TheRock fit (`ROCK`)

TheRock builds hipDNN, the integration tests and each provider as separate subprojects and tests them from installed artifacts. Check
whether that build, packaging and test path covers the change.

## Setup

The pin is `therock-ref` in `.github/actions/ci-env/action.yml`. Every `ROCm/TheRock` `@<sha>` and `ref:` under `.github/` must equal
it; a mismatch is a finding. Read TheRock at the pin, never `main`, and `.github/` from `$CLONE`:

```bash
. ./env.sh; mkdir -p "$SCRATCH/ROCK"
pin() { git -C "$CLONE" show "$1:.github/actions/ci-env/action.yml" | "$GREP" -A2 'therock-ref:' | "$GREP" -o '[0-9a-f]\{40\}'; }
ROCK_REF=$(pin "$HEAD"); ROCK_REF_BASE=$(pin "$BASE")
[ ${#ROCK_REF} -eq 40 ] && [ ${#ROCK_REF_BASE} -eq 40 ] || { echo "STOP: cannot read therock-ref at HEAD or BASE" >&2; exit 1; }
git -C "$CLONE" -c gc.auto=0 grep -nE 'ROCm/TheRock.*@[0-9a-f]{40}|ref: .*[0-9a-f]{40}' "$HEAD" -- .github | "$GREP" -v "$ROCK_REF"
T=$SCRATCH/ROCK/therock
printf '%s\n' "$ROCK_REF" > "$SCRATCH/ROCK/ref"; printf '%s\n' "$ROCK_REF_BASE" > "$SCRATCH/ROCK/ref-base"; printf '%s\n' "$T" > "$SCRATCH/ROCK/T"
git clone -q --filter=blob:none https://github.com/ROCm/TheRock "$T" && git -C "$T" checkout -q "$ROCK_REF" || { echo "STOP: cannot check out TheRock at $ROCK_REF" >&2; exit 1; }
```

When the block stops, the lane returns `STOP: <reason>`. Later blocks begin with `. ./env.sh;
T=$(cat "$SCRATCH/ROCK/T"); ROCK_REF=$(cat "$SCRATCH/ROCK/ref"); ROCK_REF_BASE=$(cat "$SCRATCH/ROCK/ref-base")` and set `PYTHONPATH` themselves.

Each line the final `grep -v` leaves must check out another repository; a TheRock ref there is a finding. When `$ROCK_REF_BASE` differs from
`$ROCK_REF`, the change bumps TheRock: every ref must move together, and you review the bump with `git -C "$T" log --oneline
"$ROCK_REF_BASE..$ROCK_REF" -- <paths in the table below>`. Record `$ROCK_REF` in the report.

| Question | Source in `$T` |
|---|---|
| Is CI skipped | `build_tools/github_actions/configure_ci_path_filters.py`, `skip-ci-base.toml` beside it (newer pins) |
| Which subtrees changed, which jobs | `build_tools/github_actions/configure_external_repo_ci.py`, `test_tools/determine_rocm_test_dependencies.py`, `build_tools/github_actions/fetch_test_configurations.py` |
| Which families test | `build_tools/github_actions/amdgpu_family_matrix.py` (`tests_on_trigger`), `build_tools/github_actions/configure_multi_arch_ci.py` |
| How a job runs ctest | `build_tools/github_actions/test_executable_scripts/test_runner.py` |
| Flags and build options | `FLAGS.cmake`, `$T/cmake/therock_flag_utils.cmake`, `ml-libs/CMakeLists.txt`, `ml-libs/post_hook_*.cmake` |
| RPATH | `$T/cmake/therock_global_post_subproject.cmake` (`_therock_post_process_rpath_target`) |
| Artifacts and components | `BUILD_TOPOLOGY.toml`, `ml-libs/artifact-*.toml`, `build_tools/_therock_utils/artifact_builder.py` |

## 1. Does TheRock CI run at all?

Classify every changed file with TheRock's skip-ci patterns: the base patterns plus the repo's `.github/skip-ci-config.toml` at
`$HEAD`. The loop prints every file (`is_ci_run_required` stops listing at ten). When the pin has no
`load_skip_ci_config`, it falls back to `_is_path_skippable(path)` (no extension patterns); record that.

```bash
. ./env.sh; T=$(cat "$SCRATCH/ROCK/T"); export PYTHONPATH=$T/build_tools/github_actions
C="$SCRATCH/ROCK/skip-ci-config.toml"
git -C "$CLONE" show "$HEAD:.github/skip-ci-config.toml" > "$C" || { echo "no skip-ci-config at HEAD" >&2; : > "$C"; }
git -C "$CLONE" diff --name-only "$BASE" "$HEAD" | python3 -c '
import sys; from pathlib import Path; import configure_ci_path_filters as f
if hasattr(f, "load_skip_ci_config"):
    p, _ = f.load_skip_ci_config(extension_config_path=Path(sys.argv[1]))
    skip = lambda x: f._is_path_skippable_for_patterns(x, p)
else:
    print("pin predates skip-ci config: _is_path_skippable only"); skip = f._is_path_skippable
for x in sys.stdin.read().splitlines(): print("[skip]" if skip(x) else "[ci]  ", x)' "$C"
```

- **Any `[ci]`:** TheRock CI runs; continue with section 2.
- **All `[skip]`, every changed file documentation:** return `N/A: documentation only; TheRock CI skipped`.
- **All `[skip]`, some file not documentation:** TheRock CI neither builds nor tests the change; the [../policy.md](../policy.md)
  *Review rules* TheRock tightening applies. A `[skip]` file that is built or installed (it feeds an `add_subdirectory`/`install()` in
  the tree, or an `artifact-*.toml` pattern ships it) is shipped code that bypasses TheRock CI. Read the base skip list the same way
  (`git -C "$CLONE" show "$BASE:.github/skip-ci-config.toml"`). If the pattern is unchanged and the PR adds no shipped code under it,
  report FUTURE WORK only.

## 2. Which jobs, families and tier?

**Jobs.** PR CI selects test jobs from the changed subtrees, not from labels. Pass the changed subtrees (for a pushed PR, `Matched
projects:` in the configure job log; otherwise the `projects/`, `dnn-providers/` or `shared/` subtree of each non-skipped file); the
output names the job keys in `fetch_test_configurations.py`:

```bash
. ./env.sh; T=$(cat "$SCRATCH/ROCK/T")
python3 "$T/test_tools/determine_rocm_test_dependencies.py" --changed-projects "projects/hipdnn,dnn-providers/miopen-provider"
```

- `hipdnn_install` (find_package consumption of the installed packages) is never selected by a hipDNN path; it runs only when all tests
  run (run-all, nightly). A change to exported CMake config, `*Config.cmake.in` or installed headers has no PR-CI install check: record
  that and name the run that covers it.
- A new `dnn-providers/<dir>` without an `_EXTERNAL_SUBTREE_ALIASES` entry makes the resolver raise: a TheRock-side change (section 5).
- On a dispatch, a `test:<job>` label matches one job key exactly: `test:hipdnn` runs no provider, integration, samples or install job.

**Families.** Covered families = the caller's families ∩ families whose `tests_on_trigger` contains the current trigger − the job's
`exclude_family`. Triggers: `pull_request` → `presubmit`, `push` → `postsubmit`, `schedule` → `nightly`. `workflow_dispatch` tests
every requested family, at the `quick` tier. A family outside that set does not test unless a dispatch requests it or the PR carries
the labels `ci:<family>` and `ci:test:<family>` (`$T/docs/development/ci_behavior_manipulation.md`). Quote the setup job's `tests
disabled (...)` lines.

**Tier.** Scoped PR or push: `standard`. Run-all (unclassified non-skipped path, shared CI change): `quick`. `schedule`:
`comprehensive`. A family's `test_type_for_family` overrides these.

Confirm it from the `test_type` output of the `configure` job in `therock-multi-arch-ci.yml` and the `Test Artifacts (<family>,
<release_type>, <test_type>, ...)` run name of each test job. `test_runner.py` `build_ctest_command` runs `-L ^<tier>$` and excludes
`<tier>_exclude` and `<tier>_therock_ci_exclude`. Tier labels come from each project's test category YAMLs (`*test_categories*.yaml`; registering a test in them is owned by `BUILD`: hand it off, do not file it)
and cascade upward: a `quick` test also carries `standard`, `comprehensive` and `full`. hipDNN core puts every test in `quick`.
Provider and integration YAMLs do not, so a test they categorize only as `comprehensive` never runs in PR CI.

**Timeout budget (main pass).** Each tier is its own CTest entry, `<binary>_<tier>_suite`, with a `category_timeouts` value in the
project's `*test_categories*.yaml` (hipkernelprovider: quick 300 s, standard 2700 s, comprehensive 25200 s). Check three exposures for
each covered family: run-all PRs (`quick`, always, because any PR touching an unclassified path runs it), scoped PRs (`standard`) and
nightly (`comprehensive`). A family's `test_type_for_family` overrides these. The baseline comes from per-job logs, because
`gh run view --log` on a whole run can come back empty. Scan recent runs for successful jobs named `Test <job> (shard <i>/<N>) (<family>)`,
fetch each such job's log, and read the entry's `Test #<n>: <entry> ... Passed <t> sec` line:

```bash
. ./env.sh; E="$EVIDENCE/r<N>/lanes/ROCK-evidence"; mkdir -p "$E"; JOB=<job>; ENTRY=<binary>_<tier>_suite
WF=therock-multi-arch-ci.yml  # run-all and scoped; the nightly exposure uses therock-multi-arch-ci-nightly.yml
for id in $(gh run list -R <owner>/<repo> --workflow "$WF" --limit 100 --json databaseId -q '.[].databaseId'); do
  gh run view "$id" -R <owner>/<repo> --json jobs -q ".jobs[] | select(.conclusion==\"success\" and (.name|test(\"Test $JOB \"))) |
    \"$id\t\(.databaseId)\t\(.name)\"" || echo "gh run view failed: $id" >&2
done > "$E/jobs-$JOB.tsv"
while IFS=$'\t' read -r run job name; do
  gh api "repos/<owner>/<repo>/actions/jobs/$job/logs" > "$E/job-$job.log" || continue
  printf '%s\t%s\t%s\n' "$run" "$name" "$("$GREP" -oE "Test +#[0-9]+: $ENTRY[ .]+Passed +[0-9.]+ sec" "$E/job-$job.log" | "$GREP" -oE '[0-9.]+ sec')"
done < "$E/jobs-$JOB.tsv"
```

Use the latest 3 runs that ran the entry for each (family, exposure). The baseline is the maximum over those runs of the maximum
shard. With fewer than 3 runs, use the runs found and write their count. With none, or with logs `gh` cannot read (record the error),
the line is `UNVERIFIED (no CI log)`. A run-all `quick` baseline can come from any run that executed the `quick` entry. Write one line
per (entry, family, exposure) under **Checked, no finding**, with no `|`: `Timing-base: <entry> <family> <run-all|scoped|nightly>
max-shard <s>s limit <s>s (<run ids>)`, or `Timing-base: <entry> <family> <exposure> UNVERIFIED (no CI log)`. `ROCK-f1` (section 6)
adds the change's delta.

## 3. Is the change tested in TheRock?

When the change adds or alters tests or shipped code, one of these must hold, or it is an IMPORTANT finding:

- For each new or changed test, a selected job's artifact installs it, its installed labels carry the covering trigger's tier and no
  exclude label, a covered family runs it, and that job's log shows it executed (counts).
- The PR description states the author verified it in a TheRock build, with flags, arch and result.

**Flags.** Take each flag or option the change depends on, with its component default, from **Options and guards** (`CONTEXT.md`);
resolve it per workflow from four places:

| Source | Effect |
|---|---|
| The component's own CMake default in `$REPO` | Applies unless something below overrides it |
| `ml-libs/CMakeLists.txt` `CMAKE_ARGS` | Hard-sets options (e.g. `-DHIP_DNN_GENERATE_SDK_HEADERS=OFF`, mapped by the project's migration shim to `HIPDNN_GENERATE_SDK_HEADERS`). Counts only while the component still defines the option |
| `FLAGS.cmake` `THEROCK_FLAG_*` | ON injects the flag's CMake vars. OFF injects nothing, so the component default stands (`therock_flag_utils.cmake`) |
| `extra_cmake_options` in each rocm-libraries workflow | Per-run overrides: `git -C "$CLONE" grep -n THEROCK_FLAG_ "$HEAD" -- .github/workflows` (PR, nightly, ASAN, flag-on lanes) |

The code is covered only by workflows where it resolves ON and whose families test it. Count a family only from executed test counts in its job logs (`mkdir -p "$EVIDENCE/r<N>/lanes/ROCK-evidence"; gh run view <run-id> -R <owner>/<repo> --log > "$EVIDENCE/r<N>/lanes/ROCK-evidence/run-<id>.log"`, then `"$GREP"` the file; the run id is the `/runs/<id>/` segment of a check's `html_url`).

## 4. Will TheRock package and run it?

Only a TheRock-layout build applies TheRock's post-processing, which *Installed CTest*, *RPATH* and *Components* need. It is an **Approval
request**: it needs `meson` (`pip install -r requirements.txt` in `$T`), fetched sources (`python3 ./build_tools/fetch_sources.py`) and
the dist AMDGPU target selection, and builds the compiler and runtime from source (hours, tens of GB); state these, the architecture
and the duration. Without approval, those three are `UNVERIFIED`: read the sources they name and say what was not run. The approval
gate and the hipDNN-core-only scope below apply to those three only: when the change touches only `dnn-providers/**` or integration
tests, they are `N/A (hipDNN core not touched)`, or `UNVERIFIED` with a Handoff when the PR claims provider packaging changes; do
not parameterize them per provider. *Arch split*, *Target-neutral outputs* and *Windows* run without the TheRock build. Once approved,
build in `$SCRATCH/ROCK/build` from a full scratch checkout of `$HEAD`, never
`$REPO` or `$REPO_BASE` (sparse checkouts lack sources) and never `-DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR="$REPO"`:

```bash
. ./env.sh; T=$(cat "$SCRATCH/ROCK/T")
fullclone "$SCRATCH/ROCK/src" "$HEAD" || exit 1
cmake -S "$T" -B "$SCRATCH/ROCK/build" -GNinja -DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR="$SCRATCH/ROCK/src" \
  -DTHEROCK_AMDGPU_TARGETS=<arch> -DTHEROCK_DIST_AMDGPU_TARGETS=<arch> -DTHEROCK_ENABLE_ALL=OFF -DTHEROCK_ENABLE_HIPDNN=ON
ninja -C "$SCRATCH/ROCK/build" -j<cores> hipDNN+stage
```

The stage is `$SCRATCH/ROCK/build/ml-libs/hipDNN/stage`.

- **Installed CTest.** A job runs ctest in `$THEROCK_BIN_DIR/<dir>`, where `<dir>` is the job's `COMPONENT_DIR_MAPPING` entry in
  `test_runner.py`. The installed `CTestTestfile.cmake` is not the build tree's: `install_hipdnn_ctest_files` in
  `projects/hipdnn/cmake/Tests.cmake` writes only `add_test(<t> "../<t>")`, the `HIPDNN_INSTALLED_CTEST_STAGING` text and labels. Diff
  `ctest --show-only=json-v1` from the build tree against `ctest --test-dir <prefix>/bin/<dir> --show-only=json-v1`: command, args,
  `ENVIRONMENT`, `WORKING_DIRECTORY`, `FIXTURES_REQUIRED`, labels.
- **RPATH.** TheRock replaces `INSTALL_RPATH` on every executable and shared library. RUNPATH becomes
  `THEROCK_PRIVATE_INSTALL_RPATH_DIRS` relative to the target's `THEROCK_INSTALL_RPATH_ORIGIN`, or to `bin/` or `lib/` when that is
  unset; nothing is set on Windows. Project `INSTALL_RPATH` is discarded; check the TheRock layer: every new executable or shared
  library installed outside `bin/` or `lib/` sets `THEROCK_INSTALL_RPATH_ORIGIN` to its install dir (`git -C "$CLONE" grep -n
  THEROCK_INSTALL_RPATH_ORIGIN "$HEAD" -- dnn-providers projects/hipdnn` shows precedent). Any extra library dir it needs is added to
  `THEROCK_PRIVATE_INSTALL_RPATH_DIRS` by an `ml-libs/post_hook_*.cmake`. On the stage, `readelf -d` (Linux only; on macOS use `otool -L` or skip and report `UNVERIFIED`) shows no absolute RUNPATH entry,
  and `"$GREP" -rl "$SCRATCH" <stage>` finds no build path baked into installed files.
- **Components.** Save this scanner as `$SCRATCH/ROCK/components.py` and run it on the stage with the artifact TOML, the artifact name
  and the build root (the TOML's `ml-libs/hipDNN/stage` key resolves under it). Run it as `. ./env.sh; T=$(cat "$SCRATCH/ROCK/T");
  PYTHONPATH="$T/build_tools" python3 "$SCRATCH/ROCK/components.py" "$T/ml-libs/artifact-hipdnn.toml" hipdnn
  "$SCRATCH/ROCK/build"`

  ```python
  import sys; from pathlib import Path; from _therock_utils import artifact_builder as ab
  d = ab.ArtifactDescriptor.load_toml_file(Path(sys.argv[1]), artifact_name=sys.argv[2])
  s = ab.ComponentScanner(Path(sys.argv[3]), d)
  for n, c in s.components.items(): print(n, sorted(str(p) for m in c.basedir_contents.values() for p in m.all))
  print("UNMATCHED", sorted(str(r) for r, _ in s.unmatched_files))
  ```

  Every new file must land in its intended component. Unmatched files are dropped silently, because `fileset_tool.py` skips `verify()`.
  Components resolve `lib → run → dbg → dev → doc → test`, and `lib` claims every `*.so*` and `*.dll` first. A test-only library or
  plugin therefore lands in the runtime package unless an exclude stops it (`artifact-hipdnn.toml` excludes `lib/test_plugins/**`, the
  Linux layout; on Windows test plugins install under `bin/test_plugins`, which that exclude does not match). Check where each new
  test-only binary lands.
- **Arch split.** `BUILD_TOPOLOGY.toml` marks each artifact `target-neutral` or `target-specific`. Where a family has a matching
  `ex_gpu_<arch>` suite, `build_ctest_command` passes `-L ^<tier>$ -L ^ex_gpu_<arch>$`, and ctest ANDs repeated `-L`: only tests with
  both labels run. With no match it adds `-LE ex_gpu`. Confirm the issued ctest line in the job log. In hipkernelprovider, per-arch
  payloads must sit under `**/engines/arch_content/**` or `**/engines/test_arch_content/**` to be split per arch.
- **Windows.** The test job sets only `THEROCK_BIN_DIR`. The runner adds no hipDNN directory to PATH: check `COMPONENT_OVERRIDES` in
  `test_runner.py` for the job. DLLs resolve because test executables install directly into `bin/` next to them, and plugins resolve
  through `HIPDNN_PLUGIN_ROOTDIR` (`bin` on WIN32, set in `projects/hipdnn/CMakeLists.txt`). A new test executable or loader in a
  subdirectory of `bin/` needs its DLL lookup shown.

  **Path length.** Windows tests run with `THEROCK_BIN_DIR: "./build/bin"` (`test_component.yml`), relative to the job's workspace, so ctest
  starts in `<workspace>\build\bin\<dir>`; the job log names that directory (`Internal ctest changing into directory: <path>`). No CI
  step sets `LongPathsEnabled`, and no hipDNN or hipkernelprovider binary declares `longPathAware`, so a file API fails on a path over
  259 characters. Compute the paths for every change that installs files or registers installed CTest entries. Sources:
  - CTest entries: run `ctest --test-dir <tree> --show-only=json-v1` on a configure tree to get each entry's arguments, `ENVIRONMENT`
    and `WORKING_DIRECTORY`. The configure tree is the *Target-neutral outputs* step 4 tree, or one configure of the component's
    preset. Cover every installed entry, including the per-arch entries installed under `test_arch_content`
    (hipkernelprovider census shards). Keep every relative path an entry passes, `../` climbs included: a climb that reaches a
    content directory is joined raw with each file under it.
  - Installed files: use `RUN`'s prefix when it exists. Otherwise take the `file(INSTALL DESTINATION ...)` rules from
    `cmake_install.cmake`. A file named at build time (packed shards, generated descriptors) takes the longest name its producing
    rule can emit for the arches and ops the content carries; cite the rule.
  - Test-created files: cache and temp files the tests create under the install tree; add the longest such relative path to the
    install prefix.
  - Map every path to the Windows layout (executables and plugins under `bin/`).

  Length = root + 1 + the string the file API receives. That string is the raw join (the entry's working directory plus its relative
  path plus the file) unless the caller normalizes it first; cite the caller's line. When an entry passes a directory, the loader opens
  files under it, so add the longest relative path of the files installed there. Take those paths from `RUN`'s prefix, or else from
  the source content tree the install copies (for hipkernelprovider, the descriptor trees under `src/engines/*/descriptors/`). The root
  is the part of that log line before `\bin\<dir>` when one is available, else 64 characters (an allowance;
  `B:\actions-runner\_work\rocm-libraries\rocm-libraries\build` is 59). Write the longest path per entry as `Win-path: <entry or file>
  <length> (<root source>)`.

  A length over 259 is a finding unless the caller uses a `\\?\` path: BLOCKING when a required set (an engine's content or a
  plugin set) provably fails to load, else IMPORTANT. Without a tree or prefix the result is `UNVERIFIED (no tree)`. Record only
  lengths and the job id; the literal runner path goes in `ROCK-evidence/`, because a literal path in the report trips `LEAK`.
- **Target-neutral outputs.** Applies to an artifact `BUILD_TOPOLOGY.toml` marks `target-specific` (`"$GREP" -n -A4 '^\[artifacts\.<name>\]'
  "$T/BUILD_TOPOLOGY.toml"`); a target-neutral artifact is `N/A (target-neutral)`. TheRock builds such an artifact once per family and
  uploads its `*_generic` part from every family job under one name without comparing the copies (`artifact_manager.py`,
  `therock_artifacts.cmake`), so host code, headers and CTest files must not depend on `GPU_TARGETS`. Cite the rule for hipkernelprovider
  (`dnn-providers/hip-kernel-provider/descriptor-packaging/README.md`: host gates never read `GPU_TARGETS`); per-arch reads of
  `GPU_TARGETS` are legitimate when they only feed the *Arch split* globs. Four steps follow the list.

**Target-neutral outputs, step 1: flags.** List every option that guards the changed code: **Options and guards** of `CONTEXT.md`
plus the options guarding each gate step 2 finds. Record each one's value at the pin (`FLAGS.cmake` in `$T`), on TheRock main
(`git -C "$T" fetch -q origin main`, then `git -C "$T" show origin/main:FLAGS.cmake`) and in `extra_cmake_options` (the table in
section 3). For each flag that is OFF at the pin and ON on main, write under **Checked, no finding**, with no `|`:
`Flag-dep: <NAME> pin=<v> main=<v> (<FLAGS.cmake line>)`. The next pin bump builds that code. Step 4 sets every listed flag ON.

**Step 2: static trace.** List every `GPU_TARGETS`, `AMDGPU_TARGETS` and `THEROCK_AMDGPU_TARGETS` read in the changed CMake files and the
files they include, then one level of `set()`/`list()` derivations of them, then every use of the source or a derived variable that
reaches `target_sources`, `target_compile_definitions`, `configure_file` into a host header, `install()` or `add_test`. A use whose
destination is per-arch is clean: for hipkernelprovider the *Arch split* globs, for any other artifact only kpack-split device code
(`THEROCK_FLAG_KPACK_SPLIT_ARTIFACTS`). Any other use is a static hit.

```bash
. ./env.sh; T=$(cat "$SCRATCH/ROCK/T")
mapfile -t CM < <(git -C "$CLONE" diff --name-only "$BASE" "$HEAD" | "$GREP" -E '(^|/)(CMakeLists\.txt|[^/]+\.cmake)$')
[ ${#CM[@]} -gt 0 ] || { echo "no changed CMake files: static trace N/A"; exit 0; }
git -C "$CLONE" -c gc.auto=0 grep -nE 'GPU_TARGETS|AMDGPU_TARGETS|THEROCK_AMDGPU_TARGETS' "$HEAD" -- "${CM[@]}"
```

Repeat with each included file as the pathspec and with each derived variable's name as the pattern.

**Step 3: arch pair.** Two archs from distinct TheRock families (`amdgpu_family_matrix.py`): one with the gated content or feature, one
without. With no gate found in step 2, use the first presubmit family's arch plus an arch from another family the toolchain accepts.

**Step 4: configure diff.** Needs `Build host: yes`; otherwise the check is static only and a static hit is IMPORTANT. Use the
superbuild preset that builds the component (root `CMakePresets.json`; for hipkernelprovider `hip-kernel-provider`). Configure it once
per arch with every step 1 flag ON (`-D<var>=ON` for each `CMAKE_VARS` entry of the flag in `FLAGS.cmake`), `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` and `-DENABLE_CLANG_TIDY=OFF`, and, when the ingestor is ON, the
prerequisites its README names (for hipkernelprovider `HIPKERNELPROVIDER_KPACK_PYTHON_DIR` and a Python with `msgpack` and `zstandard`). When a gate reads
DVC content, first run the pull the component's README names; if it fails the result is `UNVERIFIED (content not pulled)` plus the static
finding. Record whether the pull ran. Set `A1=<arch1>` and `A2=<arch2>` to the step 3 archs:

```bash
. ./env.sh; R="$SCRATCH/ROCK"; A1=<arch1>; A2=<arch2>; fullclone "$R/src" "$HEAD" || exit 1
for A in "$A1" "$A2"; do
  nice -n 19 cmake --preset <preset> -S "$R/src" -B "$R/cfg-$A" -DGPU_TARGETS="$A" <-D<var>=ON for each flag var> \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DENABLE_CLANG_TIDY=OFF > "$R/cfg-$A.log" 2>&1 ||
    { tail -n 30 "$R/cfg-$A.log"; echo "STOP: configure failed for $A" >&2; exit 1; }
done
norm() { python3 -c '
import hashlib, json, pathlib, re, subprocess, sys
b = pathlib.Path(sys.argv[1]).resolve().as_posix()
cc = json.loads(pathlib.Path(b, "compile_commands.json").read_text())
inc = set()
for e in sorted(cc, key=lambda e: e["file"]):
    cmd = e.get("command") or " ".join(e["arguments"])
    inc.update(x for x in re.findall(r"-(?:I|isystem) ?(\S+)", cmd) if x.startswith(b))
    print("CC", e["file"].replace(b, "<B>"), *sorted({d.replace(b, "<B>") for d in re.findall(r"-D\S+", cmd)}))
for p in sorted(pathlib.Path(b).rglob("cmake_install.cmake")):
    for m in re.finditer(r"file\(INSTALL DESTINATION \"([^\"]*)\"[^\n]*? FILES\b(.*?)\)\n", p.read_text(), re.S):
        for f in re.findall(r"\"([^\"]*)\"", m.group(2)): print("INSTALL", m.group(1), f.replace(b, "<B>"))
t = json.loads(subprocess.run(["ctest", "--test-dir", b, "--show-only=json-v1"], capture_output=True, text=True).stdout or "{}")
for x in sorted(t.get("tests", []), key=lambda x: x["name"]):
    lab = [p["value"] for p in x.get("properties", []) if p["name"] == "LABELS"]
    print("TEST", x["name"], *[a.replace(b, "<B>") for a in x.get("command", [])], "LABELS", lab)
for p in sorted(pathlib.Path(b).rglob("CTestTestfile.cmake")):
    for l in p.read_text().splitlines():
        if l.startswith("add_test("): print("ADD_TEST", l.replace(b, "<B>"))
for d in sorted(inc):
    for h in sorted(pathlib.Path(d).rglob("*")):
        if h.is_file() and h.suffix in (".h", ".hpp", ".hxx", ".inc"):
            print("HDR", h.as_posix().replace(b, "<B>"), hashlib.sha256(h.read_bytes().replace(b.encode(), b"<B>")).hexdigest()[:16])' "$1"; }
PERARCH='arch_content|hkp-census-'   # hipkernelprovider; set per artifact
for A in "$A1" "$A2"; do norm "$R/cfg-$A" | "$GREP" -vE "^(INSTALL|TEST|ADD_TEST) .*($PERARCH)" > "$R/norm-$A.txt"; done
diff "$R/norm-$A1.txt" "$R/norm-$A2.txt"
```

`PERARCH` matches what the artifact splits per arch: its per-arch install globs and the names of the CTest entries it installs under
them. For hipkernelprovider these are `arch_content` and the census entries `hip-kernel-provider-hkp-census-<arch>-<suite>`, which install
under `test_arch_content`. For any other artifact, derive the pattern from its *Arch split* globs. The diff compares, per host target, the sorted `-D` set and source list
of `compile_commands.json`; every file of each `file(INSTALL DESTINATION ...)` of `cmake_install.cmake`; each CTest entry (name, command
arguments, labels) from `ctest --show-only=json-v1` plus the `add_test()` lines of each `CTestTestfile.cmake` (the JSON omits the command of
an executable that is not built); and a hash of each header (`.h`, `.hpp`, `.hxx`, `.inc`) under the build tree's include directories, which
covers `configure_file` outputs. The build directory is replaced by `<B>`. The filter applies only to install and test lines, because host
compile commands legitimately carry per-arch install paths in their definitions.

A difference that survives the filter is a confirmed dependence, with one exception. A `TEST` or `ADD_TEST` difference counts only for an
entry that is installed: it appears in the installed CTest file (hipkernelprovider `install_CTestTestfile.cmake`) or in an install rule.
A build-tree-only entry ships in no artifact; record it under **Checked, no finding** with that reason.

**Target-neutral outputs, tier** ([../policy.md](../policy.md) *Review rules*): BLOCKING when the diff confirms a difference and TheRock builds
it: unguarded, or behind a flag that is ON at the pin or on TheRock main (the next pin bump builds it). IMPORTANT for a static-only hit (no
build host, or content not pulled) and for a hit behind a flag that is OFF both at the pin and on main. A gated install into a component the
artifact does not request (hipkernelprovider `share/doc`: `ml-libs/CMakeLists.txt` requests no `doc` component) is SUGGESTION, with that
reason. Record the two archs, the preset and whether the pull ran under **Checked, no finding**.

## 5. TheRock-side changes

List each TheRock file the change needs:

| Change | TheRock file |
|---|---|
| New test job or suite | `build_tools/github_actions/fetch_test_configurations.py`, `build_tools/github_actions/test_executable_scripts/test_runner.py` `COMPONENT_DIR_MAPPING` |
| New `dnn-providers/<dir>` | `test_tools/determine_rocm_test_dependencies.py` (`_EXTERNAL_SUBTREE_ALIASES`, and `_CI_TEST_SELECTOR_ALIASES` for the job key), `test_tools/therock_consumer_graph.json` (generated by `build_tools/generate_consumer_graph.py`), `test_tools/test_policies.toml`. An alias alone selects nothing |
| New installed files or dirs, or a new artifact | `ml-libs/artifact-*.toml`; `BUILD_TOPOLOGY.toml` for an artifact |
| New option value, RPATH dir or flag | `ml-libs/CMakeLists.txt`, `ml-libs/post_hook_*.cmake`; `FLAGS.cmake` for a flag |

rocm-libraries CI uses only the pinned TheRock: a companion TheRock PR has no effect until the pin moves. Report the companion PR link
and whether the pin contains it: `git -C "$T" merge-base --is-ancestor <companion-sha> "$ROCK_REF"`, where `<companion-sha>` is its
merge commit. A needed TheRock change with no companion PR, or a pin that does not contain it, means the new pieces are not built,
packaged or selected by CI.

## 6. Follow-up `ROCK-f1`: timeout budget

Runs after `RUN` returns, only when `ROCK` wrote a `Timing-base:` line. Read the `ROCK` and `RUN` reports, write the `ROCK-f1` lane
report ([../findings.md](../findings.md) *Lane report*) and do no other check. It adds the change's measured time to the baselines
`ROCK` took from CI and projects each entry per shard. It runs no test.

1. **Cases.** Added, altered and removed cases are the ones the test-source diff `git -C "$CLONE" diff "$BASE" "$HEAD"` adds, edits or
   deletes (each parametrized instance counts). Per-case times are the `[       OK ] <case> (<ms> ms)` lines in the logs that `RUN`'s suite rows cite (and in logs of
   approved remote runs), at `$HEAD` and at `$BASE`. Count only cases within the entry's filter that execute on that family; a skipped
   case counts 0. A case behind a flag (its guard in the source or in **Options and guards** of `CONTEXT.md`) counts when the flag is ON at
   the pin or on TheRock main (`ROCK`'s `Flag-dep:` line gives pin and main; `RUN`'s `Flag:` lines say what ran); when it is ON only
   on main, label the `Timing:` line `pin-bump` and tier it as below.
2. **Delta** (seconds, over all cases that differ between `$BASE` and `$HEAD`):
   - added case: + its head time;
   - altered case: + (head time − base time) when both were measured; otherwise + the head time, labelled `upper bound`;
   - removed or moved-out case: − its base time when measured, otherwise 0;
   - a time measured on another arch: label `estimate (<arch>)`;
   - an added or altered case with no time: `UNVERIFIED (no case times)` for that entry; write no `Timing:` line for it.
3. **Bound.** `N` is 1 for `quick` (`fetch_test_configurations.py` forces one shard) and for a component outside
   `GTEST_ONLY_SHARDING_COMPONENTS` (`test_runner.py`; an entry there runs in one shard); otherwise `total_shards_dict[<os>]` at
   `$ROCK_REF` (keyed by OS). With gtest-only sharding every shard runs every entry, and gtest assigns cases to shards by index, so the
   even share alone is not a bound:

   `bound = base + delta / N + longest`

   where `base` is `max-shard` from the `Timing-base:` line and `longest` is the largest single added or altered case's contribution
   (its head time when added; its increase when altered). For every job, also sum every entry's `Passed` time in the same job log for
   that exposure, add delta / N, and compare the total with the job's `timeout_minutes` (`fetch_test_configurations.py`).
4. **Line.** Write one line per (entry, family, exposure) under **Checked, no finding**, with no `|`: `Timing: <entry> <family>
   <exposure> base <s>s delta <s>s (<source>) bound <s>s/<N> shards limit <s>s`. Append any labels that apply (`pin-bump`, `upper
   bound`, `estimate (<arch>)`). `<source>` names the `RUN` log rows.
5. **Finding.** A `bound` above `limit`, or a job total above `timeout_minutes`, is a finding, tiered by [../policy.md](../policy.md) *Review rules*:
   - BLOCKING for the `run-all` (`quick`) or `scoped` (`standard`) exposure on a presubmit family, `pin-bump` cases included;
   - IMPORTANT for the `nightly` (`comprehensive`) exposure.

   `Where` cites the entry's `category_timeouts` line and the case. A `Timing-base:` line
   with `UNVERIFIED (no CI log)` carries through as `UNVERIFIED`, never as a pass.
