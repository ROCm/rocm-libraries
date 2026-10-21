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
git clone -q --filter=blob:none https://github.com/ROCm/TheRock "$T" && git -C "$T" checkout -q "$ROCK_REF" || exit 1
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

The code is covered only by workflows where it resolves ON and whose families test it. Count a family only from executed test counts in its job logs (`gh run view <run-id> -R <owner>/<repo> --log > "$EVIDENCE/r<N>/lanes/ROCK-evidence/run-<id>.log"`, then `"$GREP"` the file; the run id is the `/runs/<id>/` segment of a check's `html_url`).

## 4. Will TheRock package and run it?

Only a TheRock-layout build applies TheRock's post-processing, which the RPATH and component checks need. It is an **Approval
request**: it needs `meson` (`pip install -r requirements.txt` in `$T`), fetched sources (`python3 ./build_tools/fetch_sources.py`) and
the dist AMDGPU target selection, and builds the compiler and runtime from source (hours, tens of GB); state these, the architecture
and the duration. Without approval, *Installed CTest*, *RPATH* and *Components* are `UNVERIFIED`: read the sources they name and say
what was not run. This section covers hipDNN core only: when the change touches only `dnn-providers/**` or integration tests, those
three checks are `N/A (hipDNN core not touched)`, or `UNVERIFIED` with a Handoff when the PR claims provider packaging changes; do
not parameterize them per provider. Once approved, build in `$SCRATCH/ROCK/build` from a full scratch checkout of `$HEAD`, never
`$REPO` or `$REPO_BASE` (sparse checkouts lack sources) and never `-DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR="$REPO"`:

```bash
. ./env.sh; T=$(cat "$SCRATCH/ROCK/T")
git clone --shared --no-checkout "$CLONE" "$SCRATCH/ROCK/src" && git -C "$SCRATCH/ROCK/src" checkout --detach "$HEAD" || exit 1
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
