# Lane: build and validate (`RUN`)

Build what the change touches at `$HEAD` and run the tests that exercise it, following the PR's own instructions where it gives
them. Provider builds and tests use [hipdnn-superbuild](../../hipdnn-superbuild/SKILL.md) and
[hipdnn-superbuild-test](../../hipdnn-superbuild-test/SKILL.md). Work in `$SCRATCH/RUN/`.

## Steps

1. **Checkout.** Create a full (non-sparse) checkout without writing to `$CLONE`. `fullclone` (from `env.sh`) shares `$CLONE`'s
   objects and fetches the blobs `$CLONE` lacks from the real origin; it reuses an existing directory and moves it to the SHA:

   ```bash
   . ./env.sh; mkdir -p "$SCRATCH/RUN"; fullclone "$SCRATCH/RUN/head" "$HEAD"
   ```

   Tests resolve paths against the repository root, so `$SCRATCH/RUN/head` is a full checkout. Create `$SCRATCH/RUN/base` the same
   way (`fullclone "$SCRATCH/RUN/base" "$BASE"`) only for a base reproduction (steps 6 and 7). A change that builds nothing (docs or tooling only) still needs `$SCRATCH/RUN/head` for
   `pre-commit` (a sparse clone is fine there); other read-only checks may run in `$REPO`, but `pre-commit` never does. Never
   build in `$REPO` or `$REPO_BASE`. Without a ROCm install, CMake, Ninja and a C++ toolchain, build nothing and write
   `Build: none (<missing>)` under **Coverage**; `defect` and `test-gap` candidates then stay `UNVERIFIED`.
2. **Select tests.** Only those that reach the changed files. Record each changed file's selected tests.
3. **Route by what changed.** Use the PR's stated steps if it gives them; otherwise:

   | Changed | Build | Tests | Select |
   |---|---|---|---|
   | Core C++ or CMake, `projects/hipdnn/**` | hipdnn-superbuild (the same path on Linux and Windows) | hipdnn-superbuild-test, component `hipdnn` | Its `--filter` input; `ctest -R <regex>` or `-L <label>` where the component's tree registers CTest |
   | Provider or integration tests, `dnn-providers/**` | hipdnn-superbuild | hipdnn-superbuild-test | Its component, scope and `--filter` inputs |
   | Python bindings or wheel, `projects/hipdnn/python/**` | Superbuild with the bindings, then the wheel (`projects/hipdnn/python/README.md` § "Testing the Wheel") | pytest on `projects/hipdnn/python/frontend_wheel_package/tests` | `-k <expr>` or node ids; `-m "not gpu"` on the host |
   | Other Python tooling with a `tests/` dir | None | That pytest suite (e.g. `projects/hipdnn/tools/DescriptorGenerator/tests`) | `-k`, node ids; markers from `pytest.ini`, `pyproject.toml` or `conftest.py` |
   | AI skills, `projects/hipdnn/tools/ai/**` | None | `python3 projects/hipdnn/tools/ai/validate-skills.py`; pytest on `projects/hipdnn/tools/ai/tests` | Whole suite |
   | Docs | Doxygen (`projects/hipdnn/Doxyfile`) for frontend API docs; Sphinx (`projects/hipdnn/docs/user-guides/conf.py`) for the user guide | Relative links and anchors in changed Markdown resolve | Changed files |
   | Core C++ or CMake with new options or install rules | Also configure only: tests OFF, in the superbuild and (Linux) the standalone `projects/hipdnn` build (`projects/hipdnn/docs/Building.md`, `projects/hipdnn/CMakePresets.json`) | The configure exits 0 | — |
   | ASAN | Host-only configure with `-DBUILD_ADDRESS_SANITIZER=ON`; device ASAN (Linux, `HSA_XNACK=1`) needs a GPU: an approval request (`projects/hipdnn/docs/testing/KNOWN_TESTING_GAPS.md`, § "Sanitizers and platforms") | Host-runnable tests that exercise the ownership paths in question | `ctest -R <regex>` |
   | `plugin_sdk`, `data_sdk`, `flatbuffers_sdk` or `test_sdk` | The consumer providers' host builds, hipdnn-superbuild | The consumer providers' host tests that **Consumers** (`CONTEXT.md`) lists, hipdnn-superbuild-test | Those tests |
   | Anything | — | `pre-commit run --files <changed files>`, only in `$SCRATCH/RUN/head` (hooks include `verify-support-claims`, `check-yaml`) | Changed files |

   A raw standalone build is used only when the PR's own steps or the change itself requires it (options the superbuild does not expose), and never on Windows: it skips the superbuild's DLL staging (`amd_comgr.dll`, `amdhip64_<N>.dll`), so load failures would also occur at `$BASE`, be labelled pre-existing and drop coverage.

   Run `pre-commit` as `( . ./env.sh; cd "$SCRATCH/RUN/head" && PRE_COMMIT_HOME="$SCRATCH/RUN/pre-commit" pre-commit run --files
   <changed files> )`. Its hooks include fixers (black, `clang-format -i`, end-of-file-fixer, trailing-whitespace), which is why it
   never runs in `$REPO`. Then record `git -C "$SCRATCH/RUN/head" status --porcelain`: a hook that modified a file is a finding
   (the change is not clean); discard the modification.

   `<regex>`, `<label>`, `<pattern>` and `<expr>` come from step 2; `<changed files>` is the path list in `numstat.txt`. Enable
   every option the change depends on and say which; **Options and guards** (`CONTEXT.md`) lists them with their defaults.
   Write one `Flag: <NAME>=<value> (<configure line or preset>)` line under **Coverage** for each option enabled, for `AUTH`. Every scratch configure passes `-DENABLE_CLANG_TIDY=OFF` (the
   option of core and each provider, default ON off Windows, which makes CMake run clang-tidy on every translation unit) and,
   where the environment or a preset sets it, `-DCMAKE_CXX_CLANG_TIDY=`. Set a unique `TMPDIR` under `$SCRATCH/RUN/` for each
   attempt and clear it when a run is killed. Take `-j` and test parallelism from `Cores:` in dispatch-context.md (or `Overrides:`). Keep full logs in
   `$SCRATCH/RUN/`; read only tails and error lines.
4. **Registration.** Each new CTest test appears under its expected label: `ctest -N -L <label>`, or `ctest --show-only=json-v1`
   for every test's labels. Core registers through `add_hipdnn_test` (`projects/hipdnn/cmake/Tests.cmake`); providers through
   `add_unit_test_target`, `add_integration_test_target` (`dnn-providers/cmake/Tests.cmake`) and
   `add_external_integration_test_target` (`dnn-providers/integration-tests/cmake/HipdnnIntegrationTestHelpers.cmake`). Install
   with `cmake --install <build> --prefix $SCRATCH/RUN/prefix`; list installed core tests with
   `ctest --test-dir $SCRATCH/RUN/prefix/bin/hipdnn -N`.
5. **Run.** Run CPU-only selections locally. Run a GPU selection only when it is listed under `Approved:` in dispatch-context.md;
   put every other GPU selection under **Approval requests**.
6. **Failures.** Rerun each failing selection once at `$HEAD` to separate flaky from broken, then at `$BASE` in
   `$SCRATCH/RUN/base`. A failure at both is pre-existing; attach both logs.
7. **Regression tests (M1).** For each regression test a defect fix adds, run it at `$BASE` (or at `$HEAD` with the fix reverted in
   a scratch copy of `$SCRATCH/RUN/head`) and record that it fails there and passes at `$HEAD`. Report the result under **Checked, no
   finding**, with no finding, so `TESTQ` and `AUTH` can settle M1.
8. **Account.** One row per suite under **Checked, no finding**: suite, arch, selected, passed, skipped, failed, engine, log; read
   counts and the serving engine from the logs. Record the build directory, install prefix, binaries, `compile_commands.json` path and the exact configure line so verifiers
   reuse them. A build tree is bound to its absolute path: do not relocate it, and mutate it only as [verify.md](verify.md) *Mutating a built tree* says.

## Findings

- PR steps that do not work as written (missing option, wrong target, missing dependency).
- Build warnings introduced by the change in files it touches.
- New tests not registered, not labeled, not installed, or skipped where they should run.
- Failures introduced at `$HEAD`, with the `$HEAD` and `$BASE` logs.
- Results that contradict the PR's testing claims.
