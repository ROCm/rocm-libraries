# Lane: build and validate (`RUN`)

Build what the change touches at `$HEAD` and run the tests that exercise it, following the PR's own instructions where it gives
them. MIOpen builds standalone from `projects/miopen` (`projects/miopen/docs/install/build-source.rst`). Work in `$SCRATCH/RUN/`.

## Steps

1. **Checkout.** Create a full (non-sparse) checkout without writing to `$CLONE`. `fullclone` (from `env.sh`) shares `$CLONE`'s
   objects and fetches the blobs `$CLONE` lacks from the real origin; it reuses an existing directory and moves it to the SHA:

   ```bash
   . ./env.sh; mkdir -p "$SCRATCH/RUN"; fullclone "$SCRATCH/RUN/head" "$HEAD"
   ```

   Then pull the DVC-tracked kernel databases the build installs, and record whether the pull ran
   (`( cd "$SCRATCH/RUN/head" && dvc pull -q projects/miopen/src/kernels )`; without access, record `dvc: not pulled` and treat
   tests that need a kernel database as unrun). Create `$SCRATCH/RUN/base` the same way (`fullclone "$SCRATCH/RUN/base" "$BASE"`)
   only for a base reproduction (steps 6 and 7). A change that builds nothing (docs or tooling only) still needs
   `$SCRATCH/RUN/head` for `pre-commit` (a sparse clone is fine there); other read-only checks may run in `$REPO`, but
   `pre-commit` never does. Never build in `$REPO` or `$REPO_BASE`. Without a ROCm install, the MIOpen dependencies, CMake and
   Ninja, build nothing and write `Build: none (<missing>)` under **Coverage**; `defect` and `test-gap` candidates then stay
   `UNVERIFIED`.
2. **Select tests.** Only those that reach the changed files. Record each changed file's selected tests as a `miopen_gtest`
   `--gtest_filter` or a `ctest -R` regex.
3. **Route by what changed.** Use the PR's stated steps if it gives them; otherwise:

   | Changed | Build | Tests | Select |
   |---|---|---|---|
   | Library C++, kernels or CMake, `projects/miopen/src/**`, `include/**` | The base configure below; targets `MIOpen`, `miopen_gtest`, `MIOpenDriver` | `miopen_gtest` from the build tree | `--gtest_filter=<pattern>`; `CPU_*` cases on the host, `GPU_*` cases only as step 5 allows |
   | Code that must also compile without a device (`src/nogpu/**`, anything under `MIOPEN_BACKEND` guards, tuning paths) | The base configure with `-DMIOPEN_BACKEND=HIPNOGPU` (build only, as `component-ci-miopen.yml` does) | None | — |
   | Composable Kernel paths (`src/ck_impl/**`, CK solvers) | The base configure with CK on, and once with `-DMIOPEN_USE_COMPOSABLEKERNEL=OFF` (build only) | The CK solvers' gtests | Their filter |
   | Public API or split (`include/miopen/miopen.h`, `src/private/**`, `include_wrapper/**`, `test/public_abi/**`) | Source check first: `python3 projects/miopen/script/check_public_abi.py check-headers --require-provider`; after the base build, `check <build>/lib*/libMIOpen.so.1.0 --baseline projects/miopen/test/public_abi/public_symbols.baseline`; with flag-on code, also a configure with `-DMIOPEN_ENABLE_HIPDNN_WRAPPER=ON` and the script's `check-wrapper` | `pytest projects/miopen/script` | Whole suite |
   | Test registration or names (`test/gtest/**`, `test_categories.yaml`) | The base configure | The `miopen_gtest_check` target (`check_names.py`); `ctest -N` to list what registered | — |
   | Find-db or perf-db files (`databases` area) | The base configure with `-DMIOPEN_TEST_DBSYNC=ON` | `miopen_gtest --gtest_filter='*DBSync*'` on a GPU of that architecture (step 5), or GPU-free through `projects/miopen/test/gtest/dbsync/run_dbsync_rocjitsu.py` as its header describes (it fetches and builds the pinned rocjitsu; host-only, no approval needed) | The updated architectures |
   | MIOpenDriver (`driver/**`) | Target `MIOpenDriver` | `*GPU_MIOpenDriver*` cases; a `bin/MIOpenDriver <op> ... -V 1` run of each changed path (GPU, step 5) | — |
   | Fin (`fin/**`, `src/fin/**`) | Install the base build with `-DMIOPEN_ENABLE_FIN=ON -DMIOPEN_INSTALL_CXX_HEADERS=ON`, then configure `projects/miopen/fin` against that prefix (as `component-ci-miopen.yml` does) | None | — |
   | Python tooling (`projects/miopen/script/**`, `test/gtest/dbsync/**`, `utils/**`) | None | The pytest files beside it (`test_*.py`) | `-k`, node ids |
   | AI skills, `projects/miopen/skills/**` | None | `pytest projects/miopen/skills/<skill>/scripts/tests` | Whole suite |
   | Docs | Sphinx (`projects/miopen/docs/conf.py`, requirements in `projects/miopen/docs/sphinx/requirements.txt`) | Relative links and anchors in changed reStructuredText and Markdown resolve | Changed files |
   | `dnn-providers/miopen-provider/**` | Not built here: the provider is a hipDNN subproject. Record `UNVERIFIED (provider not built)` and name the `miopenprovider` TheRock job, or build it with the hipDNN superbuild skill when the host has it | — | — |
   | Anything | — | `pre-commit run --files <changed files>`, only in `$SCRATCH/RUN/head` (hooks include the MIOpen gtest-naming and public/private header checks) | Changed files |

   **Base configure.** TheRock's settings (`MIOPEN_TEST_DISCRETE=OFF` gives the single `miopen_gtest`, `MIOPEN_USE_MLIR=OFF`)
   plus a release build for one architecture, so results compare with CI:

   ```bash
   . ./env.sh; R="$SCRATCH/RUN"; A=<GPU: from dispatch-context.md, else gfx942>; export TMPDIR="$R/tmp-$(date +%s)"; mkdir -p "$TMPDIR"
   export CXX="${ROCM_PATH:-/opt/rocm}/llvm/bin/clang++"
   cmake -S "$R/head/projects/miopen" -B "$R/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DMIOPEN_BACKEND=HIP \
     -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm};<deps prefix>" -DGPU_TARGETS="$A" -DMIOPEN_INSTALL_GPU_DATABASES="$A" \
     -DMIOPEN_TEST_DISCRETE=OFF -DMIOPEN_USE_MLIR=OFF -DBUILD_DEV=OFF -DMIOPEN_TEST_FLAGS=' --disable-verification-cache ' \
     -DCMAKE_EXPORT_COMPILE_COMMANDS=ON > "$R/configure.log" 2>&1 || { tail -n 40 "$R/configure.log"; exit 1; }
   ninja -C "$R/build" -j<Cores> MIOpen miopen_gtest MIOpenDriver > "$R/build.log" 2>&1 || { "$GREP" -n -m 40 'error' "$R/build.log"; exit 1; }
   ```

   `<deps prefix>` is where `cmake -P projects/miopen/install_deps.cmake --prefix <dir>` (or `rbuild prepare`) installed the
   dependencies; omit it when they are under the ROCm install. The `rocm/miopen-dev` image `component-ci-miopen.yml` pins has
   them all; use it when the host offers containers. Add every option the change depends on and say which; **Options and
   guards** (`CONTEXT.md`) lists them with their defaults. Write one `Flag: <NAME>=<value> (<configure line>)` line under
   **Coverage** for each option set away from its default, for `AUTH`. Never build the clang-tidy `analyze` target: it is a
   separate CI job, not this lane's.

   **Test runs.** Run each selection with its own user database and kernel cache, so a warm cache from an earlier run cannot
   hide a compile or find failure:

   ```bash
   . ./env.sh; R="$SCRATCH/RUN"; T="$R/run-<name>"; mkdir -p "$T/udb" "$T/cache"
   MIOPEN_USER_DB_PATH="$T/udb" MIOPEN_CUSTOM_CACHE_DIR="$T/cache" MIOPEN_LOG_LEVEL=4 \
     "$R/build/bin/miopen_gtest" --gtest_filter='<pattern>' > "$T/gtest.log" 2>&1; echo "exit $?"
   ```

   To show which solver served a convolution case, add `MIOPEN_ENABLE_LOGGING=1` or pin one with `MIOPEN_DEBUG_FIND_ONLY_SOLVER`
   (`projects/miopen/docs/how-to/debug-log.rst`) and read the log.

   Run `pre-commit` as `( . ./env.sh; cd "$SCRATCH/RUN/head" && PRE_COMMIT_HOME="$SCRATCH/RUN/pre-commit" pre-commit run --files
   <changed files> )`. Its hooks include fixers (`clang-format -i` and others), which is why it never runs in `$REPO`. Then record
   `git -C "$SCRATCH/RUN/head" status --porcelain`: a hook that modified a file is a finding (the change is not clean); discard
   the modification.

   `<pattern>` comes from step 2; `<changed files>` is the path list in `numstat.txt`. Set a unique `TMPDIR` under
   `$SCRATCH/RUN/` for each attempt and clear it when a run is killed. Take `-j` and test parallelism from `Cores:` in
   dispatch-context.md (or `Overrides:`); MIOpen's kernel and CK compiles are memory-heavy, so lower `-j` before blaming the
   change for an out-of-memory kill. Keep full logs in `$SCRATCH/RUN/`; read only tails and error lines.
4. **Registration.** Each new gtest case appears in `"$R/build/bin/miopen_gtest" --gtest_list_tests` under the expected
   instantiation (`Smoke`, `Full`) and passes `check_names.py`; on a host where listing needs a device, run it under the rocjitsu
   interposer as `component-ci-miopen.yml` does or record `UNVERIFIED (no device for listing)`. Match each new case against
   `projects/miopen/test/gtest/test_categories.yaml` and say which tier selects it. Install with
   `cmake --install "$R/build" --prefix "$R/prefix"` and confirm the installed `miopen_gtest`, databases and
   `install_CTestTestfile.cmake` contain what the change adds.
5. **Run.** Run `CPU_*` and other device-free selections locally. Run a GPU selection only when `GPU:` names a local device and
   the selection is listed under `Approved:` in dispatch-context.md; put every other GPU selection under **Approval requests**
   with the filter, architecture and expected duration.
6. **Failures.** Rerun each failing selection once at `$HEAD` to separate flaky from broken, with a fresh user database and
   cache, then at `$BASE` in `$SCRATCH/RUN/base`. A failure at both is pre-existing; attach both logs.
7. **Regression tests (M1).** For each regression test a defect fix adds, run it at `$BASE` (or at `$HEAD` with the fix reverted in
   a scratch copy of `$SCRATCH/RUN/head`) and record that it fails there and passes at `$HEAD`. Report the result under **Checked, no
   finding**, with no finding, so `TESTQ` and `AUTH` can settle M1.
8. **Account.** One row per selection under **Checked, no finding**: filter, arch, selected, passed, skipped, failed, solver(s)
   seen, log; read counts and solvers from the logs. Record the build directory, install prefix, binaries,
   `compile_commands.json` path and the exact configure line so verifiers reuse them. A build tree is bound to its absolute path:
   do not relocate it, and mutate it only as [verify.md](verify.md) *Mutating a built tree* says.

## Findings

- PR steps that do not work as written (missing option, wrong target, missing dependency).
- Build warnings introduced by the change in files it touches (CI builds with `-Werror`).
- New tests not registered, misnamed, not selected by any tier, not installed, or skipped where they should run.
- Failures introduced at `$HEAD`, with the `$HEAD` and `$BASE` logs.
- Results that contradict the PR's testing claims.
