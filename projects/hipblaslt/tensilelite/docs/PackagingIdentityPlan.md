# AIHPBLAS-3989: TensileLite packaging identity — plan and decisions

Scope: the rocm-libraries TensileLite packaging stack (#11016–#11032, decision records
in #11254) and the TheRock branch `users/alvasile/tensilelite_packaging`.

Origin: archana-ramalingam's comment on #11016
(<https://github.com/ROCm/rocm-libraries/pull/11016#issuecomment-6757079254>). The
gfx1250-strict job installs ROCm from pip wheels, and tox's packaging env never sees
`ROCM_PATH`, so `setup.py` falls back to `/opt/rocm` and fails.

Part 1 says what changes in which branch and repository. Part 2 records the
decisions behind those changes. The decision log
([PythonBuildGrillingDecisions.md](../PythonBuildGrillingDecisions.md)) records
decisions 1–14 as Q121–Q134 and client-path ownership as Q135. The TheRock
packaging requirements this effort answers are written out below, with their status.

**Status (2026-11-12):** rocm-libraries stack #15582 and TheRock T1–T14 branches
are published. The #11022 generator-environment fix, #11025 tox fix, and #11019/#11025
GEKO correction are committed locally and restacked; they have not been pushed.
Section 1.5 records their scope and validation. TheRock PRs and the reply to
archana remain pending. TheRock branches merged cleanly onto `main` as checked
on 2026-11-11; a full TheRock build has not run.

The table below is the state at the start of the review.

| Ref | SHA (2026-11-10) |
| --- | --- |
| rocm-libraries `develop` | `d12be994772e` |
| #11016 `users/alvasile/AIHPBLAS-3989-wheel-handler-rocisa-foundation` | `7023260214a9` |
| #11032 `users/alvasile/AIHPBLAS-3989-python-sdk-runtime` | `f7a32c6e7b8a` |
| #11254 `users/alvasile/tensilelite_packaging-stack` | `9afc018fe71f` |
| rocm-libraries `therock-ref` | `647a8c52a206` |
| TheRock `main` | `f92083089` (pins rocm-libraries `2d3b6e98e4ef`) |
| TheRock `users/alvasile/tensilelite_packaging` (local, rebased on `main`, unpushed) | `0abec119b` |

## Packaging requirements this effort answers

TheRock's design for Python packages that sit next to native ROCm components
(Origami is the first implementation, TensileLite follows) asks for the
requirements below. They are numbered P1–P8 so the rest of this document can refer
to them.

The motivating failure is Origami (ROCM-29472, ROCM-32106). ROCm shipped one native
Origami library, and a separately installed `rocm-origami` Python package carried
another version. Both exposed overlapping symbols with incompatible memory layouts,
which corrupted the heap. TheRock's validation did not install the Python package
that PyTorch CI used, so it did not reproduce the crash, and it was found during
release-candidate testing.

| # | Requirement | Status |
| --- | --- | --- |
| P1 | A Python package is a standalone, canonical distribution inside its super-repository. | Done for the `tensilelite` wheel (#11016). rocisa, its adaptor and the client are not Python distributions yet (F3, F4). |
| P2 | `pip install path/to/project` works against the installed ROCm, and does not rebuild or replace the native libraries under test. | Done for `tensilelite`: plain pip, tox and sdist builds agree (decision 1). Kernel generation installs nothing into the build Python (#11022, decision 9). The client records the ROCm it was built against and is checked at run time (decision 14). rocisa and `stinkytofu` are still reached by path (F3, F4). |
| P3 | A dedicated TheRock stage builds the wheel once and stages it with the SDK. | Not done. The wheel is built in hipBLASLt's CMake and shipped in `blas_test` (F1). |
| P4 | Validation runs against the installed, shipped ROCm binaries, without rebuilding or substituting them. | Partly. T8–T10 and R1 test the installed wheel against the artifact's own components (decision 9). Pip-SDK and native-package validation is F2. |
| P5 | Results carry provenance: SDK build, package version, native component versions, source revisions, Python version, test inputs and generated artifacts. | Partly. T11 records the wheel, the client's version and commit, the TheRock manifest's package version and commit, and the Python version. Native component versions beyond the manifest, test inputs and generated artifacts are not recorded (F7). |
| P6 | For TensileLite, validation distinguishes tests of newly generated kernels from tests of the hipBLASLt binaries shipped in the SDK, so a pass on one is not read as a pass on the other. | Not done (F7). |
| P7 | A native Python package evaluates the Python Stable ABI before adding per-Python-version builds. | Deferred with rocisa. F4 starts from the Stable ABI; if rocisa cannot use it, that is reviewed before any per-version build is added. |
| P8 | The first implementation, Origami (rocm-libraries#10975, open), is the reference pattern that TensileLite reuses. | Not done. F1 must follow it (F8). |

---

# Part 1: Implementation plan

## 1.1 Landing order

The rocm-libraries stack, bottom to top:

```text
#11016 → … → #11019 → R0 (new) → #11020 → … → #11028 | #11030 → #11031 → #11032 → R1 (new) | #11254 (docs, stays on top)
  lands without TheRock   gated on TheRock (conditional enablement slices)
```

This keeps the stack's original design.

1. **#11016 → #11028 land without any TheRock change.** Every TheRock input is
   optional there (decisions 6–8), and TheRock `main` keeps working with the stack's
   artifacts (#11020 keeps the layout `main` expects).
2. **TheRock PRs T1–T13 (section 1.3) land independently of rocm-libraries.** Each
   works with today's rocm-libraries pin. Some depend on an earlier TheRock PR, as
   listed in section 1.3.
3. **#11030 → R1 are conditional enablement slices.** Their PR bodies already say
   each merges only after a TheRock prerequisite is available. The gates, mapped to
   section 1.3:

   | Branch | Gate |
   | --- | --- |
   | #11030 `therock-rocm-identity` | T1 in rocm-libraries' `therock-ref` (TheRock passes `THEROCK_PACKAGE_VERSION`) |
   | #11031 `rocm-version-validation` | T1, through #11030 |
   | #11032 `python-sdk-runtime` | T2 merged **and in a published nightly `rocm-sdk-core`** (not `therock-ref`; see its section) |
   | R1 (new) | T8–T10 in `therock-ref` (installed-wheel runner on both jobs) |

   They are reviewed and rebased with the stack while they wait.
4. **After TheRock's rocm-libraries submodule includes R1:** TheRock cleanup T14
   removes the runner's existing-path branch (from T8) and makes T13's artifact
   checks unconditional. Before R1 the artifact still holds the unpacked copy, so
   the existing path is still the one in use.

No step pushes or posts anything without explicit approval of the exact change or
text.

## 1.2 rocm-libraries stack

Each branch keeps its PR. Work is done in owner-branch order: amend the branch that
owns the change, then restack its descendants with `rerere` disabled. Back up refs
before every rewrite.

### #11016 `wheel-handler-rocisa-foundation` (introduces `setup.py` / `release_metadata.py`)

- **SHA resolver** (decisions 2 and 3): a new `tensilelite/_source_revision.py`.
  `setup.py` loads it with `runpy`; the runtime imports it.
  - Lookup order, exactly as in decision 3: saved value in an unpacked sdist → git
    (with the "tree belongs to this repo" guard) → `TENSILELITE_SOURCE_REVISION`
    (fails if it disagrees with git) → none.
  - The dirty value is `<sha>.dirty.<12-hex digest>` over the whole repository.
    Use sorted raw Git change records and streamed tracked contents, not patch
    text. Disable external diff/textconv; disable fsmonitor centrally on every
    resolver query, including the ownership probe, with `-c core.fsmonitor=`.
  - For nonignored untracked entries, hash sorted Git-relative path, file type,
    size and integer `lstat().st_mtime_ns` records without reading contents or
    following links. Accept same-metadata collisions (decision 3).
  - Keep sdists link-free: the real backend test must verify source symlinks and
    hardlinks become regular archive members with correct contents, then extract
    and build a stamped wheel through native and emulated Python 3.10.8 APIs.
    Do not weaken `_safe_extractall`'s link rejection on older Python.
- **Stamping**, never into the source tree:
  - a `build_py` override writes `tensilelite/_build_info.py` into `build_lib`
    (skipped for editable builds) and lists it in `get_outputs()`;
  - an `sdist` override writes it into the release tree in `make_release_tree()`,
    removing first any hard link to the source file;
  - add `tensilelite/_build_info.py` to `.gitignore` as a guard.
- **Version** (decision 1): `5.0.0+g<40-hex>[.dirty]` unless an explicit ROCm identity
  is supplied. The version never contains the diff digest or any other digits-only
  ID.
  - Remove the `ROCM_VERSION` and `ROCM_PATH`/`/opt/rocm` reads.
  - No `[testenv:.pkg]` section is added; this resolves archana's comment.
- **Logic-corpus fix:** `ModuleNotFoundError: tensilelite.Tests`. Either restore
  develop's exclusion (`tensilelite.Tests.unit.source_only*` instead of
  `tensilelite.Tests*`), or have the tests load `rocisa_test_state` from the test
  tree.
- **Tests:** replace the packaging tests' `ROCM_VERSION`/`TENSILELITE_ROCM_VERSION`
  setup with cases for:
  - a clean checkout;
  - a wheel built from the sdist;
  - a new commit after an sdist build (the new SHA is used, not the saved one);
  - a dirty tree (the three pip `setup.py` runs agree);
  - an untracked file under `tensilelite/`, and one elsewhere in the repository
    (both dirty);
  - a tree that doesn't belong to the containing repo (no SHA);
  - an explicit override that agrees, and one that disagrees (fails);
  - an editable install (no stamp);
  - `git status` clean after builds;
  - byte-identical wheels under `SOURCE_DATE_EPOCH`.
- **Validation:**
  - the tox packaging reproduction with `ROCM_PATH` unset: the sdist and wheel
    versions must match, and the source tree must be unchanged;
  - a plain `pip install path` with no ROCm environment;
  - the focused packaging tests;
  - the logic-corpus test collection;
  - after an approved push, a rerun of the gfx1250-strict job.
- **Reply to archana:** after this lands on the branch, draft a reply stating what
  changed, and post it only once you approve the text.

### #11017 `canonical-package-command`

Restack only.

### #11018 `package-client-version-identity` (introduces `TensileLiteClientVersion.hpp`)

- CMake resolves the SHA once per configure by calling the same
  `_source_revision.py` (decision 3).
  - A `TENSILELITE_SOURCE_REVISION` cache variable feeds the resolver's step 3, for
    builds of a tree without git (release tarballs).
  - If no SHA is found, CMake records none and warns at configure time: "no source
    revision found; client pairing disabled. Build from a git checkout or pass
    `-DTENSILELITE_SOURCE_REVISION=<sha>`." The client prints `none` for
    `--source-revision` and never pairs (decision 3).
  - Add `.git/HEAD`, the current ref file and `packed-refs` to
    `CMAKE_CONFIGURE_DEPENDS`, located via `git rev-parse --git-path`, so a new
    commit triggers a re-configure.
- The CMake build passes the same value into the client.
- `tensilelite-client --version` keeps its one-line contract (Q065). A new
  `--source-revision` command prints the full SHA under the same no-GPU, one-line,
  five-second rules.
- `release_metadata.py` takes the identity as an argument; it reads no ROCm root.
- **Client build ROCm** (decision 14): CMake records the ROCm the client is compiled
  against and bakes it into `TensileLiteClientVersion.hpp`.
  - Source, from the build root `hipblaslt_resolve_build_rocm_root()` selects, using
    decision 4's rules: `rocm_sdk_core.__version__` in the build Python (full), the
    TheRock manifest (full), `.info/version` (base), or `unknown`.
  - A new `tensilelite-client --build-rocm` prints `<identity> <full|base|unknown>`
    on one line, under the same no-GPU, five-second rules.
  - `hipblaslt_resolve_build_rocm_root()` and the `.info/version`/manifest
    configure dependencies therefore stay on this branch, but feed only the client.
    They never feed the wheel's version (decision 1).
  - A ctest checks `--build-rocm` against the value CMake computed.

### #11019 `runtime-client-binding-foundation` (introduces `validate_client`, `_rocm.py`)

- Own `client_executable_name()` and `built_client_path()` in the top-level binding
  module. Invoke's build/binding tasks use the same filename and CMake layout.
- Expose `invoke client-path --build-dir ...` for source-build consumers such as
  GEKO. Task discovery and this query do not import the generator package (Q135).
- `validate_client()` compares `--source-revision` with the package's own SHA, not
  the distribution version (decision 3):
  - **installed package:** read the SHA from `_build_info.py`, then require an exact
    match, including any dirty suffix;
  - **editable install:** there is no `_build_info.py`. Compute the SHA from git at
    the first client request, from the package's own location, and cache it. Compare
    commits only: a commit mismatch fails, uncommitted changes on either side warn;
  - **no SHA available:** never pairs; the error says why.
- Never use `importlib.metadata.version()` for pairing; it is fixed at install time.
- The diagnostics name both values and how to rebuild and bind a client from this
  checkout.
- `tensilelite-configure-client` applies the same check.
- Read the client's `--build-rocm` (decision 14) and include it in the diagnostics.
  No comparison yet: the runtime adapters arrive in #11023.
- **API** (audit, 2026-11-11): `Installation` gains `source_revision: str | None`
  and `editable: bool`; `validate_client(client, installation)` replaces the
  version-string signature, and `default_client_candidate`/`selected_client` take
  the `Installation`.
  - SHA lookup: installed or unpacked on `PYTHONPATH` (the #11022 build tree,
    TheRock's existing artifact path) → `read_build_info(<package_dir>/_build_info.py)`;
    editable (`direct_url.json` `dir_info.editable`) →
    `_source_revision.resolve(package_dir.parent)`.
  - Load `_source_revision.py` by file path: importing
    `tensilelite._source_revision` runs `tensilelite/__init__`, which calls
    `_runtime.initialize()` from #11025 on.
  - `invoke install` drops `ROCM_PATH=` from the editable-install environment.
  - Rewrite `test_client_binding.py` (about 25 version literals) and
    `test_install_task.py`.

### R0 (new branch) `users/alvasile/AIHPBLAS-3989-invoke-selected-rocm-root`

A new PR stacked directly on #11019, built from commit `9afc018` ("build(tensilelite):
use one selected ROCm root for every native configure"), which currently sits on
the docs branch #11254. Not gated; it lands with the lower stack.

- **Keep:** `_resolve_rocm_build_environment()` and `RocmBuildEnvironment` in
  `tasks.py`. One root (`--rocm-path` → non-empty `ROCM_PATH`, empty is an error →
  `rocm-sdk path --root` → `/opt/rocm`) supplies the compilers, `-DROCM_PATH` and
  `-DCMAKE_PREFIX_PATH` for rocisa, the StinkyTofu binding, `build-client`,
  `build-coverage` and `install`; `install` resolves it once for all its steps. Keep
  `test_install_task.py`.
- **Drop:** `env["TENSILELITE_ROCM_VERSION"] = _rocm_base_version(rocm)` in
  `install` (decisions 1 and 5: invoke passes no identity, so the editable install
  is `+g<sha>`). Add a test that `install` passes no `TENSILELITE_ROCM_VERSION`.
- **Stays on #11254:** the commit's 20 lines in `InvokeInstallPythonRocmSupport.md`.
- **Why here:** `invoke install` first exists in #11019, so this is the lowest
  branch the change applies to. Before it, tasks picked ROCm independently (the
  preset's `/opt/rocm` for `build-client`, `_detect_rocm()` for `build-coverage`,
  `ROCM_PATH`/`/opt/rocm` for `install`), so one `invoke install` could compile
  rocisa and the client against different installs. Decision 14's client build ROCm
  is only true if every native build uses the same root.
- **Expected conflict:** #11021's `tasks.install` identity change, which decision 1
  removes anyway.

### #11020 `release-wheel-installed-artifacts` (introduces CMake wheel builds, artifact unpacking, and the compatibility package)

- **Remove the compatibility package `tensilelite_tensile_compat` entirely**
  (decision 13). This branch introduces it, so the removal is a change to this
  branch rather than a later deletion:
  - drop the `compat/` directory (package, `setup.py`, `pyproject.toml`,
    `MANIFEST.in`, `README.md`, `tests/`);
  - CMake: remove `_tensilelite_needs_compatibility_wheel`, the compatibility wheel
    target, its install rule, and the `compat/tests` install;
  - `scripts/check_release_wheel_contents.py`: remove compatibility mode and the
    legacy script table;
  - `scripts/install_release_wheels.py`: remove `_COMPATIBILITY_PATHS`;
  - remove `compat/tests` from `pytest.ini` `testpaths` and from tox commands;
  - update `test_release_wheel_contents.py` and `test_packaging_metadata.py`.
  - This branch also removes the legacy `Tensile*` console scripts that develop's
    canonical `pyproject.toml` still installs (`Tensile`, `TensileCreateLibrary`,
    `TensileGetPath`, …). That stays, now with no replacement; the PR body says to
    use the `tensilelite` command family.
- Pass the configure's SHA to the wheel build (`TENSILELITE_SOURCE_REVISION`), so the
  wheel and the client always share it. Set `SOURCE_DATE_EPOCH` from
  `git log -1 --format=%ct` for reproducible wheels.
  - **Required, not optional** (audit): the staged copy under the build directory
    is not owned by git (`build*/` is ignored), so without it the wheel is a bare
    `5.0.0` with `SOURCE_REVISION = None`.
  - Remove `ROCM_PATH=${HIPBLASLT_BUILD_ROCM_ROOT}` and the `.info/version`
    dependency from the wheel build (decision 1).
  - Add `_build_info.py` to `stage_release_source.py`'s ignore list.
  - `check_release_wheel_contents.py` gains `--expected-source-revision`, compared
    with the `_build_info.py` inside the wheel.
- Keep unpacking the wheel into `share/hipblaslt/tensilelite`, because TheRock
  `main`'s runner still imports from there. Removal is the new R1 branch.
- **The wheel recipe stays inline** in `projects/hipblaslt/CMakeLists.txt` (lines
  891–1013 on this branch), reduced to one wheel (decision 9). No reusable module
  and no `TENSILELITE_WHEEL_DIR` input; both are F1 work.

### #11021 `rocm-wheel-build-identity` (introduces `TENSILELITE_ROCM_VERSION`, `.pkg`, the `0.0.0` fallback)

- `TENSILELITE_ROCM_VERSION` becomes an optional explicit input, passed only by
  CMake in TheRock builds. When it is absent, the result is the dev identity
  (decision 1).
- Remove `[testenv:.pkg]` and the `0.0.0` placeholder.
- `hipblaslt_resolve_build_rocm_version()`:
  - TheRock builds use `THEROCK_PACKAGE_VERSION` only when it is set and not `"git"`,
    otherwise no identity (a dev build);
  - standalone CMake builds and `invoke install` pass no identity, so they produce
    `+g<sha>` (decisions 1 and 5). Remove the standalone `.info/version` read.
- Drop this branch's change to `compat/setup.py` (decision 13).
- **Never introduce** `THEROCK_ROCM_VERSION`, the `0.0.0` fallback or the
  `version.json` configure dependency (audit), so #11030 has nothing to remove.
- **The `+devrocm` composition moves here from #11026** (audit): T1 can reach
  `therock-ref` before #11030, and until #11026 the label would otherwise be
  `+rocm7.12.0.dev0.<sha>`. API: `distribution_version(component, revision,
  rocm_identity=None)`.
- Revert the pre-commit hint to plain `uv sync` (no `TENSILELITE_ROCM_VERSION=0.0.0`).
- Strip `TENSILELITE_ROCM_VERSION` from `test_canonical_imports`' environment and
  rewrite `test_therock_rocm_identity.py`.
- `skip_install` first appears on this branch (not #11025).

### #11022 `client-free-device-generation` (introduces build-Python wheel install and codegen environment)

- Stop running `pip install --force-reinstall --no-deps` into `Python3_EXECUTABLE`.
  Keep this branch's `hipblaslt_configure_tensilelite_python()` /
  `hipblaslt_tensilelite_python_environment()` (they replace develop's
  `hipblaslt_configure_bundled_python_command()`, which this branch deletes), and
  replace the pip-install command with
  `install_release_wheels.py --destination <build>/tensilelite-wheel` plus a stamp.
  Put that directory on the code-generation `PYTHONPATH` in place of develop's
  `<src>/tensilelite`, still as `cmake -E env PYTHONPATH=… PATH=… --
  Python3_EXECUTABLE` (decision 9).
- Code generation depends on the wheel file, not on source globs: in
  `create_device_library`, drop the `_codegen_sources` glob and its manifest
  entries and keep `_logic_files`. The wheel recipe keeps its own source globs;
  extops already depend on `HIPBLASLT_PYTHON_DEPS`. Drop the source-tree entries
  from the code-generation `PYTHONPATH`, except `rocisa_stinkytofu_adaptor`, which
  still comes from the source tree until it has a package (F3).
- Remove `TENSILELITE_ROCM_VERSION=` from the code-generation environment. Keep
  `ROCM_PATH=<selected root>` for standalone builds (decision 14's
  `hipblaslt_resolve_build_rocm_root()`).
- No provenance sidecar next to the kernel library (dropped 2026-11-11; see F6). In
  this stack the library, wheel and client come from one configure of one checkout,
  whose SHA TheRock's manifest already records, and T11 records the wheel in the
  test job. A loose file in the installed library directory would be read by
  nothing and could trip `HIPBLASLT_VALIDATE_INSTALL` or TheRock's artifact
  descriptors.
- When `HIPBLASLT_ENABLE_THEROCK=ON`, set `TENSILELITE_ROCM_ADAPTER=therock-ci` only
  in the code-generation commands' environment (`cmake -E env …`, alongside the
  existing `PYTHONPATH` and `PATH`). Pass `THEROCK_PACKAGE_VERSION` there only when
  a real identity is known. Keep the explicit `--cxx-compiler` and
  `--offload-bundler` arguments (decision 7).
- The wheel's label is already checked since #11020 (`--expected-version`); in
  TheRock builds that expected value is the passed identity (decision 7).
- Drop this branch's CMake changes to the compatibility wheel (decision 13).

### #11023 `lazy-rocm-installation-model` (introduces the runtime adapters and `devrocm`)

- `TheRockCIRocm` is selected only by `TENSILELITE_ROCM_ADAPTER=therock-ci`. Any
  other value is an error. `THEROCK_PACKAGE_VERSION` being set no longer selects
  anything; it supplies the identity when present.
- `TheRockCIRocm` accepts a missing or `"git"` identity as a dev build. When the
  wheel carries a `+rocm…`/`+devrocm…` label and a version was passed, it compares
  the two.
- Tests: the marker selects the adapter with and without a version; without the
  marker, `THEROCK_PACKAGE_VERSION` in the environment (empty, `"git"` or real) does
  not select it, and an installed wheel still uses `PythonRocm`/`SystemRocm`; an
  unknown marker value fails.
- `SystemRocm` uses the full identity on TheRock-built roots (decision 4):
  - read `share/therock/therock_manifest.json` `rocm_package_version`, and accept
    it only if it parses as a release identity (so not `"git"`);
  - fall back to `.info/version` when the manifest is missing, unreadable or
    malformed;
  - if both exist and their base versions differ, fail with an "inconsistent ROCm
    installation" error naming both files;
  - compare the full identity against the wheel label instead of only the base;
  - **dev identities** (`X.Y.Z.devN+<commit>[.<variant>…]`, TheRock release types
    `ci`/`dev`/`dev-bkc`): drop the leading commit token of the local segment (the
    40-hex `GITHUB_SHA` or override) before comparing; keep every variant token
    (`asan`, `hostasan`, …). Release, `rc` and nightly identities, including
    `+bkc.<date>` and variant suffixes, are compared in full (decision 5);
  - tests: a valid manifest, `"git"`, missing, malformed, disagreement with
    `.info/version`, and nightly/dev/release values.
- `PythonRocm` is unchanged (`rocm_sdk_core.__version__`).
- **Base-only identities** (audit): when the identity comes from `.info/version`
  (a pre-TheRock prefix, or the manifest fallback), compare base to base, so release
  wheels work in scenario 3.
- `+g<sha>` labels parse as "no ROCm claim" here; the warn-once behaviour is
  #11031's. `ValidatedRocm.version` becomes `str | None` (only diagnostics read it).
- **Client build ROCm check** (decision 14): when a client is bound, compare its
  `--build-rocm` with the selected adapter's identity. Compare full to full, or base
  to base when either side is base-only.
  - different major version: error;
  - any other mismatch, or `unknown`: warn once per process. #11031 makes it strict
    for release-labelled packages.
  - tests: same identity, same base, different minor, different major, `unknown`,
    and full-versus-base.

### #11024 `selected-rocm-toolchain`

Restack only. (This branch never touches `tox.ini`; #11021's `.pkg` section is never
introduced, decision 1.)

### #11025 `geko-installed-package-cutover` (introduces `skip_install` and the `TOX_ENV_NAME` branch)

- Remove the `TOX_ENV_NAME` branch from `setup.py`.
- The editable installs use the #11016 resolver and read no ROCm environment.
- Keep `skip_install = true`; it is now a choice, not a workaround.
- Drop the `compat/tests` tox commands this branch adds (decision 13).
- Drop this branch's `setup.py` hunk and its `.pkg`/`0.0.0` removal hunks entirely;
  they remove things that are never introduced (audit).
- Delete geko's `utils._tensilelite_build_environment()` and its tests.
- Query #11019's `invoke client-path` before building or installing. GEKO does not
  duplicate the executable suffix or build layout. Generated bash scripts quote
  interpreter/client paths through `shell_word`; tests cover both executable
  names, Windows backslashes, spaces and apostrophes without an installed-package skip.
- Drop the `TENSILELITE_ROCM_VERSION="$(<…/.info/version)"` prefix in
  `.github/scripts/run_rocjitsu_hipblaslt_race_check.sh` (#11026 and #11027 touch it
  again).
- Fix `test_package_init_char.py:24`, which asserts
  `startswith(GENERATOR_VERSION + "+rocm")` and fails with `+g` wheels.
- Rerun gfx1250-strict here once pushed: runtime initialization at import starts on
  this branch.

### #11026 `generator-metadata-cleanup`

- Drop this branch's change to `compat/` (decision 13).
- Drop its `+devrocm` composition in `release_metadata.py` (moved to #11021); expect
  a conflict there.
- The pre-commit hint `TENSILELITE_ROCM_VERSION="$(<…)" uv sync` becomes `uv sync`.

### #11028 `strict-runtime-initialization`

Restack only.

### #11027 `installed-workflow-guidance`

Update the user guidance to describe:

- SHA pairing and how to bind a self-built client;
- the dev and release version formats;
- which ROCm installation forms are supported (decision 10);
- that the legacy `Tensile*` command names are gone, with the `tensilelite`
  subcommand to use for each (decision 13).

Also drop this branch's change to the race-check script's identity prefix, and grep
its docs for `TENSILELITE_ROCM_VERSION` and `+rocm`.

### #11030 `therock-rocm-identity` (gated on T1; requires the TheRock identity)

This branch already exists to turn the temporary TheRock fallback into a hard
requirement once TheRock passes the version. Keep that purpose, and adjust it to
the revised decisions:

- In `hipblaslt_resolve_build_rocm_version()`, when `HIPBLASLT_ENABLE_THEROCK=ON`:

  | `THEROCK_PACKAGE_VERSION` | Behaviour |
  | --- | --- |
  | Missing | `FATAL_ERROR`: TheRock did not pass a package version |
  | `"git"` | Dev identity, the explicit local TheRock build case (today this branch rejects it) |
  | Real version | Stamped, strict checks |

  Standalone (non-TheRock) builds are unchanged.
- `THEROCK_ROCM_VERSION` is never introduced (#11021), so there is nothing to stop
  accepting; after #11021's redesign this branch shrinks to the missing → error and
  `"git"` → dev rows. Flip the `"git"` test case and update the PR body, which still
  mentions `THEROCK_ROCM_VERSION`.
- It replaces the leniency #11021–#11028 provide: below this branch, a missing
  version gives a dev identity (decision 6).
- **Tests:** CMake source-only tests for all three cases, and for a standalone build.
- **Justification:** once TheRock always passes a value, a missing one means
  something broke on the way. Without this error, a nightly, CI or release build
  could silently produce an unchecked `+g<sha>` dev wheel. Together with the
  expected-identity check in `check_release_wheel_contents.py`, this guarantees
  published wheels carry the real version (decision 7).

### #11031 `rocm-version-validation`

- Strict wheel ↔ ROCm validation applies to wheels labelled `+rocm…` and
  `+devrocm…`, compared against the full identity (decision 4), except that dev
  identities ignore their commit token (decision 5, #11023).
- Dev-identity wheels (`+g<sha>[.dirty]`) warn once per process, naming the selected
  ROCm and saying it was not checked, then continue (decision 5).
- The client build ROCm check (decision 14) follows the same split: a mismatch is an
  error when the package is labelled `+rocm…`/`+devrocm…`, and stays a warning for
  `+g` packages. A different major version is an error either way (#11023).
- Tests:
  - release, nightly and dev labels match only their own channel;
  - a manifest `7.12.0.dev0+<sha>` matches the label `+devrocm7.12.0.dev0.<sha>`
    after normalization;
  - **stage reuse:** a manifest `7.12.0.dev0+<sha A>` matches a label
    `+devrocm7.12.0.dev0.<sha B>`; `7.12.0.dev0+<sha A>.asan` does not match
    `+devrocm7.12.0.dev0.<sha B>`; `7.12.0.dev0` never matches `7.12.0a20261021` or
    `7.12.0`; a commit-less custom-suffix `7.12.0.dev0` matches `7.12.0.dev0+<sha>`;
  - a `+g` wheel warns only once;
  - **round trip:** for each TheRock version kind, the label `release_metadata`
    stamps passes `_rocm.py`'s check against the same version as the installed ROCm
    would report it. The kinds: `7.12.0` (release), `7.12.0rc2`, `7.12.0a20261021`
    (nightly), `7.12.0.dev0+<sha>` (CI dev), `7.12.0a20261021+asan`,
    `10.1.0a20260811+bkc.20260814` and `7.12.0.dev0+sha.asan`. All pass against the
    stack-top code today; the test keeps it that way.

### #11032 `python-sdk-runtime`

- `invoke install` and standalone CMake stamp no ROCm identity into the wheel
  (decisions 1 and 5); only TheRock does.
- Inside a pip-SDK venv, the client's `--build-rocm` (decision 14) records the full
  `rocm_sdk_core.__version__` instead of the base `.info/version`, so the client
  check compares full identities against `PythonRocm`.
- `default_client_candidate` validates candidates by source revision (decision 3),
  not by distribution version.

This removes the full-versus-base rejection in pip-SDK environments.

- **Gate:** T2 merged in TheRock and included in a published nightly
  `rocm-sdk-core`. The PR body's "required SDK client and tool payload" is out of
  date: since Q095/Q109 made the client optional and lazy, `PythonRocm` needs only
  the tool console scripts (Q110), which means T2 (`clang-offload-bundler`).
  Shipping the client through `rocm[libraries]` is later work and should not block
  this branch. Update the PR body to name T2-in-a-nightly as the gate.
  - It is **not** `therock-ref`: TheRock CI's test jobs never import
    `rocm_sdk_core` (T8 installs only the `tensilelite` wheel), so they select
    `SystemRocm` before and after this branch.
  - It is **not** the gfx1250 lane's SDK pin: that lane runs tox, whose environment
    has no `rocm` package, so it reaches the SDK only through `ROCM_PATH` and also
    selects `SystemRocm` before and after this branch.
- **New CPU-only pip-SDK smoke check** (owned by this branch; the first slice of F2).
  No rocm-libraries job runs TensileLite with the pip SDK in the same interpreter
  today (`component-ci-tensilelite-coverage.yml` uses `setup-rocm-linux`, a native
  prefix from TheRock artifacts), so without it this branch changes runtime
  selection on a path no CI runs.
  - **Where:** a new tox environment (e.g. `pip-sdk-smoke`) and a small workflow on a
    CPU runner. Not the gfx1250 lane: the check needs no GPU, and that lane's SDK
    pin exists to match gfx1250's compiler to per-PR CI.
  - **Minimal install, unpinned:** a fresh venv;
    `pip install --pre --index-url https://nightly.repo.amd.com/rocm/whl-next/ rocm-sdk-core`
    (newest nightly; no `rocm[...]` meta-package, no device or library packages);
    then TensileLite from the checkout with only its runtime requirements. Confirm at
    implementation whether `rocisa` is needed for the import path exercised; skip it
    if not.
  - **Never trips the ROCm match:** TensileLite built from the checkout is
    `5.0.0+g<sha>`, so the wheel ↔ ROCm check only warns (decision 5); no client is
    involved, so decision 14's check doesn't run. A newer nightly therefore can't
    fail the job on version grounds.
  - **Asserts:** `PythonRocm` is selected with `rocm_sdk_core.__version__` as its
    identity, and `clang-offload-bundler` resolves through the interpreter's
    scripts directory. Records the installed `rocm-sdk-core` version in the job
    summary.
  - **Triggers:** PRs touching `tensilelite/_rocm.py`, `_runtime.py`,
    `_tensilelite_client_binding.py`, `setup.py`, `pyproject.toml`, `tox.ini` or the
    workflow; plus a daily schedule, so SDK regressions show up without a PR. A
    failure that reproduces on develop is an SDK regression; the summary names the
    nightly.
  - **Repo hygiene:** add the workflow to CODEOWNERS and the CI trigger maps in the
    same PR (AGENTS.md working rules).

### R1 (new branch) `users/alvasile/AIHPBLAS-3989-installed-wheel-test-artifacts`

A new PR stacked on #11032. It stops unpacking the `tensilelite` wheel into the
test-artifact root, so tests import only the installed wheel.

- **Change:**
  - remove the `install(CODE …)` step in hipBLASLt's CMake that runs
    `scripts/install_release_wheels.py` into `share/hipblaslt/tensilelite` (added in
    #11020);
  - keep installing the wheel under `wheels/` and the tests under `tensilelite/Tests`;
  - keep `rocisa`, `rocisa_stinkytofu_adaptor` and `stinkytofu` at their current
    paths, which T8's runner exposes through its shim directory;
  - remove `install_release_wheels.py` and its tests only if nothing else uses it.
    #11022's build-tree unpacking does use it, so expect to keep the script and drop
    only the artifact install step.
- **Tests:** an installed-tree layout test proving there is no `tensilelite/`
  package at the artifact root, and that the wheel and tests are present.
- **Gate:** land only after T8–T10 (the installed-wheel runner, wired to both jobs)
  are in rocm-libraries' `therock-ref`. TheRock `main`'s current runner imports the
  unpacked copy, so landing earlier breaks TheRock's TensileLite jobs.
- **Justification:** with `PYTHONPATH` at the component root, the unpacked copy
  shadows the installed wheel. Removing it makes the tests exercise what ships
  (decision 9).

### #11254 `tensilelite_packaging-stack` (decision records)

Restacked on top of R1, so the documentation branch stays at the top of the stack.

- **Remove code commit `9afc018`** from this branch; it moves to R0, except its
  `InvokeInstallPythonRocmSupport.md` lines, which stay here. After this, #11254
  changes only documentation.
- **Move the two root-level documents** `TensileLiteNamespaceSplitProposal.md` and
  `TensileLiteRelativeImportInvestigation.md` to `projects/hipblaslt/tensilelite/docs/`,
  in the commits that add them (`8e5dc62`, `d595095`). Both are self-described
  historical snapshots of the namespace-split stack; they belong with the other
  TensileLite design records. Search for references to the old paths (docs,
  `.github/`, tox, CODEOWNERS) and update them in the same commits.
- Record Part 2 in `PythonBuildGrillingDecisions.md` and
  `RocmWheelRuntimeGrillingDecisions.md`.
- Mark superseded: D3's build-time failure rule, D15, Q058–Q062, Q110/Q111
  (prefix identity), Q115, and Q119 plus items 3–4 of Q120 (local and invoke builds
  stamping a ROCm identity; replaced by decisions 1, 5 and 14).
- Also mark superseded (audit, 2026-11-11): Q063/Q064 (client validation by
  `--version`; now SHA, decision 3), Q081 (`pip install --force-reinstall` into the
  build Python; decision 9), Q082 (the import rewrite now lives in #11016 and covers
  13 files, not 11), Q112/Q113 (`THEROCK_PACKAGE_VERSION` selecting
  `TheRockCIRocm`; decision 7), the Q117 rows for PR 9, 18 and 20, and Q118
  (`SystemRocm` base-only matching; decision 4).
- Keep Q097/Q109 as they are.

### Note: update the decision log and related records

When the stack is amended, the written records must match Part 2. #11254 owns the
decision records; each code branch owns its own user-facing text.

- **Decision records on #11254**
  (`users/alvasile/tensilelite_packaging-stack`, `projects/hipblaslt/tensilelite/`):
  - `PythonBuildGrillingDecisions.md`: add Part 2's decisions as new entries with
    their justifications. Mark these superseded and point to the replacement:
    D3's build-time failure rule, D15, Q058–Q062, Q110/Q111's base-only prefix
    identity (replaced by decision 4's manifest rule), Q115, and the
    compatibility-package records Q031–Q035, Q039, Q053, Q084 and the compatibility
    parts of Q103/Q105 (decision 13), plus Q119 and items 3–4 of Q120 (local and
    invoke builds stamping the wheel; replaced by decisions 1, 5 and 14), and
    Q063/Q064, Q081, Q082, Q112/Q113, the Q117 rows for PR 9, 18 and 20, and Q118
    (see the #11254 entry for their replacements). Leave Q097/Q109 as they are.
  - `RocmWheelRuntimeGrillingDecisions.md`: update F5 (TheRock identity) and G004/G005
    (identity validation) for decisions 4–7, including that a missing or `"git"`
    TheRock version gives a dev identity.
  - `docs/PackagingDecisions.md`: add superseded notes on D3 and D15 pointing to the
    new entries, and on D9 (compatibility scope and lifetime) for decision 13.
    `docs/PackagingPlan.md` Phase 6 (compatibility) is updated to match.
  - `RocmVersionIdentityInvestigation.md`, `InvokeInstallPythonRocmSupport.md`,
    `PythonBuildImplementationPlan.md`, `TheRockPythonBuildImplementationPlan.md`,
    `docs/PackagingPlan.md`: correct statements that the build reads
    `.info/version`/`ROCM_PATH`, that client pairing uses the distribution version,
    or that TheRock must forward the version before the stack lands.
  - `docs/Public.md` stays the original proposal; add only a pointer to the current
    decisions.
  - Record the SHA-design evidence (the 2026-11-10 packaging review) and the
    `HIPBLASLT_VERSION_TWEAK` investigation as short findings, not full transcripts.
- **User-facing guidance on the code branches:** #11027 (`installed-workflow-guidance`)
  and the TensileLite `README.md`/`AGENTS.md` describe SHA pairing, the dev and
  release version formats, binding a self-built client, and the supported
  installation forms. Text introduced by an earlier branch is fixed in that branch.
- **PR descriptions:** each amended PR's body cites the Part 2 decisions it
  implements. Every PR-body edit is shown for approval before it is posted.
- **This review document** is committed on #11254 as
  `projects/hipblaslt/tensilelite/docs/PackagingIdentityPlan.md`, next to the other
  TensileLite design records, so the plan and the records don't drift apart.

## 1.3 TheRock: `users/alvasile/tensilelite_packaging`

### Step 1 — Squash into one commit holding the final state

1. Back up the local branch (`0abec119b`) and its remote tip (`b36e3c006`) under
   `refs/backup/<date>-tensilelite-packaging/`.
2. Create a working branch from `origin/main` and apply the branch's net change as
   one commit, then make the corrections below in that same commit.

Disposition of the 13 existing commits:

| Commit | Disposition |
| --- | --- |
| `727540988` enable TensileLite production prerequisites | Keep the requirements resolution and its tests → T4–T6. Keep the `source_only` artifact exclusion → T12. Drop `configure_stage.py` emitting `-DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR`: external-repo CI already passes it (`detect_external_repo_config.py`, `cmake_source_var`). |
| `4ffba3faf` ship client in runtime artifact, `0ac51569a` restore test-only client | Net result is "the client stays in `blas_test`". Keep only the resulting artifact checks → T13. |
| `52394ff4f`, `b808455d5`, `3e3a5fc19` wheel runner | Keep, with the corrections below → T7–T11. |
| `0b997ff1c` remove redundant artifact checks | Fold into T4 and T13. |
| `44ecb165f` pending-bump TODO comment | Drop. |
| `cbcec85c2` pass ROCm version, `0abec119b` drop it | Drop both, including the dead `THEROCK_ROCM_VERSION` propagation in the root `CMakeLists.txt`. |
| `96542cdcb` pass `THEROCK_PACKAGE_VERSION` | Keep → T1. |
| `8a3be03bc` `clang-offload-bundler` | Keep → T2. |
| `11c8eb76e` devel `.info/version` test | Keep → T3. |

Corrections to the runner (`test_tensilelite.py` / `pytest_runner.py`), all part of
the final state:

- **No label parsing.** Remove `verify_client_free_package_help`'s `rocm` prefix
  check, which rejects every CI wheel (`+devrocm…`), and its fake ROCm root. Run
  `python -m tensilelite --help` against the real artifact root.
- **Installed wheels must actually be imported.** `PYTHONPATH` becomes a temporary
  shim directory with symlinks to `rocisa`, `rocisa_stinkytofu_adaptor` and
  `stinkytofu` only. Before running tests, assert that `tensilelite.__file__` is
  under the interpreter's site-packages.
- **Existing path unless the artifact is wheel-only** (revised 2026-11-11): use the
  wheel path only when `share/hipblaslt/tensilelite/wheels/` exists **and** there is
  no unpacked `share/hipblaslt/tensilelite/tensilelite/__init__.py`; otherwise run
  today's `pytest_runner.py` path, exactly as `main` does now. This is the current
  test path, not new behaviour.
  - Why both conditions: #11020–#11032 ship `wheels/` but still unpack
    `tensilelite/` at the artifact root until R1. pytest puts the directory above
    the `tensilelite` package (the artifact root) on `sys.path` when collecting its
    tests, so the unpacked copy would shadow the installed wheel, and the import
    location check would fail on every branch from #11020 to R1 once T9/T10 are in
    `therock-ref`.
  - Changing pytest's import mode was rejected: #11016's bare test-helper imports
    rely on the same `sys.path` insertion.
  - R1 removing the unpacked copy is what switches the wheel path on. T14 removes
    the fallback once the submodule includes R1.
- **The current path's behaviour is preserved.** The branch's `pytest_runner.py`
  refactor changes `build_environment()` to *replace* `PYTHONPATH` with the component
  root, where `main` *prepends* to any existing `PYTHONPATH`. Keep `main`'s prepend in
  the shared code. Apply the reset (to the shim directory) only on the new
  installed-wheel path. Tests cover both.
- **One path for both jobs.** `tensilelite-common` uses the same runner with
  `TEST_CATEGORY=hw-common`.
- **Provenance** in the job summary and JUnit properties:
  - wheel versions;
  - client `--version` and `--source-revision` when a client is present;
  - the manifest's `rocm_package_version` and `the_rock_commit`;
  - the Python version.

The squash commit message records, per area, what changed and why, citing the Part 2
decision numbers. It also states that the client stays in `blas_test` (Q097).

### Step 2 — Split back out into small, single-purpose PRs

Status: the T1–T14 split was agreed on 2026-11-10.

Split the squash commit (the `split-to-stack` skill fits this). Each PR does exactly
one thing, carries its own justification in the commit message and PR body, and
works with TheRock's current rocm-libraries pin.

Where a PR builds on an earlier one, it lands after it. PRs with no dependency can
land in any order.

| PR | One change | Justification (decision) | Depends on | Works with today's pin because |
| --- | --- | --- | --- | --- |
| **T1** build(blas): pass `THEROCK_PACKAGE_VERSION` to hipBLASLt | One `CMAKE_ARGS` line in `math-libs/BLAS/CMakeLists.txt` | Lets hipBLASLt label TheRock wheels with the real package version and enables strict checks (decisions 5, 6) | — | develop ignores the variable |
| **T2** feat(python): `clang-offload-bundler` console script in `rocm-sdk-core` | `setup.py` entry point, `_cli.py` function, one `core_test.py` row | The pip-SDK adapter resolves tools only from the interpreter's scripts (Q110) | — | Independent of rocm-libraries |
| **T3** test(python): require `.info/version` in the devel root | One `devel_test.py` assertion | `.info/version` is the identity fallback for prefix-style roots (decision 4) | — | Independent of rocm-libraries |
| **T4** build: `configure_stage.py --rocm-libraries-source-dir` resolves the TensileLite requirements file | New option, `resolve_python_requirements()`, `configure_stage_test.py`. No CMake argument emitted. | The stage's Python dependencies must come from the rocm-libraries checkout being built, not TheRock's submodule (decision 9). Independent improvement; no rocm-libraries branch depends on T4–T6. | — | No-op until a topology entry names the file (T6) |
| **T5** ci: multi-arch workflows pass `--rocm-libraries-source-dir` | Three workflow files (portable Linux, Windows, WSL) | Points T4 at the external checkout in rocm-libraries-driven CI | T4 | No-op until T6 |
| **T6** build: install TensileLite requirements in the BLAS stage | One `python_requires` line in `BUILD_TOPOLOGY.toml` | Device generation runs TensileLite with the build Python. Today its dependencies come from a hand-copied `# hipBLASLt deps` list (`joblib`, `msgpack`) plus `PyYAML`/`packaging` in TheRock's root `requirements.txt`, so a new TensileLite dependency breaks TheRock until someone edits it. T6 makes rocm-libraries' own `requirements.txt` the only list, matching the commit being built (decision 9). **Not required by the stack:** develop and the stack top import nothing beyond what TheRock already installs. Cost: also installs `pandas`, which kernel generation doesn't import. Removing the hand-copied lines is a later cleanup, not part of T6. | T4, T5 | develop has `projects/hipblaslt/tensilelite/requirements.txt` |
| **T7** refactor(test): split `pytest_runner.py` into `build_environment` / `apply_execution_environment` / `run_phase` / `main` | Behaviour-preserving refactor, including `main`'s `PYTHONPATH` prepend, plus `pytest_runner_test.py` | Lets the wheel runner reuse category handling without duplicating it | — | No behaviour change |
| **T8** test(tensilelite): installed-wheel runner (not yet wired to any job) | New `test_tensilelite.py` and `tensilelite_runner_test.py`: install the one `tensilelite` wheel; shim `PYTHONPATH` with only rocisa, its adaptor and stinkytofu; assert the import location; `--help` against the real root; no label parsing; run today's path unless `wheels/` exists and no unpacked `tensilelite/` remains. No compatibility-wheel phase (decision 13). | Validation runs the installed artifacts (decision 9); TheRock doesn't parse labels (decision 8) | T7 | Unused until T9; runs today's path until R1's wheel-only layout. Tests include `+devrocm…`, `+rocm…` and `+g<sha>` wheels. |
| **T9** test: run the `tensilelite` job through the wheel runner | One `test_script` change in `fetch_test_configurations.py` | Starts testing the installed wheels for that job | T8 | Today's pin runs today's path |
| **T10** test: run the `tensilelite-common` job through the wheel runner | One `test_script` change (`TEST_CATEGORY=hw-common`) | Both jobs then test the same artifacts | T8 | As T9 |
| **T11** test(tensilelite): record provenance | Job summary and JUnit properties: wheel versions, client `--version`/`--source-revision` when present, manifest `rocm_package_version`/`the_rock_commit`, Python version | Requirement P5 (decision 9) | T8 | Missing values are recorded as absent |
| **T12** build(blas): exclude source-only TensileLite tests from `blas_test` | One `exclude` pattern in `artifact-blas.toml` | Source-tree contract tests don't belong in installed artifacts | — | The pattern matches nothing that shouldn't be excluded today |
| **T13** test: TensileLite artifact-layout checks | `test_artifact_structure.py`: when `wheels/` exists, the `tensilelite` wheel and installed tests are in `blas_test`, no `source_only` tests, and the client is in `blas_test` only | Guards the layout the runner relies on and Q097 client ownership | — | Checks apply only when `wheels/` exists |
| **T14** (after the submodule includes R1) test(tensilelite): remove the existing-path branch | Delete T8's existing-path branch and make T13's checks unconditional | Once the pinned rocm-libraries ships a wheel-only artifact (R1), that branch is dead code | Submodule includes R1, T8, T13 | Lands only after that bump; rebased onto `main` then |

Verification:

- the top of the split (T1–T13) equals the squash tree;
- each PR passes the unit tests it touches (`pytest_runner_test.py`,
  `tensilelite_runner_test.py`, `configure_stage_test.py`,
  `fetch_test_configurations_test.py`, `test_artifact_structure.py`, the `rocm_sdk`
  tests);
- T7 is shown to be behaviour-preserving by running the existing
  `pytest_runner_test.py` unchanged before adding new cases;
- record each branch's base and tip SHAs.

Push and open PRs only with approval.

## 1.4 Follow-ups (after this effort)

Not part of the rocm-libraries stack or TheRock T1–T14. Each is a separate effort,
recorded here so the target is explicit (decisions 11 and 12).

### F1 — TensileLite wheel as a TheRock release artifact (end state)

TheRock builds the `tensilelite` wheel once, as a release artifact. That artifact is
what generates hipBLASLt's kernel libraries and runs the TensileLite common tests.

```text
Stage W (new, architecture-independent, no compiler dependency; runs in parallel with compiler-runtime)
  in:  rocm-libraries checkout; THEROCK_PACKAGE_VERSION; TENSILELITE_SOURCE_REVISION (explicit)
  run: TheRock subproject "tensilelite-python": stage_release_source.py + pip wheel
       (or a CMake module extracted from #11020's inline recipe at that time)
  run: check_release_wheel_contents.py with the expected identity (decision 8)
  out: artifact "tensilelite": share/tensilelite/wheels/tensilelite-<version>-py3-none-any.whl
        │ fetched by later stages as a prebuilt dependency
        ▼
Stage math-libs (per architecture): hipBLASLt subproject, BUILD_DEPS tensilelite-python
  -DTENSILELITE_WHEEL_DIR=<artifact>/share/tensilelite/wheels
  1. configure: fail if the wheel's SHA differs from hipBLASLt's own (new F1 input)
  2. build _rocisa and stinkytofu in-tree (no ordering problem: the wheel is pure Python)
  3. unpack the wheel into the build tree; code generation via per-command PYTHONPATH
     with TENSILELITE_ROCM_ADAPTER=therock-ci (decisions 7, 9)
  4. build tensilelite-client from the same SHA → blas test component
        ▼
Publishing: copy the stage-W wheel unchanged next to the rocm wheels; never rebuild it
Tests: tensilelite and tensilelite-common jobs fetch the "tensilelite" artifact, install
  that wheel (T8 runner, given a wheel directory), use the client from blas_test
```

Work items:

- TheRock: the `tensilelite-python` subproject, the `tensilelite` artifact, group
  and stage. Its `source_paths` must equal `blas`'s, so TheRock CI's artifact reuse
  can't pair a wheel and a client from different commits. The configure-time SHA
  check is the backstop.
- TheRock: hipBLASLt `BUILD_DEPS` and `-DTENSILELITE_WHEEL_DIR`; test jobs fetch the
  artifact; the T8 runner takes a wheel-directory input.
- rocm-libraries: decide how stage W builds the wheel (call the Python scripts
  directly, or extract #11020's inline recipe into a CMake module), and add the
  `TENSILELITE_WHEEL_DIR` input with its configure-time SHA check. Both were moved
  here from #11020 (decision 9): they have no caller or CI until F1.
- TheRock: publishing copies the stage-W wheel unchanged.
- Stop copying the wheel into the `blas_test` component once the artifact exists,
  so there is only one wheel to trust.
- Stage W also runs on Windows, since device generation does. Wheels must be
  reproducible (`SOURCE_DATE_EPOCH`), or only one platform's wheel is published.
- Prerequisite: F3.

### F2 — Validation phases for both TheRock installation forms

- **Pip SDK:** a fresh venv with `rocm[libraries,device-<target>]` and the staged
  wheel, no `PYTHONPATH`, exercising `PythonRocm` (G010). #11032's CPU-only smoke
  check (newest nightly `rocm-sdk-core`, TensileLite from the checkout, adapter and
  tool resolution only) is the first slice; F2 extends it to the staged wheel,
  library packages and a GPU run.
- **Native packages:** install TheRock's native packages in a clean container
  (`test_native_linux_packages_install.yml`), then `pip install` the wheel,
  exercising `SystemRocm` with the manifest identity.
- Both record provenance (as T11).

### F3 — Move `rocisa_stinkytofu_adaptor` out of the source tree

Package it in the `tensilelite` wheel or in a rocisa artifact. Required before F1,
because stage W and the math-libs consumer won't have the source tree on
`PYTHONPATH`.

### F4 — rocisa packaging (D6)

rocisa, its adaptor and stinkytofu are still reached by path (the T8 shim, the
code-generation `PYTHONPATH`).

- **Target:** a standalone rocisa package built against the installed ROCm, using
  the Python stable ABI first, and linking the installed `libstinkytofu` rather than
  vendoring it.
- **Why it matters:** this is the remaining gap against requirements P1, P2 and P7,
  and the Origami-style risk of a duplicate native copy.
- **ROCm identity:** the rocisa package records the ROCm it was built against, using
  decision 4's sources: the manifest or `rocm_sdk_core` (full), or `.info/version`
  (base). It is checked against the runtime adapter with the same rules as the
  client (decision 14): a different major version is an error; any other mismatch is
  an error for TheRock-stamped builds and a warning for dev builds. Until rocisa is
  packaged it is built and used in the same tree, so build and run ROCm are the same
  by construction.

### F5 — Client in the runtime artifact (Q109)

Move `tensilelite-client` from `blas_test` to `blas_lib`, which delivers it through
`amdrocm-blas` and `rocm[libraries]` (a `tensilelite-client` console script for pip
installs). This waits for the client and the Python package to mature (decision 11).
The SHA and ROCm checks already work with a client from the test package or a
self-built one.

### F6 — Kernel-library compatibility with hipBLASLt (parked)

Generated kernel libraries carry no record of which TensileLite produced them, and
hipBLASLt doesn't check one at load. Parked; if picked up, apply the whole-repo-SHA
rule hipBLASLt already uses for tuning files (decision 12).

Recording which TensileLite generated a kernel library belongs here, together with
the check that reads it, inside the library's own serialized metadata, not as a
loose sidecar file. It becomes useful once generation and the wheel can come from
different places (F1). This replaces the sidecar #11022 used to plan.

### F7 — Say what each TensileLite job validates, and widen provenance (P5, P6)

- The `tensilelite` and `tensilelite-common` jobs test kernels that the checkout's
  own TensileLite generates. They do not test the hipBLASLt binaries shipped in the
  SDK, and a pass on one must not be read as a pass on the other. The smallest fix
  is a property in the job summary and the JUnit file, written by the T11 provenance
  step, that names which of the two a job validates. Decide whether that belongs in
  T11 or in a follow-up.
- Provenance (T11) does not yet record native component versions beyond the TheRock
  manifest, the test inputs or the generated artifacts. Decide which of these are
  worth recording and what a failing job should attach.

### F8 — Follow Origami's packaging stage (P8)

Origami is the reference implementation of the TheRock Python packaging stage, and
its PR (rocm-libraries#10975) is still open. Read it before designing F1's wheel
stage, so the two stages use one pattern for building, staging and validating a
wheel.

## 1.5 Implementation notes (updated 2026-11-12)

Where the implementation went beyond or differs from section 1.2:

- **History:** #11021, #11030 and #11031 are now one commit each, because every
  original commit implemented the replaced mandatory-identity design or a gate in
  code that #11021/#11023 rewrote. Original authors are kept. #11025's `985b133`
  became empty and was dropped.
- **#11018:** the build-ROCm lookup lives in `tensilelite/_build_rocm.py`. If the
  build root's manifest and `.info/version` disagree on the base version, the
  client records `unknown` with a configure warning instead of failing the build.
  The build can't repair the installation, and `unknown` warns at runtime
  (decision 14). The runtime treats the same disagreement in the selected ROCm as
  an error (decision 4).
- **#11019 / #11025:** the binding module owns the platform filename and CMake
  client output path. GEKO asks the source checkout through `invoke client-path`,
  because it computes the path before its editable install and its coverage env
  does not install TensileLite. Invoke defers GPU-helper imports until needed,
  so querying a path does not initialize the generator runtime. The bash writer
  owns argument quoting on Linux and Git Bash; callers contain no platform checks.
  On #11019, 63 binding/build-task tests passed. On #11025, 90 binding/build/tox
  tests passed and GEKO reported 606 passed, 40 integration/tool skips, and 89.58%
  coverage against its 80% floor. All new path and quoting tests ran. Native Windows
  execution is pending. #11032's candidate tests also use the shared helper;
  215 binding/runtime/build-task tests passed there. See Q135 for the reasons
  and first-run regression.
- **#11020:** the wheel also depends on a configured identity file (version, SHA,
  passed identity), so a new commit or identity rebuilds and renames the wheel even
  when no TensileLite file changed.
- **#11021:** an unpacked sdist keeps the version in its `PKG-INFO`, so a TheRock
  identity survives sdist → wheel; passing a different identity there is an error.
- **#11022:** codegen depends on the unpacked wheel through the new CMake variable
  `HIPBLASLT_PYTHON_INPUTS`; the client header is configured only when the client
  is enabled. The generator environment also carries `ROCM_VERSION=${hip_VERSION}`:
  TheRock's build prefix has no ambient `/opt/rocm` for the legacy version lookup.
  The regression and nine adjacent environment tests passed after the restack.
- **#11023:** `validate_client` returns the client's SHA and build ROCm together
  (`ClientIdentity`), so the client is queried once.
- **#11025 tox correction:** `skip_install` applies to all environments. The five
  environments that run package imports (`logic-corpus`, `rocisa`, `coverage-cpp`,
  `unit`, `coverage-unit`) explicitly install the checkout before their first import.
- **#11032:** the pip-SDK smoke check is `scripts/pip_sdk_smoke.py`, tox env
  `pip-sdk-smoke` and workflow `component-ci-tensilelite-pip-sdk-smoke.yml`. Run
  locally against nightly `10.3.0a20261111`, it selected `PythonRocm` and failed
  only on the missing `clang-offload-bundler` console script, as expected until T2
  is in a nightly.
- **Open gap:** with `ROCM_PATH` unset, CMake's build-root resolver falls back to
  `/opt/rocm`, not to `rocm-sdk path --root`. A client built in a pip-SDK venv
  records the full SDK identity only when `ROCM_PATH` points at the SDK root.

---

# Part 2: Decisions and justifications

All fourteen decisions were agreed in this review (decisions 1–13 on 2026-11-10,
decision 14 on 2026-11-11). Where a decision keeps
or replaces an existing record (Dn, Qnn, Gnnn), its section says so. The #11254
records are updated to match (section 1.2, "Note: update the decision log").

## Decision 1 — The build backend never discovers ROCm (Decided)

`setup.py` stamps a ROCm identity only when one is passed explicitly, and only
TheRock's CMake build passes one (`THEROCK_PACKAGE_VERSION`). Standalone CMake,
`invoke install`, tox and `pip install path` produce `5.0.0+g<sha>[.dirty]`. No
`ROCM_VERSION`, `ROCM_PATH`, `/opt/rocm` or `TOX_ENV_NAME` reads. The ROCm a native
artifact is compiled against is recorded on that artifact instead (decision 14).

Justification:

- **PEP 517 frontends isolate the backend.** tox's `.pkg` drops `[testenv]`
  `setenv`/`passenv` (confirmed in CI on #11016 `ec4c081` and `7023260`, and
  reproduced locally). pip's build isolation hides `rocm_sdk_core`. pip-SDK users
  rarely set `ROCM_PATH`.
- **Environment stamping produced inconsistent artifacts.** One local tox build of
  #11016 produced sdist `+rocm7.2.4` and wheel `+rocm9.9.9`. At the stack top, a plain
  `pip install path` fails outside tox.
- **Requirement P2** is that `pip install path/to/project` works against an
  installed SDK.
- **`ROCM_VERSION` already means the HIP build version** (`Toolchain/Component.py`,
  set by CMake from `hip_VERSION`).

Supersedes D3's "package builds fail when no ROCm version can be determined", the
#11021 `.pkg`/`0.0.0` bootstrap, and the #11025 `TOX_ENV_NAME` branch.

## Decision 2 — Pair `tensilelite` with `tensilelite-client` by the whole rocm-libraries SHA (Decided)

The Python package and the client each carry the full `git rev-parse HEAD` of
rocm-libraries. At first client use they must match exactly. Editable installs
compare the commit only and warn about uncommitted changes (decision 3).

Justification:

- **The contract between them is source-level.** `ClientWriter` writes files the
  client parses, and that format has no version of its own.
- **The `+rocm` tag only stood in for "same source"**, and it fails in both
  directions: different source with the same ROCm passes, and the same source with
  base versus full identities fails.
- **Releases pair automatically,** because the wheel and the client come from
  TheRock's single pinned checkout.
- **A whole-repo SHA was chosen over a subtree hash or an interface version for
  simplicity.** The accepted cost is that unrelated monorepo commits break pairing
  for source-built clients. That fails clearly, and generation without a client
  still works.

The ROCm identity stays the key for client ↔ ROCm, wheel ↔ ROCm, and the
published-wheel label. Supersedes Q058–Q062's semantic-version pairing.

## Decision 3 — How the SHA is obtained, stamped and compared (Decided)

**Lookup** (`tensilelite/_source_revision.py`, used by `setup.py`, the runtime and
CMake):

1. **Unpacked sdist** (`PKG-INFO` at the root): use only the saved
   `_build_info.py` value. If `TENSILELITE_SOURCE_REVISION` is set and differs, fail.
2. **Git, only if the tree belongs to the repo:**
   - `git -c safe.directory=* -c core.fsmonitor= -C <root> ls-files --error-unmatch setup.py`;
   - then the same command prefix with `rev-parse HEAD`.
3. **`TENSILELITE_SOURCE_REVISION`:** 40-hex, optionally with a dirty suffix. If git
   also produced a SHA and the base commits differ, fail instead of overriding.
4. **Nothing found:** no SHA, which never pairs, for pip and CMake alike. CMake warns
   at configure time, the client prints `none`, and `validate_client()`'s error says
   to build from a git checkout or pass the commit through
   `TENSILELITE_SOURCE_REVISION` (`-DTENSILELITE_SOURCE_REVISION` for CMake).
   No substitute ID is generated: an invented ID records nothing about which source
   was built, can't go through `parse_revision` without leaking into the version
   label, and makes wheels non-reproducible. Every known build path has git or
   receives the SHA explicitly (TheRock builds from a git submodule; #11020's staged
   copy receives CMake's value). Tarball builders know their commit and can pass it.

**Uncommitted changes:**

- dirty value `<sha>.dirty.<first 12 hex of sha256(records)>`;
- tracked changes: `git diff --raw -z --no-abbrev --no-renames --no-ext-diff
  --no-textconv --ignore-submodules=none HEAD --`. Hash sorted, length-prefixed
  path/mode/base-object/status records and streamed contents; record symlink
  targets and nested gitlink commits/changes rather than reading directories.
  Staging unchanged working-tree contents does not change the fingerprint;
- untracked, nonignored entries: sorted repository-relative path, file type,
  size and integer `st_mtime_ns` from `lstat()`. Never read their contents or
  follow links. The scope remains the whole repository, matching its SHA;
- version label `5.0.0+g<sha>.dirty` (no digest in the version).

**Cost and limits (review amendment, 2026-11-12):** Git still scans tracked state
and enumerates untracked paths. Changed tracked contents are streamed in bounded
chunks; each untracked entry needs only metadata, so large dumps and tarballs do
not add content-reading cost to repeated backend calls. This is not a fixed
runtime cap: tracked changed bytes and file count still matter. No persistent
metadata cache is introduced.

Python provides integer `st_mtime_ns` on Linux and Windows; use it directly,
not a float conversion. Filesystem precision can be coarser than nanoseconds.
Two untracked files with the same path, type, size and observed timestamp can
have different contents but the same fingerprint; this risk is explicitly
accepted. Copied dirty checkouts can acquire different timestamps and identities.
The digest is a practical build-pairing fingerprint, not proof of every byte.
Clean SHA pairing, installed exact matching, editable commit matching/warnings
and separate ROCm validation remain unchanged. These fixes require no TheRock
changes.

**Stamping:**

- `tensilelite/_build_info.py` is generated only into `build_lib` (wheels) and the
  sdist release tree, never into the source tree;
- editable installs carry no stamp;
- the client gets the same value from the same CMake configure;
- dev builds use `5.0.0+g<40-hex>[.dirty]`; releases keep `+rocm…`/`+devrocm…` with
  the SHA only in `_build_info`.

**Comparison** (at the first client request):

- installed package: exact string match;
- editable install: compute from git lazily; a commit mismatch fails, uncommitted
  changes on either side warn (option (b) of the review). There is no override
  variable.

Justification. Each point below was verified in scratch experiments by a packaging
review on 2026-11-10 (pip 26.2.1, setuptools 82, tox 4.56.1, git 2.43):

- **Reading a saved value before git gives a stale SHA.** After an sdist build wrote
  the file into the source tree, a later `pip install` from a newer commit reported
  the old SHA. Hence: stamp only build outputs, and trust the saved value only in an
  unpacked sdist.
- **A random dirty ID breaks pip.** pip runs `setup.py` three times
  (`egg_info`, `dist_info`, `bdist_wheel`), and differing IDs made pip reject the
  wheel ("unexpected file name"). The fingerprint is the same on an unchanged tree, and
  it lets tox's separate `pip install -e` and `invoke build-client` steps pair.
- **Editable installs go stale when stamped,** and `direct_url.json` records no commit
  for path installs. Hence lazy computation at runtime.
- **The editable rule balances strictness and development.** Commit-only matching
  keeps the check meaningful across commits without forcing a client rebuild after
  every Python edit. Releases and CI, which use installed packages, stay exact.
- **`safe.directory` must be the repo root or `*`;** the package directory is
  rejected. The initial probe can inspect an unrelated parent repository, so
  trusting its config is not equivalent to running the package's `setup.py`.
  Disable fsmonitor on every query with an empty value: older Git treats `false`
  as a hook pathname. Disable diff/textconv helpers separately. These controls
  are not a general sandbox for all Git configuration.
- **A tree without `.git` inside another checkout inherits that repo's HEAD;** the
  `ls-files` guard detects it.
- **`git diff --quiet HEAD` ignores untracked files,** hence the explicit untracked
  set. Porcelain `git diff` refreshes the index, so a `touch` alone doesn't count as
  a change; plumbing `diff-index` would.
- **The dirty scope is the whole repository** because the SHA identifies the whole
  repository (decision 2). A narrower scope would label a tree "clean" while the
  client or hipBLASLt sources it is paired with carry uncommitted changes. The
  cost is that any untracked, non-ignored file in the checkout marks builds dirty.
- **PEP 440 normalization strips leading zeros from all-digit segments.** Keep the
  `g` prefix and keep digests out of the version. `.dev0` labels would be excluded
  by `>=5.0`.
- **`config_settings` can't reach `setup.py` through `setuptools.build_meta`** without
  a custom backend wrapper, so the environment variable stays. tox `.pkg` drops it,
  and git supplies the value there.
- **Git works in TheRock builds.** `HIPBLASLT_VERSION_TWEAK` is set in every public
  TheRock build checked (CI run 42757568973, nightlies 27392289755 and rockrel
  29378454574, release 40359466840); the build image sets `safe.directory '*'`. Use
  `rev-parse`, not `describe --dirty`: the only empty tweak found came from an
  internal simulator SDK whose checkout had no working tree.
- **CMake records the SHA at configure time,** so it must re-configure when HEAD
  moves. The wheel filename is fixed at configure.
- **Wheels are byte-identical across builds only with `SOURCE_DATE_EPOCH` set.**
- **A separate TheRock Python stage must pass the SHA explicitly,** because the wheel
  and the client would come from two builds.

Not yet handled:

- Edits made after a CMake configure are not reflected in a dirty value until the
  next configure.
- Concurrent builds from one source tree already race on the shared `build/lib`,
  which `CleanBuildPy` deletes; this predates the SHA design.
- `export-subst` (`$Format:%H$`) could carry the SHA in `git archive` tarballs if
  TheRock's tarballs are made that way, which is not yet established.

## Decision 4 — ROCm identity per installation form (Decided)

| Scenario | Adapter | Identity |
| --- | --- | --- |
| 1. TheRock pip SDK | `PythonRocm` | `rocm_sdk_core.__version__` (full) |
| 2. TheRock native prefix | `SystemRocm` | Manifest `rocm_package_version` (full); `.info/version` only if the manifest is missing or invalid |
| 3. Pre-TheRock native prefix | `SystemRocm` | `.info/version` (base) |
| 4. TheRock CI build | `TheRockCIRocm` | Forwarded `THEROCK_PACKAGE_VERSION`, else dev |

TheRock-built installs compare the full published identity whether installed by pip
or natively.

Guardrails for reading the manifest:

- read only `rocm_package_version`;
- accept it only if it parses as a release identity (local TheRock builds record
  `"git"`);
- treat a missing or malformed file as absent and fall back to `.info/version`;
- if the manifest and `.info/version` disagree on the base version, report the
  installation as inconsistent.

Justification:

- **Releases are separated from nightly and dev builds.** With base-only matching, a
  release `7.14.0` wheel and a `7.14.0a…` nightly or `7.14.0.dev0+…` dev install look
  identical, as do two different nightlies on the same line. The full identity tells
  them apart, so a release wheel is never accepted on a nightly or dev install, or
  the reverse.
- **The pip and native forms of one TheRock build behave the same.** `PythonRocm`
  already compares the full `rocm_sdk_core.__version__` (Q108/Q110/G004, enforced
  from #11032). Native installs of the same build now compare the same value, which
  also removes the full-versus-base asymmetry between the two forms.
- **Every TheRock-built root ships the manifest.**
  `share/therock/therock_manifest.json` comes from `base/aux-overlay`. In native
  packages it ships in `amdrocm-llvm`, while `.info/version` ships in `amdrocm-base`;
  hence the consistency check.
- **The manifest is TheRock's build-provenance file, not a documented contract;**
  hence reading one field defensively with a fallback. If TheRock later publishes
  the full identity in a documented file, `SystemRocm` switches to it. That would be
  a TheRock change, so it isn't required now (decision 6).
- **Pre-TheRock roots have only `.info/version`,** so scenario 3 keeps base-only
  matching.
- **The pip SDK wins inside its own interpreter, with no borrowing across adapters**
  (unchanged from Q110).

Supersedes Q110/Q111's base-only identity for TheRock-built prefixes.

## Decision 5 — Strict ROCm checks only for release-labelled wheels (Decided)

| Wheel label | Produced by | Behaviour on mismatch with the selected ROCm |
| --- | --- | --- |
| `+rocm<release or nightly>` | TheRock release and nightly builds | Fail, comparing full identities (decision 4) |
| `+devrocm<dev identity>` | TheRock CI/dev builds, once T1 passes the version | Fail, same rule, ignoring the identity's commit token |
| `+g<sha>[.dirty]` | Builds with no ROCm identity: `pip install path`, tox, `invoke install`, standalone CMake, TheRock before T1 | Warn once per process (selected ROCm, "not checked"), then continue |

Justification:

- **A dev wheel makes no ROCm claim.** `+g<sha>` says which source it came from, not
  which ROCm it was built for, so there is nothing to enforce.
- **Failing dev wheels would break normal use:** every source install (the
  P2 requirement), every tox and CI lane, and TheRock CI before T1.
  That would contradict decision 6.
- **Releases are kept apart from nightly and dev builds.** With full identities
  (decision 4), a nightly or `+devrocm` wheel is rejected on a release install, and
  the reverse.
- **Dev identities ignore their commit token** (revised 2026-11-11). TheRock's dev
  version is `X.Y.Z.dev0+<GITHUB_SHA>[.<variant>]` (`compute_rocm_package_version.py`),
  so it changes with every commit. TheRock CI can reuse a stage's artifacts from a
  baseline `main` run: explicitly through `prebuilt_stages`/`baseline_run_id`
  (exposed by rocm-libraries' `therock-multi-arch-ci.yml` and the ASAN workflow,
  and honored in the default `dry-run` mode), or automatically with
  `stage_reuse_mode: reuse-stage`. Reusing compiler-runtime brings the baseline's
  `therock_manifest.json` (from `base/aux-overlay`), while the rebuilt math-libs
  wheel carries the current run's version, so every such run would fail a full
  comparison. The commit token identifies the run, not the ROCm content, and reuse
  is TheRock asserting the content is unchanged. Release, `rc` and nightly
  identities never reuse stages and stay full; `.dev0` still differs from both,
  and variant tokens (`asan`) still separate builds. The client check (decision 14)
  is unaffected: the client reads the same reused manifest as the runtime.
- **Without reuse, inside TheRock CI after T1,** the `+devrocm` wheel and the
  artifact root come from one build, so they match in full.
- **The file-format risk is still covered:** the wheel ↔ client SHA check
  (decisions 2 and 3) stays exact for installed packages regardless of label.
- **One warning per process** keeps the log readable while still recording that no
  check ran.

## Decision 6 — #11016–#11028 land without TheRock changes; #11030 onward are gated slices (Decided)

- **Below #11030:** a missing or `"git"` `THEROCK_PACKAGE_VERSION` produces a dev
  identity, not a configure error. This replaces the `0.0.0` placeholder those
  branches use today.
- **#11030 → R1:** conditional enablement slices, each merging only after its
  TheRock gate is available: in rocm-libraries' `therock-ref` for #11030, #11031
  and R1; in a published nightly `rocm-sdk-core` for #11032 (section 1.1).
- **From #11030 on:** a missing version is an error, and `"git"` stays the dev
  identity.

Justification:

- **This keeps the stack's original design.** The lower branches keep TheRock CI
  green without TheRock changes. The PR bodies of #11030, #11031 and #11032 already
  declare their TheRock prerequisites.
- **#11016 already passes every TheRock-built job against the pinned TheRock
  `main`.** Its remaining failures are rocm-libraries-only (gfx1250 packaging, the
  logic-corpus import) or infrastructure (an ASAN download timeout).
- **Local TheRock builds keep working,** since they default to `"git"`.
- **Making a missing version an error once TheRock always passes one** means a lost
  version can't silently produce unchecked nightly or release wheels.

Supersedes Q115 only in allowing `"git"` as the local dev identity. Its requirement
for a real forwarded identity, from #11030 on, is kept.

## Decision 7 — Select the TheRock build adapter with an internal marker (Decided)

- **Marker:** when hipBLASLt builds under TheRock (`HIPBLASLT_ENABLE_THEROCK=ON`),
  CMake sets `TENSILELITE_ROCM_ADAPTER=therock-ci` only in the environment of the
  code-generation commands. It is the only thing that selects `TheRockCIRocm`. Only
  `therock-ci` is accepted; other values are errors. The variable is internal and
  documented as such.
- **Version:** `THEROCK_PACKAGE_VERSION` carries only the ROCm version, and only
  when a real one is known. Below #11030, missing or `"git"` means a dev identity.
  From #11030 on, missing is a configure error and `"git"` is the local dev
  identity (decision 6).
- **Cross-check:** when the wheel has a `+rocm…`/`+devrocm…` label and a version was
  passed, compare them. Inside one CMake build this always holds; in the end state,
  where the wheel comes from another TheRock stage, it catches a mismatched wheel.
- **Wheel contents check:** in TheRock builds, `check_release_wheel_contents.py`
  receives the expected identity and fails on a label mismatch.
- **Kept:** the explicit `--cxx-compiler`/`--offload-bundler` arguments and the
  graph-prepared `PATH`.

Justification:

- **One variable was doing two jobs.** Today `_therock_ci_version()` treats
  "`THEROCK_PACKAGE_VERSION` is set" as "this is a TheRock build", and also uses its
  value as the ROCm version, rejecting anything that isn't a release. When the
  version is unknown (TheRock `main` passes none, and local builds use `"git"`), the
  build either fails or falls through to `PythonRocm`/`SystemRocm`. Those look for
  `/opt/rocm` or `ROCM_PATH`, which TheRock deliberately keeps out of builds (Q068)
  and its build machines don't have.
- **The build-time mode must never switch on for users.** An installed nightly, CI
  or release build must always be validated by `PythonRocm`/`SystemRocm` against the
  installed ROCm (decisions 4 and 5). Keying the mode on `THEROCK_PACKAGE_VERSION`,
  a common TheRock name that can appear in developers' and CI shells, could put an
  installed wheel into build-time mode and skip that check. An internal marker that
  only CMake sets, for its own commands, can't leak that way.
- **An environment variable reaches every worker.** The adapter is chosen at
  `import tensilelite`, and `create-library` re-imports in joblib and process-pool
  workers. A command-line argument would need plumbing through every entry point.
- **Rejected alternatives:**
  - keying on `THEROCK_PACKAGE_VERSION` with relaxed validation (simpler, but can
    leak into runtime, as above);
  - explicit tool paths only (import still selects an adapter);
  - `SystemRocm` rooted at `THEROCK_TOOLCHAIN_ROOT` (rejected by Q112; base-only
    identity; needs `ROCM_PATH`, which conflicts with Q068);
  - detecting TheRock automatically (guessing; contradicts Q113).
- **It holds in both states.** Now, the wheel is built inside hipBLASLt's CMake;
  in the end state, it is consumed from a TheRock release-artifact stage. The same
  command environment selects the adapter either way. The version round trip
  (stamped label → runtime check against the installed ROCm) was confirmed against
  the stack-top code for release, rc, nightly, CI dev, `+asan` and bkc versions.

## Decision 8 — rocm-libraries owns the version format; TheRock never parses it (Decided)

rocm-libraries owns the format of the wheel's version label (`+rocm…`, `+devrocm…`,
`+g<sha>[.dirty]`).

TheRock does:

- pass the inputs (`THEROCK_PACKAGE_VERSION`, later the SHA);
- find the wheel by project name only (exactly one `tensilelite` wheel);
- install, test and publish it;
- record the full version string, uninterpreted, as provenance.

TheRock does not inspect or rebuild the label's local part, and does not build fake
ROCm roots from it.

When TheRock wants a check, it calls rocm-libraries code:

- `check_release_wheel_contents.py` with the expected identity, at build time and,
  later, before publishing;
- the installed wheel's own runtime validation (`_rocm.py`) against the real artifact
  install, at test time.

Justification:

- **A second copy of the rules already broke.** The TheRock branch's runner
  (`verify_client_free_package_help`) requires the local version to start with
  `rocm`. It rejects every TheRock CI wheel (`+devrocm…`, confirmed by calling it
  directly) and every dev wheel (`+g<sha>`).
- **The format keeps changing** (`+rocm` → `+devrocm` → `+g<sha>`/`.dirty` during this
  work). One owner means it can change in one repository without breaking the other.
- **The fake root proved nothing.** It compared the label with a value derived from
  the label. Validating against the real artifact install is what a user's install
  does.
- **Checking stays possible without duplicating the rules.** TheRock supplies the
  expected value, and rocm-libraries judges it.

Cost: TheRock cannot detect a wrong label without rocm-libraries code. That is
accepted, because rocm-libraries code runs wherever a label is produced or
consumed.

Applies to T8 (runner), T11 (provenance) and the later TheRock wheel stage.

## Decision 9 — Test and build what ships, without modifying the builder (Decided)

- **A. Tests import the installed wheel.**
  - T8: `PYTHONPATH` is a temporary directory with links to only `rocisa`,
    `rocisa_stinkytofu_adaptor` and `stinkytofu`, and the runner asserts that
    `tensilelite` was imported from the installed location before testing.
  - R1: removes the unpacked copy from the test-artifact root once TheRock uses that
    runner.
- **B. Kernel generation uses the built wheel, without installing it.**
  - #11022: develop runs code generation from the source tree via a per-command
    `PYTHONPATH` (`hipblaslt_configure_bundled_python_command()`) and installs
    nothing. Keep that mechanism, but point `PYTHONPATH` at the wheel unpacked into
    the build tree instead of `<src>/tensilelite`.
    - The interpreter's site-packages are never modified.
    - Python imports the wheel's exact files, and `importlib.metadata` finds its
      version through the unpacked `.dist-info`.
    - No console scripts or dependency resolution, as with today's `--no-deps`;
      dependencies come from the build's Python as they do on develop.
  - The wheel-recipe module and `TENSILELITE_WHEEL_DIR` are not added in this
    effort; they move to F1 (revised 2026-11-11).
- **C. Build-time Python dependencies come from the rocm-libraries checkout being
  built** (TheRock T4–T6): an independent improvement replacing TheRock's
  hand-copied hipBLASLt dependency list; not required by any rocm-libraries branch.

Justification:

- **Tests must exercise the artifact.** With `PYTHONPATH` at the component root, the
  unpacked copy shadows the installed wheel, so the "installed wheel" test never
  runs the wheel. Requirement P4 is validation against what ships, with no
  substitution (the Origami lesson, ROCM-29472/ROCM-32106).
- **Building must not modify the builder.** `pip install --force-reinstall` changes
  TheRock's shared build venv or a developer's Python, and can collide between
  builds. Develop already shows code generation needs no install.
- **Kernel generation must use the built wheel.** Using the wheel rather than the
  source tree is what the end state requires, where the wheel comes from a TheRock
  release-artifact stage. Kernel generation already consumes the wheel file
  (#11022), so F1 only changes where that file comes from.
- **No seams without a caller.** With one pure-Python wheel and no ROCm discovery,
  the CMake recipe is thin: stage the source, `pip wheel`, run the contents check,
  copy. What's worth reusing is already in Python (`setup.py`,
  `_source_revision.py`, `check_release_wheel_contents.py`,
  `stage_release_source.py`). A module or input with no caller or CI until F1 would
  be untested and would guess at an interface F1 will define.
- **A private build-directory venv** would also leave the builder untouched, but is
  heavier, and is not develop's pattern.

Open points:

- `rocisa`, its adaptor and `stinkytofu` are still reached by path until rocisa is
  packaged (D6).
- `rocisa_stinkytofu_adaptor` is imported from the source tree during code
  generation. Before the end state it must move into the wheel or a rocisa artifact.

## Decision 10 — Support for pre-TheRock native installs (Decided: best effort, no CI)

| Capability | On a pre-TheRock `/opt/rocm` |
| --- | --- |
| `import tensilelite`, `create-library`, `logic`, other client-free commands | Supported. `SystemRocm` uses `.info/version`; a dev `+g<sha>` wheel warns once (decision 5) |
| Benchmark and tuning (`tensilelite run`) | Only with a client built from source against that ROCm and bound through `tensilelite-configure-client` (for example `invoke install`, which builds both from one checkout, so the SHAs match) |
| A client requested without a binding | Clear error: this ROCm install has no client, plus how to build and bind one |
| A release-labelled wheel (`+rocm…`) for another ROCm | Rejected by the strict check (decision 5) |
| CI | None. Tested by hand when needed, on a pre-TheRock `/opt/rocm` such as `/opt/rocm-7.2.4` |

Justification:

- **These installs ship nothing TensileLite needs beyond the toolchain.** There's no
  `tensilelite-client` (`/opt/rocm-7.2.4` has no `libexec/hipblaslt/`), no TheRock
  manifest, and no TensileLite wheel for these releases, so source installs are the
  only route. Generation is the realistic floor.
- **The SHA check works unchanged with a self-built client** (decision 3), so no
  special case is needed.
- **Every rocm-libraries CI lane that installs ROCm uses TheRock artifacts**
  (`setup-rocm-linux`), so there's no lane to extend. A dedicated lane would need a
  maintained pre-TheRock image for releases that are aging out. Math CI (Jenkins)
  may cover generation on pre-TheRock images; not checked.

#11027's guidance states this support level.

## Decision 11 — Client artifact ownership is unchanged for now (Decided; keeps Q097)

`tensilelite-client` stays in `blas_test`, packaged as `amdrocm-blas-test` (Q097).
`blas_lib`/`rocm[libraries]` remains the final state (Q109), as later work.

Justification:

- The TheRock branch's move to the runtime artifact was reverted on the branch.
- Promotion waits for the client and the Python package to mature, and shipping the
  client in every runtime install is a product decision.
- Nothing in this effort needs the move: the SHA and ROCm checks work with a client
  from the test package or a self-built one.

Follow-up: F5 (section 1.4).

## Decision 12 — Deferred to follow-ups (Decided)

- **Wheel production stays in hipBLASLt CMake for this effort.** The TheRock
  release-artifact stage, with kernel generation and tests consuming that artifact,
  is the end state (F1). This effort only makes kernel generation consume the wheel
  file (#11022, decision 9); the recipe module and `TENSILELITE_WHEEL_DIR` are F1.
  F1 needs decision 3's explicit SHA passed to both the wheel and the client.
- **Validation phases for the pip SDK and native packages** are F2.
- **`rocisa_stinkytofu_adaptor` moves out of the source tree** before F1 (F3).
- **rocisa packaging stays deferred (D6).** It is the remaining gap against
  requirement P1, standalone packaging (F4).
- **Checking generated kernel libraries against hipBLASLt's SHA** is parked (F6).
  The serialized format gained new fields in about 33 commits over 16 months,
  roughly 7 a month recently, so any cross-checkout guarantee would need the same
  whole-repo-SHA rule hipBLASLt already applies to tuning files.

Follow-ups: F1–F8 (section 1.4).

## Decision 13 — Remove the compatibility package (Decided)

`tensilelite_tensile_compat` is dropped. It was a commands-only wheel: wrappers
named `Tensile`, `TensileCreateLibrary`, `TensileLogic`, `TensileGetPath` and seven
others, each printing a deprecation warning and calling the canonical code, pinned
to exactly the same `tensilelite` version. It is removed from #11020, where it was
introduced, and its later changes come out of #11021, #11022, #11025 and #11026. The
legacy `Tensile*` console scripts are not kept anywhere; the `tensilelite` command
family is the only command surface.

Justification:

- **It has only ever been an internal package,** so there are no external users to
  migrate. The deprecation window it provided protects nobody.
- **Nothing in the stack calls the legacy names.** Device generation runs
  `python -m tensilelite …`. The remaining mentions at the stack top are comments,
  docstrings and in-process function names (for example
  `lib_logic_to_yaml.TensileLibLogicToYaml`), not command invocations.
- **It simplifies everything downstream:**
  - one wheel to build, check, stamp, unpack and publish;
  - no exact cross-wheel version pin that forces both to come from one build;
  - TheRock's runner loses its second install-and-test phase and its "one of each,
    matching versions" rule (Q105);
  - the future release-artifact stage produces one wheel.

Consequence: anyone using develop's `Tensile`, `TensileCreateLibrary` and related
console scripts switches to `tensilelite run`, `tensilelite create-library`, and so
on. #11027's guidance and #11020's PR body say so.

Supersedes D9 (compatibility scope and lifetime); Q031–Q035, Q039, Q053 and Q084; and
the compatibility parts of Q103/Q105.

## Decision 14 — Native artifacts record the ROCm they were built against (Decided)

Only TheRock stamps a ROCm identity into the `tensilelite` wheel (decisions 1 and 5).
Native artifacts that compile against ROCm record their own build ROCm instead, and
that is checked against the ROCm the runtime selects:

- `tensilelite-client` gets `--build-rocm` (#11018). The check is reported in #11019,
  enforced on major-version mismatch in #11023, and strict for release-labelled
  packages in #11031.
- rocisa does the same once it is packaged (F4).

| Client build ROCm vs. selected ROCm | `+rocm…`/`+devrocm…` package | `+g<sha>` package |
| --- | --- | --- |
| Same (full, or base when either side is base-only) | OK | OK |
| Different major version | Error | Error |
| Other mismatch | Error | Warn once |
| Client build ROCm `unknown` | Warn once | Warn once |

Justification:

- **The mismatch is real only where code is compiled against ROCm.** The
  `tensilelite` wheel is pure Python: `setup.py` builds no extension, and it runs
  whatever toolchain the runtime selects. `tensilelite-client` links HIP, HSA,
  hsakmt and amd_smi (`client/CMakeLists.txt`), and rocisa links `hip::host`
  (`rocisa/CMakeLists.txt`).
- **A build-machine label on the wheel checks the wrong artifact.** It would fail
  mismatches the wheel doesn't depend on, and miss the ones the client does.
- **The wheel's TheRock label is a different claim:** "released as part of this ROCm
  build, tested with this client". Only the release process can make that claim.
- **A different major version is always an error** because HIP's soname changes
  across major versions; within a major version HIP promises backward
  compatibility, so dev builds only warn.
- **It works in all four scenarios,** including standalone builds that carry no
  wheel identity, and needs no TheRock change (decision 6).

Supersedes Q119 and items 3–4 of Q120 (local and invoke builds stamping the wheel).
