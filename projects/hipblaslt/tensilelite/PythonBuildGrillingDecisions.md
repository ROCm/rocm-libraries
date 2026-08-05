<!-- Copyright Advanced Micro Devices, Inc., or its affiliates. -->
<!-- SPDX-License-Identifier: MIT -->

# TensileLite Python Build Grilling: Questions and Decisions

Status: Living decision log  
Started: 2026-08-06  
Scope: hipBLASLt/TensileLite source build, Python/native integration, TheRock CI,
release-wheel production, and artifact-based testing

## Purpose

This file records the questions, confirmed facts, decisions, rationale, deferred
choices, and open questions from the active design grilling session. It is
updated as the discussion continues.

No implementation should be inferred from this document until the session ends
with an explicitly confirmed shared understanding.

## Status vocabulary

- **Accepted:** explicit user decision.
- **Confirmed fact:** verified from current repository/workflow code.
- **Deferred:** deliberately postponed.
- **Open:** not yet decided or awaiting evidence.
- **Superseded:** an earlier answer replaced by a later decision.

## Wheel terminology

The **canonical ROCm artifact wheel** is the transferable, unbound TensileLite
wheel produced for the ROCm artifact set. Existing target and validator names
use “release wheel” for this artifact. Until rocisa has proper distribution
packaging (Q101), it is a controlled ROCm artifact rather than an independently
pip-installable distribution.

## Current high-level direction

The emerging source-build flow is:

```text
CI/workflow installs tensilelite/requirements.txt into the selected Python
                              │
                              ▼
                         CMake configure
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          build CMake _rocisa      build tensilelite-client
                 │                         │
                 └────────────┬────────────┘
                              ▼
       expose only build-tree rocisa through PYTHONPATH
                              │
                              ▼
 force-reinstall canonical wheel into the found Python
       and configure the exact build-tree client
                              │
                              ▼
             run real logic/create-library commands
                              │
                              ▼
              generate hipBLASLt device libraries
```

The release/test-artifact flow is separate:

```text
build canonical and compatibility release wheels
build/package tensilelite-client
package raw rocisa + _rocisa and source-only tests
upload artifacts
                              │
                              ▼
fresh test job and fresh Python venv
download/flatten artifacts into ./build
install canonical wheel
inject raw rocisa through PYTHONPATH
run canonical tests
install compatibility wheel
run compatibility-only tests with explicit flag
```

## Chronological question and decision log

### Q001 — Must raw CMake prepare the source-development Python environment?

**Question:** Must a TheRock build work through CMake without a preceding
hipBLASLt-specific editable install?

**Decision: Accepted — yes.**

TheRock invokes hipBLASLt as a CMake subproject. The build graph must own the
native/client/Python preparation ordering rather than require an undocumented
out-of-band bootstrap.

### Q002 — Is there a current installed-wheel CMake consumer?

**Question:** Is there currently a packaging or downstream job that configures
hipBLASLt CMake using an installed TensileLite wheel?

**Answer: Confirmed for current scope — no known consumer.**

The wheel exists only on the feature branch, and no checked-in preset or CI path
currently selects the installed-package `SYSTEM` CMake mode.

### Q003 — Should `SYSTEM` remain despite no current consumer?

**Decision: Accepted — remove or defer `SYSTEM`.**

Keep one CMake source-build path for TheRock. Reintroduce an installed-wheel
CMake consumer mode only when a concrete downstream or packaging workflow needs
it.

### Q004 — Must the native client remain required?

**Question:** Must TensileLite commands require `tensilelite-client`, including
commands that do not currently execute computational native code?

**Decision: Accepted — yes.**

Long-term generation work is intended to move from Python into the native
layer. The client/native dependency is therefore an intentional direction, not
an accidental dependency to remove. Q102 confirms that the dependency,
distribution, and CI wiring should be designed for that end state even though
the current client is used only for benchmarking.

### Q005 — Does import-time client validation create a bootstrap cycle?

**Confirmed fact — no.**

There is no cycle if CMake enforces this graph:

```text
build _rocisa
build tensilelite-client
install TensileLite
import/run TensileLite
```

PEP 517/660 metadata construction does not need to import the installed
TensileLite package before the client exists.

### Q006 — How are rocisa and `_rocisa` structured today?

**Confirmed fact:**

- `rocisa` is the Python package/facade.
- `_rocisa` is the native nanobind extension.
- Integrated `develop` builds `_rocisa` as a CMake target and exposes the build
  package through `PYTHONPATH`.
- Standalone `pip install -e rocisa` invokes a separate scikit-build-core CMake
  build.

### Q007 — Should the TheRock source build build `_rocisa` itself?

**Decision: Accepted — yes.**

A CMake-only TheRock source build owns both native prerequisites:

- `_rocisa`
- `tensilelite-client`

It must not assume rocisa is preinstalled in the selected Python.

### Q008 — What rocisa wiring must be restored?

**Decision: Accepted — derive rocisa wiring from the build mode.**

`HIPBLASLT_ENABLE_DEVICE=ON` requires:

- `_rocisa` as a code-generation dependency;
- the rocisa build-package parent on `PYTHONPATH`;
- CMake dependency ordering that builds `_rocisa` before Python code generation.

Standalone rocisa and coverage workflows may request `_rocisa` independently.
A true host-only build (`HIPBLASLT_ENABLE_DEVICE=OFF` and client disabled) does
not build it. Proper rocisa release packaging remains follow-up work.

### Q009 — Must rocisa be installed as a pip distribution for source builds?

**Decision: Accepted — no.**

`import rocisa` is sufficient for the current source build. The CMake-built raw
package and extension may remain build-tree artifacts exposed through
`PYTHONPATH`.

### Q010 — Should rocisa path injection remain temporary technical debt?

**Decision: Accepted — yes.**

No proper rocisa wheel/install integration is required now. Only rocisa is
injected; TensileLite itself is installed from the canonical wheel.

### Q011 — Should CMake create a private Python venv?

**Question:** Why not install into the Python found by CMake?

**Decision: Accepted — do not require a private venv.**

The relevant CI builds run in disposable containers. CMake may install into the
found Python provided it is writable. The selected interpreter is a
single-owner build resource: concurrent hipBLASLt build directories must not
install or configure TensileLite in the same Python environment.

### Q012 — Must the found Python be a virtual environment?

**Decision: Accepted — no.**

The requirement is only that the selected Python environment is writable.

### Q013 — Should CMake probe writability during configuration?

**Decision: Accepted — no.**

Let the authoritative pip operation fail if the environment is not writable.
Do not add a separate imperfect writability probe.

### Q014 — How should TensileLite be installed for source code generation?

**Decision: Accepted — TheRock installs the canonical wheel; editable installs
remain a local-development workflow.**

Local development may use:

```bash
python -m pip install \
  --editable <tensilelite-source> \
  --no-deps \
  --no-build-isolation
```

Runtime dependency provisioning belongs to CI/developer setup, not this pip
operation.

TheRock builds the canonical ROCm artifact wheel, installs that unchanged wheel
into the build-job Python, and applies the installation-local client binding
with `tensilelite-configure-client`.

The installation command must use:

```bash
python -m pip install --force-reinstall --no-deps <exact-canonical-wheel>
```

The force reinstall is required because source changes do not necessarily
change the wheel version. CMake must configure the exact build-tree client
immediately after this installation.

### Q014A — Should TheRock use the canonical wheel for build-time generation?

**Decision: Accepted — yes.**

This exercises the same production package code/resources that will be shipped.
CMake dependencies must rebuild and reinstall the wheel when relevant package
sources or metadata change.

### Q015 — Should pip install runtime dependencies implicitly?

**Decision: Accepted — no.**

Use `--no-deps` to prevent network access, upgrades, and hidden package
resolution during the CMake build.

### Q016 — Where do pure-Python dependencies come from?

**Confirmed fact:**

- Standalone developers currently install
  `projects/hipblaslt/tensilelite/requirements.txt` manually.
- The rocm-libraries TheRock wrapper installs only `TheRock/requirements.txt`.
- Native TheRock staged workflows support per-artifact `python_requires` via
  `BUILD_TOPOLOGY.toml` and `configure_stage.py`.

### Q017 — Should dependencies be copied into `TheRock/requirements.txt`?

**Decision: Accepted — no.**

TensileLite owns its requirements file; do not duplicate its dependency list in
TheRock.

### Q018 — Where should workflows install TensileLite requirements?

**Decision: Accepted.**

Use both integration mechanisms:

1. Native TheRock staged builds: add the TensileLite requirements file to the
   `blas` artifact's `python_requires` in `BUILD_TOPOLOGY.toml`.
2. rocm-libraries wrapper workflows: explicitly run:

   ```bash
   pip install -r projects/hipblaslt/tensilelite/requirements.txt
   ```

### Q019 — Which workflows must provision the requirements?

**Decision: Accepted — every workflow that builds hipBLASLt device libraries.**

This includes Linux, Windows, sanitizer, and other build workflows when they
enable the relevant hipBLASLt device-generation path. Unrelated lint-only flows
need not install them unless they configure that path.

### Q020 — Should CMake duplicate a dependency import checklist?

**Decision: Accepted — no.**

`requirements.txt` is the runtime dependency source of truth, and
`pyproject.toml` `[build-system]` is the build-backend requirement source of
truth. Because wheel commands use `--no-build-isolation`, workflows must provide
pip and those build requirements in the selected Python. Do not hard-code a
second import-name or version checklist in CMake; the authoritative pip
wheel/install command reports missing build tools.

### Q021 — Should the environment target run an explicit final import check?

**Decision: Accepted — no.**

The first real `python -m tensilelite logic` or `create-library` command imports
the package and exposes any missing dependency or native/client problem. This
matches the simpler `develop` behavior.

### Q022 — Is failure at the first real command acceptable for missing runtime dependencies?

**Decision: Accepted — yes.**

Workflows own dependency installation. The first consumer command is the
authoritative validation.

### Q023 — What should `PYTHONPATH` contain?

**Decision: Accepted.**

For source builds, `PYTHONPATH` contains only the parent of the CMake-built raw
rocisa package. TensileLite comes from the canonical wheel force-installed into
the selected Python environment; its installation directory is not added to
`PYTHONPATH` explicitly.

### Q024 — Should ordering depend on command/source order?

**Decision: Accepted — no.**

CMake must encode ordering as target dependencies:

```text
_rocisa ───────────────┐
                       ├──> TensileLite Python environment target
tensilelite-client ────┘                 │
                                         ├──> logic validation
                                         ├──> create-library
                                         └──> ext-op generation
```

### Q025 — Should there be a single CMake helper for Python generator commands?

**Decision: Accepted — yes.**

The helper must attach both the configured command environment and the required
CMake target dependencies so callers cannot forget one half.

### Q026 — Should hipBLASLt gtest-data generation use the TensileLite helper?

**Decision: Accepted — no.**

`hipblaslt_gentest.py` only converts YAML to test data. It should use
`Python3_EXECUTABLE` directly and remain outside the rocisa/client/TensileLite
generation graph.

### Q027 — Does TheRock install builds into `/opt/rocm`?

**Confirmed fact — no.**

TheRock creates build-local subproject trees:

```text
TheRock/build/math-libs/BLAS/hipBLASLt/build
TheRock/build/math-libs/BLAS/hipBLASLt/stage
TheRock/build/math-libs/BLAS/hipBLASLt/dist
TheRock/build/artifacts
TheRock/build/dist/rocm
```

Subprojects install into their local stage. The final merged developer tree is
`TheRock/build/dist/rocm`.

### Q028 — How do build artifacts reach tests?

**Confirmed fact:**

- Build and test are separate jobs/containers.
- Build artifacts are uploaded under the workflow run ID.
- The test job creates a new venv.
- Artifacts are downloaded and flattened into `./build`.
- The test job treats `./build` as the ROCm prefix.

Therefore, build-job Python/site-packages state does not survive into tests.

### Q029 — What must survive from build to test?

**Decision: Accepted.**

The combined reconstructed artifact set must contain:

- canonical TensileLite release wheel;
- temporary compatibility release wheel;
- the production-owned `tensilelite-client` from `blas_lib`;
- raw rocisa package and `_rocisa` extension;
- source-only TensileLite tests and configuration;
- compatibility tests and configuration;
- required native ROCm runtime/toolchain artifacts.

### Q030 — Is rocisa currently a release wheel in this flow?

**Confirmed fact — no.**

rocisa can technically build a wheel through scikit-build-core, but ROCm/TheRock
release packaging for that wheel is deferred. Existing integrated/test flows
use a raw package plus native extension through `PYTHONPATH`.

### Q031 — Should test jobs use a rocisa wheel now?

**Decision: Accepted — no.**

Continue injecting the raw rocisa package from test artifacts. rocisa wheel
packaging remains future work.

### Q032 — What is the accepted test-artifact layout?

**Decision: Accepted.**

```text
share/hipblaslt/tensilelite/
  wheels/
    tensilelite-*.whl
    tensilelite_tensile_compat-*.whl
  rocisa/
    __init__.py
    _rocisa.so or _rocisa.pyd
  tensilelite/Tests/
    unit/
    common/
    ...
  compat/tests/
  pytest.ini
  test_categories.yaml

libexec/hipblaslt/tensilelite/
  tensilelite-client
```

The `libexec` entry is supplied by the production runtime artifact and appears
here as part of the reconstructed test root; `blas_test` does not own a
duplicate client.

### Q033 — Should tests ship inside the production wheel?

**Decision: Accepted — no.**

Production wheels contain production package code/resources. Tests and their
configuration ship only in the `blas_test` artifact.

### Q034 — Where should wheels be installed in the test job?

**Decision: Accepted.**

The TensileLite-specific `pytest_runner.py` should install local artifact wheels
into the already-created test venv before invoking pytest. Do not burden the
generic test-environment action with component-specific installation.

Q103 clarifies that this is a thin TensileLite-owned phase runner which
delegates ordinary pytest execution to TheRock's generic runner.

### Q035 — Should the compatibility wheel be installed by the main tests?

**Decision: Accepted — install and test it while it remains supported.**

The file and test flow must explicitly note its near-term removal.

### Q036 — How do canonical and compatibility tests avoid masking each other?

**Decision: Accepted — two-phase testing.**

```text
Phase 1:
  install canonical TensileLite wheel only
  run the complete normal suite

Phase 2:
  install compatibility wheel
  run compatibility-only tests
```

When compatibility is removed, delete phase 2 and its wheel/test inputs.

### Q037 — Where do compatibility tests live?

**Decision: Accepted.**

All compatibility tests live under `compat/tests/`, separate from the canonical
TensileLite test tree.

### Q038 — How are compatibility tests selected?

**Decision: Accepted.**

- Compatibility tests are skipped by default.
- Register an explicit flag, recommended spelling `--run-compat`.
- Compatibility CI always passes that flag.
- Tests use a `compat` marker or equivalent collection hook to enforce default
  skipping.

### Q039 — What compatibility coverage is required?

**Decision: Accepted — cover every legacy entry point.**

Tests must verify:

- each expected console-script name exists in installed package metadata;
- the exact delegated target function;
- exact argument ordering and values;
- return-code propagation;
- deprecation warning behavior.

Expensive underlying commands may be mocked, but argument forwarding must be
asserted exactly.

### Q040 — Where does wheel validation occur today?

**Confirmed fact:**

The feature branch's CMake wheel target builds canonical and compatibility
wheels, then runs `scripts/check_wheel_contents.py` before stamping and
installing the wheel directory into test artifacts.

The renamed release-artifact validator must verify canonical and compatibility
versions, the compatibility wheel's exact canonical dependency pin, expected
console scripts and wheel tags, absence of custom client binding metadata, and
required canonical resources.

### Q041 — Are custom client bindings universally forbidden in wheels?

**Decision: Accepted — yes. All wheel archives remain unbound.**

Client bindings belong to an installed Python environment and are created only
by the post-install `tensilelite-configure-client` command. Canonical,
compatibility, local, non-editable, and editable wheel archives contain no
machine-local client metadata.

### Q042 — How should release-wheel validation be named?

**Decision: Accepted.**

Rename:

```text
scripts/check_wheel_contents.py
→ scripts/check_release_wheel_contents.py
```

The name distinguishes ROCm artifact-content validation from ordinary local
wheel construction.

### Q043 — How should the wheel-build target be named?

**Decision: Accepted.**

Use a name that clearly says it builds release wheels:

```text
tensilelite-build-release-wheels
```

Recommended output directory:

```text
tensilelite-release-wheels/
```

### Q044 — Does the compatibility wheel remain in the release-wheel target?

**Decision: Accepted — yes.**

The same target builds canonical and compatibility release wheels until the
compatibility package is removed.

### Q045 — How should the build-time client be resolved?

**Decision: Accepted.**

- keep the client separate from TheRock's toolchain root;
- install the canonical wheel into build-job Python;
- bind the exact CMake-built client through shared installation-local metadata;
- use `tensilelite-configure-client` for an already-installed wheel;
- do not use PATH fallback or mutate the ROCm root.

### Q046 — Can a synthetic client-only `ROCM_PATH` work?

**Confirmed fact — not by itself.**

TensileLite uses `ROCM_PATH` for:

- `.info/version` release metadata;
- `tensilelite-client` standard location;
- compiler/assembler/bundler/readelf discovery;
- Windows `amdclang++ --rocm-path`, which requires headers and libraries.

A synthetic root must be a complete usable SDK view, not merely a client
directory.

### Q047 — Where does the ROCm version for release wheels come from?

**Decision: Resolved by Q053–Q054 and Q069.**

- The base ROCm version is available from `TheRock/version.json` during
  top-level configure.
- CI computes the full package version before the build, including dev, nightly,
  prerelease, or local suffixes.
- The full value is passed to top-level TheRock as `THEROCK_PACKAGE_VERSION`.
- `THEROCK_PACKAGE_VERSION` is not currently propagated into the hipBLASLt
  subproject.
- hipBLASLt receives `HIPBLASLT_ENABLE_THEROCK`, its stage install prefix, and
  `THEROCK_STAGE_INSTALL_ROOT`, but no authoritative package-version variable.
- TheRock's `hip-clr/dist` toolchain root exists before hipBLASLt configuration
  and contains rocm-core's `.info/version`.
- Wheels use base `A.B.C` from that file.
- Full package-channel/SHA naming is deferred.

### Q048 — Should TheRock explicitly pass its version into hipBLASLt?

**Decision: No separate version variable is needed for current scope.**

Use `.info/version` from `THEROCK_TOOLCHAIN_ROOT` through the scoped build-time
ROCm-root resolver. Do not propagate full `THEROCK_PACKAGE_VERSION` now.

### Q049 — Where is `.info/version` produced, and how should client discovery work without it?

**Decision: Resolved by Q053–Q057.**

Use separate SDK-root and client-binding seams:

```text
TheRock build phase:
  SDK/version root = THEROCK_TOOLCHAIN_ROOT (hip-clr/dist)
  client = exact CMake-built executable through installation-local binding

TheRock test phase:
  SDK/version root = reconstructed ROCM_PATH
  client = production path under that root

Local source use:
  client = explicit source-install or post-install binding
```

### Q050 — Can an existing wheel receive a client binding at pip-install time?

**Confirmed fact:** Standard `pip install existing.whl` does not invoke the
wheel's PEP 517 build backend and cannot consume the source-build
`--config-settings` hook. Configure an already-built wheel with the explicit
post-install `tensilelite-configure-client` command.

### Q051 — Does the current version check validate a PATH client?

**Confirmed fact — no.**

PATH discovery is not part of the accepted resolver because it cannot identify
or validate the intended client reliably. Q052 defines the complete precedence.

### Q052 — What precedence should client discovery use?

**Decision: Accepted.**

Final contract:

1. If installation-local binding metadata exists, use only its exact client and
   never fall back.
2. Otherwise resolve only the standard client under the selected `ROCM_PATH`.
3. Do not search PATH.

### Q053 — Does TheRock have a usable ROCm root before hipBLASLt code generation?

**Confirmed fact — yes. This corrects the earlier assumption.**

The amd-hip toolchain root is the `hip-clr/dist` tree. `hip-clr` has rocm-core as
a runtime dependency, so this dist contains the SDK and `.info/version` before
hipBLASLt configures. It is not the final merged `build/dist/rocm`, but it is a
valid build-time ROCm root.

The resulting version flow is:

```text
TheRock/version.json
  → base ROCM_VERSION=A.B.C
  → rocm-core/stage/.info/version
  → hip-clr/dist/.info/version
  → hipBLASLt source build
```

The build-time client remains the only missing artifact and should be resolved
independently from the SDK root.

### Q054 — What does the final `.info/version` contain?

**Confirmed fact:** It contains base `A.B.C`, not the full dev/nightly/RC
`THEROCK_PACKAGE_VERSION`. The release-wheel ROCm tag must therefore match the
base version unless the artifact metadata policy is deliberately changed.

Current TheRock full wheel-version examples are:

```text
CI/dev:      10.1.0.dev0+<full-git-sha>
nightly:     10.1.0a20260806
prerelease:  10.1.0rcN
```

The suffix is package-publication identity, while `.info/version` currently
records only ROCm compatibility identity (`10.1.0`).

### Q055 — Is a post-install `configure-client` mechanism viable?

**Decision: Accepted.**

Configure the installed environment after pip completes:

```bash
python -m tensilelite_configure_client \
  --client /absolute/path/to/tensilelite-client
```

The command is importable outside normal `tensilelite` initialization, validates
the client, stores the installation-local binding, and leaves the original
wheel unchanged.

The binding overrides only the client executable. Commands still require a
complete matching ROCm SDK through the selected `ROCM_PATH`.

### Q056 — Can the native client report its own compatibility identity?

**Decision: Accepted and simplified by Q070–Q079.**

Add a no-GPU command:

```text
tensilelite-client --version
```

It prints the single canonical value such as `5.0.0+rocm10.1.0`. Package import
validates it once against the installed Python distribution version. Source SHA
identity is deferred.

### Q057 — Can hipBLASLt install its client into TheRock's build-time ROCm root?

**Confirmed architectural constraint:** The build-time root is `hip-clr/dist`,
owned by the hip-clr/toolchain subproject and consumed as an input by hipBLASLt.
hipBLASLt should not mutate another subproject's dist tree.

hipBLASLt content reaches the final ROCm tree later:

```text
hipBLASLt build
  → cmake --install into hipBLASLt/stage
  → artifact slicing into blas_lib / blas_test
  → flatten artifacts into TheRock/build/dist/rocm
  → test job flattens artifacts into ./build
```

The source-build client therefore exists before it can be part of the final
merged ROCm root and needs a separate build-time binding/discovery mechanism.

**Decision: Accepted — keep the build-time client separate.**

Do not mutate `hip-clr/dist`. Use it as the build-time SDK/ROCm root and specify
the CMake-built client independently for the Python TensileLite installation.

### Q058 — Can the client path be passed directly to pip install?

**Decision: Accepted — do not pass client bindings through pip.**

Pip installs an unbound wheel for every source, editable, and prebuilt workflow.
When a custom client is required, run `tensilelite-configure-client` immediately
after installation.

### Q059 — Do all installations share one binding implementation?

**Decision: Accepted — yes.**

Use one client-binding module for:

- absolute path and executable validation;
- native client identity/version query;
- Python/client compatibility checks;
- binding metadata schema and parsing;
- RECORD entry construction/update;
- runtime binding validation.

Use two thin adapters:

1. `tensilelite-configure-client`: writes the binding into an already
   installed distribution and updates its RECORD.
2. Runtime resolver: reads the binding or resolves the standard client without
   performing installed-state mutation.

The shared module must be importable before normal TensileLite package runtime
initialization, because configuration may be establishing the client required
by that initialization.

### Q060 — What must change when this becomes the only client override mechanism?

**Decision: Accepted — use one pre-runtime binding policy with thin adapters.**

The final contract is:

- keep the binding metadata as a bare JSON absolute-path string in installed
  `.dist-info`;
- share path validation, version-command policy, version comparison, metadata
  parsing, and RECORD-row construction in a pre-runtime module;
- use thin adapters for installed-distribution mutation and runtime resolution;
- make runtime use configured binding exclusively when present, with no fallback
  to `ROCM_PATH`, PATH, YAML, globals, or command-line overrides;
- retain standard `ROCM_PATH/libexec/...` lookup only when no binding exists;
- validate both configured and standard clients with plain `--version`;
- keep configuration importable before normal package initialization;
- mutate binding metadata and RECORD under an inter-process lock using atomic
  replacement for each file and explicit recovery ordering; and
- keep every wheel archive free of machine-local client bindings.

### Q061 — How do local source installs configure a custom client?

**Decision: Accepted — install first, then configure.**

```bash
python -m pip install --editable /path/to/tensilelite \
  --no-deps --no-build-isolation
python -m tensilelite_configure_client \
  --client /absolute/path/to/tensilelite-client
```

`invoke install` may perform both steps to preserve a one-command developer
workflow.

### Q062 — Can a configured binding be removed?

**Decision: Accepted — yes.**

Provide:

```bash
tensilelite-configure-client --reset
```

Under the installation lock, reset removes the binding metadata and rewrites
RECORD using atomic replacement for each file and deterministic recovery
ordering. Subsequent processes return to standard
`ROCM_PATH/libexec/...` lookup.

### Q063 — Can TheRock install and configure an unmodified release wheel for build-time use?

**Confirmed design — yes.**

Dependency flow:

```text
build canonical release wheel without custom binding
build _rocisa
build tensilelite-client
            │
            ▼
force-reinstall canonical wheel into build-job Python
run tensilelite-configure-client with exact CMake client target
            │
            ▼
run logic/create-library/ext-op generation
```

`tensilelite-configure-client` changes only the installed distribution metadata
in that Python environment and updates its installed RECORD. It does not modify
the original `.whl` archive.

Packaging therefore uses the original validated wheel file, which contains no
machine-local client binding. The separate test job installs that unchanged
wheel and resolves the packaged client from its reconstructed `ROCM_PATH`.

### Q064 — Does build-time code generation depend on the compatibility wheel?

**Decision: Accepted — no.**

Use separate CMake outputs/targets:

```text
canonical TensileLite release wheel
  → required by build-time Python environment and device generation

compatibility release wheel
  → required only by release artifact aggregation and compatibility test phase

tensilelite-build-release-wheels
  → aggregate target for both transferable wheels
```

Removing compatibility later must not change the canonical build-time generation
dependency graph.

### Q065 — Does full `THEROCK_PACKAGE_VERSION` become `.info/version`?

**Confirmed fact — no.**

TheRock passes base `A.B.C` to rocm-core as `ROCM_VERSION`; rocm-core owns and
installs `.info/version`. The full CI/dev/nightly/RC package version is separate
publication/build identity. It is recorded in TheRock manifest/package metadata,
and the manifest deliberately derives base `A.B.C` from it as a separate field.

Therefore:

```text
.info/version                = runtime compatibility identity (A.B.C)
THEROCK_PACKAGE_VERSION      = package/build publication identity
```

TensileLite's `+rocm...` compatibility tag should match the base value in
`.info/version` unless a different runtime compatibility policy is explicitly
adopted.

**Decision: Accepted — use the base `.info/version` value for ROCm compatibility.**

### Q066 — Should non-release source builds include the TensileLite git revision?

**Decision: Deferred by Q069.**

Current scope does not add source revisions or release-channel identity to the
package version. A later design must use a PEP 440-valid version and the
rocm-libraries/TensileLite source revision rather than the TheRock revision.

### Q067 — Why is the source SHA after `+`?

**Confirmed packaging rule:** PEP 440 reserves the portion after `+` for local
build/source identity. The public portion before `+` permits structured release,
prerelease, postrelease, and numeric development fields, but not an arbitrary
alphanumeric Git SHA such as `_8f3412`.

Any future layout must keep non-public identities in one local segment:

```text
+rocm10.1.0.g<rocm-libraries-sha>
```

ROCm compatibility comes first for continuity with the existing package
contract; source identity follows it. The `.dev0`, `aYYYYMMDD`, or `rcN` segment
before `+` controls release ordering.

### Q068 — Can a PEP 440 version contain multiple `+` separators?

**Confirmed fact — no.**

PEP 440 permits one local-version separator. Additional identity fields must be
dot-separated inside the single local segment:

```text
5.0.0.dev0+rocm10.1.0.g8f3412abcd12
```

Parsing should use `packaging.version.Version` and a strict local-segment grammar
rather than splitting the raw string on multiple plus signs.

### Q069 — Is source-SHA/channel versioning in the current implementation scope?

**Decision: Accepted — no; defer it.**

For the current work, every TensileLite wheel uses only:

```text
<TensileLite component version>+rocm<base .info/version>
```

Example:

```text
5.0.0+rocm10.1.0
```

Do not add release-channel or source-revision plumbing now. Q066–Q068 record
only the PEP 440 constraints for later design work.

### Q070 — Is native client identity validation in scope now?

**Decision: Accepted — yes.**

Add a no-GPU machine-readable client identity command and validate custom client
bindings and the standard ROCm-relative client through the shared binding
infrastructure.

Exact stale-source detection remains dependent on Q071: semantic versions alone
cannot distinguish two different source builds that both report component
version `5.0.0`.

### Q071 — Is exact source build identity required outside the package version?

**Decision: Accepted — semantic version only for current scope.**

Validate only the existing semantic TensileLite/generator version. Exact source
commit identity and same-version stale-build detection are deferred future work.

### Q072 — Should the native client also report its build-time ROCm version?

**Decision: Accepted — report the canonical combined distribution version.**

The canonical Python distribution version remains:

```text
5.0.0+rocm10.1.0
```

The client prints that same value through the plain `--version` contract in
Q073. Configuration and import compare it exactly with the installed Python
distribution version. The ROCm portion remains derived from `.info/version`.

Source SHA/build identity remains deferred.

### Q073 — Does client version output need JSON?

**Decision: Accepted — no.**

The current scope has one canonical compatibility value, so the native interface
is simply:

```bash
tensilelite-client --version
```

with stdout:

```text
5.0.0+rocm10.1.0
```

The configuration tool trims and parses that string with Python packaging
version utilities, then compares it exactly with the installed TensileLite
distribution version. A structured format can be introduced later only if the
client must report additional independent fields.

### Q074 — Where is an installation-local binding stored?

**Decision: Accepted.**

`tensilelite-configure-client` writes binding metadata into the installed
TensileLite `.dist-info` directory and rewrites `RECORD` under an inter-process
lock. Each file uses atomic replacement with deterministic recovery ordering.

Consequences:

- the binding belongs to one Python installation;
- the original wheel remains unchanged and transferable;
- runtime resolves it through `importlib.metadata`;
- `--reset` removes both the metadata file and RECORD row;
- uninstall can account for the configured file.

Configuration and runtime resolution identify the distribution that owns the
imported TensileLite package and reject ambiguous visible installations.

### Q075 — When is a custom client's version validated?

**Decision: Accepted — twice.**

1. `tensilelite-configure-client` validates before writing the binding.
2. Runtime validates again on first client resolution in each Python process,
   then caches the successful result for that process.

This detects an incompatible executable that later replaces the originally
configured file at the same path without paying the subprocess cost on every
client access.

### Q076 — Is the standard ROCm-relative client version also validated?

**Decision: Accepted — yes.**

Do not rely solely on artifact-set provenance. A Python TensileLite installation
can remain while `/opt/rocm` is upgraded or replaced underneath it.

For both configured and standard clients, validate:

- file exists and is executable;
- `tensilelite-client --version` exactly matches the installed Python
  TensileLite distribution version;
- the Python distribution's ROCm compatibility tag matches
  `$ROCM_PATH/.info/version` when a standard ROCm root is used.

### Q077 — Is client-version validation cached?

**Decision: Superseded by Q079.**

Q079 validates and freezes the client once during package import. There is no
subsequent filesystem or version revalidation in that process.

### Q078 — What is the native `--version` command contract?

**Decision: Accepted.**

`tensilelite-client --version` must:

- perform no GPU initialization;
- read no benchmark or generation configuration;
- print exactly one canonical version line to stdout;
- keep stderr empty and return zero on success; and
- complete within a caller-enforced five-second timeout.

This makes it safe for install-time configuration, package initialization, and
runtime client resolution. Callers report distinct diagnostics for timeout,
loader failure, signal, nonzero exit, malformed output, extra output, and
version mismatch.

### Q079 — When is client existence/version validated during normal runtime?

**Decision: Accepted — once at package import.**

Package initialization:

1. resolves configured or standard client path;
2. verifies it is a regular executable file;
3. runs `tensilelite-client --version`;
4. compares exactly with the installed Python distribution version;
5. freezes the validated path for the process.

Subsequent `client_executable()` calls return the frozen path without filesystem
or version revalidation. Replacement while a process is running is explicitly
out of scope.

### Q080 — Must TheRock pass a new build-ROCm-path cache variable?

**Confirmed correction — no.**

TheRock's generated amd-hip toolchain file already defines
`THEROCK_TOOLCHAIN_ROOT` inside the hipBLASLt subproject. The existing
`pre_hook_hipBLASLt.cmake` already requires that variable and uses it to augment
the toolchain PATH.

When `HIPBLASLT_ENABLE_THEROCK=ON`, hipBLASLt can use
`THEROCK_TOOLCHAIN_ROOT` as `ROCM_PATH` for wheel construction and Python
code-generation commands. A duplicate `HIPBLASLT_BUILD_ROCM_PATH` input is not
needed.

### Q081 — Did `develop` require a complete `ROCM_PATH` to build kernels/client?

**Confirmed fact — no.**

On `develop`, TheRock supplies native compilation through:

- the generated CMake toolchain and explicit C/C++ compiler paths;
- the hipBLASLt pre-hook adding TheRock compiler directories to PATH;
- explicit create-library compiler arguments;
- build-tree rocisa exposed through PYTHONPATH.

The client is a CMake target and kernel code generation can find tools through
the explicit compiler/PATH setup. A complete Python-visible `ROCM_PATH` is not
what makes those builds work.

The feature branch adds a new requirement: release wheel construction and
package import derive/validate ROCm compatibility through
`$ROCM_PATH/.info/version`. Mapping `THEROCK_TOOLCHAIN_ROOT` to `ROCM_PATH` is
therefore package-contract plumbing, not a new compiler requirement.

### Q082 — Why does TheRock unset `ROCM_PATH` for subprojects?

**Confirmed fact — deliberate hermeticity policy.**

TheRock unsets `ROCM_PATH`, `ROCM_DIR`, `HIP_PATH`, and `HIP_DIR` from both
subproject configure and build commands so projects cannot discover an ambient,
uncontrolled, potentially incompatible installed SDK.

The policy originated in commit:

```text
d7863db5bcd1633c7ba544be3a819cebbed9e9b2
Unset HIP_PATH and related env vars for sub-projects. (#685)
```

It addressed TheRock issue #670 and a Windows failure where an installed SDK
redirected subprojects to `C:\Program Files\AMD\ROCm`. The solution was applied
on all platforms as defense in depth.

The replacement contract is explicit and graph-owned:

- `THEROCK_TOOLCHAIN_ROOT` identifies the staged toolchain root;
- compiler flags receive explicit hip/device-library paths;
- dependency packages resolve through TheRock's provider and CMAKE prefix;
- executable directories flow through CMAKE program path and controlled PATH.

No analogous TheRock subproject globally restores `ROCM_PATH`. A narrowly scoped
`ROCM_PATH=${THEROCK_TOOLCHAIN_ROOT}` only for TensileLite wheel/package Python
subprocesses is compatible with the policy because the value is TheRock-owned,
not ambient. It must not be set globally in the subproject environment/cache.

### Q083 — What does “legacy tool discovery” mean here?

**Clarification:** It describes path-based discovery in older project build
logic, not deprecated compiler tools.

Modern CMake integration would pass explicit executable paths or imported
targets. Existing BLAS/Tensile code still locates some compiler utilities by
searching PATH, so TheRock's project-specific pre-hooks prepend graph-owned
toolchain directories. TheRock comments call this path munging a compatibility
“reacharound” for old project assumptions.

### Q084 — May hipBLASLt read from `THEROCK_TOOLCHAIN_ROOT`?

**Clarification — yes, as a read-only graph-owned dependency.**

TheRock forbids discovery through ambient, uncontrolled `ROCM_PATH` values. It
explicitly injects `THEROCK_TOOLCHAIN_ROOT` so subproject compatibility logic can
use the selected toolchain. hipBLASLt already reads executables from it through
its pre-hook.

The constraints are:

- do not mutate the toolchain root;
- do not globally restore an ambient ROCM_PATH;
- a narrowly scoped Python subprocess may read `.info/version` and toolchain
  files from the TheRock-owned root.

### Q085 — How is the build-time ROCm root selected?

**Decision: Accepted.**

Use one resolver with explicit context-sensitive precedence:

```cmake
if(HIPBLASLT_ENABLE_THEROCK)
    # Hermetic TheRock build: never use ambient ROCM_PATH.
    set(_build_rocm_root "${THEROCK_TOOLCHAIN_ROOT}")
elseif(ROCM_PATH)
    # Standalone explicit CMake selection.
    set(_build_rocm_root "${ROCM_PATH}")
elseif(DEFINED ENV{ROCM_PATH})
    # Standalone environment selection.
    set(_build_rocm_root "$ENV{ROCM_PATH}")
else()
    # Conventional standalone fallback.
    set(_build_rocm_root "/opt/rocm")
endif()
```

Use `_build_rocm_root` only as scoped `ROCM_PATH` for TensileLite release-wheel
construction, installation, and Python code-generation commands. Validate that
it contains readable `.info/version` and the required toolchain layout. Never
copy hipBLASLt outputs into it.

### Q086 — May a TheRock build fall back when `THEROCK_TOOLCHAIN_ROOT` is missing?

**Decision: Accepted — no.**

When `HIPBLASLT_ENABLE_THEROCK=ON`, missing or invalid
`THEROCK_TOOLCHAIN_ROOT` is a fatal configuration error. Do not fall back to
ambient `ROCM_PATH` or `/opt/rocm`, because that would defeat TheRock's
hermeticity guarantee.

### Q087 — Is the binding metadata format still open?

**Decision: Accepted — use the existing bare JSON absolute-path string.**

The settled format remains the existing JSON absolute-path string in installed
`.dist-info`. Client version is queried from the executable during configuration
and package import, so it is not duplicated in binding metadata.

### Q088 — Is `tensilelite-client` required on TheRock Windows builds?

**Decision: Accepted — yes.**

TheRock currently disables `TENSILELITE_ENABLE_CLIENT` on Windows. That must
change: the agreed canonical-wheel installation, client configuration, package
import, and Python generation flow requires a built client on every supported
platform. Add Windows build and validation coverage rather than maintaining a
different package contract.

### Q089 — Where is raw rocisa packaged on `develop`?

**Confirmed fact:** When TensileLite test artifacts are enabled on Linux,
hipBLASLt installs the following into the CMake `tests` component:

```text
share/hipblaslt/tensilelite/rocisa/       # Python package + _rocisa
share/hipblaslt/tensilelite/rocisa_tests/ # rocisa tests
```

It also co-installs the source-built stinkytofu shared library next to `_rocisa`
where applicable. TheRock's `blas_test` artifact includes the encompassing
`share/hipblaslt/tensilelite/**` tree.

Integrated source code generation separately imports `_rocisa` from the CMake
build tree through PYTHONPATH. Production hipBLASLt runtime dispatch does not
need the Python rocisa package.

**Decision: Accepted — preserve the raw rocisa test-artifact behavior.**

Do not move rocisa into a new wheel or production runtime component. Retain
build-tree PYTHONPATH use and the existing raw test-artifact install layout.

### Q090 — What is the governing scope rule?

**Decision: Accepted.**

Do not change existing `develop` behavior unless the canonical/compatibility
wheel and source-only test split requires it. Prefer restoring or retaining
known working build/test wiring over redesigning adjacent rocisa, TheRock, or
runtime packaging concerns.

### Q091 — How should the test runner install artifact wheels?

**Decision: Accepted — reuse the existing TheRock pattern.**

The analogous hipkernelprovider runner installs staged wheels with:

```python
[sys.executable, "-m", "pip", "install", "--no-deps", *wheels]
```

Use the same active-interpreter pip pattern for TensileLite canonical and later
compatibility phases. Do not introduce a component-specific uv invocation when a
working wheel-install precedent already exists.

Q103 assigns phase orchestration to the thin TensileLite runner and leaves
categories, markers, timeouts, workers, and JUnit execution with the generic
TheRock pytest runner.

### Q092 — What is the CMake target graph for wheel construction and generation?

**Decision: Accepted.**

```text
tensilelite-canonical-release-wheel
  → build and validate canonical wheel

tensilelite-compatibility-release-wheel
  → build and validate compatibility wheel

tensilelite-build-release-wheels
  → aggregate both release-wheel targets

_rocisa ──────────────────────────────┐
tensilelite-client ───────────────────┤
tensilelite-canonical-release-wheel ──┤
                                     ▼
tensilelite-python-build-environment
  → force-reinstall canonical wheel into found build Python
  → configure exact CMake-built client
                                     ▼
logic / create-library / ext-op generation
```

Compatibility wheel construction and tests remain outside the device-generation
dependency chain.

Canonical and compatibility wheels use separate exact output paths. Their
dependencies include package code/resources, build backend and metadata,
validator, requirements/build metadata, and `.info/version`. The configured
Python environment additionally depends on the exact client and raw rocisa
targets.

### Q093 — What do current defaults build?

**Confirmed facts:**

On upstream `develop` defaults:

- hipBLASLt host, device libraries, and hipBLASLt clients are enabled;
- `_rocisa` bundling is enabled;
- TensileLite Python runs directly from source/build paths;
- `tensilelite-client` is disabled unless TensileLite testing enables it;
- no TensileLite wheel is built;
- TensileLite test artifacts are disabled unless explicitly requested.

TheRock overrides these defaults for test builds on Linux by enabling
`TENSILELITE_ENABLE_CLIENT`, hipBLASLt testing, and TensileLite test-artifact
installation.

The feature implementation still ties release-wheel construction to the
test-artifact option. Q092 requires separating the canonical wheel target and
making it a device-generation dependency.

### Q094 — Is Windows Python artifact-test packaging in the current scope?

**Decision: Accepted — canonical production packaging is current scope; the
separate Windows artifact-test lane is deferred.**

Current scope enables and supports building and production-installing
`tensilelite-client` on Windows. Q099 also brings canonical-wheel construction,
release-content validation, installation, client binding, and code generation
into current Windows scope.

Only compatibility-wheel transfer/testing, raw `_rocisa.pyd` and DLL
test-artifact slicing, source-only test-artifact transfer, and the separate
Windows artifact-test job remain deferred to a focused follow-up.

### Q095 — What does “every device-generation build” cover?

**Clarification:** There should be one implementation in hipBLASLt CMake, but it
is reached through several entry points:

1. standalone raw CMake from `projects/hipblaslt`;
2. hipBLASLt Invoke wrappers that configure that CMake project;
3. checked-in CMake presets, especially default/full and `gemm-libs`;
4. TheRock's hipBLASLt subproject build;
5. rocm-libraries/superbuild integration;
6. CI variants such as sanitizer, coverage, and Linux/Windows builds.

The common rule is attached to `HIPBLASLT_ENABLE_DEVICE=ON`, not reimplemented
in every caller. Host-only configurations bypass the wheel/client/codegen graph.

### Q096 — Is the device-generation setup defined once or per entry point?

**Decision: Accepted — define it once.**

hipBLASLt CMake owns one centralized dependency rule:

```text
HIPBLASLT_ENABLE_DEVICE=ON
  → _rocisa
  → tensilelite-client
  → canonical release wheel
  → installed/configured build Python package
  → logic/create-library/ext-op generation
```

Raw CMake, Invoke, presets, TheRock, superbuilds, sanitizer builds, and platform
CI only select options and consume this graph. They must not replicate package
preparation or ordering logic.

### Q097 — Where does CMake install the canonical wheel for device generation?

**Decision: Accepted — install it into the found Python environment.**

Use `--force-reinstall --no-deps`, then immediately configure the exact
CMake-built client. One active hipBLASLt build owns the selected Python for the
duration of installation, client configuration, and device generation;
concurrent build directories sharing that interpreter are unsupported.

This matches the current disposable, single-build CI model while ensuring that
same-version source changes reinstall the current wheel.

### Q098 — Which ROCm artifact owns `tensilelite-client`?

**Decision: Accepted — `tensilelite-client` must be part of the ROCm production
distribution.**

Install the client as a production runtime component and include
`libexec/hipblaslt/tensilelite/tensilelite-client` in TheRock's production BLAS
runtime artifact. The test artifact must consume the production runtime artifact
through normal artifact composition/dependency rather than own a duplicate
install of the client. The canonical package requires the client, so the
production ROCm distribution must be complete without a test artifact.

### Q099 — Does the production client and canonical-wheel contract apply to Windows now?

**Decision: Accepted — yes.**

The current Windows scope includes:

- building `tensilelite-client`;
- installing it into the Windows ROCm production distribution;
- constructing the canonical TensileLite wheel;
- applying the same release-content validation as Linux;
- force-installing the canonical wheel into the selected build Python;
- binding and validating the exact CMake-built Windows client; and
- using that installed package for device-library code generation.

The canonical wheel is the device-generation input on every supported
platform, so Windows must implement the same production package contract as
Linux.

Deferred Windows follow-up scope is limited to:

- compatibility-wheel artifact transfer and compatibility-only tests;
- raw `_rocisa.pyd` and dependent-DLL test-artifact slicing;
- source-only test-artifact transfer; and
- execution in a separate reconstructed-artifact test job.

### Q100 — How do artifact tests run against the installed canonical wheel?

**Decision: Accepted — run the separately copied raw tests with pytest's
default `prepend` import mode.**

Install the canonical wheel into the artifact-test venv and keep existing test
helpers, inherited bases, and harnesses as local test modules. Rewrite the
eleven `tensilelite.Tests.*` statements as test-local imports because tests are
not part of the production wheel.

Run the reconstructed artifact suite with the checkout absent. Fix installed
package/resource assumptions that fail in that environment, and keep genuine
source-layout or package-construction tests in source CI.

This preserves the mature suite's current module structure while proving that
production imports come from the installed canonical wheel.

### Q101 — How is the canonical wheel's rocisa dependency satisfied before rocisa packaging exists?

**Decision: Accepted — classify the canonical wheel as a controlled ROCm
artifact for current scope.**

Current build and artifact test jobs install the canonical wheel with:

```bash
python -m pip install --force-reinstall --no-deps <canonical-wheel>
```

They supply the raw rocisa package and native extension through the scoped
`PYTHONPATH` already required by the CMake/test environment. This makes
`import rocisa` work but does not create installed rocisa distribution
metadata. Consequently, `pip check` will report that the declared rocisa
distribution is missing; this warning is an accepted temporary limitation.

Keep the accurate dependency metadata. Proper rocisa distribution/wheel
packaging is a follow-up that will remove the raw-package `PYTHONPATH`
workaround and make ordinary dependency checks pass.

### Q102 — Is `tensilelite-client` the intended long-term native generation seam?

**Confirmed current fact:** `tensilelite-client` is currently needed for
benchmarking and validation, not for kernel generation. Current device-library
generation remains in the Python/rocisa/toolchain path.

**Decision: Accepted — wire the client as the intended long-term native
generation seam.**

Future kernel-generation functionality is expected to move into the native
client, with Python becoming a convenience wrapper. Establish the client
dependency, production ownership, exact binding, cross-platform build, and CI
plumbing now so native generation can use the same package/runtime contract.

The stable generation command/protocol, its inputs and outputs, capability
negotiation, and error contract are future implementation work. When native
generation is added, define a separate protocol/capability version rather than
treating exact distribution-version equality as an ABI or protocol guarantee.

### Q103 — Which runner owns TensileLite's canonical and compatibility test phases?

**Decision: Accepted — use a thin TensileLite-specific phase runner which
delegates pytest execution to TheRock's generic runner.**

The TensileLite runner owns component-specific package orchestration:

- discover exactly one canonical wheel with the expected version;
- fail on zero or ambiguous matches;
- install it into the active test venv with
  `--force-reinstall --no-deps`;
- configure the reconstructed `ROCM_PATH` and raw-rocisa `PYTHONPATH`;
- invoke the generic runner for the canonical test phase;
- install the compatibility wheel only after the canonical phase passes;
- invoke the generic runner for `compat/tests` with `--run-compat`; and
- return failure when either executed phase fails.

The generic TheRock pytest runner continues to own:

- category and test-path selection;
- marker expressions;
- timeouts and worker counts;
- pytest invocation; and
- JUnit generation.

Canonical and compatibility phases write separate JUnit results. If the
canonical phase fails, preserve its JUnit output, do not install the
compatibility wheel, and return failure immediately. A compatibility failure
preserves its own JUnit output and returns failure.

The compatibility phase and its inputs are removed when the compatibility
package is retired.

### Q104 — What is the single authority for the TensileLite component version?

**Decision: Accepted — use a checked-in `VERSION` file at the TensileLite
project root.**

`VERSION` contains the component release version, for example:

```text
5.0.0
```

The base ROCm compatibility version remains independently authoritative in the
selected SDK's `.info/version`. `release_metadata.py` is the sole composition
and validation implementation for producing:

```text
<component-version>+rocm<base-rocm-version>
```

The exact composed value feeds:

- canonical wheel metadata;
- compatibility wheel metadata and its exact canonical dependency pin;
- a generated C++ header compiled into `tensilelite-client`; and
- release-wheel and client-version validation.

Remove component-version literals from canonical and compatibility setup code.
The native client embeds the generated build-time value and never derives its
identity from the runtime `ROCM_PATH`.

`GENERATOR_VERSION` remains a generator/logic compatibility concept. Any
decision to couple its lifecycle to the component release version must be
explicit rather than an accidental consequence of package-version plumbing.

### Q105 — Should the build backend rewrite wheels to embed a client binding?

**Decision: Accepted — no.**

Every wheel archive remains unbound. Source, editable, and prebuilt workflows
install the wheel first and then use `tensilelite-configure-client` when they
need a custom executable. The build backend no longer adds binding metadata or
rewrites wheel `RECORD` files.

This makes the binding an installation-local concern, uses one persistence
path for all installation modes, and preserves wheel archive contents and any
wheel signatures. It also removes the signed-wheel rewriting decision.

### Q106 — Does a multi-config build require configuration-specific Python package state?

**Decision: Accepted — no.**

The canonical wheel, installed Python package, and Python/native compatibility
contract are configuration-independent. Debug and Release clients implement the
same versioned interface; CMake build type is not part of the Python
compatibility identity.

Sequential multi-config builds are supported. Before a Python generator command
runs, the config-aware CMake wiring refreshes the installation-local binding to
the selected config's exact client when necessary and supplies the active
config's raw rocisa path. This does not require separate wheel or Python package
roots for Debug and Release.

Concurrent configurations that mutate the same selected Python remain outside
the single-owner contract in Q097. Validate a sequential Debug-to-Release build
to prove binding refresh and generator behavior.

### Q107 — How do standard and custom client install directories resolve?

**Decision: Accepted — standard ROCm lookup uses the fixed production layout;
custom layouts use an explicit installation-local binding.**

Production ROCm/TheRock artifacts place the client at:

```text
libexec/hipblaslt/tensilelite/tensilelite-client
```

The canonical wheel's standard resolver checks only that location below
`ROCM_PATH`. CMake may continue honoring a nondefault
`CMAKE_INSTALL_LIBEXECDIR` for custom/local installations; those installations
select their exact client with `tensilelite-configure-client`.

The runtime does not search alternate libexec directory names, and wheel
metadata does not encode a build-specific install layout. This preserves the
standard production contract and wheel transferability without adding code for
a nonstandard production layout that is not currently required.

### Q108 — What is the standalone Windows ROCm-root fallback?

**Decision: Accepted — standalone Windows requires an explicit SDK root.**

ROCm-root selection is:

```text
TheRock                         -> require THEROCK_TOOLCHAIN_ROOT
standalone with -DROCM_PATH     -> use the CMake value
standalone with env ROCM_PATH   -> use the environment value
standalone Linux with neither   -> fall back to /opt/rocm
standalone Windows with neither -> fail during configuration
```

The Windows diagnostic instructs users to pass `-DROCM_PATH=<SDK root>` or set
the `ROCM_PATH` environment variable. Invoke may continue discovering the
Windows SDK through `rocm-sdk` and passing the resolved root to CMake; raw CMake
does not add another implicit discovery mechanism.

### Q109 — Which targets require Python development headers?

**Decision: Accepted — derive `Development.Module` from native Python-extension
targets, not from the client executable option.**

The current `_rocisa` nanobind extension requires the Python interpreter and
`Development.Module`. The current `tensilelite-client` executable requires an
interpreter for version/package tooling but does not compile against Python and
therefore does not require development headers by itself.

The long-term native generation architecture is expected to contain a reusable
native generator library used by both `tensilelite-client` and a nanobind Python
extension. When that binding target is introduced, it becomes an explicit owner
of `Development.Module`.

The CMake rule follows actual targets:

```text
build _rocisa or future native nanobind module -> Interpreter + Development.Module
run Python tooling/client-only build           -> Interpreter only
true host-only build                           -> no TensileLite Python requirement
```

This preserves the intended native-generation direction without making the
current standalone executable require headers it does not consume.

### Q110 — Is rocisa coverage-import provenance part of this packaging change?

**Decision: Accepted — no; leave existing coverage behavior unchanged.**

The packaging work supplies raw rocisa through the scoped build/test environment
defined elsewhere in this record, but it does not add coverage-specific import
path assertions, per-object coverage gates, or coverage workflow redesign.
Coverage provenance can be addressed independently if it becomes a demonstrated
coverage problem.

### Q111 — What documentation is in scope for the packaging change?

**Decision: Accepted — update only documentation for interfaces and workflows
changed by this design.**

Documentation must reflect the canonical-wheel build/install flow, controlled
ROCm artifact status and `--no-deps` use, raw rocisa environment, unbound wheels,
post-install client configuration, standard production client path, local
editable workflow, removal of `BUILD|SYSTEM` and the private CMake venv, removal
of the pip client-path setting, and the Python 3.10 package requirement.

Unrelated pre-existing contributor-documentation cleanup is outside this
change.

### Q112 — What focused validation is required for the packaging change?

**Decision: Accepted — update focused package/native tests and rely on existing
affected CI for integration coverage.**

Required focused tests cover:

- `VERSION` plus `.info/version` composition and equality across canonical
  metadata, compatibility metadata/pin, and the generated client version;
- post-install configure/reset, installed RECORD updates, mismatched-client
  rejection, and proof that the original wheel archive remains unchanged;
- configured-versus-standard runtime resolution, no fallback, client
  `--version` success/failure/malformed handling, and process-local freezing;
- canonical/compatibility wheel version, pin, entry-point, resource, and unbound
  content validation;
- thin-runner phase ordering and failure propagation; and
- GPU-less native `--version` stdout, stderr, and exit-status behavior.

Existing Linux/Windows TheRock, artifact-test, sanitizer, and superbuild lanes
provide integrated coverage. This work does not add a separate comprehensive
CMake matrix or coverage workflow.

### Q113 — How should the `hipblaslt-clients` preset satisfy the device-build invariant?

**Decision: Accepted — remove its explicit
`TENSILELITE_ENABLE_CLIENT=OFF` override.**

The preset enables hipBLASLt device generation, so the centralized default must
enable `tensilelite-client` and the complete Python/native generation graph.
Removing the override keeps the invariant in one place rather than duplicating
an explicit `ON` value in the preset. Add a focused configure test proving the
preset no longer selects a contradictory option combination.

## Confirmed TheRock build/test facts

### Build

- hipBLASLt is configured and built under its own TheRock build directory.
- `CMAKE_INSTALL_PREFIX` points at a per-subproject stage, not `/opt/rocm`.
- TheRock builds `therock-artifacts` and a merged `therock-dist` tree.
- Ambient `ROCM_PATH` is deliberately cleared for subprojects.

### Artifact transfer

- Build artifacts are compressed and uploaded under the workflow run ID.
- Test jobs download and flatten selected artifacts into `./build`.
- The build-job Python environment is not transferred.

### Test runtime

- The test job creates a fresh venv and installs `requirements-test.txt`.
- It sets `ROCM_PATH=./build` and matching `PATH`/`LD_LIBRARY_PATH` values.
- `blas_lib` currently carries hipBLASLt runtime/device libraries but not
  `tensilelite-client`; Q098 requires adding the client.
- `blas_test` currently carries test binaries/data, Python test artifacts, and
  `tensilelite-client`; Q098 requires removing duplicate client ownership and
  receiving it through the production runtime artifact.

### Current test-runner mismatch

The current active pytest runner injects the artifact test root into
`PYTHONPATH` and expects the old `Tensile` layout. Q100/Q103 require the thin
runner to install the canonical wheel, reserve `PYTHONPATH` for raw rocisa, and
let pytest's default `prepend` mode expose the separately copied tests.

## Accepted implementation constraints

- hipBLASLt CMake has one source-build mode; an installed-package consumer mode
  returns only with a concrete consumer.
- CMake uses the selected writable Python directly. That environment is
  single-owner during installation, binding, and generation, and canonical
  installation uses `--force-reinstall --no-deps`.
- Workflows provision runtime requirements and build tools from the project
  metadata; CMake relies on authoritative pip and real generator commands for
  validation rather than duplicate import checks.
- Device-generation commands depend explicitly on `_rocisa`, the production
  client, the canonical wheel, and the configured Python environment.
- Raw rocisa remains a scoped build/test artifact on `PYTHONPATH`; proper rocisa
  distribution packaging is follow-up work.
- The canonical wheel is a controlled ROCm artifact until that rocisa packaging
  exists; its temporary `pip check` missing-distribution diagnostic is accepted.
- The checked-in `VERSION` file is the component-version authority;
  `release_metadata.py` combines it with the selected SDK's `.info/version` for
  wheel metadata and the generated native-client header.
- `tensilelite-client` is production-owned on Linux and Windows and is wired as
  the intended long-term native generation seam.
- Every wheel archive remains unbound; custom client metadata is written only
  into an installed distribution by `tensilelite-configure-client`.
- Python package state is configuration-independent. Sequential multi-config
  builds refresh the selected config's client binding before generation;
  concurrent configs sharing one Python are unsupported.
- Standard ROCm artifacts use `libexec/hipblaslt/tensilelite`; custom
  `CMAKE_INSTALL_LIBEXECDIR` layouts use an explicit post-install client binding.
- Standalone Windows device builds require an explicit ROCm SDK root; only
  standalone non-Windows builds may fall back to `/opt/rocm`.
- Python `Development.Module` is required only by native Python-extension
  targets such as `_rocisa` and the future native generator binding; the current
  client executable alone requires only an interpreter.
- Artifact tests install the canonical wheel, use pytest's default `prepend`
  mode for separately copied tests, and classify only reproduced checkout-only
  assumptions as source-CI-only.
- The thin TensileLite runner owns wheel installation and phase ordering; the
  generic runner owns pytest execution and JUnit. Canonical tests must pass
  before compatibility is installed or run.

## Deferred decisions

1. Source/nightly/prerelease SHA-bearing package version grammar and workflow
   channel propagation.
2. Exact source-build identity beyond semantic `5.0.0+rocmA.B.C` compatibility.
3. Proper rocisa distribution/wheel packaging, replacing raw-package
   `PYTHONPATH` injection and making canonical-wheel dependency checks clean.

## Next questions

Q097–Q106 resolve the renewed design choices and consistency findings from the
staff review. No current-scope design decisions remain open.

Confirm shared understanding of the complete accepted design before
implementation.

Deferred follow-ups remain source-SHA/channel version naming, exact same-version
source-build identity, and proper rocisa distribution packaging.
