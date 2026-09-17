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
packaging (Q083), it is a controlled ROCm artifact rather than an independently
pip-installable distribution.

## Current high-level direction

The current source-build flow is:

```text
CI/workflow installs tensilelite/requirements.txt into the selected Python
                              │
                              ▼
                         CMake configure
                              │
                     build CMake _rocisa
                              │
                              ▼
       expose only build-tree rocisa through PYTHONPATH
                              │
                              ▼
       force-reinstall canonical wheel into the found Python
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
package raw rocisa + _rocisa and copied installed-wheel tests
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

The optional client is packaged separately for benchmark/retune use. It is not
a prerequisite for wheel installation, import, or device-library generation.

## Chronological question and decision log

### Q001 — Must raw CMake prepare the source-development Python environment?

**Question:** Must a TheRock build work through CMake without a preceding
hipBLASLt-specific editable install?

**Decision: Accepted — yes.**

TheRock invokes hipBLASLt as a CMake subproject. The build graph must own the
native/Python preparation ordering rather than require an undocumented
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

### Q004 — How are rocisa and `_rocisa` structured today?

**Confirmed fact:**

- `rocisa` is the Python package/facade.
- `_rocisa` is the native nanobind extension.
- Integrated `develop` builds `_rocisa` as a CMake target and exposes the build
  package through `PYTHONPATH`.
- Standalone `pip install -e rocisa` invokes a separate scikit-build-core CMake
  build.

### Q005 — Should the TheRock source build build `_rocisa` itself?

**Decision: Accepted — yes.**

A CMake-only TheRock source build owns the `_rocisa` prerequisite for
device-library generation. It must not assume rocisa is preinstalled in the
selected Python.

### Q006 — What rocisa wiring must be restored?

**Decision: Accepted — derive rocisa wiring from the build mode.**

`HIPBLASLT_ENABLE_DEVICE=ON` requires:

- `_rocisa` as a code-generation dependency;
- the rocisa build-package parent on `PYTHONPATH`;
- CMake dependency ordering that builds `_rocisa` before Python code generation.

Standalone rocisa and coverage workflows may request `_rocisa` independently.
A true host-only build (`HIPBLASLT_ENABLE_DEVICE=OFF` and client disabled) does
not build it. Proper rocisa release packaging remains follow-up work.

### Q007 — Must rocisa be installed as a pip distribution for source builds?

**Decision: Accepted — no.**

`import rocisa` is sufficient for the current source build. The CMake-built raw
package and extension may remain build-tree artifacts exposed through
`PYTHONPATH`.

### Q008 — Should rocisa path injection remain temporary technical debt?

**Decision: Accepted — yes.**

No proper rocisa wheel/install integration is required now. Only rocisa is
injected; TensileLite itself is installed from the canonical wheel.

### Q009 — Should CMake create a private Python venv?

**Question:** Why not install into the Python found by CMake?

**Decision: Accepted — do not require a private venv.**

The relevant CI builds run in disposable containers. CMake may install into the
found Python provided it is writable. The selected interpreter is a
single-owner build resource: concurrent hipBLASLt build directories must not
install or configure TensileLite in the same Python environment.

### Q010 — Must the found Python be a virtual environment?

**Decision: Accepted — no.**

The requirement is only that the selected Python environment is writable.

### Q011 — Should CMake probe writability during configuration?

**Decision: Accepted — no.**

Let the authoritative pip operation fail if the environment is not writable.
Do not add a separate imperfect writability probe.

### Q012 — How should TensileLite be installed for source code generation?

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

TheRock builds the canonical ROCm artifact wheel and installs that unchanged
wheel into the build-job Python.

The installation command must use:

```bash
python -m pip install --force-reinstall --no-deps <exact-canonical-wheel>
```

The force reinstall is required because source changes do not necessarily
change the wheel version.

### Q014A — Should TheRock use the canonical wheel for build-time generation?

**Decision: Accepted — yes.**

This exercises the same production package code/resources that will be shipped.
CMake dependencies must rebuild and reinstall the wheel when relevant package
sources or metadata change.

### Q013 — Should pip install runtime dependencies implicitly?

**Decision: Accepted — no.**

Use `--no-deps` to prevent network access, upgrades, and hidden package
resolution during the CMake build.

### Q014 — Where do pure-Python dependencies come from?

**Confirmed fact:**

- Standalone developers currently install
  `projects/hipblaslt/tensilelite/requirements.txt` manually.
- The rocm-libraries TheRock wrapper installs only `TheRock/requirements.txt`.
- Native TheRock staged workflows support per-artifact `python_requires` via
  `BUILD_TOPOLOGY.toml` and `configure_stage.py`.

### Q015 — Should dependencies be copied into `TheRock/requirements.txt`?

**Decision: Accepted — add no new root duplication and do not modify the
existing compatibility entries in this change.**

TensileLite owns its requirements file for staged/native and rocm-libraries
wrapper provisioning. Leave TheRock root `requirements.txt` completely
untouched: its existing `joblib`/`msgpack` entries remain a compatibility bridge
for the documented non-staged local build, which does not consume artifact
`python_requires`. Removing those entries requires a separate local-provisioning
migration. Likewise, leave `requirements-test.txt` unchanged until its fresh
test-venv consumer installs an artifact-owned equivalent.

### Q016 — Where should workflows install TensileLite requirements?

**Decision: Accepted.**

Use both integration mechanisms:

1. Native TheRock staged builds: add the TensileLite requirements file to the
   `blas` artifact's `python_requires` in `BUILD_TOPOLOGY.toml`. Extend
   `configure_stage.py` with one explicit rocm-libraries source-root input,
   defaulting to TheRock's `rocm-libraries` submodule. Resolve the project-owned
   requirements path from that root and emit the matching
   `-DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=<root>` CMake argument from the same
   selection. Do not hard-code the default submodule path independently of the
   CMake source selection or require callers to specify the root twice.
2. rocm-libraries wrapper workflows: explicitly run:

   ```bash
   pip install -r projects/hipblaslt/tensilelite/requirements.txt
   ```

Validate the native mechanism with the default submodule, the conventional
external `rocm-libraries` checkout, and an arbitrary absolute local checkout.
Fail before pip or CMake when the selected requirements file is missing, with
the resolved path in the diagnostic.

### Q017 — Which workflows must provision the requirements?

**Decision: Accepted — every workflow that builds hipBLASLt device libraries.**

This includes Linux, Windows, sanitizer, and other build workflows when they
enable the relevant hipBLASLt device-generation path. Unrelated lint-only flows
need not install them unless they configure that path.

### Q018 — Should CMake duplicate a dependency import checklist?

**Decision: Accepted — no.**

`requirements.txt` is the runtime dependency source of truth, and
`pyproject.toml` `[build-system]` is the build-backend requirement source of
truth. Because wheel commands use `--no-build-isolation`, workflows must provide
pip and those build requirements in the selected Python. Do not hard-code a
second import-name or version checklist in CMake; the authoritative pip
wheel/install command reports missing build tools.

### Q019 — Should the environment target run an explicit final import check?

**Decision: Accepted — no.**

The first real `python -m tensilelite logic` or `create-library` command imports
the generator modules and exposes any missing rocisa or toolchain dependency.
This matches the simpler `develop` behavior.

### Q020 — Is failure at the first real command acceptable for missing runtime dependencies?

**Decision: Accepted — yes.**

Workflows own dependency installation. The first consumer command is the
authoritative validation.

### Q021 — What should `PYTHONPATH` contain?

**Decision: Accepted.**

For source builds, `PYTHONPATH` contains only the parent of the CMake-built raw
rocisa package. TensileLite comes from the canonical wheel force-installed into
the selected Python environment; its installation directory is not added to
`PYTHONPATH` explicitly.

### Q022 — Should ordering depend on command/source order?

**Decision: Accepted — no.**

CMake must encode ordering as target dependencies:

```text
_rocisa ─────────────────> TensileLite Python environment target
                                           │
                                           ├──> logic validation
                                           ├──> create-library
                                           └──> ext-op generation
```

### Q023 — Should there be a single CMake helper for Python generator commands?

**Decision: Accepted — yes.**

The helper must attach both the configured command environment and the required
CMake target dependencies so callers cannot forget one half.

### Q024 — Should hipBLASLt gtest-data generation use the TensileLite helper?

**Decision: Accepted — no.**

`hipblaslt_gentest.py` only converts YAML to test data. It should use
`Python3_EXECUTABLE` directly and remain outside the rocisa/TensileLite
generation graph.

### Q025 — Does TheRock install builds into `/opt/rocm`?

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

### Q026 — How do build artifacts reach tests?

**Confirmed fact:**

- Build and test are separate jobs/containers.
- Build artifacts are uploaded under the workflow run ID.
- The test job creates a new venv.
- Artifacts are downloaded and flattened into `./build`.
- The test job treats `./build` as the ROCm prefix.

Therefore, build-job Python/site-packages state does not survive into tests.

### Q027 — Is rocisa currently a release wheel in this flow?

**Confirmed fact — no.**

rocisa can technically build a wheel through scikit-build-core, but ROCm/TheRock
release packaging for that wheel is deferred. Existing integrated/test flows
use a raw package plus native extension through `PYTHONPATH`.

### Q028 — Should test jobs use a rocisa wheel now?

**Decision: Accepted — no.**

Continue injecting the raw rocisa package from test artifacts. rocisa wheel
packaging remains future work.

### Q029 — Should tests ship inside the production wheel?

**Decision: Accepted — no.**

Production wheels contain production package code/resources. Tests and their
configuration ship only in the `blas_test` artifact.

### Q030 — Where should wheels be installed in the test job?

**Decision: Accepted.**

The TensileLite-specific `pytest_runner.py` should install local artifact wheels
into the already-created test venv before invoking pytest. Do not burden the
generic test-environment action with component-specific installation.

Q084 clarifies that this is a thin TensileLite-owned phase runner which
delegates ordinary pytest execution to TheRock's generic runner.

### Q031 — Should the compatibility wheel be installed by the main tests?

**Decision: Accepted — install and test it while it remains supported.**

The file and test flow must explicitly note its near-term removal.

### Q032 — How do canonical and compatibility tests avoid masking each other?

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

### Q033 — Where do compatibility tests live?

**Decision: Accepted.**

All compatibility tests live under `compat/tests/`, separate from the canonical
TensileLite test tree.

### Q034 — How are compatibility tests selected?

**Decision: Accepted.**

- Compatibility tests are skipped by default.
- Register an explicit flag, recommended spelling `--run-compat`.
- Compatibility CI always passes that flag.
- Tests use a `compat` marker or equivalent collection hook to enforce default
  skipping.

### Q035 — What compatibility coverage is required?

**Decision: Accepted — cover every legacy entry point.**

Tests must verify:

- each expected console-script name exists in installed package metadata;
- the exact delegated target function;
- exact argument ordering and values;
- return-code propagation;
- deprecation warning behavior.

Expensive underlying commands may be mocked, but argument forwarding must be
asserted exactly.

Remove the compatibility wheel's `pandas` dependency rather than provisioning
it in artifact CI. `GenerateSummations.py` is the only production pandas user
and uses it only to read `benchmark.csv`, strip headers, select `SizeL`/`Cij` and
kernel columns, and compute a maximum. Replace that with standard-library
`csv.DictReader` plus the already-required NumPy arrays/`nanmax`/`polyfit`, while
preserving first-seen `SizeL` order, numeric conversion, quoting, whitespace,
and NaN behavior. Remove the test-only pandas module mock and add a focused CSV
fixture proving the parsed vectors, maximum, and fitted model. Compatibility CI
can then import the real delegated module while still mocking expensive
underlying execution for forwarding assertions.

### Q036 — Where does wheel validation occur today?

**Confirmed fact:**

The CMake wheel targets build canonical and compatibility wheels, then run
`scripts/check_release_wheel_contents.py` before staging the validated wheel
files into test artifacts. The validator verifies canonical and compatibility
versions, the compatibility wheel's exact canonical dependency pin, expected
console scripts and wheel tags, absence of custom client binding metadata, and
required canonical resources.

### Q037 — Are custom client bindings universally forbidden in wheels?

**Decision: Accepted — yes. All wheel archives remain unbound.**

Client bindings belong to one user's exact resolved TensileLite installation
directory and are created only by the post-install
`tensilelite-configure-client` command. Canonical,
compatibility, local, non-editable, and editable wheel archives contain no
machine-local client metadata.

### Q038 — How should the wheel-build target be named?

**Decision: Accepted.**

Use a name that clearly says it builds release wheels:

```text
tensilelite-build-release-wheels
```

Recommended output directory:

```text
tensilelite-release-wheels/
```

### Q039 — Does the compatibility wheel remain in the release-wheel target?

**Decision: Accepted — yes.**

The same target builds canonical and compatibility release wheels until the
compatibility package is removed.

### Q040 — Can a synthetic client-only `ROCM_PATH` work for a conventional prefix?

**Confirmed fact — not by itself.**

The conventional-prefix adapter uses `ROCM_PATH` for:

- `.info/version` release metadata;
- compiler/assembler/bundler/readelf discovery;
- Windows `amdclang++ --rocm-path`, which requires headers and libraries.

A synthetic root must be a complete usable SDK view, not merely a client
directory. This does not apply to the separate Python-SDK adapter, which does
not resolve a package root.

### Q041 — Can an existing wheel receive a client binding at pip-install time?

**Confirmed fact:** Standard `pip install existing.whl` does not invoke the
wheel's PEP 517 build backend and cannot consume the source-build
`--config-settings` hook. Configure an already-built wheel with the explicit
post-install `tensilelite-configure-client` command.

### Q042 — Does the current version check validate a PATH client?

**Confirmed fact — no.**

PATH discovery is not part of the accepted resolver because it cannot identify
or validate the intended client reliably. Q043 defines the complete precedence.

### Q043 — What precedence should client discovery use?

**Decision: Accepted.**

Final contract:

1. If keyed per-user binding metadata exists, use only its exact client and
   never fall back.
2. For a Python SDK package installation, the interim implementation reports
   that the client is unavailable: `rocm-sdk-libraries` does not yet ship the
   native client. The final state uses that package's exact
   `tensilelite-client` console script from the active Python environment.
   That trampoline owns resolution of the library payload; TensileLite does
   not inspect a core or library payload directory.
3. For a conventional-prefix installation, resolve only the standard client
   under the selected `ROCM_PATH`-style root.
4. Do not perform a broad client search on PATH. The interpreter-local console
   script in item 2 is an explicit package entry point, not PATH discovery.

### Q044 — Does TheRock have a usable ROCm root before hipBLASLt code generation?

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

This establishes the conventional-prefix base value. TheRock separately passes
its full package identity to hipBLASLt for wheel metadata when it is available.
The build-time root is used only by the graph-owned wheel and code-generation
flow; client capability is independent of that flow.

### Q045 — What does the final `.info/version` contain?

**Confirmed fact:** It contains base `A.B.C`, not the full dev/nightly/RC
`THEROCK_PACKAGE_VERSION`. That base value is the conventional-prefix
compatibility identity; it does not determine a TheRock package wheel's full
publication identity.

Current TheRock full wheel-version examples are:

```text
CI/dev:      10.1.0.dev0+<full-git-sha>
nightly:     10.1.0a20260806
prerelease:  10.1.0rcN
```

The suffix is package-publication identity, while `.info/version` currently
records only ROCm compatibility identity (`10.1.0`).

### Q046 — Is a post-install `configure-client` mechanism viable?

**Decision: Accepted.**

Configure the installed environment after pip completes:

```bash
python -m tensilelite_configure_client \
  --client /absolute/path/to/tensilelite-client
```

The command is importable outside normal `tensilelite` initialization, validates
the client, stores the keyed per-user binding, and leaves the original
wheel unchanged.

The binding overrides only the client executable. The caller still needs a
matching ROCm installation selected through the Python-SDK or
conventional-prefix adapter.

### Q047 — Can the native client report its own compatibility identity?

**Decision: Accepted and simplified by the client-resolution decisions below.**

Add a no-GPU command:

```text
tensilelite-client --version
```

It prints the installed TensileLite distribution version. Configuration and the
first explicit client request validate that value against the installed Python
distribution version. Source revision identity is deferred.

### Q048 — Can the client path be passed directly to pip install?

**Decision: Accepted — do not pass client bindings through pip.**

Pip installs an unbound wheel for every source, editable, and prebuilt workflow.
When a custom client is required, run `tensilelite-configure-client` immediately
after installation.

### Q049 — Do all installations share one binding implementation?

**Decision: Accepted — yes.**

Use one client-binding module for:

- absolute path and executable validation;
- native client identity/version query;
- Python/client compatibility checks;
- binding metadata schema and parsing;
- installation-key and per-user binding-path construction;
- runtime binding validation.

Use two thin adapters:

1. `tensilelite-configure-client`: atomically writes the binding into the
   current user's slot for the exact resolved installation directory.
2. Runtime resolver: reads the binding or resolves the standard client without
   performing installed-state mutation.

The shared module must be importable without normal TensileLite package runtime
initialization so configuration can manage an installed binding independently.

### Q050 — What must change when this becomes the only client override mechanism?

**Decision: Accepted — use one pre-runtime binding policy with thin adapters.**

The final contract is:

- keep the binding metadata as a bare JSON absolute-path string at
  `~/.tensilelite/bindings/<installation-id>/client.json`;
- derive `<installation-id>` only from the exact resolved installed package
  directory; configuring a reinstall or upgrade at the same directory
  intentionally atomically replaces the existing file and is not a collision;
- share path validation, version-command policy, version comparison, metadata
  parsing, and installation-key construction in a pre-runtime module;
- use thin adapters for per-user configuration and runtime resolution;
- make runtime use configured binding exclusively when present, with no fallback
  to `ROCM_PATH`, PATH, YAML, globals, or command-line overrides;
- use the selected installation adapter's standard client candidate only when
  no binding exists;
- validate both configured and standard clients with plain `--version`;
- keep configuration importable before normal package initialization;
- use one atomic replacement for configure and one file deletion for reset;
- treat concurrent configure/reset of the same installation as unsupported,
  without locks, tombstones, or a recovery state machine;
- intentionally leave per-user configuration outside pip uninstall ownership;
- rename the existing helper-kernel cache root from
  `~/.tensile/helper_cache` to the single shared per-user
  `~/.tensilelite/helper_cache`; never place it below an installation-keyed
  binding directory, because its existing content-derived keys safely share
  compatible entries across wheels, venvs, and worktrees; do not automatically
  migrate the disposable old cache; and
- keep every wheel archive free of machine-local client bindings.

### Q051 — How do local source installs configure a custom client?

**Decision: Accepted — install first, then configure.**

```bash
python -m pip install --editable /path/to/tensilelite \
  --no-deps --no-build-isolation
python -m tensilelite_configure_client \
  --client /absolute/path/to/tensilelite-client
```

`invoke install` may perform both steps to preserve a one-command developer
workflow.

### Q052 — Can a configured binding be removed?

**Decision: Accepted — yes.**

Provide:

```bash
tensilelite-configure-client --reset
```

Reset deletes the current installation's one keyed per-user `client.json`.
Subsequent client requests use the selected installation adapter's standard
candidate. It does not mutate the installed distribution or another
installation's binding.

### Q053 — Does build-time code generation depend on the compatibility wheel?

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

### Q054 — Does full `THEROCK_PACKAGE_VERSION` become `.info/version`?

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

For a conventional prefix, a TensileLite wheel compares the base value in
`.info/version`. For an active Python SDK, it compares the full
`rocm_sdk_core.__version__` identity.

### Q055 — Should non-release source builds include the TensileLite git revision?

**Decision: Deferred by Q058.**

Current scope does not add an independent TensileLite source revision to the
package version. A later design must use a PEP 440-valid value and the
rocm-libraries/TensileLite source revision rather than the TheRock revision.

### Q056 — Why is the source SHA after `+`?

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

### Q057 — Can a PEP 440 version contain multiple `+` separators?

**Confirmed fact — no.**

PEP 440 permits one local-version separator. Additional identity fields must be
dot-separated inside the single local segment:

```text
5.0.0.dev0+rocm10.1.0.g8f3412abcd12
```

Parsing should use `packaging.version.Version` and a strict local-segment grammar
rather than splitting the raw string on multiple plus signs.

### Q058 — Is source-SHA/channel versioning in the current implementation scope?

**Decision: Accepted — defer only source-revision identity.**

TheRock package channels are already propagated through
`THEROCK_PACKAGE_VERSION` and the wheel metadata helper. A future policy may add
an independent TensileLite source revision, but it must retain a PEP 440-valid
single local-version segment.

### Q059 — Is native client identity validation in scope now?

**Decision: Accepted — yes.**

Add a no-GPU machine-readable client identity command and validate custom client
bindings and the standard ROCm-relative client through the shared binding
infrastructure.

Exact stale-source detection remains dependent on Q060: semantic versions alone
cannot distinguish two different source builds that both report component
version `5.0.0`.

### Q060 — Is exact source build identity required outside the package version?

**Decision: Accepted — semantic version only for current scope.**

Validate only the existing semantic TensileLite/generator version. Exact source
commit identity and same-version stale-build detection are deferred future work.

### Q061 — Should the native client also report its build-time ROCm version?

**Decision: Accepted — report the canonical combined distribution version.**

The client prints the same canonical distribution version as its wheel through
the plain `--version` contract in Q062. Configuration and the first client
request compare it exactly with the installed Python distribution version. For
TheRock package builds that version may carry a nightly, RC, or development
ROCm publication suffix; standalone conventional-prefix builds use the base
compatibility value.

### Q062 — Does client version output need JSON?

**Decision: Accepted — no.**

The current scope has one canonical compatibility value, so the native interface
is simply:

```bash
tensilelite-client --version
```

with stdout equal to the installed TensileLite distribution version, for
example:

```text
<tensilelite-distribution-version>
```

The configuration tool trims and parses that string with Python packaging
version utilities, then compares it exactly with the installed TensileLite
distribution version. A structured format can be introduced later only if the
client must report additional independent fields.

### Q063 — When is a custom client's version validated?

**Decision: Accepted — twice.**

1. `tensilelite-configure-client` validates before writing the binding.
2. Runtime validates again on first client resolution in each Python process,
   then caches the successful result for that process.

This detects an incompatible executable that later replaces the originally
configured file at the same path without paying the subprocess cost on every
client access.

### Q064 — Is the standard ROCm-relative client version also validated?

**Decision: Accepted — yes.**

Do not rely solely on artifact-set provenance. A Python TensileLite installation
can remain while `/opt/rocm` is upgraded or replaced underneath it.

For both configured and standard clients, validate:

- file exists and is executable;
- `tensilelite-client --version` exactly matches the installed Python
  TensileLite distribution version;
- the Python distribution's ROCm compatibility tag matches
  `$ROCM_PATH/.info/version` when a standard ROCm root is used.

### Q065 — What is the native `--version` command contract?

**Decision: Accepted.**

`tensilelite-client --version` must:

- perform no GPU initialization;
- read no benchmark or generation configuration;
- print exactly one canonical version line to stdout;
- keep stderr empty and return zero on success; and
- complete within a caller-enforced five-second timeout.

This makes it safe for install-time configuration and runtime client
resolution. Callers report distinct diagnostics for timeout,
loader failure, signal, nonzero exit, malformed output, extra output, and
version mismatch.

### Q066 — Must TheRock pass a new build-ROCm-path cache variable?

**Confirmed correction — no.**

TheRock's generated amd-hip toolchain file already defines
`THEROCK_TOOLCHAIN_ROOT` inside the hipBLASLt subproject. The existing
`pre_hook_hipBLASLt.cmake` already requires that variable and uses it to augment
the toolchain PATH.

When `HIPBLASLT_ENABLE_THEROCK=ON`, hipBLASLt can use
`THEROCK_TOOLCHAIN_ROOT` as `ROCM_PATH` for wheel construction and Python
code-generation commands. A duplicate `HIPBLASLT_BUILD_ROCM_PATH` input is not
needed.

### Q067 — Did `develop` require a complete `ROCM_PATH` to build kernels/client?

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

### Q068 — Why does TheRock unset `ROCM_PATH` for subprojects?

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

### Q069 — What does “legacy tool discovery” mean here?

**Clarification:** It describes path-based discovery in older project build
logic, not deprecated compiler tools.

Modern CMake integration would pass explicit executable paths or imported
targets. Existing BLAS/Tensile code still locates some compiler utilities by
searching PATH, so TheRock's project-specific pre-hooks prepend graph-owned
toolchain directories. TheRock comments call this path munging a compatibility
“reacharound” for old project assumptions.

### Q070 — May hipBLASLt read from `THEROCK_TOOLCHAIN_ROOT`?

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

### Q071 — How is the build-time ROCm root selected?

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
construction and Python code-generation commands; plain installation of an
already-built wheel does not require it.
`release_metadata.py` is the sole authority that actually reads and validates
`<root>/.info/version`; do not duplicate it with a CMake existence/readability
preflight. Likewise, do not maintain a second platform-specific toolchain-layout
checklist: imported CMake targets and real generator commands validate the tools
they consume. Never copy hipBLASLt outputs into the selected root.

### Q072 — May a TheRock build fall back when `THEROCK_TOOLCHAIN_ROOT` is missing?

**Decision: Accepted — no.**

When `HIPBLASLT_ENABLE_THEROCK=ON`, missing or invalid
`THEROCK_TOOLCHAIN_ROOT` is a fatal configuration error. Do not fall back to
ambient `ROCM_PATH` or `/opt/rocm`, because that would defeat TheRock's
hermeticity guarantee.

### Q073 — Is the binding metadata format still open?

**Decision: Accepted — use the existing bare JSON absolute-path string.**

The settled value remains the existing bare JSON absolute-path string, now in
the keyed per-user `client.json`. Client version and installation identity are
derived during configuration and first client resolution, so neither is
duplicated in the JSON payload.

### Q074 — Where is raw rocisa packaged on `develop`?

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

### Q075 — What is the governing scope rule?

**Decision: Accepted.**

Do not change existing `develop` behavior unless the canonical/compatibility
wheel and source-only test split requires it. Prefer restoring or retaining
known working build/test wiring over redesigning adjacent rocisa, TheRock, or
runtime packaging concerns.

### Q076 — How should the test runner install artifact wheels?

**Decision: Accepted — reuse the existing TheRock pattern.**

The analogous hipkernelprovider runner installs staged wheels with:

```python
[sys.executable, "-m", "pip", "install", "--no-deps", *wheels]
```

Use the same active-interpreter pip pattern for TensileLite canonical and later
compatibility phases. Do not introduce a component-specific uv invocation when a
working wheel-install precedent already exists.

Both phases use the exact command shape
`sys.executable -m pip install --force-reinstall --no-deps <exact-wheel>`.
In particular, the compatibility phase must not resolve its exact canonical pin
or contact an index or replace the canonical distribution
that passed phase 1. Runner unit tests assert the complete argument lists.

Q084 assigns phase orchestration to the thin TensileLite runner and leaves
categories, markers, timeouts, workers, and JUnit execution with the generic
TheRock pytest runner.

### Q077 — What is the CMake target graph for wheel construction and generation?

**Decision: Accepted.**

```text
tensilelite-canonical-release-wheel
  → build and validate canonical wheel

tensilelite-compatibility-release-wheel
  → build and validate compatibility wheel

tensilelite-build-release-wheels
  → aggregate both release-wheel targets

_rocisa ──────────────────────────────┐
tensilelite-canonical-release-wheel ──┤
                                     ▼
tensilelite-python-build-environment
  → force-reinstall canonical wheel into found build Python
                                     ▼
logic / create-library / ext-op generation
```

Compatibility wheel construction and tests remain outside the device-generation
dependency chain.

Canonical and compatibility wheels use separate exact output paths. Their
dependencies include package code/resources, build backend and metadata,
validator, requirements/build metadata, and the selected build ROCm identity.
The configured Python environment additionally depends on raw rocisa.

Each wheel target builds only its own wheel in an independently cleaned private
staging directory and validates that staged wheel in its independent validator
mode. Neither target cleans or builds directly in the shared final release-wheel
directory. After validation, it copies that one exact wheel to its declared
release path and only then touches its target-specific completion stamp. The
wheel is a declared byproduct and the stamp is the custom-command output, so an
interrupted copy does not mark publication complete and the next build reruns
it. This follows TheRock's existing copy-then-stamp model; atomic replacement is
not required. Neither target clears or modifies the other target's output.

CMake composes the release version during configuration so the generated build
and install graph contains the exact canonical and compatibility wheel paths.
Register `VERSION`, `release_metadata.py`, and the selected build ROCm version
inputs as configure dependencies. If any changes in an existing build tree,
`cmake --build` first regenerates the CMake graph with the new filenames; do
not introduce a build-time wheel-path manifest.

Install the two configured, validated wheel files explicitly with
`install(FILES ...)`; do not install the release-wheel directory. This prevents
stale or unrelated build-tree wheels from entering the test artifact. It does
not add cleanup of an existing install prefix or change TheRock's subsequent
stage-to-artifact copy behavior.

The canonical wheel remains a dependency of every device-generation build.
When the existing `HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS` option is
enabled, make `tensilelite-build-release-wheels` part of `ALL` so the normal
build produces both wheels before TheRock runs `cmake --install`. When the
option is disabled, do not build the compatibility wheel solely for tests.
Preserve the standard build-before-install lifecycle; installation does not
invoke a nested build.

Both independent wheel-build commands use the selected Python and retain
`python -m pip wheel --no-build-isolation --no-deps`. Splitting the targets must
not create isolated build environments or let the compatibility build resolve,
download, or rebuild its declared canonical and pandas dependencies.

### Q078 — What do current defaults build?

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

The current implementation builds the canonical wheel for device generation
and builds the compatibility wheel only for test-artifact packaging.

### Q079 — What does “every device-generation build” cover?

**Clarification:** There should be one implementation in hipBLASLt CMake, but it
is reached through several entry points:

1. standalone raw CMake from `projects/hipblaslt`;
2. hipBLASLt Invoke wrappers that configure that CMake project;
3. checked-in CMake presets, especially default/full and `gemm-libs`;
4. TheRock's hipBLASLt subproject build;
5. rocm-libraries/superbuild integration;
6. CI variants such as sanitizer, coverage, and Linux/Windows builds.

The common rule is attached to `HIPBLASLT_ENABLE_DEVICE=ON`, not reimplemented
in every caller. Host-only configurations bypass the wheel/codegen graph.

### Q080 — Is the device-generation setup defined once or per entry point?

**Decision: Accepted — define it once.**

hipBLASLt CMake owns one centralized dependency rule:

```text
HIPBLASLT_ENABLE_DEVICE=ON
  → _rocisa
  → canonical release wheel
  → installed build Python package
  → logic/create-library/ext-op generation
```

Every device-generation build owns and builds the in-tree `_rocisa`. An
already-importable external rocisa is not an alternative for this graph.
`HIPBLASLT_BUNDLE_PYTHON_DEPS` therefore no longer gates `_rocisa` creation for
an integrated device build; remove that escape hatch with the obsolete Python
mode wiring. Standalone rocisa builds retain their independent entry point.

Device generation does not require the TensileLite host or client targets.
Maintained presets, Invoke wiring, TheRock, and superbuild callers may select
their own client capability independently of this graph.

Raw CMake, Invoke, presets, TheRock, superbuilds, sanitizer builds, and platform
CI only select options and consume this graph. They must not replicate package
preparation or ordering logic.

### Q081 — Where does CMake install the canonical wheel for device generation?

**Decision: Accepted — install it into the found Python environment.**

Use `--force-reinstall --no-deps`. One active hipBLASLt build owns the selected
Python for installation and device generation; concurrent build directories
sharing that interpreter are unsupported.

This matches the current disposable, single-build CI model while ensuring that
same-version source changes reinstall the current wheel.

### Q082 — How do artifact tests run against the installed canonical wheel?

**Decision: Accepted — run the separately copied raw tests with pytest's
default `prepend` import mode.**

Install the canonical wheel into the artifact-test venv and keep existing test
helpers, inherited bases, and harnesses as local test modules. Rewrite the
eleven `tensilelite.Tests.*` statements as test-local imports because tests are
not part of the production wheel.

Do not treat those eleven rewrites as the complete installed-suite migration.
Audit every test selected for the copied artifact tree for project-root paths,
adjacent production-source reads, build/version metadata, checkout-only scripts,
and source-relative resource loading. Record each dependency before deciding
whether the test should use an installed API/resource or remain source-only;
discovery alone does not require rewriting every test or expanding the artifact.

Classify by the subject under test. A test of installed production behavior must
use the installed package API or `importlib.resources`. A test of source/build
machinery such as `VERSION`, wheel construction, `release_metadata.py`,
`tasks.py`, or the `invoke install` workflow remains in source CI and is excluded
from the installed suite. Do not copy production source or build scripts merely
to preserve a path-based test, and do not classify a test as source-only merely
because conversion is inconvenient.

Move every genuine source/build test under the explicit
`tensilelite/Tests/unit/source_only/` directory. Source CI continues collecting
that directory recursively. Artifact installation excludes the whole directory
with one rule, rather than relying on per-file lists, markers, `-k`, or runtime
skips that occur after collection. Moving a test into `source_only/` is the
reviewable declaration of its scope; artifact-layout tests assert that the
directory is absent from the installed test tree.

Do not add a separate `pytest --collect-only` preflight. Each normal category
already collects its configured paths from the exact installed tree with the
checkout absent, and collection errors fail that phase. Preserve the
category-specific paths and options rather than duplicating collection in a
synthetic all-category invocation.

Run the reconstructed artifact suite with the checkout absent. Fix installed
package/resource assumptions that fail in that environment, and keep genuine
source-layout or package-construction tests in source CI.

This preserves the mature suite's current module structure while proving that
production imports come from the installed canonical wheel.

### Q083 — How is the canonical wheel's rocisa dependency satisfied before rocisa packaging exists?

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

### Q084 — Which runner owns TensileLite's canonical and compatibility test phases?

**Decision: Accepted — use a thin TensileLite-specific phase runner which
delegates pytest execution to TheRock's generic runner.**

The TensileLite runner owns component-specific package orchestration:

- construct the final reconstructed test environment first, including
  `ROCM_PATH`, platform loader paths, `PATH`, and the scoped raw-rocisa
  `PYTHONPATH`;
- discover exactly one canonical wheel with the expected version;
- fail on zero or ambiguous matches;
- install it into the active test venv with
  `--force-reinstall --no-deps`;
- invoke the generic runner for the canonical test phase;
- install the compatibility wheel only after the canonical phase passes;
- invoke the generic runner for `compat/tests` with `--run-compat`; and
- return failure when either executed phase fails.

Reuse the same constructed environment for wheel installation and both pytest
phases. A client-executing test category validates its client only when it
reaches that capability.

The generic TheRock pytest runner continues to own:

- category and test-path selection;
- marker expressions;
- timeouts and worker counts;
- pytest invocation; and
- JUnit generation.

If the canonical phase fails, do not install the compatibility wheel and return
failure immediately. A compatibility failure returns failure. Do not require
new JUnit files, directories, preservation, upload, or reporting; the generic
runner's existing optional JUnit behavior remains unchanged.

The compatibility phase and its inputs are removed when the compatibility
package is retired.

### Q085 — What is the single authority for the TensileLite component version?

**Decision: Accepted — use a checked-in `VERSION` file at the TensileLite
project root.**

`VERSION` contains the component release version, for example:

```text
5.0.0
```

The selected build ROCm identity is supplied to `release_metadata.py`, which is
the sole composition and validation implementation for producing:

```text
<component-version>+rocm<rocm-publication-or-base-identity>
```

The exact composed value feeds:

- canonical wheel metadata;
- compatibility wheel metadata and its exact canonical dependency pin;
- a generated C++ header compiled into `tensilelite-client`; and
- release-wheel and client-version validation.

Remove component-version literals from canonical and compatibility setup code.
The native client embeds the generated build-time value and never derives its
identity from the runtime `ROCM_PATH`.

Generate the native header during the same CMake configuration that composes
the exact wheel filenames. `VERSION`, `release_metadata.py`, and the selected
build ROCm identity is a configure dependency; automatic regeneration rewrites the
header, and the header is a normal input that recompiles and relinks
`tensilelite-client` when its value changes. Do not add a second build-time
version authority or path manifest.

`GENERATOR_VERSION` remains a generator/logic compatibility concept. Any
decision to couple its lifecycle to the component release version must be
explicit rather than an accidental consequence of package-version plumbing.

### Q086 — Should the build backend rewrite wheels to embed a client binding?

**Decision: Accepted — no.**

Every wheel archive remains unbound. Source, editable, and prebuilt workflows
install the wheel first and then use `tensilelite-configure-client` when they
need a custom executable. The build backend no longer adds binding metadata or
rewrites wheel `RECORD` files.

This makes the binding an external per-user, per-installation concern, uses one
persistence scheme for normal and editable installations, and preserves both
wheel archive contents and installed distribution metadata. It also removes the
signed-wheel rewriting decision.

### Q087 — Does a multi-config build require configuration-specific Python package state?

**Decision: Accepted — no.**

The canonical wheel, installed Python package, and Python/native compatibility
contract are configuration-independent. Debug and Release clients implement the
same versioned interface; CMake build type is not part of the Python
compatibility identity.

This change does not expand device-generation or raw-rocisa support beyond the
generators and layouts supported on `develop`. In particular, creating complete
per-configuration raw rocisa packages for Visual Studio-style multi-config
generation is out of scope unless a concrete supported path requires it.

The client binding is outside the generator path. Concurrent client
configure/reset operations for one selected Python remain outside the
single-owner contract.

### Q088 — How do standard and custom client install directories resolve?

**Decision: Accepted — the future production layout is fixed; custom layouts
use an explicit keyed per-user binding.**

When the client is promoted to the production ROCm libraries artifact, it will
be placed at:

```text
non-Windows: libexec/hipblaslt/tensilelite/tensilelite-client
Windows:     libexec/hipblaslt/tensilelite/tensilelite-client.exe
```

The Python-SDK adapter will use the corresponding libraries-package trampoline;
the conventional-prefix adapter will use that root-relative location. Until the
promotion lands, a default Python-SDK client request fails clearly and the
current `blas_test` artifact remains the supported packaged client path.
Custom/local installations select their exact client with
`tensilelite-configure-client`.

The directory is fixed and only the platform-native executable suffix differs.
The runtime does not search alternate libexec directory names or `PATH`, and wheel
metadata does not encode a build-specific install layout. This preserves the
standard production contract and wheel transferability without adding code for
a nonstandard production layout that is not currently required.

### Q089 — What is the standalone Windows ROCm-root fallback?

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

### Q090 — Which targets require Python development headers?

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

Preserve rocisa's existing stable-ABI specialization. When `_rocisa` is built
with `ROCISA_USE_STABLE_ABI=ON`, require Python 3.12 or newer plus
`Interpreter`, `Development.Module`, and `Development.SABIModule`. Without
stable ABI, `_rocisa` keeps the Python 3.10 floor and requires `Interpreter` plus
`Development.Module`. This conditional belongs to the extension target and does
not make client-only or host-only paths require Python development components.

This preserves the intended native-generation direction without making the
current standalone executable require headers it does not consume.

### Q091 — Is rocisa coverage-import provenance part of this packaging change?

**Decision: Accepted — no; leave existing coverage behavior unchanged.**

The packaging work supplies raw rocisa through the scoped build/test environment
defined elsewhere in this record, but it does not add coverage-specific import
path assertions, per-object coverage gates, or coverage workflow redesign.
Coverage provenance can be addressed independently if it becomes a demonstrated
coverage problem.

### Q092 — What documentation is in scope for the packaging change?

**Decision: Accepted — update only documentation for interfaces and workflows
changed by this design.**

Documentation must reflect the canonical-wheel build/install flow, controlled
ROCm artifact status and `--no-deps` use, raw rocisa environment, unbound wheels,
post-install client configuration, standard production client path, local
editable workflow, removal of `BUILD|SYSTEM` and the private CMake venv, removal
of the pip client-path setting, and the Python 3.10 package requirement.

Unrelated pre-existing contributor-documentation cleanup is outside this
change.

### Q093 — What focused validation is required for the packaging change?

**Decision: Accepted — update focused package/native tests and rely on existing
affected CI for integration coverage.**

Required focused tests cover:

- a no-clean incremental version-input change proving the rebuilt canonical
  wheel metadata advances to the selected build ROCm identity;
- distinct installation keys for regular wheels in different environments and
  editable installs in different worktrees, intentional overwrite on reinstall
  to the same resolved directory, and per-user binding-root isolation;
- successful device-generation configuration with
  `TENSILELITE_ENABLE_HOST=OFF` and `TENSILELITE_ENABLE_CLIENT=OFF`;
- `VERSION` plus the selected build ROCm identity across canonical metadata,
  compatibility metadata/pin, and the generated client version when enabled;
- post-install configure/reset, atomic single-file replacement,
  mismatched-client rejection, proof that installed distribution metadata and
  the original wheel archive remain unchanged, and the shared
  `~/.tensilelite/helper_cache` rename;
- Python-SDK versus conventional-prefix resolution, no path borrowing, and
  client `--version` success/failure/malformed handling on explicit request;
- canonical/compatibility wheel version, pin, entry-point, resource, and unbound
  content validation, including canonical `Requires-Dist: rocisa`, normalized
  distribution names, exact `py3-none-any` filename/WHEEL tag agreement,
  `_tensilelite_client_binding.py`, `tensilelite_configure_client.py`, and the
  `tensilelite-configure-client` target;
- thin-runner phase ordering and failure propagation; and
- GPU-less native `--version` stdout, stderr, and exit-status behavior for
  client-capable workflows.

Existing Linux/Windows TheRock, artifact-test, sanitizer, and superbuild lanes
provide integrated coverage. This work does not add a separate comprehensive
CMake matrix, duplicate per-caller invariant tests, or coverage workflow.

### Q094 — Where does mutable client configuration live?

**Decision: Accepted — use one keyed per-user registry and never mutate the
installed distribution.**

The common user root is renamed and organized as:

```text
~/.tensilelite/
├── helper_cache/
└── bindings/<installation-id>/client.json
```

`helper_cache/` is one shared per-user cache for all wheels, venvs, and
worktrees. Its existing content-derived cache keys decide reuse, so it must not
be nested below an installation ID. Rename the current default
`~/.tensile/helper_cache` to `~/.tensilelite/helper_cache`, retain the existing
explicit cache-directory override, and do not automatically migrate the old
disposable cache.

Derive `<installation-id>` only from the exact resolved directory of the
imported `tensilelite` package. Different venv wheel installations and editable
worktrees resolve to different keys; different users have different home roots.
A reinstall or upgrade into the same resolved directory intentionally reuses
the same key. After full client validation, configuration atomically replaces
any existing `client.json` at that key; an existing file is never a collision.
Locate the package directory with `importlib.util.find_spec` without executing
`tensilelite.__init__`, so configuration remains independent of package
initialization. The JSON payload remains one bare absolute client-path string.

Runtime uses a present keyed binding exclusively and otherwise uses the selected
installation adapter's default client candidate. `--reset` deletes the current
key's one file.
Concurrent configure/reset for the same key is unsupported. The binding is
external per-user configuration which intentionally survives pip uninstall;
there is no `.dist-info` or `RECORD` mutation, distribution-ownership algorithm,
inter-process lock, tombstone, or multi-file recovery state machine. TheRock
can configure a regular installed wheel from its exact CMake client before the
final ROCm tree exists because the installation key comes from site-packages,
not from the source checkout or final ROCm layout.

### Q095 — When must TensileLite require `tensilelite-client`?

**Question:** Must `tensilelite-client` be an import-time and device-generation
prerequisite, or only a capability-specific dependency when TensileLite actually
uses the executable?

**Decision: Accepted — require it only at actual client execution.**

`import tensilelite`, command-line help, logic validation,
`TensileCreateLibrary`, and hipBLASLt device-library generation must work when
`tensilelite-client` is absent. The package resolves and validates the client
only immediately before an operation launches it, such as benchmark or retune
execution. Q102 later refines this to validate at the first explicit request
for the client path; imports and device generation remain client-free.

**Rationale:** Current device-library generation is a Python/rocisa/toolchain
flow; it does not execute the native client. Making the client an import-time
or pre-generation prerequisite prevents otherwise valid device-library builds
and wrongly couples the generator to a benchmark/validation capability.

**Scope:** This replaces the earlier client-coupled generator design. No CMake
device-generation path may build, install, bind, or validate the client merely
to import TensileLite or generate device libraries.

### Q096 — What is TheRock's optional-client policy?

**Question:** After removing the client from the device-generation dependency
graph, should TheRock still package it, and should the feature branch's Windows
enablement remain?

**Decision: Accepted — retain a packaged client, but restore the Windows
disablement.**

`tensilelite-client` remains a separately consumable TheRock artifact for the
benchmark/validation capability that actually uses it. It is not a transient
build-time input: an artifact consumer may use the packaged executable without
requiring any later device-library-generation build to produce or bind it.

Restore `develop`'s Windows behavior: Windows does not enable, build, install,
or bind `tensilelite-client` merely because hipBLASLt device libraries are
enabled. Non-Windows client builds and their artifact packaging are selected by
the explicit benchmark/validation artifact policy, not by device generation.

**Rationale:** Packaging preserves the client for its real downstream use while
Q095 removes an unrelated build-time coupling. Restoring the Windows disablement
also removes work introduced solely to satisfy that coupling.

It retains the existence of a packaged client artifact; Q097 resolves its
current ownership.

### Q097 — Which artifact owns the optional client now?

**Question:** Should the non-Windows `tensilelite-client` be packaged with the
production BLAS runtime (`blas_lib`) or the benchmark/test payload (`blas_test`)
while the Python package and client are still maturing?

**Decision: Accepted — restore `blas_test` ownership for now.**

Restore TheRock base behavior: package
`libexec/hipblaslt/tensilelite/tensilelite-client` in the non-Windows
`blas_test` artifact. Do not include it in `blas_lib`. The executable therefore
remains available to the packaged benchmark/validation consumers that use it,
without making it part of every production BLAS runtime installation.

**Implementation consequence:** Restore the matching `develop` install
ownership: install the client only for the non-Windows TensileLite test-artifact
workflow and under CMake's `tests` component. TheRock's `blas_test` selector,
not `blas_lib`, includes that installed path. This is a consequence of the
accepted artifact ownership, not a separate decision.

**Rationale:** The current client and Python package have not yet reached the
maturity required for production-runtime ownership. `blas_test` preserves a
real artifact-based execution path while avoiding premature production
distribution.

**Future direction:** Moving the client to `blas_lib` is a deliberate future
change once both the Python package and client have matured sufficiently; it is
not an automatic consequence of a build or packaging change. The concrete
maturity criteria remain to be decided before that promotion.

### Q098 — Should `invoke install` remain a full development setup?

**Question:** Should the Linux source-development convenience command
`invoke install` stop building and binding `tensilelite-client` now that the
package and device-generation paths no longer require it by default?

**Decision: Accepted — keep the full development workflow.**

`invoke install` continues to build rocisa and `tensilelite-client`, install
TensileLite editably, and bind that exact built client. Its purpose is to make
development and end-to-end testing of both TensileLite and its benchmark/
validation client convenient in one command.

**Scope boundary:** This is an explicit source-development convenience, not a
general package-install or hipBLASLt code-generation requirement. It does not
weaken Q095: importing TensileLite or generating device libraries must not
require a client, and CMake must not build or bind one merely for those flows.

### Q099 — Which CMake model should device generation use?

**Question:** For the hipBLASLt device-generation path, should the current
wheel-install/client-binding graph keep the canonical wheel while dropping the
client prerequisite?

**Decision: Accepted — use the canonical wheel without the client.**

`HIPBLASLT_ENABLE_DEVICE=ON` must support both
`TENSILELITE_ENABLE_CLIENT=OFF` and `TENSILELITE_ENABLE_HOST=OFF`. Device
generation constructs the canonical TensileLite wheel and force-installs it
with `--no-deps` into the selected Python environment before logic validation
or device-library creation. Those Python commands import TensileLite from the
installed canonical wheel; the build-tree rocisa package remains available only
through the command-scoped `PYTHONPATH`.

The canonical wheel is part of the normal build whenever device generation or
the TensileLite test-artifact workflow needs the Python package. The
compatibility wheel remains part of the release/artifact-test workflow defined
in Q103.

Device generation does not build, configure, bind, validate, or launch
`tensilelite-client`, and it does not require the TensileLite C++ host target.
Q098 remains the intentional full source-development exception. Every other
client, package, artifact, and testing decision continues to be considered and
recorded independently.

### Q100 — Should client selection and binding change?

**Question:** Once client validation is deferred to actual use, should
TensileLite replace its existing standard-path resolver and explicit
per-installation client binding registry?

**Decision: Accepted — no mechanism change.**

Keep the explicit `tensilelite-configure-client` binding mechanism for custom
and source-development clients, its client-version validation, and the selected
adapter's default-client policy. Q102 defines the deferred lookup boundary. No
import path reads, resolves, or validates the client.

This preserves Q098's full `invoke install` workflow while leaving
Python-SDK and conventional-prefix default lookup to their respective adapters.

### Q101 — What remains eager at package import?

**Question:** After removing client resolution from package initialization,
should `import tensilelite` also defer its ROCm-release validation?

**Decision: Accepted — defer only the client.**

Package initialization continues to require a ROCm installation whose release
matches the TensileLite distribution. It does not import rocisa, resolve a
client binding, execute, or validate `tensilelite-client`. Generator commands
that load logic or code-generation modules require rocisa; client selection and
validation occur only at the later execution boundary defined by Q095.

**Rationale:** rocisa and the matching ROCm toolchain remain prerequisites for
the generator command paths, while the client is an independent
benchmark/validation capability.

### Q102 — When does a client-path request validate the client?

**Question:** Once client lookup is removed from imports, may
`getClientExecutablePath()` return a missing path while a caller merely writes
a run script, or must every returned client path already be valid?

**Decision: Accepted — a returned client path is always validated.**

Importing TensileLite or any of its modules does not look up the client.
However, the first explicit request for the client path resolves the configured
or standard client and performs the existing file, executable, and version
validation. It returns a valid path or fails with that diagnostic.

Consequently, normal script/configuration generation that asks for the client
path fails when the client is absent. This is intentional: callers may rely on
a returned path being usable. CPU-only paths remain unaffected because they do
not request a client path.

**Supersession:** This refines Q095 and Q100's deferred-validation timing. It
supersedes Q095's earlier statement that emitting a script/configuration does
not require the executable.

### Q103 — Does TheRock continue testing its release wheels?

**Question:** Since CMake device generation uses the canonical wheel, should
TheRock continue constructing and testing its canonical and compatibility
wheels from reconstructed artifacts?

**Decision: Accepted — yes, unchanged.**

Continue producing both the canonical `tensilelite` and compatibility wheels,
staging them as artifacts, and running their reconstructed-artifact test
coverage. This verifies the distributable Python packages in an environment
separate from CMake device generation.

**Scope boundary:** Q099 makes the canonical wheel the CMake device-generation
input. It does not reduce package production, packaging validation, or
artifact-test coverage.

### Q104 — Must every wheel test phase require the client?

**Question:** Should the release-wheel runner retain its feature-branch-wide
`tensilelite-client --version` gate, or follow the existing category model in
which only client-executing tests require the client?

**Decision: Accepted — retain the existing category model; no new split is
needed.**

Wheel discovery and installation must not consult `tensilelite-client`. The
existing generic test categories already separate capability use:

- rocisa and TensileLite unit categories run client-free; and
- the `ffm-quick` common-GEMM category requires and validates the client when
  it actually builds and runs kernels.

Remove the runner's universal `client --version` gate. Do not add a parallel
test-runner structure or reduce wheel coverage. This restores `develop`'s
capability-specific test behavior while retaining the branch's wheel artifact
coverage.

### Q105 — How does the artifact runner select release wheels without a client?

**Question:** After removing the client-version gate, what client-free rule
selects the canonical and compatibility wheels from a reconstructed artifact?

**Decision: Accepted — use the wheel artifact's own metadata.**

Require exactly one canonical `tensilelite` wheel and exactly one compatibility
wheel. Their parsed wheel versions must match. Keep the existing build-time
wheel-content validation, which verifies each wheel's filename and metadata
version and the compatibility wheel's exact canonical-wheel dependency pin.

Do not introduce another artifact-time version authority and do not consult
`tensilelite-client` to discover or validate the wheels.

### Q106 — How do the standard hipBLASLt presets select the client?

**Question:** After restoring `develop`'s client independence from device
generation, should the `gemm-libs` and `hipblaslt-clients` presets retain the
feature branch's client enablement?

**Decision: Accepted — restore their `develop` settings.**

`gemm-libs` enables device generation with both `TENSILELITE_ENABLE_HOST=OFF`
and `TENSILELITE_ENABLE_CLIENT=OFF`. `hipblaslt-clients` retains its TensileLite
host setting but restores `TENSILELITE_ENABLE_CLIENT=OFF`. The standalone
`tensilelite` preset remains the explicit client-build preset.

Remove the branch-added invariant tests which expect device generation to
reject client/host-off configurations. Existing device-build integration
coverage validates the restored preset contracts.

### Q107 — How do version consumers drive CMake targets?

**Question:** Which CMake targets should compute the shared TensileLite release
version, build release wheels, and generate the native client's version header?

**Decision: Accepted — separate the consumers while preserving one version
authority.**

`VERSION` plus `release_metadata.py` remain the sole logical version authority.
CMake computes `TENSILELITE_DISTRIBUTION_VERSION` whenever either a wheel or
the client version header needs it.

- The canonical wheel is built whenever device generation or test-artifact
  packaging needs the Python package. It is a direct dependency of device
  generation and is force-installed before the generator commands run.
- The compatibility wheel is built for the release/artifact-test workflow,
  where it is staged and tested.
- `TensileLiteClientVersion.hpp` is generated only when
  `TENSILELITE_ENABLE_CLIENT=ON`, because only `tensilelite-client` includes
  it.

The CMake Python environment for device generation depends on the canonical
wheel and `_rocisa`, not on `tensilelite-client`. Remove its client-binding
wrapper from that path. This preserves canonical-wheel code generation while
removing the client dependency.

### Q108 — Must the Python package runtime depend on `rocm[devel]`?

**Decision: Accepted — no.**

For a TheRock wheel installation, TensileLite must not install or expand
`rocm[devel]` merely to validate that the Python wheel and ROCm release match.
The compatibility check uses `rocm_sdk_core.__version__` as the authoritative
full Python publication identity. Compiler, assembler, bundler, and
device-enumerator commands are resolved through the active Python
environment's `rocm-sdk-core` console-script trampolines; TensileLite does not
need `rocm_sdk_core.get_core_root()` or a physical SDK root for this model.

The Python SDK adapter selects only explicit Python console-script locations:
the active interpreter's scripts directory, followed by that platform's Python
user-install scripts directory when the package is installed with `--user`.
Those are ordered locations within the same Python installation model, not an
ambient `PATH` fallback. TensileLite must not select an individual executable
from a conventional ROCm prefix when a Python SDK script is absent. `PATH`
selects an executable; it does not resolve that executable's native
dependencies. The selected Python tool instead executes its package payload
with that payload's own loader closure. Borrowing one missing tool from another
ROCm installation would therefore create a mixed toolchain, not repair the
Python SDK installation. A missing expected trampoline remains an error for
TheRock to fix in its wheel publication.

If a client-capable workflow requests `tensilelite-client`, the interim Python
SDK package implementation fails clearly because `rocm-sdk-libraries` does not
yet ship the client. The final state uses that package's console-script
trampoline. The library package owns the native payload and the trampoline
locates it; the core package neither owns the client nor provides a synthetic
combined prefix.

When no active Python core SDK is installed, the selected native ROCm root's
`.info/version` is temporarily authoritative. It contains only the base
compatibility value, so this fallback cannot distinguish different nightly, RC,
or CI publications on the same base release line. Keep that limitation visible
until native ROCm roots expose an equivalent full publication identity.

This changes only the Python SDK package runtime model. It does not change the
separate conventional-prefix contract, the build-time TheRock toolchain root,
or the client-free generation boundary in Q095.

**Rationale:** `rocm-sdk-core` is already installed by `rocm` and
`rocm[libraries]`. Pulling the large devel archive solely for a version comparison
adds an unrelated package dependency. The core package already provides the
required tool trampolines. The libraries client trampoline is the future
production state described below; until then, the native client remains
capability-specific.

### Q109 — Where will `tensilelite-client` eventually be delivered?

**Decision: Accepted final state — through the production BLAS runtime artifact
(`blas_lib`) and `rocm[libraries]`.**

Q097 describes the transitional `blas_test` ownership. The final state replaces
it after the production artifact and Python package changes land; this decision
records the target contract without claiming that the current artifact layout
has already changed.

When promoted, install the client at:

```text
<ROCm root>/libexec/hipblaslt/tensilelite/tensilelite-client[.exe]
```

in `blas_lib`, which delivers it through `rocm-sdk-libraries` and
`rocm[libraries]`.

The `rocm-sdk-libraries` wheel installs an interpreter-local
`tensilelite-client` console-script trampoline. The trampoline executes the
library-owned client below its own platform payload and forwards arguments,
including `--version`, unchanged. TensileLite's Python SDK adapter invokes this
known entry point when client capability is requested; it does not search PATH
or inspect a package payload directory.

**Why not `rocm-sdk-core`:** the client is built by hipBLASLt and has a
BLAS-side runtime closure. Making mandatory core own it would invert the
dependency direction by making core depend on the optional BLAS layer.

**Why not `rocm[devel]`:** devel is a complete SDK view for headers, CMake
metadata, static libraries, and development tooling. The client is a native
runtime benchmark/validation executable; it does not need devel-only files to
run, and devel must not own a second copy.

**Capability boundary:** Q095 remains unchanged. Import, logic validation,
`TensileCreateLibrary`, and hipBLASLt device-library generation do not require
or validate the client. Benchmark and retune execution resolve and validate the
client only when they actually launch it.

### Q110 — How do Python SDK packages and conventional ROCm prefixes coexist?

**Decision: Accepted — use two installation adapters with no path borrowing.**

The Python SDK package adapter and the conventional-prefix adapter are distinct
runtime models:

| Concern | Python SDK packages | Conventional prefix |
| --- | --- | --- |
| ROCm identity | `rocm_sdk_core.__version__` (exact full publication identity) | `<root>/.info/version` (base `A.B.C` only) |
| Toolchain | active interpreter scripts, then the platform's Python user scripts, for `rocm-sdk-core` console-script trampolines | `<root>/bin` and `<root>/lib/llvm/bin` |
| Client | Interim: clear unavailable-client error. Final: the same ordered Python script locations for `rocm-sdk-libraries` `tensilelite-client` | `<root>/libexec/hipblaslt/tensilelite/tensilelite-client[.exe]` |
| Root discovery | none; package trampolines own package payload resolution | explicit `ROCM_PATH`, `/opt/rocm`, or a ROCm tool discovered on PATH |

For ROCm identity, compiler/toolchain, and client resolution, an active Python
SDK installation never falls back to `ROCM_PATH`, `/opt/rocm`, or an ambient
tool. Conversely, the conventional-prefix adapter does not import or inspect
Python SDK package payloads. The prefix adapter's base-only identity comparison
deliberately cannot distinguish nightly, RC, or CI publications sharing the
same `A.B.C` line. This does not redefine optional benchmark conveniences such
as the separate `amd-smi` clock-pinning probe.

This no-path-borrowing rule applies to individual tools as well as root
discovery. If the Python SDK lacks a required console script in either of its
explicit script locations, TensileLite fails instead of resolving that one name
through `PATH` or a conventional prefix. The Python toolchain's native
libraries are resolved by the selected executable's own loader configuration,
not by executable `PATH`; mixing a wheel compiler with a prefix bundler is a
different toolchain model and has no compatibility guarantee.

### Q111 — What is the final ROCm-identity source for each installation model?

**Decision: Accepted — keep three explicit identity contracts.**

- A TheRock source or CI build uses the exact forwarded
  `THEROCK_PACKAGE_VERSION` publication identity.
- An active Python SDK uses the exact `rocm_sdk_core.__version__` publication
  identity.
- A conventional system ROCm prefix uses the base compatibility identity from
  `<root>/.info/version`.

TheRock's base `version.json` value is not a substitute for
`THEROCK_PACKAGE_VERSION`: it cannot distinguish nightly, RC, or CI
publications on the same base line. The TheRock branch must therefore forward
`THEROCK_PACKAGE_VERSION` into the isolated hipBLASLt CMake configure before
exact validation is enabled.

### Q112 — How does the TheRock build model enter TensileLite?

**Decision: Accepted — add a distinct `TheRockCIRocm` adapter.**

`TheRockCIRocm` is the build-graph adapter beside `PythonRocm` and
`SystemRocm`. It is selected only for the command-scoped TheRock source-build
environment and must not pretend that TheRock's staged compiler tree is a
conventional ROCm prefix.

The adapter uses the selected `THEROCK_PACKAGE_VERSION` as its exact identity.
It does not read `.info/version`, `ROCM_PATH`, `/opt/rocm`, or
`hipconfig --rocmpath`.

### Q113 — How does `TheRockCIRocm` find tools without `ROCM_PATH`?

**Decision: Accepted — preserve develop's graph-prepared `PATH` and explicit
compiler arguments.**

TheRock's hipBLASLt pre-hook already prepends graph-owned tool directories to
`PATH`, and develop captures that configured `PATH` for build-time Python
commands. Device generation also receives the explicit CMake-selected compiler.
The installed-wheel build path must retain that contract.

The command-scoped `THEROCK_PACKAGE_VERSION` both activates
`TheRockCIRocm` and supplies its identity. Do not add a second toolchain-root
environment variable and do not export `ROCM_PATH` into wheel construction,
logic validation, or create-library commands.

### Q114 — Which local installation wins when both Python and system ROCm are
present?

**Decision: Accepted — the active Python SDK wins.**

If the selected interpreter contains `rocm_sdk_core`, the build frontend and
runtime both use `rocm_sdk_core.__version__` and the interpreter-local tool
trampolines. They do not borrow `ROCM_PATH` or individual tools from a system
prefix. Only when no active Python SDK exists does the conventional-prefix
adapter select explicit `ROCM_PATH`, `/opt/rocm`, or `hipconfig --rocmpath`.

Wheel construction must use the same precedence as runtime initialization so a
locally built wheel does not encode one identity and immediately validate
against another.

When the same Python environment remains active, its Python SDK continues to
win at both build and execution. Keep the current validation strengths in this
work: `PythonRocm` compares its exact publication identity, while `SystemRocm`
compares the base `A.B.C` compatibility line. Do not add adapter-provenance
metadata or a new cross-adapter normalization matrix now; strengthen that policy
later only if a concrete workflow requires it.

### Q115 — How does the stack transition before TheRock forwards its package
version?

**Decision: Accepted — keep the PR 17 compatibility fallback only while
validation is disabled, then require the forwarded identity.**

`users/alvasile/AIHPBLAS-3989-runtime-validator-coverage` may retain the
temporary `0.0.0` identity, with a visible warning, when it is built by an
unchanged TheRock main that does not yet forward `THEROCK_PACKAGE_VERSION`.
This keeps the validation-disabled PR buildable and is not a supported release
identity.

The TheRock forwarding change is a prerequisite for enabling exact version
validation later in the stack. The validation-enabling change must reject a
missing or `git` `THEROCK_PACKAGE_VERSION`; it must not silently retain
`0.0.0`, downgrade to the base `version.json` value, or inspect the parent
CMake cache as a backchannel.

### Q116 — Does the dormant TheRock fix belong entirely in PR 17?

**Decision: Rejected — preserve the existing stack's ownership boundaries.**

PR 17 is a convenient integration point on which to develop and validate the
complete correction, but it is not the final owner of every changed line. Each
piece must be committed independently so it can be cherry-picked back to the
earliest PR that owns that contract. The affected PRs are then manually rebased
and restacked in order.

In particular, the version-input contract, canonical-wheel build environment,
ROCm installation adapters, selected-installation tool lookup, TheRock identity
requirement, and validation gate must remain separately reviewable. The
validation-enabling PR must stay small and activate behavior already established
and tested below it; it must not introduce the adapter or build-environment
correction itself.

### Q117 — Which PR owns each part of the correction?

**Decision: Accepted — assign changes to the earliest coherent contract owner,
then restack manually.**

The changes may first be developed as independent commits on top of PR 17 for
integrated validation. They must then be cherry-picked to these final owners:

| Change | Owning PR |
| --- | --- |
| Remove `ROCM_PATH` from canonical and compatibility wheel construction once `TENSILELITE_ROCM_VERSION` is authoritative | PR 9 — explicit ROCm build identity |
| Restore develop's captured graph-prepared `PATH`, omit `ROCM_PATH` from TheRock device generation, and pass command-scoped `THEROCK_PACKAGE_VERSION` | PR 10 — canonical-wheel client-free device generation |
| Add the dormant `TheRockCIRocm` adapter and its adapter-selection tests | PR 11 — dormant ROCm installation/lazy-client model |
| Make any required selected-installation validator integration changes | PR 12 — selected-installation toolchain/enumerator plumbing |
| Retain only genuine coverage fallout; add no new production mechanism | PR 17 — runtime and validator coverage repair |
| Require a real non-`git` `THEROCK_PACKAGE_VERSION` and remove the temporary fallback | PR 18 — TheRock ROCm identity |
| Remove only the version-validation gate | PR 19 — ROCm version validation |
| Select the active Python SDK identity for local wheel builds and remove only the Python-SDK runtime gate | PR 20 — Python SDK runtime |
| Reconcile the decision records and implementation plans with this final contract | PR 21 — documentation |

The TheRock-side change is only the parent-to-child forwarding of
`THEROCK_PACKAGE_VERSION`. Forwarding a second base-only
`THEROCK_ROCM_VERSION` is not required by this contract. Tests travel with the
behavior they protect rather than being accumulated in PR 17.

### Q118 — Should this work strengthen cross-adapter compatibility validation?

**Decision: Rejected — retain the existing simplified validation rules.**

`TheRockCIRocm` validates the exact `THEROCK_PACKAGE_VERSION`, `PythonRocm`
validates the exact `rocm_sdk_core.__version__`, and `SystemRocm` validates the
base `A.B.C` value exposed by `.info/version`. This change does not add build
adapter provenance to wheel metadata or attempt to normalize every possible
Python-SDK/system-prefix transition.

### Q119 — How does a local build select ROCm when its Python is a ROCm venv?

**Decision: Accepted — the active build interpreter's Python SDK wins.**

Every local build frontend queries the selected Python interpreter first. If
that interpreter contains `rocm-sdk-core`, its installed distribution identity
(equivalent to `rocm_sdk_core.__version__`) supplies the wheel's exact ROCm
identity, and its interpreter-local tool trampolines supply the Python SDK tool
model. The build does not consult `ROCM_PATH` or borrow individual tools from a
system prefix in that case.

Only when the selected interpreter has no Python ROCm SDK does the build use the
conventional-prefix selection order and `.info/version`. An active but
incomplete Python ROCm build environment fails clearly rather than silently
falling back to system ROCm.

### Q120 — What validation closes the TheRock and local-build correction?

**Decision: Accepted — prove all four supported build/runtime environments.**

1. PR 17 with unchanged TheRock main: validation remains disabled, the
   temporary identity is accepted, no build-time `ROCM_PATH` is introduced, and
   Linux device generation succeeds.
2. The final stack with TheRock package-version forwarding: the exact
   `THEROCK_PACKAGE_VERSION` reaches `TheRockCIRocm`, validation is enabled, and
   Linux and Windows device-generation paths pass.
3. A local system ROCm build: the build and runtime use the selected
   conventional prefix and its base `.info/version` identity.
4. A local ROCm venv build: the build and runtime use the active interpreter's
   `rocm-sdk-core` identity and tool trampolines even when a conflicting
   `ROCM_PATH` exists.

Each implementation commit carries the focused tests for its own PR boundary
before the cross-repository builds are run. The independent Windows `<version>`
header collision remains a separate defect and must not be confused with the
version/installation-adapter validation described here.

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
  `tensilelite-client`; Q097 retains that current ownership split.
- `blas_test` currently carries test binaries/data, Python test artifacts, and
  `tensilelite-client`; Q109 records the eventual promotion into the production
  runtime artifact.

### Current test-runner contract

The active runner installs the canonical wheel, reserves `PYTHONPATH` for raw
rocisa, and lets pytest's default `prepend` mode expose the separately copied
tests. It retains reconstructed conventional-prefix coverage.

The separate Python-SDK integration phase remains to be added: it must install
the matching ROCm Python packages, exercise the active `rocm_sdk_core` adapter,
and omit `rocm[devel]`.

## Accepted implementation constraints

- hipBLASLt CMake has one source-build mode; an installed-package consumer mode
  returns only with a concrete consumer.
- CMake uses the selected writable Python directly. That environment is
  single-owner during installation and generation, and canonical installation
  uses `--force-reinstall --no-deps`.
- Workflows provision runtime requirements and build tools from the project
  metadata; CMake relies on authoritative pip and real generator commands for
  validation rather than duplicate import checks.
- Device-generation commands depend explicitly on `_rocisa`, the canonical
  wheel, and the configured Python environment; they do not require the client.
- Raw rocisa remains a scoped build/test artifact on `PYTHONPATH`; proper rocisa
  distribution packaging is follow-up work.
- The canonical wheel is a controlled ROCm artifact until that rocisa packaging
  exists; its temporary `pip check` missing-distribution diagnostic is accepted.
- The checked-in `VERSION` file is the component-version authority;
  `release_metadata.py` combines it with the selected build ROCm identity for
  wheel metadata and the generated native-client header.
- `tensilelite-client` is currently a non-Windows `blas_test` artifact for
  benchmark/validation use. Its `blas_lib` promotion is a separate future
  decision.
- Every wheel archive and installed distribution remains unbound and unchanged;
  `tensilelite-configure-client` writes only the exact installation's keyed file
  below the current user's `~/.tensilelite/bindings` root.
- Python package state is configuration-independent. Client bindings are
  resolved only when a caller requests the client; this does not expand
  `develop`'s multi-config or raw-rocisa support.
- The future production client location is
  `libexec/hipblaslt/tensilelite`; custom layouts use an explicit keyed
  per-user binding.
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

1. Exact TensileLite source-build identity beyond the distribution version.
2. Proper rocisa distribution/wheel packaging, replacing raw-package
   `PYTHONPATH` injection and making canonical-wheel dependency checks clean.
