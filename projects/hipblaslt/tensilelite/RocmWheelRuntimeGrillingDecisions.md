# ROCm Wheel Runtime Grilling Decisions

## Purpose

This is the living record for the staff-level Python/C++ packaging review of
TensileLite compatibility with ROCm installed from TheRock wheels. The session
resolves one decision at a time. Product implementation begins only after the
session reaches shared understanding.

## Existing accepted constraints

- `tensilelite-client` is a production ROCm BLAS runtime artifact on Linux and
  Windows.
- The TensileLite wheel and native client use
  `<component-version>+rocm<base-ROCm-version>` as their compatibility identity.
- The base ROCm version is compatibility identity; TheRock dev, nightly, RC,
  and source revisions are publication identity and remain outside the current
  TensileLite version.
- Standard filesystem installations locate the client at
  `<ROCm root>/libexec/hipblaslt/tensilelite/tensilelite-client`.
- Runtime validation happens once during package initialization and freezes the
  selected client for the process.
- No manifest, binary hashes, or component ABI handshake is added.

## Verified findings

### F1 — Wheel-installed ROCm is not currently resolved

On POSIX, `tensilelite/_rocm.py` resolves an explicit `ROCM_PATH` or falls
directly back to `/opt/rocm`. It then unconditionally reads
`<root>/.info/version`. An official TheRock ROCm wheel environment without a
filesystem ROCm prefix therefore cannot initialize TensileLite.

TheRock provides `python -m rocm_sdk path --root` and
`python -m rocm_sdk version`, and its wheel packaging does not reproduce the
traditional `.info/version` contract.

### F2 — The proposed raw version equality is incompatible with the accepted version model

The TensileLite wheel records the base compatibility release, for example
`5.0.0+rocm10.1.0`. `rocm_sdk version` reports the PEP 440 publication version,
for example `10.1.0a20260807`, `10.1.0rc2`, or
`10.1.0.dev0+<sha>`. Exact raw string equality would reject matching nightly,
RC, and dev SDKs. `packaging.version.Version(value).release` recovers the
accepted base compatibility identity `10.1.0`.

### F3 — The public wheel root API requires the devel package

`python -m rocm_sdk path --root` imports the TheRock devel support and fails
when `rocm[devel]` is absent. It may expand the devel tree on first use. The
production `tensilelite-client` originates in the BLAS runtime/libraries
artifact and is also represented under the expanded devel root.

### F4 — The current TheRock TensileLite test does not cover this install shape

The test reconstructs a filesystem ROCm tree, sets `ROCM_PATH`, prepends its
tool directories to `PATH`, and then installs the TensileLite wheel. It does not
exercise a venv containing only the published ROCm wheel layout.

### F5 — D2's tool-provisioning rationale is stale

TheRock's core wheel installs console-script trampolines for `amdclang`,
`amdclang++`, `hipcc`, and `offload-arch`. Its packaged LLVM runtime payload
includes the complete installed `lib/llvm/bin/**` tree. TheRock itself directly
uses `clang-offload-bundler` from that tree, proving that binary is built and
packaged, but it has no venv console-script trampoline. The current source
checkout has no built ROCm wheel to prove whether `amdgpu-arch` is retained as
the usual symlink to `offload-arch`; only `offload-arch` is part of TheRock's
declared console-script surface. The ROCm-coupled release decision can remain
valid even though the claim that pip cannot provide the toolchain is no longer
valid.

### F6 — The POSIX tool search can silently select a stale system ROCm

`Toolchain/Validators.py` searches explicit `ROCM_PATH`, then `/opt/rocm`, then
`PATH`. With no explicit `ROCM_PATH`, a venv ROCm can therefore validate against
one SDK but compile with tools from a stale `/opt/rocm`. A characterization test
currently pins the default-before-`PATH` ordering.

### F7 — The proposed change is broader than one production file

Runtime resolution can remain centered in `_rocm.py`, but a complete change
requires focused tests and contract-document updates. If source builds inside a
wheel-installed SDK are supported, `release_metadata.py` and its CMake
dependencies also require a second metadata source. If client-validation errors
must name the selected mechanism, `_runtime.py` must retain that provenance.

### F8 — TheRock officially separates runtime deployment from a complete build prefix

The primary-source review is captured in
`TheRockRocmWheelPackageRolesResearch.md`. TheRock's stable package-role model
is:

- `rocm-sdk-core`, `rocm-sdk-libraries`, and device wheels are minimized
  runtime/deployment packages consumed through Python APIs;
- `rocm-sdk-devel` synthesizes the coherent full ROCm prefix used for headers,
  CMake/pkg-config files, static/import libraries, compiler resources, and
  traditional `ROCM_PATH`-style builds; and
- individual compiler tools may run from core, but that does not make the core
  package directory a supported general development root.

The clearest negative evidence is closed, unmerged TheRock PR #7108, which
proposed returning the core package from `rocm-sdk path --root` without devel.
Maintainer review rejected treating core as generally consumable through
`ROCM_PATH` and restated that Python projects build against devel but run
against runtime packages through `rocm_sdk` APIs.

### F9 — TheRock has separate base-compatibility and exact-publication versions

The primary-source version review is captured in
`TheRockVersionAuthorityResearch.md`. TheRock's checked-in `version.json` is the
authority for base `X.Y.Z`. Top-level CMake, rocm-core's installed
`.info/version`, the native rocm-core API, and manifest `rocm_version` expose
that derived base value.

Release tooling adds channel/build identity to produce an exact package version:
for example `X.Y.ZaYYYYMMDD`, `X.Y.ZrcN`, or `X.Y.Z.dev0+<sha>`. The Python SDK
exposes this as `rocm_sdk.__version__` / `rocm-sdk version`, while the manifest
records it as `rocm_package_version`. The manifest deliberately records both
the exact and normalized values.

`.info/version` remains supported and is not deprecated, but it is not an exact
nightly/dev/RC identity and is not exposed at a stable public root in a wheel
installation. Official TheRock guidance tells downstream Python frameworks to
pin/check the full SDK package version; TheRock-built external wheels also
embed the full ROCm publication version for nightly packages.

Consequently, the existing TensileLite `+rocmA.B.C` tag can enforce only base
release compatibility. It cannot enforce D2's stated promise of the exact ROCm
artifact set because stable, nightly, RC, and dev SDK publications on the same
`A.B.C` line all normalize to the same tag.

### F10 — `.info/version` is available before current-branch TheRock configuration

`develop` has no `release_metadata.py`, no `_rocm.py`, and no direct
`.info/version` read in the hipBLASLt/TensileLite packaging path. The current
branch adds those reads in two places: `release_metadata.py` composes the wheel
and native-client version, while `tensilelite/_rocm.py` validates the installed
runtime at import.

For a current-branch TheRock build, hipBLASLt selects the `amd-hip` compiler
toolchain. TheRock derives `THEROCK_TOOLCHAIN_ROOT` from the `hip-clr` dist
tree, not ambient ROCm. `hip-clr` has `rocm-core` as a runtime dependency;
rocm-core owns `.info/**`. A subproject's configure command depends on the
selected compiler toolchain's stage stamp, and staging copies the target plus
all transitive runtime-dependency stage directories into its dist tree.
Consequently, `<THEROCK_TOOLCHAIN_ROOT>/.info/version` is available before
hipBLASLt's current-branch configure-time call to `release_metadata.py`.

This preserves TheRock hermeticity: ambient ROCM/HIP variables are cleared for
subprojects, and the root comes from the build graph. The runtime wheel resolver
must not use this build-only variable or fall back through it; installed wheels
use their selected filesystem root or their active Python SDK separately.

## Decision queue

1. Whether official ROCm wheel installations are a supported TensileLite
   runtime/generation environment.
2. Whether that support requires `rocm[devel]`, or import must also work with a
   libraries-only wheel installation.
3. How the wheel SDK's publication version maps to TensileLite's compatibility
   identity.
4. Resolution precedence and whether a broken higher-priority source may fall
   through.
5. Whether all wheel-SDK queries must use the active Python interpreter.
6. Whether source builds inside a wheel-installed SDK are in scope.
7. How resolution provenance is represented and reported.
8. How D2's rationale should describe ROCm wheel ownership of toolchain and
   native dependencies.
9. POSIX toolchain search precedence and its relationship to the resolved SDK.
10. Whether missing TheRock tool trampolines block this change or remain a
    coordinated TheRock fix.
11. Focused unit and integration validation required before merge.

## Decisions

### G001 — Are official ROCm wheel installations supported?

**Decision: Accepted — official ROCm wheel installations are a supported
TensileLite environment.**

A released TensileLite wheel must work with the matching ROCm SDK delivered as
TheRock wheels; support is not limited to a traditional `/opt/rocm` or relocated
filesystem-prefix installation. The wheel-based SDK remains ROCm-owned and must
satisfy the same base-release, rocisa, and native-client compatibility boundary.

Consequently, POSIX runtime discovery cannot fall directly from an unset
`ROCM_PATH` to `/opt/rocm`. It must recognize a matching ROCm SDK installed in
the selected Python environment. The exact required ROCm wheel subset and
discovery mechanism are resolved by subsequent decisions.

### G002 — Does wheel-based support require `rocm[devel]`?

**Decision: Superseded by G003A.**

**Clarification from artifact audit:** `rocm[devel]` is not currently proven to
contain a unique artifact required by the released TensileLite wheel. The
matching `rocm[libraries]` installation also brings in `rocm-sdk-core`; together
those packages contain the production client, compiler payload, HIP tooling,
and LLVM tools used by generation. `rocisa` remains separately supplied in
either case.

The concrete libraries-only gaps are discoverability and command exposure:

- the public `rocm_sdk path --root` command refuses to return a unified root
  without `rocm[devel]`;
- `clang-offload-bundler` exists in the core wheel's `lib/llvm/bin` payload but
  has no venv console-script trampoline; and
- the default `amdgpu-arch` enumerator lacks a declared trampoline; TheRock
  exposes `offload-arch` instead, and explicit target selection does not require
  an enumerator.

In the installed ROCm 7.2.4 toolchain used for this review, `amdgpu-arch` is a
symlink to the same `offload-arch` executable. `offload-arch` is the newer
vendor-neutral LLVM interface and can restrict detection with
`--only=amdgpu`, `--only=nvptx`, or `--only=intel`; `amdgpu-arch` is retained as
the historical AMD command name. TheRock treats `offload-arch` as the canonical
public command. TensileLite's continued acceptance/defaulting of
`amdgpu-arch` is therefore compatibility policy, not a need for a distinct
detector implementation.

This initial libraries-only analysis is superseded by G003A. TensileLite's
supported generation workflow requires the full devel SDK, so
`rocm_sdk path --root` is a legitimate public way to obtain its coherent ROCm
prefix in a wheel installation. That root remains necessary for the fixed
production-client path and the complete compilation/linking environment; it is
not retained merely as a convenience for reading a version file.

The version source is a separate pending decision. It may use the root's
`.info/version` or the public `rocm_sdk.__version__` base value, but resolving
the devel root itself remains part of the supported wheel-SDK contract.

### G003 — What tool names must the ROCm wheel environment expose?

**Decision: Accepted — add a `clang-offload-bundler` trampoline and migrate
TensileLite to canonical `offload-arch`.**

TheRock must expose the packaged `clang-offload-bundler` binary through a venv
console-script trampoline, alongside its existing compiler and `offload-arch`
trampolines. TensileLite must try `offload-arch` first as its POSIX device
enumerator, then try `amdgpu-arch` only when `offload-arch` is unavailable,
fails, or produces no supported AMD ISA. Retain `amdgpu-arch` as this temporary
compatibility fallback for traditional ROCm installations and existing explicit
callers; do not require it from the ROCm wheel command surface. Its removal is a
later focused compatibility cleanup.

This resolves tool discovery through the selected Python environment's `PATH`
without needing a synthetic devel root.

### G003A — Does the no-devel decision cover generation that compiles or links kernels?

**Decision: Accepted — require the full devel SDK; do not define a reduced-capability TensileLite mode.**

TheRock supports runtime import, version checking, client location/loading, and
isolated explicitly guaranteed core tools without a devel root. Its current
official contract nevertheless classifies a workflow that compiles or links
generated code as build-time SDK use and directs such consumers to
`rocm[libraries,devel]`.

Therefore G002 is unambiguous for runtime initialization but too broad for
generation. Supporting core/libraries-only compilation would require TheRock to
explicitly guarantee and test the complete closure of compiler tools,
resources, headers, device files, and link inputs used by TensileLite; the mere
presence of individual binaries is insufficient.

Compilation and linking are intrinsic supported TensileLite functionality, so
the released package has one full-capability wheel-SDK contract and requires
`rocm[devel]`. There is no libraries-only, import-only, or logic-only supported
installation mode.

This is a capability requirement, not a claim that the literal `rocm[devel]`
extra is dependency-complete. Under TheRock's current package graph, a usable
installation still selects `libraries`, `devel`, and the applicable device
wheel because the production client/runtime libraries and per-ISA payload are
owned outside the devel distribution.

### G004 — How is compatibility version discovered without one ROCm root?

**Decision: Derived from G007 — read the selected root's `.info/version`.**

The previously proposed exact-publication requirement is deferred. TensileLite
validates only the base `X.Y.Z` compatibility line. Once G007 selects one root,
the existing `.info/version` read from that root is the uniform source for
filesystem, TheRock Python-SDK, and TheRock CI/artifact environments. No
manifest runtime dependency, package-manager query, or `rocm_sdk.__version__`
parsing is added for this comparison.

The runtime version-validation code itself already reads
`<resolved-root>/.info/version`. The required implementation change is only in
root resolution: on all platforms, first resolve an active-interpreter
`rocm_sdk` root when present, then explicit `ROCM_PATH`, then `/opt/rocm` on
non-Windows. The selected root's existing `.info/version` read then works
unchanged. `rocm_sdk.__version__` may appear in diagnostics but is not a
compatibility input.

Documentation must not claim that this comparison establishes identity of the
exact ROCm artifact publication. It establishes base-release compatibility.

### G005 — What is the canonical exact publication identity in each ROCm form?

**Decision: Deferred — exact publication identity is out of scope for now.**

**Discovery facts:**

- A TheRock Python SDK exposes exact publication identity through the public
  `rocm_sdk.__version__` API. This is a PEP 440 value such as
  `10.1.0a20260807` or `10.1.0.dev0+<sha>`.
- A TheRock CI artifact and extracted TheRock distribution expose exact
  publication identity as `rocm_package_version` in the official
  `share/therock/therock_manifest.json`; the CI build also injects that value as
  `THEROCK_PACKAGE_VERSION`.
- A conventional `/opt/rocm` installation's `.info/version` exposes only base
  compatibility. On this installed Debian ROCm example, `.info/version` is
  `7.2.4` and legacy `.info/version-rocm` is `7.2.4-93`, while the native
  `rocm`/`rocm-core` package-manager version is the more specific
  `7.2.4.70204-93~24.04`.

Thus all three environments can identify a publication exactly, but a generic
native `/opt/rocm` installation does not provide the same PEP 440 identity as a
TheRock wheel or CI artifact. Defining a cross-format canonicalization, adding
an installed publication marker, or carrying typed native/Python identities is
deferred. No current TensileLite behavior compares these values.

### G006 — How does a TheRock source build receive the base ROCm version?

**Decision: Accepted — build frontends supply one explicit base-version input.**

The current branch is hermetic but artifact-derived: it resolves
`THEROCK_TOOLCHAIN_ROOT`, then reads that staged root's `.info/version` while
configuring hipBLASLt. TheRock guarantees that file is available by the required
toolchain stage, but its superproject already owns the underlying authority:
top-level `TheRock/version.json` is parsed to base `X.Y.Z` and explicitly passed
to rocm-core as `-DROCM_VERSION=X.Y.Z`.

The proposed TheRock-native contract is therefore:

```text
TheRock/version.json
  -> namespaced THEROCK_ROCM_VERSION=X.Y.Z
  -> explicit CMAKE_ARGS on the hipBLASLt subproject
  -> release_metadata.py --rocm-version X.Y.Z
  -> one TENSILELITE_DISTRIBUTION_VERSION
  -> canonical wheel metadata and tensilelite-client version header
```

In TheRock mode, `release_metadata.py` validates/composes an explicitly supplied
base version and must not discover a staged root. Standalone builds retain their
selected `ROCM_PATH` and read `<root>/.info/version`. The installed-wheel
runtime separately uses its resolved SDK root for client location and base
compatibility validation.

`release_metadata.py` is a pure composer with inputs `VERSION` and a base ROCm
version string. `setup.py` and `compat/setup.py` obtain the latter from the
required, namespaced build input `TENSILELITE_ROCM_VERSION`; it is not a loose
or optional `ROCM_VERSION` override. Both canonical and compatibility wheel
builds require exactly this one ROCm-specific value.

TheRock derives `TENSILELITE_ROCM_VERSION` from its graph-owned
`THEROCK_ROCM_VERSION` value. Standalone CMake and Invoke derive it from the
already selected filesystem root's `.info/version`. CMake uses the same value
to generate the native client version header. A raw `pip wheel` invocation must
supply `TENSILELITE_ROCM_VERSION=X.Y.Z` or fail with an actionable diagnostic;
it does not infer a root itself.

### G007 — What is the installed-wheel ROCm-root resolution precedence?

**Decision: Accepted — active-interpreter TheRock SDK, explicit filesystem root,
Linux default, then failure.**

Installed-wheel runtime precedence is:

```text
1. active interpreter has rocm_sdk
   -> sys.executable -m rocm_sdk path --root

2. explicit ROCM_PATH
   -> use exactly that filesystem root

3. non-Windows default
   -> /opt/rocm

4. otherwise
   -> actionable failure
```

The active Python environment is authoritative when it contains a TheRock ROCm
SDK. This permits a venv-installed SDK to work even if the shell carries a stale
or unrelated `ROCM_PATH`. A broken selected source fails with its diagnostic; it
does not silently fall through to a lower-priority source. The wheel-SDK command
must use `sys.executable -m rocm_sdk`, never a bare `rocm-sdk` from `PATH`.

### G007A — How are TheRock source-build/codegen subprocesses kept graph-owned?

**Decision: Ignored — no demonstrated current collision.**

The installed-wheel precedence intentionally favors a Python SDK. TheRock
source-build/codegen commands instead require their graph-owned toolchain root,
even if the selected build Python happens to have `rocm_sdk` installed. Their
explicit build context must therefore bypass the installed-runtime resolver or
otherwise force the graph-selected root without reintroducing ambient discovery.

No checked-in current TheRock source-build path installs `rocm_sdk` into the
Python interpreter used for hipBLASLt/TensileLite code generation. `develop`
does not import the branch-added runtime resolver at all, and the current branch
only has a theoretical collision if a future source-build environment adds an
SDK wheel to that interpreter. Do not add a build-only root override now. Reopen
only with a concrete reproducer and a focused regression test.

### G008 — Is the selected ROCm root frozen and reused at runtime?

**Decision: Accepted — resolve once at import and reuse one private root.**

Package initialization resolves one ROCm root, validates its base release, and
freezes it for the process alongside the selected client. The same root supplies
all standard locations:

```text
<root>/.info/version
<root>/libexec/hipblaslt/tensilelite/tensilelite-client[.exe]
<root>/bin
<root>/lib/llvm/bin
```

Toolchain validation must use these root tool directories instead of independently
considering `ROCM_PATH`, `/opt/rocm`, and `PATH` as competing SDK sources. Do not
mutate global `ROCM_PATH` to communicate the selection.

An explicit per-installation client binding remains the one intentional exception:
it replaces only the frozen client executable path. It never changes the selected
ROCm root, its release validation, or the toolchain root. The custom client must
still report the installed TensileLite wheel version; no standard-client fallback
is attempted when that configured path is invalid or later disappears.

### G009 — How does the validator obtain the private frozen root?

**Decision: Accepted — a private `_runtime.rocm_root()` getter.**

`_runtime.initialize()` retains the frozen root and frozen client. A private
`_runtime.rocm_root()` getter returns the root after initialization and raises
the existing runtime-initialization diagnostic otherwise. `Validators.py` obtains
the root lazily through this getter when it constructs default search paths.

This keeps the runtime-selection interface small and private. Do not thread a
root argument through every `validateToolchain` caller, and do not mutate global
`ROCM_PATH` to communicate the selection. There is no public `RuntimeInfo` or
public root API.

### G010 — May a relative tool name fall back to `PATH` outside the selected root?

**Decision: Accepted — no; relative names are selected-root-only.**

An absolute tool path supplied by a caller is the explicit override and is
validated directly. A relative TensileLite tool name is resolved only below the
frozen root's `bin` and `lib/llvm/bin` directories. If absent, validation fails
with the selected root and expected paths in its diagnostic; it never falls back
to `PATH`, `/opt/rocm`, or another SDK.

The complete devel SDK is the required contract, so a missing relative tool is
an incomplete-install or packaging diagnosis, not a reason to silently mix
toolchains. TheRock's accepted console-script trampolines remain useful for
users and other consumers but are not a TensileLite toolchain fallback.

### G011 — How does `offload-arch` preserve AMD-only architecture detection?

**Decision: Accepted — bare `offload-arch` first, then `amdgpu-arch` fallback.**

Bare `offload-arch` works on the reviewed gfx942 host and produces the same
eight gfx942 lines as `amdgpu-arch`; the existing `gfxToIsa`/supported-ISA
filter already drops non-AMD output. Do not add `--only=amdgpu` or refactor the
detector interface solely to carry arguments.

The migration follows TheRock's current public SDK contract: it has exposed an
`offload-arch` trampoline since its 2025-10 compiler promotion, while
`amdgpu-arch` is a historical AMD-facing alias to the same current ROCm tool on
the reviewed system. Prefer the supported, vendor-neutral command that TheRock
ships in the active Python environment; retain the old name only to avoid
breaking traditional ROCm installations during the transition.

Default architecture discovery becomes an ordered operation rather than one
tool name: attempt `offload-arch`, then `amdgpu-arch` only if the first attempt
cannot yield a supported AMD ISA. Preserve the existing platform-specific
`rocm_agent_enumerator` handling as an additional fallback where it is currently
required (RHEL8/FFM); do not broaden this migration into a platform-policy
rewrite.

### G012 — Do runtime errors report the selected root and its mechanism?

**Decision: Accepted — record and report selected-root provenance.**

The private resolved-root result retains the selected root path, base version,
and mechanism (`active Python rocm_sdk`, explicit `ROCM_PATH`, or `/opt/rocm`).
Every compatibility diagnostic names the selected root and mechanism. Client
selection/validation failures include the same selected-root line, even when an
explicit custom client binding is the direct failing path.

This is private runtime state, not a public `RuntimeInfo` object or root API.
It makes mixed Python-SDK, environment, and system installations diagnosable
without changing selection behavior.

### G013 — How does D2 describe ROCm SDK ownership?

**Decision: Accepted — TensileLite is ROCm-coupled but pip may supply the SDK.**

D2 continues to require a matching complete ROCm SDK and continues to reject a
TensileLite wheel that vendors its own client/toolchain or creates an independent
portable release channel. Its rationale changes: TensileLite does not vendor a
ROCm SDK, but a matching SDK may be supplied either as a filesystem installation
or as TheRock's Python SDK with devel. Remove the inaccurate assertion that pip
cannot provide ROCm compiler tools or native components.

Associated documentation must describe compatibility as base `X.Y.Z` release
matching for the current scope, not identity of an exact nightly/dev/RC artifact
publication. Exact publication identity remains deferred by G005.

### G014 — Who owns the missing `clang-offload-bundler` trampoline?

**Decision: Accepted — TheRock owns the trampoline; TensileLite adds no shim.**

TheRock adds `clang-offload-bundler` to the `rocm-sdk-core` console-script
trampolines and verifies it in a fresh matching SDK venv. TensileLite resolves
the real binary from the frozen root's `lib/llvm/bin` directory and adds neither
a wrapper nor a `PATH` fallback. This keeps compiler-tool ownership and public
SDK behavior in TheRock while preserving one root-based TensileLite toolchain.

### G015 — How is an active-interpreter TheRock SDK detected and handled?

**Decision: Accepted — detect without importing, then use the public root command.**

Use `importlib.util.find_spec("rocm_sdk")` to determine whether the active
interpreter has a TheRock SDK. When absent, continue to the lower-priority
filesystem sources. When present, run
`sys.executable -m rocm_sdk path --root` and select the returned root.

A present SDK whose root command fails, times out, emits an empty path, or
returns a non-directory is a hard incomplete-SDK diagnostic, not a reason to
fall through to `ROCM_PATH` or `/opt/rocm`. The error identifies active Python
`rocm_sdk` as the selected-root source and directs the user to install the
matching complete `rocm[libraries,devel,device-...]` SDK.

### G016 — What focused validation proves the wheel-SDK contract?

**Decision: Accepted — extend existing unit/package suites plus one installed-SDK phase.**

- Extend `tensilelite/Tests/unit/test_rocm_runtime.py` for root precedence,
  broken-SDK no-fallback behavior, selected-root diagnostics, root/client
  freezing, and custom-client scope.
- Update the existing ToolchainValidators characterization that currently pins
  `ROCM_PATH -> /opt -> PATH`, and add a focused ordinary unit test for frozen-
  root-only relative tool resolution.
- Update the existing architecture characterization and add a focused unit test
  proving `offload-arch` first and `amdgpu-arch` fallback behavior.
- In TheRock, extend `rocm_sdk/tests/core_test.py` with the
  `clang-offload-bundler` console-script check and `rocm_sdk/tests/devel_test.py`
  with a readable `<root>/.info/version` assertion. Existing
  `test_rocm_wheels.yml` already installs `rocm[libraries,devel,device-*]` and
  runs these package tests; no new ROCm-wheel workflow is needed.
- Extend the existing TheRock TensileLite runner plus its runner-unit test with
  one phase that installs matching ROCm wheel SDK and canonical TensileLite wheel
  into the same fresh environment. The current reconstructed-artifact phase
  remains valuable but does not exercise Python-SDK-first resolution.

No comprehensive new CMake or GPU matrix is added. The installed-SDK phase uses
an explicit target and is GPU-less where possible.
