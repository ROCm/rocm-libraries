# ROCm Wheel Runtime Grilling Decisions

## Purpose

This is the current decision record for TensileLite compatibility with ROCm
installed from TheRock wheels. It records the supported installation contracts,
not superseded root-discovery experiments.

## Current constraints

- A released TensileLite wheel supports a matching ROCm SDK supplied either as
  TheRock Python packages or as a conventional ROCm prefix.
- The Python SDK and conventional-prefix forms are separate adapters. They do
  not borrow roots, tools, or native payloads from each other.
- TensileLite's Python-SDK adapter uses the active interpreter's
  `rocm_sdk_core` metadata and console-script locations. It does not construct
  a synthetic ROCm root.
- A conventional-prefix adapter uses the selected root's `.info/version` and
  its root-relative tool locations.
- `tensilelite-client` is an optional benchmark/retune capability. Import,
  help, logic validation, and device-library generation do not request it.
- The currently packaged client is a non-Windows `blas_test` artifact. Its
  eventual production destination is `blas_lib` / `rocm[libraries]`.
- No manifest comparison, binary hashes, or component ABI handshake is added.

## Current findings

### F1 — TheRock has two distinct package roles

`rocm-sdk-devel` remains TheRock's complete development-prefix package: it
provides headers, CMake/pkg-config metadata, static or import libraries, and a
coherent `ROCM_PATH`-style tree. Its public `rocm_sdk path --root` API therefore
requires the devel package.

That fact does not make devel a TensileLite Python-SDK dependency. The
Python-SDK adapter consumes the public core package metadata and its
interpreter-local tool trampolines instead of requesting a root.

### F2 — Python-package and prefix version identities differ

TheRock Python packages expose the full publication identity through
`rocm_sdk_core.__version__`; this may be a nightly, RC, or development value.
A conventional prefix exposes only its base compatibility release through
`.info/version`.

The two adapters compare their respective identities without attempting to
canonicalize a Python package publication into a conventional-prefix package
manager version.

### F3 — Compiler tools are a core-package interface

TheRock's core wheel publishes the compiler, assembler, offload bundler, and
device-enumerator console scripts used by the Python-SDK adapter. In particular,
`amdclang`, `amdclang++`, `clang-offload-bundler`, `hipcc`, `hipconfig`, and
`offload-arch` are part of the public console-script surface.

The presence of those tools does not turn the core package into a general CMake
development prefix; it only defines the Python-SDK tool interface used here.

### F4 — The current artifact runner is not a Python-SDK integration test

TheRock's TensileLite artifact runner reconstructs a conventional ROCm prefix,
sets `ROCM_PATH`, and supplies raw rocisa through `PYTHONPATH`. It validates the
installed wheel against that prefix contract, but it does not prove the active
`rocm_sdk_core` package-adapter contract in a fresh wheel-only environment.

### F5 — TheRock source builds pass an explicit ROCm identity

TheRock passes both `THEROCK_ROCM_VERSION` and `THEROCK_PACKAGE_VERSION` to the
hipBLASLt subproject. The build frontend prefers the package identity when
available and falls back to the base version otherwise. Standalone builds read
the selected conventional root's `.info/version`.

## Decisions

### G001 — Are official ROCm wheel installations supported?

**Decision: Accepted — official ROCm wheel installations are supported.**

TensileLite supports a matching SDK installed through TheRock wheels as well as
a conventional prefix. The selected adapter owns compatibility validation, tool
resolution, and client lookup for the process.

### G002 — How do the Python-SDK and conventional-prefix adapters coexist?

**Decision: Accepted — use separate adapters with no path borrowing.**

For an active Python SDK installation, TensileLite reads
`rocm_sdk_core.__version__` and resolves known tools only from the active
interpreter's scripts directory and its user-script counterpart. It never falls
through to `ROCM_PATH`, `/opt/rocm`, or ambient `PATH` for one missing tool.

Without an active Python SDK, TensileLite selects a conventional prefix in this
order: explicit `ROCM_PATH`, `/opt/rocm` on non-Windows, then a root reported by
`hipconfig --rocmpath` on `PATH`. Its tools are resolved only below that selected
prefix.

### G003 — Does the Python-SDK contract require `rocm[devel]`?

**Decision: Accepted — no.**

The Python-SDK adapter does not call `rocm_sdk path --root`, expand the devel
archive, or use a physical package root. It requires the matching ROCm core,
libraries, and applicable device payload for the requested capability, but not
the devel package merely for import, compatibility validation, or code
generation.

`rocm[libraries,devel,device-*]` remains the correct installation for users who
need a general CMake/HIP development prefix.

### G004 — How is compatibility identity validated?

**Decision: Accepted — validate through the selected adapter's native identity.**

The Python-SDK adapter compares the wheel's ROCm identity with the active
`rocm_sdk_core.__version__`, including a supported nightly, RC, or development
publication suffix. The conventional-prefix adapter compares only the base
`X.Y.Z` value from the selected root's `.info/version`.

This preserves exact package identity where TheRock exposes it while keeping
the conventional-prefix compatibility boundary explicit.

### G005 — How does a TheRock source build receive the ROCm identity?

**Decision: Accepted — the build frontend passes one explicit value.**

TheRock's graph-owned input is passed into hipBLASLt and then to
`release_metadata.py` as `TENSILELITE_ROCM_VERSION`. The metadata helper
composes the canonical and compatibility-wheel versions, and CMake uses the same
result for the native-client version header when that client is enabled.

No installed-wheel runtime resolver participates in this source-build flow.

### G006 — How are relative tools resolved?

**Decision: Accepted — resolve only within the selected adapter.**

An absolute caller-supplied executable remains an explicit override. A relative
tool name is searched only in the selected Python-SDK script locations or the
selected conventional-prefix tool directories. A missing tool is an incomplete
SDK/package diagnostic, not permission to mix a second ROCm installation into
the process.

### G007 — Which device enumerator is preferred?

**Decision: Accepted — `offload-arch` first, then compatibility fallbacks.**

Use `offload-arch` first, then `amdgpu-arch` when the first command is absent,
fails, or yields no supported AMD ISA. Retain the existing
`rocm_agent_enumerator` fallback where platform policy already requires it.

### G008 — Who owns client packaging and tool trampolines?

**Decision: Accepted — TheRock owns both public package entry points.**

TheRock publishes the core tool trampolines; TensileLite adds no wrapper or
ambient-PATH fallback. The production client will be a BLAS libraries payload
with its own interpreter-local trampoline. Until that promotion lands, a Python
SDK client request fails clearly unless an explicit client binding is present.

### G009 — What runtime provenance is retained?

**Decision: Accepted — retain selected-installation provenance.**

TensileLite retains the selected adapter, its version identity, and its
executable search locations. Diagnostics name the selected Python SDK or
conventional prefix. A client-binding failure separately identifies the explicit
binding or adapter-provided client candidate that failed.

### G010 — What validation proves the wheel-SDK contract?

**Decision: Accepted — keep focused unit coverage and add one fresh Python-SDK phase.**

- Runtime tests cover Python-SDK versus conventional-prefix selection, exact
  package identity, base-prefix comparison, selected-tool lookup, and lazy
  client validation.
- Toolchain tests cover interpreter-script lookup, no path borrowing,
  `offload-arch` preference, and the compatibility fallbacks.
- TheRock package tests cover the published core tool trampolines.
- TheRock's TensileLite artifact runner keeps its reconstructed-prefix coverage
  and gains a separate fresh environment containing the matching
  `rocm[libraries,device-*]` package set and the canonical TensileLite wheel.
  That phase must exercise the active `rocm_sdk_core` adapter without devel.
