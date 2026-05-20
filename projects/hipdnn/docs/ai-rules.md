---
alwaysApply: true
---

# AI Rules for hipDNN Project

## Project AI Skills

Reusable AI skills for hipDNN live under `tools/ai/skills/`. The skills below describe what each one does and when to suggest it.

**To use a skill, follow this sequence — do not read `SKILL.md` and execute its steps directly.** That bypasses the install path the skill is meant to be invoked through, and misses the entire point of having skills.

1. **Check whether the skill is loaded in this session.** Look at the available-skills list in the session's system reminder. The skill being present under `tools/ai/skills/` is *not* enough — it's only invocable once linked into either `<workspace>/.claude/skills/` or `~/.claude/skills/` and surfaced in the session's skill list.
2. **If not loaded, ask the user which skill(s) to link and which scope** (`<workspace>/.claude/skills` for workspace-only, `~/.claude/skills` for user-global). Then run:
   ```
   python tools/ai/link-skills.py <target-skills-dir> <skill-name> [<skill-name> ...]
   ```
   Use `--list` in place of skill names to see what's available. The user must name each skill explicitly — there is no "link everything" mode, so a skill is never installed without consent. The linker uses symlinks on Linux/macOS, junctions on Windows, and is idempotent (existing entries pointing at the same source are skipped).
3. **Invoke via the Skill tool** (e.g. `/hipdnn-superbuild`). The new skill may not appear in the session's skill list until the next message — once it does, invoke it normally.

When a user asks for a workflow covered by a project skill, tell them the project has a matching skill and offer to install and invoke it.

- `tools/ai/skills/pr-summary/SKILL.md`
  - Drafts or revises new or existing pull request titles and bodies with hipDNN's preferred summary, risk, testing, and technical-change format.
  - Suggest this skill when the user asks for PR creation, PR body updates, PR summaries, risk summaries, testing sections, or review-ready PR descriptions.
- `tools/ai/skills/hipdnn-superbuild/SKILL.md`
  - Builds hipDNN together with one or more providers via the repository-root superbuild presets (`hipdnn-providers`, `miopen-provider`, `hipblaslt-provider`, `hip-kernel-provider`, `hipdnn-samples`, etc.), in a single CMake invocation. On Windows it auto-runs the wheel-based ROCm setup when no SDK path is supplied.
  - Suggest this skill when the user asks to build hipDNN with providers, run a superbuild preset, rebuild after a rebase or merge, or set up a fresh build from the repo root. Prefer it over the standalone build whenever providers are involved.
- `tools/ai/skills/hipdnn-superbuild-test/SKILL.md`
  - Runs tests against an existing superbuild with per-component selection (`hipdnn`, `miopen`, `hipblaslt`, `hip-kernel`, `integration-tests`, or `all`), unit/integration scope, optional `--filter=<gtest_pattern>`, `--verbose`, and `--keep-going`. Handles Windows DLL PATH and the `hip-kernel-provider` target naming quirk automatically.
  - Suggest this skill when the user asks to run, filter, or triage tests against a superbuild they have already configured. It does not configure or build — pair it with `/hipdnn-superbuild` first.

## Project Overview & Architecture

hipDNN is a graph-based deep learning library for AMD GPUs with a plugin-based architecture.

### Component Architecture (Do Not Modify Boundaries (linkage))
| Component | Type | Links To | Purpose |
|-----------|------|----------|---------|
| **Backend** (`backend/`) | Shared library (C API) | Data SDK | Core engine, plugin loading, graph execution |
| **Frontend** (`frontend/`) | Header-only C++ | Backend, Data SDK | User-friendly wrapper around backend C API (uses Data SDK for types/logging, not FlatBuffers) |
| **Data SDK** (`data_sdk/`) | Header-only | (none) | Shared data types, logging |
| **FlatBuffers SDK** (`flatbuffers_sdk/`) | Header-only | FlatBuffers, nlohmann_json | FlatBuffer schemas, generated headers, JSON helpers |
| **Plugin SDK** (`plugin_sdk/`) | Header-only | Data SDK | Interfaces for plugin development |
| **Test SDK** (`test_sdk/`) | Header-only | Data SDK | Shared test utilities |

---

## Building & Testing

### Windows Build Environment

On Windows, run `scripts/windows/wheel_build_setup.ps1` to fetch the latest ROCm SDK wheels and set up the build environment:

```powershell
.\scripts\windows\wheel_build_setup.ps1                         # Latest nightlies
.\scripts\windows\wheel_build_setup.ps1 -SHA "<commit-sha>"     # Specific S3 staging build
```

The script creates a Python venv, installs ROCm wheels, initializes the SDK, and prints the CMake variables (`CMAKE_PREFIX_PATH`, `CMAKE_PROGRAM_PATH`) needed for the build.

### Build Approaches

Choose the build approach based on what is being changed:

### 1. hipDNN Only (changes only within `projects/hipdnn/`)

```bash
cd <workspace>/projects/hipdnn
mkdir -p build && cd build
cmake -GNinja ..

ninja              # Build everything
ninja check        # Build and run ALL tests
ninja unit-check   # Unit tests only (faster)
ninja integration-check  # Integration tests only
ninja doxygen      # Generate Doxygen docs (output: build/docs/html/)
```

### 2. hipDNN + Providers via Superbuild (recommended when building providers)

Use the superbuild when building hipDNN together with one or more providers. See `docs/Superbuild.md` for full details.

```bash
cd <workspace>
cmake --preset <preset-name>
cmake --build build
```

Available presets (from the repository root `CMakePresets.json`):

| Preset | Components Built |
|--------|-----------------|
| `hipdnn` | hipDNN only |
| `miopen-provider` | hipDNN + miopen-provider |
| `hipblaslt-provider` | hipDNN + hipblaslt-provider |

Or manually specifying components:
```bash
cmake -B build -GNinja -DROCM_LIBS_ENABLE_COMPONENTS="hipdnn;miopen-provider;hipblaslt-provider" .
cmake --build build
```

In the superbuild, targets are prefixed with the project name (e.g., `hipdnn-check`, `miopen-provider-unit-check`).

### 3. Standalone Provider Build (fallback — provider not in superbuild)

If a provider is not part of the superbuild, build and install hipDNN first, then build the provider with `CMAKE_PREFIX_PATH` pointing to the hipDNN install:

```bash
# Step 1: Build and install hipDNN
cd <workspace>/projects/hipdnn
mkdir -p build && cd build
cmake -GNinja -DCMAKE_INSTALL_PREFIX=<hipdnn-install-path> ..
ninja && ninja install

# Step 2: Build the provider against the hipDNN install
cd <workspace>/dnn-providers/<provider-name>
mkdir -p build && cd build
cmake -GNinja -DCMAKE_PREFIX_PATH=<hipdnn-install-path> ..
ninja
```

### Test Binaries in `build/bin/`
| Binary | Tests | Typical Use |
|--------|-------|-------------|
| `hipdnn_backend_tests` | Backend unit tests | Core backend logic |
| `hipdnn_frontend_tests` | Frontend unit tests | Frontend wrapper logic |
| `hipdnn_data_sdk_tests` | Data SDK unit tests | Flatbuffer objects, utilities |
| `hipdnn_plugin_sdk_tests` | Plugin SDK unit tests | Plugin interface utilities |
| `hipdnn_test_sdk_tests` | Test SDK unit tests | Test utility functions |
| `hipdnn_public_backend_tests` | Backend API tests | Public C API black-box tests |
| `hipdnn_public_frontend_tests` | Frontend integration tests | E2E frontend tests |

### Running Specific Tests
Use `--gtest_filter` for fast iteration:
```bash
./build/bin/hipdnn_backend_tests --gtest_filter="TestBackendDescriptor.*"
```

### When Modifying Code
**Only build or run tests if explicitly requested in the user's prompt.** Do not proactively run `ninja`, `ninja check`, or test binaries unless asked.

When requested to build/test:
1. Rebuild with `ninja` (from build directory)
2. Run relevant tests using `--gtest_filter` for fast iteration

---

## C++ Code Style

### Naming Conventions
- CamelCase for class/struct names (e.g., `BatchNormTestCase`, `SimpleTensorBundle`)
- camelCase for functions, variables, and private members (e.g., `setupEnvironment()`, `tensorData`)
- Underscore prefix for private members (e.g., `_handle`, `_testData`)

### File Headers
- Copyright header on all source files: `// Copyright © Advanced Micro Devices, Inc., or its affiliates.` / `// SPDX-License-Identifier:  MIT`
- `#pragma once` immediately after copyright in header files (.h, .hpp)

### Code Practices
- Use `auto` with casts to avoid type duplication: `auto tensor = static_cast<float*>(data)`
- Use `auto` when initializing variables, unless the type is not obvious
- **Avoid implicit casts** — use explicit `static_cast<>`. The codebase compiles with `-Wconversion` and `-Wsign-conversion`
- Always use braces for if/for/while bodies, even single-line
- Use CMake for managing C/C++ dependencies
- Use Flatbuffers for serialization needs (backend/data_sdk only; frontend uses the backend C API)

### RAII & Resource Management

**All owning raw pointers must be wrapped in RAII (smart pointers) immediately after acquisition.** Never rely on manual `delete` — if an assertion, exception, or early return occurs between allocation and `delete`, the resource leaks. This project runs under AddressSanitizer (ASAN) in CI, and any leak fails the build.

**Common patterns and pitfalls:**

| Pattern | Wrong | Right |
|---------|-------|-------|
| FlatBuffers Graph unpacking (backend/data_sdk) | `auto obj = graph->UnPack();` (raw `GraphT*`) | `auto obj = UnPackGraph(serialized.ptr);` (returns `unique_ptr<GraphT>`) |
| FlatBuffers `UnPack()` (backend/data_sdk) | `auto obj = table->UnPack();` (raw `T*`) | `auto obj = std::unique_ptr<T>(table->UnPack());` |
| `getAttribute()` for tensor arrays | Use raw pointer, forget cleanup | Wrap immediately: `auto owned = std::unique_ptr<HipdnnBackendDescriptor>(rawPtr);` |
| `createDescriptorPtr<T>()` | Store raw pointer | Use `createDescriptor<T>()` which returns `unique_ptr` |

**FlatBuffers `UnPack()` returns owning raw pointers (backend/data_sdk only).** The generated `Graph::UnPack()`, `Node::UnPack()`, etc. all return `T*` allocated with `new`. Prefer the generated helpers like `UnPackGraph()` which return `std::unique_ptr<T>` directly. For types without helpers, wrap manually: `std::unique_ptr<T>(table->UnPack())`.

**`getAttribute()` with `HIPDNN_TYPE_BACKEND_DESCRIPTOR` allocates new descriptors.** The C API packs shared descriptors into fresh `HipdnnBackendDescriptor*` via `packDescriptor()`. Ownership transfers to the caller — wrap in `std::unique_ptr<HipdnnBackendDescriptor>` immediately after retrieval.

### Testing
- Use Google Test (gtest) framework for all C/C++ tests
- Never generate a `main()` function in test files — gtest provides its own
- Use TEST(), TEST_F(), TEST_P(), or TYPED_TEST() macros as appropriate
- **Prefer TYPED_TEST for multi-datatype tests** — when testing across float, half, bfloat16, use TYPED_TEST instead of duplicating test code

#### Test Suite Naming

Rules apply to the TestSuite name (first param of `TEST` / `TEST_F` / `TEST_P`). TestCase (second param) should be PascalCase.

**Composition (left → right):**

1. Required `Test` (unit tests) or `Integration` (integration tests) prefix, always first
2. Optional `Gpu` immediately after `Test`/`Integration` if the test needs GPU support
3. Core Feature / Subject under test (PascalCase, no underscores)
4. Optional Datatype token: `Bfp16`, `Fp16`, `Fp32`

Omit any optional position that does not apply.

**Unit tests**: Mirror the class under test — `TestMyClass` or `TestGpuMyClass` if GPU is required.

**Valid examples:**
```
IntegrationGpuConvolutionPlannerNchwFp32   TestGpuActivationKernelNchwFp32
TestGpuExecutionPlanBuilderFp32            IntegrationGraphFusion
TestConvolutionHeuristicsFp32              TestConvolutionHeuristics
```

#### Choosing Between TYPED_TEST and TEST_P

| Scenario | Approach |
|----------|----------|
| Single type, no params | `TEST()` / `TEST_F()` |
| Single type, with params | `TEST_P()` |
| Multi-type, no params | `TYPED_TEST` |
| Multi-type, with params | `TEST_P` + multi-declarations |

**Key Principle:** Prefer `TEST_P` over `TYPED_TEST` when both type variation and parameterized cases are needed. `TYPED_TEST` and `TEST_P` don't mix well together.

**Multi-declarations**: Use explicit type aliases (`using ConvTestFp32 = ConvTest<float>`) with separate `TEST_P` and `INSTANTIATE_TEST_SUITE_P` per type. Prefer this over macros for debuggability.

**TypePair pattern**: For testing type combinations (e.g., input + compute type), define a `TypePair<T1, T2>` struct with `InputType`/`ComputeType` aliases and use with `TYPED_TEST_SUITE`.

---

## Conditional Guidelines

When modifying public API headers (under `backend/include/`, `frontend/include/hipdnn_frontend/`, `plugin_sdk/include/`), add Doxygen comments (`/** @brief ... */` for classes/functions/files, `///<` for enum values) to any new or changed public API. Exclude `detail/` subdirectories and generated files. For full style details, see `docs/doxygen-guidelines.md`.
