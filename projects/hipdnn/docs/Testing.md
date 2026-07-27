# hipDNN Testing

This document is the hub for hipDNN's testing approach: how to run tests, how the ctest/check targets relate, what you need to build, and what is expected of tests you write.

## Running Tests

Both the superbuild and the standalone build follow the same three-step flow using cmake presets:

- `cmake --preset <configure-preset>`
- `cmake --build <binaryDir>`
- `ctest --test-dir <binaryDir> --output-on-failure` (or a `ninja` check target)

### Superbuild (root `CMakePresets.json`)

Run from the repository root. binaryDir is `build`. The toolchain is baked into these presets; hipDNN developers may prefer to override it with the hipDNN toolchain (see [Building § Superbuild](./Building.md#superbuild)).

hipDNN-relevant configure presets:

| Preset | Components enabled |
|--------|--------------------|
| `hipdnn` | core only |
| `miopen-provider` | core + integration-tests + MIOpen provider |
| `hipblaslt-provider` | core + integration-tests + hipBLASLt provider |
| `hip-kernel-provider` | core + integration-tests + HIP-kernel provider |
| `hipdnn-providers` | core + integration-tests + MIOpen + hipBLASLt providers |
| `hipdnn-providers-all` | `hipdnn-providers` + HIP-kernel provider |
| `hipdnn-samples` | providers + samples |
| `hipdnn-dev-all` | everything (all providers + integration-tests + samples) |

> [!IMPORTANT]
> Root-level `ctest` is gated by `ROCM_LIBS_ENABLE_ROOT_CTEST`, which defaults **OFF** and is not set by any checked-in preset. Enable it, otherwise `ctest --test-dir build` sees no tests. Set it with `-DROCM_LIBS_ENABLE_ROOT_CTEST=ON` at configure time, or in the environment before the first (or a fresh) configure; once configured, the value is stored in the CMake cache and reused.

```bash
# From the repository root
cmake --preset hipdnn-dev-all -DROCM_LIBS_ENABLE_ROOT_CTEST=ON
cmake --build build
ctest --test-dir build # --output-on-failure is optional
```

Equivalently, set it in the environment. The variable is only read on a first or fresh configure, so use `--fresh` to force it to take effect on an already-configured build directory:

```bash
# From the repository root
export ROCM_LIBS_ENABLE_ROOT_CTEST=ON
cmake --preset hipdnn-dev-all --fresh
cmake --build build
ctest --test-dir build # --output-on-failure is optional
```

Superbuild ninja check targets are prefixed with `hipdnn-` (the bare, unprefixed aliases are not created in superbuild):

- `hipdnn-check`
- `hipdnn-quick-check`
- `hipdnn-unit-check`
- `hipdnn-integration-check`
- each also has a `-verbose` variant (e.g. `hipdnn-check-verbose`)

### Standalone (`projects/hipdnn/CMakePresets.json`)

Run from the `projects/hipdnn/` directory.

> [!NOTE]
> The standalone build defaults to the `cmake/ClangToolChain.cmake` toolchain, which auto-detects the ROCm Clang compiler from your PATH (via `hipconfig`). If ROCm is not on your PATH, point CMake at it with `-DROCM_CMAKE_PATH=<rocm-root>`. See [Building § ROCM_PATH, ROCM_CMAKE_PATH, and CMAKE_INSTALL_PREFIX](../Building.md#rocm_path-rocm_cmake_path-and-cmake_install_prefix) for the full toolchain-discovery details.

```bash
# From projects/hipdnn/
cmake --preset debug
cmake --build build/debug # binaryDir is `build/debug`
ctest --test-dir build/debug # --output-on-failure is optional
```

Standalone creates **unprefixed** aliases in addition to the prefixed targets:

- `check` (alias of `hipdnn-check`)
- `quick-check`
- `unit-check`
- `integration-check`
- plus the prefixed `hipdnn-*-check` names and their `-verbose` variants

### Address Sanitizer

> [!IMPORTANT]
> ASAN builds require a ROCm / TheRock build that was itself compiled with address-sanitizer support. Most developers do not have one; obtaining it means rebuilding TheRock/ROCm, which is out of scope here. ASAN checks are currently a **manual process** run by developers who have such a build; there is no ASAN coverage in CI yet (planned for the future). The steps below are for developers who **do** have an ASAN-capable ROCm build.

Add `-DBUILD_ADDRESS_SANITIZER=ON` to the configure step of whichever path, then run the normal check target or `ctest`:

```bash
# Superbuild example
cmake --preset hipdnn-dev-all -DROCM_LIBS_ENABLE_ROOT_CTEST=ON -DBUILD_ADDRESS_SANITIZER=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

> [!NOTE]
> Some HIP-related tests may be skipped due to AddressSanitizer incompatibility.

## ctest vs. check targets

The key difference is what each one does with the test binaries:

- **A check target (`ninja hipdnn-check`) builds the tests first, then runs them.** After editing source, it recompiles what changed and runs the result, so you never test a stale binary. This is the everyday choice while developing.
- **`ctest` runs already-built tests and does not build anything.** It is faster when nothing has changed, but you must have built the tests first, and it gives full control over which tests run and how.

Use a check target when you want to build-and-run a canned scope in one step. Use `ctest` directly when the tests are already built and you want to filter, parallelize, or repeat the run (see the flags below).

Each check target runs a fixed `ctest` invocation:

| ninja target (superbuild) | Equivalent ctest command |
|---------------------------|--------------------------|
| `hipdnn-check` | `ctest --output-on-failure` |
| `hipdnn-quick-check` | `ctest -L quick --output-on-failure` |
| `hipdnn-unit-check` | `ctest -L unit --output-on-failure` |
| `hipdnn-integration-check` | `ctest -L integration --output-on-failure` |

Check targets **hard-code their flags**; there is no pass-through. Use a raw `ctest --test-dir <binaryDir>` invocation when you need:

- `-R <regex>` - filter (include) tests by name
- `-E <regex>` - filter (exclude) tests by name
- `-L <regex>` - filter (include) tests by label (category)
- `-LE <regex>` - filter (exclude) tests by label (category)
- `-j <N>` - run tests in parallel
- `--repeat until-fail:<N>` - flake hunting

> [!TIP]
> Running `ctest --test-dir <binaryDir>` directly drops the check target's baked-in `-L <category>`. To reproduce a category's filtering, re-add the label yourself (e.g. `ctest --test-dir build -L quick -j 8` matches `hipdnn-quick-check` plus parallelism).

## What you need to build

hipDNN **core tests need no real provider.** Every plugin the core suites load is an in-tree fake from `tests/test_plugins/` (linking only the SDKs + `hip::host`), wired in via `add_dependencies`. A provider-free build (standalone or the core-only `hipdnn` superbuild preset) fully exercises the core suites.

Real-provider and cross-project integration suites (`hipdnn_integration_tests`, `miopen_plugin_*`) live under `dnn-providers/`. They are opt-in components and are not required to run core tests.

Guidance:

| Goal | Use |
|------|-----|
| Full validation (all providers + integration + samples) | `hipdnn-dev-all` superbuild preset |
| Everyday core work | `hipdnn` (core-only superbuild preset) |
| Minimal / offline core loop | standalone (`cmake --preset debug`) |

## Quick Reference

### Test Organization

| Component | Test Location | Type |
|-----------|--------------|------|
| Backend | `backend/tests/` | Unit tests |
| Frontend | `frontend/tests/` | Unit tests |
| Data SDK | `data_sdk/tests/` | Unit tests |
| FlatBuffers SDK | `flatbuffers_sdk/tests/` | Unit tests |
| Plugin SDK | `plugin_sdk/tests/` | Unit tests |
| Test SDK | `test_sdk/tests/` | Unit tests |
| Backend logging | `tests/backend_logging/` | Unit tests |
| API | `tests/backend/` | Black box API tests |
| Frontend integration | `tests/frontend/` | Integration tests |

> [!NOTE]
> Provider tests live under `dnn-providers/` and are built only when the corresponding provider component is enabled.

### Test Categories

Categories are defined in `test_categories.yaml`. There are six:

- `quick` - pattern matches all tests
- `unit` - selects named unit suites directly
- `integration` - selects named integration suites directly
- `standard`, `comprehensive`, `full` - no direct patterns; they inherit `quick`'s tests via tier labels

The tiers are cumulative: `standard` includes everything in `quick`, `comprehensive` includes everything in `standard`, and `full` includes everything in `comprehensive`. `unit` and `integration` are not tiers; they select their own named suites independently.

> [!NOTE]
> Today the higher tiers add no tests of their own, so `quick`, `standard`, `comprehensive`, and `full` all run the same set. The distinction exists for when higher tiers gain their own tests later.

### Choosing Between TYPED_TEST and TEST_P

```
Need to test across multiple data types (float, half, bfloat16)?
│
├── NO → Use TEST() or TEST_F()
│
└── YES → Also need parameterized test cases?
          │
          ├── NO → Use TYPED_TEST
          │        (Compile-time type safety, simple)
          │
          └── YES → Use TEST_P with multi-declarations
                    (Handles both types AND parameters)
```

**Key Principle:** Prefer `TEST_P` over `TYPED_TEST` when both type variation and parameterized cases are needed. `TYPED_TEST` and `TEST_P` don't mix well together.

| Scenario | Approach |
|----------|----------|
| Single type, no params | `TEST()` / `TEST_F()` |
| Single type, with params | `TEST_P()` |
| Multi-type, no params | `TYPED_TEST` |
| Multi-type, with params | `TEST_P` + multi-declarations |

### Multi-Declaration Pattern (for types + parameters)

When you need both type variation AND parameterized tests, use explicit type aliases with `TEST_P`:

```cpp
template <typename DataType>
class ConvTest : public ::testing::TestWithParam<ConvTestCase> { };

// Explicit type aliases - one per type
using ConvTestFp32 = ConvTest<float>;
using ConvTestFp16 = ConvTest<half>;
using ConvTestBfp16 = ConvTest<hip_bfloat16>;

// Separate TEST_P for each type
TEST_P(ConvTestFp32, Correctness) { runTest(); }
TEST_P(ConvTestFp16, Correctness) { runTest(); }
TEST_P(ConvTestBfp16, Correctness) { runTest(); }

// Separate instantiation for each type
INSTANTIATE_TEST_SUITE_P(Smoke, ConvTestFp32, testing::ValuesIn(getCases()));
INSTANTIATE_TEST_SUITE_P(Smoke, ConvTestFp16, testing::ValuesIn(getCases()));
INSTANTIATE_TEST_SUITE_P(Smoke, ConvTestBfp16, testing::ValuesIn(getCases()));
```

**Why multi-declarations over macros?** Modern tooling makes boilerplate easy to handle, and avoiding macros makes tests easier to debug and understand.

### Type Combinations with TypePair

For testing type combinations (e.g., input type + compute type), define a struct to hold the types:

```cpp
template <typename T1, typename T2>
struct TypePair
{
    using InputType = T1;
    using ComputeType = T2;
};

using TypeCombinations = ::testing::Types<TypePair<float, float>,
                                          TypePair<half, float>,
                                          TypePair<hip_bfloat16, float>>;
TYPED_TEST_SUITE(MyTypedTest, TypeCombinations);

TYPED_TEST(MyTypedTest, Correctness)
{
    using InputType = typename TypeParam::InputType;
    using ComputeType = typename TypeParam::ComputeType;
    // Test implementation using InputType and ComputeType
}
```

### Testing Requirements

- **Coverage Target**: 80% overall is the goal, with each component aiming for >80% individually (a target, not machine-enforced)
- **GPU Tests**: Must be marked with the `SKIP_IF_NO_DEVICES()` macro
- **Platform Support**: All tests must work on Windows and Linux
- **Performance**: Unit tests must execute quickly
- **CI**: All CI pipelines must pass on every PR

## Expectations During Development

Tests are a merge gate, not an afterthought. The following expectations apply to every PR.

**MUST:**

- Defect fixes ship a regression test proven to fail before / pass after the fix - locks the bug so it can't silently return.
- Product-code changes carry a test, a safe-default flag, or a written waiver - no behavior change ships unverified.
- Never disable/skip/weaken a test to green CI (no waiver) - greening by removing coverage hides real failures.
- Tests must assert real behavior - a test no source change could break is coverage padding, not coverage.
- ASAN/leak-clean - sanitizer checks are run manually today (CI coverage is planned); write RAII / failure-path cleanup so a leak does not surface later.
- PR body carries an honest Testing Summary + Checklist - `[x]` only for validation that actually passed, with the exact command.

**SHOULD:**

- Record the "why" for non-obvious tolerances or parameter sets - prevents unmaintainable orphaned tests.
- Follow existing test naming/placement so the suite actually picks the test up.
- Provider/support-surface changes (`dnn-providers/**`) need multi-arch coverage - a family that builds but skips tests is uncovered, not covered.
- Verify negative/error paths - unsupported op/layout/dtype combinations must fail predictably, not run a wrong path.

## Testing Strategy

To understand **how to design and create tests** in hipDNN (what to test, at which layer, and to what coverage), the authoritative reference is:

- **[Testing Strategy](./testing/TestingStrategy.md)** - the primary guide for contributors writing tests: white-box unit, black-box API, and end-to-end integration testing; per-component test categories; GPU-skip rules; coverage expectations; and the performance-testing roadmap.

The Quick Reference patterns above (TYPED_TEST / TEST_P, TypePair) are the day-to-day authoring shortcuts; the strategy doc is the fuller treatment. Contributors should read it before adding tests.

## Release / Milestone Verification

The following docs are for **hipDNN milestone and release verification**: validating a release build and capturing evidence for sign-off. They are **not** part of the per-PR development workflow; day-to-day contributors do not need them (see [Expectations During Development](#expectations-during-development) above for the PR-time bar).

- [Test Plan](./testing/TestPlan.md) - the release verification checklist and expected results.
- [Test Run Template](./testing/TestRunTemplate.md) - template for recording a release validation run and capturing evidence.
