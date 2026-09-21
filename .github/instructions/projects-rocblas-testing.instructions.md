---
description: "rocBLAS Copilot code review: comment on missing or incomplete tests using projects/rocblas/TESTING.md"
applyTo: "projects/rocblas/**"
excludeAgent: "cloud-agent"
---

# rocBLAS Copilot code review — testing

## Purpose

When performing a code review of files under `projects/rocblas`, apply the testing strategy in `projects/rocblas/TESTING.md`. Comment on relevant test changes and on missing tests. Do not apply this guidance to other projects in the monorepo.

Treat `projects/rocblas/TESTING.md` as the source of truth. Prefer its coverage tables over general testing advice.

## When to comment

Consider adding a review comment when any of the following is true:

- Library, API, harness, YAML, CTest, or CI behavior changed and the matching validation from `TESTING.md` is missing.
- Tests were added but they skip required files, dispatch wiring, CMake/YAML registration, or the wrong test type for the change.
- A bug fix has no regression case that would fail without the fix.
- `known_bugs.yaml` quarantines a case with no tracking ticket or with no intent to fix.
- `gpu_arch` / `os_flags` appear in YAML but `type_filter()` does not go through `RocBLAS_Test<>::type_filter_functor`.
- GEMM / Tensile logic changed with no targeted `*gemm*` / `*_tensile` coverage called out.
- Performance-sensitive GEMM changed with no `rocblas-bench` spot-check note (there is no automated PR performance gate).
- Kernel algorithm or HMM tests changed (`category: HMM`, `HMM: true`, or `*HMM*`) and the PR does not have the `ci:extended` label.
- Stress tests changed (`category: stress` or `*stress*`) and the PR does not have the `ci:weekly` label.

Stay silent for license-header, clang-format, comment-only, and docs-only diffs, unless the change updates test tiers, CTest layout, or quarantine policy without updating `TESTING.md` or the matching `test_categories.yaml` / `rtest.xml`.  Do not nit-pick.

## Coverage expectations by change type

Use this table from `TESTING.md`. Comment when the expected validation is absent.

| Change type | Expected validation |
| --- | --- |
| New BLAS routine | `testing_*.hpp`, `*_gtest.cpp`, `*_gtest.yaml`, CMake registration, YAML included in `rocblas_gtest.yaml` |
| Bug fix | Regression gtest that fails without the fix |
| New public API / handle mode | Auxiliary gtest + YAML case |
| GEMM / Tensile logic | Rebuild Tensile; gemm / `*_tensile` filters; bench spot-check if performance-related |
| CI / CTest only | Update `test_categories.yaml` and `rtest.xml`; verify `ctest -N` |
| Packaging | Install-tree `ctest` from `bin/rocblas/` |
| Format / hooks only | No client tests required |

## Choosing the right test type

- **Bug fix** — regression test that fails before the fix.
- **GPU numerical BLAS behavior** — integration case in `*_gtest.yaml` plus `testing_*.hpp`.
- **Invalid arguments / status codes** — `testing_*_bad_arg` or a small dedicated gtest.
- **Handle / logging / stride APIs** — auxiliary gtest pattern (see existing `set_get_*` tests).
- **Performance** — `rocblas-bench` with representative sizes; results belong in the PR, not as a merge gate.
- **CTest / CI tier change** — `clients/gtest/test_categories.yaml`, `rtest.xml`, and `ctest -L`.

## Required pieces for a new data-driven suite

Flag the change if any of these are missing for a new or substantially extended routine:

1. `clients/include/.../testing_<fn>.hpp` harness (`Arguments`, host reference, host and device pointer modes, `UNIT_CHECK` / `NEAR_CHECK`).
2. `clients/gtest/<fn>_gtest.cpp` using `RocBLAS_Test<>`, `type_filter()` via `type_filter_functor` (not `return true`), and `INSTANTIATE_TEST_CATEGORIES`.
3. `clients/gtest/<fn>_gtest.yaml` parameter matrix with a `category` (`quick`, `pre_checkin`, `nightly`, `stress`, or `known_bug`).
4. YAML included from `clients/gtest/rocblas_gtest.yaml` and listed in `clients/gtest/CMakeLists.txt` so `rocblas_gtest.data` regenerates.
5. The `.cpp` added to the `rocblas-test` source list in CMake.

Walkthrough: `projects/rocblas/clients/gtest/README.md`.

## Test-change review checks

- YAML `gpu_arch` is an allowlist of arch-name suffixes (for example `942` matches `gfx942`). Empty means all architectures. These fields are ignored unless `type_filter()` uses `type_filter_functor`.
- Do not suggest host-only unit tests as a substitute for GPU client tests. There is no separate host-only unit binary; almost all `rocblas-test` cases need a GPU.
- Do not treat missing automated performance thresholds or missing ASAN-on-every-PR as defects. Those are documented gaps in `TESTING.md`.
- Prefer fixing a failure over widening filters or adding `known_bug` without a ticket.
- `*known_bug*` must stay excluded from normal smoke / pre-checkin / nightly runs.

## CI label suggestions

HMM and stress suites are outside default PR CI. Comment when the matching GitHub label from `TESTING.md` is missing:

| Test change | PR label |
| --- | --- |
| HMM tests (`category: HMM`, `HMM: true`, `*HMM*`) | `ci:extended` |
| Stress tests (`category: stress`, `*stress*`) | `ci:weekly` |

If both kinds of tests change, both labels are required. Do not treat these as TheRock PR-lane substitutes.

## How to phrase comments

- Point at the missing path (`testing_*.hpp`, `*_gtest.yaml`, CMake, filter, or bench note).
- Name the `TESTING.md` expectation (change type or test type) that is unmet.
- Suggest a concrete gtest filter or YAML category when possible, for example `*quick*<routine>*` or a `pre_checkin` YAML entry.
- Do not request overview-comment checklists, merge blocking, or emoji-only severity labels.
