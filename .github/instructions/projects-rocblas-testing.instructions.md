---
description: "rocBLAS Copilot code review: comment on missing or incomplete tests using projects/rocblas/TESTING.md"
applyTo: "projects/rocblas/**"
excludeAgent: "cloud-agent"
---

# rocBLAS Copilot code review — testing

## Purpose

When performing a code review of files under `projects/rocblas`, read `projects/rocblas/TESTING.md` and apply the named tables and lists below as written. Do not copy or paraphrase them here. Do not apply this guidance to other projects in the monorepo.

## Named tables and lists in TESTING.md

- **Table: `Coverage Expectations by Change Type`** — comment when the expected validation is absent.
- **List: `Choosing the Right Test Type`** — comment when the wrong test type is used for the change.
- **Table: `CI Label Suggestions`** — comment when a matching GitHub PR label is missing.

## When to comment

Consider adding a review comment when any of the following is true:

- Library, API, harness, YAML, CTest, or CI behavior changed and the matching validation from Table `Coverage Expectations by Change Type` is missing.
- Tests were added but they skip required files, dispatch wiring, CMake/YAML registration, or the wrong test type from List `Choosing the Right Test Type`.
- A change matches a row in Table `CI Label Suggestions` and the PR does not have that label.
- A bug fix has no regression case that would fail without the fix.
- `known_bugs.yaml` quarantines a case with no tracking ticket or with no intent to fix.
- `gpu_arch` / `os_flags` appear in YAML but `type_filter()` does not go through `RocBLAS_Test<>::type_filter_functor`.
- GEMM / Tensile logic changed with no targeted `*gemm*` / `*_tensile` coverage called out.
- Performance-sensitive GEMM changed with no `rocblas-bench` spot-check note (there is no automated PR performance gate).

Stay silent for license-header, clang-format, comment-only, and docs-only diffs, unless the change updates test tiers, CTest layout, or quarantine policy without updating `TESTING.md` or the matching `test_categories.yaml` / `rtest.xml`.  Do not nit-pick.

## Required pieces for a new data-driven suite

Flag the change if any of these are missing for a new or substantially extended routine:

1. `clients/include/.../testing_<fn>.hpp` harness (`Arguments`, host reference, host and device pointer modes, `UNIT_CHECK` / `NEAR_CHECK`).
2. `clients/gtest/<fn>_gtest.cpp` using `RocBLAS_Test<>`, `type_filter()` via `type_filter_functor` (not `return true`), and `INSTANTIATE_TEST_CATEGORIES`.
3. `clients/gtest/<fn>_gtest.yaml` parameter matrix with a `category` (`quick`, `pre_checkin`, `nightly`, `HMM`, `multi_gpu`, `stress`, or `known_bug`).
4. YAML included from `clients/gtest/rocblas_gtest.yaml` and listed in `clients/gtest/CMakeLists.txt` so `rocblas_gtest.data` regenerates.
5. The `.cpp` added to the `rocblas-test` source list in CMake.

Walkthrough: `projects/rocblas/clients/gtest/README.md`.

## Test-change review checks

- YAML `gpu_arch` is an allowlist of arch-name suffixes (for example `942` matches `gfx942`). Empty means all architectures. These fields are ignored unless `type_filter()` uses `type_filter_functor`.
- Do not suggest host-only unit tests as a substitute for GPU client tests. There is no separate host-only unit binary; almost all `rocblas-test` cases need a GPU.
- Do not treat missing automated performance thresholds or missing ASAN-on-every-PR as defects. Those are documented gaps in `TESTING.md`.
- Prefer fixing a failure over widening filters or adding `known_bug` without a ticket.
- `*known_bug*` must stay excluded from normal smoke / pre-checkin / nightly runs.

## How to phrase comments

- Point at the missing path (`testing_*.hpp`, `*_gtest.yaml`, CMake, filter, label, or bench note).
- Name the `TESTING.md` table or list that is unmet (`Coverage Expectations by Change Type`, `Choosing the Right Test Type`, or `CI Label Suggestions`).
- Suggest a concrete gtest filter or YAML category when possible, for example `*quick*<routine>*` or a `pre_checkin` YAML entry.
- Do not request overview-comment checklists, merge blocking, or emoji-only severity labels.
