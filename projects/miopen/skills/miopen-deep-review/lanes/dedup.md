# Lane: de-duplication (`DUP`)

Find code the change could reuse instead of writing, and code it repeats within itself.

## Against existing code

For each new or changed function, helper, fixture, CMake function or script, look for an existing equivalent in the code units
the change already belongs to or consumes, before reporting none. Search by name, signature and body, not by file listing. Scope is
bounded: the translation unit and its target first, then the same project, then the libraries that target already links (CK,
rocBLAS, hipBLASLt). Do not search other projects or units the change does not consume, and do not propose a new dependency to
reach a helper.

- The same translation unit: nearly identical blocks, functions or test bodies; merge them there when possible (a shared helper,
  a template, a parameterized test) before looking further out.
- Solver scaffolding: `IsApplicable` checks, perf-config search and invoker factories that an existing solver of the same family,
  a base class (`projects/miopen/src/include/miopen/conv/solvers.hpp` and the other `*/solvers.hpp`), `generic_search.hpp`, or the
  shared helpers under `projects/miopen/src/include/miopen/solver/` already provide; kernel-build helpers in `hip_build_utils`
  and `kernel_build_params`.
- Test infrastructure: `projects/miopen/test/tensor_holder.hpp`, `verify.hpp`, `random.hpp`, `workspace.hpp`, the CPU references
  (`projects/miopen/test/cpu_*.hpp`), `projects/miopen/test/gtest/gtest_common.hpp` (device checks, environment guards),
  `compare_helper.hpp` and the `*_common.hpp` fixtures beside each op's tests.
- CMake functions the project already uses (`add_gtest` and `add_gtest_negative_filter` in `projects/miopen/test/gtest/CMakeLists.txt`,
  `add_test_executable` in `projects/miopen/test/CMakeLists.txt`, `miopen_apply_ck_compile_definitions`).
- Kernels: a new `.cl`, `.cpp` or `.s` kernel that differs from an existing one by a compile-time parameter the solver could pass.

## Within the change

- Near-identical functions, test bodies or CMake blocks differing by a type, value or name: one parameterized form.
- Copy-pasted files across data types, directions (forward, backward data, backward weights) or architectures.
- The same constant or table defined in more than one place.

## Bar

Report only when the shared form is simpler than the copies and the copies must change together. Similar code that will diverge
is not duplication. Cite both locations and show the shared form. Report the top 5 candidates, ranked by how much code the shared
form removes; each states the **touched by this change** test: what in the diff creates or extends the duplication (a copy that
exists unchanged at `$BASE` is `FUTURE WORK`). File the rest as one theme finding listing the locations. Tier per [../policy.md](../policy.md) *Tiers* only (`SUGGESTION`; `IMPORTANT` when the copies already diverge or this change edits one copy and not the others). Copy-pasted tests that match [../policy.md](../policy.md) *Test smells* are `TESTQ`'s: hand them off (Handoffs: `TESTQ: <locations>`) instead of filing them.
