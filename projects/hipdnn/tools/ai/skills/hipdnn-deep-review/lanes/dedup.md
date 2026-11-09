# Lane: de-duplication (`DUP`)

Find code the change could reuse instead of writing, and code it repeats within itself.

## Against existing code

For each new or changed function, helper, fixture, CMake function or script, look for an existing equivalent in the code units
the change already belongs to or consumes, before reporting none. Search by name, signature and body, not by file listing. Scope is
bounded: the translation unit and its target first, then the same project, then the SDKs that target links (`test_sdk`,
`data_sdk`, `plugin_sdk`). Do not search other projects or units the change does not consume, and do not propose a new
dependency to reach a helper.

- The same translation unit: nearly identical blocks, functions or test bodies; merge them there when possible (a shared helper,
  a template, a parameterized test) before looking further out.
- The target's own sources and the linked SDKs' utilities (`test_sdk/`, `data_sdk/`, `plugin_sdk/`), and provider counterparts.
- CMake functions the project already uses (`add_hipdnn_test`, `hipdnn_add_dependency`, the helpers in
  `dnn-providers/integration-tests/cmake/HipdnnIntegrationTestHelpers.cmake`).
- Integration-test harness and bundle mechanisms that would replace hand-written C++ graph tests.
- Descriptor, packer and unpacker boilerplate written by hand where the descriptor generator produces it
  (`projects/hipdnn/tools/DescriptorGenerator/`; `projects/hipdnn/docs/AddingNewOperations.md`).

## Within the change

- Near-identical functions, test bodies or CMake blocks differing by a type, value or name: one parameterized form.
- Copy-pasted files across providers or architectures.
- The same constant or table defined in more than one place.

## Bar

Report only when the shared form is simpler than the copies and the copies must change together. Similar code that will diverge
is not duplication. Cite both locations and show the shared form. Report the top 5 candidates, ranked by how much code the shared
form removes; each states the **touched by this change** test: what in the diff creates or extends the duplication (a copy that
exists unchanged at `$BASE` is `FUTURE WORK`). File the rest as one theme finding listing the locations. Tier per [../policy.md](../policy.md) *Tiers* only (`SUGGESTION`; `IMPORTANT` when the copies already diverge or this change edits one copy and not the others). Copy-pasted tests that match [../policy.md](../policy.md) *Test smells* are `TESTQ`'s: hand them off (Handoffs: `TESTQ: <locations>`) instead of filing them.
