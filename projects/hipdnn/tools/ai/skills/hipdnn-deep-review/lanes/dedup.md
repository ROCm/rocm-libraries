# Lane: de-duplication (`DUP`)

Find code the change could reuse instead of writing, and code it repeats within itself.

## Against existing code

For each new helper, utility, fixture, CMake function or script, search the repository for an equivalent before reporting none:

- `projects/hipdnn/test_sdk/`, `data_sdk/` and `plugin_sdk/` utilities, and provider counterparts.
- CMake functions (`add_hipdnn_test`, `hipdnn_add_dependency`, the helpers in
  `dnn-providers/integration-tests/cmake/HipdnnIntegrationTestHelpers.cmake`).
- Integration-test harness and bundle mechanisms that would replace hand-written C++ graph tests.
- Shared descriptor, tensor, layout and dtype helpers.
- Descriptor, packer and unpacker boilerplate written by hand where the descriptor generator produces it
  (`projects/hipdnn/tools/DescriptorGenerator/`; `projects/hipdnn/docs/AddingNewOperations.md`).

## Within the change

- Near-identical functions, test bodies or CMake blocks differing by a type, value or name: one parameterized form.
- Copy-pasted files across providers or architectures.
- The same constant or table defined in more than one place.

## Bar

Report only when the shared form is simpler than the copies and the copies must change together. Similar code that will diverge
is not duplication. Cite both locations and show the shared form. Tier per [../policy.md](../policy.md) *Test smells* and *Tiers*.
