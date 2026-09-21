# AI Plugin Rename Prompt

Use this file with an AI coding agent to create a new hipDNN engine plugin
by copying and renaming the example. The example ReLU and ConvFwd
operations are kept as placeholders.

## Usage

Instruct the AI to read this file and provide the brand name and target using a prompt:

```
Read ai_plugin_rename_prompt.md and follow the instructions.
- Brand name: YourName (your_name / YOUR_NAME)
- Target directory: ../your-name-provider/
```

The three forms of the brand name are required because PascalCase word
boundaries can be ambiguous (e.g., "HTTPSProxy" could become "https_proxy"
or "h_t_t_p_s_proxy").

---

## Instructions

Read the README.md in this directory. Use the "Step-by-Step Adaptation
Workflow" and "Adaptation Reference" sections to create a new hipDNN engine
plugin by copying and renaming the example template. The user provides a
brand name (PascalCase, snake_case, and UPPER_SNAKE_CASE) and a target
directory. Ask if either is missing.

1. Copy this directory to the target directory.

2. Replace the copied `README.md` with `README_TEMPLATE.md` (rename
   `README_TEMPLATE.md` to `README.md`) and fill in the placeholders for the
   new plugin. Remove `ai_plugin_rename_prompt.md` from the target directory.

3. Apply ALL renames from the Adaptation Reference tables in the README:
   files, CMake targets/options/project name, C++ classes, and namespaces.
   Use the Case Conversion Rules for all identifier replacements.

4. Scope rules -- common pitfalls to avoid:
   - Do NOT rename SDK identifiers (see "SDK Identifiers" table in README).
   - Do NOT rename files in `hip/` or `tests/mocks/` -- update namespace only.
   - Do NOT rename kernel infrastructure: `templates/*.in`,
     `EmbedKernelSources.cmake`, `KERNEL_GENERATED_SOURCES`.
   - DO rename the `HIPDNN_REGISTER_ENGINE` arguments
     (`EXAMPLE_PROVIDER_RELU_ENGINE`, `EXAMPLE_PROVIDER_CONV_FWD_ENGINE`) and
     every `_NAME`/`_ID` constant derived from them, in
     `src/ExampleProviderContainer.cpp` and `sample/ExampleProviderSample.cpp`.
     An engine's ID is a hash of its name, and a name identifies an engine
     across the whole process, so a renamed plugin that keeps the example names
     collides with the example plugin: whichever loads second has those engines
     refused, and it then serves nothing.
   - Rename version infrastructure: `version.json` (update key name),
     `templates/version.h.in` (rename header guard and macro prefix),
     `cmake/VersionUtils.cmake` (rename the three function names).
   - Handle, Context, and Settings are at global scope (outside namespace).
     Keep them there.
   - Update nested namespaces: `example_provider::test_helpers` in tests,
     `example_provider` in `kernels/templates/*.in`.

5. Grep the target directory for remaining occurrences of
   `example_provider`, `ExampleProvider`, `EXAMPLE_PROVIDER`,
   `EXAMPLEPROVIDER_`, `hipdnn_example`, and `hipdnn-example` to verify nothing was missed.
   Nothing should remain, including the engine registration identifiers.

6. Build and run tests if ROCm/hipDNN is available.
