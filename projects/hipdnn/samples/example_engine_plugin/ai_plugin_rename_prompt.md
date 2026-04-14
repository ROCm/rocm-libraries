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

2. Apply ALL renames from the Adaptation Reference tables in the README:
   files, CMake targets/options/project name, C++ classes, and namespaces.
   Use the Case Conversion Rules for all identifier replacements.

3. Scope rules -- common pitfalls to avoid:
   - Do NOT rename SDK identifiers (see "SDK Identifiers" table in README).
   - Do NOT rename files in `hip/` or `tests/mocks/` -- update namespace only.
   - Do NOT rename kernel infrastructure: `templates/*.in`,
     `EmbedKernelSources.cmake`, `KERNEL_GENERATED_SOURCES`.
   - Do NOT rename `HIPDNN_REGISTER_ENGINE` macro calls or their arguments
     (`EXAMPLE_PROVIDER_RELU_ENGINE`, `EXAMPLE_PROVIDER_CONV_FWD_ENGINE`)
     -- the example engines are placeholders to be replaced by the user.
   - Handle, Context, and Settings are at global scope (outside namespace).
     Keep them there.
   - Update nested namespaces: `example_provider::test_helpers` in tests,
     `example_provider` in `kernels/templates/*.in`.

4. Grep the target directory for remaining occurrences of
   `example_provider`, `ExampleProvider`, `EXAMPLE_PROVIDER`, and
   `hipdnn_example` to verify nothing was missed. Engine registration
   identifiers (`EXAMPLE_PROVIDER_RELU_ENGINE`,
   `EXAMPLE_PROVIDER_CONV_FWD_ENGINE`) are expected to remain.

5. Build and run tests if ROCm/hipDNN is available.
