# RemoveInstructionPass

`RemoveInstructionPass` strips instructions whose unified opcode matches a
configured target set from every basic block approved by
`PassContext::shouldProcessBasicBlock`.

Typical use cases include experiments that remove specific instruction classes
(e.g. `tensor_load_to_lds`) without hand-editing generated assembly.

## gfx1250 pipeline integration

On gfx1250, the pass is registered as the **last kernel-scope pass** (after
`AccumulateInstructionSizePass`). It is **disabled by default**: the factory
returns `nullptr` when `ModuleOptions.RemoveInstructions` is empty, so nothing
is removed unless you opt in.

## Configuration

### Via `stinkytofu-opt`

Pass mnemonics are comma-separated unified opcode names (the same strings used
in `GFX::` / `HwInstDesc::mnemonic`):

```bash
./build/tools/stinkytofu-opt/stinkytofu-opt \
    --arch gfx1250 input.stir \
    --RemoveInstructionPass=tensor_load_to_lds \
    --print-output
```

Remove multiple instruction types in one run:

```bash
./build/tools/stinkytofu-opt/stinkytofu-opt \
    --arch gfx1250 input.stir \
    --RemoveInstructionPass=tensor_load_to_lds,ds_load_b128,v_wmma_f32_16x16x16_bf16 \
    --print-output
```

Mnemonics are resolved against the target architecture at pass run time.
Unknown names are skipped with a `PASS_DEBUG` message; if no valid opcode
remains, the pass is a no-op.

### Via Tensile / rocisa (`ModuleOptions`)

Set `RemoveInstructions` in the `options` dict passed to
`rocisa.toStinkyTofuModule()`. In `KernelWriter.py`, add it to
`stinky_module_options`:

```python
stinky_module_options = {
    "OptLevel": stinky_opt_level,
    # ...
    "RemoveInstructions": "tensor_load_to_lds",
}
```

Comma-separated values remove multiple opcodes, for example:

```python
"RemoveInstructions": "tensor_load_to_lds,ds_load_b128,v_wmma_f32_16x16x16_bf16"
```

Leave `RemoveInstructions` unset or as `""` to keep the pass disabled.

### Programmatic C++ API

```cpp
// Disabled (returns nullptr)
auto disabled = createRemoveInstructionPass();

// Comma-separated mnemonics (gfx1250 backend uses this form)
auto pass = createRemoveInstructionPass("tensor_load_to_lds,s_nop");
if (pass) pm.addPass(std::move(pass));

// Explicit unified opcodes
pm.addPass(createRemoveInstructionPass(
    std::vector<UnifiedOpcode>{GFX::tensor_load_to_lds}));

// Mnemonic list (resolved at run time)
pm.addPass(createRemoveInstructionPass(
    std::vector<std::string>{"tensor_load_to_lds", "ds_load_b128"}));
```

## Common mnemonics

| Mnemonic | Instruction class |
|----------|-------------------|
| `tensor_load_to_lds` | Tensor prefetch to LDS |
| `ds_load_b128` | DS load (example LDS read) |
| `v_wmma_f32_16x16x16_bf16` | WMMA matrix multiply |

Use the exact mnemonic string for your target architecture. Operand variants
(e.g. different WMMA shapes) are separate unified opcodes.

## Tests

- Unit: `tests/unit/asm/RemoveInstructionPassTest.cpp`
- FileCheck: `tests/filecheck/remove_instruction.stir`

```bash
cd build && ctest -R 'RemoveInstructionPassTest|FileCheck.remove_instruction'
```

## Implementation

- `include/stinkytofu/transforms/asm/RemoveInstructionPass.hpp`
- `src/transforms/asm/RemoveInstructionPass.cpp`
- `src/pipeline/backend/Gfx1250Backend.cpp` (conditional pipeline hook)
