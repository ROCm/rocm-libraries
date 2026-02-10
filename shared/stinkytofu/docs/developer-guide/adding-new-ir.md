# How to Add a New Instruction

This guide shows you how to add a new instruction from bottom (Assembly IR) to top (Logical IR).

**Note**: Python API is decoupled via a separate wrapper layer and is not directly affected by these changes.

**Instruction definitions and costs** use the [DEF_T system](../design/instruction-def-t-system.md): they live in `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def` (and optionally `GfxXXXFormats.def` in the same folder). Tablegen generates `*_init.inc`, `*_costs.inc`, and `*_operands.inc`; the architecture `.cpp`/`.hpp` files only include those. Do not add DEF_T or cost tables manually in the `.cpp`.

---

## Quick checklist: adding an instruction (as few edits as possible)

| Goal | What to do |
|------|------------|
| **Assembly-only instruction** | 1. Add one `DEF_T(Class, "mnemonic", .format = X, .flags = {...}, .cost = {...})` in `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def`. 2. Rebuild. |
| **Used from Logical IR** | Same as above, plus: add entry in `tools/tablegen/LogicalInstructionDefs.inc` and add `{"LogicalName", "mnemonic"}` in `setGfxXXXLogicalToArchMap()` in `hardware/src/gfx/GfxXXX/GfxXXX.cpp`. |
| **Used from Rocisa** | Same as above; the LogicalToArch map is what Rocisa uses. Optionally add conversion in `setGfxXXXConversionMap()` if needed. |
| **Non-default cost** | Add `.cost = {cycle, latency}` to the same `DEF_T` in the .def. No .cpp edit. |
| **Operand requirements** (e.g. 4 SGPRs, 8 SGPRs) | Add `.operand_widths = { {0, 4, false, S}, {1, 8, false, S} }` to the same `DEF_T` in the .def. Tablegen generates `*_operands.inc`; GfxXXX.hpp already includes and applies it. No manual .hpp edit. |
| **New flag or format** | Add flag in `include/stinkytofu/hardware/Flags.def` and/or format in `hardware/src/gfx/GfxXXX/GfxXXXFormats.def`; then use in DEF_T. |
| **Custom instruction class** (optional) | Add struct in `hardware/include/gfx/CommonInstsDSL.hpp` and reference via flags. |

**Single place to add a normal instruction:** `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def` (one DEF_T). Rebuild. If it must be visible to Logical IR, also add the Logical def and the LogicalToArch mapping in the .cpp.

---

## Chapter 1: Assembly IR (Hardware Layer)

To add a new StinkyTofu assembly IR, you'll need to access the following files:

```bash
hardware/include/gfx/CommonInstsDSL.hpp    # Common instruction structures (optional)
include/stinkytofu/hardware/Flags.def                  # Instruction flags (optional)
hardware/src/gfx/GfxXXX/GfxXXXInstructions.def   # Instruction definitions + costs + .operand_widths (DEF_T)
hardware/src/gfx/GfxXXX/GfxXXXFormats.def        # Format definitions (if new format needed)
hardware/src/gfx/GfxXXX/GfxXXX.cpp               # Rocisa LogicalToArch map only
```

Here we will add the instruction `s_wait_tensorcnt` step by step.

### Step 1. Add a flag (Optional)

Add a flag in `include/stinkytofu/hardware/Flags.def` for a new instruction type if needed.

```c++
MACRO(IF_WaitTensorCnt)
```

### Step 2. Create a new instruction structure (Optional)

Add a new instruction type in `hardware/include/gfx/CommonInstsDSL.hpp` if needed.

```c++
struct WaitTensorCntInst : GfxInstDef
{
    WaitTensorCntInst()
    {
        hwInstDesc.flags.set(IF_WaitTensorCnt);
    }
};
```

**Note**: If your instruction is commutative (e.g., `v_add_f32` where `a+b = b+a`), mark it as such:

```c++
struct FloatAddInst : GfxInstDef
{
    FloatAddInst()
    {
        hwInstDesc.flags.set(IF_Commutative);  // Allow operand swapping in optimizer
    }
};
```

### Step 3. Add definition and optional cost in the architecture .def file

`s_wait_tensorcnt` is a Gfx1250 feature. Add it to `hardware/src/gfx/Gfx1250/Gfx1250Instructions.def` (not in the .cpp).

Use `DEF_T(ClassName, "mnemonic", ...)` with optional `.format`, `.flags`, and `.cost`. The tablegen will generate the init and cost tables; the .cpp only includes the generated files.

Example (minimal -- uses arch default cost). Use a flag that exists in `include/stinkytofu/hardware/Flags.def` (without the `IF_` prefix, e.g. `WaitTensorCnt`):

```c
DEF_T(SWaitTensorcntInst, "s_wait_tensorcnt",
    .format = SOPP,
    .flags = {WaitTensorCnt}
)
```

Example with explicit cost (cycle, latency):

```c
DEF_T(SWaitTensorcntInst, "s_wait_tensorcnt",
    .format = SOPP,
    .flags = {WaitTensorCnt},
    .cost = {1, 1}
)
```

Then rebuild so tablegen runs (e.g. `cmake --build .`). See [Instruction DEF_T System](../design/instruction-def-t-system.md) for full syntax and format inheritance.

### Step 4. Add the Rocisa mapping in the architecture .cpp

In `hardware/src/gfx/Gfx1250/Gfx1250.cpp`, inside `setGfx1250LogicalToArchMap()`, add the Rocisa logical name and the assembly mnemonic:

```c++
{"SWaitTensorcnt", "s_wait_tensorcnt"},
```

---

## Chapter 2: Logical IR (High-Level IR Layer)

### Step 5: Add Logical IR Definition

Add to `shared/stinkytofu/tools/tablegen/LogicalInstructionDefs.inc`:

```c++
{"SWaitTensorcnt",     // className
 "s_wait_tensorcnt",   // mnemonic (must match assembly)
 "Wait for tensor operations to complete",  // comment
 0,                    // numSrcs (no source operands)
 false,                // hasDest (no destination)
 "Synchronization",    // category
 false,                // supportsDPP
 false,                // supportsSDWA
 false,                // hasDS
 false},               // isCommutative
```

### Step 6: Add Hardware Metadata (Costs + Operand Requirements)

#### 6a. Instruction Costs

Costs are defined in the **architecture .def file**, not in the .cpp.

- **Default cost**: Each arch sets a default (e.g. Gfx1250 uses 1,1; Gfx942/Gfx950 use 4,4) in its .cpp. Instructions without `.cost` in the .def use that default.
- **Override**: In `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def`, add `.cost = {cycle, latency}` to the instruction's `DEF_T(...)`. Tablegen emits only non-default costs into `GfxXXX_costs.inc`; the .cpp includes that file and applies them.

Example in `Gfx1250Instructions.def`:

```c
DEF_T(WaitTensorCntInst, "s_wait_tensorcnt",
    .format = SOPP,
    .flags = {SALU},
    .cost = {1, 1}   // optional; omit to use arch default
)
```

Do **not** add cost entries to any array in `GfxXXX.cpp`; they are generated from the .def. See [Instruction DEF_T System](../design/instruction-def-t-system.md).

#### 6b. Add Operand Requirements (Optional)

If your instruction has specific register width or type requirements (e.g., `tensor_load_to_lds` requires 4 SGPRs for src0, 8 SGPRs for src1), add them **in the same DEF_T** in `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def` using `.operand_widths`. Tablegen generates `GfxXXX_operands.inc`; each `GfxXXX.hpp` already includes it and applies requirements to the MCID table. **You do not edit the .hpp.**

Example in `Gfx1250Instructions.def`:

```c
DEF_T(TensorLoadToLdsInst, "tensor_load_to_lds",
    .format = TENSOR,
    .flags = {TENSORLoadToLds},
    .operand_widths = { {0, 4, false, S}, {1, 8, false, S} }
)
```

**Operand width tuple**: `{operandIndex, width, isDest, regType}` -- `operandIndex` 0-based; `width` = number of consecutive registers; `isDest` true for dest, false for source; `regType` = `S`, `V`, or `A` (SGPR, VGPR, AGPR).

**When to add**: Instruction requires specific register counts or types for the IR verifier. Single-register instructions usually need no `.operand_widths`.

### Step 7: Run Tablegen

Regenerate IR classes and bindings:

```bash
cd build
cmake ..
make tablegen_generated
```

This auto-generates:
- `LogicalInstructions_generated.hpp` - C++ IR classes
- `LogicalOpcodes_generated.inc` - Opcode enums
- `StinkyBuilder_*.inc` - C++ builder methods
- `IRMnemonics_generated.inc` - Mnemonic -> ASM mappings
- `RocisaGfx1250Mappings.inc` - Rocisa -> ASM opcode mappings

The mnemonic mapping automatically enables Logical IR -> Assembly IR conversion.

---

## Quick Reference: Instruction Metadata Locations

Instruction **definitions** and **costs** live in `.def` files; tablegen generates `.inc` files that the `.cpp` includes. Rocisa mappings stay in the `.cpp`.

### Per-architecture layout

| Metadata Type | Location | What to Modify |
|---------------|----------|----------------|
| **Definitions + costs** (DEF_T, .cost) | `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def` | Add or edit `DEF_T(..., .format, .flags, .cost)` |
| **Formats** (optional) | `hardware/src/gfx/GfxXXX/GfxXXXFormats.def` | Add `DEF_FORMAT` if needed |
| **Generated** (do not edit) | `hardware/generated/GfxXXX_init.inc`, `GfxXXX_costs.inc` | Produced by tablegen from .def |
| **Rocisa LogicalToArch map** | `hardware/src/gfx/GfxXXX/GfxXXX.cpp` | `setGfxXXXLogicalToArchMap()` |
| **Operand requirements** | `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def` | Add `.operand_widths = { ... }` to DEF_T; tablegen -> `*_operands.inc` (included by GfxXXX.hpp) |

### Gfx1250, Gfx942, Gfx950

Same pattern for all three: edit `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def` (and optionally `GfxXXXFormats.def` in same folder), then rebuild. The corresponding `GfxXXX.cpp` only includes the generated .inc and holds the Rocisa map.

### See also

- [Instruction DEF_T System](../design/instruction-def-t-system.md) -- design and data flow for .def -> tablegen -> .inc.
