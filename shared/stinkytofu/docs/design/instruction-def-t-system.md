# Instruction DEF_T System

## Overview

Hardware instruction definitions and costs are **not** written in C++. They live in architecture-specific `.def` files. A tablegen step reads those files and generates `.inc` files that the architecture `.cpp` files include. This keeps a single source of truth per architecture and avoids manual DEF_T or cost tables in code.

## Data Flow

```
hardware/defs/
  GfxXXXFormats.def      ->  (parsed for format defaults)
  GfxXXXInstructions.def ->  (parsed for DEF_T + optional .cost)

        ? tablegen --gen-instructions --arch=GfxXXX

hardware/generated/  (build directory)
  GfxXXX_init.inc    ->  DEF_T(Class, "mnemonic");  (one line per instruction)
  GfxXXX_costs.inc   ->  { "mnemonic", cycle, latency },  (only non-default costs)

        ? #include in hardware/src/gfx/GfxXXX.cpp

GfxXXX.cpp  ->  defineGfxXXXInsts() includes _init.inc, applies _costs.inc
```

- **Formats** define default unit, encoding, flags, etc. Instructions can reference a format and inherit those defaults.
- **Instructions** are defined with `DEF_T(Class, "mnemonic", ...)` and optional `.format`, `.flags`, and `.cost`.
- **No manual DEF_T or cost arrays** are added in any `Gfx*.cpp`; all of that comes from the generated `.inc` files.

## File Roles

| File | Role |
|------|------|
| `hardware/defs/GfxXXXFormats.def` | Format definitions: `DEF_FORMAT(NAME, .unit = ..., .maxOperands = ..., .flags = {...})`. The tablegen parser uses only `.unit`, `.maxOperands`, and `.flags` for inheritance. Instructions inherit from a format when they set `.format = NAME`. |
| `hardware/defs/GfxXXXInstructions.def` | Instruction definitions: `DEF_T(ClassName, "mnemonic", .format = FMT, .flags = {...}, .cost = {cycle, latency})`. ClassName in the .def is used for documentation; the generated _init.inc uses a class derived from flags (e.g. VALU, SALU, GfxInstDef). |
| `hardware/generated/GfxXXX_init.inc` | Generated. One `DEF_T(Class, "mnemonic");` per instruction. Included by `GfxXXX.cpp` inside `defineGfxXXXInsts()`. |
| `hardware/generated/GfxXXX_costs.inc` | Generated. Array of `{"mnemonic", cycle, latency}` for instructions that override the architecture default. Included and applied in `GfxXXX.cpp`. |
| `hardware/src/gfx/GfxXXX.cpp` | Includes the two generated .inc files, sets wavefront/limits (if any), default cycle/latency, and the Rocisa LogicalToArch and conversion maps. Does **not** define instructions or cost tables by hand. |

## DEF_T Syntax (in Instructions.def)

```c
DEF_T(SomeInstClass, "mnemonic",
    .format = FORMAT_NAME,        // optional; inherits unit, flags from format
    .flags = {Flag1, Flag2},     // optional; merged with format flags
    .cost = {cycle, latency},    // optional; omit to use arch default
    .operand_widths = { {0, 4, false, S}, {1, 8, false, S} }  // optional; for IR verifier
)
```

- **ClassName** (first argument): Used in the .def for readability; the generator maps flags to an actual C++ class (e.g. VALU, SALU, GfxInstDef) when emitting _init.inc.
- **mnemonic**: Assembly opcode string (e.g. `"s_wait_tensorcnt"`). Must match the name used in Rocisa mappings and in cost tables.
- **.format**: Must match a `DEF_FORMAT` name in the same arch's `GfxXXXFormats.def`.
- **.flags**: Flag names from `include/isa/gfx/Flags.def`, used without the `IF_` prefix (e.g. VALU, SALU, MUBUFLoad, WaitTensorCnt). Format flags and instruction flags are merged. The generator maps flags to a C++ class (e.g. SALU, VALU, WaitCntInst, GfxInstDef) when emitting _init.inc.
- **.cost**: Override cycle and latency for this instruction only. If omitted, the architecture default (set in `GfxXXX.cpp`) is used.
- **.operand_widths**: Optional list of `{operandIndex, width, isDest, regType}` for the IR verifier (register width/type requirements). `regType` is `S`, `V`, or `A`. Tablegen emits `*_operands.inc`; each `GfxXXX.hpp` includes it and applies requirements to the MCID table. Adding this field is the only change needed--no .hpp edit.

## Format Inheritance

Each instruction can specify `.format = NAME`. The tablegen applies that format's parsed fields (`.unit`, `.maxOperands`, `.flags`) and then merges instruction-specific `.flags`. Unknown format names produce a warning and are skipped for inheritance.

## Cost Semantics

- **Architecture default**: Set in `GfxXXX.cpp` with `registry.setDefaultCosts(cycle, latency)` (e.g. 1,1 for RDNA, 4,4 for CDNA).
- **Per-instruction override**: In the .def, set `.cost = {cycle, latency}`. Only those instructions are emitted into `GfxXXX_costs.inc`; the rest use the default.
- The .cpp calls `setInstructionCost()` for each entry in the generated cost table, then `applyInstructionCosts()`.

## Build Integration

- CMake target `instruction_generated` runs the tablegen for each arch (Gfx942, Gfx950, Gfx1250), producing `*_init.inc` and `*_costs.inc` under the build's generated include directory.
- The gfxisa library compiles `GfxXXX.cpp` with that directory on the include path so `#include "hardware/generated/GfxXXX_init.inc"` and `_costs.inc` resolve.

## Adding or Changing an Instruction

1. Edit **only** the appropriate `hardware/defs/GfxXXXInstructions.def` (and, if needed, `GfxXXXFormats.def`).
2. Rebuild so tablegen runs (e.g. `cmake --build .` from the build directory).
3. If the instruction is exposed to Rocisa, add or update the LogicalToArch mapping in `GfxXXX.cpp` (e.g. `setGfxXXXLogicalToArchMap`).

Do **not** add DEF_T or cost entries in `GfxXXX.cpp`; they would be overwritten or ignored by the generated includes.

## See Also

- [How to Add a New Instruction](../developer-guide/adding-new-ir.md) -- step-by-step for assembly + Logical IR.
- `tools/tablegen/GenInstructions.cpp` -- parser and codegen for DEF_T and DEF_FORMAT.
