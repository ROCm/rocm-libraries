# Instruction DEF_T System

## Overview

Hardware instruction definitions and costs are **not** written in C++. They live in architecture-specific `.def` files. A tablegen step reads those files and generates `.inc` files that the architecture `.cpp` files include. This keeps a single source of truth per architecture and avoids manual DEF_T or cost tables in code.

## Data Flow

```
hardware/src/gfx/GfxXXX/
  arch.cmake             ->  ARCH_MAJOR, ARCH_WAVEFRONT, ARCH_DEFAULT_CYCLE, ARCH_MAX_VGPR, etc.
  GfxXXXFormats.def      ->  (parsed for format defaults)
  GfxXXXInstructions.def ->  (parsed for DEF_T + optional .cost)

        ? tablegen_inst_gen --input-dir=hardware/src/gfx (reads GfxXXX/GfxXXX*.def)

hardware/generated/  (build directory)
  GfxXXX_init.inc    ->  DEF_T(Class, "mnemonic");  (one line per instruction)
  GfxXXX_costs.inc   ->  { "mnemonic", cycle, latency },  (only non-default costs)
  GfxXXX_block.inc   ->  defineGfxXXXInsts() body (from GfxArchDefines_block.inc.in + arch.cmake)

        ? GfxArchDefines.cpp (generated) #includes Gfx942_block.inc, Gfx950_block.inc, ...
        ? Each _block.inc includes _init.inc, applies _costs.inc, sets wavefront/limits from arch.cmake

GfxXXX.cpp  ->  Only setGfxXXXLogicalToArchMap, setGfxXXXRocisaToArchMap, setGfxXXXConversionMap
```

- **Formats** define default unit, encoding, flags, etc. Instructions can reference a format and inherit those defaults.
- **Instructions** are defined with `DEF_T(Class, "mnemonic", ...)` and optional `.format`, `.flags`, and `.cost`.
- **No manual DEF_T or cost arrays** are added in any `Gfx*.cpp`; all of that comes from the generated `.inc` files.

## File Roles

| File | Role |
|------|------|
| `hardware/src/gfx/GfxXXX/GfxXXXFormats.def` | Format definitions: `DEF_FORMAT(NAME, .unit = ..., .maxOperands = ..., .flags = {...})`. The tablegen parser uses only `.unit`, `.maxOperands`, and `.flags` for inheritance. Instructions inherit from a format when they set `.format = NAME`. |
| `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def` | Instruction definitions: `DEF_T(ClassName, "mnemonic", .format = FMT, .flags = {...}, .cost = {cycle, latency})`. ClassName in the .def is used for documentation; the generated _init.inc uses a class derived from flags (e.g. VALU, SALU, GfxInstDef). |
| `hardware/src/gfx/GfxXXX/arch.cmake` | Architecture metadata: ARCH_MAJOR, ARCH_WAVEFRONT, ARCH_DEFAULT_CYCLE, ARCH_DEFAULT_LATENCY, ARCH_MAX_VGPR, ARCH_MAX_SGPR, ARCH_MAX_AGPR. Used by CMake to generate `*_block.inc`. |
| `hardware/generated/GfxXXX_init.inc` | Generated. One `DEF_T(Class, "mnemonic");` per instruction. Included by `GfxXXX_block.inc` inside `defineGfxXXXInsts()`. |
| `hardware/generated/GfxXXX_costs.inc` | Generated. Array of `{"mnemonic", cycle, latency}` for instructions that override the architecture default. Included and applied in `GfxXXX_block.inc`. |
| `hardware/generated/GfxXXX_block.inc` | Generated from `GfxArchDefines_block.inc.in` + arch.cmake. Contains `defineGfxXXXInsts()` body: wavefront, register limits, _init.inc, default costs, _costs.inc, applyInstructionCosts. |
| `hardware/src/gfx/GfxXXX/GfxXXX.cpp` | Only the Rocisa LogicalToArch, RocisaToArch, and Conversion maps. Does **not** define instructions, costs, or `defineGfxXXXInsts`. |

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
- **.flags**: Flag names from `include/stinkytofu/hardware/Flags.def`, used without the `IF_` prefix (e.g. VALU, SALU, MUBUFLoad, WaitTensorCnt). Format flags and instruction flags are merged. The generator maps flags to a C++ class (e.g. SALU, VALU, WaitCntInst, GfxInstDef) when emitting _init.inc.
- **.cost**: Override cycle and latency for this instruction only. If omitted, the architecture default (set in `arch.cmake`) is used.
- **.operand_widths**: Optional list of `{operandIndex, width, isDest, regType}` for the IR verifier (register width/type requirements). `regType` is `S`, `V`, or `A`. Tablegen emits `*_operands.inc`; each `GfxXXX.hpp` includes it and applies requirements to the MCID table. Adding this field is the only change needed--no .hpp edit.

## Format Inheritance

Each instruction can specify `.format = NAME`. The tablegen applies that format's parsed fields (`.unit`, `.maxOperands`, `.flags`) and then merges instruction-specific `.flags`. Unknown format names produce a warning and are skipped for inheritance.

## Cost Semantics

- **Architecture default**: Set in `arch.cmake` as `ARCH_DEFAULT_CYCLE` and `ARCH_DEFAULT_LATENCY`. The generated `GfxXXX_block.inc` calls `registry.setDefaultCosts()` with these values.
- **Per-instruction override**: In the .def, set `.cost = {cycle, latency}`. Only those instructions are emitted into `GfxXXX_costs.inc`; the rest use the default.
- The generated block calls `setInstructionCost()` for each entry in the cost table, then `applyInstructionCosts()`.

## Build Integration

- CMake target `instruction_generated` runs the tablegen for each arch (Gfx942, Gfx950, Gfx1250), producing `*_init.inc` and `*_costs.inc` under the build's generated include directory.
- The gfxisa library compiles `GfxArchDefines.cpp` (generated) and `GfxXXX.cpp` with that directory on the include path. The generated `GfxArchDefines.cpp` includes each `GfxXXX_block.inc`, which in turn includes `_init.inc` and `_costs.inc`.

## Adding or Changing an Instruction

1. Edit **only** the appropriate `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def` (and, if needed, `GfxXXXFormats.def` in the same folder).
2. Rebuild so tablegen runs (e.g. `cmake --build .` from the build directory).
3. If the instruction is exposed to Logical IR or Rocisa, add or update the mapping in `GfxXXX.cpp` (`setGfxXXXLogicalToArchMap`, `setGfxXXXRocisaToArchMap`, or `setGfxXXXConversionMap`).

Do **not** add DEF_T or cost entries in `GfxXXX.cpp`; they come from the .def files and generated .inc files.

## See Also

- [How to Add a New Instruction](../developer-guide/adding-new-ir.md) -- step-by-step for assembly + Logical IR.
- `tools/tablegen/GenInstructions.cpp` -- parser and codegen for DEF_T and DEF_FORMAT.
