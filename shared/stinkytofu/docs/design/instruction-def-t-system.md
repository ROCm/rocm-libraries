# Instruction DEF_T System

## Overview

Hardware instruction definitions and costs are **not** written in C++. They live in architecture-specific `.def` files, using `DEF_T` for standalone instructions or `DEF_BATCH` for groups sharing a format. A tablegen step reads those files and generates `.inc` files that the architecture `.cpp` files include. This keeps a single source of truth per architecture and avoids manual DEF_T or cost tables in code.

## Data Flow

```
hardware/src/gfx/GfxXXX/
  arch.cmake             ->  ARCH_MAJOR, ARCH_WAVEFRONT, ARCH_DEFAULT_CYCLE, ARCH_MAX_VGPR, etc.
  GfxXXXFormats.def      ->  (parsed for format defaults)
  GfxXXXInstructions.def ->  (parsed for DEF_T / DEF_BATCH + optional .cost, .logical)

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
- **Instructions** are defined with `DEF_T(Class, "mnemonic", ...)` or grouped in `DEF_BATCH(.format = X, ...)` with optional `.format`, `.flags`, `.cost`, `.logical`, `.operand_fields`.
- **LogicalToArch maps** are auto-generated from `.logical` in the `.def`. No manual `.cpp` edit needed for the mapping.
- **No manual DEF_T or cost arrays** are added in any `Gfx*.cpp`; all of that comes from the generated `.inc` files.

## File Roles

| File | Role |
|------|------|
| `hardware/src/gfx/GfxXXX/GfxXXXFormats.def` | Format definitions: `DEF_FORMAT(NAME, .unit = ..., .maxOperands = ..., .flags = {...})`. The tablegen parser uses only `.unit`, `.maxOperands`, and `.flags` for inheritance. Instructions inherit from a format when they set `.format = NAME`. |
| `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def` | Instruction definitions using `DEF_T(...)` or `DEF_BATCH(...)`. Each instruction has a ClassName (for readability), mnemonic, and optional `.format`, `.flags`, `.cost`, `.logical`, `.operand_fields`. The generated _init.inc maps flags to a C++ class (e.g. VALU, SALU, GfxInstDef). `.logical` auto-generates LogicalToArch maps. |
| `hardware/src/gfx/GfxXXX/arch.cmake` | Architecture metadata: ARCH_MAJOR, ARCH_WAVEFRONT, ARCH_DEFAULT_CYCLE, ARCH_DEFAULT_LATENCY, ARCH_MAX_VGPR, ARCH_MAX_SGPR, ARCH_MAX_AGPR. Used by CMake to generate `*_block.inc`. |
| `hardware/generated/GfxXXX_init.inc` | Generated. One `DEF_T(Class, "mnemonic");` per instruction. Included by `GfxXXX_block.inc` inside `defineGfxXXXInsts()`. |
| `hardware/generated/GfxXXX_costs.inc` | Generated. Array of `{"mnemonic", cycle, latency}` for instructions that override the architecture default. Included and applied in `GfxXXX_block.inc`. |
| `hardware/generated/GfxXXX_block.inc` | Generated from `GfxArchDefines_block.inc.in` + arch.cmake. Contains `defineGfxXXXInsts()` body: wavefront, register limits, _init.inc, default costs, _costs.inc, applyInstructionCosts. |
| `hardware/src/gfx/GfxXXX/GfxXXX.cpp` | Rocisa RocisaToArch and Conversion maps. LogicalToArch is auto-generated from `.logical` in the `.def` (manual entries in `.cpp` are still supported as fallback). Does **not** define instructions, costs, or `defineGfxXXXInsts`. |

## DEF_T Syntax (in Instructions.def)

```c
DEF_T(SomeInstClass, "mnemonic",
    .format = FORMAT_NAME,        // optional; inherits unit, flags from format
    .flags = {Flag1, Flag2},     // optional; merged with format flags
    .cost = {cycle, latency},    // optional; omit to use arch default
    .operand_fields = { {D0, .size=64} }  // optional; override format field sizes/types
)
```

- **ClassName** (first argument): Used in the .def for readability; the generator maps flags to an actual C++ class (e.g. VALU, SALU, GfxInstDef) when emitting _init.inc.
- **mnemonic**: Assembly opcode string (e.g. `"s_wait_tensorcnt"`). Must match the name used in Rocisa mappings and in cost tables.
- **.format**: Must match a `DEF_FORMAT` name in the same arch's `GfxXXXFormats.def`.
- **.flags**: Flag names from `include/stinkytofu/hardware/Flags.def`, used without the `IF_` prefix (e.g. VALU, SALU, MUBUFLoad, WaitTensorCnt). Format flags and instruction flags are merged. The generator maps flags to a C++ class (e.g. SALU, VALU, WaitCntInst, GfxInstDef) when emitting _init.inc.
- **.cost**: Override cycle and latency for this instruction only. If omitted, the architecture default (set in `arch.cmake`) is used.
- **.costOverride**: Modifier-dependent cost overrides, e.g. `.costOverride = { { MatrixFmtData(FP4, FP4), 3, 4 } }`. Used alongside `.cost` for matrix instructions whose cost varies by data format.
- **.logical**: Logical IR name(s) that map to this mnemonic. Single: `.logical = "SWaitTensorcnt"`. Multiple: `.logical = {"SMovB32", "SSetMask"}`. Tablegen auto-generates `setGfxXXXLogicalToArchMap()` from these — no manual `.cpp` edit required.
- **.operand_fields**: Optional per-instruction overrides for operand field descriptions. Partial overrides (e.g. `{D0, .size=64}`) merge with the format's `.fields`; full entries (e.g. `{D0, vdst, vdst, vgpr, 32}`) replace the format default. Tablegen emits `*_operands.inc`; each `GfxXXX.hpp` includes it and applies requirements to the MCID table.
- **.alt_operand_fields**: Operand layout for the promoted encoding of dual-encoding instructions (e.g. VOP3 form of a VOP2). Same syntax as `.operand_fields`.

## DEF_BATCH Syntax (in Instructions.def)

Groups multiple instructions that share a format. Shared header fields are inherited by all entries; per-entry fields override or extend them (flags are additive). Entries must **not** redefine `.format`.

```c
DEF_BATCH(.format = FORMAT_NAME,
    // [optional shared header fields: .flags = {...}, .unit = X, ...]

    ClassName1, "mnemonic1",
    ClassName2, "mnemonic2", .logical = "LogicalName",
    ClassName3, "mnemonic3", .flags = {ExtraFlag}, .cost = {4, 8},
        .operand_fields = { {D0, .size=128} },
)
```

Each entry is: `ClassName, "mnemonic"` followed by optional per-entry fields (`.flags`, `.cost`, `.costOverride`, `.logical`, `.operand_fields`, `.alt_operand_fields`). All entries inherit the shared `.format` and `.flags` from the batch header.

Real-world example:

```c
DEF_BATCH(.format = MUBUF_LOAD,
    BufferLoadU8Inst, "buffer_load_u8", .cost = {12, 108}, .logical = "BufferLoadU8",
    BufferLoadB32Inst, "buffer_load_b32", .cost = {12, 108}, .logical = "BufferLoadB32",
    BufferLoadB128Inst, "buffer_load_b128", .cost = {12, 116}, .logical = "BufferLoadB128",
        .operand_fields = { {D0, .size=128} },
)
```

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

1. Edit **only** the appropriate `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def` (and, if needed, `GfxXXXFormats.def` in the same folder). Use `DEF_T` or add an entry to an existing `DEF_BATCH`.
2. If the instruction is exposed to Logical IR or Rocisa, add `.logical = "LogicalName"` to the `DEF_T` / `DEF_BATCH` entry. Tablegen auto-generates the `setGfxXXXLogicalToArchMap()`.
3. Rebuild so tablegen runs (e.g. `cmake --build .` from the build directory).

Do **not** add DEF_T or cost entries in `GfxXXX.cpp`; they come from the .def files and generated .inc files.

## See Also

- [How to Add a New Instruction](../developer-guide/adding-new-ir.md) -- step-by-step for assembly + Logical IR.
- `tools/tablegen/GenInstructions.cpp` -- parser and codegen for DEF_T, DEF_BATCH, and DEF_FORMAT.
