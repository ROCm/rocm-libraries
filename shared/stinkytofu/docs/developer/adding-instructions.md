# How to Add a New Instruction

This guide shows you how to add a new instruction from bottom (Assembly IR) to top (Logical IR).

**Note**: Python API is decoupled via a separate wrapper layer and is not directly affected by these changes.

**Instruction definitions and costs** use the [DEF_T system](#chapter-1-assembly-ir-hardware-layer): they live in `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def` (and optionally `GfxXXXFormats.def` in the same folder), using `DEF_T` for standalone instructions or `DEF_BATCH` for groups sharing a format. Tablegen generates `*_init.inc`, `*_costs.inc`, and `*_operands.inc`; the architecture `.cpp`/`.hpp` files only include those. Do not add DEF_T or cost tables manually in the `.cpp`.

---

## Quick checklist: adding an instruction (as few edits as possible)

| Goal | What to do |
|------|------------|
| **Assembly-only instruction** | Add `DEF_T(Class, "mnemonic", .format = X, .flags = {...})` or an entry inside a `DEF_BATCH` in `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def`. Rebuild. |
| **Group of related instructions** | Use `DEF_BATCH(.format = X, ...)` to share the format across entries. Each entry is `Class, "mnemonic"` with optional per-entry `.flags`, `.cost`, `.operand_fields`, `.logical`. |
| **Used from Logical IR / Rocisa** | Add `.logical = "LogicalName"` to the `DEF_T` / `DEF_BATCH` entry. Tablegen auto-generates the `setGfxXXXLogicalToArchMap()` — no manual `.cpp` edit needed. Also add an entry in `tools/tablegen/LogicalInstructionDefs.inc`. |
| **Non-default cost** | Add `.cost = {cycle, latency}` to the `DEF_T` or `DEF_BATCH` entry. No .cpp edit. |
| **Modifier-dependent cost** (e.g. matrix fmt) | Add `.costOverride = { { MatrixFmtData(FP4, FP4), cycle, latency } }` alongside `.cost`. |
| **Operand requirements** (e.g. 4 SGPRs, 8 SGPRs) | Define `.fields` in the format (e.g., `DEF_FORMAT(TENSOR, .fields = { {S0, ..., sgpr, 128}, {S1, ..., sgpr, 256} })`) or override per-instruction via `.operand_fields = { {D0, .size=64} }`. Tablegen generates `*_operands.inc`. No manual .hpp edit. |
| **Dual-encoding operand override** | Add `.alt_operand_fields = { ... }` for the promoted encoding (e.g. VOP3 form of a VOP2 instruction). |
| **New flag or format** | Add flag in `include/stinkytofu/hardware/Flags.def` and/or format in `hardware/src/gfx/GfxXXX/GfxXXXFormats.def`; then use in DEF_T / DEF_BATCH. |
| **New instruction flag** | Add flag in `include/stinkytofu/hardware/Flags.def` and use in DEF_T / DEF_BATCH `.flags`. |

**Single place to add a normal instruction:** `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def` — one `DEF_T` or one entry in an existing `DEF_BATCH`. Rebuild. If it must be visible to Logical IR, add `.logical = "Name"` in the same `.def` entry and add the Logical def in `LogicalInstructionDefs.inc`.

---

## Chapter 1: Assembly IR (Hardware Layer)

To add a new StinkyTofu assembly IR, you'll need to access the following files:

```bash
include/stinkytofu/hardware/Flags.def                  # Instruction flags (optional)
hardware/src/gfx/GfxXXX/GfxXXXInstructions.def   # Instruction defs + costs + operands (DEF_T / DEF_BATCH)
hardware/src/gfx/GfxXXX/GfxXXXFormats.def        # Format definitions (if new format needed)
hardware/src/gfx/GfxXXX/GfxXXX.cpp               # Rocisa maps (only if not using .logical in .def)
```

Here we will add the instruction `s_wait_tensorcnt` step by step.

### Step 1. Add a flag (Optional)

Add a flag in `include/stinkytofu/hardware/Flags.def` for a new instruction type if needed.

```c++
MACRO(IF_WaitTensorCnt)
```

### Step 2. Use the flag in DEF_T (Optional)

If you added a new flag in Step 1, use it in the instruction's `.flags` in the .def file. Tablegen will emit `DEF_T("mnemonic", IF_WaitTensorCnt)` in the generated _init.inc. No separate struct needed.

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

Use `DEF_T(ClassName, "mnemonic", ...)` or add an entry inside a `DEF_BATCH(...)`. Both accept the same optional fields: `.format`, `.flags`, `.cost`, `.logical`, `.operand_fields`. The tablegen generates the init and cost tables; the .cpp only includes the generated files.

#### DEF_T — single instruction

Example (minimal — uses arch default cost). Use a flag that exists in `include/stinkytofu/hardware/Flags.def` (without the `IF_` prefix, e.g. `WaitTensorCnt`):

```c
DEF_T(SWaitTensorcntInst, "s_wait_tensorcnt",
    .format = SOPP,
    .flags = {WaitTensorCnt},
    .logical = "SWaitTensorcnt"
)
```

Example with explicit cost (cycle, latency):

```c
DEF_T(SWaitTensorcntInst, "s_wait_tensorcnt",
    .format = SOPP,
    .flags = {WaitTensorCnt},
    .cost = {1, 1},
    .logical = "SWaitTensorcnt"
)
```

#### DEF_BATCH — group of instructions sharing a format

When several instructions share the same format, use `DEF_BATCH` to avoid repeating `.format` on every entry. Shared header fields (`.format`, `.flags`, etc.) are inherited by all entries; per-entry fields override or extend them (flags are additive).

```c
DEF_BATCH(.format = SOPP_WAIT16,
    SWaitLoadcntInst, "s_wait_loadcnt", .flags = {WaitCnt},
    SWaitStorecntInst, "s_wait_storecnt", .flags = {WaitCnt},
    SWaitTensorcntInst, "s_wait_tensorcnt", .flags = {WaitTensorCnt}, .logical = "SWaitTensorcnt",
    SWaitAluInst, "s_wait_alu", .flags = {HasSideEffect},
        .operand_fields = { {S0, .type=wait_alu} },
)
```

Each entry is: `ClassName, "mnemonic"` followed by optional per-entry fields (`.flags`, `.cost`, `.logical`, `.operand_fields`, `.costOverride`). Entries must **not** redefine `.format` — that comes from the batch header.

Real-world example with load instructions, costs, and operand overrides:

```c
DEF_BATCH(.format = MUBUF_LOAD,
    BufferLoadU8Inst, "buffer_load_u8", .cost = {12, 108}, .logical = "BufferLoadU8",
    BufferLoadB32Inst, "buffer_load_b32", .cost = {12, 108}, .logical = "BufferLoadB32",
    BufferLoadB128Inst, "buffer_load_b128", .cost = {12, 116}, .logical = "BufferLoadB128",
        .operand_fields = { {D0, .size=128} },
)
```

Then rebuild so tablegen runs (e.g. `cmake --build .`). See [Architecture Overview](architecture.md) for full syntax and format inheritance.

### Step 4. Rocisa LogicalToArch mapping (automatic via `.logical`)

If you added `.logical = "LogicalName"` to the `DEF_T` or `DEF_BATCH` entry in Step 3, the tablegen **auto-generates** the `setGfxXXXLogicalToArchMap()` function. No manual `.cpp` edit is needed.

Multiple logical names can map to the same mnemonic:

```c
SMovB32Inst, "s_mov_b32", .logical = {"SMovB32", "SSetMask"},
```

If you do **not** use `.logical` in the `.def`, you can still add the mapping manually in `hardware/src/gfx/Gfx1250/Gfx1250.cpp` inside `setGfx1250LogicalToArchMap()`:

```c++
{"SWaitTensorcnt", "s_wait_tensorcnt"},
```

Prefer `.logical` in the `.def` — it keeps the mapping next to the definition.

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

Costs are defined in the **architecture .def file** and **arch.cmake**, not in the .cpp.

- **Default cost**: Each arch sets a default in `arch.cmake` (ARCH_DEFAULT_CYCLE, ARCH_DEFAULT_LATENCY). E.g. Gfx1250 uses 1,1. Instructions without `.cost` in the .def use that default.
- **Override**: In `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def`, add `.cost = {cycle, latency}` to the instruction's `DEF_T(...)` or `DEF_BATCH` entry. Tablegen emits only non-default costs into `GfxXXX_costs.inc`; the generated block applies them.

Example — standalone `DEF_T` with cost:

```c
DEF_T(WaitTensorCntInst, "s_wait_tensorcnt",
    .format = SOPP,
    .flags = {SALU},
    .cost = {1, 1}
)
```

Example — inside a `DEF_BATCH`:

```c
DEF_BATCH(.format = MUBUF_LOAD,
    BufferLoadU8Inst, "buffer_load_u8", .cost = {12, 108}, .logical = "BufferLoadU8",
    BufferLoadB32Inst, "buffer_load_b32", .cost = {12, 108}, .logical = "BufferLoadB32",
)
```

##### Modifier-dependent cost overrides (`.costOverride`)

Matrix instructions can have different costs depending on the data format. Use `.costOverride` to specify alternate `{cycle, latency}` for specific format combinations:

```c
DEF_T(VWmmaF3216x16x128F8f6f4Inst, "v_wmma_f32_16x16x128_f8f6f4",
    .format = WMMA,
    .cost = {5, 8},
    .costOverride = { { MatrixFmtData(FP4, FP4), 3, 4 } },
    .operand_fields = { {S0, .size=512}, {S1, .size=512} }
)
```

Inside a `DEF_BATCH`:

```c
DEF_BATCH(.format = MXWMMA_SCALE16,
    VWmmaScale16F3216x16x128F8f6f4Inst, "v_wmma_scale16_f32_16x16x128_f8f6f4",
        .cost = {6, 8}, .costOverride = { { MatrixFmtData(FP4, FP4), 4, 4 } },
)
```

Do **not** add cost entries to any array in `GfxXXX.cpp`; they are generated from the .def. See [Architecture Overview](architecture.md).

#### 6b. Add Operand Requirements (Optional)

If your instruction has specific register width or type requirements (e.g., `tensor_load_to_lds` requires 4 SGPRs for src0, 8 SGPRs for src1), define them via format-level `.fields` in `GfxXXXFormats.def` or per-instruction `.operand_fields` overrides in `GfxXXXInstructions.def`. Tablegen generates `GfxXXX_operands.inc`; each `GfxXXX.hpp` already includes it and applies requirements to the MCID table. **You do not edit the .hpp.**

Example — format-level fields in `Gfx1250Formats.def` (applies to all instructions using this format):

```c
DEF_FORMAT(TENSOR,
    .unit = TensorUnit,
    .encoding = {64},
    .maxOperands = 4,
    .fields = {
        {S0, vaddr0, sgpr_srg0, sgpr, 128},
        {S1, vaddr1, sgpr_srg1, sgpr, 256},
        {S2, vaddr2, sgpr_srg2, sreg, 128},
        {S3, vaddr3, sgpr_addr, sreg, 128},
    },
)
```

Example — per-instruction override with `DEF_T`:

```c
DEF_T(BufferLoadB128Inst, "buffer_load_b128",
    .format = MUBUF_LOAD,
    .operand_fields = { {D0, .size=128} }
)
```

Example — per-instruction override inside `DEF_BATCH`:

```c
DEF_BATCH(.format = MUBUF_LOAD,
    BufferLoadB32Inst, "buffer_load_b32", .cost = {12, 108}, .logical = "BufferLoadB32",
    BufferLoadB128Inst, "buffer_load_b128", .cost = {12, 116}, .logical = "BufferLoadB128",
        .operand_fields = { {D0, .size=128} },
)
```

##### Dual-encoding operand overrides (`.alt_operand_fields`)

Instructions with dual encodings (e.g. VOP2 primary + VOP3 promoted) may need different operand layouts for each encoding. Use `.alt_operand_fields` for the promoted encoding:

```c
DEF_T(VCndmaskB32Inst, "v_cndmask_b32", .format = VOP2, .logical = "VCndMaskB32",
    .operand_fields = {
        {D0, vdst, vgpr, 32},
        {S0, src0, src, 32},
        {S1, vsrc1, vgpr, 32},
        {S2, src2, vcc, 64},
    },
    .alt_operand_fields = {
        {D0, vdst, vgpr, 32},
        {S0, src0, src, 32},
        {S1, src1, src, 32},
        {S2, src2, sreg, 64},
    },
)
```

**When to add**: Instruction requires non-default register sizes or types for the IR verifier. Single-register instructions (32-bit) usually need no override.

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
| **Definitions + costs** (DEF_T / DEF_BATCH) | `hardware/src/gfx/GfxXXX/GfxXXXInstructions.def` | Add `DEF_T(...)` or entry in `DEF_BATCH(...)` with `.format`, `.flags`, `.cost`, `.logical` |
| **Formats** (optional) | `hardware/src/gfx/GfxXXX/GfxXXXFormats.def` | Add `DEF_FORMAT` if needed |
| **Generated** (do not edit) | `hardware/generated/GfxXXX_init.inc`, `GfxXXX_costs.inc`, `GfxXXX_operands.inc` | Produced by tablegen from .def |
| **Rocisa LogicalToArch map** | Auto-generated from `.logical` in .def (or manually in `GfxXXX.cpp`) | Prefer `.logical = "Name"` in the DEF_T / DEF_BATCH entry |
| **Operand requirements** | `GfxXXXFormats.def` / `GfxXXXInstructions.def` | Define `.fields` in format or `.operand_fields` / `.alt_operand_fields` per-instruction; tablegen -> `*_operands.inc` (included by GfxXXX.hpp) |
| **Modifier-dependent costs** | `GfxXXXInstructions.def` | `.costOverride = { { MatrixFmtData(...), cycle, latency } }` alongside `.cost` |

### Gfx1250

Edit `hardware/src/gfx/Gfx1250/Gfx1250Instructions.def` (and optionally `Gfx1250Formats.def` in same folder), then rebuild. Use `DEF_T` for standalone instructions or `DEF_BATCH` for groups sharing a format. Add `.logical` in the `.def` to auto-generate the Rocisa LogicalToArch map. The corresponding `Gfx1250.cpp` only holds manually-added Rocisa maps if needed. Instruction definitions and costs come from the .def files and generated .inc files.

### See also

- [Architecture Overview](architecture.md) -- design and data flow for .def -> tablegen -> .inc.
