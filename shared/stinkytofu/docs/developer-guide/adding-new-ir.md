# How to Add a New Instruction

This guide shows you how to add a new instruction from bottom (Assembly IR) to top (Logical IR).

**Note**: Python API is decoupled via a separate wrapper layer and is not directly affected by these changes.

---

## Chapter 1: Assembly IR (Hardware Layer)

To add a new StinkyTofu assembly IR, you'll need to access the following files:

```bash
CommonInstsDSL.cpp                    # Common instruction structures
Flags.def                             # Instruction flags
hardware/src/gfx/Gfx1250.cpp          # Instruction definitions + costs
src/hardware/Gfx1250ArchInfo.hpp      # Operand requirements (register width/type)
RocisaHwInstMappings.hpp              # Rocisa mappings
```

**Navigation Tip**: Each architecture file (Gfx1250.cpp, Gfx942.cpp, Gfx950.cpp) contains cross-reference comments at the top pointing to related metadata locations. Use these to quickly navigate between costs and requirements.

Here we will add the instruction `s_wait_tensorcnt` step by step.

### Step 1. Add a flag (Optional)

Add a flag in `Flags.def` for a new instruction type if needed.

```c++
MACRO(IF_WaitTensorCnt)
```

### Step 2. Create a new instruction structure (Optional)

Add a new instruction type `CommonInstsDSL.cpp` if needed.

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

### Step 3. Add definition to the corresponding architecture

`s_wait_tensorcnt` is a new feature in GFX1250. We'll add the definition to `hardware/src/gfx/Gfx1250.cpp`.

**Quick Navigation**: Open `Gfx1250.cpp` and check the header comment at the top - it shows you where to find:
- Instruction definitions (DEF_T calls) - typically lines 200-800
- Instruction costs - see Step 6 below
- Operand requirements - in `src/hardware/Gfx1250ArchInfo.hpp`

Add the definition:

```c++
DEF_T(WaitTensorCntInst, "s_wait_tensorcnt");
```

### Step 4. At last, add the mapping to 'rocisa'

In `RocisaHwInstMapping.cpp`, add the name of the 'struct' in 'rocisa' (`SWaitTensorcnt`) and the assembly instruction `s_wait_tensorcnt`.

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

#### 6a. Add Instruction Costs

Add to the cost table in `hardware/src/gfx/Gfx1250.cpp`:

**Navigation**: Open the file and read the header comment - it tells you:
- Where costs are defined (GFX1250_COSTS[] array)
- Where operand requirements are (src/hardware/Gfx1250ArchInfo.hpp)
- Where definitions are (DEF_T calls)

```cpp
constexpr InstructionCost GFX1250_COSTS[] = {
    // ... existing costs ...

    // Add new instruction costs
    {"s_wait_tensorcnt", 1, 1},  // cycle, latency

    // ... rest of costs ...
};
```

**Important**:
- Use the assembly mnemonic (`s_wait_tensorcnt`), not the class name
- If the instruction uses default costs, you don't need to add it to the table
- For Gfx1250 (RDNA4), default is cycle=1, latency=1
- For Gfx942/Gfx950 (CDNA), default is cycle=4, latency=4

#### 6b. Add Operand Requirements (Optional)

If your instruction has specific register width or type requirements (e.g., `tensor_load_to_lds` requires 4 SGPRs for src0, 8 SGPRs for src1), add them in `src/hardware/Gfx1250ArchInfo.hpp`:

**Navigation**: The header comment at the top tells you where to find costs (hardware/src/gfx/Gfx1250.cpp).

```cpp
// In getMCIDTable() method:

// 1. Define operand requirements (before the instRequirements table)
static constexpr HwInstDesc::OperandWidth myInstructionReqs[] = {
    {0, 4, false, RegType::S},  // src[0]: 4 SGPRs
    {1, 8, false, RegType::S},  // src[1]: 8 SGPRs
};

// 2. Add to the instRequirements table
static constexpr InstRequirement instRequirements[] = {
    {"tensor_load_to_lds", tensorLoadToLdsReqs},
    {"my_instruction", myInstructionReqs},  // Add your instruction
};
```

**Operand Width Fields**:
- `operandIndex`: 0-based operand index
- `width`: Number of consecutive registers (4 = 4 registers)
- `isDest`: `true` for destination operand, `false` for source
- `expectedType`: Register type (`RegType::S` for SGPR, `RegType::V` for VGPR, `RegType::A` for AGPR)

**When to Add Requirements**:
- ? Instruction requires specific register counts (e.g., 4 or 8 consecutive registers)
- ? Instruction requires specific register types (must be SGPR/VGPR/AGPR)
- ? Instruction uses single registers with no constraints (verifier will skip these)

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

Each architecture has metadata split across 2 files for modularity:

### Gfx1250 (RDNA4)

| Metadata Type | Location | What to Modify |
|---------------|----------|----------------|
| **Costs** (cycle, latency) | `hardware/src/gfx/Gfx1250.cpp` | `GFX1250_COSTS[]` array |
| **Definitions** (DEF_T) | `hardware/src/gfx/Gfx1250.cpp` | `DEF_T()` calls (lines ~200-800) |
| **Operand Requirements** | `src/hardware/Gfx1250ArchInfo.hpp` | `getMCIDTable()` method |

### Gfx942 (CDNA2/MI200) & Gfx950 (CDNA3/MI300)

Same structure as Gfx1250:
- `hardware/src/gfx/Gfx942.cpp` + `src/hardware/Gfx942ArchInfo.hpp`
- `hardware/src/gfx/Gfx950.cpp` + `src/hardware/Gfx950ArchInfo.hpp`

### Navigation Tips

? **Each file has cross-reference comments at the top** pointing to related metadata locations.

? **To modify an instruction**:
1. Open the architecture's `.cpp` file (e.g., `Gfx1250.cpp`)
2. Read the header comment - it tells you where to find costs, definitions, and requirements
3. Update costs in the same file (scroll to `GFX*_COSTS[]`)
4. Update requirements by opening the corresponding `ArchInfo.hpp` file

? **All metadata for one architecture is just a Ctrl+Click away** - no hunting through dozens of files!
