# How to Add a New Instruction

This guide shows you how to add a new instruction from bottom (Assembly IR) to top (Logical IR).

**Note**: Python API is decoupled via a separate wrapper layer and is not directly affected by these changes.

---

## Chapter 1: Assembly IR (Hardware Layer)

To add a new StinkyTofu assembly IR, you'll need to access the following files:

```bash
CommonInstsDSL.cpp
Flags.def
Gfx1250.cpp
RocisaHwInstMappings.hpp
```

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

`s_wait_tensorcnt` is a new feature in GFX1250. We'll add the definition to `Gfx1250.cpp`.

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

### Step 6: Add Hardware Timing Info

Add to the cost table in `hardware/src/gfx/Gfx1250.cpp`:

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
