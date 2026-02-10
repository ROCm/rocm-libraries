# StinkyTofu TableGen

A code generation tool that produces architecture-specific instruction tables
and mappings from hardware definitions.

## Overview

TableGen is a compiler-like tool that reads GPU architecture definitions and
generates:
- Architecture-specific instruction descriptor tables (`.inc` files)
- Rocisa-to-hardware instruction mappings and conversion functions
- Unified opcode enumerations across all architectures

Each GPU architecture's instruction definitions are written in C++ using a
Domain-Specific Language (DSL) methodology.

This table-driven approach eliminates hardcoded instruction opcodes and enables
easy addition of new GPU architectures. In the future, it may be expanded to
generate additional tables, such as multi-level IR conversion boilerplate code.

## Usage

```bash
tablegen <outdir> <hardwareDir>
```

**Arguments:**
- `outdir`: Output directory for generated `.inc` files
- `hardwareDir`: Path to hardware definition directory (e.g., `shared/stinkytofu/hardware`)

**Example:**
```bash
./tablegen build/generated shared/stinkytofu/hardware
```

## Generated Files

### Per-Architecture Files

For each architecture (e.g., `gfx942`, `gfx950`):

**`hardware/<arch>Isa.inc`**
- Instruction descriptor table with ISA opcode, unified opcode, flags, latency, and issue cycles
- Opcode lookup functions for both mnemonic-to-ISA-opcode and unified-opcode-to-ISA-opcode mappings
- Mnemonic string table with efficient binary search

**`stinkytofu/ir/rocisa/Rocisa<arch>Mappings.inc`**
- Simple one-to-one mappings between Rocisa types and hardware instructions
- Conversion function pointers for complex instruction lowering (e.g., `v_mfma`, `v_smfmac`)

### Global Files

**`hardware/gfxIsa.inc`**
- Unified `GFX` enum containing all instruction mnemonics across all architectures
- Architecture availability comments for each instruction
- Sorted alphabetically for deterministic enumeration

## Architecture Definition Structure

### Hardware Directory Layout

```
hardware/
+-- CMakeLists.txt          # Builds gfxisa library
+-- include/gfx/
|   +-- GpuArchManager.hpp  # Architecture manager and common definitions
|   +-- CommonInstsDSL.hpp  # Instruction type definitions (shared across architectures)
|   +-- InstDefDSL.hpp      # Instruction definition DSL with cost map system
+-- src/gfx/
    +-- GpuArchManager.cpp  # Architecture registration and management
    +-- InstDefDSL.cpp      # DSL implementation
    +-- Gfx942/             # per-arch folder
    |   +-- Gfx942.cpp
    |   +-- Gfx942Formats.def
    |   +-- Gfx942Instructions.def
    +-- Gfx950/
    |   +-- Gfx950.cpp
    |   +-- Gfx950Formats.def
    |   +-- Gfx950Instructions.def
    +-- Gfx1250/
        +-- Gfx1250.cpp
        +-- Gfx1250Formats.def
        +-- Gfx1250Instructions.def
```

## Instruction Definition DSL

Instructions are defined using a declarative C++ DSL:

```cpp
// Define instruction types with specific flags (from CommonInstsDSL.hpp)
struct DSRead : GfxInstDef {
    DSRead() {
        hwInstDesc.flags.set(IF_DSRead);
    }
};

struct MFMA : GfxInstDef {
    int M, N, K, B;
    std::string outTy, inTy;
    bool sparse;
    // Constructor sets IF_MFMA or IF_SMFMA flag
};

// Define instructions for an architecture (in Gfx942.cpp)
void defineGfx942Insts(GpuArch& registry) {
    // Simple ALU instructions
    DEF_T(SALU, "s_add_i32");
    DEF_T(SALU, "s_mul_i32");

    // Memory instructions
    DEF_T(DSRead, "ds_read_b32");
    DEF_T(DSWrite, "ds_write_b32");
    DEF_T(GLOBALLoad, "global_load_b32");

    // Complex MFMA instructions with helper macro
    MatInstDesc desc = {16, 16, 16, 1, "f32", "f16"};
    MFMA* mfma = GEN_MFMA(registry, desc, /*sparse=*/false);
    // Latency is automatically computed based on matrix dimensions
}
```

## Adding a New Architecture

1. **Declare architecture function:**
   ```cpp
   // hardware/include/gfx/GpuArchManager.hpp
   namespace stinkytofu {
       void defineGfx942Insts(GpuArch& registry);
       void defineGfx950Insts(GpuArch& registry);
       void defineGfx1000Insts(GpuArch& registry);  // Add declaration
   }
   ```

2. **Implement instruction set:**
   ```cpp
   // hardware/src/gfx/Gfx1000/Gfx1000.cpp
   #include "gfx/GpuArchManager.hpp"
   #include "gfx/CommonInstsDSL.hpp"  // For instruction type definitions
   #include "gfx/InstDefDSL.hpp"

   namespace stinkytofu {
       void defineGfx1000Insts(GpuArch& registry) {
           DEF_T(SALU, "s_add_i32");
           DEF_T(SALU, "s_mul_i32");
           DEF_T(DSRead, "ds_read_b32");
           // ... more instructions
       }
   }
   ```

3. **Register architecture:**
   ```cpp
   // hardware/src/gfx/GpuArchManager.cpp
   bool GpuArchManager::initAllArchs(GpuArchManager& manager,
                                     const std::string& hardwareDir) {
       manager.addArch("gfx942", defineGfx942Insts, hardwareDir);
       manager.addArch("gfx950", defineGfx950Insts, hardwareDir);
       manager.addArch("gfx1000", defineGfx1000Insts, hardwareDir);  // Add here

       manager.enumAllOpcodes();
       // ... error checking
   }
   ```

4. **Add logical-to-architecture instruction mappings:**
   ```cpp
   // tools/tablegen/GenRocisaHwMapping.cpp
   static Map getGfx1000LogicalToArchMappings() {
       return {
           {"SAddI32", "s_add_i32"},
           {"DSReadB32", "ds_read_b32"},
           // ... more simple one-to-one mappings
       };
   }

   static Map getGfx1000Conversion() {
       return {
           {"MFMAInstruction", "lowerRocisaMFMA"},
           // ... complex conversion functions
       };
   }

   bool genAllArchRocisaMappings(GpuArchManager& manager, const std::string& outdir) {
       // ... existing architectures
       success &= genRocisaMappings(
           manager, "gfx1000", outdir,
           getGfx1000LogicalToArchMappings(),
           getGfx1000Conversion());
       return success;
   }
   ```

5. **Add instruction costs in the architecture file:**
   ```cpp
   // In hardware/src/gfx/Gfx1000/Gfx1000.cpp
   namespace {
       struct InstructionCost {
           const char* opcode;
           uint16_t cycle;
           uint16_t latency;
       };

       constexpr InstructionCost GFX1000_COSTS[] = {
           {"buffer_load_dword", 12, 108},
           {"ds_read_b32", 4, 48},
           // ... add exception costs
       };

       constexpr uint16_t GFX1000_DEFAULT_CYCLE = 1;
       constexpr uint16_t GFX1000_DEFAULT_LATENCY = 1;
   }

   void defineGfx1000Insts(GpuArch& registry) {
       registry.setDefaultCosts(GFX1000_DEFAULT_CYCLE, GFX1000_DEFAULT_LATENCY);

       // ... define instructions

       for(const auto& cost : GFX1000_COSTS) {
           registry.setInstructionCost(cost.opcode, cost.cycle, cost.latency);
       }

       if(!registry.applyInstructionCosts()) {
           std::cerr << "FATAL: Failed to apply instruction costs\n";
           return;
       }
   }
   ```

6. **Update build:**
   ```cmake
   # hardware/CMakeLists.txt
   set(GFX_SOURCES
      src/gfx/Gfx942/Gfx942.cpp
      src/gfx/Gfx950/Gfx950.cpp
      src/gfx/Gfx1000/Gfx1000.cpp  # Add new arch folder + file
       src/gfx/GpuArchManager.cpp
       src/gfx/InstDefDSL.cpp
   )
   ```

## Build Integration

TableGen is integrated into the CMake build:

```cmake
# Generate instruction tables at build time
add_custom_command(
    OUTPUT ${TABLEGEN_GENERATED_FILES}
    COMMAND tablegen ${CMAKE_BINARY_DIR}/generated ${HARDWARE_DIR}
    DEPENDS tablegen gfxisa
)

add_custom_target(tablegen_generated DEPENDS ${TABLEGEN_GENERATED_FILES})
add_dependencies(stinkytofu tablegen_generated)
```

## Instruction Flags

Instructions are categorized using flags defined in `IsaFlag.def` (using X-macro pattern):

| Flag | Description |
|------|-------------|
| `IF_MUBUFLoad` | Buffer memory load |
| `IF_MUBUFStore` | Buffer memory store |
| `IF_MUBUFAtomic` | Buffer memory atomic operation |
| `IF_FLATLoad` | Flat memory load |
| `IF_FLATStore` | Flat memory store |
| `IF_FLATAtomic` | Flat memory atomic operation |
| `IF_GLOBALLoad` | Global memory load |
| `IF_GLOBALStore` | Global memory store |
| `IF_SMemLoad` | Scalar memory load |
| `IF_SMemStore` | Scalar memory store |
| `IF_SMemAtomic` | Scalar memory atomic operation |
| `IF_DSRead` | LDS (shared memory) read |
| `IF_DSStore` | LDS (shared memory) write |
| `IF_DSAtomic` | LDS atomic operation |
| `IF_Barrier` | Synchronization barrier |
| `IF_Branch` | Control flow branch |
| `IF_WaitCnt` | Wait counter instruction |
| `IF_HasSideEffect` | Instruction with side effects (excluding memory stores, ds writes, branches, barriers, and waitcnts) |
| `IF_MFMA` | Matrix multiply-accumulate |
| `IF_SMFMA` | Sparse matrix multiply-accumulate |

Flags enable efficient instruction classification using bitset operations:

```cpp
if (inst->hwInstDesc.has(IF_DSRead)) {
    // Handle LDS read
}

// Check multiple flags
if (inst->is(IF_MFMA) || inst->is(IF_SMFMA)) {
    // Handle matrix instructions
}
```

## Design Philosophy

1. **Separation of Concerns**: Hardware definitions are isolated from compiler logic, enabling independent evolution
2. **Declarative**: Instructions are described with properties rather than implemented with behavior
3. **Extensible**: New architectures can be added without modifying existing code or core infrastructure
4. **Type-Safe**: C++ type system enforces correctness at compile-time via DSL classes
5. **Single Source of Truth**: All instruction properties (opcodes, flags, latency, etc.) defined once in hardware definitions
6. **Compile-Time Generation**: Tables are generated at build time, ensuring zero runtime overhead

## Performance Characteristics

- **Opcode Lookup**: O(log n) binary search by mnemonic string (std::lower_bound)
- **Unified Opcode Mapping**: O(1) direct array indexing
- **Flag Queries**: O(1) bitset test operations
- **Memory Footprint**: Approximately 32 bytes per instruction descriptor
- **Build Time**: Incremental; only regenerates when hardware definitions change

## See Also

- `hardware/include/gfx/InstDefDSL.hpp` - DSL implementation and API reference
- `hardware/include/gfx/GpuArchManager.hpp` - Architecture manager API
- `hardware/include/gfx/CommonInstsDSL.hpp` - Instruction type definitions
- `tools/tablegen/GenIsa.cpp` - ISA table generation implementation
- `tools/tablegen/GenRocisaHwMapping.cpp` - Rocisa mapping generation
- Generated Rocisa mappings: `stinkytofu/ir/rocisa/Rocisa<arch>Mappings.inc` (build output). Rocisa conversion sources and AllHwMappings live under `src/conversion/rocisa/` (internal).
