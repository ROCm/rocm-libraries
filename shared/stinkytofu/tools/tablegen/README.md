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

For each architecture (e.g., `gfx1250`):

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
|   +-- InstDefDSL.hpp      # Instruction definition DSL with cost map system
+-- src/gfx/
    +-- GpuArchManager.cpp  # Architecture registration and management
    +-- InstDefDSL.cpp      # DSL implementation
    +-- Gfx1250/            # per-arch folder
        +-- Gfx1250.cpp
        +-- Gfx1250Formats.def
        +-- Gfx1250Instructions.def
```

## Instruction Definition DSL

Instructions are defined using a declarative C++ DSL:

```cpp
// Instructions are defined in GfxXXXInstructions.def with DEF_T(ClassName, "mnemonic", .flags = {...}).
// Tablegen generates _init.inc with DEF_T("mnemonic", IF_X, IF_Y) - flags come from .def.
// Example .def entry:
//   DEF_T(BufferLoadB32Inst, "buffer_load_b32", .format = MUBUF, .flags = {MUBUFLoad})
// Generated _init.inc:
//   DEF_T("buffer_load_b32", IF_MUBUFLoad);
```

## Adding a New Architecture

See the full guide: [How to Add a New Architecture](../../docs/developer-guide/how-to-add-new-architecture.md).

**Summary:** Add your arch to `cmake/StinkytofuArchList.cmake` and `Config.h.in`. Create `hardware/src/gfx/GfxYourArch/` with:
- `arch.cmake` -- ARCH_MAJOR, ARCH_WAVEFRONT, ARCH_DEFAULT_CYCLE, ARCH_MAX_VGPR, etc.
- `GfxYourArchInstructions.def` -- DEF_T for all instructions (tablegen generates `*_init.inc`, `*_costs.inc`)
- `GfxYourArchFormats.def`
- `GfxYourArch.cpp` -- only `setGfxYourArchLogicalToArchMap`, `setGfxYourArchRocisaToArchMap`, `setGfxYourArchConversionMap`

Update `tools/tablegen/CMakeLists.txt` to add your arch to INSTRUCTION_GEN_FILES and INSTRUCTION_DEF_FILES. The `defineGfxYourArchInsts` and cost application are **auto-generated** from templates.

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
- `tools/tablegen/GenIsa.cpp` - ISA table generation implementation
- `tools/tablegen/GenRocisaHwMapping.cpp` - Rocisa mapping generation
- Generated Rocisa mappings: `stinkytofu/ir/rocisa/Rocisa<arch>Mappings.inc` (build output). Rocisa conversion sources and AllHwMappings live under `src/conversion/rocisa/` (internal).
