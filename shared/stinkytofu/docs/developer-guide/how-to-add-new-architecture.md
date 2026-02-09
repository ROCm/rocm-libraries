# Adding a New GPU Architecture to StinkyTofu

This guide walks through all the steps required to add support for a new GPU architecture to the StinkyTofu framework.

> **Note:** This guide uses **Gfx942** (CDNA3/MI300) as a concrete example to show how an actual architecture is implemented. When adding your own architecture, follow the same pattern using Gfx942, Gfx950, or Gfx1250 as your template.

## Overview

Adding a new architecture involves:
1. Updating the architecture list in CMake (step 1 and step 2)
2. Creating architecture definitions (step 3)
    - Add new architecture in *gfxisa* library used by table generation
    - This includes defining the instruction set, instruction costs, and Rocisa mappings (string to string)
3. Implementing architecture 'ArchInfo' class (step 4)
    - This class wraps the generated architecture information and provides helper functions
4. Implementing Rocisa-related header (step 5)
    - Provides helper functions for Rocisa-dependent mapping

**NOTE: Use existing architectures as templates:**

| Architecture | File | Type | Defaults | Notes |
|--------------|------|------|----------|-------|
| **Gfx942** | `hardware/src/gfx/Gfx942.cpp` | CDNA3/MI300 | cycle=4, latency=4 | ~986 lines, has MFMA |
| **Gfx950** | `hardware/src/gfx/Gfx950.cpp` | CDNA5 | cycle=4, latency=4 | Has WMMA |
| **Gfx1250** | `hardware/src/gfx/Gfx1250.cpp` | RDNA4 | cycle=1, latency=1 | Simpler ISA |

**This guide uses Gfx942 as the example** - all code snippets are from actual files you can reference.

## Step-by-Step Guide

### Step 1: Update Architecture List

Add the new architecture to `cmake/StinkytofuArchList.cmake`.

**Example (how Gfx942 is listed):**

```cmake
set(STINKYTOFU_ALL_ARCHS
    Gfx942    # <-- CDNA3/MI300 (existing example)
    Gfx950    # <-- Your new architecture would be added here
    Gfx1250
)
```

### Step 2: Update Configuration Template

Add a `#cmakedefine` entry in `include/Config.h.in`.

**Example (how Gfx942 is configured):**

```cpp
// Architecture support definitions
#cmakedefine STINKYTOFU_ARCH_GFX942    // <-- CDNA3/MI300 (existing example)
#cmakedefine STINKYTOFU_ARCH_GFX950    // <-- Your new architecture would be added here
#cmakedefine STINKYTOFU_ARCH_GFX1250
```

### Step 3: Create Arch Definitions with Instruction Costs

**Best approach:** Copy an existing architecture file and modify it.

**Example:** Copy a similar architecture as your starting point:
```bash
# For CDNA-like architecture, copy Gfx942 or Gfx950
cp hardware/src/gfx/Gfx942.cpp hardware/src/gfx/GfxYourArch.cpp

# For RDNA-like architecture, copy Gfx1250
cp hardware/src/gfx/Gfx1250.cpp hardware/src/gfx/GfxYourArch.cpp
```

Here's the actual structure from **Gfx942** (CDNA3/MI300):

```cpp
#include <iostream>
#include <string>

#include "gfx/CommonInstsDSL.hpp"
#include "gfx/InstDefDSL.hpp"

namespace stinkytofu
{
    namespace
    {
        // Instruction cost structure (same for all architectures)
        struct InstructionCost
        {
            const char* opcode;
            uint16_t    cycle;
            uint16_t    latency;
        };

        // Architecture-specific instruction costs
        // Only instructions that differ from defaults are listed here
        constexpr InstructionCost GFX942_COSTS[] = {
            // Buffer loads
            {"buffer_load_dword", 12, 108},
            {"buffer_load_dwordx2", 12, 128},
            {"buffer_load_dwordx3", 12, 120},
            {"buffer_load_dwordx4", 12, 124},
            // ... ~200 more instructions ...

            // MFMA instructions (must enumerate ALL variants)
            {"v_mfma_f32_16x16x16_f16", 4, 16},
            {"v_mfma_f32_16x16x1_4b_f32", 4, 32},
            {"v_mfma_f32_16x16x2_bf16", 4, 16},
            {"v_mfma_f32_16x16x4_f32", 4, 32},
            // ... all MFMA variants ...
        };

        // Architecture default costs
        constexpr uint16_t GFX942_DEFAULT_CYCLE = 4;
        constexpr uint16_t GFX942_DEFAULT_LATENCY = 4;
    }

    void defineGfx942Insts(GpuArch& registry)
    {
        // Architecture properties
        registry.setWaveFrontSize(64);
        registry.setRegisterLimits({256, 102, 256});  // VGPR, SGPR, AGPR

        // STEP 1: Set default costs (REQUIRED!)
        registry.setDefaultCosts(GFX942_DEFAULT_CYCLE, GFX942_DEFAULT_LATENCY);

        // STEP 2: Define all instructions
        // (See full file for complete instruction definitions)
        genScalarALU(registry);
        genVectorALU(registry);
        genBufferInsts(registry);
        genDSInsts(registry);
        genMFMAInsts(registry);
        // ... etc.

        // STEP 3: Register instruction-specific costs (exceptions from defaults)
        for(const auto& cost : GFX942_COSTS)
        {
            registry.setInstructionCost(cost.opcode, cost.cycle, cost.latency);
        }

        // STEP 4: Apply costs with strict validation
        if(!registry.applyInstructionCosts())
        {
            std::cerr << "FATAL: Failed to apply instruction costs for Gfx942\n";
            return;
        }
    }

    void setGfx942LogicalToArchMap(GpuArch& registry)
    {
        // Logical-to-architecture instruction name mappings
        // (See full file for complete mappings)
    }

    void setGfx942ConversionMap(GpuArch& registry)
    {
        // Rocisa conversion mappings
        // (See full file for complete conversions)
    }

} // namespace stinkytofu
```

**To adapt Gfx942 for your architecture:**
1. Replace all occurrences of `942` with your architecture number
2. Update default costs based on your architecture family:
   - CDNA (MI200/MI300): cycle=4, latency=4
   - RDNA: cycle=1, latency=1
3. Update `GFX942_COSTS[]` table with your architecture's instruction timings
4. Modify instruction definitions if your ISA differs
5. Update register limits (`setRegisterLimits`) and wave front size (`setWaveFrontSize`)

**Important Notes for Instruction Costs:**

1. **Default Costs:** Choose appropriate defaults for your architecture
   - CDNA (Gfx942/Gfx950): cycle=4, latency=4
   - RDNA (Gfx1250): cycle=1, latency=1

2. **Exception Table:** Only include instructions that differ from defaults
   - This keeps the table small and maintainable
   - Most instructions will use the default values

3. **Enumerate All MFMA/WMMA Variants:**
   - Each MFMA/WMMA variant must be listed explicitly
   - Cannot use formula-based calculation (removed for compile-time validation)
   - See existing architectures for complete lists

4. **Compile-Time Validation:**
   - Build fails if `setDefaultCosts()` is not called
   - Build fails if any instruction ends up with 0 cycle or latency
   - Prevents silent errors from typos or missing costs

**Tips for Instruction Definitions:**

- Use `DEF_T` macro for standard instructions
- Use `GEN_MFMA` / `GEN_WMMA` for matrix instructions
- **Copy from existing architectures:**
  - `hardware/src/gfx/Gfx942.cpp` - ~986 lines, CDNA3 with MFMA
  - `hardware/src/gfx/Gfx950.cpp` - CDNA5 with WMMA
  - `hardware/src/gfx/Gfx1250.cpp` - RDNA4, simpler instruction set
- Use `grep "v_mfma\|v_wmma" instruction_costs_groundtruth.txt` to see all matrix variants

**Where to find instruction costs:**
- Reference existing Gfx942/Gfx950/Gfx1250 cost tables
- Use hardware documentation for your architecture
- Start with defaults, then add exceptions as needed

### Step 4: Create ArchInfo Class

#### 1. Create `hardware/include/Gfx942.hpp`

**Template:** Copy from an existing ArchInfo file.

**Example (actual Gfx942 structure):**

```cpp
#include "isa/ArchHelper.hpp"

namespace
{

#define GET_ISAINFO_UOP_MAPPINGS
#include "hardware/Gfx942Isa.inc"

}

using namespace stinkytofu;

struct Gfx942ArchInfo : public ArchHelper::ArchInfo
{
    Gfx942ArchInfo()
        : ArchInfo(9, 4, 2)   // <-- Architecture version (major=9, minor=4, stepping=2)
    {
    }

    IsaOpcode getIsaOpcode(UnifiedOpcode unifiedOpcode) const override
    {
        return getGfx942Opcode(unifiedOpcode);
    }

    const HwInstDesc* getMCIDTable() const override
    {
#define GET_ISAINFO_HWINSTDESC_TABLE
#include "hardware/Gfx942Isa.inc"
        return MCIDTable;
    }

    const std::unordered_map<std::string, uint16_t>& getMnemonicToIsaOpcodeMap() const override
    {
#define GET_ISAINFO_MNEMONIC_TO_OPCODE_MAPPINGS
#include "hardware/Gfx942Isa.inc"
        return MnemonicToIsaOpcodeMap;
    }
};
```

**To adapt:** Replace `942` and `9, 4, 2` with your architecture's identifiers.

#### 2. Update `src/hardware/ArchHelper.cpp`

**Example (how Gfx942 is included):**

```cpp
/* Architecture-specific headers (GfxXXX.hpp defines GfxXXXArchInfo) */

#ifdef STINKYTOFU_ARCH_GFX942
#include "Gfx942.hpp"    // <-- CDNA3/MI300 (existing example)
#endif

#ifdef STINKYTOFU_ARCH_GFX950
#include "Gfx950.hpp"
#endif

#ifdef STINKYTOFU_ARCH_GFX1250
#include "Gfx1250.hpp"
#endif
```

### Step 5: Create Rocisa-related header

#### 1. Create `src/ir/rocisa/Gfx942RocisaArchInfo.hpp`

**Template:** Copy from an existing RocisaArchInfo file.

**Example (actual Gfx942 structure):**

```cpp
namespace
{
    using namespace stinkytofu;

    const std::unordered_map<std::type_index, uint16_t>* Gfx942RocisaToHwInstMap()
    {
#define GET_ROCISA_HW_MAPPING_TABLE
#include "ir/rocisa/RocisaGfx942Mappings.inc"
        return &rocisaToHwInstMap;
    }

    const std::unordered_map<std::type_index, stinkytofu::ConvertRocisaToHwInstFunc>*
        Gfx942RocisaToHwInstLoweringMap()
    {
#define GET_ROCISA_TO_HW_CONVERSION_TABLE
#include "ir/rocisa/RocisaGfx942Mappings.inc"
        return &convertRocisaToHwInstFunc;
    }
};
```

**To adapt:** Replace `942` with your architecture number.

#### 2. Update `src/ir/rocisa/RocisaArchHelper.cpp`

**Example (how Gfx942 is included):**

```cpp
/* Begin architecture-specific ArchInfo headers */

// GFX942
#ifdef STINKYTOFU_ARCH_GFX942
#include "Gfx942RocisaArchInfo.hpp"    // <-- CDNA3/MI300 (existing example)
#endif

// GFX950 (your new architecture would be added here)
#ifdef STINKYTOFU_ARCH_GFX950
#include "Gfx950RocisaArchInfo.hpp"
#endif

/* End of architecture-specific ArchInfo headers */
```

---

## Summary Checklist

When adding a new architecture, follow the Gfx942 pattern:

- [ ] Update `cmake/StinkytofuArchList.cmake` - add your architecture to the list
- [ ] Update `include/Config.h.in` - add `#cmakedefine STINKYTOFU_ARCH_GFXYOURARCH`
- [ ] Create `hardware/src/gfx/GfxYourArch.cpp` (copy from Gfx942/Gfx950/Gfx1250)
  - [ ] Define instruction cost table `GFXYOURARCH_COSTS[]`
  - [ ] Set default costs `GFXYOURARCH_DEFAULT_CYCLE/LATENCY`
  - [ ] Define instructions (`defineGfxYourArchInsts`)
  - [ ] Apply costs with validation (`applyInstructionCosts()`)
  - [ ] Set logical-to-arch mappings (`setGfxYourArchLogicalToArchMap`)
  - [ ] Set Rocisa conversion mappings (`setGfxYourArchConversionMap`)
- [ ] Add to `hardware/CMakeLists.txt` in `GFX_SOURCES`
- [ ] Create `hardware/include/GfxYourArch.hpp` (copy from similar arch)
- [ ] Update `src/hardware/ArchHelper.cpp` to include new header
- [ ] Create `src/ir/rocisa/GfxYourArchRocisaArchInfo.hpp` (copy from similar arch)
- [ ] Update `src/ir/rocisa/RocisaArchHelper.cpp` to include new header
- [ ] Rebuild and test:
  ```bash
  cd build
  cmake ..
  make -j
  make test
  ```

**Pro tip:** Use search-and-replace on the copied files:
```bash
# Example: Copy Gfx942 and replace 942 -> YourArch
sed -i 's/942/YourArch/g' hardware/src/gfx/GfxYourArch.cpp
sed -i 's/9, 4, 2/X, Y, Z/g' hardware/include/GfxYourArch.hpp
```
