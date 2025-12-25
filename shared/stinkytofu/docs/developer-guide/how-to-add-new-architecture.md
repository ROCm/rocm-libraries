# Adding a New GPU Architecture to StinkyTofu

This guide walks through all the steps required to add support for a new GPU architecture to the StinkyTofu framework.

## Overview

Adding a new architecture involves:
1. Updating the architecture list in CMake (step 1 and step 2)
2. Creating architecture definitions (step 3)
    - Add new architecture in *gfxisa* library used by table generation
    - This includes defining the instruction set and Rocisa mappings (string to string)
3. Adding hardware data YAML files (step 4)
4. Implementing architecture `ArchInfo` class (step 5)
    - This class wraps the generated architecture information and provides helper functions
5. Implementing Rocisa-related header (step 6)
    - Provides helper functions for Rocisa-dependent mapping

**NOTE: Most of these steps can be implemented by leveraging existing architectures (such as Gfx1250) as a template, followed by minor adjustments.**

## Step-by-Step Guide

### Step 1: Update Architecture List

Add the new architecture to `cmake/StinkytofuArchList.cmake`:

```cmake
set(STINKYTOFU_ALL_ARCHS
    Gfx942
    Gfx950
    Gfx1250
    Gfx1300    # <-- Add your new architecture here
)
```

### Step 2: Update Configuration Template

Add a `#cmakedefine` entry in `include/Config.h.in`:

```cpp
// Architecture support definitions
#cmakedefine STINKYTOFU_ARCH_GFX942
#cmakedefine STINKYTOFU_ARCH_GFX950
#cmakedefine STINKYTOFU_ARCH_GFX1250
#cmakedefine STINKYTOFU_ARCH_GFX1300    // <-- Add this line
```

### Step 3: Create Arch Definitions

Create `hardware/src/gfx/Gfx1300.cpp`:

```cpp
...
#include "gfx/CommonInstsDSL.hpp"
#include "gfx/InstDefDSL.hpp"

namespace stinkytofu
{
    void defineGfx1300Insts(GpuArch& registry)
    {
        // Gfx1300 instruction definitions
        ...
    }

    void setGfx1300RocisaSimpleMap(GpuArch& registry)
    {
        // Rocisa simple mapping for Gfx1300
        ...
    }

    void setGfx1300ConversionMap(GpuArch& registry)
    {
        // Rocisa conversion mapping for Gfx1300
        ...
    }

} // namespace stinkytofu
```

**Tips for Instruction Definitions:**

- Use `DEF_T` macro for standard instructions
- Use `GEN_MFMA` / `GEN_WMMA` for matrix instructions

### Step 4: Create Hardware Data File

Create `hardware/data/Gfx1300.yaml` with hardware latency data:

```yaml
# A sample hardware configuration for GFX13.0.0
# This file is used by StinkyTofu to model the performance of kernels on this architecture
# The format is YAML
- target: [13,0,0]
- instructions:
  - default_cycle: 4
  - cycle:
    - ds_write_b8: 8
    - ds_write_b16: 8
    ...
  - latency:
    - ds_read_u8: 48
    - ds_read_u8_d16_hi: 48
    ...
```

### Step 5: Create ArchInfo Class

#### 1. Create `src/hardware/Gfx1300ArchInfo.hpp`:

```cpp
...

#include "isa/ArchHelper.hpp"
#include "isa/gfx/GfxIsa.hpp"

namespace
{

#define GET_ISAINFO_UOP_MAPPINGS
#include "hardware/Gfx1300Isa.inc"

}

using namespace stinkytofu;

struct Gfx1300ArchInfo : public ArchHelper::ArchInfo
{
    Gfx1300ArchInfo()
        : ArchInfo(13, 0, 0)   // <-- Specify the architecture version, (major, minor, stepping)
    {
    }

    IsaOpcode getIsaOpcode(UnifiedOpcode unifiedOpcode) const override
    {
        return getGfx1300Opcode(unifiedOpcode);
    }

    const HwInstDesc* getMCIDTable() const override
    {
#define GET_ISAINFO_HWINSTDESC_TABLE
#include "hardware/Gfx1300Isa.inc"
        return MCIDTable;
    }

    const std::unordered_map<std::string, uint16_t>& getMnemonicToIsaOpcodeMap() const override
    {
#define GET_ISAINFO_MNEMONIC_TO_OPCODE_MAPPINGS
#include "hardware/Gfx1300Isa.inc"
        return MnemonicToIsaOpcodeMap;
    }
};
```

#### 2. Update `src/hardware/ArchHelper.cpp`:

```cpp
...
/* Begin architecture-specific ArchInfo headers */
...
// GFX1250
#ifdef STINKYTOFU_ARCH_GFX1250
#include "Gfx1250ArchInfo.hpp"
#endif

// GFX1300
#ifdef STINKYTOFU_ARCH_GFX1300
#include "Gfx1300ArchInfo.hpp"
#endif

/* End of architecture-specific ArchInfo headers */
...
```

### Step 6: Create Rocisa-related header

#### 1. Create `src/ir/rocisa/Gfx1300RocisaArchInfo.hpp` to support the new architecture:

```cpp
...

namespace
{
    using namespace stinkytofu;

    const std::unordered_map<std::type_index, uint16_t>* Gfx1300RocisaToHwInstMap()
    {
#define GET_ROCISA_HW_MAPPING_TABLE
#include "ir/rocisa/RocisaGfx1300Mappings.inc"
        return &rocisaToHwInstMap;
    }

    const std::unordered_map<std::type_index, stinkytofu::ConvertRocisaToHwInstFunc>*
        Gfx1300RocisaToHwInstLoweringMap()
    {
#define GET_ROCISA_TO_HW_CONVERSION_TABLE
#include "ir/rocisa/RocisaGfx1300Mappings.inc"
        return &convertRocisaToHwInstFunc;
    }
};
```

#### 2. Update `hardware/include/gfx/ArchHelper.hpp`:

```cpp
...
/* Begin architecture-specific ArchInfo headers */
...
// GFX1250
#ifdef STINKYTOFU_ARCH_GFX1250
#include "Gfx1250RocisaArchInfo.hpp"
#endif

// GFX1300
#ifdef STINKYTOFU_ARCH_GFX1300
#include "Gfx1300RocisaArchInfo.hpp"
#endif

/* End of architecture-specific ArchInfo headers */
...
```
