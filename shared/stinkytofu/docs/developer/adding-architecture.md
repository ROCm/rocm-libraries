# Adding a New GPU Architecture to StinkyTofu

This guide walks through all the steps required to add support for a new GPU architecture to the StinkyTofu framework.

> **Note:** This guide uses **Gfx1250** as a concrete example. When adding your own architecture, follow the same pattern using Gfx1250 as your template.

## Overview

Adding a new architecture involves:
1. Updating the architecture list (Step 1 and 2)
2. Creating the architecture folder with `arch.cmake`, `.def` files, and `.cpp` (Step 3)
3. Updating generated-file lists: Config.h.in, tablegen CMakeLists, RocisaArchInfo (Step 4)
4. Implementing Rocisa-related header (Step 5)

**Instruction definitions and costs** are in `.def` files; tablegen generates `*_init.inc`, `*_costs.inc`. The `defineGfxXXXInsts` and cost application are **auto-generated** from `GfxArchDefines_block.inc.in` and `arch.cmake`. You only provide the maps in `GfxXXX.cpp`.

| Architecture | Type | arch.cmake defaults | Notes |
|--------------|------|--------------------|-------|
| **Gfx1250** | RDNA4 | cycle=1, latency=1, VGPR=256, SGPR=102, AGPR=0 | Currently the only supported architecture |

## Step-by-Step Guide

### Step 1: Update Architecture List

Add the new architecture to `cmake/StinkytofuArchList.cmake`:

```cmake
set(STINKYTOFU_ALL_ARCHS
    Gfx1250
    GfxYourArch    # <-- Add here
)
```

### Step 2: Update Config.h.in

Add a `#cmakedefine` entry in `include/stinkytofu/Config.h.in`:

```cpp
#cmakedefine STINKYTOFU_ARCH_GFX1250
#cmakedefine STINKYTOFU_ARCH_GFXYOURARCH    // <-- Add here
```

### Step 3: Create Architecture Definitions

**Best approach:** Copy an existing architecture folder and modify it.

```bash
cp -r hardware/src/gfx/Gfx1250 hardware/src/gfx/GfxYourArch
cd hardware/src/gfx/GfxYourArch
mv Gfx1250.cpp GfxYourArch.cpp
mv Gfx1250Formats.def GfxYourArchFormats.def
mv Gfx1250Instructions.def GfxYourArchInstructions.def
```

#### 3a. Create `arch.cmake`

This file defines architecture metadata. **All values are required.** ARCH_MAX_AGPR may be 0 for RDNA.

```cmake
# GfxYourArch arch properties
set(ARCH_MAJOR X)
set(ARCH_MINOR Y)
set(ARCH_STEPPING Z)
set(ARCH_WAVEFRONT 64)        # 64 for CDNA, 32 for RDNA
set(ARCH_DEFAULT_CYCLE 4)     # 4 for CDNA, 1 for RDNA
set(ARCH_DEFAULT_LATENCY 4)   # 4 for CDNA, 1 for RDNA
set(ARCH_MAX_VGPR 256)        # Must be > 0
set(ARCH_MAX_SGPR 102)        # Must be > 0
set(ARCH_MAX_AGPR 256)        # May be 0 for RDNA
```

#### 3b. Create `GfxYourArchInstructions.def` and `GfxYourArchFormats.def`

- **Instructions**: Add `DEF_T(ClassName, "mnemonic", .format = X, .flags = {...}, .cost = {cycle, latency})` for each instruction. Tablegen generates `*_init.inc` and `*_costs.inc`.
- **Formats**: Copy from a similar arch and adjust. See [Adding Instructions](adding-instructions.md) for DEF_T syntax details.

#### 3c. Create `GfxYourArch.cpp`

The `.cpp` file **only** contains the three map functions. No instruction definitions, no cost tables--those are generated.

```cpp
#include "gfx/InstDefDSL.hpp"

namespace stinkytofu
{
    void setGfxYourArchLogicalToArchMap(GpuArch& registry)
    {
        std::unordered_map<std::string, std::string> logicalToHwInstMap = {
            {"SBranch", "s_branch"},
            // ... add Logical IR name -> assembly mnemonic mappings
        };
        registry.setLogicalToArchMap(std::move(logicalToHwInstMap));
    }

    void setGfxYourArchRocisaToArchMap(GpuArch& registry)
    {
        std::unordered_map<std::string, std::string> rocisaToArchMap = {
            // ... Rocisa type name -> mnemonic
        };
        registry.setRocisaToArchMap(std::move(rocisaToArchMap));
    }

    void setGfxYourArchConversionMap(GpuArch& registry)
    {
        std::unordered_map<std::string, std::string> rocisaConversionMap = {
            // ... Rocisa conversion mappings
        };
        registry.setRocisaConversionMap(std::move(rocisaConversionMap));
    }
}
```

**What is auto-generated (no manual code):**
- `defineGfxYourArchInsts()` -- from `GfxArchDefines_block.inc.in`, configured per arch
- Instruction definitions -- from `GfxYourArchInstructions.def` via tablegen
- Cost tables -- from `.cost` in DEF_T, tablegen emits `*_costs.inc`
- Wavefront size, register limits, default costs -- from `arch.cmake`

### Step 4: Update Tablegen and Generated Headers

#### 4a. Update `tools/tablegen/CMakeLists.txt`

Add your arch to the `INSTRUCTION_GEN_FILES` and `INSTRUCTION_DEF_FILES` lists:

```cmake
foreach(arch Gfx1250 GfxYourArch)   # Add GfxYourArch
    ...
endforeach()
set(INSTRUCTION_DEF_FILES
    ...
    "${INSTRUCTION_DEF_BASE_DIR}/GfxYourArch/GfxYourArchFormats.def"
    "${INSTRUCTION_DEF_BASE_DIR}/GfxYourArch/GfxYourArchInstructions.def"
)
```

#### 4b. GfxXXX.hpp and ArchHelper

`GfxXXX.hpp` is **auto-generated** from `hardware/GfxArch.hpp.in` by CMake. No manual file needed. `ArchHelper_includes.inc` is also generated from the arch list. As long as your arch is in `StinkytofuArchList.cmake`, it will be included.

### Step 5: Create Rocisa-related Header

#### 5a. Create `src/conversion/rocisa/GfxYourArchRocisaArchInfo.hpp`

Copy from `Gfx1250RocisaArchInfo.hpp` and replace `1250` with your arch number.

#### 5b. Update `src/conversion/rocisa/RocisaArchInfo.hpp`

Add:

```cpp
#ifdef STINKYTOFU_ARCH_GFXYOURARCH
#include "GfxYourArchRocisaArchInfo.hpp"
#endif
```

### Step 6: Create Backend Pipeline and Register Anchor

Each architecture needs an optimization pipeline registered with `BackendRegistry`. Create `src/pipeline/backend/GfxYourArchBackend.cpp` (copy from `Gfx1250Backend.cpp`) containing:

1. A `buildGfxYourArchPipeline()` function that populates a `PassManager`
2. A static registrar struct that calls `BackendRegistry::setArchPipeline()`
3. An **anchor function** to prevent dead-stripping in static builds

```cpp
namespace stinkytofu {
namespace {
bool buildGfxYourArchPipeline(PassManager& pm, StinkyAsmModule& module) {
    // Add passes here...
    return true;
}

struct GfxYourArchRegistrar {
    GfxYourArchRegistrar() {
        BackendRegistry::setArchPipeline(
            GFXYOURARCH_ARCH, {buildGfxYourArchPipeline, {"group0", "group1"}});
    }
};
static GfxYourArchRegistrar s_gfxYourArchRegistrar;
}  // namespace

// Anchor: prevents linker from dead-stripping this TU in static builds.
void anchorGfxYourArchBackend() {}

}  // namespace stinkytofu
```

Then register the anchor in `BackendRegistry.cpp`:

```cpp
// In BackendRegistry::registerAllBackends():
void anchorGfxYourArchBackend();  // Add declaration

void BackendRegistry::registerAllBackends() {
    anchorGfx1250Backend();
    anchorGfxYourArchBackend();   // Add call
}
```

Finally, add the source file to `src/pipeline/backend/CMakeLists.txt`.

---

## Summary Checklist

- [ ] Add to `cmake/StinkytofuArchList.cmake`
- [ ] Add `#cmakedefine STINKYTOFU_ARCH_GFXYOURARCH` in `include/stinkytofu/Config.h.in`
- [ ] Create `hardware/src/gfx/GfxYourArch/`:
  - [ ] `arch.cmake` -- ARCH_MAJOR, ARCH_MINOR, ARCH_STEPPING, ARCH_WAVEFRONT, ARCH_DEFAULT_CYCLE, ARCH_DEFAULT_LATENCY, ARCH_MAX_VGPR, ARCH_MAX_SGPR, ARCH_MAX_AGPR
  - [ ] `GfxYourArchFormats.def`
  - [ ] `GfxYourArchInstructions.def` -- DEF_T for all instructions
  - [ ] `GfxYourArch.cpp` -- only `setGfxYourArchLogicalToArchMap`, `setGfxYourArchRocisaToArchMap`, `setGfxYourArchConversionMap`
- [ ] Update `tools/tablegen/CMakeLists.txt` -- add arch to INSTRUCTION_GEN_FILES and INSTRUCTION_DEF_FILES
- [ ] Create `src/conversion/rocisa/GfxYourArchRocisaArchInfo.hpp`
- [ ] Update `src/conversion/rocisa/RocisaArchInfo.hpp` -- add #include for new arch
- [ ] Create `src/pipeline/backend/GfxYourArchBackend.cpp` with pipeline, registrar, and anchor function
- [ ] Add anchor call to `BackendRegistry::registerAllBackends()` in `BackendRegistry.cpp`
- [ ] Add source to `src/pipeline/backend/CMakeLists.txt`
- [ ] Rebuild and test:
  ```bash
  cd build && cmake .. && cmake --build . -j && ctest -j
  ```

**Pro tip:** Use search-and-replace on copied files:
```bash
sed -i 's/1250/YourArch/g' hardware/src/gfx/GfxYourArch/GfxYourArch.cpp
sed -i 's/12, 5, 0/X, Y, Z/g' hardware/src/gfx/GfxYourArch/arch.cmake
```
