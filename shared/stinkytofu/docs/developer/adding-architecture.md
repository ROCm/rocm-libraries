# Adding a New GPU Architecture to StinkyTofu

This guide walks through all the steps required to add support for a new GPU architecture to the StinkyTofu framework.

> **Note:** This guide uses **Gfx1250** as a concrete example. When adding your own architecture, follow the same pattern using Gfx1250 as your template.

## Overview

Adding a new architecture involves:
1. Updating the architecture list (Step 1 and 2)
2. Creating the architecture folder with `.def` files and `.cpp` (Step 3)
3. Updating generated-file lists: Config.h.in, tablegen CMakeLists, RocisaArchInfo (Step 4)
4. Implementing Rocisa-related header (Step 5)

**Instruction definitions and costs** are in `.def` files; tablegen generates `*_init.inc`, `*_costs.inc`. The `defineGfxXXXInsts` and cost application are **auto-generated** from `GfxArchDefines_block.inc.in` and the `DEF_ARCH` block. You only provide the maps in `GfxXXX.cpp`.

| Architecture | Type | `DEF_ARCH` defaults | Notes |
|--------------|------|--------------------|-------|
| **Gfx1250** | RDNA4 | cycle=1, latency=1, VGPR=256, SGPR=102, AGPR=0 | v1 silicon (shipping stepping); what `"all"` builds |
| **Gfx1250v0** | RDNA4 | same | v0 stepping of the same chip; opt-in, cost-only delta, cannot share a build with Gfx1250 |

Gfx1250v0 is not a separate ISA: it shares Gfx1250's entire instruction set (same
mnemonics, opcodes and encodings, same `DEF_ARCH` metadata and `{12,5,0}` triple). It differs
only in the cost of a few non-sparse fp8/fp4 WMMA rows, edited inline in
`Gfx1250v0Instructions.def` / `Gfx1250v0Formats.def` (marked `v0:`); everything else is a
verbatim copy of the Gfx1250 `.def` files.

> **Gfx1250v0 is a tests-off configuration.** The test suite names `GfxArchID::Gfx1250` in
> roughly two dozen files and stays that way by design, so Gfx1250v0 must be built as
> `-DSTINKYTOFU_ARCHS_TO_BUILD=Gfx1250v0 -DSTINKYTOFU_BUILD_TESTS=OFF`. Asking for Gfx1250v0 with
> tests enabled is rejected at configure time rather than left to the compiler. Run the suite
> against Gfx1250; what is stepping-specific is instruction timing, which lives entirely in
> the `Gfx1250v0` `.def` files.

## Step-by-Step Guide

### Step 1: Update Architecture List

Add the new architecture to `STINKYTOFU_ALL_ARCHS` in `cmake/StinkytofuArchList.cmake`.
`"all"` (the default) expands to this list; a build that wants a subset asks for it by name
via `-DSTINKYTOFU_ARCHS_TO_BUILD=...`.

```cmake
set(STINKYTOFU_ALL_ARCHS
    Gfx1250
    Gfx1250v0
    GfxYourArch    # <-- Add here (or leave out to keep it opt-in)
)
```

Leave it out of `STINKYTOFU_ALL_ARCHS` if it must be asked for by name. `Gfx1250v0` is opt-in
for that reason: it is a second tapeout of Gfx1250 and reports the same `{12,5,0}` ISA triple.
`ArchHelper` resolves a triple to the first architecture registered with it, so two
architectures sharing one triple cannot go in the same library — the second would silently
receive the first's instruction costs. `CMakeLists.txt` rejects that combination; add a
matching check if you introduce another stepping.

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

> **Check what `Gfx1250.cpp` delegates to.** It does not hold its own rocisa maps; it calls
> into `../common/Gfx125xRocisaMaps.hpp`, which is shared by the gfx12.5 steppings. A copy
> inherits that call and will compile and link happily while producing gfx12.5's rocisa
> mappings for your architecture. Unless your architecture genuinely shares them, replace the
> delegation with your own `setGfxYourArchRocisaToArchMap` and `setGfxYourArchConversionMap`
> bodies.

#### 3a. Declare architecture metadata with `DEF_ARCH`

Metadata lives in a `DEF_ARCH` block at the top of `GfxYourArchFormats.def`, before the first
`DEF_FORMAT`. **All values are required.** `maxAGPR` may be 0 for RDNA.

```cpp
DEF_ARCH(GfxYourArch,
    .major = X, .minor = Y, .stepping = Z,
    .wavefront = 64,            // 64 for CDNA, 32 for RDNA
    .maxVGPR = 256, .maxSGPR = 102, .maxAGPR = 256,
    .totalVgprPerSimd = 512,    // physical VGPR file per SIMD
    .vgprAllocGranule = 8,
    .defaultCycle = 4, .defaultLatency = 4)   // 4 for CDNA, 1 for RDNA
```

`.major/.minor/.stepping` is the ISA triple. If it collides with an existing architecture's,
see the mutual-exclusion note in Step 1: `ArchHelper` resolves a triple to the first
architecture registered with it, so the two can never share a build.

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
- Wavefront size, register limits, default costs -- from `DEF_ARCH`

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

> **Skip this step if you are adding a stepping of an existing chip.** `BackendRegistry` is
> keyed by ISA triple, not by `GfxArchID`, so a second stepping of `{12,5,0}` picks up the
> existing pipeline with no new file. That is why Gfx1250v0 has no backend of its own. It also
> means `Gfx1250Backend.cpp` must keep spelling its triple as a literal: resolving it through
> `GfxArchID::Gfx1250` would stop compiling in a Gfx1250v0-only build.

An architecture on a *new* triple needs its own optimization pipeline. Create `src/pipeline/backend/GfxYourArchBackend.cpp` (copy from `Gfx1250Backend.cpp`) containing:

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

- [ ] Add to `STINKYTOFU_ALL_ARCHS` in `cmake/StinkytofuArchList.cmake` (or leave out to keep
      it opt-in)
- [ ] Add `#cmakedefine STINKYTOFU_ARCH_GFXYOURARCH` in `include/stinkytofu/Config.h.in`
- [ ] Create `hardware/src/gfx/GfxYourArch/`:
  - [ ] `GfxYourArchFormats.def` -- opening `DEF_ARCH` block with the ISA triple, wavefront,
        register limits, and default cycle/latency
  - [ ] `GfxYourArchInstructions.def` -- DEF_T for all instructions
  - [ ] `GfxYourArch.cpp` -- only `setGfxYourArchLogicalToArchMap`, `setGfxYourArchRocisaToArchMap`, `setGfxYourArchConversionMap`
- [ ] Update `tools/tablegen/CMakeLists.txt` -- add arch to INSTRUCTION_GEN_FILES and INSTRUCTION_DEF_FILES
- [ ] Create `src/conversion/rocisa/GfxYourArchRocisaArchInfo.hpp`
- [ ] Update `src/conversion/rocisa/RocisaArchInfo.hpp` -- add #include for new arch
- [ ] Only for a new ISA triple (a new stepping reuses the existing pipeline):
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
sed -i 's/\.major = 12, \.minor = 5, \.stepping = 0/.major = X, .minor = Y, .stepping = Z/' \
    hardware/src/gfx/GfxYourArch/GfxYourArchFormats.def
```

---

## Building a specific architecture / stepping

`STINKYTOFU_ARCHS_TO_BUILD` selects which architectures go into the library. Because the
`GfxArchID` enum, the `Config/Archs.def` X-macro expansion, and every generated `.inc` are
produced per build from this list, only the selected architectures exist in a given
`libstinkytofu.so` / `_stinkytofu*.so`.

```bash
# Default ("all" -> just Gfx1250, the v1 stepping):
cmake -S . -B build && cmake --build build -j          # or: invoke build

# Gfx1250 (v1) explicitly, with tests:
cmake -S . -B build -DSTINKYTOFU_ARCHS_TO_BUILD=Gfx1250 && cmake --build build -j

# Gfx1250v0 (v0), the cost-only stepping. It shares Gfx1250's {12,5,0} triple, so it cannot
# share a build with Gfx1250, and the Gfx1250-only test suite must be disabled:
cmake -S . -B build_v0 \
      -DSTINKYTOFU_ARCHS_TO_BUILD=Gfx1250v0 \
      -DSTINKYTOFU_BUILD_TESTS=OFF && cmake --build build_v0 -j
```

Import the freshly built Python binding by putting its `lib/` on `PYTHONPATH`:

```bash
PYTHONPATH=build_v0/lib python -c "import stinkytofu; print(stinkytofu.__file__)"
```

At runtime the ISA triple `{12,5,0}` resolves to whichever gfx12.5 stepping the loaded
library was built with, so a v0-only build hands callers v0's cost table transparently.

## Running a tensilelite YAML through stinkytofu

tensilelite reaches stinkytofu through `rocisa`. Two switches drive it:

- **Per-solution:** set `ScheduleIterAlg: 4` in the solution. `Solution.py` remaps this to
  `_StinkyTofuOptLevel=3` and rejects it if rocisa was built without the stinkytofu backend or
  the ISA has no backend.
- **Per-process:** `ROCISA_BACKEND=stinkytofu` redirects `import rocisa` to the
  `rocisa_stinkytofu_adaptor` shim (backed by `_stinkytofu.so`) instead of native `_rocisa`.

```bash
export PYTHONPATH=/path/to/shared/stinkytofu/build/lib:$PYTHONPATH
export ROCISA_BACKEND=stinkytofu
./Tensile/bin/Tensile config.yaml ./out    # config.yaml uses ScheduleIterAlg: 4
```
