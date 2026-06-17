# TheRock Change Required

To integrate `HipFlash2Engine` into the full build, the following change is
needed in `ROCm/TheRock/ml-libs/CMakeLists.txt`.

Add `rocwmma` as a build-time dependency for `hipkernelprovider`:

```cmake
# In therock_cmake_subproject_declare(hipkernelprovider ...) section
# Add rocwmma to BUILD_DEPS (after therock-nlohmann-json):
    BUILD_DEPS
      hipDNN
      therock-googletest
      therock-flatbuffers
      therock-nlohmann-json
      rocwmma              # <-- ADD THIS
```

Also add the CMake arg to point to rocWMMA:

```cmake
    CMAKE_ARGS
      -DHIP_PLATFORM=amd
      -DROCM_PATH=
      -DROCM_DIR=
      -DHIPKERNELPROVIDER_ENABLE_TESTS=${THEROCK_BUILD_TESTING}
      -DENABLE_HIP_FLASH2_ENGINE=${THEROCK_ENABLE_HIPKERNELPROVIDER}  # <-- ADD
      -DENABLE_CLANG_TIDY=OFF
      -DENABLE_CLANG_FORMAT=OFF
```

## Multi-arch Notes

Brian Harrison noted that static compiled kernels are non-preferred for
packaging. Two options discussed:

**Option A (hipRTC — preferred for no pre-compiled binaries):**
Compile the kernel at first use via hipRTC. rocWMMA headers would need to
be bundled as string literals. Adds ~1-2s JIT warmup on first call.

**Option B (precompiled hsaco — current approach):**
Pre-compile for each arch offline, ship as binary data:
  - `gfx942.hsaco` → MI300X/MI325X
  - `gfx950.hsaco` → MI355X/MI350X

**Current implementation uses static compilation** (Option B variant via
`--offload-arch=gfx942;gfx950` CMake flag). This is the simplest path to
validate correctness. Migration to hsaco loading follows the ASM_SDPA_ENGINE
pattern and can be done after performance/correctness validation.

Brian and Daryl have indicated they can guide on the preferred approach for
their packaging requirements.
