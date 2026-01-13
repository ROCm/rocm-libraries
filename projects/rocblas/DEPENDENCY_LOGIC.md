# rocBLAS Dependency Logic

This document describes the decision flow for managing dependencies when building rocBLAS, including AOCL/BLAS library selection and CMake version management.

**STATUS: ✅ IMPLEMENTED in install.sh**

## Decision Flow (install.sh)

```
1. Are we building clients (-c flag)?
   NO → AOCL not needed (done) ✓

2. Is -d (dependencies) flag set?
   NO → Skip AOCL setup, CMake will search for existing libraries

3. Detect AOCL (via detect_aocl function):
   a) Is AOCL_ROOT set?
      YES → Validate it
            ├─ VALID → Use AOCL at AOCL_ROOT (done) ✓
            └─ INVALID → Error and exit ✗

   b) Is AOCL 5.x installed on system?
      YES → Use system AOCL 5.x (done) ✓

4. Is --skip-aocl flag set?
   YES → Skip AOCL 5.2 build, CMake will search for:
         - AOCL 4.x (if installed) ✓
         - System CBLAS via pkg-config ⚠
         - Error if nothing found ✗

   NO → Build AOCL 5.2 as local dependency (done) ✓
        (CMake auto-downloaded if needed - no flag required)
```

## CMake Local Dependency Logic

CMake is **automatically downloaded as a local build dependency** when needed (no system installation):

```
Auto-Download Triggers:
1. System CMake < 3.24.4 (rocBLAS minimum)
   → Downloads CMake 3.24.4 to ${build_dir}/deps/cmake-3.24.4/

2. System CMake < 3.26.0 AND building AOCL 5.2
   → Downloads CMake 3.26.0 to ${build_dir}/deps/cmake-3.26.0/

3. --cmake_install flag specified
   → Force-downloads CMake 3.26.0 (even if system version is sufficient)

wget is automatically added to dependencies when CMake download is needed.
```

**Key Points:**
- No `--cmake_install` flag required for normal use
- CMake is downloaded as a pre-built binary (fast!)
- Downloaded to local build directory, **NOT installed to system**
- System CMake remains untouched
- `cmake_executable` variable is updated to point to local version
- Downloaded CMake only used for this build

## BLAS Library Search Order (CMakeLists.txt)

After install.sh completes, CMake searches for BLAS libraries in this order:
```
1. Local build:     ${BUILD_DIR}/deps/aocl/install_package/lib/libaocl.*
2. AOCL_ROOT:       $AOCL_ROOT/lib*/libaocl.*
3. System AOCL 5.x: $HOME/aocl/*/*/lib*/libaocl.*
4. System AOCL 5.x: /opt/AMD/aocl/[0-9]*/lib*/libaocl.*
5. Legacy AOCL 4.x: /opt/AMD/aocl/aocl-linux-*/lib*/libaocl.* (rare)
6. System AOCL 4.x: /opt/AMD/aocl/aocl-linux-*/gcc/lib_ILP64/libblis-mt.a
7. System AOCL 4.x: /opt/AMD/aocl/aocl-linux-*/aocc/lib*/libblis-mt.a
8. Bundled BLIS:    ${BUILD_DIR}/deps/amd-blis/lib*/libblis-mt.a
9. System CBLAS:    via pkg-config (OpenBLAS, etc.)
```

## Implementation Notes

- **install.sh** controls the build decision and environment setup
- **CMakeLists.txt** searches in natural order (local build → AOCL_ROOT → system paths)
- AOCL_ROOT validation happens in install.sh via `detect_aocl()`
- System detection happens in install.sh via `detect_aocl()`
- CMake automatically downloaded as local dependency when needed

## Key Functions

**Detection/Validation (pure detection, no side effects):**
- `detect_aocl()` - Detects and validates all available AOCL
  - Checks AOCL_ROOT (validates and errors if invalid)
  - Checks system AOCL 5.x installations
  - Returns status and sets AOCL_DETECTED variable

**Local Dependencies:**
- `install_cmake(version)` - Downloads CMake pre-built binary as local dependency
  - Downloads and extracts to ${build_dir}/deps/ (NOT a system install)
  - Updates cmake_executable variable to use local version
- `build_aocl_5_2()` - Builds AOCL 5.2 from source
  - Uses cmake_executable (should already be 3.26+ at this point)
  - Clones, builds, and installs to ${build_dir}/deps/aocl/
- `setup_aocl()` - Main orchestrator
  - Calls detect_aocl() to see what's available
  - Decides whether to use existing or build new
  - Handles --skip-aocl and --clean-deps flags

## Dependency Installation Flow

1. **Check if building clients** - AOCL/BLAS only needed for clients
   - If NOT building clients → Skip all AOCL logic
2. **Early Detection** - Check if we'll need to build AOCL 5.2
   - Call detect_aocl() silently
   - Check if --skip-aocl is set
3. **Decide CMake Needs** - Will we need to download CMake?
   - Auto-download if: system CMake < 3.24.4 (rocBLAS minimum)
   - Auto-download if: building AOCL AND system CMake < 3.26
   - Force download if: --cmake_install flag set (even if version sufficient)
4. **Add wget** - If CMake download needed, add wget to package dependencies
5. **Install Packages** - Install all system packages (including wget if needed)
6. **Download CMake** - If needed, download CMake binary to local build directory
7. **Later: setup_aocl()** - Uses the local CMake (only called if building clients)
