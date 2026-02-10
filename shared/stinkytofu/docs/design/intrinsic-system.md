# Intrinsic System Design

## Overview

StinkyTofu's intrinsic system provides automatic loading and management of pre-defined high-level operations, similar to LLVM's device libraries (ocml.bc, ockl.bc).

## Design Goals

1. **No manual path management** - Intrinsics loaded automatically
2. **Two-stage build** - Avoid circular dependencies with TableGen
3. **Binary distribution** - Ship pre-compiled `.st.bc` files
4. **Runtime flexibility** - Override paths for testing/debugging

## Architecture

### Components

```
+-----------------------------------------------------+
| Intrinsics.intrinsic (source)                       |
| - Human-readable intrinsic definitions              |
| - Config-style syntax                               |
+----------------+------------------------------------+
                 |
                 v
+-----------------------------------------------------+
| intrinsic-compiler (build tool)                     |
| - Parses Intrinsics.intrinsic                       |
| - Validates syntax                                  |
| - Serializes to binary                              |
+----------------+------------------------------------+
                 |
                 v
+-----------------------------------------------------+
| intrinsics.st.bc (binary)                           |
| - Compact binary format                             |
| - ~1KB for 5 intrinsics                             |
| - Distributed with library                          |
+----------------+------------------------------------+
                 |
                 v
+-----------------------------------------------------+
| IntrinsicRegistry (runtime)                         |
| - Automatic loading from search paths               |
| - Singleton pattern                                 |
| - Query and lookup API                              |
+-----------------------------------------------------+
```

### Two-Stage Build Process

**Problem:** TableGen generates IR classes, but intrinsic compilation needs those classes.

**Solution:**
1. **Stage 1:** TableGen generates IR classes
2. **Stage 2:** Build `libstinkytofu.so` with generated classes
3. **Stage 3:** `intrinsic-compiler` links against `libstinkytofu.so` to compile intrinsics

```cmake
# CMakeLists.txt structure
add_custom_command(tablegen ...)           # Generate IR classes
add_library(stinkytofu ...)                # Build core library
add_subdirectory(tools/intrinsic-compiler) # Build intrinsic compiler
```

## Comparison with LLVM

### LLVM Device Libraries

```cpp
// User code - linking automatic
float result = __ocml_exp_f32(x);
```

**LLVM's approach:**
- Compiler driver (clang) finds `*.bc` files automatically
- Search: `<llvm-install>/lib/clang/<version>/amdgcn-amd-amdhsa/*.bc`
- Bitcode linked at compile time
- Override with `-L` flags

### StinkyTofu Intrinsics

```cpp
// Code generator - loading automatic
auto& intrinsics = IntrinsicRegistry::instance();
auto pattern = intrinsics.lookup("ReluF32");
```

**StinkyTofu's approach:**
- Library finds `intrinsics.st.bc` automatically
- Search: env var, current dir, build dir, install dir, system dirs
- Loaded once at runtime (singleton)
- Override with `STINKYTOFU_INTRINSICS_PATH`

## Search Path Priority

1. **`STINKYTOFU_INTRINSICS_PATH`** (environment variable)
2. **`./intrinsics.st.bc`** (current directory)
3. **Build directory** (relative to executable)
4. **Install directory** (`<prefix>/lib/stinkytofu/intrinsics.st.bc`)
5. **System directories** (`/usr/local/lib`, `/opt/rocm/lib`, etc.)

See `IntrinsicRegistry::findIntrinsicPath()` for implementation.

## Binary Format

### IRSerializer

The `IRSerializer` class handles binary I/O for intrinsic patterns:

```cpp
class IRSerializer {
public:
    static bool serialize(const IRModule& module, const std::string& path);
    static std::unique_ptr<IRModule> deserialize(const std::string& path);
};
```

**Format (simplified):**
```
[Magic: "STBC"]
[Version: uint32_t]
[NumIntrinsics: uint32_t]
[Intrinsic 1]
  [NameLength: uint32_t][Name: string]
  [NumArgs: uint32_t]
    [ArgName: string][ArgType: string] ...
  [NumInstructions: uint32_t]
    [Instruction data] ...
[Intrinsic 2]
...
```

## Runtime API

### IntrinsicRegistry (Singleton)

```cpp
class IntrinsicRegistry {
public:
    static IntrinsicRegistry& instance();

    bool isInitialized() const;
    bool hasIntrinsic(const std::string& name) const;
    const IntrinsicPattern* lookup(const std::string& name) const;
    bool reload(const std::string& path);
};
```

**Usage:**
```cpp
auto& reg = IntrinsicRegistry::instance();  // Auto-loads on first call
if (reg.hasIntrinsic("ReluF32")) {
    auto pattern = reg.lookup("ReluF32");
    // Generate code from pattern
}
```

### IntrinsicLibrary (Internal)

Lower-level API used by `IntrinsicRegistry`:

```cpp
class IntrinsicLibrary {
public:
    static std::unique_ptr<IntrinsicLibrary> loadFromFile(const std::string& path);

    bool hasIntrinsic(const std::string& name) const;
    const IntrinsicPattern* lookup(const std::string& name) const;
    size_t size() const;
};
```

## Integration Points

### TensileLite

```cpp
// In TensileLite code generator
auto& intrinsics = IntrinsicRegistry::instance();

if (intrinsics.hasIntrinsic("ReluF32")) {
    auto pattern = intrinsics.lookup("ReluF32");

    // Generate code from pattern
    for (const auto& inst : pattern->body) {
        emitInstruction(inst);
    }
}
```

### IntrinsicExpansionPass (Future)

The `IntrinsicExpansionPass` will expand `IntrinsicCall` instructions during optimization:

```cpp
// High-level IR with intrinsic call
IntrinsicCall("ReluF32", {dest, src, temp})

// After IntrinsicExpansionPass
temp = VCmpGtF32(src, 0.0)
dest = VCndMaskB32(0.0, src, temp)
```

This happens **before** lowering to assembly IR, allowing cross-module optimization.

## File Locations

### Source Files

- `src/ir/logical/Intrinsics.intrinsic` - Source definitions
- `tools/intrinsic-compiler/intrinsic-compiler.cpp` - Compiler tool

### Build Artifacts

- `build/intrinsics.st.bc` - Compiled binary (~1KB)

### Installed Files

```
<prefix>/
  lib/
    libstinkytofu.so
    stinkytofu/
      intrinsics.st.bc     <- Installed here
  include/
    stinkytofu/
      ir/logical/IntrinsicRegistry.hpp
```

## Implementation Notes

### Why Not TableGen?

TableGen runs in Stage 1 (before core library is built). Intrinsic compilation needs:
- `PatternParser` (text parsing)
- `IRSerializer` (binary I/O)
- Full IR infrastructure

These are only available after Stage 2. Hence, `intrinsic-compiler` is a separate tool.

### Why Binary Format?

1. **Compact:** ~1KB vs ~5KB text
2. **Fast loading:** No parsing at runtime
3. **Validation:** Done at build time, not runtime
4. **Distribution:** Ship single binary file

### Why High-Level IR?

Intrinsics are defined in high-level IR (not assembly) to enable:
- Cross-module optimization
- Pattern matching and fusion
- Architecture portability

Lowering to assembly happens later in the pipeline.

## Testing

### Build System Test

```bash
ninja intrinsics_compiled
# Verifies: Intrinsics.intrinsic -> intrinsics.st.bc
```

### Runtime Test

```bash
./tools/intrinsic-compiler/demo_auto_load
# Verifies: Automatic loading from search paths
```

### Unit Tests

```bash
./build/tests/stinkytofu_tests --gtest_filter="IntrinsicFlow*"
# Verifies: Full flow (load -> query -> expand)
```

## Future Extensions

1. **Python Bindings:** Expose intrinsics to Python API
2. **Optimization:** Apply peephole patterns to intrinsic bodies
3. **Versioning:** Handle multiple `.st.bc` versions
4. **Dynamic Reload:** Hot-reload intrinsics during development

## See Also

- [Adding Intrinsics Guide](../user-guide/adding-intrinsics.md) - How to add new intrinsics
- `include/stinkytofu/ir/logical/IntrinsicRegistry.hpp` - Registry API
- `include/stinkytofu/serialization/logical/IRSerializer.hpp` - Binary format
- `tools/intrinsic-compiler/` - Compiler implementation
