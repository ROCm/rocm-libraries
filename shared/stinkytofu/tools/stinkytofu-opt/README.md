# stinkytofu-opt

`stinkytofu-opt` is a command-line tool for parsing, analyzing, and optimizing StinkyTofu IR (Intermediate Representation).

---

## Table of Contents

- [Build](#build)
- [Quick Start](#quick-start)
- [User Guide](#user-guide)
  - [Basic Usage](#basic-usage)
  - [Available Passes](#available-passes)
  - [Examples](#examples)
- [Developer Guide](#developer-guide)
  - [Architecture Overview](#architecture-overview)
  - [Configuration Components](#configuration-components)
- [IR Format](#ir-format)

---

## Build

To directly build the **stinkytofu** project, you need to do the workaround by modifying the `CMakeLists.txt` as below
```diff
diff --git a/shared/stinkytofu/CMakeLists.txt b/shared/stinkytofu/CMakeLists.txt
index 8651cac86e..1a2b50d476 100644
--- a/shared/stinkytofu/CMakeLists.txt
+++ b/shared/stinkytofu/CMakeLists.txt
@@ -6,7 +6,7 @@ set(CMAKE_CXX_STANDARD 20)
 set(CMAKE_CXX_STANDARD_REQUIRED ON)
 set(CMAKE_CXX_EXTENSIONS OFF)

-add_compile_options(-fno-rtti -fno-exceptions)
+add_compile_options(-fno-rtti)

 set(STINKYTOFU_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include" PARENT_SCOPE)
```

Then, navigate to the `rocm-libraries_p/shared/stinkytofu` directory:

```bash
cmake -B build .
cmake --build build --target stinkytofu-opt -j
```

The binary will be located at: `build/tools/stinkytofu-opt/stinkytofu-opt`

---

## Quick Start

```bash
# List all available passes
./build/tools/stinkytofu-opt/stinkytofu-opt --list-passes

# Parse and deserialize IR (no optimization)
./build/tools/stinkytofu-opt/stinkytofu-opt input.txt

# Apply optimization passes
./build/tools/stinkytofu-opt/stinkytofu-opt input.txt --StinkyDAGSchedulerPass --ScheduleFirstLRsPass
```

---

## User Guide

### Basic Usage

```bash
stinkytofu-opt [options] <ir_file> [--pass1] [--pass2] ...
```

**Arguments:**
- `<ir_file>`: Path to the StinkyTofu IR file to process
- `--passN`: Optional pass names to apply (use `--` prefix)

**Options:**
- `--list-passes`: Display all available optimization passes
- `--help`: Show usage information

### Available Passes

```bash
./stinkytofu-opt --list-passes
```

Output:
```
Available passes:
=================
  --StinkyClusterDSReadPass
  --StinkyDAGSchedulerPass
  --StinkyConfigurableWaitCntPass
  --ScheduleLastLRsPass
  --ScheduleFirstLRsPass
```

**Note:** Passes are applied in the order they appear on the command line, after the initial deserialization pass.

### Examples

#### Example 1: Parse and Validate IR
Simply parse the IR file without applying any optimization passes:

```bash
./build/tools/stinkytofu-opt/stinkytofu-opt tools/stinkytofu-opt/tests/stinkytofu_ir.txt
```

This will:
- Parse the IR file
- Display parsed instruction count
- Generate debug output files (`before.txt`, `after.txt`)

#### Example 2: Apply DAG Scheduling
Apply the DAG scheduler to optimize instruction ordering:

```bash
./build/tools/stinkytofu-opt/stinkytofu-opt input.txt --StinkyDAGSchedulerPass
```

#### Example 3: Multiple Passes
Apply multiple optimization passes in sequence:

```bash
./build/tools/stinkytofu-opt/stinkytofu-opt input.txt \
    --StinkyClusterDSReadPass \
    --StinkyDAGSchedulerPass
```

### Output Files

The tool generates debug output files in the same directory as the input:
- `before.txt`: IR state before each pass
- `after.txt`: IR state after each pass

---

## Developer Guide

### Architecture Overview

`stinkytofu-opt` is built using a modular pass-based architecture inspired by LLVM:

1. **IRLexer**: Tokenizes the input IR text into a token stream (`stinkytofu/serialization/asm/`, impl in `src/serialization/asm/IRLexer.cpp`)
2. **IRParser**: Parses tokens into `StinkyInstruction` objects (`stinkytofu/serialization/asm/IRParser.hpp`, impl in `src/serialization/asm/IRParser.cpp`)
3. **PassManager**: Orchestrates the execution of optimization passes
4. **Passes**: Individual optimization transformations on the IR

#### Component Flow

```
IR File → IRLexer → Tokens → IRParser → Instructions → PassManager → Optimized IR
                                                            ↓
                                                        Pass Pipeline
                                                        ├─ DeserializePass
                                                        ├─ Pass 1
                                                        ├─ Pass 2
                                                        └─ Pass N
```

### Configuration Components

The tool uses several configuration components defined in `stinkytofu-opt.hpp`:

#### 1. Available Passes (`availablePasses`)

A vector of `PassInfo` structures that register all available optimization passes:

```cpp
struct PassInfo
{
    const char* name;                               // Pass identifier (e.g., "StinkyDAGSchedulerPass")
    std::function<std::unique_ptr<Pass>()> creator; // Factory function to create the pass
};

const std::vector<PassInfo> availablePasses = {
    { "StinkyClusterDSReadPass", []() { return createStinkyClusterDSReadPass(); } },
    { "StinkyDAGSchedulerPass", []() { return createStinkyDAGSchedulerPass(); } },
    { "StinkyUnrollWaitCntPass", []() { return createStinkyUnrollWaitCntPass(); } },
    { "ScheduleLastLRsPass", []() { return createScheduleLastLRsPass(); } },
    { "ScheduleFirstLRsPass", []() { return createScheduleFirstLRsPass(); } },
};
```

**Purpose:** This registry allows dynamic pass creation based on command-line arguments.

#### 2. PassManagerDebugConfig (`getPassManagerDebugConfig()`)

Controls debug output and logging during pass execution:

```cpp
std::unique_ptr<stinkytofu::PassManagerDebugConfig> getPassManagerDebugConfig()
{
    auto debugConfig = std::make_unique<stinkytofu::PassManagerDebugConfig>();
    debugConfig->setPrintBeforeAll(true);     // Print IR before each pass
    debugConfig->setPrintAfterAll(true);      // Print IR after each pass
    debugConfig->setDumpToFileInBefore("before.txt");  // Save pre-pass state
    debugConfig->setDumpToFileInAfter("after.txt");    // Save post-pass state
    return debugConfig;
}
```

**Use Case:** Enable detailed debugging when developing or troubleshooting passes.

#### 3. StinkyOptInfo (`getStinkyOptInfo()`)

Global optimization settings that control pass behavior:

```cpp
stinkytofu::StinkyOptInfo getStinkyOptInfo()
{
    stinkytofu::StinkyOptInfo optInfo;
    optInfo.unrollGemmMovableBarrier = true;
    optInfo.unrollGemm = true;
    optInfo.distributeGlobalRead = true;
    return optInfo;
}
```

**Use Case:** Tune optimization aggressiveness for different workloads.

#### 4. Kernel Configuration (`setKernelConfig()`)

Hardware-specific configuration for the target GPU architecture:

```cpp
void setKernelConfig(stinkytofu::PassManager& passManager)
{
    passManager.setKernelConfig(
        {9, 4, 2},  // arch: GPU architecture version (e.g., gfx942)
        0,          // ta0: TileA0
        0,          // tb0: TileB0
        0,          // tm0: TileM0
        0,          // nGRA: NumGRA
        0,          // nGRB: NumGRB
        0,          // nGRM: NumGRM
        0           // numWaves: Number of waves per workgroup
                    // Note: wavefrontSz is automatically determined from architecture
    );
}
```

**Use Case:** Configure passes for specific GPU architectures and kernel characteristics.

---

## IR Format

StinkyTofu IR uses a text-based format with the following structure:

```
<instruction_mnemonic>
Dest:
    <destination_register> [optional_modifiers]
    ...
Src:
    <source_operand_1> [optional_modifiers]
    <source_operand_2> [optional_modifiers]
    ...
IssueCycles: <cycles>
LatencyCycles: <cycles>
```

### Field Descriptions

- **instruction_mnemonic**: The instruction mnemonic (e.g., `v_mfma_f32_32x32x8f16`, `ds_read_b128`)
- **Dest**: Destination register(s) with optional modifiers
- **Src**: Source operand(s) with optional modifiers (can be registers, immediates, or offsets)
- **IssueCycles**: Number of cycles to issue the instruction (use `-1` for default value)
- **LatencyCycles**: Number of cycles until the instruction completes (use `-1` for default value)

### Example IR

```
ds_read_b128
  Dest: v[34:37]
  Src : v[5]
        BARRIER[0]
  issueCycles: 4
  latencyCycles: 52

v_mfma_f32_16x16x16_f16
  Dest: acc[0:3]
  Src : v[6:7]
        v[22:23]
        acc[0:3]
  issueCycles: 4
  latencyCycles: 16
```

See `tools/stinkytofu-opt/tests/stinkytofu_ir.txt` for more examples.
