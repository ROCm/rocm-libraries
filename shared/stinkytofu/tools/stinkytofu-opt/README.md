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

Then, navigate to the `rocm-libraries/shared/stinkytofu` directory:

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
./build/tools/stinkytofu-opt/stinkytofu-opt --arch gfx1250 func.arch-gfx1250.stir

# Apply optimization passes
./build/tools/stinkytofu-opt/stinkytofu-opt \
    --arch gfx1250 func.arch-gfx1250.stir \
    --StinkyDAGSchedulerPass \
    --ScheduleFirstLRsPass
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
- `--arch <arch>`: Target GPU architecture (default: gfx1250). Supported: `gfx1250`
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
./build/tools/stinkytofu-opt/stinkytofu-opt \
    --arch gfx1250 tools/stinkytofu-opt/tests/func.arch-gfx1250.stir
```

This will:
- Parse the IR file
- Display parsed instruction count
- Generate debug output files (`before.txt`, `after.txt`)

#### Example 2: Apply DAG Scheduling
Apply the DAG scheduler to optimize instruction ordering:

```bash
./build/tools/stinkytofu-opt/stinkytofu-opt \
    --arch gfx1250 tools/stinkytofu-opt/tests/func.arch-gfx1250.stir \
    --StinkyDAGSchedulerPass
```

#### Example 3: Multiple Passes
Apply multiple optimization passes in sequence:

```bash
./build/tools/stinkytofu-opt/stinkytofu-opt \
    --arch gfx1250 tools/stinkytofu-opt/tests/func.arch-gfx1250.stir \
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

#### 2. Debug Print Instrumentation (`createDebugPrintInstrumentation()`)

Controls debug output and logging during pass execution via the PassInstrumentation callback interface:

```cpp
std::shared_ptr<stinkytofu::PassInstrumentation> createDebugPrintInstrumentation()
{
    auto debugConfig = std::make_unique<stinkytofu::PassManagerDebugConfig>();
    debugConfig->setPrintBeforeAll(true);     // Print IR before each pass
    debugConfig->setPrintAfterAll(true);      // Print IR after each pass
    debugConfig->setDumpToFileInBefore("before.txt");  // Save pre-pass state
    debugConfig->setDumpToFileInAfter("after.txt");    // Save post-pass state
    return std::make_shared<stinkytofu::DebugPrintInstrumentation>(std::move(debugConfig));
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
        {12, 5, 0},  // arch: GPU architecture version (e.g., gfx1250)
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

StinkyTofu IR uses an MLIR-style text format. Each instruction has the form:

```
[destRegs =] "st.<mnemonic>"(srcRegs) { issueCycles = N, latencyCycles = N [, mod.X = { ... }, ...] }
```

### Field Descriptions

- **destRegs** (optional): Destination register(s), e.g. `v[0]`, `v[34:37]`, `acc[0:3]`. Omit for instructions with no destination. Follow with `=`.
- **"st.\<mnemonic>"**: Quoted operation name with `st.` namespace (e.g. `"st.ds_read_b128"`, `"st.v_mfma_f32_16x16x16_f16"`).
- **srcRegs**: Comma-separated source operands in parentheses (registers, immediates, BARRIER[0], etc.).
- **attributes** (in braces): Required **issueCycles** and **latencyCycles** (integer values). Optional modifier attributes: **mod.X = { field = value, ... }** for each modifier type (e.g. `mod.ds`, `mod.flat`, `mod.mfma`). Use `-1` for default cycle values when appropriate.

### Modifier Attributes

Modifiers are serialized as attribute keys `mod.<type>` with a dict value. Examples:

- **mod.ds**: `mod.ds = { na = 1, offset = 0, offset0 = 0, offset1 = 0, gds = false }`
- **mod.flat**: `mod.flat = { offset12 = 0, glc = false, slc = false, lds = false }`
- **mod.mfma**: `mod.mfma = { inputPermute = "...", scaleStr = "...", negStr = "...", reuseA = false, reuseB = false, neg_lo = false, neg_hi = false }`

Other modifier types (e.g. `mod.vop3`, `mod.swaitcnt`, `mod.delayalu`) follow the same pattern. The serializer emits only fields that apply to the modifier instance.

### Example IR

```
v[34:37] = "st.ds_read_b128"(v[5]) { issueCycles = 4, latencyCycles = 52, mod.ds = { na = 1, offset = 0, gds = false } }

acc[0:3] = "st.v_mfma_f32_16x16x16_f16"(v[6:7], v[22:23], acc[0:3]) { issueCycles = 4, latencyCycles = 16 }
```

Hierarchical format with functions and blocks:

```
st.func @kernel() {
^entry:
  v[0] = "st.v_add_f32"(v[1], v[2]) { issueCycles = 4, latencyCycles = 8 }
Successors: ^next
^next:
  ...
}
```

See `tests/unit/asm/IRParserTest.cpp` and the serialization layer (`StinkyAsmPrinter`, `IRParser`) for more examples.
