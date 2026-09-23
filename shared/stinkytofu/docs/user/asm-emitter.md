# StinkyAsmEmitter User Guide

## Overview

'StinkyAsmEmitter' converts StinkyTofu IR (MLIR-style intermediate representation) into actual GPU assembly code that can be processed by assemblers and compilers.

## Basic Usage

### Simple Assembly Emission

'''cpp
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/core/stinkytofu.hpp"

using namespace stinkytofu;

// Assuming you have an IRList populated with instructions
IRList irlist;
// ... populate irlist ...

// Create an emitter with default options
StinkyAsmEmitter emitter;

// Emit assembly code as a string
std::string assembly = emitter.emit(irlist);
std::cout << assembly << std::endl;
'''

### Emit to Stream

'''cpp
// Emit directly to a stream (e.g., file)
std::ofstream outFile("output.s");
StinkyAsmEmitter emitter;
emitter.emit(outFile, irlist);
outFile.close();
'''

### Using the Utility Function

'''cpp
// Quick conversion using the toAssembly() utility
std::string assembly = toAssembly(irlist);
'''

## Configuration Options

The 'AsmEmitterOptions' struct allows you to customize the assembly output:

'''cpp
struct AsmEmitterOptions {
    bool emitComments   = true;   // Include header comments
    bool emitCycleInfo  = true;   // Emit cycle count information
    int  indent         = 4;      // Instruction indentation (spaces)
    bool emitBlankLines = false;  // Add blank lines between instruction groups
};
'''

### Example: Custom Configuration

'''cpp
AsmEmitterOptions options;
options.emitComments  = false;  // No header comments
options.emitCycleInfo = true;   // Show cycle counts
options.indent        = 8;      // Use 8-space indentation

StinkyAsmEmitter emitter(options);
std::string assembly = emitter.emit(irlist);
'''

## Complete Example

Here's a complete example showing how to create instructions and emit them as assembly:

'''cpp
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include "stinkytofu/core/stinkytofu.hpp"
#include <iostream>

using namespace stinkytofu;

int main() {
    // Create a PassContext (manages IR and builders)
    auto ctx = PassContext::create(GfxArchID::gfx1250);

    // Get the IRList and builder
    IRList& irlist = ctx->getIRList();
    AsmIRBuilder* builder = ctx->getIRBuilder();

    // Create a label (builder must be bound to a block, e.g. via setInsertionPoint)
    builder->createLabel("loop_start");

    // Create a ds_read instruction
    StinkyInstruction* inst1 = builder->createStinkyInstruction(
        "ds_read_b128", irlist.end());
    inst1->destRegs.push_back(StinkyRegister("v", 0, 4));   // v[0:3]
    inst1->srcRegs.push_back(StinkyRegister("v", 40, 1));   // v[40]
    inst1->issueCycles   = 4;
    inst1->latencyCycles = 52;

    // Create another ds_read instruction
    StinkyInstruction* inst2 = builder->createStinkyInstruction(
        "ds_read_b128", irlist.end());
    inst2->destRegs.push_back(StinkyRegister("v", 4, 4));   // v[4:7]
    inst2->srcRegs.push_back(StinkyRegister("v", 40, 1));   // v[40]
    inst2->issueCycles   = 4;
    inst2->latencyCycles = 52;

    // Configure emission options
    AsmEmitterOptions options;
    options.emitComments  = true;
    options.emitCycleInfo = true;
    options.indent        = 4;

    // Emit assembly
    StinkyAsmEmitter emitter(options);
    std::string assembly = emitter.emit(irlist);

    // Print the result
    std::cout << assembly << std::endl;

    return 0;
}
'''

### Expected Output

'''assembly
// ==================================================
// StinkyTofu Assembly Output
// Instructions: 3
// ==================================================

loop_start:
    ds_read_b128 v[0:3], v[40] // issue=4 latency=52
    ds_read_b128 v[4:7], v[40] // issue=4 latency=52
'''

## Without Cycle Information

If you don't want cycle count comments:

'''cpp
AsmEmitterOptions options;
options.emitCycleInfo = false;

StinkyAsmEmitter emitter(options);
std::string assembly = emitter.emit(irlist);
'''

### Output

'''assembly
// ==================================================
// StinkyTofu Assembly Output
// Instructions: 3
// ==================================================

loop_start:
    ds_read_b128 v[0:3], v[40]
    ds_read_b128 v[4:7], v[40]
'''

## Minimal Output (No Comments)

For production assembly without any comments:

'''cpp
AsmEmitterOptions options;
options.emitComments  = false;
options.emitCycleInfo = false;

StinkyAsmEmitter emitter(options);
std::string assembly = emitter.emit(irlist);
'''

### Output

'''assembly
loop_start:
    ds_read_b128 v[0:3], v[40]
    ds_read_b128 v[4:7], v[40]
'''

## Emitting Single Instructions

You can also emit individual instructions:

'''cpp
StinkyInstruction* inst = /* ... get instruction ... */;

AsmEmitterOptions options;
options.emitCycleInfo = true;

StinkyAsmEmitter emitter(options);
std::string assembly = emitter.emit(*inst);
// Output: "    ds_read_b128 v[0:3], v[40] // issue=4 latency=52\n"
'''

## Common Use Cases

### 1. Debug Output
Use default options with comments and cycle info to understand instruction timing:
'''cpp
std::cout << toAssembly(irlist) << std::endl;
'''

### 2. Assembly File Generation
Generate clean assembly for the assembler:
'''cpp
AsmEmitterOptions options;
options.emitComments  = false;
options.emitCycleInfo = false;

std::ofstream asmFile("kernel.s");
StinkyAsmEmitter emitter(options);
emitter.emit(asmFile, irlist);
asmFile.close();
'''

### 3. Annotated Assembly for Analysis
Include cycle information for performance analysis:
'''cpp
AsmEmitterOptions options;
options.emitComments  = true;
options.emitCycleInfo = true;

std::ofstream annotated("kernel_annotated.s");
StinkyAsmEmitter emitter(options);
emitter.emit(annotated, irlist);
annotated.close();
'''

## Register Formats

The emitter handles different register types correctly:

| Register Type | Example | Assembly Output |
|--------------|---------|-----------------|
| Vector (single) | 'StinkyRegister("v", 5, 1)' | 'v[5]' |
| Vector (range) | 'StinkyRegister("v", 0, 4)' | 'v[0:3]' |
| Accumulator | 'StinkyRegister("acc", 0, 16)' | 'acc[0:15]' |
| Scalar | 'StinkyRegister("s", 10, 1)' | 's10' |
| Literal int | 'StinkyRegister(0)' | '0' |

## Integration with Pass Pipeline

The emitter works seamlessly with the pass manager:

'''cpp
// Create context and run optimization passes
auto ctx = PassContext::create(GfxArchID::gfx1250);
IRList& irlist = ctx->getIRList();

// ... populate irlist and run passes ...

PassManager pm;
pm.addPass(std::make_unique<SomeOptimizationPass>());
pm.run(*ctx);

// Emit optimized assembly
AsmEmitterOptions options;
options.emitComments = true;
std::string assembly = toAssembly(irlist, options);
std::cout << assembly << std::endl;
'''

## See Also

- [Error Codes](error-codes.md) - Common error codes and their meanings
