# IR Converter - Converting Strings to IRList

StinkyTofu provides the 'StinkyIRConverter' class to programmatically convert MLIR-style instruction strings to 'IRList' objects.

> **Note:** For detailed information about error codes, see [Error Codes](error-codes.md).

## Basic Usage

'''cpp
#include "stinkytofu/core/stinkytofu.hpp"

using namespace stinkytofu;

std::string irText = R"(
  v[0:3] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 56 }
  v[4:7] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 56 }
)";

// Create converter with default architecture (gfx1250)
StinkyIRConverter converter;

// Convert string to IRList
IRList* irlist = converter.convertToIRList(irText);

if(irlist) {
    // Successfully converted, use the IRList
    std::cout << "Converted " << irlist->size() << " instructions\n";
} else {
    std::cerr << "Conversion failed\n";
}
'''

## Custom Architecture

You can specify a different target architecture:

'''cpp
// For gfx1250
std::array<int, 3> arch = {12, 5, 0};
StinkyIRConverter converter(arch);

IRList* irlist = converter.convertToIRList(irText);
'''

## Accessing the PassContext

After conversion, you can access the 'PassContext' to run optimization passes:

'''cpp
StinkyIRConverter converter;
IRList* irlist = converter.convertToIRList(irText);

if(irlist) {
    PassContext* ctx = converter.getPassContext();
    if(ctx) {
        // Run passes on the IR
        PassManager pm;
        pm.addPass(std::make_unique<SomeOptimizationPass>());
        pm.run(*ctx);
    }
}
'''

## Using the Static Method

For more control, use the static 'populateFunctionFromString' method:

'''cpp
// Create your own PassContext
auto ctx = PassContext::create(GfxArchID::gfx1250);
IRList& irlist = ctx->getIRList();

// Populate it with instructions from string
std::string irText = /* ... */;
GfxArchID arch = GfxArchID::gfx1250;

StinkyErrorCode result = StinkyIRConverter::populateFunctionFromString(
    irText, irlist, *ctx, arch);

if(result == StinkyErrorCode::SUCCESS) {
    // Successfully populated
    std::cout << "Populated " << irlist.size() << " instructions\n";
} else {
    // Handle error
    std::cerr << "Error: " << static_cast<int>(result) << "\n";
}
'''

## Complete Example

Here's a complete example showing conversion and assembly emission:

'''cpp
#include "stinkytofu/core/stinkytofu.hpp"
#include "stinkytofu/serialization/asm/StinkyAsmEmitter.hpp"
#include <iostream>

using namespace stinkytofu;

int main() {
    // IR text with two ds_read instructions
    std::string irText = R"(
        v[0:3] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 52 }
        v[4:7] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 52 }
    )";

    // Convert to IRList
    StinkyIRConverter converter;
    IRList* irlist = converter.convertToIRList(irText);

    if(!irlist) {
        std::cerr << "Failed to convert IR\n";
        return 1;
    }

    std::cout << "Converted " << irlist->size() << " instructions\n";

    // Emit as assembly
    StinkyAsmEmitter emitter;
    std::string assembly = emitter.emit(*irlist);

    std::cout << "\nGenerated Assembly:\n" << assembly << std::endl;

    return 0;
}
'''

### Expected Output

'''
Converted 2 instructions

Generated Assembly:
// ==================================================
// StinkyTofu Assembly Output
// Instructions: 2
// ==================================================

    ds_read_b128 v[0:3], v[40] // issue=4 latency=52
    ds_read_b128 v[4:7], v[40] // issue=4 latency=52
'''

## IR Format

The IR string should follow the MLIR-style StinkyTofu format:

'''
destRegs = "st.mnemonic"(srcRegs) { attributes }
'''

### Examples

**Simple instruction:**
'''
v[0] = "st.v_mov_b32"(v1)
'''

**Multiple destinations:**
'''
v[0:3] = "st.ds_load_b128"(v40)
'''

**With attributes:**
'''
v[0:3] = "st.ds_load_b128"(v40) { issueCycles = 4, latencyCycles = 52 }
'''

**Multiple source registers:**
'''
acc[0:15] = "st.v_mfma_f32_16x16x16_f16"(v6:7, v22:23, acc0:15)
'''

## Error Handling

Always check the return value:

'''cpp
StinkyIRConverter converter;
IRList* irlist = converter.convertToIRList(irText);

if(!irlist) {
    std::cerr << "Conversion failed. Possible reasons:\n"
              << "  - Invalid IR syntax\n"
              << "  - Unsupported instruction mnemonic\n"
              << "  - Invalid register format\n";
    return 1;
}
'''

For the static method, check the error code:

'''cpp
StinkyErrorCode result = StinkyIRConverter::populateFunctionFromString(
    irText, irlist, *ctx, arch);

switch(result) {
    case StinkyErrorCode::SUCCESS:
        // Success
        break;
    case StinkyErrorCode::PASSCTX_EMPTY:
        std::cerr << "PassContext is empty or invalid\n";
        break;
    default:
        std::cerr << "Unknown error\n";
        break;
}
'''

## Integration with Optimization Passes

After converting IR, you can run optimization passes:

'''cpp
// Convert IR
StinkyIRConverter converter;
IRList* irlist = converter.convertToIRList(irText);

if(irlist) {
    PassContext* ctx = converter.getPassContext();
    if(ctx) {
        // Create pass manager and add passes
        PassManager pm;
        pm.addPass(std::make_unique<StinkyDAGSchedulerPass>());
        pm.addPass(std::make_unique<StinkyWaitCntInsertionPass>());

        // Run the passes
        pm.run(*ctx);

        // Emit optimized assembly
        StinkyAsmEmitter emitter;
        std::string optimizedAsm = emitter.emit(*irlist);
        std::cout << optimizedAsm << std::endl;
    }
}
'''

## Thread Safety

'StinkyIRConverter' is not thread-safe. If you need to convert IR in multiple threads, create a separate converter instance for each thread:

'''cpp
// In each thread:
void processIRInThread(const std::string& irText) {
    StinkyIRConverter converter;  // Thread-local instance
    IRList* irlist = converter.convertToIRList(irText);
    // ... process irlist ...
}
'''

## See Also

- [Assembly Emitter Guide](asm-emitter.md) - Converting IR to assembly code
- [Error Codes Reference](error-codes.md) - Error codes and handling
