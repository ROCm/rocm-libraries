# StinkyTofu Error Codes

This document describes the error codes used throughout the StinkyTofu library.

## StinkyErrorCode

Error codes for 'StinkyIRConverter' operations and IR conversion functions.

'''cpp
enum class StinkyErrorCode : int
{
    SUCCESS       = 0,  // Operation completed successfully
    PASSCTX_EMPTY = 1,  // PassContext is invalid or empty
};
'''

### Error Code Descriptions

| Code | Value | Description |
|------|-------|-------------|
| 'StinkyErrorCode::SUCCESS' | 0 | Operation completed successfully |
| 'StinkyErrorCode::PASSCTX_EMPTY' | 1 | PassContext is invalid or empty. This typically occurs when trying to use a PassContext that hasn't been properly initialized or has been cleaned up. |

### Usage

#### Checking Return Values

'''cpp
StinkyErrorCode result = StinkyIRConverter::populateFunctionFromString(irText, irlist, passCtx, arch);
if(result != StinkyErrorCode::SUCCESS) {
    // Handle error
    if(result == StinkyErrorCode::PASSCTX_EMPTY) {
        std::cerr << "PassContext is empty or invalid\n";
    }
}
'''

#### Converting to int

If you need to convert the error code to an integer (e.g., for logging):

'''cpp
StinkyErrorCode result = /* ... */;
int errorValue = static_cast<int>(result);
std::cerr << "Error code: " << errorValue << "\n";
'''

### Functions That Return StinkyErrorCode

- 'StinkyIRConverter::populateFunctionFromString()' - Returns error codes when populating an IRList from a string

### Error Handling Best Practices

1. **Always check return values** - Don't assume operations will succeed
2. **Use specific error codes** - Check for specific error codes rather than just non-SUCCESS
3. **Log errors appropriately** - Include the error code value in error messages for debugging
4. **Clean up on error** - Make sure to clean up resources when errors occur

### Example: Complete Error Handling

'''cpp
#include "stinkytofu/core/stinkytofu.hpp"

using namespace stinkytofu;

int processIR(const std::string& irText) {
    // Create a PassContext
    auto ctx = PassContext::create(GfxArchID::gfx1250);
    if(!ctx) {
        std::cerr << "Failed to create PassContext\n";
        return -1;
    }

    IRList& irlist = ctx->getIRList();

    // Convert IR string to IRList
    StinkyIRConverter converter;
    IRList* result = converter.convertToIRList(irText);

    if(!result) {
        std::cerr << "Failed to convert IR string to IRList\n";
        return -1;
    }

    // Or use the static method directly:
    PassContext* passCtx = converter.getPassContext();
    if(!passCtx) {
        std::cerr << "Failed to get PassContext\n";
        return -1;
    }

    GfxArchID arch = GfxArchID::gfx1250;
    StinkyErrorCode errorCode = StinkyIRConverter::populateFunctionFromString(
        irText, irlist, *passCtx, arch);

    if(errorCode != StinkyErrorCode::SUCCESS) {
        std::cerr << "Failed to populate IRList. Error: ";

        switch(errorCode) {
            case StinkyErrorCode::PASSCTX_EMPTY:
                std::cerr << "PassContext is empty\n";
                break;
            default:
                std::cerr << "Unknown error (" << static_cast<int>(errorCode) << ")\n";
                break;
        }

        return static_cast<int>(errorCode);
    }

    // Process the IRList...

    return 0;
}
'''

## Future Error Codes

As the library evolves, additional error codes may be added to 'StinkyErrorCode'. Always check the return value against 'SUCCESS' rather than assuming any non-zero value is an error, as error codes may be expanded in the future.
