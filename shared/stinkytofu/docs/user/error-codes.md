# StinkyTofu Error Codes

This document describes the error codes the StinkyTofu library returns, and the ids its diagnostics carry.

## StinkyErrorCode

Error codes for `StinkyIRConverter` operations and IR conversion functions.

```cpp
enum class StinkyErrorCode : int
{
    SUCCESS       = 0,  // Operation completed successfully
    PASSCTX_EMPTY = 1,  // No library function returns this now
    PARSE_ERROR   = 2,  // The text does not parse
    CONVERT_ERROR = 3,  // An instruction or block edge cannot be built for the arch
};
```

### Error Code Descriptions

| Code | Value | Description |
|------|-------|-------------|
| `StinkyErrorCode::SUCCESS` | 0 | Operation completed successfully |
| `StinkyErrorCode::PASSCTX_EMPTY` | 1 | The `PassContext` was invalid or empty. No library function returns it now. |
| `StinkyErrorCode::PARSE_ERROR` | 2 | The IR text does not parse, or the parsed function has no blocks. |
| `StinkyErrorCode::CONVERT_ERROR` | 3 | The text parsed, but an instruction has no opcode or hardware descriptor for the arch, or a successor label names no block. `populateFunctionFromParsed` reports each one as an error `Diagnostic` and still builds the rest; do not use the `Function` it built. `replaceBlockBody` reports them the same way and leaves the block unchanged. |

### Usage

#### Checking Return Values

```cpp
Function func("kernel");
PassContext passCtx;
StinkyErrorCode result =
    StinkyIRConverter::populateFunctionFromString(irText, func, passCtx, GfxArchID::Gfx1250);
if (result != StinkyErrorCode::SUCCESS) {
    // populateFunctionFromString has printed each diagnostic to stderr.
    std::cerr << "Failed to load the IR: error code " << static_cast<int>(result) << "\n";
}
```

#### Converting to int

If you need to convert the error code to an integer (e.g., for logging):

```cpp
StinkyErrorCode result = /* ... */;
int errorValue = static_cast<int>(result);
std::cerr << "Error code: " << errorValue << "\n";
```

### Functions That Return StinkyErrorCode

- `StinkyIRConverter::populateFunctionFromString()` - Returns `PARSE_ERROR` when the text does not parse, and otherwise what `populateFunctionFromParsed()` returns. It prints the diagnostics to stderr
- `StinkyIRConverter::populateFunctionFromParsed()` - Returns `PARSE_ERROR` or `CONVERT_ERROR`, with the reasons in its diagnostics
- `StinkyIRConverter::replaceBlockBody()` - Returns `CONVERT_ERROR`, with the reasons in its diagnostics, and leaves the block unchanged

`StinkyIRConverter::convertToFunction()` returns a `Function*` instead, which is null when the text does not load.

### Error Handling Best Practices

1. **Always check return values** - Don't assume operations will succeed
2. **Use specific error codes** - Check for specific error codes rather than just non-`SUCCESS`
3. **Read the diagnostics** - The code says that a call failed; each `Diagnostic` says where and why. Print it with `format()`, and match on its id rather than its message
4. **Clean up on error** - Make sure to clean up resources when errors occur

### Example: Complete Error Handling

```cpp
#include <iostream>
#include <string>
#include <vector>

#include "stinkytofu/core/stinkytofu.hpp"
#include "stinkytofu/serialization/asm/IRParser.hpp"

using namespace stinkytofu;

int loadFunction(const std::string& irText, Function& func) {
    ParseResult parsed = parseSourceStringWithDiagnostics(irText);
    for (const Diagnostic& diagnostic : parsed.diagnostics)
        std::cerr << diagnostic.format() << "\n";
    if (parsed.hasErrors() || !parsed.parsedFunction)
        return static_cast<int>(StinkyErrorCode::PARSE_ERROR);

    std::vector<Diagnostic> diagnostics;
    StinkyErrorCode errorCode = StinkyIRConverter::populateFunctionFromParsed(
        *parsed.parsedFunction, func, GfxArchID::Gfx1250, diagnostics);
    for (const Diagnostic& diagnostic : diagnostics) {
        std::cerr << diagnostic.format() << "\n";
        if (diagnostic.getId() == DiagId::UnknownMnemonic)
            std::cerr << "  Check the opcode's spelling for the arch\n";
    }

    switch (errorCode) {
        case StinkyErrorCode::SUCCESS:
            return 0;
        case StinkyErrorCode::PARSE_ERROR:
            std::cerr << "The function has no blocks\n";
            break;
        case StinkyErrorCode::CONVERT_ERROR:
            std::cerr << "Instructions or edges were left out; do not use the function\n";
            break;
        default:
            std::cerr << "Unknown error (" << static_cast<int>(errorCode) << ")\n";
            break;
    }
    return static_cast<int>(errorCode);
}
```

For text with a typo in an opcode, this prints the diagnostic and returns `CONVERT_ERROR`:

```text
3:3: error: Unknown mnemonic 's_nopp' for gfx1250 [unknown-mnemonic]
  Check the opcode's spelling for the arch
Instructions or edges were left out; do not use the function
```

## Diagnostic ids

Every diagnostic from the parsers, the converter, and the verifier carries an id. An id is one stable name for a kind of failure, taken from `include/stinkytofu/support/DiagnosticIds.def`. `Diagnostic::format()` prints it at the end of the line, and Python's `Diagnostic.id` returns it:

```text
3:3: error: Unknown mnemonic 's_nopp' for gfx1250 [unknown-mnemonic]
@k ^entry #0: error: 'ds_load_b128' operand dest[0] has register width 2, expected 4 [register-width]
```

A diagnostic about text starts with `line:column`. One about built IR starts with `@function ^block #index`. The index counts the block's instructions without its labels, as Python's `instructions()` does. Match on the id, not the message: a message can be reworded, but an id keeps its spelling.

| Id | Reported for |
| --- | --- |
| `syntax` | Text that does not follow the grammar: an `Expected ...` error, an early end of input, or a block header, `Successors:` line, or `}` in a block body |
| `unknown-register-type` | A register whose class StinkyTofu does not know, such as `q[0]` |
| `out-of-range` | A number that does not fit its field, or a reversed register range such as `v[3:1]` |
| `alignment-without-label` | A block header that gives `alignment` but not `label` |
| `missing-attributes` | Warning. An instruction without `issueCycles` or `latencyCycles` |
| `unknown-attribute` | Warning. An instruction or label attribute StinkyTofu does not read |
| `unknown-mnemonic` | An opcode the arch does not have. An error in STIR. In assembler source it is a warning, and the line is kept as text |
| `unreadable-operands` | Warning. Assembler source whose operands could not be read. The line is kept as text |
| `no-hardware-descriptor` | An instruction with no hardware descriptor for the arch |
| `unknown-successor` | A `Successors:` entry that names no block in the function |
| `cannot-read-file` | A pattern file that could not be opened |
| `empty-function` | A function with no blocks, or with no instructions |
| `logical-ir` | Logical IR left in an assembly function |
| `missing-operand` | An instruction without an operand its encoding needs |
| `register-width` | A register operand of the wrong width |
| `register-type` | A register operand of the wrong register class |
| `read-write-operand` | A register the instruction both reads and writes that is missing from its defs or its uses |
| `undefined-label` | A branch to a label the function does not define. Only `verifyStinkyIR()` reports it, not `validateStinkyIR()` |
| `invalid-ssa` | Attached SSA that breaks an invariant. Only `verifyStinkyIR()` reports it, not `validateStinkyIR()` |

## Future Error Codes

As the library evolves, additional error codes may be added to `StinkyErrorCode`. Compare the return value with `StinkyErrorCode::SUCCESS` rather than with `0`, and treat a code you do not recognize as a failure.
