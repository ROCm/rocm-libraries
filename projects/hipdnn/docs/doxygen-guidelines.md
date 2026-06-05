# Doxygen Documentation Guidelines

## Public API Scope

The following directories contain public API that **requires** Doxygen documentation:

| Directory | Scope | Notes |
|-----------|-------|-------|
| `backend/include/` | All `.h` files | C API headers (hipdnn_backend.h, enums, status codes) |
| `frontend/include/hipdnn_frontend/` | All `.hpp` files **except** `detail/` subdirectory | Frontend classes, attributes, error handling |
| `frontend/include/hipdnn_frontend/attributes/` | All attribute classes | Operation configuration (convolution, batchnorm, matmul, pointwise) |
| `frontend/include/hipdnn_frontend/knob/` | Knob system | Engine configuration knobs |
| `plugin_sdk/include/` | Plugin interfaces | For plugin developers |

## What to Exclude

- `**/detail/**` subdirectories - internal implementation
- `**/src/**` directories - private implementation files
- Test files (`**/tests/**`)
- Generated files (`*_generated.h`, `*_generated.hpp`)

## Required Documentation Elements

**File-level (all public headers):**
```cpp
/**
 * @file FileName.hpp
 * @brief One-line description of the file's purpose
 *
 * Optional longer description of what the file contains and when to use it.
 */
```

**Class-level:**
```cpp
/**
 * @class ClassName
 * @brief One-line description
 *
 * Detailed description of the class purpose and usage.
 *
 * @code{.cpp}
 * // Example usage
 * ClassName obj;
 * obj.doSomething();
 * @endcode
 *
 * @see RelatedClass, relatedFunction()
 */
```

**Function/Method-level:**
```cpp
/**
 * @brief Brief description of what the function does
 * @param paramName Description of the parameter
 * @return Description of return value
 * @throws ExceptionType When this is thrown
 * @see relatedFunction()
 */
```

**Enum values:**
```cpp
enum class MyEnum
{
    VALUE_A,  ///< Description of VALUE_A
    VALUE_B,  ///< Description of VALUE_B
    VALUE_C   ///< Description of VALUE_C
};
```

## Graph Operation Methods

`Graph.hpp` is the primary entry point for users. Operation methods (e.g., `conv_fprop`, `matmul`, `pointwise`, `batchnorm`) should include:

1. A `@brief` one-liner
2. A short formula showing the mathematical operation (use `@code` blocks)
3. `@param` tags documenting each input tensor with its expected shape
4. `@return` documenting each output tensor with its shape
5. A `@see` reference to the corresponding attributes class

```cpp
/** @brief Convolution forward pass
 *
 * For 2D with NCHW layout:
 * @code
 * y[n,k,oh,ow] = sum_c,r,s  x[n, c, oh*stride+r*dilation-pad, ...]  *  w[k,c,r,s]
 * @endcode
 *
 * @param x Input activation tensor [N, C, H, W]
 * @param w Filter/weight tensor [K, C, R, S]
 * @param attributes Convolution parameters: padding, stride, dilation
 * @return y: Output activation tensor [N, K, OH, OW]
 *
 * @see ConvFpropAttributes
 */
```

When adding a new operation to `Graph.hpp`, add a Doxygen comment following this pattern before the method declaration.

## Doxygen Comment Style

- Use `/** ... */` for multi-line Doxygen comments
- Use `///< ` for inline enum/member documentation
- **Never use `/* */` inside Doxygen blocks** - this causes nested comment warnings
- Use `@code{.cpp} ... @endcode` for code examples

## When Adding New Public API

1. Add file-level `@file` and `@brief` documentation
2. Document all public classes with `@class`, `@brief`, and usage example
3. Document all public methods with `@brief`, `@param`, `@return`
4. Document all enum values with `///<` inline comments
5. Run `doxygen Doxyfile` to verify documentation builds without warnings
