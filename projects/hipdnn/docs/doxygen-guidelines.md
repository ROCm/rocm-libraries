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
