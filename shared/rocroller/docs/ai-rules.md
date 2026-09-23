# # AI Rules for rocRoller

This file provides guidance to AI agents when working with code in this repository.

## Overview

RocRoller is a software library for generating optimized AMDGPU assembly kernels.
It transforms high-level kernel specifications (`Command`) through a dual-graph IR (`KernelGraph` with `ControlGraph` + `CoordinateGraph`) into optimized GPU assembly, then assembles to binary via AMD Comgr and executes via HIP.

## Conventions

- **Default layout is column-major.** N (non-transposed) = column-major, T (transposed) = row-major.

## Git Workflow

- **Main branch**: `develop` (use for PRs, not `main` or `master`)
- **Branch naming**: `users/<username>/<feature-name>`

## File Structure Conventions

- `Foo.hpp`: Class/concept definitions with declaration-only functions
- `Foo_impl.hpp`: Short inlinable function definitions (included at bottom of `Foo.hpp`)
- `Foo_fwd.hpp`: Forward declarations, type aliases (e.g., `FooPtr = std::shared_ptr<Foo>`), minimal includes
- `Foo.cpp`: Longer function definitions

## Coding Style

- **Formatting**: Follows `clang-format` version 13 (`./scripts/fix-format`)
- **Functions**: Static/free functions start with uppercase; instance functions start with lowercase
- **Variables**: Private members start with `m_`; public members do not
- **Naming**: camelCase (not snake_case)
- **Macros/CMake**: `UPPER_CASE` with `ROCROLLER_` prefix
- **C++ Standard**: C++20 is available but prefer C++17 modern practices unless explicitly reviewed
- **Memory**: Use `std::make_unique`/`std::make_shared` instead of `new`; use `std::vector` for arrays
- **Type aliases**: Use `using` instead of `typedef`

### Adding New Tests

- **Preferred**: Add Catch2 tests in `test/catch/` → builds `rocroller-tests-catch`
- **Legacy**: GTest tests in `test/unit/` → builds `rocroller-tests`

### Debugging and Error Handling

- Use `Log::debug()` etc. from `Utilities/Logging.hpp` for debug output.
- Prefer `AssertFatal` with `ShowValue()` over `assert`.
