## Error handling

HostNumerics-owned terminal errors throw `std::runtime_error`.
Expected support and compatibility failures should use explicit results
or checks. Allocation, system, dependency and visitor errors retain their
original types. Dispatch rejects invalid enum values before invoking the visitor
or moving its arguments. This policy does not promise exceptions-disabled builds.

## Internal C++ results

`detail::Expected<T, E>` in `result.hpp` represents either a value or an error.
Use `detail::Unexpected(error)` to construct a failure, check `has_value()` or
its boolean conversion, and use `*`/`->` or `error()` only in the matching state.
The wrapper stores its payload inline and adds no allocation or exception
translation. An owning payload can still allocate or throw.

This follows the core `std::expected<T, E>` model while retaining C++20 support;
HostNumerics cannot yet assume C++23 and compatible standard libraries across
its consumers. It deliberately omits assignment, void values, converting result
constructors, throwing `value()` access, fallback access, comparisons, and
monadic operations. Assignment is disabled to keep the stored variant from
becoming valueless after a throwing payload assignment. The utility remains an
implementation helper in `detail`, rather than a new supported public API.
