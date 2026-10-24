## Error handling

HostNumerics-owned terminal errors throw `std::runtime_error`.
Expected support and compatibility failures should use explicit results
or checks. Allocation, system, dependency and visitor errors retain their
original types. Dispatch rejects invalid enum values before invoking the visitor
or moving its arguments. This policy does not promise exceptions-disabled builds.

## Internal C++ results

Failures that a caller can handle are returned rather than thrown.
`detail::Expected<T, E>` in `expected.hpp` holds either a value or an error, such
as an enum that names the failure. Return `detail::Unexpected(error)` for a
failure, check `has_value()` or the boolean conversion, then read the value with
`*`/`->` or the error with `error()`. To pass a failure on to a caller that
expects a different value type, return `detail::Unexpected(result.error())`.

`Expected` is a C++20 subset of C++23 `std::expected`. `expected.hpp` explains
why, and how it will be replaced. It is internal, with no source or ABI
compatibility promise.

Copy and move assignment are available when both payload types have nonthrowing
move constructors and support the corresponding assignment. This keeps the
stored value or error present even if copying a replacement throws.
`Expected<void, E>` is not supported; add and test that specialization alongside
the first check that needs to return only success or an error.
