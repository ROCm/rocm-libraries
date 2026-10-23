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

## Layout boundaries

`Layout::elementOffset` returns `detail::Expected<ptrdiff_t, detail::LayoutFailure>`
and is `noexcept`. Rank, coordinate, and arithmetic errors retain an error code
and numeric context. Checking an offset neither allocates nor formats diagnostics.
The Python `element_offset` binding reports these errors as `RuntimeError` with
the same rank, coordinate, and overflow messages.

`Layout::getContiguousLayout` returns validation errors through `Expected`.
It copies the Shape and allocates strides, so allocation errors can still throw.
`Layout::contiguous` is a convenience boundary that reports validation errors as
`std::runtime_error`. Both use orders listing axes from slowest to fastest, as in
`Shape::linearIndex`.

Assertions check established internal preconditions. Public checked operations
return errors for invalid input in both debug and release builds. Returning an
error does not require callers to choose a fallback: an outer boundary can
report it as an exception when that operation cannot continue.
