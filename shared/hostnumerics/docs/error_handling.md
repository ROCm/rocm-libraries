## Error handling

HostNumerics-owned terminal errors throw `std::runtime_error`.
Expected support and compatibility failures should use explicit results
or checks. Allocation, system, dependency and visitor errors retain their
original types. Dispatch rejects invalid enum values before invoking the visitor
or moving its arguments. This policy does not promise exceptions-disabled builds.

## Internal C++ results

`detail::Expected<T, E>` in `result.hpp` owns either a value or a typed failure.
Construct a failure with `detail::Unexpected(error)`, check `has_value()` or the
boolean conversion, then access the active payload with `*`/`->` or `error()`.
Accessing the inactive alternative is a programming error checked by assertions.

The helper is HostNumerics code under the existing MIT license. It wraps
`std::variant<T, E>`, which manages inline storage, copy, move, and destruction.
No third-party implementation is copied or bundled. The wrapper adds no allocation
or exception translation; an owning payload can still allocate or throw.
Constructors preserve payload-based `noexcept` and support in-place values and
move-only payloads. Observers preserve constness and value category.

This is an internal value-or-error interface for C++20, with no source or ABI
compatibility promise. It deliberately omits assignment, void values, converting
result constructors, throwing `value()` access, fallback access, comparisons, and
monadic operations. Propagate a value or tagged failure explicitly when changing
result types. Assignment is disabled so throwing payload assignment cannot leave
a result's variant valueless; construct another result instead.
