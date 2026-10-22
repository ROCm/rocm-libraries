## Error handling

Expected compatibility, support and descriptor-translation failures use explicit
results or checks. Callers inspect those results to select a backend, materialize
an operand, or reject a descriptor. They must not infer recovery decisions from
an exception type or its message.

Internal fallible helpers return `detail::Expected<T, E>`, a C++20 subset of the
`std::expected` model. Values construct directly; errors use `detail::Unexpected`.
The result provides `[[nodiscard]]`, boolean/`has_value()` checks, value access,
and `error()` for propagation. It retains C++20 consumer compatibility; unused
standard APIs are omitted. Layout, storage, scalar conversion and
descriptor checks use a closed failure vocabulary; numerical conversion also carries the
destination width and signedness. Diagnostics are obtained from the failure at
reporting boundaries, without error-message output parameters. Checks needing
dynamic diagnostic text can carry an owned string in the same result interface.

The integer-conversion leaf is independent of throwing tensor/metadata headers
and is tested with exception handling disabled and allocation attempts fatal.
Its success and failure paths allocate no storage. Result payload construction
retains its ordinary exception behaviour; the wrapper does not make allocating
operations or arbitrary user conversions nonthrowing.

HostNumerics-owned terminal errors throw `std::runtime_error`. Boundary handlers may report these failures, cleanup may catch and
rethrow, and worker code may transport the original exception to the joining
caller. Allocation, system, dependency and user-callback exceptions retain their
original types. Python sequence/index protocols retain their native exceptions;
HostNumerics-owned operation errors become Python `RuntimeError`.

TensileLite's benchmark client may catch a terminal error to report the current
solution as `INVALID` and continue to the next solution. This is a failed-operation
reporting boundary; it does not select a numerical backend or retry the operation.

This policy does not promise exceptions-disabled builds or transactional output:
a failed operation can have written part of a caller-owned output.

## Internal C++ results

`detail::Expected<T, E>` in `expected.hpp` owns either a value or a typed failure.
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
compatibility promise. It deliberately omits void values, converting
result constructors, throwing `value()` access, fallback access, comparisons, and
monadic operations. Propagate a value or tagged failure explicitly when changing
result types. Copy and move assignment require both payloads to have nonthrowing move
constructors and support the corresponding assignment. This keeps an active
payload present when copying a replacement throws. Add and test `Expected<void, E>`
alongside the first check that needs to return only success or an error.
