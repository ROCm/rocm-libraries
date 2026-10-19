## Error handling

Expected compatibility, support and descriptor-translation failures use explicit
results or checks. Callers inspect those results to select a backend, materialize
an operand, or reject a descriptor. They must not infer recovery decisions from
an exception type or its message.

Internal fallible helpers return `detail::Expected<T, E>`, a C++20 subset of the
`std::expected` model. Values construct directly; errors use `detail::Unexpected`.
The result provides `[[nodiscard]]`, boolean/`has_value()` checks, value access,
and `error()` for propagation. It retains C++20 consumer compatibility; assignment
and unused standard APIs are omitted. Layout, storage, scalar conversion and
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
