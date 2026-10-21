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

`compat::expected<T, E>` in `compat/expected.hpp` represents either a value or an error.
Use `compat::unexpected(error)` to construct a failure, check `has_value()` or
its boolean conversion, and use `*`/`->` or `error()` only in the matching state.
The upstream implementation stores its payload inline and adds no allocation or exception
translation. An owning payload can still allocate or throw.

The implementation is [TartanLlama/expected v1.3.1](https://github.com/TartanLlama/expected/tree/v1.3.1), pinned at commit `1770e3559f2f6ea4a5fb4f577ad22aeb30fbd8e4`. Its header is bundled unchanged under CC0-1.0, with the upstream dedication, license link, and [source provenance](../third_party/expected/README.md) retained. HostNumerics owns only the restricted interface; upstream handles the active payload, copy, move, and destruction.

The facade accepts `std::in_place` for success construction and provides the standard `unexpected::error()` name. It deliberately omits assignment, void values, converting result constructors, throwing `value()` access, fallback access, comparisons, and monadic operations. Converting results must propagate their value or tagged error explicitly. This also avoids the upstream converting-result lifetime defect tracked in [issue #180](https://github.com/TartanLlama/expected/issues/180). Results cannot be assigned, so no state-changing payload assignment is exposed.

HostNumerics retains C++20 because it cannot yet assume C++23 standard-library support across its consumers. The `compat` namespace keeps the result call sites independent of the backport so they can move to `std::expected` once that baseline is available. The supported subset is internal and carries no source or ABI compatibility promise. Constructors retain payload-based `noexcept`; observers follow the upstream exception specifications. `result.hpp` temporarily provides the old `detail::Expected` and error-factory names for prototype consumers.
