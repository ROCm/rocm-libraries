## Error handling

Native single-value rounding and conversion return values without throwing or
returning a failure result. `convertScalar<T>` is `noexcept`; like a native cast,
its caller must satisfy the documented conversion preconditions.

For non-Boolean integer destinations, conversion is `cast<T>(f_T(g(x)))`.
The rounding operation `g` is `TowardZero` or `NearestEven` and leaves integral
inputs unchanged. The range operation `f_T` is `Identity` (leave the rounded value
unchanged), `Saturate` (clamp to T's range), or `ModuloWrap` (reduce modulo 2^N,
where N is T's bit width, with a two's-complement interpretation for signed T).
`Identity` requires the rounded value to fit in T before the final cast.
NaN is outside the domain, and infinity requires `Saturate`.
Complex-to-real input must have a zero imaginary component.
Floating destinations use native casts, and Boolean conversion tests
for a nonzero value; both ignore the integer policies.

These conversions do not validate individual values or assert numerical
preconditions. Tensor storage conversion uses the same numerical preconditions;
descriptor and storage-access validation remain separate.

Native conversion requires explicit integer rounding and overflow policies.
These policies do not check floating-to-floating precision loss.

Expected compatibility, support and descriptor-translation failures use explicit
results or checks. Callers inspect those results to select a backend, materialize
an operand, or reject a descriptor. They must not infer recovery decisions from
an exception type or its message.

Internal fallible helpers return `detail::Expected<T, E>`, a C++20 subset of the
`std::expected` model. Values construct directly; errors use `detail::Unexpected`.
The result provides `[[nodiscard]]`, boolean/`has_value()` checks, value access,
and `error()` for propagation. It retains C++20 consumer compatibility; unused
standard APIs are omitted. Layout, storage and descriptor checks use a closed
failure vocabulary. Diagnostics are obtained from the failure at reporting
boundaries, without error-message output parameters. Checks needing
dynamic diagnostic text can carry an owned string in the same result interface.

The integer-conversion and offset leaves are independent of throwing tensor and
metadata headers and are tested with exception handling disabled and allocation
attempts fatal. Native numerical conversion allocates no storage; offset checks
also allocate no storage on success or failure. Result payload construction
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

## Prepared execution

`detail::IndexIteration` validates and owns the shape and index order during
setup. Its `noexcept` coordinate methods assert that spans have the correct rank,
indices are in range, and the shape is nonempty. `preparedElementOffset` asserts
coordinate rank and range and requires bounds already checked during Tensor
construction. These assertions diagnose a broken internal loop in Debug builds;
Release builds use the established preconditions without repeated validation.

Prepared numerical conversion uses the same cast preconditions as public
single-value conversion. A valid floating Tensor can contain infinity, so
validated geometry alone does not establish representability for an integer
destination. The caller must select a policy whose domain includes the input
values. Numerical conversion does not check these preconditions in either
Debug or Release builds.

The internal `detail/layout_offset.hpp` leaf checks borrowed extent, stride and
coordinate spans without including Shape or Layout. It is tested alongside
integer conversion with exceptions disabled and allocation attempts prohibited.
Layout, Tensor and owning preparation still permit allocation exceptions.
This split does not establish an exceptions-disabled build of the entire library.

Fallible queries and factories use `get` names, including `getContiguousLayout`,
`getElementBounds`, `getStorageBytesForLayout` and descriptor mappings. Their
result type states whether failure is possible. The optional structured-sparsity
executor uses `runTwoOfFourByteAlignedIfSupported`, since it executes a supported
case rather than computing a query result. Throwing conveniences retain their
existing names.
