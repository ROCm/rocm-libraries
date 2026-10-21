## Error handling

HostNumerics-owned terminal errors throw `std::runtime_error`.
Expected support and compatibility failures should use explicit results
or checks. Allocation, system, dependency and visitor errors retain their
original types. Dispatch rejects invalid enum values before invoking the visitor
or moving its arguments. This policy does not promise exceptions-disabled builds.

## Internal C++ results

`compat::expected<T, E>` in `compat/expected.hpp` represents either a value or an error.
Use `compat::unexpected(error)` to construct a failure, check `has_value()` or
its boolean conversion, and use `*`/`->` or `error()` only in the matching state.
The upstream implementation stores its payload inline and adds no allocation or exception
translation. An owning payload can still allocate or throw.

The implementation is [TartanLlama/expected v1.3.1](https://github.com/TartanLlama/expected/tree/v1.3.1), pinned at commit `1770e3559f2f6ea4a5fb4f577ad22aeb30fbd8e4`. Its header is bundled unchanged under CC0-1.0, with the upstream dedication, license link, and [source provenance](../third_party/expected/README.md) retained. HostNumerics owns only the restricted interface; upstream handles the active payload, copy, move, and destruction.

The facade accepts `std::in_place` for success construction and provides the standard `unexpected::error()` name. It deliberately omits assignment, void values, converting result constructors, throwing `value()` access, fallback access, comparisons, and monadic operations. Converting results must propagate their value or tagged error explicitly. This also avoids the upstream converting-result lifetime defect tracked in [issue #180](https://github.com/TartanLlama/expected/issues/180). Results cannot be assigned, so no state-changing payload assignment is exposed.

HostNumerics retains C++20 because it cannot yet assume C++23 standard-library support across its consumers. The `compat` namespace keeps the result call sites independent of the backport so they can move to `std::expected` once that baseline is available. The supported subset is internal and carries no source or ABI compatibility promise. Constructors retain payload-based `noexcept`; observers follow the upstream exception specifications. `result.hpp` temporarily provides the old `detail::Expected` and error-factory names for prototype consumers.
