# TartanLlama expected

HostNumerics bundles the unmodified header from [TartanLlama/expected v1.3.1](https://github.com/TartanLlama/expected/tree/v1.3.1), commit `1770e3559f2f6ea4a5fb4f577ad22aeb30fbd8e4`.

Upstream file: `include/tl/expected.hpp`. Its SHA-256 is `0788f7d544a2fdec1dd7a794e1e9863ad4cc6a8817982b6fd93fd98724b9fa19`. The bundled copy is `include/roc/hostnumerics/compat/tl/expected.hpp`; it is copied byte-for-byte, with no upstream implementation edits. Only this upstream header is vendored, not upstream tests or build files.

Upstream applies [CC0-1.0](https://creativecommons.org/publicdomain/zero/1.0/legalcode). CC0 permits reuse and redistribution without attribution or a requirement to ship the legal text. The bundled header retains the upstream dedication and license link. The restricted facade in `include/roc/hostnumerics/compat/expected.hpp`, the existing-name bridge in `include/roc/hostnumerics/result.hpp`, and the contract tests are HostNumerics code, not copied upstream implementation. They keep the HostNumerics MIT license.

See [the result contract](../../docs/error_handling.md#internal-c-results) for the supported interface and limits.
