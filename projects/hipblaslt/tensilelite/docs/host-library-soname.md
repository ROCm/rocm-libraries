# Host library SONAME

`libtensilelite-host` is installed as `libtensilelite-host.so.1`
(`rocm_set_soversion(tensilelite-host "1.0")`). The YAML build
(`TENSILE_YAML`) and the msgpack build (`TENSILE_MSGPACK`) use that same
SONAME. They export different functions and accept different library files.

A shared client bound to `.so.1` can load the other build. A missing function
fails with `undefined symbol` before the library prints a diagnostic.

## What this draft does not change

The SONAME stays `1.0`. Bumping it, or giving the two builds different names,
changes the library ROCm ships. hipBLASLt and every other consumer relink, and
packaging changes. That needs agreement from the TensileLite owners.

The dev client is handled separately: it links this host statically, so the
installed `.so.1` is not on its load path. Packaged hipBLASLt stays shared.
That client already names a parser mismatch (YAML versus msgpack). Adding a
version to that message needs another exported function, which an older host
does not provide, so the version belongs with the SONAME change.

## Symbol check

`hipblaslt.symbol-contract.tensilelite-host` compares the `TensileLite` symbols
hipBLASLt imports, plus `library/symbol-contract/tensilelite-host-imports.txt`,
with the `tensilelite-host` built beside it. It runs only when both libraries
are shared and testing is enabled. It does not compare a YAML host with a
msgpack host.

## Separate case

`hipblaslt-bench` loads `libhipblaslt.so.1`. A profiler that prepends the
installed ROCm library directory on `LD_LIBRARY_PATH` can select that installed
library instead of a dev build. Linking `tensilelite-host` statically does not
change that search. This is not reproduced here.
