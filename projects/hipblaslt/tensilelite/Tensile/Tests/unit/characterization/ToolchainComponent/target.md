# Toolchain/Component.py — characterization target

Pins the ROCm toolchain wrapper classes (Component / Assembler / Compiler /
Bundler / Linker) and helpers (_invoke, _getVersion, get_rocm_version).

Coverage: 107 stmts, 0 missed → 100% line (99.19% blended; residual branch
`233->exit` is the Assembler true16-set fall-through, covered the other way).

Subprocess-free: module-level `_invoke` (the single subprocess chokepoint for
the `__call__` methods) and `_getVersion` (the version probe in every
`__init__`) are stubbed. We pin argv *construction* — the exact command line
each wrapper hands to the toolchain — plus the asan/save-temps/Windows/true16/
wavefront/response-file branches.

Note: Linker/Bundler are constructed with **str** paths to mirror real callers
(`validateToolchain` returns str). Passing a `Path` would crash
`_use_response_file` (`len(Path)`), which is existing behavior, not pinned here.
