# Design of the hipBLASLt Python bindings

This file records the decisions behind the bindings that a maintainer would otherwise have to rediscover: what each one is, and why. It describes the design as it is. When a decision changes, edit its section.

## Scope

The bindings are a development and testing tool for hipBLASLt and its consumers, not a ROCm deliverable. They're opt-in and source-build only: built from a rocm-libraries checkout against an installed ROCm, and installed into the active Python environment. hipBLASLt's CMake never adds or installs them, and TheRock never builds them, so nothing about hipBLASLt's own build, install rules or packages depends on them.

## Version

The package version, `hipblaslt.__version__`, is hipBLASLt's own `MAJOR.MINOR.PATCH`, read from the `hipblaslt-version.h` of the hipBLASLt the bindings are built against. A separate bindings version would say nothing about which hipBLASLt a user has, and a solution index, a default heuristic choice or a bug is only meaningful for one hipBLASLt build.

- **Package metadata.** `skbuild_plugins/hipblaslt_version_provider.py` is a scikit-build-core dynamic-metadata provider (the standard `[[tool.dynamic-metadata]]` table, scikit-build-core 1.1.2 or later). It searches `hipblaslt_ROOT`, `ROCM_PATH`, `ROCM_HOME`, then `/opt/rocm`. `CMakeLists.txt` stops the build if `find_package(hipblaslt)` found a different version, so the two can't disagree.
- **The loaded build.** The version alone doesn't identify a build: hipBLASLt's kernels can be retuned without a version change. So the bindings also compare git revisions. `hipblasLtGetGitRevision` reports `HIPBLASLT_VERSION_TWEAK`, which is the revision, and the compiled-in value comes from the same header.
- **When it's checked.** The first `Handle()` created in a process compares the loaded hipBLASLt's version and git revision with the compiled-in ones, and warns with `HipblasLtLoadWarning` when either differs. It doesn't happen at import, because `hipblasLtGetVersion` and `hipblasLtGetGitRevision` need a handle, and creating one allocates GPU memory. `hipblasLtCreate` calls `exit()` when that allocation fails. A process that never creates a handle is never checked.

## Which libhipblaslt loads

The library that `_core` binds to isn't decided by link-time paths alone. With torch-ROCm or another ROCm in the process, a naive import can end up on the wrong library, on two copies at once, or on a mix of builds. So `import hipblaslt` chooses one library and preloads it by full path before importing `_core`, whose `NEEDED` entry then binds to the copy already in the process.

- **Lookup order:**
  1. `HIPBLASLT_PY_LIBRARY`, a full path to `libhipblaslt.so.1` or `libhipblaslt.dll`, for a user who wants one specific build;
  2. TheRock's rocm-sdk wheels (`rocm_sdk.find_libraries("hipblaslt")`), which torch-ROCm also uses;
  3. `ROCM_PATH`, then `ROCM_HOME`;
  4. the ROCm the bindings were built against, recorded in `_build_info.py`, and also in `RUNPATH` on Linux.
- **Already loaded.** If no override is set and the process already has a `libhipblaslt` (torch imported first, say), the bindings use that copy rather than preload a second one. It must have the name `_core` links against, recorded at build time (`libhipblaslt.so.1` on Linux): a copy with another SONAME, such as `libhipblaslt.so.0`, raises `HipblasLtLoadError`, because `_core` would load a second hipBLASLt beside it.
- **The build-time HIP runtime.** A hipBLASLt installed on its own (for example by `invoke build`) can sit outside the ROCm whose HIP runtime it needs. When the build-time library is the one chosen, its HIP runtime, also recorded at build time, is preloaded first.
- **`LD_LIBRARY_PATH` on Linux.** Loading `libhipblaslt` by full path chooses it, not its dependencies. In an install that uses `DT_RUNPATH`, as ROCm's packages and `invoke build` do, the dynamic loader searches `LD_LIBRARY_PATH` first, so another ROCm there would supply the companion libraries and the preload would fail with an `undefined symbol` error. So when `LD_LIBRARY_PATH` is set and the chosen library uses `DT_RUNPATH`, the loader reads its `NEEDED` entries and preloads the ones in its own directory by full path first, dependencies before dependents, skipping any the process already has. TheRock's wheels use `DT_RPATH`, which `LD_LIBRARY_PATH` can't override. A preload that still fails raises `HipblasLtLoadError`, naming `LD_LIBRARY_PATH` when it's set.
- **After import,** the process must hold exactly one `libhipblaslt`, the one chosen, or the import raises `HipblasLtLoadError`.
- **Checks.** Each turns a cryptic failure (an `undefined symbol`, or "The specified procedure could not be found") or a silent mix into a clear message. They read the list of libraries in the process: `/proc/self/maps` on Linux, `EnumProcessModules` on Windows. `hipblaslt._loader` keeps the checks as pure functions over that list, so they're testable without a second ROCm.
  - The override can't take effect, because the process already uses a different `libhipblaslt`: raise `HipblasLtLoadError`, and say to import `hipblaslt` first or unset the override.
  - The override is active: log which build loaded, and warn that importing a framework built against a different ROCm afterward will fail to load. A framework imported afterward that does load, torch for example, shares the override's `libhipblaslt`: there's one copy per process. That's what lets you run a framework on a custom hipBLASLt, but it's only safe when that build is ABI-compatible with the one the framework was built against. The matching SONAME is necessary, not sufficient, and frameworks can use the C++ `hipblaslt-ext` API as well as the C API.
  - hipBLASLt's companion libraries (`libtensilelite-host`, `liborigami`, `librocroller`; there's no rocRoller on Windows) aren't in `libhipblaslt`'s directory: raise `HipblasLtLoadError`. This runs before the preload too, because a companion from another build makes the preload fail, or, between ABI-compatible builds, mix silently and corrupt memory.
  - A different build than the one the bindings were compiled against: warn, as described under Version. Not when the library came from the override, which already warned and is what the caller asked for.

## Windows DLLs

Since Python 3.8, Windows resolves an extension's dependent DLLs only from the system directories, the extension's own directory, and directories registered with `os.add_dll_directory`; `PATH` is ignored. In the rocm-sdk wheels, the HIP runtime (`amdhip64_7.dll`) lives in `_rocm_sdk_core\bin`, not next to `libhipblaslt.dll` in `_rocm_sdk_libraries\bin`.

- The lookup order is the one above, with `<root>\bin` in place of `<root>/lib`.
- Before preloading, the loader lists candidate directories: the library's own directory; its HIP runtime's directory (`_rocm_sdk_core\bin` next to a wheel's `_rocm_sdk_libraries`, then the directories `rocm_sdk` reports for `amdhip64`); `ROCM_PATH`, `HIP_PATH` and `ROCM_HOME`, each with `\bin`; and the build-time ROCm's `bin`.
- It registers only two of them: the library's own directory, and the first candidate that holds an `amdhip64*.dll`. Windows doesn't define the search order among `os.add_dll_directory` entries, so registering every candidate could load one build's `libhipblaslt.dll` next to another build's HIP runtime.
- Both steps are pure functions, `windows_dll_dirs` and `windows_dirs_to_register`, tested on every platform.
- hipBLASLt's headers need clang, so Windows builds use ROCm's `amdclang++` with Ninja rather than MSVC.

## Errors

Every `hipblasStatus_t` other than success becomes a `HipblasLtError` naming the failing call, the status and its number. No status is swallowed. A test parses the status enum from the `hipblas-common.h` the bindings were built against, so a status added later fails until it's mapped. Errors in the Python values a caller passes are ordinary `ValueError`s and `TypeError`s; `HipblasLtError` is for hipBLASLt's own status codes and the bindings' library-state errors.
