# Design of the hipBLASLt Python bindings

This file records the decisions behind the bindings that a maintainer would otherwise have to rediscover: what each one is, and why. It describes the design as it is. When a decision changes, edit its section.

## Scope

The bindings are a development and testing tool for hipBLASLt and its consumers, not a ROCm deliverable. They're opt-in and source-build only: built from a rocm-libraries checkout against an installed ROCm, and installed into the active Python environment. hipBLASLt's CMake never adds or installs them, and TheRock never builds them, so nothing about hipBLASLt's own build, install rules or packages depends on them.

## HIP Python

AMD's [HIP Python](https://github.com/ROCm/hip-python) (`rocm-bindings-libraries` on PyPI) includes `rocm.bindings.hipblaslt`, Cython bindings generated from `hipblaslt.h`. These bindings are separate from it, and a program can use both.

- **They follow the checkout.** HIP Python is generated from a released header, once per ROCm release, so it trails hipBLASLt's development branch. These build against the checkout's headers, so the PR that changes hipBLASLt's API can update and test them in the same change.
- **They reach the C++ API.** HIP Python binds the C header only. The solution-index calls in `hipblaslt-ext.hpp` are C++, and they're much of what a tuning or testing tool needs from Python.
- **They check before they call.** HIP Python maps C calls directly, so a wrong scale type, a short workspace or an undersized buffer reaches hipBLASLt as is, and several such mistakes give a wrong answer, corrupt memory or crash. These bindings validate those inputs and raise instead.
- **They build on Windows.** HIP Python publishes Linux wheels only.

## Name

The import name is `hipblaslt` and the distribution reports hipBLASLt's own version (see Version). The package is named for what it wraps. It's never published (see Scope), so it doesn't claim a name on PyPI, and HIP Python's module is `rocm.bindings.hipblaslt`, so the two don't collide. If ROCm settles on a common naming scheme for its libraries' bindings, the package follows it.

## Layout

`src/hipblaslt/` is the Python package and `csrc/` is the nanobind extension, with the `_build_info.py` template CMake fills in. The `src/` layout, which origami's bindings also use, means a Python started in this directory imports the installed package rather than the source tree, which has no `_core`, and nothing outside the package ends up in the wheel.

## Version

The package version, `hipblaslt.__version__`, is hipBLASLt's own `MAJOR.MINOR.PATCH`, read from the `hipblaslt-version.h` of the hipBLASLt the bindings are built against. A separate bindings version would say nothing about which hipBLASLt a user has, and a solution index, a default heuristic choice or a bug is only meaningful for one hipBLASLt build.

- **Package metadata.** `skbuild_plugins/hipblaslt_version_provider.py` is a scikit-build-core dynamic-metadata provider (the standard `[[tool.dynamic-metadata]]` table, scikit-build-core 1.1.2 or later). It searches `hipblaslt_ROOT`, `ROCM_PATH`, `ROCM_HOME`, then `/opt/rocm`. `CMakeLists.txt` runs the same file as a script to get that install's root, and `find_package(hipblaslt)` looks only there, so the version and the library `_core` links come from one install even when another with the same version is on `CMAKE_PREFIX_PATH`. The bindings need a shared `libhipblaslt`; CMake stops on a static one, and `invoke build` rejects `--static` with `--python`.
- **The loaded build.** The version alone doesn't identify a build: hipBLASLt's kernels can be retuned without a version change. So `_core` reports the git revision as well as the version, both for the build it was compiled against and for the one loaded. `hipblasLtGetGitRevision` reports `HIPBLASLT_VERSION_TWEAK`, which is the revision, and the compiled-in value comes from the same header.
- **When it's checked.** The first `Handle()` created in a process compares the loaded hipBLASLt's version and git revision with the compiled-in ones, and warns with `HipblasLtLoadWarning` when either differs. It doesn't happen at import, because `hipblasLtGetVersion` and `hipblasLtGetGitRevision` need a handle, and creating one allocates GPU memory. `hipblasLtCreate` calls `exit()` when that allocation fails. A process that never creates a handle is never checked.

## Which libhipblaslt loads

The library that `_core` binds to isn't decided by link-time paths alone. With torch-ROCm or another ROCm in the process, a naive import can end up on the wrong library, on two copies at once, or on a mix of builds. So `import hipblaslt` chooses one library and preloads it by full path before importing `_core`, whose `NEEDED` entry then binds to the copy already in the process.

- **Lookup order:**
  1. `HIPBLASLT_PY_LIBRARY`, a full path to `libhipblaslt.so.1` or `libhipblaslt.dll`, for a user who wants one specific build;
  2. TheRock's rocm-sdk wheels (`rocm_sdk.find_libraries("hipblaslt")`), which torch-ROCm also uses;
  3. the hipBLASLt the bindings were built against, recorded in `_build_info.py`, and also in `RUNPATH` on Linux;
  4. `ROCM_PATH`, then `ROCM_HOME`, only if the build-time library is gone.
- **Why the build-time library comes before `ROCM_PATH`.** `_core` is compiled against one hipBLASLt, and `ROCM_PATH` usually names a base ROCm, not a particular hipBLASLt. After `invoke build --python`, the bindings are built against `hipblaslt-install/`, while `ROCM_PATH` still names the base ROCm that ships its own hipBLASLt. The rocm-sdk wheels stay ahead of it so that the process shares one copy with torch.
- **Saying so when it's not the build.** When the import ends up on any library other than the build-time one without the override (the rocm-sdk wheels, a copy torch already loaded, or `ROCM_PATH` because the build-time library is gone), it warns with `HipblasLtLoadWarning`, naming both. The same file under another name, such as the wheels' hard links, doesn't count.
- **Already loaded.** If no override is set and the process already has a `libhipblaslt` (torch imported first, say), the bindings use that copy rather than preload a second one. It must have the name `_core` links against, recorded at build time (`libhipblaslt.so.1` on Linux): a copy with another SONAME, such as `libhipblaslt.so.0`, raises `HipblasLtLoadError`, because `_core` would load a second hipBLASLt beside it.
- **The build-time HIP runtime.** A hipBLASLt installed on its own (for example by `invoke build`) can sit outside the ROCm whose HIP runtime it needs. When the build-time library is the one chosen, its HIP runtime, also recorded at build time, is preloaded first.
- **`LD_LIBRARY_PATH` on Linux.** Loading `libhipblaslt` by full path chooses it, not its dependencies. In an install that uses `DT_RUNPATH`, as ROCm's packages and `invoke build` do, the dynamic loader searches `LD_LIBRARY_PATH` first, so another ROCm there would supply the companion libraries and the preload would fail with an `undefined symbol` error. So when `LD_LIBRARY_PATH` is set and the chosen library uses `DT_RUNPATH`, the loader reads its `NEEDED` entries and preloads the ones in its own directory by full path first, dependencies before dependents, skipping any the process already has. TheRock's wheels use `DT_RPATH`, which `LD_LIBRARY_PATH` can't override. A preload that still fails raises `HipblasLtLoadError`, naming `LD_LIBRARY_PATH` when it's set.
- **Its SONAME, before it's loaded.** On Linux the chosen file's SONAME must be the name `_core` links against. A mismatch raises `HipblasLtLoadError` before anything is mapped: loading it would map it and then a second hipBLASLt for `_core`, and neither can be unloaded, so the process couldn't recover.
- **One HIP runtime.** After the preload, and again after import, the process must hold exactly one HIP runtime (`libamdhip64.so.*`, `amdhip64*.dll`), or the import raises `HipblasLtLoadError`: an already-loaded `libamdhip64.so.5` beside the `.so.7` hipBLASLt needs can't share a process. Because an earlier import can supply the runtime, the warnings name the runtime actually loaded, and a build-time hipBLASLt running on another runtime than its build's warns.
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

Every `hipblasStatus_t` other than success becomes a `HipblasLtError` naming the failing call, the status and its number. No status is swallowed, including `hipblasLtDestroy`'s after a call that otherwise succeeded. `status.hpp` names statuses by number, not by enumerator, and takes the status as an `int`, so the PR that adds a status can name it before the ROCm the bindings compile against has it, and a value outside the installed header's enumerators is never stored in a `hipblasStatus_t`. A test parses the status enum from the checkout's `hipblas-common.h` when there is one (otherwise the installed one), so a status added in a PR fails until it's mapped. Errors in the Python values a caller passes are ordinary `ValueError`s and `TypeError`s; `HipblasLtError` is for hipBLASLt's own status codes and the bindings' library-state errors.

## CI

The bindings' Component CI lanes build them against an installed ROCm, the way a user does, rather than against the PR's own hipBLASLt.

- **Which ROCm.** Every lane installs TheRock's nightly wheels into a fresh venv: `rocm[libraries,devel]`, plus a `device-<gfx>` wheel on GPU lanes. The nightly is pinned in one place, `HIPBLASLT_PY_ROCM_VERSION` in `.github/workflows/component-ci-hipblaslt-python.yml`, and bumped deliberately in its own PR, so a failure that arrives with a new nightly is attributed to the nightly. `scripts/ci_lane.sh` holds the lane's steps (create the venv, install ROCm, build, test). The workflow runs them one at a time, so a failure names its step, and with no step named the script runs them all, the same way locally.
- **Why not the in-tree hipBLASLt.** The in-tree headers can't be used on their own: they need three generated headers (`hipblaslt-version.h`, `hipblaslt-export.h` and `hipblaslt-opt-in-features.h`) that exist only in an installed tree. And what changes in a bindings PR is the bindings, not the library.
- **Status values and enums follow the PR.** The status test reads the checkout's `hipblas-common.h`, and checks of the public enums read the PR's own headers, with the generated headers taken from the install. The lanes run when `projects/hipblaslt/library/include/hipblaslt/` or `projects/hipblas-common/library/include/` change, so a new status or enum value fails on the PR that adds it. Since `status.hpp` maps statuses by number, that PR can also map a new status before the pinned nightly has it. A new enum value is bound by name, so it needs a nightly that has it, or an `ALLOWED_MISSING` entry until then.
- **The gap this leaves.** The bindings compile against the installed headers. Changes to other declarations, or to the library's behavior, reach these lanes only once the nightly contains them and the pinned version is bumped. `TESTING.md` lists this under Known Risks.
- **Lanes.** Linux CPU and gfx942 run on every PR that changes the bindings or the public headers; gfx950, whose machines are scarce, runs only with the `ci:run-gfx950` label or on a manual run; Windows CPU runs informationally; Windows gfx1151 runs when the lanes themselves change. GPU lanes set `HIPBLASLT_PY_REQUIRE_GPU=1`, so a lane that can't see its device fails instead of skipping every GPU test. `TESTING.md` says which lanes gate.

## Enums

`_core` binds every value of every public enum in `hipblaslt.h`, `hipblaslt-ext.hpp` and `hipblas-common.h`, plus the `hipDataType` values hipBLASLt accepts, as Python `IntEnum`s with the C prefix dropped (`HIPBLASLT_EPILOGUE_RELU` is `Epilogue.RELU`). A missing or misnumbered value makes every call that uses it quietly wrong, so completeness is checked, not hoped for.

- **One registration form.** Each value is one `reg(e, "<C enum>", "<Python name>", <C value>)` call in `csrc/enums.cpp`, which also records it in a registry the tests read.
- **Opt-in values.** A value inside `#if HIPBLASLT_HAS_...` is bound under the same `#if`, so it exists exactly when the hipBLASLt the bindings are built against turns that feature on.
- **What's left out.** `ALLOWED_MISSING` in `src/hipblaslt/_coverage.py` lists the values deliberately not bound, each with a reason. Only three kinds belong there:
  - sentinels such as `..._MAX`, which grow whenever a value is added;
  - values the header marks "Not supported yet";
  - the HIP data types and hipBLAS compute types hipBLASLt doesn't accept.
- **One parser, two checks.** `src/hipblaslt/_coverage.py` reads enums from C and C++ headers: comments containing braces, implicit, hex and expression values, `typedef enum` with or without a tag, `enum class`, CRLF line endings, and values under `#if HIPBLASLT_HAS_...`. It uses only the standard library.
  - The `hipblaslt-python-enum-sync` pre-commit hook (`scripts/check_enum_sync.py`) compares the raw headers with `enums.cpp`, with nothing built. It's the first line of defense, and fires on the PR that adds a value. CI's pre-commit checkout can be sparse, so files that aren't on disk are read from git, and an unreadable one is an error.
  - The coverage tests compare the headers after the real preprocessor has run, so the build configuration decides which opt-in values count. In a source checkout they read the checkout's own headers.

## Handles and descriptor attributes

- **`Handle()` checks for a device first.** `hipblasLtCreate` calls `exit()` when it can't allocate its device memory, including when there's no device, which would end the Python process with no error. So `Handle()` raises `HipblasLtError` instead, and tests that run without a GPU never construct one. Using a handle after `close()` raises rather than reaching hipBLASLt.
- **Descriptors don't hold a handle.** hipBLASLt creates matmul descriptors, matrix layouts and preferences without one, so they work without a GPU and have no lifetime tie to a handle.
- **One table-driven attribute API.** `set_attribute(attr, value)` and `get_attribute(attr)` look each attribute up in a table of its C storage type (`int32`, `uint32`, `int64`, `uint64`, `float32` or pointer) and its enum, if it has one, in `csrc/descriptors.cpp`. hipBLASLt checks only that an attribute buffer is big enough, copies `sizeof(attribute)` bytes, and accepts most values unchecked: an 8-byte `TRANSA`, a float's bits as an epilogue, a 4-byte workspace size. So the setter writes exactly the attribute's size, and refuses a value that doesn't fit, isn't a member of the attribute's enum, or is a member of a different enum, before hipBLASLt sees it. An attribute missing from the table raises rather than being guessed.
- **Where the header and the implementation disagree, the table follows the implementation.** `EPILOGUE` and `BIAS_BATCH_STRIDE` are stored as `int32_t`; `BATCH_MODE` is `int32_t` and `OFFSET` is `int64_t`.

## Device memory

`DeviceArray` owns device memory holding one `hipDataType`. Its failure mode is the quiet one: a float16 array copied into memory hipBLASLt treats as float32 gives numbers that look plausible and are wrong.

- **Every check is in C++.** `from_numpy`, `copy_from_host` and `copy_to_host` check, before any HIP call, that the host array's dtype is the one the type holds, that it's C-contiguous (a strided array is refused, not copied), and that the byte counts match exactly. Calling `_core` directly gets the same checks as the Python wrapper. `R_8I` also takes `uint8`, for byte buffers such as workspaces.
- **`ml_dtypes` types cross as views.** nanobind accepts only NumPy's own scalar types, so a bfloat16 or fp8 array crosses into C++ as its same-width unsigned view, with its dtype name alongside. The C++ side checks both, so a raw `uint16` isn't accepted as bfloat16 by accident. `hipblaslt.from_numpy` and `DeviceArray.to_numpy` do the viewing.
- **Supported types.** float32, float64, float16, bfloat16, int8, int32, and the four fp8 formats. fp4, fp6 and int4 aren't supported: hipBLASLt packs them more densely than `ml_dtypes`' one value per byte.
