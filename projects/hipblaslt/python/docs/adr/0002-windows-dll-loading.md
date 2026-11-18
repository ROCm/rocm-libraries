# ADR 0002: How Windows finds the DLLs

Status:  Accepted
Decision:  Review A, Decision 5 (AIHPBLAS-5274)

## Context

Since Python 3.8, Windows resolves an extension's dependent DLLs only from the system directories, the extension's own directory, and directories registered with `os.add_dll_directory`. `PATH` is ignored. The first spike (AIHPBLAS-5272) saw `import hipblaslt` fail with "DLL load failed while importing _core" until `<ROCm>\bin` was registered.

Spike 3 (AIHPBLAS-5626) added two facts:
- `rocm_sdk.find_libraries("hipblaslt")` returns `libhipblaslt.dll` on Windows too.
- In TheRock's rocm-sdk wheels the HIP runtime (`amdhip64_7.dll`) lives in `_rocm_sdk_core\bin`, not next to `libhipblaslt.dll` in `_rocm_sdk_libraries\bin`. So registering only the library's directory isn't enough, and an override build needs its own runtime directory, not the active environment's.

rocISA already solved the general problem with `_candidate_dll_dirs` (`tensilelite/rocisa/rocisa/__init__.py`), a pure function with a unit test. Its "three directories up from the `.pyd`" guess doesn't fit a pip install in a venv, though.

## Decision

- Use the same lookup as Linux (Decision 10): `HIPBLASLT_PY_LIBRARY`, then the rocm-sdk wheels, then `ROCM_PATH` and `ROCM_HOME`, then the build-time ROCm. Windows looks in `<root>\bin` for the DLL.
- List candidate directories in this order, skipping duplicates and directories that don't exist:
  1. the library's own directory;
  2. its HIP runtime's directory: `_rocm_sdk_core\bin` next to a wheel's `_rocm_sdk_libraries`, then the directories `rocm_sdk` reports for `amdhip64`;
  3. `ROCM_PATH`, `HIP_PATH` and `ROCM_HOME`, each with `\bin`;
  4. the build-time ROCm's `bin`.
- Register only two of them with `os.add_dll_directory`: the library's own directory, and the first candidate that holds an `amdhip64*.dll`. Register every candidate only if none holds a runtime. Then preload the chosen `libhipblaslt.dll` by full path.
- The selection is two pure functions in rocISA's style, `windows_dll_dirs` and `windows_dirs_to_register` in `hipblaslt._loader`. `tests/test_dll_dirs.py` tests both on every platform.

## Consequences

- An override of the form `...\_rocm_sdk_libraries\bin\libhipblaslt.dll` gets its own build's HIP runtime, which Spike 3 showed is what makes it load.
- Windows doesn't define the search order among `os.add_dll_directory` entries. Registering every candidate was tried first: with an override, the old build's `libhipblaslt.dll` loaded next to the active environment's `amdhip64_7.dll`. Registering one runtime directory removes that ambiguity. The load checks from Decision 10 still report mixed companion libraries.
- The Windows CPU CI lane exercises the import, but no lane runs a second ROCm, so the override cases rest on Spike 3's evidence and the pure-function tests.
