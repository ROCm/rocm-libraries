# Lane: public API (`API`)

Every change to a public surface must be intentional and described in the PR.

## Surfaces

| Surface | Where |
|---|---|
| C API and ABI | `projects/miopen/include/miopen/miopen.h` (`MIOPEN_EXPORT` declarations, enums, status codes); the exported symbol set and SONAME, pinned by `projects/miopen/test/public_abi/public_symbols.baseline`; the three `miopenConvolution*GetWorkSpaceSizeRange` entry points exported but declared only by their consumers |
| Public/private split | The five hand-maintained files `projects/miopen/script/check_public_abi.py` names (`miopen.h`, `src/private/miopen_impl.h`, `src/private/miopen_private_rename.h`, `src/private/wrapper.cpp`, and `dnn-providers/miopen-provider/MiopenApiPrivateRename.hpp`); the wrapper-only API `projects/miopen/include_wrapper/miopen/hipdnn_forwarding.h` with `test/public_abi/wrapper_only_symbols.txt` (flag-on builds only) |
| Generated headers | `projects/miopen/include/miopen/config.h.in`, `version.h.in`; the C++ headers installed with `MIOPEN_INSTALL_CXX_HEADERS=ON` (Fin and tuning tools consume them) |
| Solver identity | Solver IDs from the registration order in `projects/miopen/src/solver.cpp` (returned as `solution_id`, accepted by the immediate-mode API) and each solver's `SolverDbId` name |
| Database formats | Find-db, perf-db and user-db keys and record formats (problem `Serialize`, network config, perf-config strings), the shipped files under `projects/miopen/src/kernels/`, and the kernel cache layout (`projects/miopen/docs/conceptual/`) |
| Runtime contract | Environment variables (`MIOPEN_DECLARE_ENV_VAR_*`, documented in `projects/miopen/docs/reference/env_variables.rst`), default find mode and tuning policy, log formats others parse, workspace sizes the API reports |
| MIOpenDriver | Command-line options and output that users and the tuning infrastructure script against (`projects/miopen/driver/`) |
| CMake package | Exported targets, `PACKAGE_DEPENDS`, option names consumers or TheRock set |

`projects/miopen/src/include/` is internal even when `MIOPEN_INSTALL_CXX_HEADERS=ON` ships it; flag an internal type reaching
`miopen.h`. Behavior is a surface too: status codes, defaults, environment variables, database formats or log formats changed in
source behind unchanged signatures are API changes.

## Method

1. Diff the surfaces between `$BASE` and `$HEAD`. Classify each changed declaration: added, removed, signature changed,
   semantics changed, default changed.
2. Find each in the PR description. Described and intended → note it. Not described → **Needs human decision**: unintended API
   change, with the exact declaration ([../policy.md](../policy.md) *Review rules*).
3. Breaking changes (removal, signature or semantics change, enum value reuse or renumbering, struct layout change, a solver ID
   or `SolverDbId` change) are `BLOCKING` unless described with a deprecation and migration path. New `miopen.h` entry points use
   `MIOPEN_EXPORT`, sit behind `MIOPEN_BETA_API` when experimental, and append enum values rather than inserting them.
4. A new or changed public entry point appears in all five split files with matching signatures and a dispatch macro in its
   wrapper stub, and in `public_symbols.baseline` (`check_public_abi.py check-headers` and `check`). A baseline edited without a
   matching header change, or the reverse, is a finding. The provider mirror moves in the same PR or a linked one; otherwise the
   flag-on build or the hipDNN provider breaks.
5. Database compatibility: a change to a problem key, network config or perf-config serialization either keeps existing records
   readable (old records parse or are ignored cleanly, never misapplied) or updates every shipped database it affects and says so;
   check the code that reads the record, not only the writer.
6. New environment variables users are expected to set are documented in `projects/miopen/docs/reference/env_variables.rst`;
   a changed default is described in the PR and in `projects/miopen/CHANGELOG.md`.
7. New public declarations carry Doxygen documentation in the style of the surrounding `miopen.h` entries
   (`projects/miopen/docs/doxygen/Doxyfile` generates the API reference from it).
