---
name: hipdnn-tidy
description: Run clang-tidy on hipDNN after a build, scoped to the translation units affected by changed files. Much faster than ENABLE_CLANG_TIDY=ON, which runs clang-tidy inline with every compile. Resolves changed headers to the translation units that include them. Use when asked to run clang-tidy, lint changed files, or check tidy cleanliness before pushing.
argument-hint: "[build-dir] [base=<ref>] [files=<path>...] [all] [dry-run] [jobs=<n>]"
allowed-tools: Bash, Read, Grep, Glob
---

# hipDNN clang-tidy (post-build, changed files only)

Use this skill when the user asks to run clang-tidy over hipDNN, lint what they
changed, or confirm a branch is tidy-clean before pushing.

Do **not** reach for `-DENABLE_CLANG_TIDY=ON` for this. That option runs
clang-tidy inline with every compile: it roughly quadruples a clean Windows
build and re-runs on every recompile. clang-tidy needs only the compile
database, so this skill runs it afterwards over a normal build and narrows the
work to what changed.

## How it works

`scripts/tidy_changed.py` drives LLVM's `run-clang-tidy` against
`compile_commands.json`:

1. Derives changed files from git: the merge base with the base ref, plus
   staged, unstaged, and untracked edits.
2. Resolves changed **headers** to translation units. Headers are not
   translation units and never appear in the compile database, so a changed
   header is walked backwards through the include graph to every `.cpp` that
   reaches it, transitively. The scan reads `#include` lines, so it works on a
   configured-but-never-compiled tree.
3. Writes a filtered `compile_commands.json` containing exactly the selected
   entries and points `run-clang-tidy` at it. This is exact, unlike
   `run-clang-tidy`'s positional path regexes, which both over- and
   under-match.

Diagnostics and exit code match a full run: the configuration comes from
`projects/hipdnn/.clang-tidy` (211 checks, `WarningsAsErrors: '*'`), so any
finding exits non-zero.

## Inputs

Infer from the user request:

- **Build directory**: required. Any configured hipDNN build directory holding
  `compile_commands.json`. It does **not** need `ENABLE_CLANG_TIDY=ON`; a normal
  build (or even a configure-only tree) is the intended input.
- **Base ref**: `--base <ref>`, default `origin/develop`. Use the branch the
  work will merge into.
- **Explicit files**: `--files <path>...` to check specific paths instead of
  deriving them from git. Headers are resolved to their includers exactly as
  git-derived changes are.
- **Everything**: `--all` to check every translation unit in the database.
- **Preview**: `--dry-run` to list the selected translation units without
  running clang-tidy. Use this first when a change touches a widely-included
  header, so the user can see the scope before committing to the run.
- **Jobs**: `-j <n>`, default processor count.
- **Tool overrides**: `--clang-tidy` and `--run-clang-tidy`. Normally omitted:
  both are read from the build's `CMakeCache.txt` (`CLANG_TIDY_EXE`,
  `RUN_CLANG_TIDY_EXE`), which the configure step already version-checked.

## Workflow

1. Find the build directory. Prefer one the user names; otherwise look for
   `compile_commands.json` under the repository's build directories and confirm
   the choice before running.

   Then choose the scope. Default to changed files; use `--all` for a full pass
   over every translation unit in the compile database when the user asks to
   check the whole project, wants a CI-equivalent result, or when the change
   invalidates a per-file scope: edits to `.clang-tidy`, to compiler flags, or
   to a build option that alters preprocessor branches. A full pass on hipDNN
   core is roughly 554 translation units and takes minutes, so say so before
   starting one, and prefer changed-file scope when a branch is the subject.

2. Run the script from the repository root:

   ```bash
   python <skill-directory>/scripts/tidy_changed.py --build-dir <build-dir> [options]
   ```

   Use the scripts bundled with the skill you were invoked from. Keep full
   output in a log when the run is large, and show the diagnostics rather than
   the progress lines.

3. Report the selected translation unit count against the database total, then
   the diagnostics. Group repeated header diagnostics: a defect in a header is
   reported once per including translation unit, so the raw count overstates
   the number of distinct problems.

4. On a non-zero exit, show each unique diagnostic with its check name. Fix the
   code rather than adding suppressions unless the user asks otherwise; a
   finding inside a system or vendor header is the exception worth raising with
   them.

## Report

Summarize:

- Changed files detected and the base ref used
- Selected translation units versus the compile database total
- Unique diagnostics, each with file, line, and check name
- Clean or not, and the exit code

## Notes

- The build directory must be configured. Compiled objects are not required:
  hipDNN's generated headers (`*_export.h`, `version.h`) are written at
  configure time and the FlatBuffers `*_generated.h` files are committed, so a
  configure-only tree analyses identically to a fully built one.
- The compile database is regenerated at configure time. After adding a source
  file, reconfigure or the new translation unit will not be in the database.
- The database encodes one configuration's flags. Checking against a Debug or
  sanitizer build analyses different preprocessor branches than a Release one.
- On Windows the script disables `bugprone-exception-escape` and
  `performance-noexcept-move-constructor`, mirroring the `WIN32` block in
  `projects/hipdnn/cmake/ClangTidy.cmake`. Both fire on Microsoft STL internals
  rather than on hipDNN code and are clean against libstdc++. Keep the two in
  sync.
- Per-directory `.clang-tidy` files are not applied. Passing `-config-file`
  disables clang-tidy's directory discovery, so the nested configs under
  `frontend/tests/`, `tests/frontend/`, and `test_sdk/tests/` are inert. This
  matches the behaviour of the in-tree tidy targets and of an
  `ENABLE_CLANG_TIDY=ON` build.
- Changing a widely-included header selects many translation units, which is
  correct but slow. `--dry-run` shows the scope first.
